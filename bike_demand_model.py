# # Bike Demand Forecasting & Capacity-Pricing Model
#
# An end-to-end supply-demand matching model for a retailer selling three bikes
# (Comfort Mountain, Tough Track, Ultimate Mountain), built from order-level sales transactions:
#
#   Part 1 – Demand forecasting: four OLS models forecast total demand 1 to 4 two-week periods ahead,
#            using lagged demand, trend, annual seasonality and orders already received (advance demand).
#   Part 2 – Product choice: a multinomial logit model estimates how the three posted prices drive
#            the probability that a customer chooses each bike.
#   Part 3 – Capacity, pricing and order quantities: capacity is reserved from the 8-week-ahead forecast;
#            two weeks before the selling period, prices and order quantities are adjusted with the updated
#            forecast so that expected demand matches the committed capacity. Evaluated out-of-sample.
#
# Data: data/sales_transactions.csv (see data/README.md for the required columns). Run from the repository root.

# Part 1 – Two-week aggregation and OLS forecasting models (h = 1 to 4)
# All products are aggregated into two-week periods and one OLS model is developed for each forecast horizon of one to four two-week periods.
# Demand is assigned to the **delivery period** (the period in which the bikes are delivered),
# and **advance demand** is used as information that is already known when the forecast is made: the units of a delivery period that were *ordered in an earlier period*. 
# For a forecast made $h$ periods ahead, advance demand is the part of the delivery period's units that was ordered at least $h$ periods earlier.


# Part 1 – Step 1. Libraries and data

# %%
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.stats.diagnostic import het_breuschpagan
import matplotlib.pyplot as plt

os.makedirs('figures', exist_ok=True)                    # charts are saved here

df = pd.read_csv('data/sales_transactions.csv', parse_dates=['order_date', 'delivery_date'])
print(df.shape)
print(df.head())
print(df.isna().sum())

# Part 1 – Step 2. Aggregate transactions into two-week delivery periods

# Period 0 starts on the first order date (2021-02-07); every period is 14 days long.
first_day = df['order_date'].min()
df['order_period'] = (df['order_date'] - first_day).dt.days // 14           # period in which the customer ordered
df['delivery_period'] = (df['delivery_date'] - first_day).dt.days // 14     # period in which the bike is delivered = demand period

# Lead time between order and delivery
print('Lead time (delivery date - order date) in days:')
print((df['delivery_date'] - df['order_date']).dt.days.describe().round(1))

# Units per bike in each delivery period (0 if a bike was not delivered in that period)
units = df.pivot_table(index='delivery_period', columns='Product', values='quantity', aggfunc='sum', fill_value=0)

# Average posted price (at the time of the order) of the transactions delivered in each period
prices = df.groupby('delivery_period')[['Price_Comfort_Mountain', 'Price_Tough_Track', 'Price_Ultimate_Mountain']].mean()

biweekly = pd.concat([units, prices], axis=1)
biweekly['Total_Demand'] = units.sum(axis=1)          # all three bikes together
biweekly['Period_Start'] = first_day + pd.to_timedelta(biweekly.index * 14, unit='D')
print('Delivery periods in the data:', biweekly.index.min(), 'to', biweekly.index.max(), ' number:', len(biweekly))

# The first and the last delivery periods are incomplete: orders placed before 2021-02-07 or after 2026-01-31 are not in the data.
print(biweekly.iloc[[0, 1, -2, -1]][['Period_Start', 'Total_Demand']])
biweekly = biweekly.iloc[1:-1]                        # keep the complete delivery periods only

print(biweekly.head())
print(biweekly.tail())
print('Number of two-week periods:', len(biweekly), ' (consecutive:', (np.diff(biweekly.index) == 1).all(), ')')
print(biweekly['Total_Demand'].describe())

# Part 1 – Step 3. Advance demand for each forecast horizon

# Advance demand for horizon $h$ in delivery period $t$ = units delivered in $t$ that were ordered in period $t-h$ or earlier. 
# This generalises the monthly definition (`is_advance = order_month < delivery_month`) to $h$ periods.

# Units delivered in period t that were ordered at least h periods earlier
adv_1 = df[df['order_period'] <= df['delivery_period'] - 1].groupby('delivery_period')['quantity'].sum()
adv_2 = df[df['order_period'] <= df['delivery_period'] - 2].groupby('delivery_period')['quantity'].sum()
adv_3 = df[df['order_period'] <= df['delivery_period'] - 3].groupby('delivery_period')['quantity'].sum()
adv_4 = df[df['order_period'] <= df['delivery_period'] - 4].groupby('delivery_period')['quantity'].sum()

advance = pd.concat([adv_1, adv_2, adv_3, adv_4], axis=1, keys=['Adv_h1', 'Adv_h2', 'Adv_h3', 'Adv_h4'])
advance = advance.reindex(biweekly.index).fillna(0)     # 0 if nothing was ordered in advance
biweekly = biweekly.join(advance)

print(biweekly[['Period_Start', 'Total_Demand', 'Adv_h1', 'Adv_h2', 'Adv_h3', 'Adv_h4']].head(8))
print('Share of total demand that is known h periods ahead:')
print((advance.sum() / biweekly['Total_Demand'].sum()).round(4))
print('Periods with any advance demand:', (advance > 0).sum().values)
# Orders are delivered 3 to 15 days later, so only for h = 1 a large part of the demand is already known;
# for h = 2, 3, 4 (more than 28 days ahead) no order exists yet and advance demand is (almost) zero.

# Part 1 – Step 4. Look at the pattern in total demand (guides the choice of regressors)

# Autocorrelation of total demand at lags 1-8 (lag 1 = two weeks)
print('Autocorrelation of total demand:')
print(pd.Series({k: biweekly['Total_Demand'].autocorr(k) for k in range(1, 9)}).round(3))

# Units by delivery month (seasonality) and by delivery year (trend), complete periods only
kept = df[df['delivery_period'].isin(biweekly.index)]
print(kept.groupby(kept['delivery_date'].dt.month)['quantity'].sum())
print(kept.groupby(kept['delivery_date'].dt.year)['quantity'].sum())


# Part 1 – Step 5. Do prices help explain TOTAL demand? (check before deciding whether to include them)


# Prices mainly shift demand between bikes (Part 2 - logistic regression), so test whether
# the average prices add anything to total demand once trend and seasonality are controlled for.
periods_per_year = 365.25 / 14                         # about 26.09 two-week periods per year
t = biweekly.index.to_series()

check = pd.DataFrame({'y': biweekly['Total_Demand']})
check['trend'] = t
check['sin1'] = np.sin(2 * np.pi * t / periods_per_year)
check['cos1'] = np.cos(2 * np.pi * t / periods_per_year)
check['sin2'] = np.sin(4 * np.pi * t / periods_per_year)
check['cos2'] = np.cos(4 * np.pi * t / periods_per_year)
check['p_CM'] = biweekly['Price_Comfort_Mountain']
check['p_TT'] = biweekly['Price_Tough_Track']
check['p_UM'] = biweekly['Price_Ultimate_Mountain']

price_check = sm.OLS(check['y'], sm.add_constant(check.drop(columns='y'))).fit()
print(price_check.summary().tables[1])
# All three price coefficients have large p-values -> prices are not used in the total-demand OLS.


# Part 1 – Step 6. Starting point: the baseline specification, and the OLS assumptions

# The baseline model is $D_t=\beta_0+\beta_1 D_{t-h}+\beta_2\,\text{Advance}_{h,t}+\epsilon_t$. It is estimated for each $h$ (for $h\ge 2$ there is no advance demand,
# so only the lagged demand is used). The assumptions of OLS are checked with the **Durbin–Watson** statistic (should be close to 2; 
# a low value means positive autocorrelation, assumption 3) and the **Breusch–Pagan** test (p-value above 0.05: the variance of the errors is constant, assumption 4).

D = biweekly['Total_Demand']

base = pd.DataFrame({'y': D})
base['lag1'] = D.shift(1)
base['lag2'] = D.shift(2)
base['lag3'] = D.shift(3)
base['lag4'] = D.shift(4)
base['adv1'] = biweekly['Adv_h1']
base = base.dropna()                                    # same sample for the four models

base_h1 = sm.OLS(base['y'], sm.add_constant(base[['lag1', 'adv1']])).fit()
base_h2 = sm.OLS(base['y'], sm.add_constant(base[['lag2']])).fit()
base_h3 = sm.OLS(base['y'], sm.add_constant(base[['lag3']])).fit()
base_h4 = sm.OLS(base['y'], sm.add_constant(base[['lag4']])).fit()

base_models = [base_h1, base_h2, base_h3, base_h4]
base_table = pd.DataFrame({
    'R2': [m.rsquared for m in base_models],
    'RMSE': [np.sqrt(m.mse_resid) for m in base_models],
    'Durbin_Watson': [durbin_watson(m.resid) for m in base_models],
    'Breusch_Pagan_p': [het_breuschpagan(m.resid, m.model.exog)[1] for m in base_models]},
    index=['h=1 (lag + advance)', 'h=2 (lag)', 'h=3 (lag)', 'h=4 (lag)'])
print('Observations:', int(base_h1.nobs))
print(base_table.round(3))
print(base_h1.params.round(3))
# h = 1 is already very accurate because most of the demand is known in advance.
# For h = 2, 3, 4 the Durbin-Watson statistic is far below 2: seasonality and trend are still in the errors.

# Part 1 – Step 7. Build the data set for each forecast horizon (final models)

# Direct forecast made at the end of period $t$ for period $t+h$ (only information known at time $t$ is used, so a separate model is estimated for every $h$):

# $$D_{t+h}=\beta_0+\beta_1 D_t+\beta_2 D_{t-1}+\beta_3 D_{t-2}+\beta_4 (t+h)+\beta_5\sin_1+\beta_6\cos_1+\beta_7\sin_2+\beta_8\cos_2\;[+\;\beta_9\,\text{Advance}_{1,t+1}\ \text{for }h=1]$$

# Trend and the two seasonal sine/cosine pairs (annual cycle) remove the autocorrelation that the baseline specification leaves in the errors for $h\ge 2$; 
# the advance demand is added for $h=1$, the only horizon for which it exists.


# ---------- h = 1 ----------
target_1 = t + 1                                       # index of the period being forecast
h1 = pd.DataFrame({'y': D.shift(-1)})                  # demand one period ahead
h1['adv'] = biweekly['Adv_h1'].shift(-1)               # orders already placed for delivery in t + 1 (known at time t)
h1['D_t'] = D
h1['D_t1'] = D.shift(1)
h1['D_t2'] = D.shift(2)
h1['trend'] = target_1
h1['sin1'] = np.sin(2 * np.pi * target_1 / periods_per_year)
h1['cos1'] = np.cos(2 * np.pi * target_1 / periods_per_year)
h1['sin2'] = np.sin(4 * np.pi * target_1 / periods_per_year)
h1['cos2'] = np.cos(4 * np.pi * target_1 / periods_per_year)
h1 = h1.dropna()

# ---------- h = 2 ----------
target_2 = t + 2
h2 = pd.DataFrame({'y': D.shift(-2)})
h2['D_t'] = D
h2['D_t1'] = D.shift(1)
h2['D_t2'] = D.shift(2)
h2['trend'] = target_2
h2['sin1'] = np.sin(2 * np.pi * target_2 / periods_per_year)
h2['cos1'] = np.cos(2 * np.pi * target_2 / periods_per_year)
h2['sin2'] = np.sin(4 * np.pi * target_2 / periods_per_year)
h2['cos2'] = np.cos(4 * np.pi * target_2 / periods_per_year)
h2 = h2.dropna()

# ---------- h = 3 ----------
target_3 = t + 3
h3 = pd.DataFrame({'y': D.shift(-3)})
h3['D_t'] = D
h3['D_t1'] = D.shift(1)
h3['D_t2'] = D.shift(2)
h3['trend'] = target_3
h3['sin1'] = np.sin(2 * np.pi * target_3 / periods_per_year)
h3['cos1'] = np.cos(2 * np.pi * target_3 / periods_per_year)
h3['sin2'] = np.sin(4 * np.pi * target_3 / periods_per_year)
h3['cos2'] = np.cos(4 * np.pi * target_3 / periods_per_year)
h3 = h3.dropna()

# ---------- h = 4 (two months ahead: the capacity-reservation horizon in Part 3) ----------
target_4 = t + 4
h4 = pd.DataFrame({'y': D.shift(-4)})
h4['D_t'] = D
h4['D_t1'] = D.shift(1)
h4['D_t2'] = D.shift(2)
h4['trend'] = target_4
h4['sin1'] = np.sin(2 * np.pi * target_4 / periods_per_year)
h4['cos1'] = np.cos(2 * np.pi * target_4 / periods_per_year)
h4['sin2'] = np.sin(4 * np.pi * target_4 / periods_per_year)
h4['cos2'] = np.cos(4 * np.pi * target_4 / periods_per_year)
h4 = h4.dropna()

print('Observations:', len(h1), len(h2), len(h3), len(h4))

# Part 1 – Step 8. Estimate the four OLS models

# %%
ols_h1 = sm.OLS(h1['y'], sm.add_constant(h1.drop(columns='y'))).fit()
ols_h2 = sm.OLS(h2['y'], sm.add_constant(h2.drop(columns='y'))).fit()
ols_h3 = sm.OLS(h3['y'], sm.add_constant(h3.drop(columns='y'))).fit()
ols_h4 = sm.OLS(h4['y'], sm.add_constant(h4.drop(columns='y'))).fit()

print(ols_h1.summary())
print(ols_h2.summary())
print(ols_h3.summary())
print(ols_h4.summary())

# ### Part 1 – Step 9. Compare the four models side by side, with the OLS assumption checks

coef_table = pd.DataFrame({'h=1': ols_h1.params, 'h=2': ols_h2.params,
                           'h=3': ols_h3.params, 'h=4': ols_h4.params}).round(3)
pval_table = pd.DataFrame({'h=1': ols_h1.pvalues, 'h=2': ols_h2.pvalues,
                           'h=3': ols_h3.pvalues, 'h=4': ols_h4.pvalues}).round(4)
print('Coefficients:')
print(coef_table)
print('p-values:')
print(pval_table)

fit_table = pd.DataFrame({
    'Observations': [int(ols_h1.nobs), int(ols_h2.nobs), int(ols_h3.nobs), int(ols_h4.nobs)],
    'R2': [ols_h1.rsquared, ols_h2.rsquared, ols_h3.rsquared, ols_h4.rsquared],
    'Adj_R2': [ols_h1.rsquared_adj, ols_h2.rsquared_adj, ols_h3.rsquared_adj, ols_h4.rsquared_adj],
    'RMSE': [np.sqrt(ols_h1.mse_resid), np.sqrt(ols_h2.mse_resid), np.sqrt(ols_h3.mse_resid), np.sqrt(ols_h4.mse_resid)],
    'AIC': [ols_h1.aic, ols_h2.aic, ols_h3.aic, ols_h4.aic],
    'BIC': [ols_h1.bic, ols_h2.bic, ols_h3.bic, ols_h4.bic],
    'Durbin_Watson': [durbin_watson(ols_h1.resid), durbin_watson(ols_h2.resid),
                      durbin_watson(ols_h3.resid), durbin_watson(ols_h4.resid)],
    'Breusch_Pagan_p': [het_breuschpagan(ols_h1.resid, ols_h1.model.exog)[1], het_breuschpagan(ols_h2.resid, ols_h2.model.exog)[1],
                        het_breuschpagan(ols_h3.resid, ols_h3.model.exog)[1], het_breuschpagan(ols_h4.resid, ols_h4.model.exog)[1]],
    'Jarque_Bera_p': [jarque_bera(ols_h1.resid)[1], jarque_bera(ols_h2.resid)[1],
                      jarque_bera(ols_h3.resid)[1], jarque_bera(ols_h4.resid)[1]],
}, index=['h=1', 'h=2', 'h=3', 'h=4']).round(3)
print(fit_table)

# The Breusch-Pagan p-values are close to or below 0.05 (the variance of the errors is not perfectly constant, assumption 4).
# The OLS coefficients, and therefore the forecasts, stay unbiased. Heteroscedasticity only affects the standard errors,
# so the p-values are re-computed with heteroscedasticity-robust (HC3) standard errors as a check.
robust_h1 = sm.OLS(h1['y'], sm.add_constant(h1.drop(columns='y'))).fit(cov_type='HC3')
robust_h2 = sm.OLS(h2['y'], sm.add_constant(h2.drop(columns='y'))).fit(cov_type='HC3')
robust_h3 = sm.OLS(h3['y'], sm.add_constant(h3.drop(columns='y'))).fit(cov_type='HC3')
robust_h4 = sm.OLS(h4['y'], sm.add_constant(h4.drop(columns='y'))).fit(cov_type='HC3')
robust_pval_table = pd.DataFrame({'h=1': robust_h1.pvalues, 'h=2': robust_h2.pvalues,
                                  'h=3': robust_h3.pvalues, 'h=4': robust_h4.pvalues}).round(4)
print('p-values with robust (HC3) standard errors:')
print(robust_pval_table)
same_conclusion = ((pval_table < 0.05) == (robust_pval_table < 0.05)).drop(index='const')
print('Same 5% conclusion with OLS and robust standard errors for every variable (constant excluded):', same_conclusion.all().values)

# Total persistence of recent demand (sum of the three lag coefficients) at each horizon
persistence = pd.Series({
    'h=1': ols_h1.params[['D_t', 'D_t1', 'D_t2']].sum(),
    'h=2': ols_h2.params[['D_t', 'D_t1', 'D_t2']].sum(),
    'h=3': ols_h3.params[['D_t', 'D_t1', 'D_t2']].sum(),
    'h=4': ols_h4.params[['D_t', 'D_t1', 'D_t2']].sum()}).round(3)
print('Sum of lag coefficients:')
print(persistence)

# Lowest fitted value of each model (OLS can return negative demand in the low season)
print('Minimum fitted demand:',
      round(ols_h1.fittedvalues.min(), 2), round(ols_h2.fittedvalues.min(), 2),
      round(ols_h3.fittedvalues.min(), 2), round(ols_h4.fittedvalues.min(), 2))

# Part 1 – Step 10. Chart: actual demand vs. fitted values for h = 1 and h = 4

# Index of each fitted value = index of the period being forecast (t + h)
fitted_h1 = ols_h1.fittedvalues.copy()
fitted_h1.index = fitted_h1.index + 1
fitted_h4 = ols_h4.fittedvalues.copy()
fitted_h4.index = fitted_h4.index + 4

fig, ax = plt.subplots(figsize=(10, 4.2))
ax.plot(biweekly['Period_Start'], D, color='#9a9a94', linewidth=1.2, label='Actual total demand')
ax.plot(biweekly['Period_Start'].loc[fitted_h1.index], fitted_h1.values, color='#2a78d6', linewidth=2,
        label='Fitted, h = 1 (two weeks ahead)')
ax.plot(biweekly['Period_Start'].loc[fitted_h4.index], fitted_h4.values, color='#eb6834', linewidth=2,
        label='Fitted, h = 4 (eight weeks ahead)')
ax.set_title('Total demand per two-week delivery period: actual vs. OLS fitted values', loc='left', fontsize=12)
ax.set_ylabel('Units per two-week period')
ax.set_xlabel('Start of period')
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='y', color='#e6e6e2', linewidth=0.8)
ax.legend(frameon=False, loc='upper left')
plt.tight_layout()
plt.savefig('figures/actual_vs_fitted.png', dpi=200)
plt.show()

# Part 2 – Choice probabilities with logistic regression

# Each transaction is one customer's choice among three bikes, so the dependent variable has three categories and a **multinomial logistic regression** (multinomial logit) is used. 
# Comfort Mountain is the base category. Let $V_T$ and $V_U$ be the log-odds of Tough Track and Ultimate Mountain relative to Comfort Mountain:

# $$V_j = \alpha_j + \beta_{j,C}\,P_C + \beta_{j,T}\,P_T + \beta_{j,U}\,P_U, \qquad j \in \{T, U\}$$

# $$P(\text{Comfort}) = \frac{1}{1 + e^{V_T} + e^{V_U}}, \quad P(\text{Tough}) = \frac{e^{V_T}}{1 + e^{V_T} + e^{V_U}}, \quad P(\text{Ultimate}) = \frac{e^{V_U}}{1 + e^{V_T} + e^{V_U}}$$

# All three posted prices enter both equations because a customer compares the prices of all three bikes. Each transaction counts as one choice (the quantity bought is not part of the choice model).

# Part 2 – Step 1. Dependent variable and explanatory variables

# Dependent variable: the bike chosen (0 = Comfort Mountain = base category, 1 = Tough Track, 2 = Ultimate Mountain)
choice = df['Product'].map({'Comfort Mountain': 0, 'Tough Track': 1, 'Ultimate Mountain': 2})

# Explanatory variables: posted prices of the three bikes at the time of the order, plus an intercept
price_cols = ['Price_Comfort_Mountain', 'Price_Tough_Track', 'Price_Ultimate_Mountain']
X_choice = sm.add_constant(df[price_cols])

print('Number of choices:', len(choice))
print(choice.value_counts().sort_index())
print(choice.value_counts(normalize=True).sort_index().round(3))
print(df[price_cols].describe().round(0))

# Part 2 – Step 2. Estimate the multinomial logit model

logit_model = sm.MNLogit(choice, X_choice).fit(disp=0)
print(logit_model.summary())

# Part 2 – Step 3. Coefficient tables

# Column 0 = Tough Track vs. Comfort Mountain, column 1 = Ultimate Mountain vs. Comfort Mountain
coef_table = pd.DataFrame({
    'TT_coef': logit_model.params[0], 'TT_std_err': logit_model.bse[0], 'TT_z': logit_model.tvalues[0], 'TT_p': logit_model.pvalues[0],
    'UM_coef': logit_model.params[1], 'UM_std_err': logit_model.bse[1], 'UM_z': logit_model.tvalues[1], 'UM_p': logit_model.pvalues[1]})
print(coef_table.round(4))

# Odds ratio for a $100 price increase = exp(100 * coefficient)
odds_ratio_100 = np.exp(100 * logit_model.params.drop('const'))
odds_ratio_100.columns = ['Tough Track vs Comfort', 'Ultimate Mountain vs Comfort']
print(odds_ratio_100.round(3))

# Part 2 – Step 4. Model fit

# %%
print('McFadden pseudo R-squared:', round(logit_model.prsquared, 4))
print('Likelihood-ratio chi-square:', round(logit_model.llr, 1), ' df =', int(logit_model.df_model), ' p-value =', logit_model.llr_pvalue)
print('Log-likelihood:', round(logit_model.llf, 2), ' AIC:', round(logit_model.aic, 1))

# Classification: predict the bike with the highest choice probability
predicted_choice = logit_model.predict(X_choice).values.argmax(axis=1)
accuracy = (predicted_choice == choice.values).mean()
baseline = choice.value_counts(normalize=True).max()
print('In-sample accuracy:', round(accuracy, 4), ' | always predicting the most common bike:', round(baseline, 4))

confusion = pd.crosstab(choice.map({0: 'Comfort', 1: 'Tough', 2: 'Ultimate'}),
                        pd.Series(predicted_choice, index=choice.index).map({0: 'Comfort', 1: 'Tough', 2: 'Ultimate'}),
                        rownames=['Actual'], colnames=['Predicted'])
print(confusion)

# Average predicted probability equals the observed share of each bike (a property of the maximum-likelihood fit)
print(logit_model.predict(X_choice).mean().round(4).values, choice.value_counts(normalize=True).sort_index().round(4).values)

# Part 2 – Step 5. Choice probabilities at the scenario prices ($4,200, $1,950, $500)

assumed_prices = pd.DataFrame({'const': [1.0],
                               'Price_Comfort_Mountain': [4200.0],
                               'Price_Tough_Track': [1950.0],
                               'Price_Ultimate_Mountain': [500.0]})

prob_assumed = logit_model.predict(assumed_prices)
prob_assumed.columns = ['Comfort Mountain', 'Tough Track', 'Ultimate Mountain']
print(prob_assumed.round(4))
print('Sum of the three probabilities:', prob_assumed.sum(axis=1).values.round(4))

# Cross-check with the logit formula written out by hand
b = logit_model.params
v_tough = (b.loc['const', 0] + b.loc['Price_Comfort_Mountain', 0] * 4200
           + b.loc['Price_Tough_Track', 0] * 1950 + b.loc['Price_Ultimate_Mountain', 0] * 500)
v_ultimate = (b.loc['const', 1] + b.loc['Price_Comfort_Mountain', 1] * 4200
              + b.loc['Price_Tough_Track', 1] * 1950 + b.loc['Price_Ultimate_Mountain', 1] * 500)

denominator = 1 + np.exp(v_tough) + np.exp(v_ultimate)
print('V_T =', round(v_tough, 4), ' V_U =', round(v_ultimate, 4))
print('P(Comfort Mountain)  =', round(1 / denominator, 4))
print('P(Tough Track)       =', round(np.exp(v_tough) / denominator, 4))
print('P(Ultimate Mountain) =', round(np.exp(v_ultimate) / denominator, 4))

# Part 2 – Step 6. Price sensitivity: change in choice probability (percentage points) for a $100 price increase

plus_comfort = assumed_prices.copy()
plus_comfort['Price_Comfort_Mountain'] = 4300.0

plus_tough = assumed_prices.copy()
plus_tough['Price_Tough_Track'] = 2050.0

plus_ultimate = assumed_prices.copy()
plus_ultimate['Price_Ultimate_Mountain'] = 600.0

change_table = pd.DataFrame({
    '+$100 on Comfort Mountain': (logit_model.predict(plus_comfort).values[0] - prob_assumed.values[0]) * 100,
    '+$100 on Tough Track': (logit_model.predict(plus_tough).values[0] - prob_assumed.values[0]) * 100,
    '+$100 on Ultimate Mountain': (logit_model.predict(plus_ultimate).values[0] - prob_assumed.values[0]) * 100},
    index=['Comfort Mountain', 'Tough Track', 'Ultimate Mountain'])
print(change_table.T.round(2))

# Part 2 – Step 7. Cross-check with scikit-learn

# The same model is estimated with scikit-learn's `LogisticRegression` and assessed with the F1 score. The probabilities at the scenario prices are the same as with statsmodels.

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

y_sk = df['Product'].values                                   # bike chosen (text labels)
X_sk = df[price_cols].values                                  # the three posted prices

model_classification = LogisticRegression(random_state=0, max_iter=1000).fit(X_sk, y_sk)
print('Classes:', model_classification.classes_)
print('F1 score (weighted):', round(f1_score(y_sk, model_classification.predict(X_sk), average='weighted'), 4))

prob_sklearn = model_classification.predict_proba(np.array([[4200, 1950, 500]]))
print('scikit-learn probabilities at 4,200 / 1,950 / 500:', prob_sklearn.round(4))
print('statsmodels probabilities (Step 5)                :', prob_assumed.values.round(4))
print('Largest difference:', np.abs(prob_sklearn - prob_assumed.values).max().round(6))

# Part 3 – Matching capacity, prices and order quantities to demand
#
# **Principle.** Supply is matched to demand with the information that is available at each decision point. 
# No cost data are needed: the reserved capacity is the demand forecast, and the later price and quantity decisions use the newer forecast to close the gap between demand and the capacity that is already committed.
#
# | Decision point | Information available | Decision |
# |---|---|---|
# | 2 months (4 periods) before selling period $s$ | Demand observed up to period $s-4$; no order for period $s$ exists yet → OLS forecast $\hat D^{(4)}_s$ | Reserve capacity $K_s=\hat D^{(4)}_s$ (total demand of the three bikes) |
# | 2 weeks (1 period) before $s$ | Demand observed up to period $s-1$ **and the orders already received for delivery in $s$ (advance demand)** → updated OLS forecast $\hat D^{(1)}_s$ | $K_s$ is already committed. Set the price and the order quantity of every bike so that expected demand equals $K_s$ |
# | Selling period $s$ | Customers arrive and choose | Sales of each bike = min(order, demand) |
#
# **Models.** The OLS models of Part 1 forecast total demand. A logit choice model (Part 2, extended below with a no-purchase option) describes how the three prices change the demand for each bike and how many customers leave without buying. Both are re-estimated on the **training** periods only. The decisions are then made for the **test** periods and evaluated with the actual test demand.


# ### Assumption for the customers who do not buy

# The transactions contain only completed purchases. Customers who looked at the bikes and walked away are missing, and without them the three choice probabilities add up to one: a price change can then move customers between bikes but never out of the market, so prices could not be used to steer total demand towards the committed capacity. I therefore add a fourth alternative, *no purchase*, and make the following assumptions.

# 1. **Constant outside option, own-price utilities.** Not buying has utility $0$ (it does not depend on the bike prices). Bike $i$ has utility $V_i=c+\alpha_i+\beta_i\,p_i$, which depends on its own price $p_i$. Then
# $$P_i=\frac{e^{V_i}}{1+e^{V_C}+e^{V_T}+e^{V_U}},\qquad P_0=1-\sum_i P_i,\qquad S=\sum_i P_i \;(\text{probability of buying anything}).$$
# 2. **What the purchase data identify.** Among buyers $P_i/P_j=e^{V_i-V_j}$ does not contain the outside option, so the transactions identify the price coefficients $\beta_i$ and the differences $\alpha_i-\alpha_j$ (maximum likelihood, Step 3). They cannot reveal the common level $c$, i.e. how many customers leave.
# 3. **Calibration of $c$ (the missing observations).** At the reference prices $p^0$ (the average prices in the training data) a share $\theta=50\%$ of the customers who consider a bike buy one, i.e. for every purchase there is on average one customer who did not buy. 50% is the neutral mid-point; Step 9 repeats the analysis with 30% and 70%.
# 4. **Reference demand.** Prices are not in the OLS models (Part 1 found no significant effect on total demand), so the OLS forecast $\hat D$ is read as the demand expected at the reference prices $p^0$. The number of interested customers is $M=\hat D/\theta$, and at any prices $p$ the expected demand is $M\,P_i(p)$.

# Part 3 – Step 1. Chronological training / test split and re-estimation of the OLS models

# The complete delivery periods are 1 to 129. Training = periods 1-103 (80%), test = periods 104-129 (the last 26 periods,
# 2025-02-02 to 2026-01-31, one full year). A chronological split is used because the OLS models forecast the future from the past.
n_train = 104                                           # first test period
n_test = 130 - n_train
print('Test period starts on:', first_day + pd.Timedelta(days=14 * n_train))

# h1..h4 come from Part 1: the index is the last observed period t, the target period is t + h.
# A row belongs to the training set if its TARGET period is before period 104.
train_h1 = h1[h1.index + 1 < n_train]
test_h1 = h1[h1.index + 1 >= n_train]
train_h2 = h2[h2.index + 2 < n_train]
test_h2 = h2[h2.index + 2 >= n_train]
train_h3 = h3[h3.index + 3 < n_train]
test_h3 = h3[h3.index + 3 >= n_train]
train_h4 = h4[h4.index + 4 < n_train]
test_h4 = h4[h4.index + 4 >= n_train]

ols_tr_h1 = sm.OLS(train_h1['y'], sm.add_constant(train_h1.drop(columns='y'))).fit()
ols_tr_h2 = sm.OLS(train_h2['y'], sm.add_constant(train_h2.drop(columns='y'))).fit()
ols_tr_h3 = sm.OLS(train_h3['y'], sm.add_constant(train_h3.drop(columns='y'))).fit()
ols_tr_h4 = sm.OLS(train_h4['y'], sm.add_constant(train_h4.drop(columns='y'))).fit()

print('Training observations:', int(ols_tr_h1.nobs), int(ols_tr_h2.nobs), int(ols_tr_h3.nobs), int(ols_tr_h4.nobs))
print('Test observations:', len(test_h1), len(test_h2), len(test_h3), len(test_h4))
print(pd.DataFrame({'h=1': ols_tr_h1.params, 'h=2': ols_tr_h2.params, 'h=3': ols_tr_h3.params, 'h=4': ols_tr_h4.params}).round(3))

# Part 3 – Step 2. Out-of-sample accuracy of the OLS forecasts (test periods)

# Forecasts for the test periods (negative forecasts are set to zero because demand cannot be negative)
pred_h1 = ols_tr_h1.predict(sm.add_constant(test_h1.drop(columns='y'), has_constant='add')).clip(lower=0)
pred_h2 = ols_tr_h2.predict(sm.add_constant(test_h2.drop(columns='y'), has_constant='add')).clip(lower=0)
pred_h3 = ols_tr_h3.predict(sm.add_constant(test_h3.drop(columns='y'), has_constant='add')).clip(lower=0)
pred_h4 = ols_tr_h4.predict(sm.add_constant(test_h4.drop(columns='y'), has_constant='add')).clip(lower=0)

# Re-index every forecast by the period it refers to (origin + h)
forecast_h1 = pd.Series(pred_h1.values, index=pred_h1.index + 1)
forecast_h2 = pd.Series(pred_h2.values, index=pred_h2.index + 2)
forecast_h3 = pd.Series(pred_h3.values, index=pred_h3.index + 3)
forecast_h4 = pd.Series(pred_h4.values, index=pred_h4.index + 4)

oos_table = pd.DataFrame({
    'Train_RMSE': [np.sqrt(ols_tr_h1.mse_resid), np.sqrt(ols_tr_h2.mse_resid), np.sqrt(ols_tr_h3.mse_resid), np.sqrt(ols_tr_h4.mse_resid)],
    'Test_RMSE': [np.sqrt(((test_h1['y'] - pred_h1) ** 2).mean()), np.sqrt(((test_h2['y'] - pred_h2) ** 2).mean()),
                  np.sqrt(((test_h3['y'] - pred_h3) ** 2).mean()), np.sqrt(((test_h4['y'] - pred_h4) ** 2).mean())],
    'Test_MAE': [(test_h1['y'] - pred_h1).abs().mean(), (test_h2['y'] - pred_h2).abs().mean(),
                 (test_h3['y'] - pred_h3).abs().mean(), (test_h4['y'] - pred_h4).abs().mean()],
    'Test_bias (actual - forecast)': [(test_h1['y'] - pred_h1).mean(), (test_h2['y'] - pred_h2).mean(),
                                      (test_h3['y'] - pred_h3).mean(), (test_h4['y'] - pred_h4).mean()],
    'Test_mean_demand': [test_h1['y'].mean(), test_h2['y'].mean(), test_h3['y'].mean(), test_h4['y'].mean()]},
    index=['h=1', 'h=2', 'h=3', 'h=4'])
print(oos_table.round(2))

# Part 3 – Step 3. Choice model with the no-purchase option: estimation on the training transactions
# Assumptions 1–2: the own-price logit among the three bikes is estimated by maximum likelihood on the training purchases. 
# Comfort Mountain is the reference bike ($\alpha_C=0$). A function is needed here only because `minimize()` has to call the log-likelihood repeatedly. 
# The Part 2 model (all three prices in every equation) is re-estimated on the same data as a benchmark for a likelihood-ratio test.

# %%
from scipy.optimize import minimize
from scipy.stats import chi2
from statsmodels.tools.numdiff import approx_hess

# Training transactions = orders placed before period 104 (the choice model describes the order decision, so the split uses the order date);
# test transactions = the later orders
train_mask = (df['order_period'] < n_train).values
P_train = df.loc[train_mask, price_cols].values          # posted prices at every training order (n x 3)
P_test = df.loc[~train_mask, price_cols].values
y_train = choice[train_mask].values                      # chosen bike: 0 = Comfort, 1 = Tough, 2 = Ultimate
y_test = choice[~train_mask].values
print('Training transactions:', len(y_train), ' Test transactions:', len(y_test))

def neg_loglik(theta, prices, bike):
    alpha = np.array([0.0, theta[0], theta[1]])          # alpha_C = 0 (reference); alpha_T - alpha_C, alpha_U - alpha_C
    beta = theta[2:5]                                    # own-price coefficients of Comfort, Tough, Ultimate
    V = alpha + beta * prices                            # utility of each bike at every order (n x 3)
    V = V - V.max(axis=1, keepdims=True)                 # only for numerical stability; does not change the probabilities
    log_den = np.log(np.exp(V).sum(axis=1))
    return -(V[np.arange(len(bike)), bike] - log_den).sum()

start = np.array([0.0, 0.0, -0.005, -0.01, -0.02])
fit_cl = minimize(neg_loglik, start, args=(P_train, y_train), method='BFGS')
theta_hat = fit_cl.x
alpha_hat = np.array([0.0, theta_hat[0], theta_hat[1]])
beta_hat = theta_hat[2:5]

# Standard errors from the numerical Hessian of the negative log-likelihood
se_hat = np.sqrt(np.diag(np.linalg.inv(approx_hess(theta_hat, neg_loglik, args=(P_train, y_train)))))
cl_table = pd.DataFrame({'Estimate': theta_hat, 'Std_err': se_hat, 'z': theta_hat / se_hat},
                        index=['alpha_T - alpha_C', 'alpha_U - alpha_C', 'beta_Comfort', 'beta_Tough', 'beta_Ultimate'])
print(cl_table.round(5))

# Benchmark: Part 2 model (all three prices in both equations) estimated on the training data
full_train = sm.MNLogit(choice[train_mask], X_choice[train_mask]).fit(disp=0)
lr_stat = 2 * (full_train.llf + fit_cl.fun)
print('Log-likelihood own-price logit:', round(-fit_cl.fun, 2), ' Part 2 model:', round(full_train.llf, 2))
print('Likelihood-ratio test (own-price vs. full): chi2 =', round(lr_stat, 2), ' df = 7  p =', round(chi2.sf(lr_stat, 7), 3))

# Accuracy of the own-price logit: bike with the highest probability = bike actually bought
V_train = alpha_hat + beta_hat * P_train
V_test = alpha_hat + beta_hat * P_test
print('Accuracy  training:', round((V_train.argmax(axis=1) == y_train).mean(), 4), ' test:', round((V_test.argmax(axis=1) == y_test).mean(), 4))

# ### Part 3 – Step 4. Calibrating the no-purchase level (assumptions 3 and 4)
# At the reference prices $p^0$ the probability of buying must equal $\theta$. Because $S=\dfrac{E}{1+E}$ with $E=\sum_i e^{V_i}$, this gives $E=\theta/(1-\theta)$ and 
# therefore $c=\ln\dfrac{\theta}{1-\theta}-\ln\sum_i e^{\alpha_i+\beta_i p^0_i}$.

theta_buy = 0.50                                         # share of interested customers who buy at the reference prices

p0 = P_train.mean(axis=0)                                # reference prices = average training prices
print('Reference prices p0 (Comfort, Tough, Ultimate):', p0.round(1))

V0 = alpha_hat + beta_hat * p0                           # bike utilities at p0 without the constant c
share_buyers = np.exp(V0) / np.exp(V0).sum()             # split among the bikes of a customer who buys (does not depend on c)
c_hat = np.log(theta_buy / (1 - theta_buy)) - np.log(np.exp(V0).sum())
print('Constant c:', round(c_hat, 3))

# Probabilities at the reference prices
P_ref = np.exp(c_hat + V0) / (1 + np.exp(c_hat + V0).sum())
print('P(Comfort, Tough, Ultimate) at p0:', P_ref.round(4), ' P(buy) =', round(P_ref.sum(), 4), ' P(no purchase) =', round(1 - P_ref.sum(), 4))
print('Split among bikes of a buyer     :', share_buyers.round(4))
print('Non-buyers per purchase at p0    :', round((1 - theta_buy) / theta_buy, 2))

# How strongly does total demand react to prices in this model? (all three prices up or down by the same percentage)
for pct in [-0.02, -0.01, 0.01, 0.02]:
    V_shift = c_hat + alpha_hat + beta_hat * p0 * (1 + pct)
    S_shift = (np.exp(V_shift) / (1 + np.exp(V_shift).sum())).sum()
    print('All prices', f'{pct:+.0%}', '-> total demand', f'{S_shift / theta_buy - 1:+.1%}')

# Part 3 – Step 5. Decision point 1 (two months ahead): reserve the capacity
#
# The capacity for selling period $s$ is the total-demand forecast made two months earlier, $K_s=\hat D^{(4)}_s$. 
# At the second decision point the horizon-1 forecast $\hat D^{(1)}_s$ is available as well; the difference is the information gained in between. 
# Two kinds of information arrive in these six weeks: the demand of the periods $s-3$ to $s-1$, and the **orders already received for period $s$** (advance demand, Part 1, Step 3). 
# Orders are delivered 3 to 15 days after they are placed, so no order exists two months ahead, whereas most of the demand is already known two weeks ahead.

test_periods = np.arange(n_train, 130)
D4_test = forecast_h4.loc[test_periods].values           # forecast made 8 weeks before the period
D1_test = forecast_h1.loc[test_periods].values           # updated forecast made 2 weeks before the period
D_actual = D.loc[test_periods].values                    # demand that actually occurred

capacity = D4_test                                       # decision 1: capacity K_s = h=4 forecast of total demand
revision = D1_test - capacity                            # information gained between the two decisions

adv_test = biweekly['Adv_h1'].loc[test_periods].values   # orders already received two weeks before the period (known at decision 2)
print('Share of the test-period demand already ordered  8 weeks ahead:', round(biweekly['Adv_h4'].loc[test_periods].sum() / D_actual.sum(), 3),
      ' 2 weeks ahead:', round(adv_test.sum() / D_actual.sum(), 3))
print('Total reserved capacity over the test period:', round(capacity.sum(), 1), ' units (actual demand', D_actual.sum(), ')')
print('Revision of the forecast (h=1 minus h=4): mean', round(revision.mean(), 2), ' std', round(revision.std(), 2),
      ' min', round(revision.min(), 1), ' max', round(revision.max(), 1))
print('Periods with upward revision (demand above capacity):', (revision > 0).sum(), ' downward:', (revision < 0).sum())
print('Test RMSE of the forecast against actual demand: h=4', round(np.sqrt(((D_actual - D4_test) ** 2).mean()), 2),
      ' h=1', round(np.sqrt(((D_actual - D1_test) ** 2).mean()), 2))

# Part 3 – Step 6. Decision point 2 (two weeks ahead): prices and order quantities

# Expected demand for bike $i$ at prices $p$ is $q_i(p)=\dfrac{\hat D^{(1)}}{\theta}P_i(p)$. If $\hat D^{(1)}>K$ the prices have to go up, if $\hat D^{(1)}<K$ they can go down, until expected demand equals the committed capacity.

# **Rule.** The three prices move so that the utility of every bike changes by the same amount $\Delta$, i.e. $p_i=p^0_i+\Delta/\beta_i$ (the less price-sensitive bike moves more in dollars). 
# Then the demand of all three bikes is scaled by the same factor, so the product mix expected at the reference prices is kept and the order quantity of each bike is its share of the capacity. 
# Requiring $\sum_i q_i=K$ gives in closed form

# $$S^{*}=\frac{\theta K}{\hat D^{(1)}},\qquad \Delta=\ln\frac{S^{*}/(1-S^{*})}{\theta/(1-\theta)},\qquad q_i=K\cdot\text{share}_i .$$

# The order quantity is the expected demand at the new prices, so supply equals demand in expectation.

# **Limit.** If $\theta K/\hat D^{(1)}\ge 1$ the committed capacity is larger than the number of all interested customers $\hat D^{(1)}/\theta$; even if all of them bought, 
# the capacity would not be filled. The purchase probability is then capped at 99.9% (a very large price cut) and the rest of the capacity stays unused.

# Purchase probability that makes expected demand equal to the capacity (cannot exceed 100%)
S_target = theta_buy * capacity / D1_test
fillable = S_target < 1                                  # periods in which the capacity can be filled
print('Periods in which the capacity exceeds the demand of all interested customers (cap applies):', (~fillable).sum())
S_target = np.minimum(S_target, 0.999)

delta = np.log((S_target / (1 - S_target)) / (theta_buy / (1 - theta_buy)))      # common utility shift
opt_price = p0 + delta[:, None] / beta_hat                                         # new prices (n_test x 3)

# Check with the logit model: choice probabilities at the new prices and expected demand of every bike
V_opt = c_hat + alpha_hat + beta_hat * opt_price
P_opt = np.exp(V_opt) / (1 + np.exp(V_opt).sum(axis=1, keepdims=True))
opt_qty = D1_test[:, None] / theta_buy * P_opt                                     # order quantity of every bike

print('Largest difference between total order and capacity (periods without the cap):', np.abs(opt_qty.sum(axis=1) - capacity)[fillable].max().round(6))
print('Difference in the capped period(s): capacity - order =', (capacity - opt_qty.sum(axis=1))[~fillable].round(2))
print('Price change vs reference (%): min', ((opt_price / p0 - 1) * 100).min(axis=0).round(1), ' max', ((opt_price / p0 - 1) * 100).max(axis=0).round(1))
print('Price range training data  : min', df.loc[train_mask, price_cols].min().values, ' max', df.loc[train_mask, price_cols].max().values)
print('Price range of the new plan: min', opt_price.min(axis=0).round(0), ' max', opt_price.max(axis=0).round(0))

# Part 3 – Step 7. Capacity, prices and order quantities over the entire test period

test_plan = pd.DataFrame({
    'Period_Start': biweekly['Period_Start'].loc[test_periods].dt.date.values,
    'Capacity_K': capacity,
    'Orders_received': adv_test,
    'Forecast_h1': D1_test,
    'Gap (h1 - K)': revision,
    'Price_Comfort': opt_price[:, 0],
    'Price_Tough': opt_price[:, 1],
    'Price_Ultimate': opt_price[:, 2],
    'Qty_Comfort': opt_qty[:, 0],
    'Qty_Tough': opt_qty[:, 1],
    'Qty_Ultimate': opt_qty[:, 2],
    'Actual_Demand': D_actual}, index=test_periods)
pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 20)
print(test_plan.round(1).to_string())

print('Totals over the test period:')
print(test_plan[['Capacity_K', 'Orders_received', 'Forecast_h1', 'Qty_Comfort', 'Qty_Tough', 'Qty_Ultimate', 'Actual_Demand']].sum().round(1))

# Part 3 – Step 8. Evaluation with the actual test demand

# The actual demand of a test period is the demand at the reference prices (it is what the OLS models forecast). At the chosen prices it becomes $D^{actual}\,P_i(p)/\theta$ for bike $i$. 
# Sales of bike $i$ = min(order, demand). Revenue = sales × price (no costs are assumed). Three plans are compared:

# * **(A) Model plan:** capacity $K=\hat D^{(4)}$, then prices and orders adjusted with $\hat D^{(1)}$ (Step 6).
# * **(B) Quantity update only:** same capacity, reference prices, total order = min($K$, $\hat D^{(1)}$) split by the buyer shares. The update is used for the quantity but not for the price.
# * **(C) Hindsight:** capacity equal to the actual demand and reference prices, i.e. what a perfect forecast would allow without using prices.
# * **(A′) Model plan, advance orders fixed:** the same plan as (A), but the orders that were already received two weeks ahead were placed at the old prices 
# and do **not** react to the new price; only the rest of the demand does. This is the stricter test of the price adjustment.

# Mismatch = lost demand + unsold bikes + reserved capacity that was not ordered.

# (A) Model plan
demand_A = D_actual[:, None] * P_opt / theta_buy              # demand of every bike at the chosen prices
sold_A = np.minimum(opt_qty, demand_A)
revenue_A = (sold_A * opt_price).sum()

# (A') Same plan, but the advance orders keep the reference product mix and do not react to the new prices
demand_A2 = adv_test[:, None] * share_buyers + (D_actual - adv_test)[:, None] * P_opt / theta_buy
sold_A2 = np.minimum(opt_qty, demand_A2)
revenue_A2 = (sold_A2 * opt_price).sum()

# (B) Reference prices, order = min(capacity, updated forecast), split by the buyer shares
order_B = np.minimum(capacity, D1_test)[:, None] * share_buyers
demand_B = D_actual[:, None] * share_buyers
sold_B = np.minimum(order_B, demand_B)
revenue_B = (sold_B * p0).sum()

# (C) Hindsight: capacity = order = actual demand, reference prices
sold_C = D_actual[:, None] * share_buyers
revenue_C = (sold_C * p0).sum()

evaluation = pd.DataFrame({
    'Revenue': [revenue_A, revenue_A2, revenue_B, revenue_C],
    'Units_sold': [sold_A.sum(), sold_A2.sum(), sold_B.sum(), sold_C.sum()],
    'Lost_demand': [(demand_A - sold_A).sum(), (demand_A2 - sold_A2).sum(), (demand_B - sold_B).sum(), 0.0],
    'Unsold_bikes': [(opt_qty - sold_A).sum(), (opt_qty - sold_A2).sum(), (order_B - sold_B).sum(), 0.0],
    'Unused_capacity': [(capacity - opt_qty.sum(axis=1)).sum(), (capacity - opt_qty.sum(axis=1)).sum(), (capacity - order_B.sum(axis=1)).sum(), 0.0]},
    index=['(A) Model: price and quantity update', "(A') Model, advance orders fixed", '(B) Quantity update only', '(C) Hindsight (perfect forecast)'])
evaluation['Mismatch'] = evaluation['Lost_demand'] + evaluation['Unsold_bikes'] + evaluation['Unused_capacity']
print(evaluation.round(1))
print('Revenue gain over (B):  (A)', round(revenue_A / revenue_B - 1, 4), " (A')", round(revenue_A2 / revenue_B - 1, 4))
print('Mismatch reduction vs (B):  (A)', round(1 - evaluation.iloc[0]['Mismatch'] / evaluation.iloc[2]['Mismatch'], 4),
      " (A')", round(1 - evaluation.iloc[1]['Mismatch'] / evaluation.iloc[2]['Mismatch'], 4))
print('Average price paid per bike sold (A):', round(revenue_A / sold_A.sum(), 1), ' (B):', round(revenue_B / sold_B.sum(), 1))

# ### Part 3 – Step 9. Sensitivity to the assumed share of customers who buy ($\theta$)

# $\theta$ is the only quantity that the purchase data cannot provide. The loop repeats the calculation of Steps 4–8 for three values of $\theta$.

sens_rows = []
for theta_s in [0.30, 0.50, 0.70]:                       # loop only repeats the same calculation for three values of theta
    c_s = np.log(theta_s / (1 - theta_s)) - np.log(np.exp(V0).sum())
    S_raw = theta_s * capacity / D1_test
    S_s = np.minimum(S_raw, 0.999)
    delta_s = np.log((S_s / (1 - S_s)) / (theta_s / (1 - theta_s)))
    price_s = p0 + delta_s[:, None] / beta_hat
    V_s = c_s + alpha_hat + beta_hat * price_s
    P_s = np.exp(V_s) / (1 + np.exp(V_s).sum(axis=1, keepdims=True))
    qty_s = D1_test[:, None] / theta_s * P_s
    dem_s = D_actual[:, None] * P_s / theta_s
    sold_s = np.minimum(qty_s, dem_s)
    dem_s2 = adv_test[:, None] * share_buyers + (D_actual - adv_test)[:, None] * P_s / theta_s     # advance orders fixed (A')
    sold_s2 = np.minimum(qty_s, dem_s2)
    V_up = c_s + alpha_hat + beta_hat * p0 * 1.01         # all prices +1%
    elasticity = (np.exp(V_up).sum() / (1 + np.exp(V_up).sum())) / theta_s - 1
    sens_rows.append([theta_s, (S_raw >= 1).sum(), elasticity * 100,
                      *(np.abs(price_s / p0 - 1).mean(axis=0) * 100), *(np.abs(price_s / p0 - 1).max(axis=0) * 100),
                      (sold_s * price_s).sum(), (sold_s * price_s).sum() / revenue_B - 1,
                      (dem_s - sold_s).sum() + (qty_s - sold_s).sum() + (capacity - qty_s.sum(axis=1)).sum(),
                      (sold_s2 * price_s).sum() / revenue_B - 1,
                      (dem_s2 - sold_s2).sum() + (qty_s - sold_s2).sum() + (capacity - qty_s.sum(axis=1)).sum()])

sensitivity = pd.DataFrame(sens_rows, columns=['theta', 'Periods_cap_not_fillable', 'Demand_change_for_+1%_prices_(%)',
                                               'Avg_abs_change_Comfort_%', 'Avg_abs_change_Tough_%', 'Avg_abs_change_Ultimate_%',
                                               'Max_abs_change_Comfort_%', 'Max_abs_change_Tough_%', 'Max_abs_change_Ultimate_%',
                                               'Revenue_A', 'Revenue_gain_vs_B', 'Mismatch_A',
                                               "Revenue_gain_vs_B_(A')", "Mismatch_A'"])
print(sensitivity.round(3).T.to_string(header=False))

# Part 3 – Step 10. Check: why the product mix is kept

# The capacity is counted in bikes, no costs are assumed, and Comfort Mountain has the highest price. 
# Letting the three prices move freely to maximize revenue would therefore push demand towards Comfort Mountain alone. 
# The cell shows this corner solution: Tough Track and Ultimate Mountain are set to the highest prices ever observed, 
# and the Comfort Mountain price is lowered by bisection until expected demand equals the capacity.

top_T = df.loc[train_mask, 'Price_Tough_Track'].max()
top_U = df.loc[train_mask, 'Price_Ultimate_Mountain'].max()
low_C = np.full(n_test, df.loc[train_mask, 'Price_Comfort_Mountain'].min())
high_C = np.full(n_test, df.loc[train_mask, 'Price_Comfort_Mountain'].max())

for iteration in range(60):                              # bisection on the Comfort Mountain price (demand falls when the price rises)
    mid_C = (low_C + high_C) / 2
    V_mid = c_hat + alpha_hat + beta_hat * np.column_stack([mid_C, np.full(n_test, top_T), np.full(n_test, top_U)])
    demand_mid = D1_test / theta_buy * np.exp(V_mid).sum(axis=1) / (1 + np.exp(V_mid).sum(axis=1))
    low_C = np.where(demand_mid > capacity, mid_C, low_C)
    high_C = np.where(demand_mid > capacity, high_C, mid_C)

free_price = np.column_stack([mid_C, np.full(n_test, top_T), np.full(n_test, top_U)])
V_free = c_hat + alpha_hat + beta_hat * free_price
P_free = np.exp(V_free) / (1 + np.exp(V_free).sum(axis=1, keepdims=True))
qty_free = D1_test[:, None] / theta_buy * P_free
sold_free = np.minimum(qty_free, D_actual[:, None] * P_free / theta_buy)
print('Comfort Mountain price in this alternative: min', mid_C.min().round(0), ' max', mid_C.max().round(0), '(2640 = lowest training price = lower bound of the search)')
print('Share of Comfort Mountain in the planned units:', round(qty_free[:, 0].sum() / qty_free.sum(), 4), '  (model plan A:', round(opt_qty[:, 0].sum() / opt_qty.sum(), 4), ')')
print('Revenue of this alternative:', round((sold_free * free_price).sum(), 0), ' vs. model plan A:', round(revenue_A, 0))

# Part 3 – Step 11. Chart: capacity, updated forecast, actual demand and price adjustment

# %%
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.6), sharex=True, gridspec_kw={'height_ratios': [1.15, 1]})

ax1.plot(test_plan['Period_Start'], test_plan['Capacity_K'], color='#2a78d6', linewidth=2, label='Reserved capacity K (h = 4 forecast, 8 weeks ahead)')
ax1.plot(test_plan['Period_Start'], test_plan['Forecast_h1'], color='#eb6834', linewidth=2, label='Updated forecast (h = 1, 2 weeks ahead)')
ax1.plot(test_plan['Period_Start'], test_plan['Actual_Demand'], color='#9a9a94', linewidth=1.2, label='Actual total demand')
ax1.set_title('Test period: capacity, updated forecast and actual demand', loc='left', fontsize=12)
ax1.set_ylabel('Units per two-week period')
ax1.legend(frameon=False, loc='upper right', fontsize=9)

ax2.axhline(0, color='#9a9a94', linewidth=0.8)
ax2.plot(test_plan['Period_Start'], (opt_price[:, 0] / p0[0] - 1) * 100, color='#2a78d6', linewidth=2, label='Comfort Mountain')
ax2.plot(test_plan['Period_Start'], (opt_price[:, 1] / p0[1] - 1) * 100, color='#eb6834', linewidth=2, label='Tough Track')
ax2.plot(test_plan['Period_Start'], (opt_price[:, 2] / p0[2] - 1) * 100, color='#9a9a94', linewidth=2, label='Ultimate Mountain')
ax2.set_title('Price set two weeks ahead: change from the reference price', loc='left', fontsize=12)
ax2.set_ylim(-80, 22)                                    # the Ultimate Mountain price falls by 68% in the one capped period
ax2.set_ylabel('Price change (%)')
ax2.set_xlabel('Start of period')
ax2.legend(frameon=False, loc='upper right', fontsize=9, ncol=3)
ax2.annotate('capacity exceeds all interested customers\n(purchase probability capped at 99.9%)',
             xy=(test_plan['Period_Start'].iloc[18], (opt_price[18, 2] / p0[2] - 1) * 100), xytext=(test_plan['Period_Start'].iloc[8], -64),
             fontsize=8, color='#555550', arrowprops=dict(arrowstyle='-', color='#9a9a94'))

for ax in (ax1, ax2):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#e6e6e2', linewidth=0.8)
plt.tight_layout()
plt.savefig('figures/capacity_prices.png', dpi=200)
plt.show()
