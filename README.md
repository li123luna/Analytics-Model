# Bike Demand Forecasting & Capacity-Pricing Model

A Python model that matches supply to demand for a retailer selling three bikes — **Comfort Mountain**, **Tough Track** and **Ultimate Mountain**. Starting from order-level sales transactions, it forecasts demand, estimates how prices drive the customer's choice between the bikes, and uses both to decide how much capacity to reserve, what prices to charge and how many units of each bike to order.

## What the model does

| Part | Question it answers | Method |
|---|---|---|
| 1. Demand forecasting | How many bikes will be delivered 2, 4, 6 and 8 weeks from now? | Four OLS regressions (one per horizon h = 1–4 two-week periods) on lagged demand, linear trend, annual seasonality (sine/cosine terms) and **advance demand** — orders already received for a future delivery period |
| 2. Product choice | Given the three posted prices, which bike will a customer choose? | Multinomial logistic regression (Comfort Mountain = base category), cross-checked with scikit-learn |
| 3. Capacity, pricing & ordering | How much capacity to reserve, and how to set prices and order quantities once newer information arrives? | Two-stage decision rule: reserve capacity from the 8-week forecast, then two weeks ahead adjust all three prices so that expected demand matches the committed capacity, keeping the product mix |

### Two-stage planning logic (Part 3)

| Decision point | Information available | Decision |
|---|---|---|
| 8 weeks before the selling period | Demand observed so far; no orders exist yet | Reserve capacity = h = 4 forecast of total demand |
| 2 weeks before the selling period | Recent demand **plus orders already received** for that period | Set prices and order quantities so that expected demand = reserved capacity |
| Selling period | Customers arrive and choose | Sales = min(order quantity, demand) |

Because the transactions only contain completed purchases, the choice model is extended with a **no-purchase option**. The share of interested customers who buy at reference prices (θ) cannot be identified from purchase data, so it is set to 50% and tested at 30% and 70% in a sensitivity analysis.

## Key results

- **Forecasting:** accuracy is highest for the shortest horizon (h = 1: R² = 0.810, RMSE = 5.68 units per two-week period) because most of the demand is already known from advance orders; accuracy decreases for longer horizons. Prices had no significant effect on *total* demand, so they are not used in the forecasting models.
- **Choice probabilities** at example prices of $4,200 / $1,950 / $500: Comfort Mountain 12.37%, Tough Track 61.25%, Ultimate Mountain 26.38%.
- **Planning (out-of-sample):** trained on the first 103 two-week periods and tested on the last 26 (one full year). Compared with a static benchmark at reference prices, adjusting prices with the updated forecast increases total revenue while keeping the product mix and limiting excess inventory, unused capacity and lost demand. The script prints the full evaluation table, including a stricter test in which orders already received do not react to the new prices.

Full write-up: [`docs/Bike_Demand_Model_Report.pdf`](docs/Bike_Demand_Model_Report.pdf)

## Repository structure

```
├── bike_demand_model.py        # the full model (Parts 1–3)
├── data/
│   └── README.md               # required input columns (dataset not included)
├── docs/
│   ├── Bike_Demand_Model_Report.pdf
│   └── Bike_Demand_Model_Report.docx
├── figures/                    # charts created when the script runs
├── web/                        # Django project skeleton for a future web front end
├── requirements.txt
└── README.md
```

## How to run

```bash
git clone https://github.com/<your-username>/bike-demand-forecasting-pricing.git
cd bike-demand-forecasting-pricing
pip install -r requirements.txt

# put the transaction data at data/sales_transactions.csv (see data/README.md), then:
python bike_demand_model.py
```

The script prints the model summaries, diagnostics (Durbin–Watson, Breusch–Pagan, Jarque–Bera, HC3 robust standard errors) and the evaluation tables, and saves two charts to `figures/`:

- `actual_vs_fitted.png` — actual demand vs. fitted values for h = 1 and h = 4
- `capacity_prices.png` — reserved capacity, updated forecast, actual demand and the price adjustments over the test year

The file also contains `# %%` cell markers, so it can be run cell by cell in VS Code or Jupyter.

## Web skeleton (`web/`)

An empty Django project (admin only) intended as the starting point for a web front end. The secret key is **not** stored in the repository; it is read from environment variables:

| Variable | Default | Production |
|---|---|---|
| `DJANGO_SECRET_KEY` | random key generated at start-up | set a fixed secret value |
| `DJANGO_DEBUG` | `True` | `False` |
| `DJANGO_ALLOWED_HOSTS` | `localhost,127.0.0.1` | your domain |

```bash
cd web
python manage.py migrate
python manage.py runserver
```

## Assumptions and limitations

- The share of interested customers who buy (θ) is an assumption, not an estimate; results are shown for θ = 30%, 50% and 70%.
- Prices enter the choice model through own-price utilities only; a likelihood-ratio test against the full model (all three prices in every equation) is reported in the script.
- In a period where the reserved capacity exceeds all interested customers, the required price cut falls outside the price range seen in the training data.
- OLS forecasts can be negative in the low season; negative forecasts are set to zero.

## Tech stack

Python · pandas · NumPy · statsmodels · SciPy · scikit-learn · Matplotlib · Django
