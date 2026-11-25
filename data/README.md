# Data

The dataset is **not included** in this repository. To run the model, save the transaction file as `data/sales_transactions.csv`.

One row per sales transaction, with these columns:

| Column | Type | Description |
|---|---|---|
| `order_date` | date | Date the customer placed the order |
| `delivery_date` | date | Date the bike was delivered (demand is assigned to the delivery period) |
| `Product` | text | `Comfort Mountain`, `Tough Track` or `Ultimate Mountain` |
| `quantity` | integer | Units in the transaction |
| `Price_Comfort_Mountain` | number | Posted price of Comfort Mountain at the time of the order |
| `Price_Tough_Track` | number | Posted price of Tough Track at the time of the order |
| `Price_Ultimate_Mountain` | number | Posted price of Ultimate Mountain at the time of the order |

The script was built for orders from 2021-02-07 to 2026-01-31 (129 complete two-week delivery periods; the last 26 are used as the test year). With a different date range, update `n_train` and the period count `130` in Part 3 of `bike_demand_model.py`.
