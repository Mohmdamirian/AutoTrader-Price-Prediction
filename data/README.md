# Data

The dataset is deliberately excluded from version control.

## Expected file

Download the AutoTrader car sale adverts dataset from
[Kaggle](https://www.kaggle.com/datasets/shayanshahid997/autotrader-at-car-sale-adverts-dataset),
review the current dataset terms, and save it as:

```text
data/raw/adverts.csv
```

Do not commit the CSV to GitHub.

## Expected schema

| Column | Role |
|---|---|
| `public_reference` | Listing identifier; removed before modelling |
| `mileage` | Numerical predictor |
| `reg_code` | Registration code; categorical predictor |
| `standard_colour` | Categorical predictor |
| `standard_make` | High-cardinality categorical predictor |
| `standard_model` | High-cardinality categorical predictor |
| `vehicle_condition` | Categorical predictor |
| `year_of_registration` | Numerical predictor |
| `price` | Regression target |
| `body_type` | Categorical predictor |
| `crossover_car_and_van` | Boolean field; excluded in the academic workflow |
| `fuel_type` | Categorical predictor |

The loader validates these columns and raises an explicit error when any are
missing.

## Cleaning policy

- Derive missing registration years from valid two-digit UK registration codes
  where possible.
- Keep listings from registration year 2000 onwards.
- Keep mileage at or below 200,000.
- Keep positive target prices.
- Split the data before fitting imputers, encoders or scalers.

These choices are documented modelling decisions, not universal rules for
vehicle valuation.
