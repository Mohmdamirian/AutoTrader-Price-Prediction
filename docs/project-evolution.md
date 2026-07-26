# Project evolution

This repository contains two related stages of one vehicle-price modelling
project. They should be read as a progression, not as independent or competing
solutions.

## Stage 1: Machine Learning Concepts

The MLC stage established the problem and baseline workflow:

1. explored 402,005 anonymised vehicle adverts;
2. investigated mileage, registration year and make;
3. cleaned inconsistent values and missing data;
4. encoded mixed categorical and numerical inputs;
5. compared linear regression, KNN and decision-tree regression;
6. tuned model hyperparameters with cross-validation; and
7. interpreted predictors using permutation importance.

The recorded best baseline was a decision tree with a test R² of 0.847.

## Stage 2: Advanced Machine Learning

The AML stage extended the same data and target with:

- recursive and sequential feature selection;
- Random Forest and XGBoost;
- a stacked ensemble with a linear meta-learner;
- permutation importance and SHAP;
- partial dependence and individual conditional expectation;
- PCA and Isomap; and
- exploratory K-Means clustering.

The recorded stacking result was a test R² of 0.862 and test MAE of 1,839 in
the dataset's price units.

## Portfolio refactor

The original submissions were designed around assessment requirements. The
portfolio edition retains their analytical identity but makes the following
engineering changes:

| Original issue | Portfolio treatment |
|---|---|
| Learned transformations were applied before splitting the data | Split first; fit imputers, encoders and scalers on training data only |
| MLC fine-grained errors were simulated | Calculate errors from the fitted models |
| PCA and Isomap used differently scaled inputs | Standardise transformed inputs before projection |
| Clusters were described as feature engineering but not used downstream | Describe clustering as exploratory and avoid target-dependent encoding |
| Notebooks contained machine-specific warning paths | Clear outputs and use repository-relative paths |
| Reports and filenames exposed a student number | Include privacy-clean portfolio copies |

The recorded academic metrics are retained for transparency. Because the
preprocessing has been corrected, rerunning the portfolio notebooks may produce
different results.
