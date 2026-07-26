"""Leakage-aware preprocessing for mixed vehicle attributes."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    PowerTransformer,
    StandardScaler,
    TargetEncoder,
)

HIGH_CARDINALITY = [
    "reg_code",
    "standard_colour",
    "standard_make",
    "standard_model",
    "body_type",
]
LOW_CARDINALITY = ["fuel_type", "vehicle_condition"]
NUMERICAL = ["mileage", "year_of_registration"]
ALL_CATEGORICAL = HIGH_CARDINALITY + LOW_CARDINALITY


def make_preprocessor(*, scale_output: bool = False) -> Pipeline | ColumnTransformer:
    """Build a preprocessor that is fitted on training data inside a pipeline."""
    high_cardinality = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "target_encoder",
                TargetEncoder(
                    target_type="continuous",
                    cv=5,
                    shuffle=True,
                    random_state=42,
                ),
            ),
        ]
    )

    low_cardinality = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "one_hot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    numerical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("power", PowerTransformer(method="yeo-johnson")),
        ]
    )

    columns = ColumnTransformer(
        transformers=[
            ("high_cardinality", high_cardinality, HIGH_CARDINALITY),
            ("low_cardinality", low_cardinality, LOW_CARDINALITY),
            ("numerical", numerical, NUMERICAL),
        ],
        verbose_feature_names_out=False,
    )

    if not scale_output:
        return columns

    return Pipeline(
        steps=[
            ("columns", columns),
            ("scaler", StandardScaler()),
        ]
    )


def make_unsupervised_preprocessor() -> Pipeline:
    """Build a target-independent sparse representation for clustering."""
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "one_hot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=25,
                ),
            ),
        ]
    )
    numerical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("power", PowerTransformer(method="yeo-johnson")),
        ]
    )
    columns = ColumnTransformer(
        transformers=[
            ("categorical", categorical, ALL_CATEGORICAL),
            ("numerical", numerical, NUMERICAL),
        ],
        sparse_threshold=1.0,
    )
    return Pipeline(
        steps=[
            ("columns", columns),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )
