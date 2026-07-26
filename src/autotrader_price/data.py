"""Data loading and deterministic cleaning."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "price"
DROP_COLUMNS = ("public_reference", "crossover_car_and_van")
EXPECTED_COLUMNS = {
    "public_reference",
    "mileage",
    "reg_code",
    "standard_colour",
    "standard_make",
    "standard_model",
    "vehicle_condition",
    "year_of_registration",
    TARGET,
    "body_type",
    "crossover_car_and_van",
    "fuel_type",
}


def registration_year(reg_code: object) -> float | None:
    """Convert the two-digit age identifier in a UK registration code to a year."""
    if pd.isna(reg_code):
        return None

    code = str(reg_code).strip().zfill(2)
    if len(code) < 2 or not code[:2].isdigit():
        return None

    identifier = int(code[:2])
    if 0 <= identifier <= 49:
        return float(2000 + identifier)
    if 50 <= identifier <= 99:
        return float(1950 + identifier)
    return None


def clean_vehicle_data(data: pd.DataFrame) -> pd.DataFrame:
    """Apply row-level domain checks without learning from the full dataset."""
    missing = EXPECTED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    cleaned = data.copy()
    derived_year = cleaned["reg_code"].map(registration_year)
    cleaned["year_of_registration"] = cleaned["year_of_registration"].fillna(
        derived_year
    )

    valid_year = cleaned["year_of_registration"].isna() | (
        cleaned["year_of_registration"] >= 2000
    )
    valid_mileage = cleaned["mileage"].isna() | (cleaned["mileage"] <= 200_000)
    valid_price = cleaned[TARGET].notna() & (cleaned[TARGET] > 0)

    cleaned = cleaned.loc[valid_year & valid_mileage & valid_price]
    cleaned = cleaned.drop(columns=list(DROP_COLUMNS))
    return cleaned.reset_index(drop=True)


def load_vehicle_data(path: str | Path) -> pd.DataFrame:
    """Load and validate the expected CSV."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. See data/README.md for setup."
        )
    return clean_vehicle_data(pd.read_csv(path))


def split_features_target(
    data: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a reproducible hold-out split before learned preprocessing."""
    X = data.drop(columns=TARGET)
    y = data[TARGET]
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )
