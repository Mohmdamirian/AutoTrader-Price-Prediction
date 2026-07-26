import pandas as pd
import pytest

from autotrader_price.data import clean_vehicle_data, registration_year


def sample_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "public_reference": [1, 2, 3],
            "mileage": [10_000, 250_000, None],
            "reg_code": ["61", "19", "08"],
            "standard_colour": ["Black", "Blue", "Red"],
            "standard_make": ["A", "B", "C"],
            "standard_model": ["One", "Two", "Three"],
            "vehicle_condition": ["USED", "USED", "USED"],
            "year_of_registration": [None, 2019, None],
            "price": [10_000, 20_000, 8_000],
            "body_type": ["SUV", "Saloon", "Hatchback"],
            "crossover_car_and_van": [False, False, False],
            "fuel_type": ["Petrol", "Diesel", "Petrol"],
        }
    )


def test_registration_year_handles_two_digit_codes():
    assert registration_year("08") == 2008
    assert registration_year("61") == 2011
    assert registration_year(None) is None


def test_cleaning_derives_year_and_applies_domain_filter():
    cleaned = clean_vehicle_data(sample_data())

    assert len(cleaned) == 2
    assert cleaned.loc[0, "year_of_registration"] == 2011
    assert cleaned.loc[1, "year_of_registration"] == 2008
    assert "public_reference" not in cleaned
    assert "crossover_car_and_van" not in cleaned


def test_missing_columns_raise_clear_error():
    with pytest.raises(ValueError, match="missing required columns"):
        clean_vehicle_data(pd.DataFrame({"price": [1]}))
