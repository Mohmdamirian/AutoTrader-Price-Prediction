import pandas as pd

from autotrader_price.preprocessing import make_preprocessor


def test_preprocessor_returns_finite_numeric_matrix():
    X = pd.DataFrame(
        {
            "reg_code": ["18", "19", "20", "21", "22", "23"],
            "standard_colour": ["Black", "Blue", "Black", "Red", "Blue", "Red"],
            "standard_make": ["A", "A", "B", "B", "C", "C"],
            "standard_model": ["One", "Two", "One", "Two", "One", "Two"],
            "body_type": ["SUV", "SUV", "Saloon", "Saloon", "SUV", "Saloon"],
            "fuel_type": ["Petrol", "Diesel", "Petrol", "Diesel", "Petrol", "Diesel"],
            "vehicle_condition": ["USED"] * 6,
            "mileage": [10_000, 20_000, 30_000, 40_000, 50_000, None],
            "year_of_registration": [2018, 2019, 2020, 2021, 2022, 2023],
        }
    )
    y = pd.Series([10_000, 11_000, 12_000, 13_000, 14_000, 15_000])

    transformed = make_preprocessor().fit_transform(X, y)

    assert transformed.shape[0] == len(X)
    assert pd.notna(transformed).all()
