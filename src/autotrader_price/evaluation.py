"""Evaluation helpers with metrics in interpretable target units."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_regressor(model, X, y) -> dict[str, float]:
    """Return R², MAE and RMSE for a fitted regressor."""
    predictions = model.predict(X)
    return {
        "r2": float(r2_score(y, predictions)),
        "mae": float(mean_absolute_error(y, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y, predictions))),
    }
