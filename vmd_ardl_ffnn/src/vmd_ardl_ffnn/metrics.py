from __future__ import annotations

import numpy as np


def evaluate_forecast(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) < 1e-12, np.nan, y_true)
    if len(y_true) > 1:
        actual_direction = np.sign(np.diff(y_true))
        predicted_direction = np.sign(np.diff(y_pred))
        directional_accuracy = float((actual_direction == predicted_direction).mean() * 100.0)
    else:
        directional_accuracy = float("nan")
    return {
        "RMSE": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "MAE": float(np.mean(np.abs(y_true - y_pred))),
        "MAPE": float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100.0),
        "Directional_Accuracy": directional_accuracy,
    }
