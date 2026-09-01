from __future__ import annotations

import numpy as np


def evaluate_forecast(y_true, y_pred) -> dict[str, float]:
    """Tính lỗi và directional accuracy cho rolling one-step forecast.

    Direction tại ``t`` được đo từ actual ``t-1`` tới actual/prediction ``t``.
    Điểm đầu tiên không có previous actual trong mảng nên không tham gia metric.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) < 1e-12, np.nan, y_true)
    if len(y_true) > 1:
        actual_direction = np.sign(np.diff(y_true))
        predicted_direction = np.sign(y_pred[1:] - y_true[:-1])
        directional_accuracy = float((actual_direction == predicted_direction).mean() * 100.0)
    else:
        directional_accuracy = float("nan")
    centered_sum = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r_squared = (
        float(1.0 - np.sum((y_true - y_pred) ** 2) / centered_sum)
        if centered_sum > 1e-12
        else float("nan")
    )
    return {
        "RMSE": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "MAE": float(np.mean(np.abs(y_true - y_pred))),
        "MAPE": float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100.0),
        "R2": r_squared,
        "Directional_Accuracy": directional_accuracy,
    }
