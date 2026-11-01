"""Forecast-safe Ridge-ARX baseline using lagged target and exogenous history."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class RidgeARXConfig:
    lookback_grid: tuple[int, ...] = (1, 3, 6, 12)
    alpha_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)
    fit_intercept: bool = True

    def __post_init__(self) -> None:
        if not self.lookback_grid or any(value < 1 for value in self.lookback_grid):
            raise ValueError("Ridge-ARX lookback grid must contain positive integers.")
        if not self.alpha_grid or any(value < 0 for value in self.alpha_grid):
            raise ValueError("Ridge-ARX alpha grid must contain nonnegative values.")


@dataclass(frozen=True)
class RidgeARXWindows:
    X_fit: np.ndarray
    y_fit: np.ndarray
    X_pred: np.ndarray
    actual: np.ndarray
    dates: pd.DatetimeIndex


def prepare_ridge_arx_windows(
    frame: pd.DataFrame,
    *,
    target: str,
    fit_end: int,
    prediction_start: int,
    prediction_end: int,
    lookback: int,
) -> RidgeARXWindows:
    """Create flattened causal windows, ending at t-1 for a target at t."""
    if target not in frame:
        raise ValueError(f"Ridge-ARX target is missing: {target}")
    if not (1 <= lookback < fit_end <= prediction_start < prediction_end <= len(frame)):
        raise ValueError("Invalid chronological Ridge-ARX fit/prediction boundaries.")
    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Ridge-ARX input contains missing or nonfinite values.")
    target_column = frame.columns.get_loc(target)

    def windows(start: int, end: int) -> np.ndarray:
        return np.stack([
            values[index - lookback:index].reshape(-1)
            for index in range(start, end)
        ])

    return RidgeARXWindows(
        X_fit=windows(lookback, fit_end),
        y_fit=values[lookback:fit_end, target_column],
        X_pred=windows(prediction_start, prediction_end),
        actual=values[prediction_start:prediction_end, target_column],
        dates=pd.DatetimeIndex(frame.index[prediction_start:prediction_end]),
    )


def fit_predict_ridge_arx(
    frame: pd.DataFrame,
    *,
    target: str,
    fit_end: int,
    prediction_start: int,
    prediction_end: int,
    lookback: int,
    alpha: float,
    config: RidgeARXConfig,
) -> np.ndarray:
    """Fit Ridge-ARX before ``fit_end`` and predict causal one-step windows."""
    windows = prepare_ridge_arx_windows(
        frame,
        target=target,
        fit_end=fit_end,
        prediction_start=prediction_start,
        prediction_end=prediction_end,
        lookback=lookback,
    )
    model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=float(alpha), fit_intercept=config.fit_intercept),
    )
    model.fit(windows.X_fit, windows.y_fit)
    predicted = np.asarray(model.predict(windows.X_pred), dtype=float).reshape(-1)
    if len(predicted) != len(windows.dates) or not np.isfinite(predicted).all():
        raise ValueError("Ridge-ARX produced incomplete or nonfinite forecasts.")
    return predicted
