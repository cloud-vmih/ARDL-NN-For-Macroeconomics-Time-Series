"""Forecast-safe multivariate XGBoost baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class XGBoostConfig:
    lookback_grid: tuple[int, ...] = (6, 12)
    max_depth_grid: tuple[int, ...] = (2, 3)
    learning_rate_grid: tuple[float, ...] = (0.03, 0.1)
    n_estimators: int = 200
    min_child_weight: float = 1.0
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    n_jobs: int = 1

    def __post_init__(self) -> None:
        if not self.lookback_grid or any(value < 1 for value in self.lookback_grid):
            raise ValueError("XGBoost lookback grid must contain positive integers.")
        if not self.max_depth_grid or any(value < 1 for value in self.max_depth_grid):
            raise ValueError("XGBoost max-depth grid must contain positive integers.")
        if not self.learning_rate_grid or any(value <= 0 for value in self.learning_rate_grid):
            raise ValueError("XGBoost learning-rate grid must contain positive values.")
        if self.n_estimators < 1 or self.min_child_weight < 0:
            raise ValueError("XGBoost tree counts/weights are invalid.")
        if not 0 < self.subsample <= 1 or not 0 < self.colsample_bytree <= 1:
            raise ValueError("XGBoost sampling fractions must be in (0, 1].")
        if self.n_jobs < 1:
            raise ValueError("XGBoost n_jobs must be positive.")


@dataclass(frozen=True)
class XGBoostWindows:
    X_fit: np.ndarray
    y_fit: np.ndarray
    X_pred: np.ndarray
    actual: np.ndarray
    dates: pd.DatetimeIndex


def prepare_xgboost_windows(
    frame: pd.DataFrame,
    *,
    target: str,
    fit_end: int,
    prediction_start: int,
    prediction_end: int,
    lookback: int,
) -> XGBoostWindows:
    """Flatten causal multivariate windows ending immediately before target time."""
    if target not in frame:
        raise ValueError(f"XGBoost target is missing: {target}")
    if not (1 <= lookback < fit_end <= prediction_start < prediction_end <= len(frame)):
        raise ValueError("Invalid chronological XGBoost fit/prediction boundaries.")
    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("XGBoost input contains missing or nonfinite values.")
    target_column = frame.columns.get_loc(target)

    def windows(start: int, end: int) -> np.ndarray:
        return np.stack([
            values[index - lookback:index].reshape(-1)
            for index in range(start, end)
        ])

    return XGBoostWindows(
        X_fit=windows(lookback, fit_end),
        y_fit=values[lookback:fit_end, target_column],
        X_pred=windows(prediction_start, prediction_end),
        actual=values[prediction_start:prediction_end, target_column],
        dates=pd.DatetimeIndex(frame.index[prediction_start:prediction_end]),
    )


def _xgb_regressor() -> Any:
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError(
            "XGBoost is required for the XGBoost comparator. Install project dependencies."
        ) from exc
    return XGBRegressor


def fit_predict_xgboost(
    frame: pd.DataFrame,
    *,
    target: str,
    fit_end: int,
    prediction_start: int,
    prediction_end: int,
    lookback: int,
    max_depth: int,
    learning_rate: float,
    seed: int,
    xgboost_config: XGBoostConfig,
) -> np.ndarray:
    """Fit XGBoost on causal windows and predict one-step windows in level scale."""
    windows = prepare_xgboost_windows(
        frame,
        target=target,
        fit_end=fit_end,
        prediction_start=prediction_start,
        prediction_end=prediction_end,
        lookback=lookback,
    )
    regressor = _xgb_regressor()
    model = regressor(
        objective="reg:squarederror",
        n_estimators=xgboost_config.n_estimators,
        max_depth=int(max_depth),
        learning_rate=float(learning_rate),
        min_child_weight=xgboost_config.min_child_weight,
        subsample=xgboost_config.subsample,
        colsample_bytree=xgboost_config.colsample_bytree,
        reg_alpha=xgboost_config.reg_alpha,
        reg_lambda=xgboost_config.reg_lambda,
        random_state=int(seed),
        n_jobs=xgboost_config.n_jobs,
        tree_method="hist",
        verbosity=0,
    )
    model.fit(windows.X_fit, windows.y_fit, verbose=False)
    predicted = np.asarray(model.predict(windows.X_pred), dtype=float).reshape(-1)
    if len(predicted) != len(windows.dates) or not np.isfinite(predicted).all():
        raise ValueError("XGBoost produced incomplete or nonfinite forecasts.")
    return predicted
