"""Forecast-safe scikit-learn MLP-ARX baseline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class MLPARXConfig:
    lookback_grid: tuple[int, ...] = (6, 12)
    hidden_layer_grid: tuple[tuple[int, ...], ...] = ((32,), (64,), (32, 16))
    alpha_grid: tuple[float, ...] = (0.0001, 0.001, 0.01)
    learning_rate_init: float = 0.001
    max_iter: int = 300
    batch_size: int = 16
    early_stopping: bool = False
    random_state: int = 7

    def __post_init__(self) -> None:
        if not self.lookback_grid or any(v < 1 for v in self.lookback_grid):
            raise ValueError("MLP-ARX lookback grid must be positive.")
        if not self.hidden_layer_grid or any(
            not layers or any(v < 1 for v in layers) for layers in self.hidden_layer_grid
        ):
            raise ValueError("MLP-ARX hidden-layer grid is invalid.")
        if not self.alpha_grid or any(v < 0 for v in self.alpha_grid):
            raise ValueError("MLP-ARX alpha grid must be nonnegative.")
        if self.learning_rate_init <= 0 or self.max_iter < 1 or self.batch_size < 1:
            raise ValueError("MLP-ARX training parameters are invalid.")


def _windows(frame: pd.DataFrame, target: str, fit_end: int,
             prediction_start: int, prediction_end: int, lookback: int):
    if target not in frame:
        raise ValueError(f"MLP-ARX target is missing: {target}")
    if not (1 <= lookback < fit_end <= prediction_start < prediction_end <= len(frame)):
        raise ValueError("Invalid chronological MLP-ARX boundaries.")
    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("MLP-ARX input contains missing or nonfinite values.")
    target_idx = frame.columns.get_loc(target)
    make_x = lambda start, end: np.stack([
        values[i - lookback:i].reshape(-1) for i in range(start, end)
    ])
    return (
        make_x(lookback, fit_end), values[lookback:fit_end, target_idx],
        make_x(prediction_start, prediction_end),
        pd.DatetimeIndex(frame.index[prediction_start:prediction_end]),
    )


def fit_predict_mlp_arx(frame: pd.DataFrame, *, target: str, fit_end: int,
                        prediction_start: int, prediction_end: int,
                        lookback: int, hidden_layers: tuple[int, ...], alpha: float,
                        seed: int, config: MLPARXConfig) -> np.ndarray:
    X_fit, y_fit, X_pred, dates = _windows(
        frame, target, fit_end, prediction_start, prediction_end, lookback
    )
    model = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=tuple(hidden_layers), alpha=float(alpha),
            learning_rate_init=config.learning_rate_init, max_iter=config.max_iter,
            batch_size=config.batch_size, early_stopping=config.early_stopping,
            shuffle=False, random_state=int(seed), tol=1e-5,
        ),
    )
    model.fit(X_fit, y_fit)
    predicted = np.asarray(model.predict(X_pred), dtype=float).reshape(-1)
    if len(predicted) != len(dates) or not np.isfinite(predicted).all():
        raise ValueError("MLP-ARX produced incomplete or nonfinite forecasts.")
    return predicted
