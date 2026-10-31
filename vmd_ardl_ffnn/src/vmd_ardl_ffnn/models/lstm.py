"""Causal multivariate LSTM baseline for monthly one-step forecasts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ..config import TensorFlowRuntimeConfig
from .tensorflow_runtime import bootstrap_tensorflow


@dataclass(frozen=True)
class LSTMConfig:
    lookback_grid: tuple[int, ...] = (6, 12)
    hidden_units_grid: tuple[int, ...] = (16, 32)
    learning_rate_grid: tuple[float, ...] = (0.001,)
    epochs: int = 60
    batch_size: int = 16
    patience: int = 6

    def __post_init__(self) -> None:
        if not self.lookback_grid or any(value < 1 for value in self.lookback_grid):
            raise ValueError("LSTM lookback grid must contain positive integers.")
        if not self.hidden_units_grid or any(value < 1 for value in self.hidden_units_grid):
            raise ValueError("LSTM hidden-unit grid must contain positive integers.")
        if not self.learning_rate_grid or any(value <= 0 for value in self.learning_rate_grid):
            raise ValueError("LSTM learning-rate grid must contain positive values.")
        if min(self.epochs, self.batch_size, self.patience) < 1:
            raise ValueError("LSTM training counts must be positive.")


@dataclass(frozen=True)
class LSTMWindows:
    X_fit: np.ndarray
    y_fit: np.ndarray
    X_pred: np.ndarray
    actual: np.ndarray
    dates: pd.DatetimeIndex
    scaler: StandardScaler
    target_column: int


def prepare_lstm_windows(
    frame: pd.DataFrame,
    *,
    target: str,
    fit_end: int,
    prediction_start: int,
    prediction_end: int,
    lookback: int,
) -> LSTMWindows:
    """Fit the scaler on history and build windows ending before each target date."""
    if target not in frame:
        raise ValueError(f"LSTM target is missing: {target}")
    if not (1 <= lookback < fit_end <= prediction_start < prediction_end <= len(frame)):
        raise ValueError("Invalid chronological LSTM fit/prediction boundaries.")
    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("LSTM input contains missing or nonfinite values.")
    scaler = StandardScaler().fit(values[:fit_end])
    scaled = scaler.transform(values)
    target_column = frame.columns.get_loc(target)

    def windows(start: int, end: int) -> np.ndarray:
        return np.stack([scaled[index - lookback:index] for index in range(start, end)])

    return LSTMWindows(
        X_fit=windows(lookback, fit_end),
        y_fit=scaled[lookback:fit_end, target_column],
        X_pred=windows(prediction_start, prediction_end),
        actual=values[prediction_start:prediction_end, target_column],
        dates=pd.DatetimeIndex(frame.index[prediction_start:prediction_end]),
        scaler=scaler,
        target_column=target_column,
    )


def fit_predict_lstm(
    frame: pd.DataFrame,
    *,
    target: str,
    fit_end: int,
    prediction_start: int,
    prediction_end: int,
    lookback: int,
    hidden_units: int,
    learning_rate: float,
    seed: int,
    lstm_config: LSTMConfig,
    runtime: TensorFlowRuntimeConfig,
) -> tuple[np.ndarray, int]:
    """Train on observations before fit_end and predict later one-step origins."""
    windows = prepare_lstm_windows(
        frame,
        target=target,
        fit_end=fit_end,
        prediction_start=prediction_start,
        prediction_end=prediction_end,
        lookback=lookback,
    )
    tf, device, _ = bootstrap_tensorflow(runtime)
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(int(seed))
    with tf.device(device):
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(lookback, frame.shape[1])),
            tf.keras.layers.LSTM(int(hidden_units)),
            tf.keras.layers.Dense(1),
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=float(learning_rate)),
            loss="mse",
        )
        history = model.fit(
            windows.X_fit,
            windows.y_fit,
            epochs=lstm_config.epochs,
            batch_size=lstm_config.batch_size,
            shuffle=False,
            verbose=0,
            callbacks=[tf.keras.callbacks.EarlyStopping(
                monitor="loss", patience=lstm_config.patience,
                restore_best_weights=True,
            )],
        )
        scaled_prediction = np.asarray(model.predict(windows.X_pred, verbose=0)).reshape(-1)
    predicted = (
        scaled_prediction * windows.scaler.scale_[windows.target_column]
        + windows.scaler.mean_[windows.target_column]
    )
    tf.keras.backend.clear_session()
    if len(predicted) != len(windows.dates) or not np.isfinite(predicted).all():
        raise ValueError("LSTM produced incomplete or nonfinite forecasts.")
    return predicted, len(history.history["loss"])
