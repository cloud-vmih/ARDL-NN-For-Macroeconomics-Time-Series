from __future__ import annotations

from dataclasses import dataclass, field
import warnings

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from ..config import TensorFlowRuntimeConfig
from .tensorflow_runtime import bootstrap_tensorflow


@dataclass
class TensorFlowFFNNRegressor:
    """scikit-learn-shaped TensorFlow/Keras feed-forward regressor."""

    hidden_layer_sizes: tuple[int, ...] = (8,)
    alpha: float = 1e-3
    seed: int = 7
    activation: str = "relu"
    learning_rate_init: float = 0.01
    max_iter: int = 500
    tol: float = 1e-6
    n_iter_no_change: int = 30
    scaler: str = "standard"
    batch_size: int = 16
    validation_fraction: float = 0.20
    early_stopping_patience: int = 15
    reduce_lr_patience: int = 6
    reduce_lr_factor: float = 0.5
    min_learning_rate: float = 1e-5
    dropout_rate: float = 0.0
    runtime: TensorFlowRuntimeConfig = field(default_factory=TensorFlowRuntimeConfig)

    model_: object | None = field(default=None, init=False)
    x_scaler_: StandardScaler | MinMaxScaler | None = field(default=None, init=False)
    y_scaler_: StandardScaler | MinMaxScaler | None = field(default=None, init=False)
    history_: dict[str, list[float]] = field(default_factory=dict, init=False)
    loss_curve_: list[float] = field(default_factory=list, init=False)
    n_iter_: int = field(default=0, init=False)
    parameter_count_: int = field(default=0, init=False)
    device_: str | None = field(default=None, init=False)
    runtime_metadata_: dict[str, object] = field(default_factory=dict, init=False)

    def _new_scaler(self) -> StandardScaler | MinMaxScaler:
        name = self.scaler.replace("-", "_").lower()
        if name == "standard":
            return StandardScaler()
        if name == "minmax":
            return MinMaxScaler(feature_range=(0.0, 1.0))
        raise ValueError("scaler must be either 'standard' or 'minmax'.")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TensorFlowFFNNRegressor":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of rows.")
        if len(X) == 0:
            raise ValueError("X and y must not be empty.")
        if not self.hidden_layer_sizes or any(
            int(size) < 1 for size in self.hidden_layer_sizes
        ):
            raise ValueError("hidden_layer_sizes must contain positive widths.")
        if self.max_iter < 1 or self.batch_size < 1:
            raise ValueError("max_iter and batch_size must be >= 1.")
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in [0, 1).")

        tf, self.device_, self.runtime_metadata_ = bootstrap_tensorflow(self.runtime)
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(int(self.seed))

        self.x_scaler_ = self._new_scaler().fit(X)
        self.y_scaler_ = self._new_scaler().fit(y)
        scaled_X = self.x_scaler_.transform(X).astype(np.float32)
        scaled_y = self.y_scaler_.transform(y).astype(np.float32)

        regularizer = (
            tf.keras.regularizers.L2(float(self.alpha) / 2.0)
            if self.alpha > 0.0
            else None
        )
        with tf.device(self.device_):
            layers: list[object] = [
                tf.keras.layers.Input(shape=(scaled_X.shape[1],))
            ]
            for width in self.hidden_layer_sizes:
                layers.append(
                    tf.keras.layers.Dense(
                        int(width),
                        activation=self.activation,
                        kernel_regularizer=regularizer,
                    )
                )
                if self.dropout_rate > 0.0:
                    layers.append(tf.keras.layers.Dropout(float(self.dropout_rate)))
            layers.append(tf.keras.layers.Dense(1, kernel_regularizer=regularizer))
            self.model_ = tf.keras.Sequential(layers)
            self.model_.compile(
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=float(self.learning_rate_init)
                ),
                loss="mse",
                metrics=["mae"],
                run_eagerly=bool(self.runtime.run_eagerly),
            )

            monitor = "val_loss" if self.validation_fraction > 0.0 else "loss"
            callbacks: list[object] = [
                tf.keras.callbacks.EarlyStopping(
                    monitor=monitor,
                    patience=int(self.early_stopping_patience),
                    min_delta=float(self.tol),
                    restore_best_weights=True,
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor=monitor,
                    factor=float(self.reduce_lr_factor),
                    patience=int(self.reduce_lr_patience),
                    min_lr=float(self.min_learning_rate),
                ),
            ]
            history = self.model_.fit(
                scaled_X,
                scaled_y,
                validation_split=float(self.validation_fraction),
                epochs=int(self.max_iter),
                batch_size=min(int(self.batch_size), len(scaled_X)),
                shuffle=False,
                callbacks=callbacks,
                verbose=0,
            )

        self.history_ = {
            key: [float(value) for value in values]
            for key, values in history.history.items()
        }
        self.loss_curve_ = list(self.history_.get("loss", ()))
        self.n_iter_ = len(self.loss_curve_)
        self.parameter_count_ = int(self.model_.count_params())
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None or self.x_scaler_ is None or self.y_scaler_ is None:
            raise RuntimeError("Call fit before predict.")
        values = np.asarray(X, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError("X must be a 2D array.")
        scaled = self.x_scaler_.transform(values).astype(np.float32)
        prediction = self.model_.predict(
            scaled,
            batch_size=min(int(self.batch_size), max(1, len(scaled))),
            verbose=0,
        )
        return self.y_scaler_.inverse_transform(prediction).ravel()


def __getattr__(name: str):
    if name in {"NumpyFFNNRegressor", "SklearnFFNNRegressor"}:
        warnings.warn(
            f"{name} is deprecated; use TensorFlowFFNNRegressor.",
            DeprecationWarning,
            stacklevel=2,
        )
        return TensorFlowFFNNRegressor
    raise AttributeError(name)


__all__ = ["TensorFlowFFNNRegressor"]
