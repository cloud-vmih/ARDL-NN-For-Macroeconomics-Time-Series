from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


@dataclass
class SklearnFFNNRegressor:
    """Bọc MLPRegressor với chuẩn hóa X và y."""

    hidden_layer_sizes: tuple[int, ...] = (8,)
    alpha: float = 1e-3
    seed: int = 7 
    activation: str = "relu"
    learning_rate_init: float = 0.01
    max_iter: int = 500

    model_: MLPRegressor | None = None
    x_scaler_: StandardScaler | None = None
    y_scaler_: StandardScaler | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SklearnFFNNRegressor":
        """Fit scaler và FFNN trên dữ liệu huấn luyện."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        self.x_scaler_ = StandardScaler().fit(X)
        self.y_scaler_ = StandardScaler().fit(y)
        scaled_X = self.x_scaler_.transform(X)
        scaled_y = self.y_scaler_.transform(y).ravel()
        self.model_ = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            solver="adam",
            alpha=self.alpha,
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            early_stopping=True,
            n_iter_no_change=30,
            random_state=self.seed,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.model_.fit(scaled_X, scaled_y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Dự báo và đưa kết quả về thang đo gốc của y."""
        if self.model_ is None or self.x_scaler_ is None or self.y_scaler_ is None:
            raise RuntimeError("Call fit before predict.")
        scaled = self.x_scaler_.transform(np.asarray(X, dtype=float))
        pred = self.model_.predict(scaled).reshape(-1, 1)
        return self.y_scaler_.inverse_transform(pred).ravel()
