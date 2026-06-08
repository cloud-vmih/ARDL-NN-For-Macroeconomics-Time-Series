from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class NumpyStandardScaler:
    """Scaler tối giản tương đương StandardScaler, cài bằng NumPy."""

    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> "NumpyStandardScaler":
        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        self.mean_ = values.mean(axis=0)
        scale = values.std(axis=0)
        self.scale_ = np.where(scale == 0.0, 1.0, scale)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Call fit before transform.")
        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        return (values - self.mean_) / self.scale_

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Call fit before inverse_transform.")
        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        return values * self.scale_ + self.mean_


@dataclass
class NumpyFFNNRegressor:
    """FFNN tự cài bằng NumPy với chuẩn hóa X/y và tối ưu Adam."""

    hidden_layer_sizes: tuple[int, ...] = (8,)
    alpha: float = 1e-3
    seed: int = 7
    activation: str = "relu"
    learning_rate_init: float = 0.01
    max_iter: int = 500
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-8
    tol: float = 1e-6
    n_iter_no_change: int = 30

    weights_: list[np.ndarray] = field(default_factory=list)
    biases_: list[np.ndarray] = field(default_factory=list)
    x_scaler_: NumpyStandardScaler | None = None
    y_scaler_: NumpyStandardScaler | None = None
    loss_curve_: list[float] = field(default_factory=list)
    n_iter_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NumpyFFNNRegressor":
        """Fit scaler và FFNN trên dữ liệu huấn luyện."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of rows.")
        if len(X) == 0:
            raise ValueError("X and y must not be empty.")

        self.x_scaler_ = NumpyStandardScaler().fit(X)
        self.y_scaler_ = NumpyStandardScaler().fit(y)
        scaled_X = self.x_scaler_.transform(X)
        scaled_y = self.y_scaler_.transform(y)

        layer_sizes = (scaled_X.shape[1], *self.hidden_layer_sizes, 1)
        self._initialize_parameters(layer_sizes)
        self._fit_adam(scaled_X, scaled_y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Dự báo và đưa kết quả về thang đo gốc của y."""
        if not self.weights_ or self.x_scaler_ is None or self.y_scaler_ is None:
            raise RuntimeError("Call fit before predict.")
        scaled = self.x_scaler_.transform(np.asarray(X, dtype=float))
        pred = self._forward(scaled)[0][-1]
        return self.y_scaler_.inverse_transform(pred).ravel()

    def _initialize_parameters(self, layer_sizes: tuple[int, ...]) -> None:
        rng = np.random.default_rng(self.seed)
        self.weights_ = []
        self.biases_ = []
        for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            if self.activation == "relu":
                limit = np.sqrt(2.0 / max(1, fan_in))
                weights = rng.normal(0.0, limit, size=(fan_in, fan_out))
            else:
                limit = np.sqrt(6.0 / max(1, fan_in + fan_out))
                weights = rng.uniform(-limit, limit, size=(fan_in, fan_out))
            self.weights_.append(weights)
            self.biases_.append(np.zeros((1, fan_out), dtype=float))

    def _fit_adam(self, X: np.ndarray, y: np.ndarray) -> None:
        weight_m = [np.zeros_like(weight) for weight in self.weights_]
        weight_v = [np.zeros_like(weight) for weight in self.weights_]
        bias_m = [np.zeros_like(bias) for bias in self.biases_]
        bias_v = [np.zeros_like(bias) for bias in self.biases_]

        best_loss = np.inf
        best_weights = [w.copy() for w in self.weights_]
        best_biases = [b.copy() for b in self.biases_]
        stale_iters = 0
        self.loss_curve_ = []

        for iteration in range(1, int(self.max_iter) + 1):
            activations, pre_activations = self._forward(X)
            loss = self._loss(activations[-1], y)

            if not np.isfinite(loss):
                break

            grad_weights, grad_biases = self._backward(activations, pre_activations, y)

            # optional gradient clipping
            grad_norm = np.sqrt(
                sum(np.sum(g ** 2) for g in grad_weights)
                + sum(np.sum(g ** 2) for g in grad_biases)
            )
            max_norm = 10.0
            if grad_norm > max_norm:
                scale = max_norm / (grad_norm + self.epsilon)
                grad_weights = [g * scale for g in grad_weights]
                grad_biases = [g * scale for g in grad_biases]

            for idx, (grad_w, grad_b) in enumerate(zip(grad_weights, grad_biases)):
                weight_m[idx] = self.beta_1 * weight_m[idx] + (1.0 - self.beta_1) * grad_w
                weight_v[idx] = self.beta_2 * weight_v[idx] + (1.0 - self.beta_2) * np.square(grad_w)
                bias_m[idx] = self.beta_1 * bias_m[idx] + (1.0 - self.beta_1) * grad_b
                bias_v[idx] = self.beta_2 * bias_v[idx] + (1.0 - self.beta_2) * np.square(grad_b)

                corrected_weight_m = weight_m[idx] / (1.0 - self.beta_1**iteration)
                corrected_weight_v = weight_v[idx] / (1.0 - self.beta_2**iteration)
                corrected_bias_m = bias_m[idx] / (1.0 - self.beta_1**iteration)
                corrected_bias_v = bias_v[idx] / (1.0 - self.beta_2**iteration)

                self.weights_[idx] -= self.learning_rate_init * corrected_weight_m / (
                    np.sqrt(corrected_weight_v) + self.epsilon
                )
                self.biases_[idx] -= self.learning_rate_init * corrected_bias_m / (
                    np.sqrt(corrected_bias_v) + self.epsilon
                )

            self.loss_curve_.append(float(loss))
            self.n_iter_ = iteration

            if best_loss - loss > self.tol:
                best_loss = loss
                best_weights = [w.copy() for w in self.weights_]
                best_biases = [b.copy() for b in self.biases_]
                stale_iters = 0
            else:
                stale_iters += 1
                if stale_iters >= self.n_iter_no_change:
                    break

        self.weights_ = best_weights
        self.biases_ = best_biases

    def _forward(self, X: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        activations = [X]
        pre_activations: list[np.ndarray] = []
        for idx, (weight, bias) in enumerate(zip(self.weights_, self.biases_)):
            z = activations[-1] @ weight + bias
            pre_activations.append(z)
            if idx == len(self.weights_) - 1:
                activations.append(z)
            else:
                activations.append(self._activate(z))
        return activations, pre_activations

    def _backward(
        self,
        activations: list[np.ndarray],
        pre_activations: list[np.ndarray],
        y: np.ndarray,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        n_samples = y.shape[0]
        delta = 2.0 * (activations[-1] - y) / n_samples
        grad_weights: list[np.ndarray] = []
        grad_biases: list[np.ndarray] = []

        for idx in range(len(self.weights_) - 1, -1, -1):
            grad_w = activations[idx].T @ delta + self.alpha * self.weights_[idx]
            grad_b = delta.sum(axis=0, keepdims=True)
            grad_weights.insert(0, grad_w)
            grad_biases.insert(0, grad_b)
            if idx > 0:
                delta = (delta @ self.weights_[idx].T) * self._activation_derivative(pre_activations[idx - 1])
        return grad_weights, grad_biases

    def _loss(self, prediction: np.ndarray, y: np.ndarray) -> float:
        mse = np.mean(np.square(prediction - y))
        penalty = 0.5 * self.alpha * sum(float(np.sum(np.square(weight))) for weight in self.weights_)
        return float(mse + penalty)

    def _activate(self, values: np.ndarray) -> np.ndarray:
        activation = self.activation.lower()
        if activation == "relu":
            return np.maximum(values, 0.0)
        if activation == "tanh":
            return np.tanh(values)
        if activation in {"logistic", "sigmoid"}:
            return 1.0 / (1.0 + np.exp(-np.clip(values, -500.0, 500.0)))
        if activation == "identity":
            return values
        raise ValueError("activation must be one of: relu, tanh, logistic, sigmoid, identity.")

    def _activation_derivative(self, values: np.ndarray) -> np.ndarray:
        activation = self.activation.lower()
        if activation == "relu":
            return (values > 0.0).astype(float)
        if activation == "tanh":
            activated = np.tanh(values)
            return 1.0 - np.square(activated)
        if activation in {"logistic", "sigmoid"}:
            activated = 1.0 / (1.0 + np.exp(-np.clip(values, -500.0, 500.0)))
            return activated * (1.0 - activated)
        if activation == "identity":
            return np.ones_like(values)
        raise ValueError("activation must be one of: relu, tanh, logistic, sigmoid, identity.")


# Giữ tên cũ để các module hiện tại không phải đổi import.
SklearnFFNNRegressor = NumpyFFNNRegressor
