from __future__ import annotations

import numpy as np
import pytest
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from vmd_ardl_ffnn.config import FFNNConfig, TensorFlowRuntimeConfig
from vmd_ardl_ffnn.models.ffnn import TensorFlowFFNNRegressor
from vmd_ardl_ffnn.models import tensorflow_runtime
from vmd_ardl_ffnn.models.tensorflow_runtime import (
    bootstrap_tensorflow,
    select_tensorflow_device,
)


def test_tensorflow_runtime_requires_requested_gpu() -> None:
    config = TensorFlowRuntimeConfig(device="gpu", gpu_index=0)

    with pytest.raises(RuntimeError, match="GPU index 0 was requested"):
        select_tensorflow_device(config, physical_gpus=())


def test_tensorflow_runtime_selects_explicit_cpu() -> None:
    config = TensorFlowRuntimeConfig(device="cpu")

    assert select_tensorflow_device(config, physical_gpus=()) == "/CPU:0"


@pytest.mark.parametrize(
    ("scaler_name", "scaler_type"),
    [("standard", StandardScaler), ("minmax", MinMaxScaler)],
)
def test_tensorflow_ffnn_uses_configured_train_only_scalers(
    scaler_name: str,
    scaler_type: type[StandardScaler] | type[MinMaxScaler],
) -> None:
    X = np.arange(48, dtype=np.float32).reshape(24, 2)
    y = (0.5 * X[:, 0] - 0.25 * X[:, 1]).astype(np.float32)
    model = TensorFlowFFNNRegressor(
        hidden_layer_sizes=(4,),
        alpha=1e-3,
        seed=11,
        activation="tanh",
        learning_rate_init=1e-3,
        max_iter=4,
        scaler=scaler_name,
        batch_size=8,
        validation_fraction=0.25,
        early_stopping_patience=2,
        reduce_lr_patience=1,
        runtime=TensorFlowRuntimeConfig(device="cpu"),
    ).fit(X, y)

    assert isinstance(model.x_scaler_, scaler_type)
    assert isinstance(model.y_scaler_, scaler_type)
    assert model.x_scaler_.n_samples_seen_ == len(X)
    assert model.y_scaler_.n_samples_seen_ == len(y)
    assert model.predict(X[:3]).shape == (3,)
    assert np.isfinite(model.predict(X[:3])).all()
    assert 1 <= model.n_iter_ <= 4
    assert model.parameter_count_ > 0
    assert model.device_ == "/CPU:0"


def test_ffnn_config_rejects_unknown_scaler() -> None:
    with pytest.raises(ValueError, match="scaler"):
        FFNNConfig(scaler="robust")


def test_ffnn_runtime_can_train_eagerly() -> None:
    X = np.arange(48, dtype=np.float32).reshape(24, 2)
    y = (X[:, 0] - X[:, 1]).astype(np.float32)
    model = TensorFlowFFNNRegressor(
        hidden_layer_sizes=(4,), max_iter=2, validation_fraction=0.0,
        runtime=TensorFlowRuntimeConfig(device="cpu", run_eagerly=True),
    ).fit(X, y)

    assert model.model_.run_eagerly is True
    assert np.isfinite(model.predict(X[:2])).all()


def test_legacy_ffnn_alias_warns_without_numpy_fallback() -> None:
    import vmd_ardl_ffnn.models.ffnn as ffnn

    with pytest.warns(DeprecationWarning, match="deprecated"):
        legacy = ffnn.NumpyFFNNRegressor
    assert legacy is TensorFlowFFNNRegressor


def test_gpu_index_remains_the_tensorflow_logical_device(monkeypatch) -> None:
    physical = (object(), object())

    class Experimental:
        def __init__(self):
            self.memory_growth_target = None

        def set_memory_growth(self, target, enabled):
            self.memory_growth_target = (target, enabled)

        def enable_op_determinism(self):
            return None

        def get_device_details(self, target):
            return {"device_name": "GPU one"}

    experimental = Experimental()

    class Config:
        @staticmethod
        def list_physical_devices(kind):
            assert kind == "GPU"
            return physical

    Config.experimental = experimental

    class FakeTensorFlow:
        __version__ = "test"
        config = Config()
        keras = type("Keras", (), {"__version__": "test"})()
        test = type("Test", (), {"is_built_with_cuda": staticmethod(lambda: True)})()
        sysconfig = type(
            "SysConfig", (), {"get_build_info": staticmethod(lambda: {})}
        )()

    monkeypatch.setattr(tensorflow_runtime, "_tensorflow", lambda: FakeTensorFlow())
    monkeypatch.setattr(tensorflow_runtime, "_CONFIGURED_GPU_INDEX", None)
    _, device, metadata = bootstrap_tensorflow(
        TensorFlowRuntimeConfig(device="gpu", gpu_index=1)
    )
    assert device == "/GPU:1"
    assert experimental.memory_growth_target == (physical[1], True)
    assert metadata["selected_gpu_name"] == "GPU one"
