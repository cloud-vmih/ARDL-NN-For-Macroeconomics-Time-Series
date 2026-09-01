from __future__ import annotations

from typing import Any, Sequence

from ..config import TensorFlowRuntimeConfig


_CONFIGURED_GPU_INDEX: int | None = None


def _tensorflow() -> Any:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "TensorFlow/Keras is required for FFNN training. Install the package "
            "with its neural dependencies."
        ) from exc
    return tf


def _keras_version(tf: Any) -> str:
    """Return a Keras version across TensorFlow/Keras packaging variants.

    TensorFlow 2.15 exposes ``tf.keras`` as a lazy module without a
    ``__version__`` attribute, while the standalone ``keras`` package does.
    Runtime metadata must not make model startup fail just because that
    informational attribute is absent.
    """
    version = getattr(tf.keras, "__version__", None)
    if version is not None:
        return str(version)
    try:
        import keras

        version = getattr(keras, "__version__", None)
    except ImportError:
        version = None
    return str(version or "unknown")


def select_tensorflow_device(
    config: TensorFlowRuntimeConfig,
    *,
    physical_gpus: Sequence[object],
) -> str:
    """Resolve an explicit TensorFlow device without silently violating policy."""
    requested = config.device.lower()
    if requested == "cpu":
        return "/CPU:0"
    if requested == "gpu":
        if config.gpu_index >= len(physical_gpus):
            raise RuntimeError(
                f"GPU index {config.gpu_index} was requested, but TensorFlow found "
                f"{len(physical_gpus)} physical GPU(s)."
            )
        return f"/GPU:{config.gpu_index}"
    if physical_gpus:
        index = min(config.gpu_index, len(physical_gpus) - 1)
        return f"/GPU:{index}"
    return "/CPU:0"


def bootstrap_tensorflow(
    config: TensorFlowRuntimeConfig,
) -> tuple[Any, str, dict[str, object]]:
    """Initialize TensorFlow once and return selected device plus audit data."""
    global _CONFIGURED_GPU_INDEX

    tf = _tensorflow()
    physical_gpus = tuple(tf.config.list_physical_devices("GPU"))
    device = select_tensorflow_device(config, physical_gpus=physical_gpus)
    if device.startswith("/GPU"):
        selected_index = int(device.rsplit(":", maxsplit=1)[1])
        selected_gpu = physical_gpus[selected_index]
        if _CONFIGURED_GPU_INDEX is None:
            try:
                if config.memory_growth:
                    tf.config.experimental.set_memory_growth(selected_gpu, True)
            except RuntimeError as exc:
                raise RuntimeError(
                    "TensorFlow GPU runtime was initialized before memory-growth "
                    "policy could be applied. Bootstrap it before building models."
                ) from exc
            _CONFIGURED_GPU_INDEX = selected_index
        elif _CONFIGURED_GPU_INDEX != selected_index:
            raise RuntimeError(
                "TensorFlow is already configured for GPU index "
                f"{_CONFIGURED_GPU_INDEX}; cannot switch to {selected_index}."
            )
    if config.deterministic_ops:
        tf.config.experimental.enable_op_determinism()

    selected_name: str | None = None
    if device.startswith("/GPU"):
        selected_index = int(device.rsplit(":", maxsplit=1)[1])
        details = tf.config.experimental.get_device_details(physical_gpus[selected_index])
        selected_name = str(details.get("device_name") or physical_gpus[selected_index].name)
    build_info = tf.sysconfig.get_build_info()
    metadata: dict[str, object] = {
        "tensorflow_version": str(tf.__version__),
        "keras_version": _keras_version(tf),
        "tensorflow_built_with_cuda": bool(tf.test.is_built_with_cuda()),
        "cuda_version": build_info.get("cuda_version"),
        "cudnn_version": build_info.get("cudnn_version"),
        "requested_device": config.device.lower(),
        "selected_device": device,
        "physical_gpu_count": len(physical_gpus),
        "selected_gpu_name": selected_name,
        "deterministic_ops": bool(config.deterministic_ops),
        "memory_growth": bool(config.memory_growth),
    }
    return tf, device, metadata
