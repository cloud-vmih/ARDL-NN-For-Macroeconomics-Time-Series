from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
from typing import Any

import pandas as pd


def _dependency_versions() -> dict[str, str | None]:
    dependencies = (
        "numpy", "pandas", "statsmodels", "scikit-learn", "matplotlib",
        "tensorflow", "keras", "sktime",
    )
    resolved: dict[str, str | None] = {}
    for dependency in dependencies:
        try:
            resolved[dependency] = version(dependency)
        except PackageNotFoundError:
            resolved[dependency] = None
    return resolved


def _json_value(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _json_value(item) for key, item in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


def config_hash(config: Any) -> str:
    normalized = _json_value(config)
    if isinstance(normalized, dict):
        normalized = dict(normalized)
        normalized.pop("output_dir", None)
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]


def make_run_id(config: Any, now: datetime | None = None) -> str:
    timestamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{timestamp}_{config_hash(config)}"


def prepare_run_directory(root: str | Path, pipeline: str, config: Any, run_id: str | None = None) -> Path:
    path = Path(root) / pipeline / (run_id or make_run_id(config))
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_manifest(
    output_dir: str | Path,
    *,
    pipeline: str,
    dataset: str | Path,
    config: Any,
    status: str,
    error: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": output_path.name,
        "pipeline": pipeline,
        "dataset": str(Path(dataset).resolve()),
        "status": status,
        "error": error,
        "forecast_protocol": getattr(config, "forecast_protocol", None),
        "config_hash": config_hash(config),
        "config": _json_value(config),
        "metadata": _json_value(metadata or {}),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "dependencies": _dependency_versions(),
        },
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    path = output_path / "manifest.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def write_csv(frame: pd.DataFrame, path: str | Path) -> None:
    """Write one run-scoped artifact atomically enough for local experiments."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(output)
