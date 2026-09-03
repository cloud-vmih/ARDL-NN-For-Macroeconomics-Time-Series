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


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_resume_identity(
    *,
    dataset: str | Path,
    config: Any,
    selected_features: tuple[str, ...],
    tuning_seed: int,
    schema_version: int,
) -> dict[str, Any]:
    """Build the immutable identity used to accept or reject a resumed run."""
    return {
        "schema_version": int(schema_version),
        "dataset": str(Path(dataset).resolve()),
        "dataset_sha256": file_sha256(dataset),
        "config_hash": config_hash(config),
        "selected_features": list(selected_features),
        "tuning_seed": int(tuning_seed),
    }


def validate_resume_manifest(
    run_dir: str | Path,
    expected_identity: dict[str, Any],
) -> dict[str, Any]:
    """Return an existing manifest only when its resume identity is exact."""
    path = Path(run_dir) / "manifest.json"
    if not path.is_file():
        raise ValueError(f"Cannot resume without manifest: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read resume manifest: {path}") from exc
    actual = manifest.get("metadata", {}).get("resume_identity")
    if actual != _json_value(expected_identity):
        raise ValueError("Run resume identity mismatch; use a new --run-id.")
    return manifest


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
