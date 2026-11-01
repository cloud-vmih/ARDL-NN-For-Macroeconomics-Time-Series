"""Dataset-profile validation for audited macro studies."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import file_sha256
from .config import ExperimentConfig, TensorFlowRuntimeConfig
from .macro_benchmark import build_macro_experiment_config
from .shared_mean_benchmark_experiments import NAIVE_PIPELINE, SHARED_MEAN_BENCHMARK_PIPELINES
from .sarimax_shared_mean import SARIMAX_PIPELINE
from .ridge_arx_shared_mean import RIDGE_ARX_PIPELINE

US_UNEMPLOYMENT_FEATURES = (
    "manufacturing_hours",
    "nonfarm_payrolls",
    "cpi_all_items",
    "industrial_prod",
)

def build_us_unemployment_profile(
    csv_path: Path,
    output_dir: Path,
    *,
    runtime: TensorFlowRuntimeConfig,
    scaler: str,
    seeds: tuple[int, ...],
) -> tuple[ExperimentConfig, dict[str, object]]:
    """Validate the fixed, audited panel before starting a benchmark run."""
    if csv_path.name == "df_final.csv":
        raise ValueError("df_final.csv is excluded from the unemployment study.")
    audit_path = csv_path.with_suffix(".audit.json")
    if not audit_path.is_file():
        raise ValueError(f"Source audit is missing: {audit_path}")
    audit = json.loads(audit_path.read_text())
    panel_hash = file_sha256(csv_path)
    if audit.get("panel_sha256") != panel_hash:
        raise ValueError("Panel hash does not match source audit.")
    expected_columns = ("date", "unemployment_rate", *US_UNEMPLOYMENT_FEATURES)
    frame = pd.read_csv(csv_path)
    if tuple(frame.columns) != expected_columns:
        raise ValueError(f"Unemployment panel columns must be {expected_columns}.")
    dates = pd.to_datetime(frame["date"], errors="raise")
    expected_dates = pd.date_range("2000-01-01", "2025-09-01", freq="MS")
    if not dates.equals(pd.Series(expected_dates, name="date")) or len(frame) != 309:
        raise ValueError("Unemployment panel must cover every month 2000-01 to 2025-09.")
    values = frame.loc[:, expected_columns[1:]].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(values.to_numpy(float)).all():
        raise ValueError("Unemployment panel contains missing or nonfinite values.")
    required_series = {"UNRATE", "AWHMAN", "PAYEMS", "CPIAUCSL", "INDPRO"}
    series_ids = {item.get("series_id") for item in audit.get("series", ())}
    snapshot_ids = {item.get("series_id") for item in audit.get("snapshots", ())}
    if (series_ids != required_series or snapshot_ids != required_series
            or audit.get("row_count") != 309
            or audit.get("start") != "2000-01-01"
            or audit.get("end") != "2025-09-01"):
        raise ValueError("Source audit does not describe the registered study panel.")
    config = build_macro_experiment_config(
        US_UNEMPLOYMENT_FEATURES, output_dir, tensorflow=runtime,
        scaler=scaler, seeds=seeds,
    )
    metadata: dict[str, object] = {
        "dataset_profile": "us-unemployment",
        "panel_sha256": panel_hash,
        "source_audit_sha256": file_sha256(audit_path),
        "source_audit_path": str(audit_path.resolve()),
        "forecast_origin_rule": "last calendar day of t, 23:59 America/New_York; inputs through t-1",
        "benchmark_model_set": (*SHARED_MEAN_BENCHMARK_PIPELINES, NAIVE_PIPELINE),
        "paper_seeds": seeds,
    }
    return config, metadata

def _validate_us_unemployment_result(
    results: dict,
    metadata: dict[str, object],
    csv_path: Path,
) -> None:
    """Reject an incomplete study before its manifest can say success."""
    expected = set((*SHARED_MEAN_BENCHMARK_PIPELINES, NAIVE_PIPELINE))
    for key in ("comparison_metrics", "comparison_forecasts", "settings", "date_coverage"):
        frame = results.get(key)
        if not isinstance(frame, pd.DataFrame) or "pipeline" not in frame:
            raise ValueError(f"Unemployment benchmark is missing {key}.")
        actual = set(frame["pipeline"])
        if actual != expected:
            raise ValueError(f"Unemployment benchmark {key} model set mismatch: {sorted(actual)}.")
    coverage = results["date_coverage"]
    for split, count in (("validation", 46), ("test", 47)):
        part = coverage.loc[coverage["split"].eq(split)]
        if len(part) != len(expected) or not part["n_dates"].eq(count).all() or not part["same_dates_for_all_models"].all():
            raise ValueError(f"Unemployment benchmark has incomplete {split} dates.")
        for key in ("seed_metrics", "comparison_forecasts"):
            table = results.get(key)
            if not isinstance(table, pd.DataFrame) or not {"pipeline", "split", "seed"}.issubset(table.columns):
                raise ValueError(f"Unemployment benchmark is missing {key} seed coverage.")
            for model in expected:
                member = table.loc[table["pipeline"].eq(model) & table["split"].eq(split)]
                if model in {
                    NAIVE_PIPELINE,
                    SARIMAX_PIPELINE,
                    RIDGE_ARX_PIPELINE,
                }:
                    expected_rows = 1 if key == "seed_metrics" else count
                    if len(member) != expected_rows or not member["seed"].isna().all():
                        raise ValueError(
                            f"{model} deterministic seed/date coverage is invalid for {split}."
                        )
                    continue
                actual_seeds = set(pd.to_numeric(member["seed"], errors="coerce").dropna().astype(int))
                if actual_seeds != set(metadata["paper_seeds"]):
                    raise ValueError(f"{model} {split} reporting seed coverage is incomplete.")
                if key == "seed_metrics" and len(member) != len(metadata["paper_seeds"]):
                    raise ValueError(f"{model} {split} has duplicate seed metrics.")
                if key == "comparison_forecasts" and not member.groupby("seed").size().eq(count).all():
                    raise ValueError(f"{model} {split} has incomplete per-seed dates.")
    if file_sha256(csv_path) != metadata.get("panel_sha256"):
        raise ValueError("Panel hash changed during benchmark run.")
    if file_sha256(Path(str(metadata["source_audit_path"]))) != metadata.get("source_audit_sha256"):
        raise ValueError("Source audit hash changed during benchmark run.")

def _is_macro_dataset(path: Path) -> bool:
    """Detect the Kaggle macro schema without loading the full dataset."""
    try:
        with path.open(newline="") as handle:
            columns = set(next(csv.reader(handle)))
    except (OSError, StopIteration, csv.Error):
        return False
    return {"date", "unemployment_rate"}.issubset(columns)
