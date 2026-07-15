from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


SCALE_ALIASES = {
    "raw_transformed": "transformed",
    "reconstructed_transformed": "transformed",
    "level": "level",
}


def normalize_forecasts(
    forecasts: pd.DataFrame,
    *,
    pipeline: str,
    predicted_col: str,
) -> pd.DataFrame:
    actual_col = "actual_level" if "actual_level" in forecasts.columns else "actual_raw_transformed"
    display_predicted = "predicted_level" if "predicted_level" in forecasts.columns else predicted_col
    columns = ["date", "split", actual_col, display_predicted]
    if "seed" in forecasts.columns:
        columns.append("seed")
    out = forecasts[columns].copy()
    out = out.rename(columns={actual_col: "actual", display_predicted: "predicted"})
    out["pipeline"] = pipeline
    out["scale"] = "level" if actual_col == "actual_level" else "transformed"
    return out


def compare_two_metric_tables(
    first_metrics: pd.DataFrame,
    second_metrics: pd.DataFrame,
    *,
    first_key: str,
    second_key: str,
    key_column: str,
) -> pd.DataFrame:
    combined = pd.concat([first_metrics, second_metrics], ignore_index=True).copy()
    combined["comparison_scale"] = combined["scale"].astype(str).map(SCALE_ALIASES).fillna(combined["scale"])
    metadata = {
        "dataset", "model", "uses_vmd", "vmd_setting", "vmd_modes", "activation",
        "split", "scale", "comparison_scale", key_column, "forecast_protocol",
    }
    metric_columns = [
        col for col in combined.columns
        if col not in metadata and pd.api.types.is_numeric_dtype(combined[col])
    ]
    rows: list[dict[str, Any]] = []
    for (split, scale), group in combined.groupby(["split", "comparison_scale"], sort=False):
        indexed = group.set_index(key_column)
        row: dict[str, Any] = {"split": split, "scale": scale}
        for metric in metric_columns:
            first = float(indexed.loc[first_key, metric]) if first_key in indexed.index else np.nan
            second = float(indexed.loc[second_key, metric]) if second_key in indexed.index else np.nan
            row[f"{first_key}_{metric}"] = first
            row[f"{second_key}_{metric}"] = second
            row[f"delta_{metric}_{second_key}_minus_{first_key}"] = second - first
            if metric in {"RMSE", "MAE", "MAPE"} and np.isfinite(first) and abs(first) > 1e-12:
                row[f"{second_key}_improvement_{metric}_pct"] = (first - second) / first * 100.0
        rows.append(row)
    return pd.DataFrame(rows)
