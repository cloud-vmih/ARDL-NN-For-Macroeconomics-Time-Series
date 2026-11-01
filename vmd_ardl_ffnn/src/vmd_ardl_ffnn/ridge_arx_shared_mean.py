"""Validation-selected deterministic Ridge-ARX baseline."""

from __future__ import annotations

from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import write_csv
from .config import ExperimentConfig
from .data import GenericDataLoader
from .metrics import evaluate_forecast
from .models.ridge_arx import RidgeARXConfig, fit_predict_ridge_arx


RIDGE_ARX_PIPELINE = "ridge_arx"
RIDGE_ARX_MODEL = "Ridge_ARX_ValidationSelected"


def _forecast_frame(frame: pd.DataFrame, *, target: str, start: int, end: int,
                    predicted: np.ndarray, split: str) -> pd.DataFrame:
    actual = frame[target].iloc[start:end].to_numpy(float)
    if len(predicted) != len(actual) or not np.isfinite(predicted).all():
        raise ValueError(f"Ridge-ARX has incomplete {split} predictions.")
    return pd.DataFrame({
        "date": frame.index[start:end], "split": split, "actual": actual,
        "predicted": predicted, "seed": pd.NA,
        "pipeline": RIDGE_ARX_PIPELINE, "scale": "level",
    })


def _metric_frame(forecasts: pd.DataFrame, *, target: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split, group in forecasts.groupby("split", sort=False):
        rows.append({
            "model": RIDGE_ARX_MODEL, "pipeline": RIDGE_ARX_PIPELINE,
            "target": target, "split": split, "scale": "level", "seed": pd.NA,
            **evaluate_forecast(group["actual"], group["predicted"]),
        })
    return pd.DataFrame(rows)


def run_ridge_arx_validation_selected(
    csv_path: str | Path,
    config: ExperimentConfig,
    ridge_config: RidgeARXConfig,
    *,
    output_dir: Path,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    """Select Ridge-ARX by validation RMSE, refit train+validation for test."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_config = replace(config.data, log_transform=False)
    frame = GenericDataLoader(raw_config).load_csv(csv_path)
    train_end = int(len(frame) * config.train_ratio)
    validation_end = int(len(frame) * (config.train_ratio + config.val_ratio))
    if train_end - max(ridge_config.lookback_grid) < 4 or not (
        train_end < validation_end < len(frame)
    ):
        raise ValueError("Ridge-ARX needs enough chronological train, validation and test rows.")

    rows: list[dict[str, Any]] = []
    validation_predictions: dict[tuple[int, float], np.ndarray] = {}
    for lookback, alpha in product(ridge_config.lookback_grid, ridge_config.alpha_grid):
        predicted = fit_predict_ridge_arx(
            frame, target=config.data.target, fit_end=train_end,
            prediction_start=train_end, prediction_end=validation_end,
            lookback=lookback, alpha=alpha, config=ridge_config,
        )
        validation_predictions[(lookback, alpha)] = predicted
        validation = _forecast_frame(
            frame, target=config.data.target, start=train_end, end=validation_end,
            predicted=predicted, split="validation",
        )
        score = evaluate_forecast(validation["actual"], validation["predicted"])
        rows.append({
            "lookback": lookback, "alpha": alpha,
            "Val Level RMSE": score["RMSE"], "Val Level MAE": score["MAE"],
            "Val Level MAPE": score["MAPE"], "Val Level R2": score["R2"],
            "Val Level Directional_Accuracy": score["Directional_Accuracy"],
        })
    search = pd.DataFrame(rows).sort_values(
        ["Val Level RMSE", "Val Level MAE", "Val Level MAPE", "lookback", "alpha"]
    ).reset_index(drop=True)
    search.insert(0, "validation_rank", range(1, len(search) + 1))
    search["selected"] = search["validation_rank"].eq(1)
    search["selection_protocol"] = "level_validation_rmse"
    winner = search.iloc[0]
    selected = (int(winner["lookback"]), float(winner["alpha"]))
    write_csv(search, output_dir / "ridge_arx_validation_search.csv")
    write_csv(search, output_dir / "ridge_arx_selection.csv")

    validation = _forecast_frame(
        frame, target=config.data.target, start=train_end, end=validation_end,
        predicted=validation_predictions[selected], split="validation",
    )
    test_prediction = fit_predict_ridge_arx(
        frame, target=config.data.target, fit_end=validation_end,
        prediction_start=validation_end, prediction_end=len(frame),
        lookback=selected[0], alpha=selected[1], config=ridge_config,
    )
    test = _forecast_frame(
        frame, target=config.data.target, start=validation_end, end=len(frame),
        predicted=test_prediction, split="test",
    )
    forecasts = pd.concat([validation, test], ignore_index=True)
    metrics = _metric_frame(forecasts, target=config.data.target)
    write_csv(forecasts, output_dir / "forecasts.csv")
    write_csv(metrics, output_dir / "metrics.csv")
    return (
        {"lookback": selected[0], "alpha": selected[1], "fit_intercept": ridge_config.fit_intercept},
        metrics, forecasts, search, len(search) + 1,
    )
