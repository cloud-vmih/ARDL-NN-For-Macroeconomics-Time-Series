"""Mean-seed validation selection and refit for the XGBoost baseline."""

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
from .models.xgboost import XGBoostConfig, fit_predict_xgboost


XGBOOST_PIPELINE = "xgboost"
XGBOOST_MODEL = "XGBoost_SharedMeanSeed"


def _forecast_frame(
    frame: pd.DataFrame,
    *,
    target: str,
    start: int,
    end: int,
    predicted: np.ndarray,
    split: str,
    seed: int,
) -> pd.DataFrame:
    actual = frame[target].iloc[start:end].to_numpy(float)
    if len(predicted) != len(actual) or not np.isfinite(predicted).all():
        raise ValueError(f"XGBoost seed {seed} has incomplete {split} predictions.")
    return pd.DataFrame({
        "date": frame.index[start:end],
        "split": split,
        "actual": actual,
        "predicted": predicted,
        "seed": int(seed),
        "pipeline": XGBOOST_PIPELINE,
        "scale": "level",
    })


def _metric_frame(forecasts: pd.DataFrame, *, target: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (split, seed), group in forecasts.groupby(["split", "seed"], sort=False):
        rows.append({
            "model": XGBOOST_MODEL,
            "pipeline": XGBOOST_PIPELINE,
            "target": target,
            "split": split,
            "scale": "level",
            "seed": int(seed),
            **evaluate_forecast(group["actual"], group["predicted"]),
        })
    return pd.DataFrame(rows)


def run_xgboost_shared_mean_seed(
    csv_path: str | Path,
    config: ExperimentConfig,
    xgboost_config: XGBoostConfig,
    *,
    seeds: tuple[int, ...],
    output_dir: Path,
) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    """Select one XGBoost signature by mean validation RMSE, then refit each seed."""
    seeds = tuple(dict.fromkeys(int(seed) for seed in seeds))
    if not seeds:
        raise ValueError("XGBoost requires at least one reporting seed.")
    raw_config = replace(config.data, log_transform=False)
    frame = GenericDataLoader(raw_config).load_csv(csv_path)
    train_end = int(len(frame) * config.train_ratio)
    validation_end = int(len(frame) * (config.train_ratio + config.val_ratio))
    if train_end - max(xgboost_config.lookback_grid) < 4 or not (
        train_end < validation_end < len(frame)
    ):
        raise ValueError("XGBoost needs chronological train, validation and test rows.")

    candidates = tuple(product(
        xgboost_config.lookback_grid,
        xgboost_config.max_depth_grid,
        xgboost_config.learning_rate_grid,
    ))
    search_rows: list[dict[str, Any]] = []
    validation_predictions: dict[tuple[int, int, float, int], np.ndarray] = {}
    for lookback, max_depth, learning_rate in candidates:
        for seed in seeds:
            predicted = fit_predict_xgboost(
                frame,
                target=config.data.target,
                fit_end=train_end,
                prediction_start=train_end,
                prediction_end=validation_end,
                lookback=lookback,
                max_depth=max_depth,
                learning_rate=learning_rate,
                seed=seed,
                xgboost_config=xgboost_config,
            )
            key = (lookback, max_depth, learning_rate, seed)
            validation_predictions[key] = predicted
            validation = _forecast_frame(
                frame,
                target=config.data.target,
                start=train_end,
                end=validation_end,
                predicted=predicted,
                split="validation",
                seed=seed,
            )
            score = evaluate_forecast(validation["actual"], validation["predicted"])
            search_rows.append({
                "lookback": lookback,
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "n_estimators": xgboost_config.n_estimators,
                "subsample": xgboost_config.subsample,
                "colsample_bytree": xgboost_config.colsample_bytree,
                "seed": seed,
                "Val Level RMSE": score["RMSE"],
                "Val Level MAE": score["MAE"],
                "Val Level MAPE": score["MAPE"],
            })

    search = pd.DataFrame(search_rows)
    signature = ["lookback", "max_depth", "learning_rate"]
    ranking = search.groupby(signature, as_index=False).agg(
        mean_val_level_RMSE=("Val Level RMSE", "mean"),
        std_val_level_RMSE=("Val Level RMSE", "std"),
        mean_val_level_MAE=("Val Level MAE", "mean"),
        mean_val_level_MAPE=("Val Level MAPE", "mean"),
        seed_count=("seed", "nunique"),
    )
    ranking = ranking.loc[ranking["seed_count"].eq(len(seeds))].copy()
    if ranking.empty:
        raise ValueError("No XGBoost signature covers every reporting seed.")
    ranking["std_val_level_RMSE"] = ranking["std_val_level_RMSE"].fillna(0.0)
    ranking = ranking.sort_values([
        "mean_val_level_RMSE", "mean_val_level_MAE", "mean_val_level_MAPE",
        "lookback", "max_depth", "learning_rate",
    ]).reset_index(drop=True)
    ranking.insert(0, "mean_seed_rank", range(1, len(ranking) + 1))
    ranking["selected"] = ranking["mean_seed_rank"].eq(1)
    ranking["selection_protocol"] = "mean_level_validation_across_reporting_seeds"
    winner = ranking.iloc[0]
    selected = (
        int(winner["lookback"]),
        int(winner["max_depth"]),
        float(winner["learning_rate"]),
    )
    search["selected_shared_mean_seed"] = (
        search["lookback"].eq(selected[0])
        & search["max_depth"].eq(selected[1])
        & search["learning_rate"].eq(selected[2])
    )
    write_csv(search, output_dir / "xgboost_shared_validation_search.csv")
    write_csv(ranking, output_dir / "xgboost_shared_mean_seed_selection.csv")

    results: list[dict[str, Any]] = []
    forecast_frames: list[pd.DataFrame] = []
    for seed in seeds:
        key = (*selected, seed)
        validation = _forecast_frame(
            frame,
            target=config.data.target,
            start=train_end,
            end=validation_end,
            predicted=validation_predictions[key],
            split="validation",
            seed=seed,
        )
        test_prediction = fit_predict_xgboost(
            frame,
            target=config.data.target,
            fit_end=validation_end,
            prediction_start=validation_end,
            prediction_end=len(frame),
            lookback=selected[0],
            max_depth=selected[1],
            learning_rate=selected[2],
            seed=seed,
            xgboost_config=xgboost_config,
        )
        test = _forecast_frame(
            frame,
            target=config.data.target,
            start=validation_end,
            end=len(frame),
            predicted=test_prediction,
            split="test",
            seed=seed,
        )
        seed_forecasts = pd.concat([validation, test], ignore_index=True)
        seed_metrics = _metric_frame(seed_forecasts, target=config.data.target)
        write_csv(seed_forecasts, output_dir / f"seed_{seed}" / "forecasts.csv")
        write_csv(seed_metrics, output_dir / f"seed_{seed}" / "metrics.csv")
        forecast_frames.append(seed_forecasts)
        results.append({
            "seed": seed,
            "lookback": selected[0],
            "max_depth": selected[1],
            "learning_rate": selected[2],
            "n_estimators": xgboost_config.n_estimators,
        })
    forecasts = pd.concat(forecast_frames, ignore_index=True)
    metrics = _metric_frame(forecasts, target=config.data.target)
    return results, metrics, forecasts, ranking, len(search) + len(seeds)
