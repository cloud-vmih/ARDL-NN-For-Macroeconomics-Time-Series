"""Mean-seed validation selection and refit for the multivariate GRU baseline."""

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
from .models.gru import GRUConfig, fit_predict_gru


GRU_PIPELINE = "gru"
GRU_MODEL = "GRU_SharedMeanSeed"


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
        raise ValueError(f"GRU seed {seed} has incomplete {split} predictions.")
    return pd.DataFrame({
        "date": frame.index[start:end],
        "split": split,
        "actual": actual,
        "predicted": predicted,
        "seed": int(seed),
        "pipeline": GRU_PIPELINE,
        "scale": "level",
    })


def _metric_frame(forecasts: pd.DataFrame, *, target: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (split, seed), group in forecasts.groupby(["split", "seed"], sort=False):
        rows.append({
            "model": GRU_MODEL,
            "pipeline": GRU_PIPELINE,
            "target": target,
            "split": split,
            "scale": "level",
            "seed": int(seed),
            **evaluate_forecast(group["actual"], group["predicted"]),
        })
    return pd.DataFrame(rows)


def run_gru_shared_mean_seed(
    csv_path: str | Path,
    config: ExperimentConfig,
    gru_config: GRUConfig,
    *,
    seeds: tuple[int, ...],
    output_dir: Path,
) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    """Select one GRU signature by mean validation RMSE, then refit each seed."""
    seeds = tuple(dict.fromkeys(int(seed) for seed in seeds))
    if not seeds:
        raise ValueError("GRU requires at least one reporting seed.")
    raw_config = replace(config.data, log_transform=False)
    frame = GenericDataLoader(raw_config).load_csv(csv_path)
    train_end = int(len(frame) * config.train_ratio)
    validation_end = int(len(frame) * (config.train_ratio + config.val_ratio))
    if train_end - max(gru_config.lookback_grid) < 4 or not (
        train_end < validation_end < len(frame)
    ):
        raise ValueError("GRU needs enough chronological train, validation and test rows.")

    candidates = tuple(product(
        gru_config.lookback_grid,
        gru_config.hidden_units_grid,
        gru_config.learning_rate_grid,
    ))
    search_rows: list[dict[str, Any]] = []
    validation_predictions: dict[tuple[int, int, float, int], np.ndarray] = {}
    validation_epochs: dict[tuple[int, int, float, int], int] = {}
    for lookback, hidden_units, learning_rate in candidates:
        for seed in seeds:
            predicted, epochs = fit_predict_gru(
                frame,
                target=config.data.target,
                fit_end=train_end,
                prediction_start=train_end,
                prediction_end=validation_end,
                lookback=lookback,
                hidden_units=hidden_units,
                learning_rate=learning_rate,
                seed=seed,
                gru_config=gru_config,
                runtime=config.tensorflow,
            )
            key = (lookback, hidden_units, learning_rate, seed)
            validation_predictions[key] = np.asarray(predicted, dtype=float)
            validation_epochs[key] = int(epochs)
            validation = _forecast_frame(
                frame,
                target=config.data.target,
                start=train_end,
                end=validation_end,
                predicted=validation_predictions[key],
                split="validation",
                seed=seed,
            )
            score = evaluate_forecast(validation["actual"], validation["predicted"])
            search_rows.append({
                "lookback": lookback,
                "hidden_units": hidden_units,
                "learning_rate": learning_rate,
                "seed": seed,
                "epochs_trained": int(epochs),
                "Val Level RMSE": score["RMSE"],
                "Val Level MAE": score["MAE"],
                "Val Level MAPE": score["MAPE"],
            })

    search = pd.DataFrame(search_rows)
    signature = ["lookback", "hidden_units", "learning_rate"]
    ranking = search.groupby(signature, as_index=False).agg(
        mean_val_level_RMSE=("Val Level RMSE", "mean"),
        std_val_level_RMSE=("Val Level RMSE", "std"),
        mean_val_level_MAE=("Val Level MAE", "mean"),
        mean_val_level_MAPE=("Val Level MAPE", "mean"),
        seed_count=("seed", "nunique"),
    )
    ranking = ranking.loc[ranking["seed_count"].eq(len(seeds))].copy()
    if ranking.empty:
        raise ValueError("No GRU signature covers every reporting seed.")
    ranking["std_val_level_RMSE"] = ranking["std_val_level_RMSE"].fillna(0.0)
    ranking = ranking.sort_values([
        "mean_val_level_RMSE",
        "mean_val_level_MAE",
        "mean_val_level_MAPE",
        "lookback",
        "hidden_units",
        "learning_rate",
    ]).reset_index(drop=True)
    ranking.insert(0, "mean_seed_rank", range(1, len(ranking) + 1))
    ranking["selected"] = ranking["mean_seed_rank"].eq(1)
    ranking["selection_protocol"] = "mean_level_validation_across_reporting_seeds"
    winner = ranking.iloc[0]
    selected = (
        int(winner["lookback"]),
        int(winner["hidden_units"]),
        float(winner["learning_rate"]),
    )
    search["selected_shared_mean_seed"] = (
        search["lookback"].eq(selected[0])
        & search["hidden_units"].eq(selected[1])
        & search["learning_rate"].eq(selected[2])
    )
    write_csv(search, output_dir / "gru_shared_validation_search.csv")
    write_csv(ranking, output_dir / "gru_shared_mean_seed_selection.csv")

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
        test_prediction, refit_epochs = fit_predict_gru(
            frame,
            target=config.data.target,
            fit_end=validation_end,
            prediction_start=validation_end,
            prediction_end=len(frame),
            lookback=selected[0],
            hidden_units=selected[1],
            learning_rate=selected[2],
            seed=seed,
            gru_config=gru_config,
            runtime=config.tensorflow,
        )
        test = _forecast_frame(
            frame,
            target=config.data.target,
            start=validation_end,
            end=len(frame),
            predicted=np.asarray(test_prediction, dtype=float),
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
            "hidden_units": selected[1],
            "learning_rate": selected[2],
            "validation_epochs": validation_epochs[key],
            "refit_epochs": int(refit_epochs),
        })
    forecasts = pd.concat(forecast_frames, ignore_index=True)
    metrics = _metric_frame(forecasts, target=config.data.target)
    return results, metrics, forecasts, ranking, len(search) + len(seeds)
