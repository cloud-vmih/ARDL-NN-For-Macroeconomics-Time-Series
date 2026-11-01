"""Shared-mean validation selection and refit for MLP-ARX."""

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
from .models.mlp_arx import MLPARXConfig, fit_predict_mlp_arx


MLP_ARX_PIPELINE = "mlp_arx"
MLP_ARX_MODEL = "MLP_ARX_SharedMeanSeed"


def _frame(frame, target, start, end, predicted, split, seed):
    actual = frame[target].iloc[start:end].to_numpy(float)
    if len(actual) != len(predicted) or not np.isfinite(predicted).all():
        raise ValueError(f"MLP-ARX seed {seed} has incomplete {split} forecasts.")
    return pd.DataFrame({"date": frame.index[start:end], "split": split,
        "actual": actual, "predicted": predicted, "seed": int(seed),
        "pipeline": MLP_ARX_PIPELINE, "scale": "level"})


def _metrics(forecasts, target):
    rows = []
    for (split, seed), group in forecasts.groupby(["split", "seed"], sort=False):
        rows.append({"model": MLP_ARX_MODEL, "pipeline": MLP_ARX_PIPELINE,
            "target": target, "split": split, "scale": "level", "seed": int(seed),
            **evaluate_forecast(group["actual"], group["predicted"])})
    return pd.DataFrame(rows)


def run_mlp_arx_shared_mean_seed(csv_path: str | Path, config: ExperimentConfig,
                                 mlp_config: MLPARXConfig, *, seeds: tuple[int, ...],
                                 output_dir: Path):
    seeds = tuple(dict.fromkeys(int(s) for s in seeds))
    if not seeds:
        raise ValueError("MLP-ARX requires reporting seeds.")
    frame = GenericDataLoader(replace(config.data, log_transform=False)).load_csv(csv_path)
    train_end = int(len(frame) * config.train_ratio)
    val_end = int(len(frame) * (config.train_ratio + config.val_ratio))
    candidates = tuple(product(mlp_config.lookback_grid, mlp_config.hidden_layer_grid,
                               mlp_config.alpha_grid))
    rows, val_predictions = [], {}
    for lookback, hidden_layers, alpha in candidates:
        for seed in seeds:
            pred = fit_predict_mlp_arx(frame, target=config.data.target, fit_end=train_end,
                prediction_start=train_end, prediction_end=val_end, lookback=lookback,
                hidden_layers=hidden_layers, alpha=alpha, seed=seed, config=mlp_config)
            key = (lookback, tuple(hidden_layers), alpha, seed)
            val_predictions[key] = pred
            val = _frame(frame, config.data.target, train_end, val_end, pred, "validation", seed)
            score = evaluate_forecast(val["actual"], val["predicted"])
            rows.append({"lookback": lookback, "hidden_layers": tuple(hidden_layers),
                "alpha": alpha, "seed": seed, "Val Level RMSE": score["RMSE"],
                "Val Level MAE": score["MAE"], "Val Level MAPE": score["MAPE"]})
    search = pd.DataFrame(rows)
    ranking = search.groupby(["lookback", "hidden_layers", "alpha"], as_index=False).agg(
        mean_val_level_RMSE=("Val Level RMSE", "mean"),
        std_val_level_RMSE=("Val Level RMSE", "std"),
        mean_val_level_MAE=("Val Level MAE", "mean"),
        mean_val_level_MAPE=("Val Level MAPE", "mean"), seed_count=("seed", "nunique"))
    ranking = ranking[ranking.seed_count.eq(len(seeds))].copy()
    if ranking.empty:
        raise ValueError("No MLP-ARX signature covers every reporting seed.")
    ranking["std_val_level_RMSE"] = ranking["std_val_level_RMSE"].fillna(0.0)
    ranking = ranking.sort_values(["mean_val_level_RMSE", "mean_val_level_MAE",
        "mean_val_level_MAPE", "lookback", "hidden_layers", "alpha"]).reset_index(drop=True)
    ranking.insert(0, "mean_seed_rank", range(1, len(ranking) + 1))
    ranking["selected"] = ranking.mean_seed_rank.eq(1)
    ranking["selection_protocol"] = "mean_level_validation_across_reporting_seeds"
    winner = ranking.iloc[0]
    selected = (int(winner.lookback), tuple(winner.hidden_layers), float(winner.alpha))
    search["selected_shared_mean_seed"] = (
        search.lookback.eq(selected[0])
        & search.hidden_layers.map(lambda value: tuple(value) == selected[1])
        & search.alpha.eq(selected[2]))
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(search, output_dir / "mlp_arx_shared_validation_search.csv")
    write_csv(ranking, output_dir / "mlp_arx_shared_mean_seed_selection.csv")
    forecasts, results = [], []
    for seed in seeds:
        key = (*selected, seed)
        val = _frame(frame, config.data.target, train_end, val_end, val_predictions[key], "validation", seed)
        test_pred = fit_predict_mlp_arx(frame, target=config.data.target, fit_end=val_end,
            prediction_start=val_end, prediction_end=len(frame), lookback=selected[0],
            hidden_layers=selected[1], alpha=selected[2], seed=seed, config=mlp_config)
        test = _frame(frame, config.data.target, val_end, len(frame), test_pred, "test", seed)
        sf = pd.concat([val, test], ignore_index=True)
        write_csv(sf, output_dir / f"seed_{seed}" / "forecasts.csv")
        write_csv(_metrics(sf, config.data.target), output_dir / f"seed_{seed}" / "metrics.csv")
        forecasts.append(sf)
        results.append({"seed": seed, "lookback": selected[0],
            "hidden_layers": selected[1], "alpha": selected[2]})
    forecasts = pd.concat(forecasts, ignore_index=True)
    metrics = _metrics(forecasts, config.data.target)
    return results, metrics, forecasts, ranking, len(search) + len(seeds)
