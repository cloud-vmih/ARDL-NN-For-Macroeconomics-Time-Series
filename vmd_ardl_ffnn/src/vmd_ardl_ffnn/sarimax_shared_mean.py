"""Validation selection and rolling refit for the deterministic SARIMAX baseline."""

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
from .models.sarimax import SARIMAXConfig, fit_predict_sarimax


SARIMAX_PIPELINE = "sarimax"
SARIMAX_MODEL = "SARIMAX_ValidationSelected"


def _forecast_frame(
    frame: pd.DataFrame,
    *,
    target: str,
    start: int,
    end: int,
    predicted: np.ndarray,
    split: str,
) -> pd.DataFrame:
    actual = frame[target].iloc[start:end].to_numpy(dtype=float)
    if len(predicted) != len(actual) or not np.isfinite(predicted).all():
        raise ValueError(f"SARIMAX has incomplete {split} predictions.")
    return pd.DataFrame({
        "date": frame.index[start:end],
        "split": split,
        "actual": actual,
        "predicted": predicted,
        "seed": pd.NA,
        "pipeline": SARIMAX_PIPELINE,
        "scale": "level",
    })


def _metric_frame(forecasts: pd.DataFrame, *, target: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split, group in forecasts.groupby("split", sort=False):
        rows.append({
            "model": SARIMAX_MODEL,
            "pipeline": SARIMAX_PIPELINE,
            "target": target,
            "split": split,
            "scale": "level",
            "seed": pd.NA,
            **evaluate_forecast(group["actual"], group["predicted"]),
        })
    return pd.DataFrame(rows)


def run_sarimax_validation_selected(
    csv_path: str | Path,
    config: ExperimentConfig,
    sarimax_config: SARIMAXConfig,
    *,
    output_dir: Path,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    """Select one SARIMAX signature on validation, then refit for test."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_config = replace(config.data, log_transform=False)
    frame = GenericDataLoader(raw_config).load_csv(csv_path)
    train_end = int(len(frame) * config.train_ratio)
    validation_end = int(len(frame) * (config.train_ratio + config.val_ratio))
    if not (train_end < validation_end < len(frame)):
        raise ValueError("SARIMAX needs chronological train, validation and test rows.")

    search_rows: list[dict[str, Any]] = []
    validation_predictions: dict[
        tuple[tuple[int, int, int], tuple[int, int, int, int]], np.ndarray
    ] = {}
    for order, seasonal_order in product(
        sarimax_config.order_grid,
        sarimax_config.seasonal_order_grid,
    ):
        row: dict[str, Any] = {
            "order": order,
            "seasonal_order": seasonal_order,
            "exog_lag": sarimax_config.exog_lag,
            "trend": sarimax_config.trend,
            "status": "failed",
            "error": "",
        }
        try:
            predicted, aic, converged, iterations = fit_predict_sarimax(
                frame,
                target=config.data.target,
                features=config.data.features,
                fit_end=train_end,
                prediction_start=train_end,
                prediction_end=validation_end,
                order=order,
                seasonal_order=seasonal_order,
                config=sarimax_config,
            )
            validation = _forecast_frame(
                frame,
                target=config.data.target,
                start=train_end,
                end=validation_end,
                predicted=predicted,
                split="validation",
            )
            score = evaluate_forecast(validation["actual"], validation["predicted"])
            validation_predictions[(order, seasonal_order)] = predicted
            row.update({
                "status": "success",
                "aic": aic,
                "converged": converged,
                "iterations": iterations,
                "Val Level RMSE": score["RMSE"],
                "Val Level MAE": score["MAE"],
                "Val Level MAPE": score["MAPE"],
                "Val Level R2": score["R2"],
                "Val Level Directional_Accuracy": score["Directional_Accuracy"],
            })
        except (ValueError, np.linalg.LinAlgError) as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        search_rows.append(row)

    search = pd.DataFrame(search_rows)
    search["eligible_for_selection"] = (
        search["status"].eq("success")
        & search["converged"].fillna(False).astype(bool)
    )
    successful = search.loc[search["eligible_for_selection"]].copy()
    if successful.empty:
        write_csv(search, output_dir / "sarimax_validation_search.csv")
        raise ValueError("No SARIMAX validation candidate converged successfully.")
    successful = successful.sort_values(
        ["Val Level RMSE", "Val Level MAE", "Val Level MAPE", "aic"],
        na_position="last",
    ).reset_index(drop=True)
    successful.insert(0, "validation_rank", range(1, len(successful) + 1))
    successful["selected"] = False
    successful["refit_attempted"] = False
    successful["refit_converged"] = False
    successful["refit_error"] = ""
    successful["selection_protocol"] = (
        "level_validation_rmse_subject_to_train_validation_refit_convergence"
    )

    selected_order: tuple[int, int, int] | None = None
    selected_seasonal_order: tuple[int, int, int, int] | None = None
    test_prediction: np.ndarray | None = None
    refit_aic = float("nan")
    refit_converged = False
    refit_iterations = 0
    refit_attempts = 0
    for index, candidate in successful.iterrows():
        order = tuple(candidate["order"])
        seasonal_order = tuple(candidate["seasonal_order"])
        successful.loc[index, "refit_attempted"] = True
        refit_attempts += 1
        try:
            prediction, candidate_aic, converged, iterations = fit_predict_sarimax(
                frame,
                target=config.data.target,
                features=config.data.features,
                fit_end=validation_end,
                prediction_start=validation_end,
                prediction_end=len(frame),
                order=order,
                seasonal_order=seasonal_order,
                config=sarimax_config,
            )
        except (ValueError, np.linalg.LinAlgError) as exc:
            successful.loc[index, "refit_error"] = f"{type(exc).__name__}: {exc}"
            continue
        successful.loc[index, "refit_converged"] = bool(converged)
        if not converged:
            successful.loc[index, "refit_error"] = "optimizer_did_not_converge"
            continue
        successful.loc[index, "selected"] = True
        selected_order = order
        selected_seasonal_order = seasonal_order
        test_prediction = prediction
        refit_aic = candidate_aic
        refit_converged = True
        refit_iterations = iterations
        break
    if selected_order is None or selected_seasonal_order is None or test_prediction is None:
        write_csv(search, output_dir / "sarimax_validation_search.csv")
        write_csv(successful, output_dir / "sarimax_selection.csv")
        raise ValueError("No validation-ranked SARIMAX candidate converged after refit.")

    search["selected"] = (
        search["order"].map(lambda value: tuple(value) == selected_order)
        & search["seasonal_order"].map(
            lambda value: tuple(value) == selected_seasonal_order
        )
    )
    write_csv(search, output_dir / "sarimax_validation_search.csv")
    write_csv(successful, output_dir / "sarimax_selection.csv")

    validation = _forecast_frame(
        frame,
        target=config.data.target,
        start=train_end,
        end=validation_end,
        predicted=validation_predictions[(selected_order, selected_seasonal_order)],
        split="validation",
    )
    test = _forecast_frame(
        frame,
        target=config.data.target,
        start=validation_end,
        end=len(frame),
        predicted=test_prediction,
        split="test",
    )
    forecasts = pd.concat([validation, test], ignore_index=True)
    metrics = _metric_frame(forecasts, target=config.data.target)
    write_csv(forecasts, output_dir / "forecasts.csv")
    write_csv(metrics, output_dir / "metrics.csv")
    result = {
        "order": selected_order,
        "seasonal_order": selected_seasonal_order,
        "exog_lag": sarimax_config.exog_lag,
        "trend": sarimax_config.trend,
        "refit_aic": refit_aic,
        "refit_converged": refit_converged,
        "refit_iterations": refit_iterations,
    }
    return result, metrics, forecasts, successful, len(search) + refit_attempts
