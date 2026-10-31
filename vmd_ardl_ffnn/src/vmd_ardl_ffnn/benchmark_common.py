"""Shared scoring and audit contracts for benchmark workflows."""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ExperimentConfig
from .evaluation import normalize_forecasts
from .experiment import VMDARDLFFNNExperiment
from .metrics import evaluate_forecast

def aggregate_seed_metrics(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        column for column in ("RMSE", "MAE", "MAPE", "R2", "Directional_Accuracy")
        if column in seed_metrics
    ]
    grouped = seed_metrics.groupby(["model", "pipeline", "target", "split", "scale"], dropna=False)
    aggregate = grouped[metric_columns].agg(["mean", "std", "median", "count"]).reset_index()
    aggregate.columns = [
        column if isinstance(column, str) else "_".join(part for part in column if part)
        for column in aggregate.columns
    ]
    aggregate.insert(5, "estimand", "distribution_of_per_seed_metrics")
    return aggregate

def ensemble_prediction_metrics(
    forecasts: pd.DataFrame,
    seed_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Score the date-wise mean prediction across seeds for every pipeline."""
    if forecasts.empty:
        return pd.DataFrame()
    group_columns = ["pipeline", "split", "scale", "date"]
    actual_counts = forecasts.groupby(group_columns)["actual"].nunique(dropna=False)
    if actual_counts.gt(1).any():
        raise ValueError(
            "Cannot score ensemble predictions because actual values differ "
            "within a pipeline/split/date."
        )
    ensemble = (
        forecasts.groupby(group_columns, as_index=False)
        .agg(
            actual=("actual", "first"),
            predicted=("predicted", "mean"),
            ensemble_member_count=("seed", "nunique"),
        )
    )
    ensemble["ensemble_member_count"] = ensemble[
        "ensemble_member_count"
    ].clip(lower=1)
    metadata = (
        seed_metrics.groupby("pipeline", as_index=False)
        .agg(model=("model", "first"), target=("target", "first"))
        .set_index("pipeline")
    )
    rows: list[dict[str, Any]] = []
    for (pipeline, split, scale), frame in ensemble.groupby(
        ["pipeline", "split", "scale"],
        sort=True,
    ):
        metrics = evaluate_forecast(frame["actual"], frame["predicted"])
        rows.append(
            {
                "model": metadata.loc[pipeline, "model"],
                "pipeline": pipeline,
                "target": metadata.loc[pipeline, "target"],
                "split": split,
                "scale": scale,
                "estimand": "ensemble_mean_prediction",
                "ensemble_member_count": int(
                    frame["ensemble_member_count"].max()
                ),
                **metrics,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["pipeline", "split", "scale"]
    ).reset_index(drop=True)

def lag_count(value: Any) -> int:
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return 0
    if isinstance(value, dict):
        return sum(len(tuple(lags)) for lags in value.values())
    if isinstance(value, (tuple, list, set)):
        return len(value)
    return 0

def lag_selection_rows(
    result: dict[str, pd.DataFrame],
    *,
    pipeline: str,
    seed: int,
) -> pd.DataFrame:
    best = result.get("best_component_models", pd.DataFrame()).copy()
    if best.empty:
        return pd.DataFrame()
    if "seed" in best.columns:
        best = best[best["seed"].eq(seed)].copy()
        best = best.rename(columns={"seed": "ffnn_seed"})
    best.insert(0, "pipeline", pipeline)
    best.insert(1, "seed", seed)
    best["target_lag_count"] = best["target_lags"].map(lag_count)
    best["exogenous_lag_count"] = best["exog_lags"].map(lag_count)
    best["ffnn_input_dimension"] = best["target_lag_count"] + best["exogenous_lag_count"]
    columns = [
        "pipeline", "seed", "component", "target_lags", "exog_lags", "lag_spec",
        "target_lag_count", "exogenous_lag_count", "ffnn_input_dimension",
        "hidden_layer_sizes", "alpha", "ffnn_seed", "activation", "learning_rate_init",
    ]
    return best[[column for column in columns if column in best.columns]]

def efficiency_row(
    result: dict[str, pd.DataFrame],
    config: ExperimentConfig,
    *,
    pipeline: str,
    seed: int | None,
    elapsed_seconds: float,
) -> dict[str, Any]:
    orders = result.get("ardl_orders", pd.DataFrame())
    best = result.get("best_component_models", pd.DataFrame())
    components = max(1, int(best["component"].nunique())) if "component" in best else 1
    theoretical = int(config.ardl.max_exog_lag) ** len(config.data.features) * components
    if not orders.empty and "feature" in orders:
        retained = int(orders["feature"].eq("__ffnn_combo__").sum())
    else:
        retained = 0
    search = result.get("component_search_results", result.get("search_results", pd.DataFrame()))
    lag_screen = result.get("lag_screen_fold_metrics", pd.DataFrame())
    block_subset_fast_screen = result.get(
        "block_subset_fast_screen_metrics",
        pd.DataFrame(),
    )
    evaluated_fits = (
        int(len(search))
        + int(len(lag_screen))
        + int(len(block_subset_fast_screen))
    )
    timing = result.get("timing", pd.DataFrame())
    phase = timing.iloc[0].to_dict() if isinstance(timing, pd.DataFrame) and not timing.empty else {}
    reduction = (
        100.0 * (theoretical - retained) / theoretical
        if theoretical > 0 and retained <= theoretical
        else float("nan")
    )
    return {
        "pipeline": pipeline,
        "seed": seed,
        "runtime_seconds": elapsed_seconds,
        "ardl_selection_seconds": phase.get("ardl_selection_seconds"),
        "vmd_cache_seconds": phase.get("vmd_cache_seconds"),
        "ffnn_search_seconds": phase.get("ffnn_search_seconds"),
        "refit_forecast_seconds": phase.get("refit_forecast_seconds"),
        "component_count": components,
        "theoretical_full_product_lag_specs": theoretical,
        "retained_lag_specs_after_ardl": retained,
        "lag_search_reduction_pct": reduction,
        "evaluated_ffnn_fits": evaluated_fits,
    }

def dataset_audit(
    csv_path: str | Path,
    config: ExperimentConfig,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    experiment = VMDARDLFFNNExperiment(replace(config, output_dir=output_dir / "study_audit"))
    data = experiment.load_data(csv_path)
    train, validation, test = experiment.split_raw_data(data)
    experiment._apply_stationarity_transform(data, train)
    dataset = pd.DataFrame(
        [
            {
                "variable": variable,
                "role": "target" if variable == config.data.target else "exogenous",
                "frequency": config.data.freq,
                "log_transform": config.data.log_transform,
                "stationarity_transform": experiment.transform_info_.get(variable, "level"),
                "n_observations": len(data),
                "sample_start": data.index.min(),
                "sample_end": data.index.max(),
            }
            for variable in (config.data.target, *config.data.features)
        ]
    )
    splits = pd.DataFrame(
        [
            {
                "split": name,
                "n_observations": len(frame),
                "start": frame.index.min(),
                "end": frame.index.max(),
            }
            for name, frame in (("train", train), ("validation", validation), ("test", test))
        ]
    )
    stationarity = (
        experiment.stationarity_screen_.copy()
        if experiment.stationarity_screen_ is not None
        else pd.DataFrame()
    )
    return dataset, splits, stationarity

def forecast_date_coverage(forecasts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split, split_frame in forecasts.groupby("split", sort=False):
        date_sets = {
            pipeline: set(pd.to_datetime(group["date"]))
            for pipeline, group in split_frame.groupby("pipeline", sort=True)
        }
        common = set.intersection(*date_sets.values()) if date_sets else set()
        union = set.union(*date_sets.values()) if date_sets else set()
        for pipeline, dates in date_sets.items():
            rows.append(
                {
                    "pipeline": pipeline,
                    "split": split,
                    "n_dates": len(dates),
                    "common_date_count": len(common),
                    "union_date_count": len(union),
                    "same_dates_for_all_models": dates == union == common,
                    "start": min(dates) if dates else pd.NaT,
                    "end": max(dates) if dates else pd.NaT,
                }
            )
    coverage = pd.DataFrame(rows)
    if not coverage.empty and not coverage["same_dates_for_all_models"].all():
        mismatches = coverage.loc[
            ~coverage["same_dates_for_all_models"], ["pipeline", "split", "n_dates"]
        ].to_dict("records")
        raise ValueError(f"Benchmark models do not share identical forecast dates: {mismatches}")
    return coverage

def normalized_result(
    result: dict[str, Any],
    *,
    pipeline: str,
    model: str,
    uses_vmd: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = result["final_metrics"].copy()
    metrics = metrics[metrics["scale"].eq("level")].copy()
    metrics["model"] = model
    metrics["pipeline"] = pipeline
    predicted_col = "predicted_reconstructed" if uses_vmd else "predicted_raw_transformed"
    forecasts = normalize_forecasts(
        result["final_forecasts"],
        pipeline=pipeline,
        predicted_col=predicted_col,
    )
    return metrics, forecasts
