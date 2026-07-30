from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd

from .artifacts import write_csv
from .config import ExperimentConfig
from .evaluation import normalize_forecasts
from .experiment import VMDARDLFFNNExperiment
from .hard_ffnn_experiments import (
    HARD_FFNN_MODEL,
    HARD_FFNN_PIPELINE,
    run_hard_ffnn_variant as _run_hard_ffnn_variant,
)
from .metrics import evaluate_forecast
from .models.varnn import VARNNConfig, run_varnn_pipeline
from .paper_baselines import run_linear_ardl_baseline, run_naive_baseline
from .paper_reporting import lag_selection_frequency, write_paper_figures
from .statistical_tests import pairwise_dm_tests


BLOCK_SUBSET_PIPELINE = "vmd_ardl_block_subset_mean_seed"
BLOCK_SUBSET_MODEL = "VMD_ARDL_BlockSubset_MeanSeed_FFNN"


NEURAL_VARIANTS: dict[str, dict[str, Any]] = {
    "ffnn_fixed_no_vmd": {
        "model": "FFNN_FixedLag",
        "uses_vmd": False,
        "target_strategy": "fixed",
        "exog_strategy": "fixed",
    },
    "vmd_ffnn_fixed": {
        "model": "VMD_FFNN_FixedLag",
        "uses_vmd": True,
        "target_strategy": "fixed",
        "exog_strategy": "fixed",
    },
    "vmd_target_select_fixed_exog": {
        "model": "VMD_ARDLTarget_FixedExog_FFNN",
        "uses_vmd": True,
        "target_strategy": "ardl_select_order",
        "exog_strategy": "fixed",
    },
    "vmd_fixed_target_ardl_exog": {
        "model": "VMD_FixedTarget_ARDLExog_FFNN",
        "uses_vmd": True,
        "target_strategy": "fixed",
        "exog_strategy": "ardl_screen",
    },
    "vmd_ardl_ffnn_proposed": {
        "model": "VMD_ARDL_FFNN_Proposed",
        "uses_vmd": True,
        "target_strategy": "ardl_select_order",
        "exog_strategy": "ardl_screen",
    },
    "vmd_ardl_joint_select": {
        "model": "VMD_ARDL_JointSelect_FFNN",
        "uses_vmd": True,
        "target_strategy": "ardl_select_order_joint",
        "exog_strategy": "ardl_screen",
        "reported_exog_strategy": "ardl_select_order_joint",
    },
    "ardl_ffnn_no_vmd": {
        "model": "ARDL_FFNN_NoVMD",
        "uses_vmd": False,
        "target_strategy": "ardl_select_order",
        "exog_strategy": "ardl_screen",
    },
}

BENCHMARK_PIPELINES = (
    "naive_last_observation",
    "naive_seasonal",
    "standalone_ardl",
    "ffnn_fixed_no_vmd",
    "ardl_ffnn_no_vmd",
    "vmd_ffnn_fixed",
    "varnn",
    "vmd_ardl_ffnn_proposed",
    BLOCK_SUBSET_PIPELINE,
    HARD_FFNN_PIPELINE,
)

ABLATION_IDS = {
    "ffnn_fixed_no_vmd": "A0",
    "vmd_ffnn_fixed": "A1",
    "vmd_target_select_fixed_exog": "A2",
    "vmd_fixed_target_ardl_exog": "A3",
    "vmd_ardl_ffnn_proposed": "A4",
    "vmd_ardl_joint_select": "A5",
}


def _normalized_neural_result(
    result: dict[str, pd.DataFrame],
    *,
    pipeline: str,
    model: str,
    seed: int,
    uses_vmd: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = result["final_metrics"].copy()
    metrics = metrics[metrics["scale"].eq("level")].copy()
    metrics["model"] = model
    metrics["pipeline"] = pipeline
    metrics["seed"] = seed
    predicted_col = "predicted_reconstructed" if uses_vmd else "predicted_raw_transformed"
    forecasts = normalize_forecasts(
        result["final_forecasts"],
        pipeline=pipeline,
        predicted_col=predicted_col,
    )
    forecasts["seed"] = seed
    return metrics, forecasts


def _run_neural_variant(
    csv_path: str | Path,
    config: ExperimentConfig,
    *,
    pipeline: str,
    definition: dict[str, Any],
    seed: int,
    output_dir: Path,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    variant_config = replace(
        config,
        output_dir=output_dir,
        ardl=replace(
            config.ardl,
            target_lag_strategy=definition["target_strategy"],
            exog_lag_strategy=definition["exog_strategy"],
        ),
        ffnn=replace(config.ffnn, seed_grid=(seed,)),
    )
    experiment = VMDARDLFFNNExperiment(variant_config)
    result = experiment.run(csv_path) if definition["uses_vmd"] else experiment.run_without_vmd(csv_path)
    metrics, forecasts = _normalized_neural_result(
        result,
        pipeline=pipeline,
        model=definition["model"],
        seed=seed,
        uses_vmd=definition["uses_vmd"],
    )
    return result, metrics, forecasts


def _run_block_subset_mean_seed_variant(
    csv_path: str | Path,
    config: ExperimentConfig,
    *,
    seeds: tuple[int, ...],
    output_dir: Path,
    search_strategy: str,
) -> tuple[
    dict[str, pd.DataFrame],
    pd.DataFrame,
    pd.DataFrame,
    ExperimentConfig,
]:
    """Run the block-subset comparator once with selection shared across seeds."""
    variant_config = replace(
        config,
        output_dir=output_dir,
        ardl=replace(
            config.ardl,
            ic="aic",
            target_lag_strategy="ardl_select_order",
            exog_lag_strategy="ardl_block_subsets",
        ),
        ffnn=replace(
            config.ffnn,
            seed_grid=tuple(seeds),
            search_strategy="full_grid",
        ),
    )
    result = VMDARDLFFNNExperiment(variant_config).run_block_subset_mean_seed(
        csv_path,
        reporting_seeds=tuple(seeds),
        search_strategy=search_strategy,
    )
    metrics = result["final_metrics"].copy()
    metrics = metrics[metrics["scale"].eq("level")].copy()
    metrics["model"] = BLOCK_SUBSET_MODEL
    metrics["pipeline"] = BLOCK_SUBSET_PIPELINE
    forecasts = normalize_forecasts(
        result["final_forecasts"],
        pipeline=BLOCK_SUBSET_PIPELINE,
        predicted_col="predicted_reconstructed",
    )
    return result, metrics, forecasts, variant_config


def _aggregate_seed_metrics(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        column for column in ("RMSE", "MAE", "MAPE", "Directional_Accuracy")
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


def _ensemble_prediction_metrics(
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


def _lag_count(value: Any) -> int:
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


def _lag_selection_rows(
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
    best["target_lag_count"] = best["target_lags"].map(_lag_count)
    best["exogenous_lag_count"] = best["exog_lags"].map(_lag_count)
    best["ffnn_input_dimension"] = best["target_lag_count"] + best["exogenous_lag_count"]
    columns = [
        "pipeline", "seed", "component", "target_lags", "exog_lags", "lag_spec",
        "target_lag_count", "exogenous_lag_count", "ffnn_input_dimension",
        "hidden_layer_sizes", "alpha", "ffnn_seed", "activation", "learning_rate_init",
    ]
    return best[[column for column in columns if column in best.columns]]


def _efficiency_row(
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


def _study_settings(
    seeds: tuple[int, ...],
    include_varnn: bool,
    *,
    ffnn_learning_rate_grid: tuple[float, ...],
    varnn_learning_rate_grid: tuple[float, ...],
    lag_screen_folds: int,
    lag_screen_validation_size: int,
    lag_screen_learning_rate_grid: tuple[float, ...],
    lag_screen_selection_rule: str,
    final_selection_scale: str,
    block_subset_search_strategy: str,
    hard_ffnn_policy_version: str,
    hard_ffnn_top_component_candidates: int,
    hard_ffnn_order_count: int,
    feature_count: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pipeline, definition in NEURAL_VARIANTS.items():
        rows.append(
            {
                "pipeline": pipeline,
                "model": definition["model"],
                "ablation_id": ABLATION_IDS.get(pipeline),
                "included_in_benchmark": pipeline in BENCHMARK_PIPELINES,
                "uses_vmd": definition["uses_vmd"],
                "target_lag_strategy": definition["target_strategy"],
                "exog_lag_strategy": definition.get(
                    "reported_exog_strategy", definition["exog_strategy"]
                ),
                "reporting_seeds": seeds,
                "ffnn_learning_rate_grid": ffnn_learning_rate_grid,
                "lag_screen_folds": lag_screen_folds,
                "lag_screen_validation_size": lag_screen_validation_size,
                "lag_screen_learning_rate_grid": lag_screen_learning_rate_grid,
                "lag_screen_selection_rule": lag_screen_selection_rule,
                "final_selection_scale": final_selection_scale,
                "varnn_learning_rate_grid": tuple(),
                "enabled": True,
            }
        )
    rows.append(
        {
            "pipeline": BLOCK_SUBSET_PIPELINE,
            "model": BLOCK_SUBSET_MODEL,
            "ablation_id": None,
            "included_in_benchmark": True,
            "uses_vmd": True,
            "target_lag_strategy": "ardl_select_order",
            "exog_lag_strategy": "aic_rank1_variable_block_subsets",
            "reporting_seeds": seeds,
            "ffnn_learning_rate_grid": ffnn_learning_rate_grid,
            "lag_screen_folds": 0,
            "lag_screen_validation_size": 0,
            "lag_screen_learning_rate_grid": tuple(),
            "lag_screen_selection_rule": None,
            "final_selection_scale": "reconstructed_level",
            "selection_seed_aggregation": (
                "mean_reconstructed_level_validation_across_reporting_seeds"
            ),
            "block_subset_search_strategy": block_subset_search_strategy,
            "block_subset_fast_screen_top_k": (
                4 if block_subset_search_strategy == "two_stage" else None
            ),
            "block_subset_fast_screen_alpha": (
                0.1 if block_subset_search_strategy == "two_stage" else None
            ),
            "block_subset_fast_screen_learning_rate_init": (
                0.001 if block_subset_search_strategy == "two_stage" else None
            ),
            "varnn_learning_rate_grid": tuple(),
            "enabled": True,
        }
    )
    rows.append(
        {
            "pipeline": HARD_FFNN_PIPELINE,
            "model": HARD_FFNN_MODEL,
            "ablation_id": None,
            "included_in_benchmark": True,
            "uses_vmd": True,
            "target_lag_strategy": "ardl_select_order",
            "exog_lag_strategy": "ardl_screen_top3_full_product",
            "reporting_seeds": seeds,
            "ffnn_learning_rate_grid": tuple(),
            "lag_screen_folds": 0,
            "lag_screen_validation_size": 0,
            "lag_screen_learning_rate_grid": tuple(),
            "lag_screen_selection_rule": None,
            "final_selection_scale": "reconstructed_level",
            "selection_seed_aggregation": (
                "mean_reconstructed_level_validation_across_reporting_seeds"
            ),
            "search_strategy": "full_product_lag_search_with_hard_ffnn_policy",
            "ffnn_config_source": "train_only_frequency_lag_policy",
            "hard_ffnn_policy_version": hard_ffnn_policy_version,
            "top_component_candidates": hard_ffnn_top_component_candidates,
            "theoretical_lag_spec_count_per_component": (
                hard_ffnn_order_count ** feature_count
            ),
            "varnn_learning_rate_grid": tuple(),
            "enabled": True,
        }
    )
    rows.extend(
        {
            "pipeline": pipeline,
            "model": model,
            "ablation_id": None,
            "included_in_benchmark": True,
            "uses_vmd": False,
            "target_lag_strategy": None,
            "exog_lag_strategy": None,
            "reporting_seeds": seeds if pipeline == "varnn" else tuple(),
            "ffnn_learning_rate_grid": tuple(),
            "lag_screen_folds": None,
            "lag_screen_validation_size": None,
            "lag_screen_learning_rate_grid": tuple(),
            "lag_screen_selection_rule": None,
            "final_selection_scale": None,
            "varnn_learning_rate_grid": varnn_learning_rate_grid
            if pipeline == "varnn"
            else tuple(),
            "enabled": include_varnn if pipeline == "varnn" else True,
        }
        for pipeline, model in (
            ("naive_last_observation", "Naive_LastObservation"),
            ("naive_seasonal", "Naive_Seasonal12"),
            ("standalone_ardl", "Standalone_ARDL"),
            ("varnn", "VARNN"),
        )
    )
    return pd.DataFrame(rows)


def _dataset_audit(
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


def _forecast_date_coverage(forecasts: pd.DataFrame) -> pd.DataFrame:
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


def run_paper_experiments(
    csv_path: str | Path,
    config: ExperimentConfig,
    varnn_config: VARNNConfig | None = None,
    *,
    seeds: tuple[int, ...] = (7, 42, 123, 2024, 2025),
    include_varnn: bool = True,
    block_subset_search_strategy: str = "full_grid",
) -> dict[str, Any]:
    """Run the B0–B7 benchmark, A0–A5 ablations, summaries and DM tests."""
    seeds = tuple(dict.fromkeys(int(seed) for seed in seeds))
    if not seeds:
        raise ValueError("Paper experiments require at least one reporting seed.")
    block_subset_search_strategy = block_subset_search_strategy.replace("-", "_")
    if block_subset_search_strategy not in {"full_grid", "two_stage"}:
        raise ValueError(
            "block_subset_search_strategy must be 'full_grid' or 'two_stage'."
        )
    root = Path(config.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    dataset_summary, split_summary, stationarity_summary = _dataset_audit(csv_path, config, root)
    metric_frames: list[pd.DataFrame] = []
    forecast_frames: list[pd.DataFrame] = []
    lag_frames: list[pd.DataFrame] = []
    lag_screen_fold_frames: list[pd.DataFrame] = []
    lag_screen_stability_frames: list[pd.DataFrame] = []
    efficiency_rows: list[dict[str, Any]] = []
    variant_results: dict[str, list[dict[str, pd.DataFrame]]] = {}

    deterministic_baselines = (
        (
            "naive_last_observation",
            run_naive_baseline,
            {"lag": 1, "model_name": "Naive_LastObservation"},
        ),
        (
            "naive_seasonal",
            run_naive_baseline,
            {"lag": 12, "model_name": "Naive_Seasonal12"},
        ),
        (
            "standalone_ardl",
            run_linear_ardl_baseline,
            {},
        ),
    )
    for pipeline, runner, kwargs in deterministic_baselines:
        started = perf_counter()
        result = runner(csv_path, replace(config, output_dir=root / pipeline), **kwargs)
        efficiency_rows.append(
            {
                "pipeline": pipeline,
                "seed": None,
                "runtime_seconds": perf_counter() - started,
                "ardl_selection_seconds": None,
                "vmd_cache_seconds": None,
                "ffnn_search_seconds": None,
                "refit_forecast_seconds": None,
                "component_count": 1,
                "theoretical_full_product_lag_specs": None,
                "retained_lag_specs_after_ardl": None,
                "lag_search_reduction_pct": None,
                "evaluated_ffnn_fits": 0,
            }
        )
        metrics = result["metrics"].copy()
        metrics["pipeline"] = pipeline
        metrics["seed"] = pd.NA
        forecasts = result["forecasts"].copy()
        forecasts["pipeline"] = pipeline
        forecasts["seed"] = pd.NA
        metric_frames.append(metrics)
        forecast_frames.append(forecasts)

    for pipeline, definition in NEURAL_VARIANTS.items():
        variant_results[pipeline] = []
        for seed in seeds:
            started = perf_counter()
            result, metrics, forecasts = _run_neural_variant(
                csv_path,
                config,
                pipeline=pipeline,
                definition=definition,
                seed=seed,
                output_dir=root / pipeline / f"seed_{seed}",
            )
            variant_results[pipeline].append(result)
            metric_frames.append(metrics)
            forecast_frames.append(forecasts)
            lag_frames.append(_lag_selection_rows(result, pipeline=pipeline, seed=seed))
            fold_metrics = result.get("lag_screen_fold_metrics", pd.DataFrame())
            if not fold_metrics.empty:
                lag_screen_fold_frames.append(
                    fold_metrics.assign(pipeline=pipeline, seed=seed)
                )
            stability = result.get("lag_screen_stability", pd.DataFrame())
            if not stability.empty:
                lag_screen_stability_frames.append(
                    stability.assign(pipeline=pipeline, seed=seed)
                )
            efficiency_rows.append(
                _efficiency_row(
                    result,
                    config,
                    pipeline=pipeline,
                    seed=seed,
                    elapsed_seconds=perf_counter() - started,
                )
            )

    started = perf_counter()
    block_result, block_metrics, block_forecasts, block_config = (
        _run_block_subset_mean_seed_variant(
            csv_path,
            config,
            seeds=tuple(seeds),
            output_dir=root / BLOCK_SUBSET_PIPELINE,
            search_strategy=block_subset_search_strategy,
        )
    )
    variant_results[BLOCK_SUBSET_PIPELINE] = [block_result]
    metric_frames.append(block_metrics)
    forecast_frames.append(block_forecasts)
    for seed in seeds:
        lag_frames.append(
            _lag_selection_rows(
                block_result,
                pipeline=BLOCK_SUBSET_PIPELINE,
                seed=seed,
            )
        )
    efficiency_rows.append(
        _efficiency_row(
            block_result,
            block_config,
            pipeline=BLOCK_SUBSET_PIPELINE,
            seed=None,
            elapsed_seconds=perf_counter() - started,
        )
    )

    started = perf_counter()
    hard_result, hard_metrics, hard_forecasts, hard_config = (
        _run_hard_ffnn_variant(
            csv_path,
            config,
            seeds=tuple(seeds),
            output_dir=root / HARD_FFNN_PIPELINE,
        )
    )
    variant_results[HARD_FFNN_PIPELINE] = [hard_result]
    metric_frames.append(hard_metrics)
    forecast_frames.append(hard_forecasts)
    for seed in seeds:
        lag_frames.append(
            _lag_selection_rows(
                hard_result,
                pipeline=HARD_FFNN_PIPELINE,
                seed=seed,
            )
        )
    hard_efficiency = _efficiency_row(
        hard_result,
        hard_config,
        pipeline=HARD_FFNN_PIPELINE,
        seed=None,
        elapsed_seconds=perf_counter() - started,
    )
    hard_component_count = int(
        hard_result["best_component_models"]["component"].nunique()
    )
    hard_order_count = min(3, int(config.ardl.max_exog_lag))
    hard_theoretical_specs = (
        hard_order_count ** len(config.data.features)
    ) * hard_component_count
    hard_retained_specs = len(
        hard_result.get("hard_ffnn_policy_assignments", pd.DataFrame())
    )
    hard_efficiency.update(
        {
            "theoretical_full_product_lag_specs": hard_theoretical_specs,
            "retained_lag_specs_after_ardl": hard_retained_specs,
            "lag_search_reduction_pct": (
                100.0
                * (hard_theoretical_specs - hard_retained_specs)
                / hard_theoretical_specs
                if hard_theoretical_specs
                and hard_retained_specs <= hard_theoretical_specs
                else float("nan")
            ),
        }
    )
    efficiency_rows.append(hard_efficiency)

    skipped_rows: list[dict[str, str]] = []
    if include_varnn:
        variant_results["varnn"] = []
        for seed in seeds:
            started = perf_counter()
            try:
                current_varnn = replace(varnn_config or VARNNConfig(), seed=seed)
                result = run_varnn_pipeline(
                    csv_path,
                    replace(config, output_dir=root / "varnn" / f"seed_{seed}"),
                    current_varnn,
                )
            except ImportError as exc:
                skipped_rows.append({"pipeline": "varnn", "reason": str(exc)})
                break
            metrics = result["metrics"]
            metrics = metrics[metrics["scale"].eq("level")].copy()
            metrics["pipeline"] = "varnn"
            metrics["seed"] = seed
            forecasts = normalize_forecasts(
                result["forecasts"], pipeline="varnn", predicted_col="predicted_raw_transformed"
            )
            forecasts["seed"] = seed
            variant_results["varnn"].append(result)
            metric_frames.append(metrics)
            forecast_frames.append(forecasts)
            efficiency_rows.append(
                {
                    "pipeline": "varnn",
                    "seed": seed,
                    "runtime_seconds": perf_counter() - started,
                    "ardl_selection_seconds": None,
                    "vmd_cache_seconds": None,
                    "ffnn_search_seconds": None,
                    "refit_forecast_seconds": None,
                    "component_count": 1,
                    "theoretical_full_product_lag_specs": None,
                    "retained_lag_specs_after_ardl": None,
                    "lag_search_reduction_pct": None,
                    "evaluated_ffnn_fits": 1,
                }
            )

    seed_metrics = pd.concat(metric_frames, ignore_index=True, sort=False)
    forecasts = pd.concat(forecast_frames, ignore_index=True, sort=False)
    aggregate_metrics = _aggregate_seed_metrics(seed_metrics)
    benchmark_forecasts = forecasts[forecasts["pipeline"].isin(BENCHMARK_PIPELINES)].copy()
    benchmark_seed_robustness = aggregate_metrics[
        aggregate_metrics["pipeline"].isin(BENCHMARK_PIPELINES)
    ].copy()
    benchmark_ensemble_metrics = _ensemble_prediction_metrics(
        benchmark_forecasts,
        seed_metrics,
    )
    # Primary reporting estimand: expected single-model performance across
    # random initializations.  The mean-prediction ensemble remains an
    # explicitly named secondary artifact.
    benchmark_metrics = benchmark_seed_robustness
    date_coverage = _forecast_date_coverage(benchmark_forecasts)
    ablation_metrics = aggregate_metrics[aggregate_metrics["pipeline"].isin(ABLATION_IDS)].copy()
    ablation_metrics.insert(0, "ablation_id", ablation_metrics["pipeline"].map(ABLATION_IDS))
    ablation_forecasts = forecasts[forecasts["pipeline"].isin(ABLATION_IDS)].copy()
    dm_tests = pairwise_dm_tests(benchmark_forecasts, split="test", loss="squared", horizon=1)
    skipped = pd.DataFrame(skipped_rows)
    lag_selection = pd.concat(
        [frame for frame in lag_frames if not frame.empty], ignore_index=True, sort=False
    ) if any(not frame.empty for frame in lag_frames) else pd.DataFrame()
    efficiency = pd.DataFrame(efficiency_rows)
    settings = _study_settings(
        tuple(seeds),
        include_varnn,
        ffnn_learning_rate_grid=tuple(config.ffnn.learning_rate_grid),
        varnn_learning_rate_grid=tuple((varnn_config or VARNNConfig()).learning_rate_grid),
        lag_screen_folds=int(config.ffnn.lag_screen_folds),
        lag_screen_validation_size=int(config.ffnn.lag_screen_validation_size),
        lag_screen_learning_rate_grid=tuple(config.ffnn.lag_screen_learning_rates()),
        lag_screen_selection_rule=config.ffnn.lag_screen_selection_rule,
        final_selection_scale=config.ffnn.final_selection_scale,
        block_subset_search_strategy=block_subset_search_strategy,
        hard_ffnn_policy_version=config.hard_ffnn.policy_version,
        hard_ffnn_top_component_candidates=(
            config.hard_ffnn.top_component_candidates
        ),
        hard_ffnn_order_count=min(3, int(config.ardl.max_exog_lag)),
        feature_count=len(config.data.features),
    )
    lag_frequency = lag_selection_frequency(lag_selection)
    lag_screen_folds = (
        pd.concat(lag_screen_fold_frames, ignore_index=True, sort=False)
        if lag_screen_fold_frames
        else pd.DataFrame()
    )
    lag_screen_stability = (
        pd.concat(lag_screen_stability_frames, ignore_index=True, sort=False)
        if lag_screen_stability_frames
        else pd.DataFrame()
    )

    write_csv(seed_metrics, root / "paper_seed_metrics.csv")
    write_csv(dataset_summary, root / "paper_dataset_summary.csv")
    write_csv(split_summary, root / "paper_split_summary.csv")
    write_csv(stationarity_summary, root / "paper_stationarity_screen.csv")
    write_csv(date_coverage, root / "paper_forecast_date_coverage.csv")
    write_csv(aggregate_metrics, root / "paper_metrics_summary.csv")
    write_csv(
        _ensemble_prediction_metrics(forecasts, seed_metrics),
        root / "paper_ensemble_metrics_summary.csv",
    )
    write_csv(forecasts, root / "paper_forecasts.csv")
    write_csv(benchmark_metrics, root / "benchmark_metrics_comparison.csv")
    write_csv(
        benchmark_ensemble_metrics,
        root / "benchmark_ensemble_metrics_comparison.csv",
    )
    write_csv(
        benchmark_seed_robustness,
        root / "benchmark_seed_robustness_metrics.csv",
    )
    write_csv(benchmark_forecasts, root / "benchmark_forecasts_comparison.csv")
    write_csv(ablation_metrics, root / "ablation_metrics.csv")
    write_csv(ablation_forecasts, root / "ablation_forecasts.csv")
    write_csv(dm_tests, root / "diebold_mariano_tests.csv")
    write_csv(lag_selection, root / "lag_selection_summary.csv")
    write_csv(lag_frequency, root / "lag_selection_frequency.csv")
    if not lag_screen_folds.empty:
        write_csv(lag_screen_folds, root / "lag_screen_fold_metrics.csv")
    if not lag_screen_stability.empty:
        write_csv(lag_screen_stability, root / "lag_screen_stability.csv")
    write_csv(efficiency, root / "efficiency_summary.csv")
    write_csv(settings, root / "paper_experiment_settings.csv")
    if not skipped.empty:
        write_csv(skipped, root / "skipped_models.csv")
    write_paper_figures(
        benchmark_forecasts,
        benchmark_ensemble_metrics,
        efficiency,
        lag_frequency,
        root,
        seed_robustness_metrics=benchmark_seed_robustness,
        training_components=variant_results["vmd_ardl_ffnn_proposed"][0].get(
            "training_components"
        ),
        target=config.data.target,
    )

    return {
        "comparison_metrics": benchmark_metrics,
        "ensemble_comparison_metrics": benchmark_ensemble_metrics,
        "seed_robustness_metrics": benchmark_seed_robustness,
        "comparison_forecasts": benchmark_forecasts,
        "seed_metrics": seed_metrics,
        "aggregate_metrics": aggregate_metrics,
        "ablation_metrics": ablation_metrics,
        "ablation_forecasts": ablation_forecasts,
        "dm_tests": dm_tests,
        "lag_selection": lag_selection,
        "lag_frequency": lag_frequency,
        "lag_screen_fold_metrics": lag_screen_folds,
        "lag_screen_stability": lag_screen_stability,
        "efficiency": efficiency,
        "settings": settings,
        "dataset_summary": dataset_summary,
        "split_summary": split_summary,
        "stationarity_summary": stationarity_summary,
        "date_coverage": date_coverage,
        "skipped_models": skipped,
        "variant_results": variant_results,
        "vmd_ardl_ffnn": variant_results["vmd_ardl_ffnn_proposed"],
        "vmd_ardl_select_order_ffnn": variant_results["vmd_ardl_ffnn_proposed"],
        "no_vmd_ardl_ffnn": variant_results["ardl_ffnn_no_vmd"],
        BLOCK_SUBSET_PIPELINE: variant_results[BLOCK_SUBSET_PIPELINE],
        HARD_FFNN_PIPELINE: variant_results[HARD_FFNN_PIPELINE],
        "varnn": variant_results.get("varnn", []),
    }
