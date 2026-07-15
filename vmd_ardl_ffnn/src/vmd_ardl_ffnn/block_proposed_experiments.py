from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd

from .artifacts import write_csv
from .config import ExperimentConfig
from .evaluation import normalize_forecasts
from .experiment import VMDARDLFFNNExperiment
from .models.varnn import VARNNConfig, run_varnn_pipeline
from .paper_baselines import run_linear_ardl_baseline, run_naive_baseline
from .paper_experiments import (
    _aggregate_seed_metrics,
    _dataset_audit,
    _efficiency_row,
    _forecast_date_coverage,
    _lag_selection_rows,
)
from .paper_reporting import lag_selection_frequency, write_paper_figures
from .statistical_tests import pairwise_dm_tests


PROPOSED_PIPELINE = "vmd_ardl_block_subset_proposed"
PROPOSED_MODEL = "VMD_ARDL_BlockSubset_TwoStage_Proposed"
SEQUENTIAL_PIPELINE = "vmd_ardl_sequential_screen"
JOINT_SELECT_PIPELINE = "vmd_ardl_joint_select"

BENCHMARK_PIPELINES = (
    "naive_last_observation",
    "naive_seasonal",
    "standalone_ardl",
    "ffnn_fixed_no_vmd",
    "vmd_ffnn_fixed",
    "ardl_ffnn_no_vmd",
    "varnn",
    SEQUENTIAL_PIPELINE,
    JOINT_SELECT_PIPELINE,
    PROPOSED_PIPELINE,
)

ABLATION_IDS = {
    "ffnn_fixed_no_vmd": "A0",
    "vmd_ffnn_fixed": "A1",
    "a2_vmd_target_fixed_exog": "A2",
    "a3_vmd_aic_all_blocks": "A3",
    "a4_vmd_block_subset_component_mean_seed": "A4",
    "a5_vmd_block_subset_shared_full_grid": "A5",
    "a6_vmd_block_subset_per_seed_full_grid": "A6",
    PROPOSED_PIPELINE: "A7",
}

ABLATION_SCHEMA_VERSION = "block_subset_proposed_v1"


def _shared_config(
    config: ExperimentConfig,
    *,
    output_dir: Path,
    seeds: tuple[int, ...],
    target_strategy: str,
    exog_strategy: str,
) -> ExperimentConfig:
    return replace(
        config,
        output_dir=output_dir,
        ardl=replace(
            config.ardl,
            target_lag_strategy=target_strategy,
            exog_lag_strategy=exog_strategy,
        ),
        ffnn=replace(
            config.ffnn,
            seed_grid=seeds,
            search_strategy="full_grid",
        ),
    )


def _normalized_result(
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


def _run_shared_vmd_variant(
    csv_path: str | Path,
    config: ExperimentConfig,
    *,
    pipeline: str,
    model: str,
    seeds: tuple[int, ...],
    output_dir: Path,
    target_strategy: str,
    exog_strategy: str,
    selection_mode: str = "reconstructed_mean_seed",
    lag_spec_scope: str = "all",
    screening_mode: str = "none",
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, ExperimentConfig]:
    variant_config = _shared_config(
        config,
        output_dir=output_dir,
        seeds=seeds,
        target_strategy=target_strategy,
        exog_strategy=exog_strategy,
    )
    result = VMDARDLFFNNExperiment(variant_config).run_vmd_shared_mean_seed(
        csv_path,
        reporting_seeds=seeds,
        model_name=model,
        selection_mode=selection_mode,
        lag_spec_scope=lag_spec_scope,
        screening_mode=screening_mode,
        artifact_stem=pipeline,
    )
    metrics, forecasts = _normalized_result(
        result,
        pipeline=pipeline,
        model=model,
        uses_vmd=True,
    )
    return result, metrics, forecasts, variant_config


def _run_shared_no_vmd_variant(
    csv_path: str | Path,
    config: ExperimentConfig,
    *,
    pipeline: str,
    model: str,
    seeds: tuple[int, ...],
    output_dir: Path,
    target_strategy: str,
    exog_strategy: str,
    screening_mode: str = "none",
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, ExperimentConfig]:
    variant_config = _shared_config(
        config,
        output_dir=output_dir,
        seeds=seeds,
        target_strategy=target_strategy,
        exog_strategy=exog_strategy,
    )
    result = VMDARDLFFNNExperiment(variant_config).run_no_vmd_shared_mean_seed(
        csv_path,
        reporting_seeds=seeds,
        model_name=model,
        screening_mode=screening_mode,
        artifact_stem=pipeline,
    )
    metrics, forecasts = _normalized_result(
        result,
        pipeline=pipeline,
        model=model,
        uses_vmd=False,
    )
    return result, metrics, forecasts, variant_config


def _run_shared_block_variant(
    csv_path: str | Path,
    config: ExperimentConfig,
    *,
    pipeline: str,
    model: str,
    seeds: tuple[int, ...],
    output_dir: Path,
    search_strategy: str,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, ExperimentConfig]:
    variant_config = _shared_config(
        config,
        output_dir=output_dir,
        seeds=seeds,
        target_strategy="ardl_select_order",
        exog_strategy="ardl_block_subsets",
    )
    result = VMDARDLFFNNExperiment(variant_config).run_block_subset_mean_seed(
        csv_path,
        reporting_seeds=seeds,
        search_strategy=search_strategy,
    )
    metrics, forecasts = _normalized_result(
        result,
        pipeline=pipeline,
        model=model,
        uses_vmd=True,
    )
    return result, metrics, forecasts, variant_config


def _run_per_seed_block_variant(
    csv_path: str | Path,
    config: ExperimentConfig,
    *,
    pipeline: str,
    model: str,
    seeds: tuple[int, ...],
    output_dir: Path,
) -> tuple[
    list[dict[str, Any]],
    pd.DataFrame,
    pd.DataFrame,
    list[tuple[ExperimentConfig, float]],
]:
    results: list[dict[str, Any]] = []
    metric_frames: list[pd.DataFrame] = []
    forecast_frames: list[pd.DataFrame] = []
    run_metadata: list[tuple[ExperimentConfig, float]] = []
    for seed in seeds:
        seed_config = _shared_config(
            config,
            output_dir=output_dir / f"seed_{seed}",
            seeds=(seed,),
            target_strategy="ardl_select_order",
            exog_strategy="ardl_block_subsets",
        )
        started = perf_counter()
        result = VMDARDLFFNNExperiment(seed_config).run(csv_path)
        elapsed = perf_counter() - started
        metrics, forecasts = _normalized_result(
            result,
            pipeline=pipeline,
            model=model,
            uses_vmd=True,
        )
        metrics["seed"] = int(seed)
        forecasts["seed"] = int(seed)
        results.append(result)
        metric_frames.append(metrics)
        forecast_frames.append(forecasts)
        run_metadata.append((seed_config, elapsed))
    return (
        results,
        pd.concat(metric_frames, ignore_index=True, sort=False),
        pd.concat(forecast_frames, ignore_index=True, sort=False),
        run_metadata,
    )


def _run_varnn_shared_mean_seed(
    csv_path: str | Path,
    config: ExperimentConfig,
    varnn_config: VARNNConfig,
    *,
    seeds: tuple[int, ...],
    output_dir: Path,
) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    search_frames: list[pd.DataFrame] = []
    for seed in seeds:
        current = replace(varnn_config, seed=seed, tune=True)
        result = run_varnn_pipeline(
            csv_path,
            replace(config, output_dir=output_dir / "shared_search" / f"seed_{seed}"),
            current,
            validation_only=True,
        )
        search_frames.append(result["validation_search"].copy())
    search = pd.concat(search_frames, ignore_index=True, sort=False)
    signature_columns = ["hidden_units", "activation", "learning_rate"]
    ranking = (
        search.groupby(signature_columns, as_index=False, dropna=False)
        .agg(
            mean_val_level_RMSE=("Val Level RMSE", "mean"),
            std_val_level_RMSE=("Val Level RMSE", "std"),
            mean_val_level_MAE=("Val Level MAE", "mean"),
            mean_val_level_MAPE=("Val Level MAPE", "mean"),
            seed_count=("seed", "nunique"),
        )
    )
    ranking = ranking[ranking["seed_count"].eq(len(seeds))].copy()
    if ranking.empty:
        raise ValueError("No VARNN signature covers every reporting seed.")
    ranking["std_val_level_RMSE"] = ranking["std_val_level_RMSE"].fillna(0.0)
    ranking = ranking.sort_values(
        [
            "mean_val_level_RMSE",
            "mean_val_level_MAE",
            "mean_val_level_MAPE",
            "hidden_units",
            "activation",
            "learning_rate",
        ]
    ).reset_index(drop=True)
    ranking.insert(0, "mean_seed_rank", range(1, len(ranking) + 1))
    ranking["selected"] = ranking["mean_seed_rank"].eq(1)
    ranking["selection_protocol"] = (
        "mean_level_validation_across_reporting_seeds"
    )
    winner = ranking.iloc[0]
    search["selected_shared_mean_seed"] = (
        search["hidden_units"].eq(int(winner["hidden_units"]))
        & search["activation"].eq(str(winner["activation"]))
        & search["learning_rate"].eq(float(winner["learning_rate"]))
    )
    write_csv(search, output_dir / "varnn_shared_validation_search.csv")
    write_csv(ranking, output_dir / "varnn_shared_mean_seed_selection.csv")

    results: list[dict[str, Any]] = []
    metric_frames: list[pd.DataFrame] = []
    forecast_frames: list[pd.DataFrame] = []
    for seed in seeds:
        locked = replace(
            varnn_config,
            seed=seed,
            tune=False,
            hidden_units=int(winner["hidden_units"]),
            activation=str(winner["activation"]),
            learning_rate=float(winner["learning_rate"]),
        )
        result = run_varnn_pipeline(
            csv_path,
            replace(config, output_dir=output_dir / f"seed_{seed}"),
            locked,
        )
        metrics = result["metrics"].copy()
        metrics = metrics[metrics["scale"].eq("level")].copy()
        metrics["pipeline"] = "varnn"
        metrics["seed"] = int(seed)
        forecasts = normalize_forecasts(
            result["forecasts"],
            pipeline="varnn",
            predicted_col="predicted_raw_transformed",
        )
        forecasts["seed"] = int(seed)
        results.append(result)
        metric_frames.append(metrics)
        forecast_frames.append(forecasts)
    return (
        results,
        pd.concat(metric_frames, ignore_index=True, sort=False),
        pd.concat(forecast_frames, ignore_index=True, sort=False),
        ranking,
        len(search) + len(seeds),
    )


def _settings_table(
    rows: list[dict[str, Any]],
    *,
    seeds: tuple[int, ...],
    ffnn_learning_rate_grid: tuple[float, ...],
    varnn_learning_rate_grid: tuple[float, ...],
) -> pd.DataFrame:
    settings = pd.DataFrame(rows)
    settings["reporting_seeds"] = [
        seeds if value else tuple() for value in settings["stochastic"]
    ]
    settings["ffnn_learning_rate_grid"] = [
        ffnn_learning_rate_grid if value == "ffnn" else tuple()
        for value in settings["model_family"]
    ]
    settings["varnn_learning_rate_grid"] = [
        varnn_learning_rate_grid if value == "varnn" else tuple()
        for value in settings["model_family"]
    ]
    settings["ablation_schema_version"] = ABLATION_SCHEMA_VERSION
    return settings.drop(columns=["stochastic", "model_family"])


def run_block_proposed_experiments(
    csv_path: str | Path,
    config: ExperimentConfig,
    varnn_config: VARNNConfig | None = None,
    *,
    seeds: tuple[int, ...] = (7, 42, 123, 2024, 2025),
    include_varnn: bool = True,
) -> dict[str, Any]:
    """Run the shared-selection benchmark and A0--A7 block-proposed study."""
    seeds = tuple(dict.fromkeys(int(seed) for seed in seeds))
    if not seeds:
        raise ValueError("Block-proposed experiments require reporting seeds.")
    root = Path(config.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    dataset_summary, split_summary, stationarity_summary = _dataset_audit(
        csv_path,
        config,
        root,
    )

    metric_frames: list[pd.DataFrame] = []
    forecast_frames: list[pd.DataFrame] = []
    lag_frames: list[pd.DataFrame] = []
    lag_screen_fold_frames: list[pd.DataFrame] = []
    lag_screen_stability_frames: list[pd.DataFrame] = []
    efficiency_rows: list[dict[str, Any]] = []
    settings_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, str]] = []
    variant_results: dict[str, list[dict[str, Any]]] = {}

    def register_setting(
        *,
        pipeline: str,
        model: str,
        uses_vmd: bool,
        target_strategy: str | None,
        exog_strategy: str | None,
        selection_protocol: str | None,
        search_strategy: str | None,
        stochastic: bool,
        model_family: str,
    ) -> None:
        settings_rows.append(
            {
                "pipeline": pipeline,
                "model": model,
                "ablation_id": ABLATION_IDS.get(pipeline),
                "included_in_benchmark": pipeline in BENCHMARK_PIPELINES,
                "uses_vmd": uses_vmd,
                "target_lag_strategy": target_strategy,
                "exog_lag_strategy": exog_strategy,
                "selection_protocol": selection_protocol,
                "search_strategy": search_strategy,
                "stochastic": stochastic,
                "model_family": model_family,
                "enabled": True,
            }
        )

    def register_shared(
        *,
        pipeline: str,
        model: str,
        result: dict[str, Any],
        metrics: pd.DataFrame,
        forecasts: pd.DataFrame,
        variant_config: ExperimentConfig,
        elapsed: float,
        uses_vmd: bool,
        target_strategy: str,
        exog_strategy: str,
        selection_protocol: str,
        search_strategy: str,
    ) -> None:
        variant_results[pipeline] = [result]
        metric_frames.append(metrics)
        forecast_frames.append(forecasts)
        for seed in seeds:
            lag_frames.append(
                _lag_selection_rows(result, pipeline=pipeline, seed=seed)
            )
        folds = result.get("lag_screen_fold_metrics", pd.DataFrame())
        if not folds.empty:
            lag_screen_fold_frames.append(folds.assign(pipeline=pipeline))
        stability = result.get("lag_screen_stability", pd.DataFrame())
        if not stability.empty:
            lag_screen_stability_frames.append(stability.assign(pipeline=pipeline))
        efficiency_rows.append(
            _efficiency_row(
                result,
                variant_config,
                pipeline=pipeline,
                seed=None,
                elapsed_seconds=elapsed,
            )
        )
        register_setting(
            pipeline=pipeline,
            model=model,
            uses_vmd=uses_vmd,
            target_strategy=target_strategy,
            exog_strategy=exog_strategy,
            selection_protocol=selection_protocol,
            search_strategy=search_strategy,
            stochastic=True,
            model_family="ffnn",
        )

    deterministic = (
        (
            "naive_last_observation",
            "Naive_LastObservation",
            run_naive_baseline,
            {"lag": 1, "model_name": "Naive_LastObservation"},
        ),
        (
            "naive_seasonal",
            "Naive_Seasonal12",
            run_naive_baseline,
            {"lag": 12, "model_name": "Naive_Seasonal12"},
        ),
        ("standalone_ardl", "Standalone_ARDL", run_linear_ardl_baseline, {}),
    )
    for pipeline, model, runner, kwargs in deterministic:
        started = perf_counter()
        result = runner(
            csv_path,
            replace(config, output_dir=root / pipeline),
            **kwargs,
        )
        elapsed = perf_counter() - started
        metrics = result["metrics"].copy()
        metrics["pipeline"] = pipeline
        metrics["seed"] = pd.NA
        forecasts = result["forecasts"].copy()
        forecasts["pipeline"] = pipeline
        forecasts["seed"] = pd.NA
        metric_frames.append(metrics)
        forecast_frames.append(forecasts)
        variant_results[pipeline] = [result]
        efficiency_rows.append(
            {
                "pipeline": pipeline,
                "seed": None,
                "runtime_seconds": elapsed,
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
        register_setting(
            pipeline=pipeline,
            model=model,
            uses_vmd=False,
            target_strategy=None,
            exog_strategy=None,
            selection_protocol="deterministic",
            search_strategy=None,
            stochastic=False,
            model_family="deterministic",
        )

    shared_profiles = (
        {
            "pipeline": "ffnn_fixed_no_vmd",
            "model": "FFNN_FixedLag_SharedMeanSeed",
            "uses_vmd": False,
            "target": "fixed",
            "exog": "fixed",
            "screening": "none",
        },
        {
            "pipeline": "vmd_ffnn_fixed",
            "model": "VMD_FFNN_FixedLag_SharedMeanSeed",
            "uses_vmd": True,
            "target": "fixed",
            "exog": "fixed",
            "screening": "none",
        },
        {
            "pipeline": "ardl_ffnn_no_vmd",
            "model": "ARDL_FFNN_NoVMD_SharedMeanSeed",
            "uses_vmd": False,
            "target": "ardl_select_order",
            "exog": "ardl_screen",
            "screening": "rolling_mean_seed",
        },
        {
            "pipeline": SEQUENTIAL_PIPELINE,
            "model": "VMD_ARDL_SequentialScreen_SharedMeanSeed_FFNN",
            "uses_vmd": True,
            "target": "ardl_select_order",
            "exog": "ardl_screen",
            "screening": "rolling_mean_seed",
        },
        {
            "pipeline": JOINT_SELECT_PIPELINE,
            "model": "VMD_ARDL_JointSelect_SharedMeanSeed_FFNN",
            "uses_vmd": True,
            "target": "ardl_select_order_joint",
            # The joint target strategy owns both target and exogenous lags;
            # exog_strategy remains a valid config value but is not consulted.
            "exog": "ardl_screen",
            "reported_exog": "ardl_select_order_joint",
            "screening": "none",
        },
        {
            "pipeline": "a2_vmd_target_fixed_exog",
            "model": "A2_VMD_ARDLTarget_FixedExog_SharedMeanSeed_FFNN",
            "uses_vmd": True,
            "target": "ardl_select_order",
            "exog": "fixed",
            "screening": "none",
        },
        {
            "pipeline": "a3_vmd_aic_all_blocks",
            "model": "A3_VMD_AIC_AllBlocks_SharedMeanSeed_FFNN",
            "uses_vmd": True,
            "target": "ardl_select_order",
            "exog": "ardl_block_subsets",
            "screening": "none",
            "lag_spec_scope": "all_blocks_only",
        },
        {
            "pipeline": "a4_vmd_block_subset_component_mean_seed",
            "model": "A4_VMD_BlockSubset_ComponentMeanSeed_FFNN",
            "uses_vmd": True,
            "target": "ardl_select_order",
            "exog": "ardl_block_subsets",
            "screening": "none",
            "selection_mode": "component_mean_seed",
        },
    )
    for profile in shared_profiles:
        started = perf_counter()
        if profile["uses_vmd"]:
            result, metrics, forecasts, variant_config = _run_shared_vmd_variant(
                csv_path,
                config,
                pipeline=profile["pipeline"],
                model=profile["model"],
                seeds=seeds,
                output_dir=root / profile["pipeline"],
                target_strategy=profile["target"],
                exog_strategy=profile["exog"],
                selection_mode=profile.get(
                    "selection_mode",
                    "reconstructed_mean_seed",
                ),
                lag_spec_scope=profile.get("lag_spec_scope", "all"),
                screening_mode=profile["screening"],
            )
            selection_protocol = (
                "component_mean_validation_across_reporting_seeds_then_reconstruct"
                if profile.get("selection_mode") == "component_mean_seed"
                else "mean_reconstructed_level_validation_across_reporting_seeds"
            )
        else:
            result, metrics, forecasts, variant_config = _run_shared_no_vmd_variant(
                csv_path,
                config,
                pipeline=profile["pipeline"],
                model=profile["model"],
                seeds=seeds,
                output_dir=root / profile["pipeline"],
                target_strategy=profile["target"],
                exog_strategy=profile["exog"],
                screening_mode=profile["screening"],
            )
            selection_protocol = "mean_level_validation_across_reporting_seeds"
        register_shared(
            pipeline=profile["pipeline"],
            model=profile["model"],
            result=result,
            metrics=metrics,
            forecasts=forecasts,
            variant_config=variant_config,
            elapsed=perf_counter() - started,
            uses_vmd=bool(profile["uses_vmd"]),
            target_strategy=profile["target"],
            exog_strategy=profile.get("reported_exog", profile["exog"]),
            selection_protocol=selection_protocol,
            search_strategy=(
                "rolling_mean_seed_then_full_grid"
                if profile["screening"] == "rolling_mean_seed"
                else "full_grid"
            ),
        )

    for pipeline, model, strategy in (
        (
            "a5_vmd_block_subset_shared_full_grid",
            "A5_VMD_BlockSubset_SharedMeanSeed_FullGrid_FFNN",
            "full_grid",
        ),
        (PROPOSED_PIPELINE, PROPOSED_MODEL, "two_stage"),
    ):
        started = perf_counter()
        result, metrics, forecasts, variant_config = _run_shared_block_variant(
            csv_path,
            config,
            pipeline=pipeline,
            model=model,
            seeds=seeds,
            output_dir=root / pipeline,
            search_strategy=strategy,
        )
        register_shared(
            pipeline=pipeline,
            model=model,
            result=result,
            metrics=metrics,
            forecasts=forecasts,
            variant_config=variant_config,
            elapsed=perf_counter() - started,
            uses_vmd=True,
            target_strategy="ardl_select_order",
            exog_strategy="ardl_block_subsets",
            selection_protocol=(
                "mean_reconstructed_level_validation_across_reporting_seeds"
            ),
            search_strategy=strategy,
        )

    a6_pipeline = "a6_vmd_block_subset_per_seed_full_grid"
    a6_model = "A6_VMD_BlockSubset_PerSeed_FullGrid_FFNN"
    a6_results, a6_metrics, a6_forecasts, a6_runs = _run_per_seed_block_variant(
        csv_path,
        config,
        pipeline=a6_pipeline,
        model=a6_model,
        seeds=seeds,
        output_dir=root / a6_pipeline,
    )
    variant_results[a6_pipeline] = a6_results
    metric_frames.append(a6_metrics)
    forecast_frames.append(a6_forecasts)
    for seed, (result, (seed_config, elapsed)) in zip(
        seeds,
        zip(a6_results, a6_runs, strict=True),
        strict=True,
    ):
        lag_frames.append(_lag_selection_rows(result, pipeline=a6_pipeline, seed=seed))
        efficiency_rows.append(
            _efficiency_row(
                result,
                seed_config,
                pipeline=a6_pipeline,
                seed=seed,
                elapsed_seconds=elapsed,
            )
        )
    register_setting(
        pipeline=a6_pipeline,
        model=a6_model,
        uses_vmd=True,
        target_strategy="ardl_select_order",
        exog_strategy="ardl_block_subsets",
        selection_protocol="reconstructed_level_validation_per_seed",
        search_strategy="full_grid",
        stochastic=True,
        model_family="ffnn",
    )

    if include_varnn:
        started = perf_counter()
        try:
            (
                varnn_results,
                varnn_metrics,
                varnn_forecasts,
                varnn_selection,
                varnn_fits,
            ) = _run_varnn_shared_mean_seed(
                csv_path,
                config,
                varnn_config or VARNNConfig(),
                seeds=seeds,
                output_dir=root / "varnn",
            )
        except ImportError as exc:
            skipped_rows.append({"pipeline": "varnn", "reason": str(exc)})
            register_setting(
                pipeline="varnn",
                model="VARNN_SharedMeanSeed",
                uses_vmd=False,
                target_strategy="var_aic",
                exog_strategy="var_lag",
                selection_protocol="mean_level_validation_across_reporting_seeds",
                search_strategy="shared_grid",
                stochastic=True,
                model_family="varnn",
            )
            settings_rows[-1]["enabled"] = False
        else:
            variant_results["varnn"] = varnn_results
            metric_frames.append(varnn_metrics)
            forecast_frames.append(varnn_forecasts)
            efficiency_rows.append(
                {
                    "pipeline": "varnn",
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
                    "evaluated_ffnn_fits": varnn_fits,
                }
            )
            register_setting(
                pipeline="varnn",
                model="VARNN_SharedMeanSeed",
                uses_vmd=False,
                target_strategy="var_aic",
                exog_strategy="var_lag",
                selection_protocol="mean_level_validation_across_reporting_seeds",
                search_strategy="shared_grid",
                stochastic=True,
                model_family="varnn",
            )
            write_csv(varnn_selection, root / "varnn" / "shared_selection.csv")
    else:
        skipped_rows.append({"pipeline": "varnn", "reason": "disabled_by_cli"})
        register_setting(
            pipeline="varnn",
            model="VARNN_SharedMeanSeed",
            uses_vmd=False,
            target_strategy="var_aic",
            exog_strategy="var_lag",
            selection_protocol="mean_level_validation_across_reporting_seeds",
            search_strategy="shared_grid",
            stochastic=True,
            model_family="varnn",
        )
        settings_rows[-1]["enabled"] = False

    seed_metrics = pd.concat(metric_frames, ignore_index=True, sort=False)
    forecasts = pd.concat(forecast_frames, ignore_index=True, sort=False)
    aggregate_metrics = _aggregate_seed_metrics(seed_metrics)
    active_benchmarks = tuple(
        pipeline
        for pipeline in BENCHMARK_PIPELINES
        if pipeline in set(forecasts["pipeline"])
    )
    benchmark_metrics = aggregate_metrics[
        aggregate_metrics["pipeline"].isin(active_benchmarks)
    ].copy()
    benchmark_forecasts = forecasts[
        forecasts["pipeline"].isin(active_benchmarks)
    ].copy()
    date_coverage = _forecast_date_coverage(benchmark_forecasts)
    ablation_metrics = aggregate_metrics[
        aggregate_metrics["pipeline"].isin(ABLATION_IDS)
    ].copy()
    ablation_metrics.insert(
        0,
        "ablation_id",
        ablation_metrics["pipeline"].map(ABLATION_IDS),
    )
    ablation_metrics.insert(1, "ablation_schema_version", ABLATION_SCHEMA_VERSION)
    ablation_forecasts = forecasts[forecasts["pipeline"].isin(ABLATION_IDS)].copy()
    ablation_forecasts.insert(
        0,
        "ablation_id",
        ablation_forecasts["pipeline"].map(ABLATION_IDS),
    )
    dm_tests = pairwise_dm_tests(
        benchmark_forecasts,
        split="test",
        loss="squared",
        horizon=1,
    )
    lag_selection = (
        pd.concat(
            [frame for frame in lag_frames if not frame.empty],
            ignore_index=True,
            sort=False,
        )
        if any(not frame.empty for frame in lag_frames)
        else pd.DataFrame()
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
    efficiency = pd.DataFrame(efficiency_rows)
    settings = _settings_table(
        settings_rows,
        seeds=seeds,
        ffnn_learning_rate_grid=tuple(config.ffnn.learning_rate_grid),
        varnn_learning_rate_grid=tuple(
            (varnn_config or VARNNConfig()).learning_rate_grid
        ),
    )
    skipped = pd.DataFrame(skipped_rows)

    write_csv(seed_metrics, root / "paper_seed_metrics.csv")
    write_csv(dataset_summary, root / "paper_dataset_summary.csv")
    write_csv(split_summary, root / "paper_split_summary.csv")
    write_csv(stationarity_summary, root / "paper_stationarity_screen.csv")
    write_csv(date_coverage, root / "paper_forecast_date_coverage.csv")
    write_csv(aggregate_metrics, root / "paper_metrics_summary.csv")
    write_csv(forecasts, root / "paper_forecasts.csv")
    write_csv(benchmark_metrics, root / "benchmark_metrics_comparison.csv")
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

    proposed_result = variant_results[PROPOSED_PIPELINE][0]
    write_paper_figures(
        benchmark_forecasts,
        benchmark_metrics,
        efficiency,
        lag_frequency,
        root,
        training_components=proposed_result.get("training_components"),
        target=config.data.target,
        proposed_pipeline=PROPOSED_PIPELINE,
    )

    return {
        "comparison_metrics": benchmark_metrics,
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
        "proposed": variant_results[PROPOSED_PIPELINE],
        "varnn": variant_results.get("varnn", []),
    }
