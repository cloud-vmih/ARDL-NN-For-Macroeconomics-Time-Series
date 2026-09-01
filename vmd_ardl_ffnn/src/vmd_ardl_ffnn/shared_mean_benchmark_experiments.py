from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import pandas as pd

from .artifacts import write_csv
from .block_proposed_experiments import (
    JOINT_SELECT_PIPELINE,
    PROPOSED_MODEL,
    PROPOSED_PIPELINE,
    SEQUENTIAL_PIPELINE,
    _run_shared_block_variant,
    _run_shared_no_vmd_variant,
    _run_shared_vmd_variant,
)
from .config import ExperimentConfig
from .paper_experiments import (
    _aggregate_seed_metrics,
    _dataset_audit,
    _efficiency_row,
    _ensemble_prediction_metrics,
    _forecast_date_coverage,
    _lag_selection_rows,
)
from .paper_reporting import lag_selection_frequency, write_paper_figures
from .statistical_tests import pairwise_dm_tests


NO_VMD_PIPELINE = "ardl_ffnn_no_vmd"
NO_VMD_MODEL = "ARDL_FFNN_NoVMD_SharedMeanSeed"
SEQUENTIAL_MODEL = "VMD_ARDL_SequentialScreen_SharedMeanSeed_FFNN"
JOINT_SELECT_MODEL = "VMD_ARDL_JointSelect_SharedMeanSeed_FFNN"

SHARED_MEAN_BENCHMARK_PIPELINES = (
    NO_VMD_PIPELINE,
    PROPOSED_PIPELINE,
    SEQUENTIAL_PIPELINE,
    JOINT_SELECT_PIPELINE,
)


def run_shared_mean_benchmark_experiments(
    csv_path: str | Path,
    config: ExperimentConfig,
    *,
    seeds: tuple[int, ...] = (7, 42, 123, 2024, 2025),
) -> dict[str, Any]:
    """Benchmark the four requested workflows under one shared-mean protocol."""
    seeds = tuple(dict.fromkeys(int(seed) for seed in seeds))
    if not seeds:
        raise ValueError("Shared-mean benchmark requires reporting seeds.")

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
    setting_rows: list[dict[str, Any]] = []
    variant_results: dict[str, list[dict[str, Any]]] = {}

    def register(
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
        fast_screen_validation_scope: str,
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
            lag_screen_stability_frames.append(
                stability.assign(pipeline=pipeline)
            )
        efficiency_rows.append(
            _efficiency_row(
                result,
                variant_config,
                pipeline=pipeline,
                seed=None,
                elapsed_seconds=elapsed,
            )
        )
        setting_rows.append(
            {
                "pipeline": pipeline,
                "model": model,
                "included_in_benchmark": True,
                "uses_vmd": uses_vmd,
                "target_lag_strategy": target_strategy,
                "exog_lag_strategy": exog_strategy,
                "selection_protocol": selection_protocol,
                "selection_seed_aggregation": (
                    "shared_mean_validation_across_reporting_seeds"
                ),
                "shared_configuration_across_seeds": True,
                "search_strategy": search_strategy,
                "fast_screen_validation_scope": fast_screen_validation_scope,
                "reporting_seeds": seeds,
                "ffnn_learning_rate_grid": tuple(
                    variant_config.ffnn.learning_rate_grid
                ),
                "final_selection_scale": (
                    "reconstructed_level" if uses_vmd else "level"
                ),
                "enabled": True,
            }
        )

    shared_runners: tuple[
        tuple[
            str,
            str,
            bool,
            str,
            str,
            str,
            str,
            Callable[..., tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, ExperimentConfig]],
        ],
        ...,
    ] = (
        (
            NO_VMD_PIPELINE,
            NO_VMD_MODEL,
            False,
            "ardl_select_order",
            "ardl_screen",
            "rolling_mean_seed",
            "mean_level_validation_across_reporting_seeds",
            _run_shared_no_vmd_variant,
        ),
        (
            SEQUENTIAL_PIPELINE,
            SEQUENTIAL_MODEL,
            True,
            "ardl_select_order",
            "ardl_screen",
            "rolling_mean_seed",
            "mean_reconstructed_level_validation_across_reporting_seeds",
            _run_shared_vmd_variant,
        ),
        (
            JOINT_SELECT_PIPELINE,
            JOINT_SELECT_MODEL,
            True,
            "ardl_select_order_joint",
            "ardl_select_order_joint",
            "none",
            "mean_reconstructed_level_validation_across_reporting_seeds",
            _run_shared_vmd_variant,
        ),
    )

    for (
        pipeline,
        model,
        uses_vmd,
        target_strategy,
        reported_exog_strategy,
        screening_mode,
        selection_protocol,
        runner,
    ) in shared_runners:
        started = perf_counter()
        runner_exog_strategy = (
            "ardl_screen"
            if target_strategy == "ardl_select_order_joint"
            else reported_exog_strategy
        )
        result, metrics, forecasts, variant_config = runner(
            csv_path,
            config,
            pipeline=pipeline,
            model=model,
            seeds=seeds,
            output_dir=root / pipeline,
            target_strategy=target_strategy,
            exog_strategy=runner_exog_strategy,
            screening_mode=screening_mode,
        )
        register(
            pipeline=pipeline,
            model=model,
            result=result,
            metrics=metrics,
            forecasts=forecasts,
            variant_config=variant_config,
            elapsed=perf_counter() - started,
            uses_vmd=uses_vmd,
            target_strategy=target_strategy,
            exog_strategy=reported_exog_strategy,
            selection_protocol=selection_protocol,
            search_strategy=(
                "rolling_mean_seed_then_full_grid"
                if screening_mode == "rolling_mean_seed"
                else "full_grid"
            ),
            fast_screen_validation_scope=(
                "train_inner_rolling"
                if screening_mode == "rolling_mean_seed"
                else "not_applicable"
            ),
        )

    started = perf_counter()
    proposed_result, proposed_metrics, proposed_forecasts, proposed_config = (
        _run_shared_block_variant(
            csv_path,
            config,
            pipeline=PROPOSED_PIPELINE,
            model=PROPOSED_MODEL,
            seeds=seeds,
            output_dir=root / PROPOSED_PIPELINE,
            search_strategy="two_stage",
            fast_screen_validation_scope="train_inner_rolling",
        )
    )
    register(
        pipeline=PROPOSED_PIPELINE,
        model=PROPOSED_MODEL,
        result=proposed_result,
        metrics=proposed_metrics,
        forecasts=proposed_forecasts,
        variant_config=proposed_config,
        elapsed=perf_counter() - started,
        uses_vmd=True,
        target_strategy="ardl_select_order",
        exog_strategy="ardl_block_subsets",
        selection_protocol=(
            "mean_reconstructed_level_validation_across_reporting_seeds"
        ),
        search_strategy="two_stage",
        fast_screen_validation_scope="train_inner_rolling",
    )

    seed_metrics = pd.concat(metric_frames, ignore_index=True, sort=False)
    forecasts = pd.concat(forecast_frames, ignore_index=True, sort=False)
    active_pipelines = set(forecasts["pipeline"].unique())
    expected_pipelines = set(SHARED_MEAN_BENCHMARK_PIPELINES)
    if active_pipelines != expected_pipelines:
        raise ValueError(
            "Shared-mean benchmark model set mismatch: "
            f"expected={sorted(expected_pipelines)}, "
            f"actual={sorted(active_pipelines)}."
        )
    expected_seeds = set(seeds)
    for pipeline in SHARED_MEAN_BENCHMARK_PIPELINES:
        metric_seeds = set(
            seed_metrics.loc[
                seed_metrics["pipeline"].eq(pipeline),
                "seed",
            ].astype(int)
        )
        forecast_seeds = set(
            forecasts.loc[
                forecasts["pipeline"].eq(pipeline),
                "seed",
            ].astype(int)
        )
        if metric_seeds != expected_seeds or forecast_seeds != expected_seeds:
            raise ValueError(
                f"Incomplete reporting-seed coverage for {pipeline}: "
                f"expected={sorted(expected_seeds)}, "
                f"metrics={sorted(metric_seeds)}, "
                f"forecasts={sorted(forecast_seeds)}."
            )

    aggregate_metrics = _aggregate_seed_metrics(seed_metrics)
    benchmark_metrics = aggregate_metrics[
        aggregate_metrics["pipeline"].isin(SHARED_MEAN_BENCHMARK_PIPELINES)
    ].copy()
    benchmark_forecasts = forecasts[
        forecasts["pipeline"].isin(SHARED_MEAN_BENCHMARK_PIPELINES)
    ].copy()
    ensemble_metrics = _ensemble_prediction_metrics(
        benchmark_forecasts,
        seed_metrics,
    )
    date_coverage = _forecast_date_coverage(benchmark_forecasts)
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
    settings = pd.DataFrame(setting_rows)

    write_csv(seed_metrics, root / "paper_seed_metrics.csv")
    write_csv(dataset_summary, root / "paper_dataset_summary.csv")
    write_csv(split_summary, root / "paper_split_summary.csv")
    write_csv(stationarity_summary, root / "paper_stationarity_screen.csv")
    write_csv(date_coverage, root / "paper_forecast_date_coverage.csv")
    write_csv(aggregate_metrics, root / "paper_metrics_summary.csv")
    write_csv(ensemble_metrics, root / "paper_ensemble_metrics_summary.csv")
    write_csv(forecasts, root / "paper_forecasts.csv")
    write_csv(benchmark_metrics, root / "benchmark_metrics_comparison.csv")
    write_csv(
        benchmark_metrics,
        root / "benchmark_seed_robustness_metrics.csv",
    )
    write_csv(
        ensemble_metrics,
        root / "benchmark_ensemble_metrics_comparison.csv",
    )
    write_csv(
        benchmark_forecasts,
        root / "benchmark_forecasts_comparison.csv",
    )
    write_csv(dm_tests, root / "diebold_mariano_tests.csv")
    write_csv(lag_selection, root / "lag_selection_summary.csv")
    write_csv(lag_frequency, root / "lag_selection_frequency.csv")
    if not lag_screen_folds.empty:
        write_csv(lag_screen_folds, root / "lag_screen_fold_metrics.csv")
    if not lag_screen_stability.empty:
        write_csv(lag_screen_stability, root / "lag_screen_stability.csv")
    write_csv(efficiency, root / "efficiency_summary.csv")
    write_csv(settings, root / "paper_experiment_settings.csv")

    write_paper_figures(
        benchmark_forecasts,
        benchmark_metrics,
        efficiency,
        lag_frequency,
        root,
        seed_robustness_metrics=benchmark_metrics,
        training_components=proposed_result.get("training_components"),
        target=config.data.target,
        proposed_pipeline=PROPOSED_PIPELINE,
    )

    return {
        "comparison_metrics": benchmark_metrics,
        "ensemble_comparison_metrics": ensemble_metrics,
        "seed_robustness_metrics": benchmark_metrics,
        "comparison_forecasts": benchmark_forecasts,
        "seed_metrics": seed_metrics,
        "aggregate_metrics": aggregate_metrics,
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
        "variant_results": variant_results,
        "no_vmd": variant_results[NO_VMD_PIPELINE],
        "proposed": variant_results[PROPOSED_PIPELINE],
        "sequential_proposed": variant_results[SEQUENTIAL_PIPELINE],
        "joint_select": variant_results[JOINT_SELECT_PIPELINE],
    }
