from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import numpy as np
import pandas as pd

from .artifacts import write_csv
from .benchmark_common import (
    aggregate_seed_metrics,
    dataset_audit,
    efficiency_row,
    ensemble_prediction_metrics,
    forecast_date_coverage,
    lag_selection_rows,
)
from .block_proposed_experiments import (
    JOINT_SELECT_PIPELINE,
    PROPOSED_MODEL,
    PROPOSED_PIPELINE,
    SEQUENTIAL_PIPELINE,
    run_shared_block_variant,
    run_shared_no_vmd_variant,
    run_shared_vmd_variant,
    run_varnn_shared_mean_seed,
)
from .config import ExperimentConfig
from .lstm_shared_mean import LSTM_MODEL, LSTM_PIPELINE, run_lstm_shared_mean_seed
from .models.lstm import LSTMConfig
from .models.varnn import VARNNConfig
from .paper_baselines import run_naive_baseline
from .paper_reporting import lag_selection_frequency, write_paper_figures
from .statistical_tests import pairwise_dm_tests


NO_VMD_PIPELINE = "ardl_ffnn_no_vmd"
NO_VMD_MODEL = "ARDL_FFNN_NoVMD_SharedMeanSeed"
SEQUENTIAL_MODEL = "VMD_ARDL_SequentialScreen_SharedMeanSeed_FFNN"
JOINT_SELECT_MODEL = "VMD_ARDL_JointSelect_SharedMeanSeed_FFNN"
VARNN_PIPELINE = "varnn"
NAIVE_PIPELINE = "naive_last_observation"

SHARED_MEAN_BENCHMARK_PIPELINES = (
    NO_VMD_PIPELINE,
    PROPOSED_PIPELINE,
    SEQUENTIAL_PIPELINE,
    JOINT_SELECT_PIPELINE,
    VARNN_PIPELINE,
    LSTM_PIPELINE,
)


def _require_complete_coverage(
    forecasts: pd.DataFrame,
    seeds: tuple[int, ...],
    stochastic_pipelines: tuple[str, ...],
) -> None:
    """Require each model/seed to predict every date covered by naïve."""
    frame = forecasts.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    for split in ("validation", "test"):
        part = frame.loc[frame["split"].eq(split)]
        baseline = part.loc[part["pipeline"].eq(NAIVE_PIPELINE)]
        expected = set(baseline["date"])
        if not expected or len(baseline) != len(expected):
            raise ValueError(f"Naïve forecast coverage is incomplete for {split}.")
        for pipeline in stochastic_pipelines:
            for seed in seeds:
                member = part.loc[
                    part["pipeline"].eq(pipeline) & part["seed"].eq(seed)
                ]
                actual = set(member["date"])
                if actual != expected or len(member) != len(expected):
                    missing = sorted(expected - actual)
                    extra = sorted(actual - expected)
                    raise ValueError(
                        f"{pipeline} seed {seed} {split} forecast dates differ: "
                        f"missing={[day.strftime('%Y-%m-%d') for day in missing]}, "
                        f"extra={[day.strftime('%Y-%m-%d') for day in extra]}."
                    )
        reference = part["date"].map(baseline.set_index("date")["actual"])
        actual = pd.to_numeric(part["actual"], errors="coerce").to_numpy(float)
        expected_actual = pd.to_numeric(reference, errors="coerce").to_numpy(float)
        if (
            not np.isfinite(actual).all()
            or not np.isfinite(expected_actual).all()
            or not np.isclose(actual, expected_actual, rtol=1e-10, atol=1e-12).all()
        ):
            raise ValueError(f"Actual values differ across models for {split}.")


def run_shared_mean_benchmark_experiments(
    csv_path: str | Path,
    config: ExperimentConfig,
    *,
    seeds: tuple[int, ...] = (7, 42, 123, 2024, 2025),
    varnn_config: VARNNConfig | None = None,
    lstm_config: LSTMConfig | None = None,
    include_lstm: bool = True,
) -> dict[str, Any]:
    """Benchmark shared-mean stochastic workflows and a naïve reference."""
    seeds = tuple(dict.fromkeys(int(seed) for seed in seeds))
    if not seeds:
        raise ValueError("Shared-mean benchmark requires reporting seeds.")

    root = Path(config.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    dataset_summary, split_summary, stationarity_summary = dataset_audit(
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
                lag_selection_rows(result, pipeline=pipeline, seed=seed)
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
            efficiency_row(
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
            run_shared_no_vmd_variant,
        ),
        (
            SEQUENTIAL_PIPELINE,
            SEQUENTIAL_MODEL,
            True,
            "ardl_select_order",
            "ardl_screen",
            "rolling_mean_seed",
            "mean_reconstructed_level_validation_across_reporting_seeds",
            run_shared_vmd_variant,
        ),
        (
            JOINT_SELECT_PIPELINE,
            JOINT_SELECT_MODEL,
            True,
            "ardl_select_order_joint",
            "ardl_select_order_joint",
            "none",
            "mean_reconstructed_level_validation_across_reporting_seeds",
            run_shared_vmd_variant,
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
        run_shared_block_variant(
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

    started = perf_counter()
    varnn_results, varnn_metrics, varnn_forecasts, varnn_selection, varnn_fits = (
        run_varnn_shared_mean_seed(
            csv_path,
            config,
            varnn_config or VARNNConfig(),
            seeds=seeds,
            output_dir=root / VARNN_PIPELINE,
        )
    )
    variant_results[VARNN_PIPELINE] = varnn_results
    metric_frames.append(varnn_metrics)
    forecast_frames.append(varnn_forecasts)
    write_csv(varnn_selection, root / VARNN_PIPELINE / "shared_selection.csv")
    efficiency_rows.append(
        {
            "pipeline": VARNN_PIPELINE,
            "seed": None,
            "runtime_seconds": perf_counter() - started,
            "component_count": 1,
            "evaluated_ffnn_fits": varnn_fits,
        }
    )
    setting_rows.append(
        {
            "pipeline": VARNN_PIPELINE,
            "model": "VARNN_SharedMeanSeed",
            "included_in_benchmark": True,
            "uses_vmd": False,
            "target_lag_strategy": "var_aic",
            "exog_lag_strategy": "var_lag",
            "selection_protocol": "mean_level_validation_across_reporting_seeds",
            "selection_seed_aggregation": "shared_mean_validation_across_reporting_seeds",
            "shared_configuration_across_seeds": True,
            "search_strategy": "shared_grid",
            "fast_screen_validation_scope": "not_applicable",
            "reporting_seeds": seeds,
            "varnn_learning_rate_grid": tuple((varnn_config or VARNNConfig()).learning_rate_grid),
            "final_selection_scale": "level",
            "enabled": True,
        }
    )

    if include_lstm:
        selected_lstm_config = lstm_config or LSTMConfig()
        started = perf_counter()
        lstm_results, lstm_metrics, lstm_forecasts, lstm_selection, lstm_fits = (
            run_lstm_shared_mean_seed(
                csv_path,
                config,
                selected_lstm_config,
                seeds=seeds,
                output_dir=root / LSTM_PIPELINE,
            )
        )
        variant_results[LSTM_PIPELINE] = lstm_results
        metric_frames.append(lstm_metrics)
        forecast_frames.append(lstm_forecasts)
        write_csv(lstm_selection, root / LSTM_PIPELINE / "shared_selection.csv")
        winner = lstm_selection.loc[lstm_selection["selected"]].iloc[0]
        efficiency_rows.append({
            "pipeline": LSTM_PIPELINE,
            "seed": None,
            "runtime_seconds": perf_counter() - started,
            "component_count": 1,
            "evaluated_ffnn_fits": lstm_fits,
            "evaluated_lstm_fits": lstm_fits,
        })
        setting_rows.append({
            "pipeline": LSTM_PIPELINE,
            "model": LSTM_MODEL,
            "included_in_benchmark": True,
            "uses_vmd": False,
            "target_lag_strategy": "lstm_lookback_validation",
            "exog_lag_strategy": "multivariate_history_through_t_minus_1",
            "selection_protocol": "mean_level_validation_across_reporting_seeds",
            "selection_seed_aggregation": "shared_mean_validation_across_reporting_seeds",
            "shared_configuration_across_seeds": True,
            "search_strategy": "shared_grid",
            "fast_screen_validation_scope": "not_applicable",
            "reporting_seeds": seeds,
            "lstm_lookback_grid": selected_lstm_config.lookback_grid,
            "lstm_hidden_units_grid": selected_lstm_config.hidden_units_grid,
            "lstm_learning_rate_grid": selected_lstm_config.learning_rate_grid,
            "lstm_epochs": selected_lstm_config.epochs,
            "lstm_selected_lookback": int(winner["lookback"]),
            "lstm_selected_hidden_units": int(winner["hidden_units"]),
            "lstm_selected_learning_rate": float(winner["learning_rate"]),
            "final_selection_scale": "level",
            "enabled": True,
        })

    started = perf_counter()
    naive_result = run_naive_baseline(
        csv_path,
        replace(config, output_dir=root / NAIVE_PIPELINE),
        lag=1,
        model_name="Naive_LastObservation",
    )
    naive_metrics = naive_result["metrics"].copy()
    naive_forecasts = naive_result["forecasts"].copy()
    naive_metrics["pipeline"] = NAIVE_PIPELINE
    naive_forecasts["pipeline"] = NAIVE_PIPELINE
    naive_metrics["seed"] = pd.NA
    naive_forecasts["seed"] = pd.NA
    metric_frames.append(naive_metrics)
    forecast_frames.append(naive_forecasts)
    variant_results[NAIVE_PIPELINE] = [naive_result]
    efficiency_rows.append(
        {"pipeline": NAIVE_PIPELINE, "seed": None,
         "runtime_seconds": perf_counter() - started,
         "component_count": 0, "evaluated_ffnn_fits": 0}
    )
    setting_rows.append(
        {"pipeline": NAIVE_PIPELINE, "model": "Naive_LastObservation",
         "included_in_benchmark": True, "uses_vmd": False,
         "target_lag_strategy": "last_observation", "exog_lag_strategy": "none",
         "selection_protocol": "none", "selection_seed_aggregation": None,
         "shared_configuration_across_seeds": False, "search_strategy": "none",
         "fast_screen_validation_scope": "not_applicable",
         "reporting_seeds": tuple(), "final_selection_scale": "level",
         "enabled": True}
    )

    seed_metrics = pd.concat(metric_frames, ignore_index=True, sort=False)
    forecasts = pd.concat(forecast_frames, ignore_index=True, sort=False)
    active_pipelines = set(forecasts["pipeline"].unique())
    stochastic_pipelines = tuple(
        pipeline for pipeline in SHARED_MEAN_BENCHMARK_PIPELINES
        if include_lstm or pipeline != LSTM_PIPELINE
    )
    expected_pipelines = set(stochastic_pipelines) | {NAIVE_PIPELINE}
    if active_pipelines != expected_pipelines:
        raise ValueError(
            "Shared-mean benchmark model set mismatch: "
            f"expected={sorted(expected_pipelines)}, "
            f"actual={sorted(active_pipelines)}."
        )
    expected_seeds = set(seeds)
    for pipeline in stochastic_pipelines:
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
    _require_complete_coverage(forecasts, seeds, stochastic_pipelines)

    aggregate_metrics = aggregate_seed_metrics(seed_metrics)
    benchmark_metrics = aggregate_metrics[
        aggregate_metrics["pipeline"].isin(expected_pipelines)
    ].copy()
    benchmark_forecasts = forecasts[
        forecasts["pipeline"].isin(expected_pipelines)
    ].copy()
    ensemble_metrics = ensemble_prediction_metrics(
        benchmark_forecasts,
        seed_metrics,
    )
    date_coverage = forecast_date_coverage(benchmark_forecasts)
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
        "varnn": variant_results[VARNN_PIPELINE],
        "lstm": variant_results.get(LSTM_PIPELINE, []),
        "naive": variant_results[NAIVE_PIPELINE],
    }
