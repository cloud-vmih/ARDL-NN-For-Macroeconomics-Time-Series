#!/usr/bin/env python3
"""Single entrypoint for every vmd_ardl_ffnn experiment pipeline."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys
import traceback


PROJECT_ROOT = Path(__file__).resolve().parent
PACKAGE_SRC = PROJECT_ROOT / "vmd_ardl_ffnn" / "src"
if str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))

from vmd_ardl_ffnn.artifacts import prepare_run_directory, write_manifest
from vmd_ardl_ffnn.config import (
    ARDLSelectionConfig,
    DataConfig,
    DiagnosticsConfig,
    ExperimentConfig,
    FFNNConfig,
    StationarityConfig,
    TensorFlowRuntimeConfig,
    VMDConfig,
)
from vmd_ardl_ffnn.experiment import VMDARDLFFNNExperiment
from vmd_ardl_ffnn.models.varnn import VARNNConfig, run_varnn_pipeline
from vmd_ardl_ffnn.macro_benchmark import (
    CORE_PIPELINES,
    FULL_ONLY_PIPELINES,
    NOTEBOOK_SHA256,
    NOTEBOOK_URL,
    build_macro_experiment_config,
    file_sha256,
    load_macro_dataset,
    pipelines_for_profile,
    run_macro_benchmark,
    screen_macro_features,
)
from vmd_ardl_ffnn.models.tensorflow_runtime import bootstrap_tensorflow


PIPELINE_MODE = "vmd-ardl-select-order"
PIPELINE_CHOICES = (
    "vmd",
    "no-vmd",
    "both",
    "varnn",
    "model-comparison",
    "benchmark",
    "block-proposed-benchmark",
    "shared-mean-benchmark",
    "sensitivity",
    "vmd-ardl-select-order",
    "vmd-ardl-joint-select",
    "vmd-ardl-hard-ffnn",
    "sequential-vmd-ardl-ffnn",
    "macro-benchmark",
)


def strategy_for_pipeline(pipeline: str) -> str:
    """Return the target-lag policy owned by a standalone pipeline profile."""
    if pipeline == "vmd-ardl-joint-select":
        return "ardl_select_order_joint"
    return "ardl_select_order"


def build_experiment_config(pipeline: str, output_dir: Path) -> ExperimentConfig:
    """Build one validated, independent experiment profile."""
    return ExperimentConfig(
        data=DataConfig(
            date_col="TIME_PERIOD",
            target="Export_US",
            features=("PCE", "US_Retail", "USD_VND", "Import_CN"),
            freq="MS",
            log_transform=True,
            missing_period_policy="raise",
            missing_value_policy="raise",
        ),
        vmd=VMDConfig(
            modes=3,
            backend="sktime",
            alpha=2000.0,
            tau=0.0,
            dc=False,
            init=1,
            tolerance=1e-7,
            max_iterations=500,
        ),
        ardl=ARDLSelectionConfig(
            target_lag_strategy=strategy_for_pipeline(pipeline),
            lag_spec_strategy=(
                "full_product"
                if pipeline == "vmd-ardl-hard-ffnn"
                else "staged"
            ),
            max_target_lag=12,
            max_exog_lag=6,
            top_n=3,
            ic="aic",
            trend="c",
            causal=True,
            fixed_target_lags=(1, 12),
            target_lag_top_n=2,
            force_target_lag_1=True,
            target_lag_preset="acf_pacf",
            target_lag_candidates=tuple(),
            target_lag_acf_pacf_top_n=3,
            target_lag_acf_pacf_max_lags_per_set=3,
            max_lag_specs=16,
            selection_failure_policy="raise",
        ),
        ffnn=FFNNConfig(
            hidden_layer_candidates=(1, 2, 3),
            hidden_width_multipliers=(1.0, 0.5, 0.25),
            hidden_units_candidates=tuple(),
            alpha_grid=(1e-4, 1e-3, 1e-2, 1e-1),
            seed_grid=(7, 42, 123),
            activation="tanh",
            activation_grid=("tanh",),
            learning_rate_init=0.001,
            learning_rate_grid=(
                (0.01, 0.001)
                if pipeline
                in {"block-proposed-benchmark", "shared-mean-benchmark"}
                else (0.001,)
            ),
            max_iter=500,
            min_train=30,
            min_val=8,
            min_test=8,
            search_strategy="staged_halving",
            fast_hidden_units=None,
            fast_hidden_layers=1,
            fast_hidden_width_multiplier=1.0,
            fast_alpha=1e-3,
            fast_max_iter=150,
            top_k_lag_specs=4,
            top_component_candidates=3,
            lag_screen_folds=3,
            lag_screen_validation_size=12,
            lag_screen_learning_rate_grid=(0.01, 0.001),
            lag_screen_selection_rule="one_standard_error",
            final_selection_scale="level",
        ),
        diagnostics=DiagnosticsConfig(lags=12, granger_maxlag=2, alpha=0.05),
        stationarity=StationarityConfig(enabled=True, alpha=0.05, min_obs=12),
        train_ratio=0.70,
        val_ratio=0.15,
        output_dir=output_dir,
        forecast_protocol="rolling_one_step",
    )


def build_varnn_config() -> VARNNConfig:
    return VARNNConfig(
        lag_criterion="aic",
        maxlags=None,
        min_lag=1,
        max_lag=6,
        fixed_lag=None,
        hidden_units=32,
        activation="sigmoid",
        learning_rate=1e-3,
        tune=True,
        hidden_units_grid=(32,),
        activation_grid=("sigmoid",),
        learning_rate_grid=(0.01, 0.001),
        epochs=100,
        batch_size=32,
        patience=12,
        seed=7,
    )


def run_pipeline(
    pipeline: str,
    csv_path: Path,
    config: ExperimentConfig,
    varnn_config: VARNNConfig,
    *,
    paper_seeds: tuple[int, ...] = (7, 42, 123, 2024, 2025),
    include_varnn: bool = True,
    block_subset_search_strategy: str = "full_grid",
):
    experiment = VMDARDLFFNNExperiment(config)
    if pipeline == "vmd":
        return experiment.run(csv_path), "final_metrics"
    if pipeline == "no-vmd":
        return experiment.run_without_vmd(csv_path), "final_metrics"
    if pipeline == "both":
        return experiment.run_comparison(csv_path), "comparison_metrics"
    if pipeline == "varnn":
        return run_varnn_pipeline(csv_path, config, varnn_config), "metrics"
    if pipeline == "model-comparison":
        return experiment.run_model_comparison(csv_path, varnn_config), "comparison_metrics"
    if pipeline == "benchmark":
        return (
            experiment.run_benchmark(
                csv_path,
                varnn_config,
                seeds=paper_seeds,
                include_varnn=include_varnn,
                block_subset_search_strategy=block_subset_search_strategy,
            ),
            "comparison_metrics",
        )
    if pipeline == "block-proposed-benchmark":
        return (
            experiment.run_block_proposed_benchmark(
                csv_path,
                varnn_config,
                seeds=paper_seeds,
                include_varnn=include_varnn,
            ),
            "comparison_metrics",
        )
    if pipeline == "shared-mean-benchmark":
        return (
            experiment.run_shared_mean_benchmark(
                csv_path,
                seeds=paper_seeds,
            ),
            "comparison_metrics",
        )
    if pipeline == "vmd-ardl-hard-ffnn":
        hard_config = replace(
            config,
            ardl=replace(
                config.ardl,
                ic="aic",
                target_lag_strategy="ardl_select_order",
                exog_lag_strategy="ardl_screen",
                lag_spec_strategy="full_product",
                top_n=3,
            ),
            ffnn=replace(config.ffnn, seed_grid=paper_seeds),
        )
        return (
            VMDARDLFFNNExperiment(
                hard_config
            ).run_hard_ffnn_comparator(
                csv_path,
                reporting_seeds=paper_seeds,
            ),
            "final_metrics",
        )
    if pipeline == "sensitivity":
        from vmd_ardl_ffnn.sensitivity_experiments import run_sensitivity_experiments

        return (
            run_sensitivity_experiments(csv_path, config, seeds=paper_seeds),
            "metrics",
        )
    if pipeline in {
        "vmd-ardl-select-order",
        "vmd-ardl-joint-select",
        "sequential-vmd-ardl-ffnn",
    }:
        return experiment.run(csv_path), "final_metrics"
    raise ValueError(f"Unsupported pipeline: {pipeline}")


def print_metrics_summary(
    results,
    metric_key: str,
    *,
    title: str = "Metrics summary",
) -> None:
    if metric_key not in results:
        return
    frame = results[metric_key]
    columns = [
        col
        for col in (
            "model",
            "seed",
            "split",
            "scale",
            "RMSE",
            "MAE",
            "MAPE",
            "RMSE_mean",
            "RMSE_std",
            "RMSE_median",
            "RMSE_count",
            "MAE_mean",
            "MAE_std",
            "MAE_median",
            "MAE_count",
            "MAPE_mean",
            "MAPE_std",
            "MAPE_median",
            "MAPE_count",
            "R2",
            "R2_mean",
            "R2_std",
            "R2_median",
            "R2_count",
            "Directional_Accuracy",
            "Directional_Accuracy_mean",
            "Directional_Accuracy_std",
        )
        if col in frame.columns
    ]
    print(f"\n{title}:")
    print(frame[columns].to_string(index=False))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", choices=PIPELINE_CHOICES, default=PIPELINE_MODE)
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "df_final.csv")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--run-id", default=None, help="Optional deterministic run directory name.")
    parser.add_argument(
        "--paper-seeds",
        nargs="+",
        type=int,
        default=(7, 42, 123, 2024, 2025),
        help=(
            "Reporting seeds used by benchmark/ablation, shared-mean benchmark, "
            "the standalone hard-FFNN comparator, and sensitivity studies."
        ),
    )
    parser.add_argument(
        "--skip-varnn",
        action="store_true",
        help="Skip the optional TensorFlow VARNN benchmark.",
    )
    parser.add_argument(
        "--block-subset-search-strategy",
        choices=("full-grid", "two-stage"),
        default="two-stage",
        help=(
            "Search strategy for the AIC variable-block comparator inside the legacy "
            "benchmark pipeline only. block-proposed-benchmark always runs both the "
            "A5 full-grid reference and A7 two-stage Proposed. "
            "two-stage fast-screens all block specs with alpha=0.1 and LR=0.001, "
            "then full-grids the best four specs per component."
        ),
    )
    parser.add_argument(
        "--macro-profile",
        choices=("core-only", "full"),
        default="core-only",
        help=(
            "core-only runs only the five project workflows; full additionally "
            "runs the Kaggle reproduction, Naive, MLP, LSTM, and GRU."
        ),
    )
    parser.add_argument(
        "--tensorflow-device",
        choices=("gpu", "cpu", "auto"),
        default="gpu",
        help="TensorFlow device policy. macro-benchmark defaults to strict GPU.",
    )
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument(
        "--ffnn-scaler",
        choices=("standard", "minmax"),
        default="standard",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    csv_path = args.data.resolve()
    if args.pipeline == "macro-benchmark" and csv_path == (PROJECT_ROOT / "df_final.csv"):
        csv_path = PROJECT_ROOT / "macro_indicators.csv"
    if not csv_path.exists():
        print(f"Dataset not found: {csv_path}", file=sys.stderr)
        return 2

    if args.pipeline == "macro-benchmark":
        runtime = TensorFlowRuntimeConfig(
            device=args.tensorflow_device,
            gpu_index=args.gpu_index,
        )
        try:
            _, _, tensorflow_metadata = bootstrap_tensorflow(runtime)
            macro_data = load_macro_dataset(csv_path)
            selected_features, _ = screen_macro_features(macro_data.train)
            provisional = build_macro_experiment_config(
                selected_features,
                Path("results"),
                tensorflow=runtime,
                scaler=args.ffnn_scaler,
                seeds=tuple(args.paper_seeds),
            )
            run_dir = prepare_run_directory(
                args.output_root, args.pipeline, provisional, args.run_id
            )
        except (FileExistsError, ImportError, RuntimeError, ValueError) as exc:
            print(f"Cannot start macro pipeline: {exc}", file=sys.stderr)
            return 2
        config = replace(provisional, output_dir=run_dir)
        enabled_pipelines = pipelines_for_profile(args.macro_profile)
        metadata = {
            "macro_profile": args.macro_profile,
            "enabled_pipelines": enabled_pipelines,
            "skipped_pipelines": tuple(
                pipeline
                for pipeline in CORE_PIPELINES + FULL_ONLY_PIPELINES
                if pipeline not in enabled_pipelines
            ),
            "selected_features": selected_features,
            "paper_seeds": tuple(args.paper_seeds),
            "tensorflow": tensorflow_metadata,
            "dataset_sha256": file_sha256(csv_path),
            "notebook_source_sha256": NOTEBOOK_SHA256,
            "notebook_url": NOTEBOOK_URL,
            "result_scope": (
                "internal_workflow_comparison"
                if args.macro_profile == "core-only"
                else "kaggle_reproduction_and_fair_comparison"
            ),
        }
        write_manifest(
            run_dir,
            pipeline=args.pipeline,
            dataset=csv_path,
            config=config,
            status="running",
            metadata=metadata,
        )
        print(f"Pipeline: macro-benchmark ({args.macro_profile})")
        print(f"TensorFlow device: {tensorflow_metadata['selected_device']}")
        print(f"Selected train-only macro features: {selected_features}")
        print(f"Output: {run_dir}")
        try:
            result = run_macro_benchmark(
                csv_path,
                run_dir,
                profile=args.macro_profile,
                tensorflow=runtime,
                scaler=args.ffnn_scaler,
                seeds=tuple(args.paper_seeds),
            )
            metadata.update(
                enabled_pipelines=result["enabled_pipelines"],
                skipped_pipelines=result["skipped_pipelines"],
                dataset_sha256=result["dataset_sha256"],
                notebook_source_sha256=result["notebook_source_sha256"],
                notebook_url=result["notebook_url"],
            )
            print_metrics_summary(
                result["project"],
                "comparison_metrics",
                title=(
                    "Internal project-workflow comparison"
                    if args.macro_profile == "core-only"
                    else "Project-workflow comparison"
                ),
            )
            write_manifest(
                run_dir,
                pipeline=args.pipeline,
                dataset=csv_path,
                config=result["config"],
                status="success",
                metadata=metadata,
            )
        except Exception as exc:
            write_manifest(
                run_dir,
                pipeline=args.pipeline,
                dataset=csv_path,
                config=config,
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
                metadata=metadata,
            )
            traceback.print_exc()
            return 1
        print(f"\nPipeline completed successfully. Results: {run_dir}")
        return 0

    provisional = build_experiment_config(args.pipeline, Path("results"))
    if args.pipeline == "vmd-ardl-hard-ffnn":
        provisional = replace(
            provisional,
            ffnn=replace(
                provisional.ffnn,
                seed_grid=tuple(args.paper_seeds),
            ),
        )
    try:
        run_dir = prepare_run_directory(args.output_root, args.pipeline, provisional, args.run_id)
    except FileExistsError as exc:
        print(f"Run directory already exists: {exc.filename}", file=sys.stderr)
        return 2
    config = replace(provisional, output_dir=run_dir)
    varnn_config = build_varnn_config()
    run_metadata = {
        "paper_seeds": tuple(args.paper_seeds)
        if args.pipeline
        in {
            "benchmark",
            "block-proposed-benchmark",
            "shared-mean-benchmark",
            "sensitivity",
            "vmd-ardl-hard-ffnn",
        }
        else tuple(),
        "include_varnn": (
            not args.skip_varnn
            if args.pipeline in {"benchmark", "block-proposed-benchmark"}
            else None
        ),
        "block_subset_search_strategy": (
            args.block_subset_search_strategy.replace("-", "_")
            if args.pipeline == "benchmark"
            else None
        ),
        "selection_seed_aggregation": (
            "shared_mean_validation_across_reporting_seeds"
            if args.pipeline
            in {
                "block-proposed-benchmark",
                "shared-mean-benchmark",
                "vmd-ardl-hard-ffnn",
            }
            else None
        ),
        "ablation_schema_version": (
            "block_subset_proposed_v1"
            if args.pipeline == "block-proposed-benchmark"
            else None
        ),
        "benchmark_model_set": (
            (
                "ardl_ffnn_no_vmd",
                "vmd_ardl_block_subset_proposed",
                "vmd_ardl_sequential_screen",
                "vmd_ardl_joint_select",
            )
            if args.pipeline == "shared-mean-benchmark"
            else None
        ),
        "proposed_block_subset_search_strategy": (
            "two_stage"
            if args.pipeline == "shared-mean-benchmark"
            else None
        ),
        "proposed_fast_screen_validation_scope": (
            "train_inner_rolling"
            if args.pipeline == "shared-mean-benchmark"
            else None
        ),
    }
    write_manifest(
        run_dir,
        pipeline=args.pipeline,
        dataset=csv_path,
        config=config,
        status="running",
        metadata=run_metadata,
    )

    print(f"Pipeline: {args.pipeline}")
    print(f"Target-lag strategy: {config.ardl.target_lag_strategy}")
    print(f"Forecast protocol: {config.forecast_protocol}")
    if args.pipeline in {
        "benchmark",
        "block-proposed-benchmark",
        "shared-mean-benchmark",
        "sensitivity",
        "vmd-ardl-hard-ffnn",
    }:
        print(f"Paper seeds: {tuple(args.paper_seeds)}")
    if args.pipeline in {"benchmark", "block-proposed-benchmark"}:
        print(f"Include VARNN: {not args.skip_varnn}")
    if args.pipeline == "benchmark":
        print(f"Block-subset search: {args.block_subset_search_strategy}")
    print(f"Output: {run_dir}")
    try:
        results, metric_key = run_pipeline(
            args.pipeline,
            csv_path,
            config,
            varnn_config,
            paper_seeds=tuple(args.paper_seeds),
            include_varnn=not args.skip_varnn,
            block_subset_search_strategy=args.block_subset_search_strategy.replace(
                "-", "_"
            ),
        )
        print_metrics_summary(
            results,
            metric_key,
            title=(
                "Primary per-seed mean/std metrics"
                if args.pipeline
                in {
                    "benchmark",
                    "block-proposed-benchmark",
                    "shared-mean-benchmark",
                }
                else "Metrics summary"
            ),
        )
        if "ensemble_comparison_metrics" in results:
            print_metrics_summary(
                results,
                "ensemble_comparison_metrics",
                title="Secondary ensemble mean-prediction metrics",
            )
        write_manifest(
            run_dir,
            pipeline=args.pipeline,
            dataset=csv_path,
            config=config,
            status="success",
            metadata=run_metadata,
        )
    except Exception as exc:
        write_manifest(
            run_dir,
            pipeline=args.pipeline,
            dataset=csv_path,
            config=config,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            metadata=run_metadata,
        )
        traceback.print_exc()
        return 1

    print(f"\nPipeline completed successfully. Results: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
