#!/usr/bin/env python3
"""Run-scoped orchestration for the forecasting experiment pipelines."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import traceback

from vmd_ardl_ffnn.artifacts import (
    build_resume_identity,
    prepare_run_directory,
    validate_resume_manifest,
    write_csv,
    write_manifest,
)
from vmd_ardl_ffnn.checkpoints import CandidateCheckpointStore
from vmd_ardl_ffnn.config import TensorFlowRuntimeConfig
from vmd_ardl_ffnn.experiment import VMDARDLFFNNExperiment
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
from vmd_ardl_ffnn.models.lstm import LSTMConfig
from vmd_ardl_ffnn.models.gru import GRUConfig
from vmd_ardl_ffnn.models.xgboost import XGBoostConfig
from vmd_ardl_ffnn.models.ridge_arx import RidgeARXConfig
from vmd_ardl_ffnn.models.mlp_arx import MLPARXConfig
from vmd_ardl_ffnn.models.sarimax import SARIMAXConfig
from vmd_ardl_ffnn.sarimax_shared_mean import SARIMAX_PIPELINE
from vmd_ardl_ffnn.ridge_arx_shared_mean import RIDGE_ARX_PIPELINE
from vmd_ardl_ffnn.shared_mean_benchmark_experiments import (
    NAIVE_PIPELINE,
    SHARED_MEAN_BENCHMARK_PIPELINES,
)
from vmd_ardl_ffnn.vmd_cache import ContentAddressedVMDCache
from vmd_ardl_ffnn.advanced_arguments import parse_args
from vmd_ardl_ffnn.advanced_config import build_experiment_config, build_varnn_config
from vmd_ardl_ffnn.dataset_profiles import build_us_unemployment_profile, _validate_us_unemployment_result, _is_macro_dataset
from vmd_ardl_ffnn.workflow_dispatch import run_pipeline, print_metrics_summary


# Edit this tuple when you intentionally want to change the four macro inputs.
SEQUENTIAL_MACRO_FEATURES = (
    "initial_claims",
    "nonfarm_payrolls",
    "cpi_all_items",
    "industrial_prod",
)
SEQUENTIAL_TUNING_SEED = 7
SEQUENTIAL_CHECKPOINT_SCHEMA_VERSION = 1


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    csv_path = args.data.resolve()
    if not csv_path.exists():
        print(f"Dataset not found: {csv_path}", file=sys.stderr)
        return 2

    if args.pipeline == "macro-benchmark":
        runtime = TensorFlowRuntimeConfig(
            device=args.tensorflow_device,
            gpu_index=args.gpu_index,
            run_eagerly=args.tensorflow_run_eagerly,
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

    # The sequential profile also supports macro_indicators.csv, but keeps its
    # execution intentionally narrow: one VMD -> ARDL -> FFNN experiment only.
    if args.pipeline == "sequential-vmd-ardl-ffnn" and _is_macro_dataset(csv_path):
        runtime = TensorFlowRuntimeConfig(
            device=args.tensorflow_device,
            gpu_index=args.gpu_index,
            run_eagerly=args.tensorflow_run_eagerly,
        )
        try:
            macro_data = load_macro_dataset(csv_path)
            selected_features = SEQUENTIAL_MACRO_FEATURES
            missing_features = [
                feature
                for feature in selected_features
                if feature not in macro_data.model_data.columns
            ]
            if missing_features:
                raise ValueError(
                    f"Configured sequential macro features are missing: {missing_features}"
                )
            provisional = build_macro_experiment_config(
                selected_features,
                Path("results"),
                tensorflow=runtime,
                scaler=args.ffnn_scaler,
                seeds=(SEQUENTIAL_TUNING_SEED,),
            )
            resume_identity = build_resume_identity(
                dataset=csv_path,
                config=provisional,
                selected_features=selected_features,
                tuning_seed=SEQUENTIAL_TUNING_SEED,
                schema_version=SEQUENTIAL_CHECKPOINT_SCHEMA_VERSION,
            )
            if args.resume:
                run_dir = args.output_root / args.pipeline / str(args.run_id)
                if not run_dir.is_dir():
                    raise ValueError(f"Resume run directory does not exist: {run_dir}")
                validate_resume_manifest(run_dir, resume_identity)
            else:
                run_dir = prepare_run_directory(
                    args.output_root, args.pipeline, provisional, args.run_id
                )
            config = replace(provisional, output_dir=run_dir)
            prepared_csv = run_dir / "prepared_macro.csv"
            write_csv(macro_data.model_data.reset_index(), prepared_csv)
            metadata = {
                "dataset_schema": "kaggle_macro",
                "selected_features": selected_features,
                "tuning_seeds": (SEQUENTIAL_TUNING_SEED,),
                "feature_screening": "fixed_from_feature_screening_not_rerun",
                "execution": "single_sequential_vmd_ardl_ffnn",
                "resume": bool(args.resume),
                "resume_identity": resume_identity,
            }
            write_manifest(
                run_dir,
                pipeline=args.pipeline,
                dataset=csv_path,
                config=config,
                status="running",
                metadata=metadata,
            )
            _, _, tensorflow_metadata = bootstrap_tensorflow(runtime)
            metadata["tensorflow"] = tensorflow_metadata
        except (FileExistsError, ImportError, RuntimeError, ValueError) as exc:
            print(f"Cannot start sequential macro pipeline: {exc}", file=sys.stderr)
            return 2
        write_manifest(
            run_dir,
            pipeline=args.pipeline,
            dataset=csv_path,
            config=config,
            status="running",
            metadata=metadata,
        )
        print(f"Pipeline: {args.pipeline} (macro schema)")
        print(f"TensorFlow device: {tensorflow_metadata['selected_device']}")
        print(f"Fixed macro features: {selected_features}")
        print(f"Tuning seeds: {(SEQUENTIAL_TUNING_SEED,)}")
        print(f"Output: {run_dir}")
        try:
            vmd_cache = ContentAddressedVMDCache(
                args.output_root / "_cache" / "vmd",
                config.vmd,
                max_memory_entries=2,
            )
            candidate_checkpoints = CandidateCheckpointStore(
                run_dir / "checkpoints" / "ffnn",
                resume_identity,
            )
            experiment = VMDARDLFFNNExperiment(
                config,
                vmd_cache=vmd_cache,
                candidate_checkpoints=candidate_checkpoints,
            )
            results, metric_key = experiment.run(prepared_csv), "final_metrics"
            print_metrics_summary(results, metric_key)
            metadata["vmd_cache"] = vars(vmd_cache.stats)
            metadata["candidate_checkpoints"] = vars(candidate_checkpoints.stats)
            write_manifest(
                run_dir,
                pipeline=args.pipeline,
                dataset=csv_path,
                config=config,
                status="success",
                metadata=metadata,
            )
        except Exception as exc:
            if "vmd_cache" in locals():
                metadata["vmd_cache"] = vars(vmd_cache.stats)
            if "candidate_checkpoints" in locals():
                metadata["candidate_checkpoints"] = vars(candidate_checkpoints.stats)
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

    profile_metadata: dict[str, object] = {}
    if (
        args.pipeline in {
            "shared-mean-benchmark", "gru", "xgboost", "ridge-arx", "mlp-arx", "sarimax"
        }
        and args.dataset_profile == "us-unemployment"
    ):
        try:
            provisional, profile_metadata = build_us_unemployment_profile(
                csv_path, Path("results"),
                runtime=TensorFlowRuntimeConfig(
                    device=args.tensorflow_device, gpu_index=args.gpu_index,
                    run_eagerly=args.tensorflow_run_eagerly,
                ),
                scaler=args.ffnn_scaler, seeds=tuple(args.paper_seeds),
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"Cannot start unemployment study: {exc}", file=sys.stderr)
            return 2
        if args.pipeline == "sarimax":
            profile_metadata["benchmark_model_set"] = (SARIMAX_PIPELINE,)
        if args.pipeline == "gru":
            profile_metadata["benchmark_model_set"] = ("gru",)
        if args.pipeline == "xgboost":
            profile_metadata["benchmark_model_set"] = ("xgboost",)
        if args.pipeline == "ridge-arx":
            profile_metadata["benchmark_model_set"] = (RIDGE_ARX_PIPELINE,)
        if args.pipeline == "mlp-arx":
            profile_metadata["benchmark_model_set"] = ("mlp_arx",)
    else:
        provisional = build_experiment_config(args.pipeline, Path("results"))
        provisional = replace(
            provisional,
            tensorflow=TensorFlowRuntimeConfig(
                device=args.tensorflow_device,
                gpu_index=args.gpu_index,
                run_eagerly=args.tensorflow_run_eagerly,
            ),
        )
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
        if profile_metadata:
            source_audit = Path(str(profile_metadata["source_audit_path"]))
            copied_audit = run_dir / "source_audit.json"
            copied_audit.write_bytes(source_audit.read_bytes())
            if file_sha256(copied_audit) != profile_metadata["source_audit_sha256"]:
                raise ValueError("Source audit changed while preparing the run.")
            profile_metadata["run_source_audit_path"] = str(copied_audit.resolve())
    except FileExistsError as exc:
        print(f"Run directory already exists: {exc.filename}", file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        print(f"Cannot prepare run: {exc}", file=sys.stderr)
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
            "gru",
            "xgboost",
            "ridge-arx",
            "mlp-arx",
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
                "gru",
                "xgboost",
                "ridge-arx",
                "mlp-arx",
            }
            else None
        ),
        "ablation_schema_version": (
            "block_subset_proposed_v1"
            if args.pipeline == "block-proposed-benchmark"
            else None
        ),
        "benchmark_model_set": (
            (*SHARED_MEAN_BENCHMARK_PIPELINES, NAIVE_PIPELINE)
            if args.pipeline == "shared-mean-benchmark"
            else None
        ),
        "lstm_config": (
            asdict(LSTMConfig())
            if args.pipeline == "shared-mean-benchmark"
            else None
        ),
        "gru_config": (
            asdict(GRUConfig())
            if args.pipeline in {"shared-mean-benchmark", "gru"}
            else None
        ),
        "xgboost_config": (
            asdict(XGBoostConfig())
            if args.pipeline in {"shared-mean-benchmark", "xgboost"}
            else None
        ),
        "ridge_arx_config": (
            asdict(RidgeARXConfig())
            if args.pipeline in {"shared-mean-benchmark", "ridge-arx"}
            else None
        ),
        "mlp_arx_config": (
            asdict(MLPARXConfig())
            if args.pipeline in {"shared-mean-benchmark", "mlp-arx"}
            else None
        ),
        "sarimax_config": (
            asdict(SARIMAXConfig())
            if args.pipeline in {"shared-mean-benchmark", "sarimax"}
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
    run_metadata.update(profile_metadata)
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
        "gru",
        "xgboost",
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
        if args.pipeline == "shared-mean-benchmark" and args.dataset_profile == "us-unemployment":
            _validate_us_unemployment_result(results, run_metadata, csv_path)
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
