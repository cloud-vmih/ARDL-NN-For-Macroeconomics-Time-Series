"""Dispatch and summarize individual forecasting workflows."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .config import ExperimentConfig
from .experiment import VMDARDLFFNNExperiment
from .models.varnn import VARNNConfig, run_varnn_pipeline
from .models.sarimax import SARIMAXConfig
from .models.gru import GRUConfig
from .gru_shared_mean import run_gru_shared_mean_seed
from .models.xgboost import XGBoostConfig
from .xgboost_shared_mean import run_xgboost_shared_mean_seed
from .models.ridge_arx import RidgeARXConfig
from .ridge_arx_shared_mean import run_ridge_arx_validation_selected
from .models.mlp_arx import MLPARXConfig
from .mlp_arx_shared_mean import run_mlp_arx_shared_mean_seed
from .sarimax_shared_mean import run_sarimax_validation_selected

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
    if pipeline == "sarimax":
        result, metrics, forecasts, selection, fit_count = (
            run_sarimax_validation_selected(
                csv_path,
                config,
                SARIMAXConfig(),
                output_dir=Path(config.output_dir),
            )
        )
        return {
            "selected_model": result,
            "metrics": metrics,
            "forecasts": forecasts,
            "selection": selection,
            "fit_count": fit_count,
        }, "metrics"
    if pipeline == "gru":
        results, metrics, forecasts, selection, fit_count = run_gru_shared_mean_seed(
            csv_path,
            config,
            GRUConfig(),
            seeds=paper_seeds,
            output_dir=Path(config.output_dir),
        )
        return {
            "seed_results": results,
            "metrics": metrics,
            "forecasts": forecasts,
            "selection": selection,
            "fit_count": fit_count,
        }, "metrics"
    if pipeline == "xgboost":
        results, metrics, forecasts, selection, fit_count = (
            run_xgboost_shared_mean_seed(
                csv_path,
                config,
                XGBoostConfig(),
                seeds=paper_seeds,
                output_dir=Path(config.output_dir),
            )
        )
        return {
            "seed_results": results,
            "metrics": metrics,
            "forecasts": forecasts,
            "selection": selection,
            "fit_count": fit_count,
        }, "metrics"
    if pipeline == "ridge-arx":
        result, metrics, forecasts, selection, fit_count = (
            run_ridge_arx_validation_selected(
                csv_path,
                config,
                RidgeARXConfig(),
                output_dir=Path(config.output_dir),
            )
        )
        return {
            "selected_model": result,
            "metrics": metrics,
            "forecasts": forecasts,
            "selection": selection,
            "fit_count": fit_count,
        }, "metrics"
    if pipeline == "mlp-arx":
        results, metrics, forecasts, selection, fit_count = run_mlp_arx_shared_mean_seed(
            csv_path, config, MLPARXConfig(), seeds=paper_seeds,
            output_dir=Path(config.output_dir),
        )
        return {"seed_results": results, "metrics": metrics,
                "forecasts": forecasts, "selection": selection,
                "fit_count": fit_count}, "metrics"
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
                varnn_config=varnn_config,
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
