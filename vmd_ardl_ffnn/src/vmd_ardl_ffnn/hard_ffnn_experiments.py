from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ExperimentConfig
from .evaluation import normalize_forecasts
from .experiment import VMDARDLFFNNExperiment


HARD_FFNN_PIPELINE = "vmd_ardl_hard_config_ffnn"
HARD_FFNN_MODEL = "VMD_ARDL_HardConfig_FFNN"


def run_hard_ffnn_variant(
    csv_path: str | Path,
    config: ExperimentConfig,
    *,
    seeds: tuple[int, ...],
    output_dir: Path,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, ExperimentConfig]:
    """Run the shared hard-FFNN comparator for any benchmark study."""
    variant_config = replace(
        config,
        output_dir=output_dir,
        ardl=replace(
            config.ardl,
            ic="aic",
            target_lag_strategy="ardl_select_order",
            exog_lag_strategy="ardl_screen",
            lag_spec_strategy="full_product",
            top_n=3,
        ),
        ffnn=replace(config.ffnn, seed_grid=seeds),
    )
    result = VMDARDLFFNNExperiment(
        variant_config
    ).run_hard_ffnn_comparator(
        csv_path,
        reporting_seeds=seeds,
        model_name=HARD_FFNN_MODEL,
    )
    metrics = result["final_metrics"].copy()
    metrics = metrics[metrics["scale"].eq("level")].copy()
    metrics["model"] = HARD_FFNN_MODEL
    metrics["pipeline"] = HARD_FFNN_PIPELINE
    forecasts = normalize_forecasts(
        result["final_forecasts"],
        pipeline=HARD_FFNN_PIPELINE,
        predicted_col="predicted_reconstructed",
    )
    return result, metrics, forecasts, variant_config
