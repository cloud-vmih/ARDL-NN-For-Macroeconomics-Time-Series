"""Predeclared configurations for the legacy experiment profiles."""

from __future__ import annotations

from pathlib import Path

from .config import ARDLSelectionConfig, DataConfig, DiagnosticsConfig, ExperimentConfig, FFNNConfig, StationarityConfig, VMDConfig
from .models.varnn import VARNNConfig

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
            seed_grid=(7, ),
            activation="tanh",
            activation_grid=("tanh",),
            learning_rate_init=0.001,
            learning_rate_grid=(0.001,),
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
            lag_screen_learning_rate_grid=(0.001,),
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
        activation="tanh",
        learning_rate=1e-3,
        tune=True,
        hidden_units_grid=(32,),
        activation_grid=("tanh",),
        learning_rate_grid=(0.001,),
        epochs=100,
        batch_size=32,
        patience=12,
        seed=7,
    )
