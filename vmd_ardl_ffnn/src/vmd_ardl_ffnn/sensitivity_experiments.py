from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd

from .artifacts import write_csv
from .benchmark_common import aggregate_seed_metrics, efficiency_row
from .config import ExperimentConfig
from .paper_experiments import (
    NEURAL_VARIANTS,
    run_neural_variant,
)


def sensitivity_profiles(config: ExperimentConfig) -> dict[str, ExperimentConfig]:
    """Build a compact one-factor-at-a-time robustness grid around the main setup."""
    profiles: dict[str, ExperimentConfig] = {"S0_base": config}

    for modes in (2, 4):
        if modes != config.vmd.modes:
            profiles[f"S_vmd_modes_{modes}"] = replace(config, vmd=replace(config.vmd, modes=modes))
    for label, alpha in (
        ("low", config.vmd.alpha / 2.0),
        ("high", config.vmd.alpha * 2.0),
    ):
        profiles[f"S_vmd_alpha_{label}"] = replace(
            config, vmd=replace(config.vmd, alpha=float(alpha))
        )

    compact_target = max(2, min(config.ardl.max_target_lag, 6))
    compact_exog = max(1, min(config.ardl.max_exog_lag, 3))
    profiles["S_compact_lag_bounds"] = replace(
        config,
        ardl=replace(
            config.ardl,
            max_target_lag=compact_target,
            max_exog_lag=compact_exog,
            fixed_target_lags=tuple(
                lag for lag in config.ardl.fixed_target_lags if lag <= compact_target
            ) or (1,),
            fixed_exog_lags=tuple(
                lag for lag in config.ardl.fixed_exog_lags if lag <= compact_exog
            ) or (1,),
        ),
    )
    profiles["S_bic"] = replace(config, ardl=replace(config.ardl, ic="bic"))
    profiles["S_full_product_feasible"] = replace(
        config,
        ardl=replace(
            config.ardl,
            max_exog_lag=compact_exog,
            top_n=min(config.ardl.top_n, 2),
            lag_spec_strategy="full_product",
            fixed_exog_lags=tuple(
                lag for lag in config.ardl.fixed_exog_lags if lag <= compact_exog
            ) or (1,),
        ),
    )
    profiles["S_split_65_20_15"] = replace(config, train_ratio=0.65, val_ratio=0.20)
    profiles["S_split_75_10_15"] = replace(config, train_ratio=0.75, val_ratio=0.10)
    return profiles


def _profile_settings(profile: str, config: ExperimentConfig, seeds: tuple[int, ...]) -> dict[str, Any]:
    return {
        "profile": profile,
        "reporting_seeds": seeds,
        "vmd_modes": config.vmd.modes,
        "vmd_alpha": config.vmd.alpha,
        "max_target_lag": config.ardl.max_target_lag,
        "max_exog_lag": config.ardl.max_exog_lag,
        "information_criterion": config.ardl.ic,
        "lag_spec_strategy": config.ardl.lag_spec_strategy,
        "top_n": config.ardl.top_n,
        "ffnn_learning_rate_grid": config.ffnn.learning_rate_grid,
        "train_ratio": config.train_ratio,
        "validation_ratio": config.val_ratio,
        "test_ratio": 1.0 - config.train_ratio - config.val_ratio,
        "forecast_protocol": config.forecast_protocol,
    }


def run_sensitivity_experiments(
    csv_path: str | Path,
    config: ExperimentConfig,
    *,
    seeds: tuple[int, ...] = (7, 42, 123, 2024, 2025),
) -> dict[str, pd.DataFrame]:
    """Run the predeclared one-factor-at-a-time proposed-model sensitivity study."""
    if not seeds:
        raise ValueError("Sensitivity experiments require at least one reporting seed.")
    root = Path(config.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    metric_frames: list[pd.DataFrame] = []
    forecast_frames: list[pd.DataFrame] = []
    efficiency_rows: list[dict[str, Any]] = []
    setting_rows: list[dict[str, Any]] = []
    proposed = NEURAL_VARIANTS["vmd_ardl_ffnn_proposed"]

    for profile, profile_config in sensitivity_profiles(config).items():
        setting_rows.append(_profile_settings(profile, profile_config, tuple(seeds)))
        for seed in seeds:
            started = perf_counter()
            result, metrics, forecasts = run_neural_variant(
                csv_path,
                profile_config,
                pipeline=profile,
                definition={**proposed, "model": f"Proposed_{profile}"},
                seed=seed,
                output_dir=root / profile / f"seed_{seed}",
            )
            metrics["sensitivity_profile"] = profile
            forecasts["sensitivity_profile"] = profile
            metric_frames.append(metrics)
            forecast_frames.append(forecasts)
            efficiency_rows.append(
                {
                    **efficiency_row(
                        result,
                        profile_config,
                        pipeline=profile,
                        seed=seed,
                        elapsed_seconds=perf_counter() - started,
                    ),
                    "sensitivity_profile": profile,
                }
            )

    seed_metrics = pd.concat(metric_frames, ignore_index=True, sort=False)
    forecasts = pd.concat(forecast_frames, ignore_index=True, sort=False)
    summary = aggregate_seed_metrics(seed_metrics)
    efficiency = pd.DataFrame(efficiency_rows)
    settings = pd.DataFrame(setting_rows)
    write_csv(seed_metrics, root / "sensitivity_seed_metrics.csv")
    write_csv(summary, root / "sensitivity_metrics_summary.csv")
    write_csv(forecasts, root / "sensitivity_forecasts.csv")
    write_csv(efficiency, root / "sensitivity_efficiency.csv")
    write_csv(settings, root / "sensitivity_settings.csv")
    return {
        "metrics": summary,
        "seed_metrics": seed_metrics,
        "forecasts": forecasts,
        "efficiency": efficiency,
        "settings": settings,
    }
