"""End-to-end orchestration for the Son La ARDL-FFNN flood-proxy experiment."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .flood_config import FloodARDLConfig, FloodDataConfig, FloodFFNNConfig, FloodRoutingConfig
from .flood_data import FloodPanelLoader
from .flood_evaluation import (
    LeadConformalIntervals,
    attach_actuals,
    score_proxy_alerts,
    score_runoff_forecasts,
)
from .flood_hybrid import ARDLFFNNResidualForecaster
from .flood_routing import BasinRouter, RunoffThresholds, apply_proxy_alerts


def utc_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def run_flood_experiment(
    *,
    static_path: str | Path,
    observation_path: str | Path,
    forecast_path: str | Path | None,
    output_dir: str | Path,
    train_end: str | pd.Timestamp,
    validation_end: str | pd.Timestamp,
    test_end: str | pd.Timestamp,
    data_config: FloodDataConfig,
    ardl_config: FloodARDLConfig | None = None,
    ffnn_config: FloodFFNNConfig | None = None,
    routing_config: FloodRoutingConfig | None = None,
    river_threshold_path: str | Path | None = None,
    state_mode: str = "recursive",
    oracle_exogenous: bool = False,
) -> Path:
    """Fit, recursively backtest, route and score one leakage-audited experiment."""
    ardl_config = ardl_config or FloodARDLConfig()
    ffnn_config = ffnn_config or FloodFFNNConfig()
    routing_config = routing_config or FloodRoutingConfig()
    train_end = utc_timestamp(train_end)
    validation_end = utc_timestamp(validation_end)
    test_end = utc_timestamp(test_end)
    if not train_end < validation_end < test_end:
        raise ValueError("Expected train_end < validation_end < test_end.")

    loader = FloodPanelLoader(data_config)
    static = loader.load_static(static_path)
    observations = loader.load_observations(observation_path, require_target=True)
    if oracle_exogenous:
        forecasts = loader.make_oracle_forecasts(
            observations,
            max_horizon=ffnn_config.max_horizon_hours,
            issue_stride_hours=ffnn_config.issue_stride_hours,
        )
    else:
        if forecast_path is None:
            raise ValueError("forecast_path is required unless oracle_exogenous=True.")
        forecasts = loader.load_forecasts(forecast_path)
    loader.validate_zone_coverage(observations, static, forecasts)
    audit = loader.audit(observations, static)

    experiment = ARDLFFNNResidualForecaster(data_config, ardl_config, ffnn_config)
    experiment.fit(observations, static, forecasts, train_end=train_end)
    evaluation_forecasts = forecasts.loc[
        (forecasts[data_config.issue_time_col] > train_end)
        & (forecasts[data_config.issue_time_col] <= test_end)
        & (forecasts["lead_hour"] <= ffnn_config.max_horizon_hours)
    ].copy()
    predictions = experiment.forecast_issues(
        observations,
        evaluation_forecasts,
        state_mode=state_mode,
    )
    if predictions.empty:
        raise ValueError("Evaluation produced no forecasts.")
    predictions = attach_actuals(predictions, observations, data_config)
    predictions["split"] = np.where(
        predictions[data_config.issue_time_col] <= validation_end, "validation", "test"
    )
    if predictions["actual_runoff_mm_h"].isna().all():
        raise ValueError("No actual runoff values overlap the validation/test forecast windows.")

    validation = predictions.loc[predictions["split"].eq("validation")]
    intervals = LeadConformalIntervals(coverage=0.90).fit(validation)
    predictions = intervals.apply(predictions)

    train_observations = observations.loc[observations[data_config.time_col] <= train_end]
    runoff_thresholds = RunoffThresholds(data_config, routing_config).fit(train_observations)
    router = BasinRouter(static, data_config, routing_config)
    routed, routing_audit = router.route(predictions)
    river_thresholds = None
    if river_threshold_path is not None:
        river_thresholds = pd.read_csv(river_threshold_path)
    warned = apply_proxy_alerts(
        routed,
        runoff_thresholds,
        data_config,
        routing_config,
        river_thresholds,
    )

    model_forecasts: list[pd.DataFrame] = []
    hybrid = predictions.copy()
    hybrid["model"] = "ARDL_FFNN_RESIDUAL"
    model_forecasts.append(hybrid)
    ardl = predictions.copy()
    ardl["runoff_prediction_mm_h"] = np.maximum(0.0, ardl["ardl_prediction"])
    ardl["model"] = "ARDL"
    model_forecasts.append(ardl)
    fixed_ffnn = predictions.copy()
    fixed_ffnn["runoff_prediction_mm_h"] = fixed_ffnn[
        "ffnn_fixed_lag_prediction_mm_h"
    ]
    fixed_ffnn["model"] = "FFNN_FIXED_LAGS"
    model_forecasts.append(fixed_ffnn)
    guided_ffnn = predictions.copy()
    guided_ffnn["runoff_prediction_mm_h"] = guided_ffnn[
        "ardl_guided_ffnn_prediction_mm_h"
    ]
    guided_ffnn["model"] = "ARDL_GUIDED_FFNN_DIRECT"
    model_forecasts.append(guided_ffnn)
    persistence = predictions.copy()
    state_lookup = observations.set_index([data_config.zone_col, data_config.time_col])[
        data_config.target_col
    ]
    persistence_values = []
    for _, row in persistence.iterrows():
        key = (str(row[data_config.zone_col]), row[data_config.issue_time_col])
        persistence_values.append(float(state_lookup.loc[key]) if key in state_lookup.index else np.nan)
    persistence["runoff_prediction_mm_h"] = persistence_values
    persistence["model"] = "PERSISTENCE"
    model_forecasts.append(persistence)

    metric_tables: list[pd.DataFrame] = []
    for frame in model_forecasts:
        metrics = score_runoff_forecasts(frame, data_config)
        metrics.insert(0, "model", frame["model"].iloc[0])
        metric_tables.append(metrics)
    metrics = pd.concat(metric_tables, ignore_index=True)
    alert_metrics = score_proxy_alerts(predictions, runoff_thresholds, data_config)
    acceptance = _acceptance_summary(metrics)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    pd.DataFrame([audit.to_dict()]).to_csv(run_dir / "data_audit.csv", index=False)
    experiment.ardl_selection_table().to_csv(run_dir / "ardl_lag_selection.csv", index=False)
    pd.DataFrame([experiment.residual_training_summary_]).to_csv(
        run_dir / "residual_training_summary.csv", index=False
    )
    predictions.to_csv(run_dir / "runoff_forecasts.csv", index=False)
    warned.to_csv(run_dir / "routed_proxy_warnings.csv", index=False)
    routing_audit.to_csv(run_dir / "routing_mass_balance.csv", index=False)
    runoff_thresholds.table().to_csv(run_dir / "runoff_thresholds_train_only.csv", index=False)
    intervals.table().to_csv(run_dir / "conformal_intervals_validation_only.csv", index=False)
    metrics.to_csv(run_dir / "runoff_metrics.csv", index=False)
    alert_metrics.to_csv(run_dir / "proxy_alert_metrics.csv", index=False)
    acceptance.to_csv(run_dir / "acceptance_summary.csv", index=False)
    manifest = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "warning_type": "model_proxy",
        "oracle_exogenous": bool(oracle_exogenous),
        "state_mode": state_mode,
        "paths": {
            "static": str(Path(static_path).resolve()),
            "observations": str(Path(observation_path).resolve()),
            "forecasts": str(Path(forecast_path).resolve()) if forecast_path else None,
        },
        "splits": {
            "train_end": train_end.isoformat(),
            "validation_end": validation_end.isoformat(),
            "test_end": test_end.isoformat(),
        },
        "data_config": asdict(data_config),
        "ardl_config": asdict(ardl_config),
        "ffnn_config": asdict(ffnn_config),
        "routing_config": asdict(routing_config),
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return run_dir


def _acceptance_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    test = metrics.loc[metrics["split"].eq("test")]
    summary = test.groupby("model", observed=True).agg(
        mean_RMSE=("RMSE", "mean"),
        mean_MAE=("MAE", "mean"),
        mean_NSE=("NSE", "mean"),
        mean_KGE=("KGE", "mean"),
    )
    hybrid = summary.loc["ARDL_FFNN_RESIDUAL"] if "ARDL_FFNN_RESIDUAL" in summary.index else None
    rows: list[dict[str, object]] = []
    for baseline in [
        "ARDL",
        "PERSISTENCE",
        "FFNN_FIXED_LAGS",
        "ARDL_GUIDED_FFNN_DIRECT",
    ]:
        passed = False
        if hybrid is not None and baseline in summary.index:
            competitor = summary.loc[baseline]
            passed = bool(
                hybrid["mean_RMSE"] < competitor["mean_RMSE"]
                and hybrid["mean_KGE"] > competitor["mean_KGE"]
            )
        rows.append(
            {
                "criterion": f"hybrid_beats_{baseline.lower()}",
                "passed": passed,
                "rule": "lower mean test RMSE and higher mean test KGE",
            }
        )
    return pd.DataFrame.from_records(rows)
