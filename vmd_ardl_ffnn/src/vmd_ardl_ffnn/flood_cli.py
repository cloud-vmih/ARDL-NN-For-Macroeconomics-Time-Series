"""Command-line interface for panel audit and ARDL-FFNN flood-proxy experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .flood_config import FloodARDLConfig, FloodDataConfig, FloodFFNNConfig, FloodRoutingConfig
from .flood_data import FloodPanelLoader
from .flood_pipeline import run_flood_experiment


def _csv_tuple(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Son La hourly runoff and model-proxy flood warning pipeline."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit", help="Audit the current static/weather panel.")
    audit.add_argument("--static", default="static_features.csv")
    audit.add_argument("--observations", default="weather.csv")
    audit.add_argument("--output", default=None, help="Optional JSON audit output path.")
    audit.add_argument("--target", default="surface_runoff_mm_h")
    audit.add_argument(
        "--exog-cols",
        default="rain,temperature_2m,soil_moisture_1_to_3cm",
    )

    run = subparsers.add_parser("run", help="Fit and backtest the ARDL + FFNN residual hybrid.")
    run.add_argument("--static", required=True)
    run.add_argument("--observations", required=True)
    run.add_argument("--forecasts", default=None)
    run.add_argument("--output-dir", default="results/flood-proxy")
    run.add_argument("--train-end", required=True)
    run.add_argument("--validation-end", required=True)
    run.add_argument("--test-end", required=True)
    run.add_argument("--target", default="surface_runoff_mm_h")
    run.add_argument("--zone-col", default="HYBAS_ID")
    run.add_argument("--time-col", default="datetime")
    run.add_argument("--issue-time-col", default="issue_time")
    run.add_argument("--valid-time-col", default="valid_time")
    run.add_argument(
        "--exog-cols",
        default="rain,temperature_2m,soil_moisture_1_to_3cm",
    )
    run.add_argument("--nwp-cols", default="")
    run.add_argument(
        "--static-numeric-cols",
        default="SUB_AREA,UP_AREA,ORDER,min_elevation_m,max_elevation_m,mean_elevation_m",
    )
    run.add_argument("--static-categorical-cols", default="soil_name")
    run.add_argument("--max-target-lag", type=int, default=24)
    run.add_argument("--max-exog-lag", type=int, default=24)
    run.add_argument("--min-train-rows", type=int, default=720)
    run.add_argument("--max-horizon", type=int, default=24)
    run.add_argument("--issue-stride", type=int, default=6)
    run.add_argument("--max-iter", type=int, default=300)
    run.add_argument("--max-residual-samples", type=int, default=50_000)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--state-mode", choices=("recursive", "observed"), default="recursive")
    run.add_argument("--oracle-exogenous", action="store_true")
    run.add_argument("--river-thresholds", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "audit":
        _run_audit(args)
        return
    data_config = FloodDataConfig(
        zone_col=args.zone_col,
        time_col=args.time_col,
        issue_time_col=args.issue_time_col,
        valid_time_col=args.valid_time_col,
        target_col=args.target,
        exogenous_cols=_csv_tuple(args.exog_cols),
        forecast_exogenous_cols=_csv_tuple(args.nwp_cols),
        static_numeric_cols=_csv_tuple(args.static_numeric_cols),
        static_categorical_cols=_csv_tuple(args.static_categorical_cols),
    )
    run_dir = run_flood_experiment(
        static_path=args.static,
        observation_path=args.observations,
        forecast_path=args.forecasts,
        output_dir=args.output_dir,
        train_end=args.train_end,
        validation_end=args.validation_end,
        test_end=args.test_end,
        data_config=data_config,
        ardl_config=FloodARDLConfig(
            max_target_lag=args.max_target_lag,
            max_exog_lag=args.max_exog_lag,
            min_train_rows=args.min_train_rows,
        ),
        ffnn_config=FloodFFNNConfig(
            max_horizon_hours=args.max_horizon,
            issue_stride_hours=args.issue_stride,
            max_iter=args.max_iter,
            max_residual_samples=args.max_residual_samples,
            seed=args.seed,
        ),
        routing_config=FloodRoutingConfig(),
        river_threshold_path=args.river_thresholds,
        state_mode=args.state_mode,
        oracle_exogenous=args.oracle_exogenous,
    )
    print(run_dir)


def _run_audit(args: argparse.Namespace) -> None:
    config = FloodDataConfig(target_col=args.target, exogenous_cols=_csv_tuple(args.exog_cols))
    loader = FloodPanelLoader(config)
    static = loader.load_static(args.static)
    observations = loader.load_observations(args.observations, require_target=False)
    loader.validate_zone_coverage(observations, static)
    report = loader.audit(observations, static).to_dict()
    has_target = bool(report["has_target"])
    report.update(
        {
            "status": "target_required" if not has_target else "ready_for_oracle_backtest",
            "target_column": config.target_col,
            "run_dates": sorted(observations["run_date"].astype(str).unique().tolist())
            if "run_date" in observations
            else [],
            "soil_counts": static["soil_name"].value_counts().to_dict()
            if "soil_name" in static
            else {},
            "notes": (
                [
                    "Current files are suitable for ingestion/topology smoke tests only.",
                    "A runoff target and issue-time NWP forecast panel are required for deployable training.",
                ]
                if not has_target
                else [
                    "The panel contains a runoff target and supports historical oracle-exogenous backtesting.",
                    "An issue-time NWP panel is still required for a deployable 1-24 hour forecast evaluation.",
                ]
            ),
        }
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
