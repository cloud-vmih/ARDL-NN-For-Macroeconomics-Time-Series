from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from .config import (
    ARDLSelectionConfig,
    DataConfig,
    ExperimentConfig,
    FFNNConfig,
    StationarityConfig,
    VMDConfig,
)
from .experiment import VMDARDLFFNNExperiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VMD + ARDL order selection + FFNN grid search.")
    parser.add_argument("--data", required=True, help="CSV input path.")
    parser.add_argument("--out", default="results", help="Output directory.")
    parser.add_argument("--date-col", default="TIME_PERIOD")
    parser.add_argument("--target", default="Export_US")
    parser.add_argument("--features", nargs="+", required=True)
    parser.add_argument("--freq", default="MS")
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--vmd-modes", type=int, default=3)
    parser.add_argument("--max-target-lag", type=int, default=12)
    parser.add_argument("--max-exog-lag", type=int, default=6)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--hr", type=int, nargs="+", default=[4, 8, 12, 16])
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--no-stationarity", action="store_true")
    parser.add_argument(
        "--pipeline",
        choices=["vmd", "no-vmd"],
        default="vmd",
        help="Forecasting pipeline to run.",
    )
    args = parser.parse_args()

    config = ExperimentConfig(
        data=DataConfig(
            date_col=args.date_col,
            target=args.target,
            features=tuple(args.features),
            freq=args.freq if args.freq.lower() != "none" else None,
            log_transform=not args.no_log,
        ),
        vmd=replace(VMDConfig(), modes=args.vmd_modes),
        ardl=replace(ARDLSelectionConfig(), max_target_lag=args.max_target_lag, max_exog_lag=args.max_exog_lag, top_n=args.top_n),
        ffnn=replace(FFNNConfig(), hidden_units_candidates=tuple(args.hr), max_iter=args.max_iter),
        stationarity=replace(StationarityConfig(), enabled=not args.no_stationarity),
        output_dir=Path(args.out),
    )
    experiment = VMDARDLFFNNExperiment(config)
    result = experiment.run_without_vmd(args.data) if args.pipeline == "no-vmd" else experiment.run(args.data)
    print("Best model rows:")
    print(result["best_component_models"].to_string(index=False))
    print("\nFinal metrics:")
    print(result["final_metrics"].to_string(index=False))


if __name__ == "__main__":
    main()
