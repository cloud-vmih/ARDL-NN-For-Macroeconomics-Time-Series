"""Arguments and validation for the run-scoped CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

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

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", choices=PIPELINE_CHOICES, default=PIPELINE_MODE)
    parser.add_argument("--data", type=Path, required=True, help="Input CSV; must be selected explicitly.")
    parser.add_argument(
        "--dataset-profile", choices=("legacy-export", "us-unemployment"),
        default="legacy-export",
        help="Audited US unemployment data for shared-mean-benchmark, or legacy export data.",
    )
    parser.add_argument("--output-root", type=Path, default=Path.cwd() / "results")
    parser.add_argument("--run-id", default=None, help="Optional deterministic run directory name.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing sequential macro run. Requires --run-id.",
    )
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
        help="Skip VARNN only in benchmark and block-proposed-benchmark.",
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
            "core-only runs seven project/reference workflows; full additionally "
            "runs the Kaggle reproduction, MLP, LSTM, and GRU."
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
        "--tensorflow-run-eagerly", action="store_true",
        help="Train Keras FFNN models eagerly to reduce memory growth across many fits.",
    )
    parser.add_argument(
        "--ffnn-scaler",
        choices=("standard", "minmax"),
        default="standard",
    )
    args = parser.parse_args(argv)
    if args.resume and not args.run_id:
        parser.error("--resume requires an explicit --run-id")
    if args.resume and args.pipeline != "sequential-vmd-ardl-ffnn":
        parser.error("--resume is supported only by sequential-vmd-ardl-ffnn")
    if args.dataset_profile == "us-unemployment":
        if args.pipeline != "shared-mean-benchmark":
            parser.error("--dataset-profile us-unemployment requires shared-mean-benchmark")
    if args.skip_varnn and args.pipeline not in {"benchmark", "block-proposed-benchmark"}:
        parser.error("--skip-varnn is supported only by benchmark and block-proposed-benchmark")
    if "df_final.csv" in {args.data.name, args.data.resolve().name}:
        parser.error("df_final.csv is excluded because its source compilation is invalid")
    return args
