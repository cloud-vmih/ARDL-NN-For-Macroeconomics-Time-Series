#!/usr/bin/env python3
"""Repository-local compatibility entrypoint for experiment runs."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
PACKAGE_SRC = PROJECT_ROOT / "vmd_ardl_ffnn" / "src"
if str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))

from vmd_ardl_ffnn.advanced_arguments import PIPELINE_CHOICES, PIPELINE_MODE, parse_args
from vmd_ardl_ffnn.advanced_config import (
    build_experiment_config,
    build_varnn_config,
    strategy_for_pipeline,
)
from vmd_ardl_ffnn.advanced_runner import (
    SEQUENTIAL_CHECKPOINT_SCHEMA_VERSION,
    SEQUENTIAL_MACRO_FEATURES,
    SEQUENTIAL_TUNING_SEED,
    main,
)
from vmd_ardl_ffnn.artifacts import file_sha256
from vmd_ardl_ffnn.dataset_profiles import (
    US_UNEMPLOYMENT_FEATURES,
    _is_macro_dataset,
    _validate_us_unemployment_result,
    build_us_unemployment_profile,
)
from vmd_ardl_ffnn.shared_mean_benchmark_experiments import (
    NAIVE_PIPELINE,
    SHARED_MEAN_BENCHMARK_PIPELINES,
)
from vmd_ardl_ffnn.workflow_dispatch import print_metrics_summary, run_pipeline


if __name__ == "__main__":
    raise SystemExit(main())
