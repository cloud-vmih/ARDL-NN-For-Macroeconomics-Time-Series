from __future__ import annotations

import pandas as pd
import pytest


def test_seed_summary_keeps_per_seed_estimand() -> None:
    from vmd_ardl_ffnn.benchmark_common import aggregate_seed_metrics

    seeds = pd.DataFrame([
        {"model": "M", "pipeline": "p", "target": "y", "split": "test", "scale": "level", "seed": 1, "RMSE": 2.0},
        {"model": "M", "pipeline": "p", "target": "y", "split": "test", "scale": "level", "seed": 2, "RMSE": 4.0},
    ])
    summary = aggregate_seed_metrics(seeds)

    assert summary.loc[0, "estimand"] == "distribution_of_per_seed_metrics"
    assert summary.loc[0, "RMSE_mean"] == 3.0
    assert summary.loc[0, "RMSE_count"] == 2


def test_date_coverage_rejects_unequal_forecast_dates() -> None:
    from vmd_ardl_ffnn.benchmark_common import forecast_date_coverage

    forecasts = pd.DataFrame([
        {"pipeline": "a", "split": "test", "date": "2025-01-01"},
        {"pipeline": "a", "split": "test", "date": "2025-02-01"},
        {"pipeline": "b", "split": "test", "date": "2025-01-01"},
    ])
    with pytest.raises(ValueError, match="identical forecast dates"):
        forecast_date_coverage(forecasts)
