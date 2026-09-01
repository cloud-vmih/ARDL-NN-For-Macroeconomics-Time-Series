from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vmd_ardl_ffnn.config import TensorFlowRuntimeConfig
from vmd_ardl_ffnn.macro_benchmark import (
    CORE_PIPELINES,
    FULL_ONLY_PIPELINES,
    build_macro_experiment_config,
    _notebook_inner_split,
    load_macro_dataset,
    pipelines_for_profile,
    run_macro_benchmark,
    screen_macro_features,
)


def _macro_frame() -> pd.DataFrame:
    dates = pd.date_range("2000-01-01", "2026-08-01", freq="MS")
    values = np.arange(len(dates), dtype=float)
    frame = pd.DataFrame(
        {
            "date": dates,
            "unemployment_rate": 4.0 + values / 1000.0,
            "initial_claims": 200_000 + values,
            "nonfarm_payrolls": 100_000 + values,
            "consumer_sentiment": 80 + values / 10,
            "industrial_prod": 90 + values / 20,
            "retail_sales": 300_000 + values,
            "building_permits": 1_000 + values,
        }
    )
    frame.loc[frame["date"].eq(pd.Timestamp("2025-10-01")), "unemployment_rate"] = np.nan
    return frame


def test_macro_loader_uses_continuous_observed_prefix_and_fixed_splits(tmp_path: Path) -> None:
    path = tmp_path / "macro.csv"
    _macro_frame().to_csv(path, index=False)

    dataset = load_macro_dataset(path)

    assert len(dataset.model_data) == 309
    assert dataset.model_data.index.max() == pd.Timestamp("2025-09-01")
    assert dataset.train.index.max() == pd.Timestamp("2017-12-01")
    assert dataset.validation.index[[0, -1]].tolist() == [
        pd.Timestamp("2018-01-01"),
        pd.Timestamp("2021-10-01"),
    ]
    assert dataset.test.index[[0, -1]].tolist() == [
        pd.Timestamp("2021-11-01"),
        pd.Timestamp("2025-09-01"),
    ]


def test_macro_feature_screen_is_train_only_and_keeps_exactly_four(monkeypatch) -> None:
    frame = _macro_frame().set_index("date").loc[:"2017-12-01"]
    score = {
        "initial_claims": 6.0,
        "nonfarm_payrolls": 2.0,
        "consumer_sentiment": 1.0,
        "industrial_prod": 4.0,
        "retail_sales": 3.0,
        "building_permits": 5.0,
    }

    def fake_score(data, target, feature, target_lag, exog_lag):
        assert data.index.max() == pd.Timestamp("2017-12-01")
        return score[feature] + target_lag / 100 + exog_lag / 1000

    monkeypatch.setattr(
        "vmd_ardl_ffnn.macro_benchmark._fit_candidate_aic", fake_score
    )
    selected, audit = screen_macro_features(frame)

    assert selected == (
        "consumer_sentiment",
        "nonfarm_payrolls",
        "retail_sales",
        "industrial_prod",
    )
    assert len(audit) == 6 * 12 * 6
    assert audit["used_for_model"].sum() == 4 * 12 * 6


def test_macro_profiles_make_notebook_models_strictly_opt_in() -> None:
    assert pipelines_for_profile("core-only") == CORE_PIPELINES
    assert pipelines_for_profile("full") == CORE_PIPELINES + FULL_ONLY_PIPELINES
    with pytest.raises(ValueError, match="macro profile"):
        pipelines_for_profile("unknown")


def test_notebook_early_stopping_split_stays_inside_training_data() -> None:
    X = np.arange(1200, dtype=float).reshape(100, 12)
    y = np.arange(100, dtype=float)
    fit_X, fit_y, validation_X, validation_y = _notebook_inner_split(X, y)
    assert len(fit_X) == len(fit_y) == 80
    assert len(validation_X) == len(validation_y) == 20
    assert fit_y[-1] < validation_y[0]


def test_macro_experiment_requires_sktime_and_requested_gpu(tmp_path: Path) -> None:
    config = build_macro_experiment_config(
        ("initial_claims", "retail_sales", "industrial_prod", "nonfarm_payrolls"),
        tmp_path,
        tensorflow=TensorFlowRuntimeConfig(device="gpu", gpu_index=1),
    )
    assert config.vmd.backend == "sktime"
    assert config.tensorflow.device == "gpu"
    assert config.tensorflow.gpu_index == 1
    assert config.ffnn.scaler == "standard"
    assert config.ffnn.seed_grid == (7, 42, 123, 2024, 2025)


def test_core_only_does_not_invoke_or_create_notebook_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "macro.csv"
    _macro_frame().to_csv(source, index=False)
    selected = (
        "initial_claims",
        "nonfarm_payrolls",
        "retail_sales",
        "industrial_prod",
    )
    monkeypatch.setattr(
        "vmd_ardl_ffnn.macro_benchmark.screen_macro_features",
        lambda train: (selected, pd.DataFrame({"feature": selected})),
    )
    monkeypatch.setattr(
        "vmd_ardl_ffnn.macro_benchmark._run_project_workflows",
        lambda *args, **kwargs: {
            "comparison_metrics": pd.DataFrame([{"pipeline": CORE_PIPELINES[0], "RMSE": 1.0}]),
            "comparison_forecasts": pd.DataFrame([{"pipeline": CORE_PIPELINES[0], "predicted": 4.0}]),
            "settings": pd.DataFrame([{"pipeline": CORE_PIPELINES[0]}]),
        },
    )
    monkeypatch.setattr(
        "vmd_ardl_ffnn.macro_benchmark._run_notebook_benchmarks",
        lambda *args, **kwargs: pytest.fail("core-only imported notebook execution branch"),
    )

    result = run_macro_benchmark(
        source,
        tmp_path / "out",
        profile="core-only",
        tensorflow=TensorFlowRuntimeConfig(device="cpu"),
    )

    assert result["enabled_pipelines"] == CORE_PIPELINES
    assert set(result["skipped_pipelines"]) == set(FULL_ONLY_PIPELINES)
    assert len(result["dataset_sha256"]) == 64
    assert len(result["notebook_source_sha256"]) == 64
    assert not (tmp_path / "out" / "notebook_reproduction").exists()
    assert not (tmp_path / "out" / "notebook_fair_models").exists()


def test_macro_gpu_policy_fails_before_creating_artifacts(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "must_not_exist"
    monkeypatch.setattr(
        "vmd_ardl_ffnn.macro_benchmark.bootstrap_tensorflow",
        lambda runtime: (_ for _ in ()).throw(RuntimeError("no requested GPU")),
    )
    with pytest.raises(RuntimeError, match="requested GPU"):
        run_macro_benchmark(
            tmp_path / "input_does_not_need_to_be_read.csv",
            output,
            tensorflow=TensorFlowRuntimeConfig(device="gpu"),
        )
    assert not output.exists()
