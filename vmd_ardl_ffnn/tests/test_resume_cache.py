from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from vmd_ardl_ffnn.artifacts import (
    build_resume_identity,
    validate_resume_manifest,
)
from vmd_ardl_ffnn.checkpoints import CandidateCheckpointStore
from vmd_ardl_ffnn.config import DataConfig, ExperimentConfig, FFNNConfig, VMDConfig
from vmd_ardl_ffnn.experiment import VMDARDLFFNNExperiment
from vmd_ardl_ffnn.features import LagSpec
from vmd_ardl_ffnn.vmd_cache import ContentAddressedVMDCache, VMDCacheRef


def test_vmd_disk_cache_reuses_complete_entry_and_bounds_memory(tmp_path: Path) -> None:
    cache = ContentAddressedVMDCache(
        tmp_path / "vmd",
        VMDConfig(modes=1, backend="self_contained"),
        max_memory_entries=1,
    )
    frame = pd.DataFrame(
        {"Y": [1.0, 2.0, 3.0, 4.0], "X": [4.0, 3.0, 2.0, 1.0]},
        index=pd.date_range("2020-01-01", periods=4, freq="MS"),
    )
    calls = 0

    def compute(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
        nonlocal calls
        calls += 1
        return {"VMD1": data * 0.5, "RES": data * 0.5}

    first = cache.get_or_compute(frame, compute)
    second = cache.get_or_compute(frame.copy(), compute)

    pd.testing.assert_frame_equal(second.load("VMD1"), frame * 0.5)
    second.load("RES")
    assert calls == 1
    assert cache.stats.hits == 1
    assert cache.stats.misses == 1
    assert cache.memory_entry_count == 1
    assert first.key == second.key


def test_vmd_cache_key_changes_with_column_order(tmp_path: Path) -> None:
    cache = ContentAddressedVMDCache(tmp_path, VMDConfig(modes=1))
    frame = pd.DataFrame(
        {"Y": [1.0, 2.0, 3.0, 4.0], "X": [2.0, 3.0, 4.0, 5.0]},
        index=pd.date_range("2020-01-01", periods=4, freq="MS"),
    )
    assert cache.key_for(frame) != cache.key_for(frame[["X", "Y"]])


def test_candidate_checkpoint_is_complete_only_after_marker(tmp_path: Path) -> None:
    descriptor = {"component": "VMD1", "seed": 7, "alpha": 0.001}
    store = CandidateCheckpointStore(tmp_path, {"dataset_sha256": "abc"})
    prediction = pd.DataFrame({"predicted_component": [1.25]})
    store.save(descriptor, {"candidate_id": 9, "Val RMSE": 0.2}, prediction)

    loaded = store.load(descriptor)
    assert loaded is not None
    row, restored_prediction = loaded
    assert row["Val RMSE"] == 0.2
    pd.testing.assert_frame_equal(restored_prediction, prediction)

    marker = next(tmp_path.glob("*/complete.json"))
    marker.unlink()
    assert CandidateCheckpointStore(tmp_path, {"dataset_sha256": "abc"}).load(descriptor) is None


def test_resume_manifest_rejects_different_identity(tmp_path: Path) -> None:
    dataset = tmp_path / "macro.csv"
    dataset.write_text("date,value\n2020-01-01,1\n", encoding="utf-8")
    expected = build_resume_identity(
        dataset=dataset,
        config={"model": "vmd"},
        selected_features=("a", "b", "c", "d"),
        tuning_seed=7,
        schema_version=1,
    )
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(
        json.dumps({"metadata": {"resume_identity": expected}}), encoding="utf-8"
    )
    validate_resume_manifest(run_dir, expected)

    changed = dict(expected, selected_features=["x", "b", "c", "d"])
    with pytest.raises(ValueError, match="resume identity mismatch"):
        validate_resume_manifest(run_dir, changed)


def test_sequential_resume_requires_run_id_and_uses_fixed_profile() -> None:
    path = Path(__file__).resolve().parents[2] / "run_pipeline_advanced.py"
    spec = importlib.util.spec_from_file_location("run_pipeline_advanced", path)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    with pytest.raises(SystemExit):
        runner.parse_args(["--pipeline", "sequential-vmd-ardl-ffnn", "--resume", "--data", "macro_indicators.csv"])
    assert runner.SEQUENTIAL_MACRO_FEATURES == (
        "initial_claims",
        "nonfarm_payrolls",
        "cpi_all_items",
        "industrial_prod",
    )
    assert runner.SEQUENTIAL_TUNING_SEED == 7


def test_experiment_origin_cache_is_disk_backed_and_causal(tmp_path: Path) -> None:
    config = ExperimentConfig(
        data=DataConfig(target="Y", features=("X",), log_transform=False),
        vmd=VMDConfig(modes=1, backend="self_contained"),
        output_dir=tmp_path / "out",
    )
    disk_cache = ContentAddressedVMDCache(tmp_path / "cache", config.vmd)
    experiment = VMDARDLFFNNExperiment(config, vmd_cache=disk_cache)
    experiment.decompose = lambda data: {"VMD1": data.copy(), "RES": data * 0.0}
    history = pd.DataFrame(
        {"Y": [1.0, 2.0, 3.0, 4.0], "X": [2.0, 3.0, 4.0, 5.0]},
        index=pd.date_range("2020-01-01", periods=4, freq="MS"),
    )
    evaluation = pd.DataFrame(
        {"Y": [5.0, 6.0], "X": [6.0, 7.0]},
        index=pd.date_range("2020-05-01", periods=2, freq="MS"),
    )

    origins = experiment._make_origin_cache(history, evaluation, "validation")

    first, second = list(origins.values())
    assert isinstance(first, VMDCacheRef)
    assert len(first.load("VMD1")) == 4
    assert len(second.load("VMD1")) == 5


def test_ffnn_search_reuses_completed_candidate_checkpoint(tmp_path: Path) -> None:
    config = ExperimentConfig(
        data=DataConfig(target="Y", features=("X",), log_transform=False),
        ffnn=FFNNConfig(
            hidden_layer_candidates=(1,),
            hidden_width_multipliers=(1.0,),
            alpha_grid=(0.001,),
            seed_grid=(7,),
            activation_grid=("tanh",),
            learning_rate_grid=(0.001,),
            min_train=2,
            min_val=1,
            search_strategy="full_grid",
        ),
        output_dir=tmp_path / "out",
    )
    store = CandidateCheckpointStore(tmp_path / "checkpoints", {"run": "same"})
    spec = LagSpec(target_lags=(1,), exog_lags={"X": (1,)})
    train = pd.DataFrame(
        {"Y": [1.0, 2.0, 3.0, 4.0], "X": [2.0, 3.0, 4.0, 5.0]},
        index=pd.date_range("2020-01-01", periods=4, freq="MS"),
    )

    class FakeModel:
        scaler = "standard"
        n_iter_ = 3
        parameter_count_ = 8
        device_ = "/CPU:0"

    def prediction(_model, _spec, candidate_id):
        return pd.DataFrame(
            {
                "candidate_id": [candidate_id],
                "actual_component": [1.0],
                "predicted_component": [1.1],
            }
        )

    first = VMDARDLFFNNExperiment(config, candidate_checkpoints=store)
    first._fit_model = lambda *args, **kwargs: (FakeModel(), 3)
    rows, _, _, _ = first._ffnn_search_candidates(
        "VMD1", train, [spec], prediction,
        lambda pred: {"RMSE": 0.1, "MAE": 0.1, "MAPE": 10.0}, True,
    )
    assert len(rows) == 1

    resumed = VMDARDLFFNNExperiment(
        config,
        candidate_checkpoints=CandidateCheckpointStore(
            tmp_path / "checkpoints", {"run": "same"}
        ),
    )
    resumed._fit_model = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("completed candidate was fitted again")
    )
    resumed_rows, _, _, _ = resumed._ffnn_search_candidates(
        "VMD1", train, [spec], prediction,
        lambda pred: {"RMSE": 0.1, "MAE": 0.1, "MAPE": 10.0}, True,
    )
    assert len(resumed_rows) == 1
    assert resumed.candidate_checkpoints.stats.hits == 1


def test_vmd_lag_screen_resumes_completed_fold(tmp_path: Path) -> None:
    config = ExperimentConfig(
        data=DataConfig(target="Y", features=("X",), log_transform=False),
        ffnn=FFNNConfig(min_train=2, min_val=1, seed_grid=(7,)),
        output_dir=tmp_path / "out",
    )
    spec = LagSpec(target_lags=(1,), exog_lags={"X": (1,)})
    train = pd.DataFrame(
        {"Y": [1.0, 2.0, 3.0, 4.0], "X": [2.0, 3.0, 4.0, 5.0]},
        index=pd.date_range("2020-01-01", periods=4, freq="MS"),
    )
    validation = pd.DataFrame(
        {"Y": [5.0], "X": [6.0]}, index=pd.DatetimeIndex(["2020-05-01"])
    )
    context = {
        "fold": 1,
        "train": train,
        "validation": validation,
        "train_components": {"VMD1": train},
        "origin_cache": {},
        "actual_components": {},
    }

    class FakeModel:
        pass

    def fake_prediction(*args, **kwargs):
        return pd.DataFrame(
            {"actual_component": [5.0], "predicted_component": [4.5]}
        )

    root = tmp_path / "fold-checkpoints"
    first = VMDARDLFFNNExperiment(
        config,
        candidate_checkpoints=CandidateCheckpointStore(root, {"run": "same"}),
    )
    first._fit_model = lambda *args, **kwargs: (FakeModel(), 3)
    first._predict_component_from_cache = fake_prediction
    expected = first._screen_vmd_lag_spec("VMD1", spec, 0.001, [context])

    resumed = VMDARDLFFNNExperiment(
        config,
        candidate_checkpoints=CandidateCheckpointStore(root, {"run": "same"}),
    )
    resumed._fit_model = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("completed fold was fitted again")
    )
    actual = resumed._screen_vmd_lag_spec("VMD1", spec, 0.001, [context])
    assert actual[0]["screen_RMSE"] == expected[0]["screen_RMSE"]
    assert actual[0]["n_train_after_lag"] == 3
    assert resumed.candidate_checkpoints.stats.hits == 1
