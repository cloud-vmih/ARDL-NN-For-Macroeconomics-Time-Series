from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vmd_ardl_ffnn.config import DataConfig, ExperimentConfig, TensorFlowRuntimeConfig


def test_shared_mean_model_set_includes_lstm_without_changing_macro_core() -> None:
    from vmd_ardl_ffnn.macro_benchmark import CORE_PIPELINES
    from vmd_ardl_ffnn.shared_mean_benchmark_experiments import SHARED_MEAN_BENCHMARK_PIPELINES

    assert "lstm" in SHARED_MEAN_BENCHMARK_PIPELINES
    assert "lstm" not in CORE_PIPELINES


def test_shared_coverage_accepts_log_roundtrip_but_rejects_wrong_actual() -> None:
    from vmd_ardl_ffnn.shared_mean_benchmark_experiments import _require_complete_coverage

    dates = pd.date_range("2020-01-01", periods=2, freq="MS")
    raw = np.array([0.001, 0.002])
    one_split = pd.DataFrame({
        "date": [*dates, *dates],
        "split": ["test"] * 4,
        "pipeline": ["naive_last_observation"] * 2 + ["lstm"] * 2,
        "seed": [pd.NA, pd.NA, 7, 7],
        "actual": [*np.exp(np.log(raw)), *raw],
    })
    forecasts = pd.concat([
        one_split.assign(split="validation"), one_split,
    ], ignore_index=True)
    _require_complete_coverage(forecasts, (7,), ("lstm",))
    forecasts.loc[7, "actual"] = 0.2
    with pytest.raises(ValueError, match="Actual values differ"):
        _require_complete_coverage(forecasts, (7,), ("lstm",))


def _frame(n: int = 24) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n, freq="MS")
    return pd.DataFrame(
        {
            "Y": 10.0 + np.arange(n, dtype=float),
            "X": 100.0 + np.arange(n, dtype=float),
        },
        index=dates,
    )


def test_lstm_windows_use_train_scaler_and_only_past_rows() -> None:
    from vmd_ardl_ffnn.models.lstm import prepare_lstm_windows

    frame = _frame(12)
    original = prepare_lstm_windows(
        frame, target="Y", fit_end=6, prediction_start=6,
        prediction_end=8, lookback=2,
    )
    changed = frame.copy()
    changed.iloc[6, :] = 1_000_000.0
    outlier = prepare_lstm_windows(
        changed, target="Y", fit_end=6, prediction_start=6,
        prediction_end=8, lookback=2,
    )

    assert original.X_fit.shape == (4, 2, 2)
    assert original.X_pred.shape == (2, 2, 2)
    np.testing.assert_array_equal(original.X_pred[0], outlier.X_pred[0])
    np.testing.assert_array_equal(original.X_fit, outlier.X_fit)
    assert original.scaler.mean_[0] == frame["Y"].iloc[:6].mean()
    assert original.dates.tolist() == frame.index[6:8].tolist()
    assert original.actual.tolist() == frame["Y"].iloc[6:8].tolist()


def test_lstm_shared_selection_uses_mean_validation_error_then_refits(tmp_path: Path, monkeypatch) -> None:
    from vmd_ardl_ffnn import lstm_shared_mean as study
    from vmd_ardl_ffnn.models.lstm import LSTMConfig

    source = tmp_path / "data.csv"
    _frame().rename_axis("date").to_csv(source)
    config = ExperimentConfig(
        data=DataConfig(date_col="date", target="Y", features=("X",), log_transform=False),
        train_ratio=0.5,
        val_ratio=0.25,
        tensorflow=TensorFlowRuntimeConfig(device="cpu"),
        output_dir=tmp_path / "results",
    )
    candidate_config = LSTMConfig(
        lookback_grid=(2, 3), hidden_units_grid=(4,), learning_rate_grid=(0.001,),
        epochs=2, batch_size=4, patience=1,
    )
    calls: list[tuple[int, int, int]] = []

    def fake_fit(frame, *, target, fit_end, prediction_start, prediction_end,
                 lookback, hidden_units, learning_rate, seed, lstm_config, runtime):
        calls.append((fit_end, lookback, seed))
        actual = frame[target].iloc[prediction_start:prediction_end].to_numpy(float)
        if fit_end == 12:
            offset = {2: {1: 0.1, 2: 0.9}, 3: {1: 0.4, 2: 0.4}}[lookback][seed]
        else:
            offset = 0.2
        return actual + offset, 2

    monkeypatch.setattr(study, "fit_predict_lstm", fake_fit)
    results, metrics, forecasts, ranking, fit_count = study.run_lstm_shared_mean_seed(
        source, config, candidate_config, seeds=(1, 2), output_dir=tmp_path / "lstm",
    )

    assert ranking.loc[ranking["selected"], "lookback"].tolist() == [3]
    assert fit_count == 6
    assert calls.count((18, 3, 1)) == 1
    assert calls.count((18, 3, 2)) == 1
    assert set(metrics["split"]) == {"validation", "test"}
    assert set(metrics["seed"]) == {1, 2}
    assert metrics["Directional_Accuracy"].notna().all()
    for seed in (1, 2):
        for split, expected_dates in (("validation", _frame().index[12:18]), ("test", _frame().index[18:])):
            actual_dates = pd.DatetimeIndex(
                forecasts.loc[forecasts["seed"].eq(seed) & forecasts["split"].eq(split), "date"]
            )
            assert actual_dates.equals(expected_dates)
    assert len(results) == 2
    assert (tmp_path / "lstm" / "lstm_shared_mean_seed_selection.csv").is_file()


def test_lstm_cpu_fit_produces_one_prediction_per_date() -> None:
    from vmd_ardl_ffnn.models.lstm import LSTMConfig, fit_predict_lstm

    frame = _frame(18)
    config = LSTMConfig(
        lookback_grid=(3,), hidden_units_grid=(4,), learning_rate_grid=(0.001,),
        epochs=2, batch_size=4, patience=1,
    )
    predicted, epochs = fit_predict_lstm(
        frame, target="Y", fit_end=12, prediction_start=12, prediction_end=15,
        lookback=3, hidden_units=4, learning_rate=0.001, seed=7,
        lstm_config=config, runtime=TensorFlowRuntimeConfig(device="cpu"),
    )
    assert predicted.shape == (3,)
    assert np.isfinite(predicted).all()
    assert 1 <= epochs <= 2
