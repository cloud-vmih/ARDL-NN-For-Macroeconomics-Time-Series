from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from vmd_ardl_ffnn.config import DataConfig, ExperimentConfig


def _frame(n: int = 36) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n, freq="MS")
    step = np.arange(n, dtype=float)
    return pd.DataFrame({
        "Y": 4.0 + 0.02 * step + 0.1 * np.sin(2 * np.pi * step / 12),
        "X": 100.0 + step,
    }, index=dates)


def test_xgboost_windows_use_only_past_rows() -> None:
    from vmd_ardl_ffnn.models.xgboost import prepare_xgboost_windows

    frame = _frame(12)
    original = prepare_xgboost_windows(
        frame, target="Y", fit_end=6, prediction_start=6,
        prediction_end=8, lookback=2,
    )
    changed = frame.copy()
    changed.iloc[6, :] = 1_000_000.0
    outlier = prepare_xgboost_windows(
        changed, target="Y", fit_end=6, prediction_start=6,
        prediction_end=8, lookback=2,
    )
    np.testing.assert_array_equal(original.X_fit, outlier.X_fit)
    np.testing.assert_array_equal(original.X_pred[0], outlier.X_pred[0])
    assert original.dates.tolist() == frame.index[6:8].tolist()


def test_xgboost_shared_selection_then_refits_each_seed(tmp_path: Path, monkeypatch) -> None:
    from vmd_ardl_ffnn import xgboost_shared_mean as study
    from vmd_ardl_ffnn.models.xgboost import XGBoostConfig

    source = tmp_path / "data.csv"
    _frame(40).rename_axis("date").to_csv(source)
    config = ExperimentConfig(
        data=DataConfig(date_col="date", target="Y", features=("X",), log_transform=False),
        train_ratio=0.5, val_ratio=0.25, output_dir=tmp_path / "results",
    )
    candidate_config = XGBoostConfig(
        lookback_grid=(2, 3), max_depth_grid=(2,), learning_rate_grid=(0.1,),
        n_estimators=5,
    )
    calls: list[tuple[int, int, int]] = []

    def fake_fit(frame, *, target, fit_end, prediction_start, prediction_end,
                 lookback, max_depth, learning_rate, seed, xgboost_config):
        calls.append((fit_end, lookback, seed))
        actual = frame[target].iloc[prediction_start:prediction_end].to_numpy(float)
        if fit_end == 20:
            offset = {2: {1: 0.1, 2: 0.9}, 3: {1: 0.4, 2: 0.4}}[lookback][seed]
        else:
            offset = 0.2
        return actual + offset

    monkeypatch.setattr(study, "fit_predict_xgboost", fake_fit)
    results, metrics, forecasts, ranking, fit_count = study.run_xgboost_shared_mean_seed(
        source, config, candidate_config, seeds=(1, 2), output_dir=tmp_path / "xgboost",
    )
    assert ranking.loc[ranking["selected"], "lookback"].tolist() == [3]
    assert fit_count == 6
    assert calls.count((30, 3, 1)) == 1
    assert calls.count((30, 3, 2)) == 1
    assert set(metrics["seed"]) == {1, 2}
    assert len(forecasts) == 40
    assert len(results) == 2


def test_xgboost_real_fit_produces_finite_forecasts() -> None:
    from vmd_ardl_ffnn.models.xgboost import XGBoostConfig, fit_predict_xgboost

    frame = _frame()
    config = XGBoostConfig(
        lookback_grid=(3,), max_depth_grid=(2,), learning_rate_grid=(0.1,),
        n_estimators=5,
    )
    predicted = fit_predict_xgboost(
        frame, target="Y", fit_end=24, prediction_start=24, prediction_end=30,
        lookback=3, max_depth=2, learning_rate=0.1, seed=7,
        xgboost_config=config,
    )
    assert predicted.shape == (6,)
    assert np.isfinite(predicted).all()
