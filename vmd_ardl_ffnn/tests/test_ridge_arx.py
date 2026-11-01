from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from vmd_ardl_ffnn.config import DataConfig, ExperimentConfig


def _frame(n: int = 40) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n, freq="MS")
    step = np.arange(n, dtype=float)
    return pd.DataFrame({"Y": 4.0 + 0.03 * step, "X": 100.0 + step}, index=dates)


def test_ridge_arx_window_uses_only_past_rows() -> None:
    from vmd_ardl_ffnn.models.ridge_arx import prepare_ridge_arx_windows

    frame = _frame(12)
    original = prepare_ridge_arx_windows(
        frame, target="Y", fit_end=6, prediction_start=6, prediction_end=8, lookback=2,
    )
    changed = frame.copy()
    changed.iloc[6, :] = 1_000_000.0
    after = prepare_ridge_arx_windows(
        changed, target="Y", fit_end=6, prediction_start=6, prediction_end=8, lookback=2,
    )
    np.testing.assert_array_equal(original.X_fit, after.X_fit)
    np.testing.assert_array_equal(original.X_pred[0], after.X_pred[0])


def test_ridge_arx_selects_validation_winner_then_refits(tmp_path: Path, monkeypatch) -> None:
    from vmd_ardl_ffnn import ridge_arx_shared_mean as study
    from vmd_ardl_ffnn.models.ridge_arx import RidgeARXConfig

    source = tmp_path / "data.csv"
    _frame().rename_axis("date").to_csv(source)
    config = ExperimentConfig(
        data=DataConfig(date_col="date", target="Y", features=("X",), log_transform=False),
        train_ratio=0.5, val_ratio=0.25, output_dir=tmp_path / "output",
    )
    candidate = RidgeARXConfig(lookback_grid=(1, 2), alpha_grid=(1.0,))
    calls: list[tuple[int, int]] = []

    def fake_fit(frame, *, target, fit_end, prediction_start, prediction_end,
                 lookback, alpha, config):
        calls.append((fit_end, lookback))
        actual = frame[target].iloc[prediction_start:prediction_end].to_numpy(float)
        return actual + (0.1 if lookback == 2 else 0.5)

    monkeypatch.setattr(study, "fit_predict_ridge_arx", fake_fit)
    result, metrics, forecasts, selection, fits = study.run_ridge_arx_validation_selected(
        source, config, candidate, output_dir=tmp_path / "ridge",
    )
    assert result["lookback"] == 2
    assert selection.loc[selection["selected"], "lookback"].tolist() == [2]
    assert calls.count((30, 2)) == 1
    assert fits == 3
    assert metrics["seed"].isna().all()
    assert forecasts["seed"].isna().all()
