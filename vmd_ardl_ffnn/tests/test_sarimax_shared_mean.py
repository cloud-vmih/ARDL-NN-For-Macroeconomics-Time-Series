from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from vmd_ardl_ffnn.config import DataConfig, ExperimentConfig


def _frame(n: int = 72) -> pd.DataFrame:
    dates = pd.date_range("2018-01-01", periods=n, freq="MS")
    step = np.arange(n, dtype=float)
    return pd.DataFrame(
        {
            "Y": 4.0 + 0.01 * step + 0.2 * np.sin(2 * np.pi * step / 12),
            "X": 100.0 + step,
        },
        index=dates,
    )


def test_sarimax_uses_only_lagged_exogenous_values(monkeypatch) -> None:
    from vmd_ardl_ffnn.models.sarimax import SARIMAXConfig, fit_predict_sarimax

    frame = _frame()
    config = SARIMAXConfig(
        order_grid=((1, 0, 0),),
        seasonal_order_grid=((0, 0, 0, 12),),
        exog_lag=1,
        maxiter=20,
    )
    original, *_ = fit_predict_sarimax(
        frame,
        target="Y",
        features=("X",),
        fit_end=48,
        prediction_start=48,
        prediction_end=49,
        order=(1, 0, 0),
        seasonal_order=(0, 0, 0, 12),
        config=config,
    )
    changed = frame.copy()
    changed.iloc[48, changed.columns.get_loc("X")] = 1_000_000.0
    after_change, *_ = fit_predict_sarimax(
        changed,
        target="Y",
        features=("X",),
        fit_end=48,
        prediction_start=48,
        prediction_end=49,
        order=(1, 0, 0),
        seasonal_order=(0, 0, 0, 12),
        config=config,
    )
    np.testing.assert_allclose(original, after_change)


def test_sarimax_validation_selection_then_refit(tmp_path: Path, monkeypatch) -> None:
    from vmd_ardl_ffnn import sarimax_shared_mean as study
    from vmd_ardl_ffnn.models.sarimax import SARIMAXConfig

    source = tmp_path / "data.csv"
    frame = _frame(40)
    frame.rename_axis("date").to_csv(source)
    config = ExperimentConfig(
        data=DataConfig(
            date_col="date", target="Y", features=("X",), log_transform=False
        ),
        train_ratio=0.5,
        val_ratio=0.25,
        output_dir=tmp_path / "results",
    )
    candidate_config = SARIMAXConfig(
        order_grid=((1, 0, 0), (2, 0, 0)),
        seasonal_order_grid=((0, 0, 0, 12),),
    )
    calls: list[tuple[int, tuple[int, int, int]]] = []

    def fake_fit(frame, *, target, features, fit_end, prediction_start,
                 prediction_end, order, seasonal_order, config):
        calls.append((fit_end, order))
        actual = frame[target].iloc[prediction_start:prediction_end].to_numpy(float)
        offset = 0.1 if order == (2, 0, 0) else 0.5
        return actual + offset, 10.0, True, 3

    monkeypatch.setattr(study, "fit_predict_sarimax", fake_fit)
    result, metrics, forecasts, ranking, fit_count = (
        study.run_sarimax_validation_selected(
            source,
            config,
            candidate_config,
            output_dir=tmp_path / "sarimax",
        )
    )

    assert result["order"] == (2, 0, 0)
    assert ranking.loc[ranking["selected"], "order"].tolist() == [(2, 0, 0)]
    assert fit_count == 3
    assert calls.count((30, (2, 0, 0))) == 1
    assert set(metrics["split"]) == {"validation", "test"}
    assert metrics["seed"].isna().all()
    assert forecasts["seed"].isna().all()
    assert len(forecasts) == 20
    assert (tmp_path / "sarimax" / "sarimax_validation_search.csv").is_file()
