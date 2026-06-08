from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from vmd_ardl_ffnn.config import ARDLSelectionConfig, DataConfig, ExperimentConfig, FFNNConfig, VMDConfig
from vmd_ardl_ffnn.decomposition import VariationalModeDecomposer
from vmd_ardl_ffnn.experiment import VMDARDLFFNNExperiment
from vmd_ardl_ffnn.lag_selection import ARDLOrderSelector


def synthetic_data(n: int = 120) -> pd.DataFrame:
    """Tạo dữ liệu giả dương cho smoke test."""
    rng = np.random.default_rng(2)
    x1 = np.exp(np.cumsum(rng.normal(0.0, 0.02, n))) + 2.0
    x2 = np.exp(np.cumsum(rng.normal(0.0, 0.01, n))) + 3.0
    y = np.exp(2.0 + 0.02 * np.arange(n) / n + 0.15 * np.log(x1) - 0.05 * np.log(x2) + rng.normal(0, 0.01, n))
    return pd.DataFrame({"TIME_PERIOD": pd.date_range("2020-01-01", periods=n, freq="MS"), "Y": y, "X1": x1, "X2": x2})


def test_vmd_decomposition_reconstructs_shape() -> None:
    """Kiểm tra VMD trả đủ component cùng độ dài chuỗi."""
    series = pd.Series(np.sin(np.linspace(0, 10, 80)) + 0.1 * np.arange(80), name="x")
    levels = VariationalModeDecomposer(VMDConfig(modes=2, max_iterations=50)).decompose(series)
    assert set(levels) == {"VMD1", "VMD2", "RES"}
    assert all(len(component) == len(series) for component in levels.values())


def test_ffnn_architecture_grid_uses_feature_count() -> None:
    """FFNN grid should search widths derived from n_features and multiple layer counts."""
    config = FFNNConfig(hidden_layer_candidates=(1, 2), hidden_width_multipliers=(1.0, 2.0, 0.5))
    assert config.architectures_for(6) == ((6,), (12,), (3,), (6, 6), (12, 12), (3, 3))
    assert config.fast_architecture_for(6) == (6,)


def test_experiment_smoke_run() -> None:
    """Chạy nhanh pipeline VMD và kiểm tra các bảng đầu ra chính."""
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "data.csv"
        synthetic_data().to_csv(path, index=False)
        config = ExperimentConfig(
            data=DataConfig(target="Y", features=("X1", "X2")),
            vmd=VMDConfig(modes=2, max_iterations=50),
            ardl=ARDLSelectionConfig(max_target_lag=3, max_exog_lag=2, top_n=2),
            ffnn=FFNNConfig(
                hidden_units_candidates=(3,),
                alpha_grid=(1e-3,),
                seed_grid=(1,),
                max_iter=20,
                fast_hidden_units=3,
                fast_max_iter=20,
            ),
            output_dir=Path(temp_dir) / "results",
        )
        result = VMDARDLFFNNExperiment(config).run(path)
        assert not result["search_results"].empty
        assert not result["best_component_models"].empty
        assert not result["component_forecasts"].empty
        assert not result["final_forecasts"].empty
        assert not result["final_metrics"].empty
        assert not result["series_diagnostics"].empty
        assert not result["granger_causality"].empty
        assert not result["ardl_residual_diagnostics"].empty
        assert "__target_lag_validation_screen__" in set(result["ardl_orders"]["feature"])
        assert set(result["component_forecasts"]["component"]) == {"VMD1", "VMD2", "RES"}
        assert "actual_component" in result["component_forecasts"].columns
        assert not result["component_forecasts"]["actual_component"].isna().any()
        assert set(result["final_forecasts"]["component_count"]) == {3}
        assert int(result["best_component_models"].iloc[0]["HR"]) == 3


def test_vmd_component_scoring_uses_component_actuals() -> None:
    """Component scorer should compare component actuals, not reconstructed raw target."""
    experiment = VMDARDLFFNNExperiment()
    prediction = pd.DataFrame(
        {
            "date": pd.date_range("2021-01-01", periods=3, freq="MS"),
            "split": ["validation"] * 3,
            "component": ["VMD1"] * 3,
            "actual_component": [1.0, 2.0, 3.0],
            "predicted_component": [1.0, 2.0, 3.0],
        }
    )
    actual_raw = pd.DataFrame(
        {"Export_US": [10.0, 20.0, 30.0]},
        index=prediction["date"],
    )
    metrics = experiment._score_component_prediction(prediction, actual_raw)
    assert metrics["RMSE"] == 0.0
    assert metrics["MAE"] == 0.0


def test_staged_ardl_lag_specs_respect_budget() -> None:
    """Staged ARDL lag generation should cap FFNN lag specs."""
    data = synthetic_data()
    selector = ARDLOrderSelector(
        ARDLSelectionConfig(max_exog_lag=3, top_n=3, lag_spec_strategy="staged", max_lag_specs=4)
    )
    table, specs = selector.select(data.set_index("TIME_PERIOD"), "Y", ["X1", "X2"])
    combo_rows = table[table["feature"].eq("__ffnn_combo__")]
    assert 1 <= len(specs) <= 4
    assert len(combo_rows) == len(specs)
    assert set(combo_rows["lag_spec_strategy"]) == {"staged"}


def test_fixed_target_lags_preserve_legacy_policy() -> None:
    """Fixed target-lag strategy should reproduce the old target lag policy."""
    data = synthetic_data()
    selector = ARDLOrderSelector(
        ARDLSelectionConfig(
            max_target_lag=12,
            max_exog_lag=2,
            target_lag_strategy="fixed",
            fixed_target_lags=(1, 12),
        )
    )
    table, specs = selector.select(data.set_index("TIME_PERIOD"), "Y", ["X1", "X2"])
    target_rows = table[table["feature"].eq("__target_lag__")]
    assert {tuple(spec.target_lags) for spec in specs} == {(1, 12)}
    assert set(target_rows[target_rows["selected_for_target_lags"]]["target_lag"]) == {1, 12}


def test_ic_topk_target_lags_are_bounded_and_audited() -> None:
    """IC-selected target lags should be selected once without expanding lag specs."""
    data = synthetic_data()
    selector = ARDLOrderSelector(
        ARDLSelectionConfig(
            max_target_lag=4,
            max_exog_lag=2,
            target_lag_strategy="ic_topk",
            target_lag_top_n=1,
            force_target_lag_1=True,
            lag_spec_strategy="staged",
            max_lag_specs=3,
        )
    )
    table, specs = selector.select(data.set_index("TIME_PERIOD"), "Y", ["X1", "X2"])
    target_rows = table[table["feature"].eq("__target_lag__")]
    selected_lags = set(target_rows[target_rows["selected_for_target_lags"]]["target_lag"].astype(int))
    assert 1 in selected_lags
    assert all(1 <= lag <= 4 for lag in selected_lags)
    assert all(set(spec.target_lags) == selected_lags for spec in specs)
    assert len(specs) <= 3


def test_target_lag_presets_are_bounded() -> None:
    """Preset target-lag candidates should respect max_target_lag."""
    daily = ARDLOrderSelector(ARDLSelectionConfig(max_target_lag=7, target_lag_preset="daily"))
    yearly = ARDLOrderSelector(ARDLSelectionConfig(max_target_lag=3, target_lag_preset="yearly"))
    assert all(max(candidate) <= 7 for candidate in daily.target_lag_candidate_sets())
    assert all(max(candidate) <= 3 for candidate in yearly.target_lag_candidate_sets())
    assert (1, 7) in daily.target_lag_candidate_sets()


def test_acf_pacf_target_lag_candidates_use_train_target_signal() -> None:
    """ACF/PACF preset should create bounded PACF-core candidate sets from train data."""
    rng = np.random.default_rng(4)
    n = 160
    y = np.zeros(n)
    noise = rng.normal(0, 0.2, n)
    for i in range(2, n):
        y[i] = 0.65 * y[i - 1] - 0.35 * y[i - 2] + noise[i]
    data = pd.DataFrame(
        {
            "TIME_PERIOD": pd.date_range("2020-01-01", periods=n, freq="MS"),
            "Y": y + 10.0,
            "X1": rng.normal(size=n),
        }
    ).set_index("TIME_PERIOD")
    selector = ARDLOrderSelector(
        ARDLSelectionConfig(
            max_target_lag=6,
            target_lag_preset="acf_pacf",
            target_lag_acf_pacf_top_n=3,
            target_lag_acf_pacf_max_lags_per_set=3,
        )
    )
    candidates = selector.target_lag_candidate_sets(data, "Y")
    metadata = selector.target_lag_candidate_metadata(data, "Y")
    candidate_lags = {lag for candidate in candidates for lag in candidate}
    assert 1 in candidate_lags
    assert 2 in candidate_lags
    assert all(1 <= lag <= 6 for lag in candidate_lags)
    assert all(len(candidate) <= 3 for candidate in candidates)
    assert metadata
    assert all("pacf_value" in row and "acf_value" in row for row in metadata.values())
