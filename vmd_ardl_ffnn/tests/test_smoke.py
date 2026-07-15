from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import importlib.util
import tempfile

import numpy as np
import pandas as pd
import pytest

from vmd_ardl_ffnn.config import ARDLSelectionConfig, DataConfig, ExperimentConfig, FFNNConfig, VMDConfig
from vmd_ardl_ffnn import decomposition
from vmd_ardl_ffnn.decomposition import VariationalModeDecomposer
from vmd_ardl_ffnn.experiment import VMDARDLFFNNExperiment
from vmd_ardl_ffnn.features import LagSpec
from vmd_ardl_ffnn.lag_selection import ARDLOrderSelector
from vmd_ardl_ffnn.models.varnn import VARNNConfig
from vmd_ardl_ffnn.paper_baselines import run_linear_ardl_baseline, run_naive_baseline
requires_tensorflow = pytest.mark.skipif(
    importlib.util.find_spec("tensorflow") is None,
    reason="requires optional dependency: pip install -e '.[varnn]'",
)


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


def test_vmd_library_backend_uses_optional_vmd_function(monkeypatch) -> None:
    """Library backend should call the optional VMD function and keep component shape."""

    def fake_vmd(signal, alpha, tau, modes, dc, init, tolerance):
        assert np.isclose(np.mean(signal), 0.0)
        assert alpha == 123.0
        assert tau == 0.0
        assert modes == 2
        assert dc == 0
        assert init == 1
        assert tolerance == 1e-7
        return np.vstack([signal * 0.25, signal * 0.75]), np.empty((0,)), np.empty((0,))

    monkeypatch.setattr(decomposition, "_load_library_vmd", lambda: fake_vmd)
    series = pd.Series(np.linspace(1.0, 153.0, 153), name="x")
    levels = VariationalModeDecomposer(VMDConfig(modes=2, backend="library", alpha=123.0)).decompose(series)
    assert set(levels) == {"VMD1", "VMD2", "RES"}
    assert all(len(component) == len(series) for component in levels.values())


def test_ffnn_architecture_grid_uses_feature_count() -> None:
    """FFNN grid should search widths derived from n_features and multiple layer counts."""
    config = FFNNConfig(hidden_layer_candidates=(1, 2), hidden_width_multipliers=(1.0, 2.0, 0.5))
    assert config.architectures_for(6) == ((6,), (12,), (3,), (6, 6), (12, 12), (3, 3))
    assert config.fast_architecture_for(6) == (6,)


def test_train_only_lag_screen_builds_expanding_folds() -> None:
    data = synthetic_data(120).set_index("TIME_PERIOD")
    experiment = VMDARDLFFNNExperiment(
        ExperimentConfig(
            data=DataConfig(target="Y", features=("X1", "X2")),
            ardl=ARDLSelectionConfig(max_target_lag=3, max_exog_lag=2),
            ffnn=FFNNConfig(
                lag_screen_folds=3,
                lag_screen_validation_size=12,
            ),
        )
    )
    folds = experiment._expanding_lag_screen_splits(data.iloc[:84])
    assert len(folds) == 3
    assert [len(validation) for _, _, validation in folds] == [12, 12, 12]
    assert [len(train) for _, train, _ in folds] == [48, 60, 72]
    assert all(train.index.max() < validation.index.min() for _, train, validation in folds)


def test_one_standard_error_screen_keeps_raw_and_simple_winners() -> None:
    experiment = VMDARDLFFNNExperiment(
        ExperimentConfig(
            data=DataConfig(target="Y", features=("X1",), log_transform=False),
            ffnn=FFNNConfig(top_k_lag_specs=2, lag_screen_selection_rule="one_standard_error"),
        )
    )
    complex_spec = LagSpec(target_lags=(1, 2), exog_lags={"X1": (1, 2, 3)})
    simple_spec = LagSpec(target_lags=(1,), exog_lags={"X1": (1,)})
    weak_spec = LagSpec(target_lags=(1,), exog_lags={"X1": (1, 2)})
    scores = {
        complex_spec.label: (0.8, 1.0, 1.2),
        simple_spec.label: (1.05, 1.05, 1.05),
        weak_spec.label: (1.4, 1.4, 1.4),
    }
    fold_rows = [
        {
            "lag_spec": label,
            "learning_rate_init": 0.001,
            "screen_RMSE": score,
            "screen_MAE": score,
            "screen_MAPE": score,
        }
        for label, fold_scores in scores.items()
        for score in fold_scores
    ]
    selected, _ = experiment._summarize_lag_screen(
        "raw",
        [complex_spec, simple_spec, weak_spec],
        fold_rows,
    )
    stability = pd.DataFrame(experiment._lag_screen_stability_rows)
    assert selected == {complex_spec.label, simple_spec.label}
    assert stability.loc[
        stability["lag_spec"].eq(complex_spec.label), "raw_mean_rmse_winner"
    ].all()
    assert stability.loc[
        stability["lag_spec"].eq(simple_spec.label), "one_se_winner"
    ].all()


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
                learning_rate_grid=(1e-3,),
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
        assert "Val Level RMSE" in result["search_results"]
        assert set(result["search_results"]["final_selection_scale"]) == {"level"}
        assert not result["lag_screen_fold_metrics"].empty
        assert not result["lag_screen_stability"].empty


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


def test_model_comparison_metrics_normalize_scales() -> None:
    """VMD-ARDL-FFNN and VARNN metrics should compare on common scale labels."""
    experiment = VMDARDLFFNNExperiment()
    vmd_metrics = pd.DataFrame(
        [
            {"model": "VMD_ARDL_FFNN", "split": "validation", "scale": "reconstructed_transformed", "RMSE": 2.0, "MAE": 1.0, "MAPE": 10.0},
            {"model": "VMD_ARDL_FFNN", "split": "validation", "scale": "level", "RMSE": 4.0, "MAE": 2.0, "MAPE": 20.0},
        ]
    )
    varnn_metrics = pd.DataFrame(
        [
            {"model": "VARNN", "split": "validation", "scale": "raw_transformed", "RMSE": 1.0, "MAE": 0.5, "MAPE": 5.0},
            {"model": "VARNN", "split": "validation", "scale": "level", "RMSE": 3.0, "MAE": 1.5, "MAPE": 15.0},
        ]
    )
    comparison = experiment._model_metrics_comparison_table(
        vmd_metrics,
        varnn_metrics,
        "VMD_ARDL_FFNN",
        "VARNN",
    )
    assert set(comparison["scale"]) == {"transformed", "level"}
    transformed = comparison[comparison["scale"].eq("transformed")].iloc[0]
    assert transformed["delta_RMSE_VARNN_minus_VMD_ARDL_FFNN"] == -1.0
    assert transformed["VARNN_improvement_RMSE_pct"] == 50.0


def test_varnn_default_selects_lag_by_criterion() -> None:
    """Default VARNN config should use train-only VAR lag selection."""
    assert VARNNConfig().fixed_lag is None


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


def test_joint_target_lag_strategy_builds_specs_per_pacf_candidate() -> None:
    """Joint strategy should pair each target candidate with AIC-ranked ARDL exog lags."""
    data = synthetic_data().set_index("TIME_PERIOD")
    with tempfile.TemporaryDirectory() as temp_dir:
        experiment = VMDARDLFFNNExperiment(
            ExperimentConfig(
                data=DataConfig(target="Y", features=("X1", "X2")),
                ardl=ARDLSelectionConfig(
                    max_target_lag=4,
                    max_exog_lag=3,
                    target_lag_strategy="joint_validation_screen",
                    target_lag_preset="acf_pacf",
                    lag_spec_strategy="staged",
                    max_lag_specs=2,
                ),
                output_dir=Path(temp_dir) / "results",
            )
        )
        experiment._select_specs({"raw": data})

    table = experiment.order_tables_["raw"]
    joint_rows = table[table["feature"].eq("__target_lag_joint_validation_screen__")]
    combo_rows = table[table["feature"].eq("__ffnn_combo__")]
    exog_rows = table[table["feature"].isin(["X1", "X2"])]

    assert not joint_rows.empty
    assert not combo_rows.empty
    assert experiment.selected_specs_["raw"]
    assert set(exog_rows["selected_ic"].dropna()) == {"aic"}
    assert set(combo_rows["target_lag_strategy"]) == {"joint_validation_screen"}
    assert combo_rows["target_lag_candidate_id"].nunique() >= 1


def test_ardl_select_order_strategy() -> None:
    """Test that target_lag_strategy='ardl_select_order' runs correctly."""
    data = synthetic_data()
    selector = ARDLOrderSelector(
        ARDLSelectionConfig(
            max_target_lag=4,
            max_exog_lag=2,
            target_lag_strategy="ardl_select_order",
            force_target_lag_1=True,
        )
    )
    table, specs = selector.select(data.set_index("TIME_PERIOD"), "Y", ["X1", "X2"])
    target_rows = table[table["feature"].eq("__target_lag__")]
    selected_lags = set(target_rows[target_rows["selected_for_target_lags"]]["target_lag"].astype(int))
    assert 1 in selected_lags
    assert all(1 <= lag <= 4 for lag in selected_lags)
    assert all(set(spec.target_lags) == selected_lags for spec in specs)


def test_ffnn_validation_search_compares_learning_rates() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "data.csv"
        synthetic_data().to_csv(path, index=False)
        config = ExperimentConfig(
            data=DataConfig(target="Y", features=("X1", "X2")),
            ardl=ARDLSelectionConfig(
                max_target_lag=3,
                max_exog_lag=2,
                target_lag_strategy="fixed",
                fixed_target_lags=(1,),
                exog_lag_strategy="fixed",
                fixed_exog_lags=(1,),
            ),
            ffnn=FFNNConfig(
                hidden_units_candidates=(3,),
                alpha_grid=(1e-3,),
                seed_grid=(1,),
                activation_grid=("tanh",),
                learning_rate_grid=(0.01, 0.001),
                max_iter=10,
                search_strategy="full_grid",
            ),
            output_dir=Path(temp_dir) / "results",
        )
        result = VMDARDLFFNNExperiment(config).run_without_vmd(path)
        assert set(result["search_results"]["learning_rate_init"]) == {0.01, 0.001}
        assert "Val Level RMSE" in result["search_results"]
        assert set(result["best_component_models"]["final_selection_scale"]) == {"level"}
        assert float(result["best_component_models"].iloc[0]["learning_rate_init"]) in {
            0.01,
            0.001,
        }


def test_paper_classical_baselines_share_level_scale_test_dates() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "data.csv"
        synthetic_data().to_csv(path, index=False)
        config = ExperimentConfig(
            data=DataConfig(target="Y", features=("X1", "X2")),
            ardl=ARDLSelectionConfig(max_target_lag=3, max_exog_lag=2, top_n=2),
            output_dir=Path(temp_dir) / "ardl",
        )
        ardl = run_linear_ardl_baseline(path, config)
        naive = run_naive_baseline(
            path,
            replace(config, output_dir=Path(temp_dir) / "naive"),
            lag=1,
        )
        ardl_test = ardl["forecasts"][ardl["forecasts"]["split"].eq("test")]
        naive_test = naive["forecasts"][naive["forecasts"]["split"].eq("test")]
        assert set(ardl["metrics"]["scale"]) == {"level"}
        assert set(ardl_test["date"]) == set(naive_test["date"])
        assert ardl_test[["actual", "predicted"]].notna().all().all()


def test_experiment_benchmark_run() -> None:
    """Smoke test for running the benchmark comparison pipeline."""
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
                learning_rate_grid=(1e-3,),
                max_iter=10,
                fast_hidden_units=3,
                fast_max_iter=10,
            ),
            output_dir=Path(temp_dir) / "results",
        )
        result = VMDARDLFFNNExperiment(config).run_benchmark(
            path,
            seeds=(1,),
            include_varnn=False,
        )
        assert not result["comparison_metrics"].empty
        assert not result["comparison_forecasts"].empty
        assert "vmd_ardl_ffnn" in result
        assert "vmd_ardl_select_order_ffnn" in result
        assert "no_vmd_ardl_ffnn" in result
        assert "varnn" in result
        assert set(result["ablation_metrics"]["ablation_id"]) == {"A0", "A1", "A2", "A3", "A4", "A5"}
        assert not result["dm_tests"].empty
        assert result["date_coverage"]["same_dates_for_all_models"].all()


def test_block_proposed_benchmark_run() -> None:
    """Compact end-to-end smoke test for shared benchmark and A0--A7."""
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "data.csv"
        synthetic_data().to_csv(path, index=False)
        config = ExperimentConfig(
            data=DataConfig(target="Y", features=("X1", "X2")),
            vmd=VMDConfig(modes=2, max_iterations=20),
            ardl=ARDLSelectionConfig(
                max_target_lag=2,
                max_exog_lag=2,
                top_n=1,
                max_lag_specs=4,
            ),
            ffnn=FFNNConfig(
                hidden_units_candidates=(3,),
                alpha_grid=(1e-3,),
                seed_grid=(1,),
                activation_grid=("tanh",),
                learning_rate_grid=(1e-3,),
                max_iter=6,
                fast_hidden_units=3,
                fast_max_iter=6,
                top_k_lag_specs=2,
                top_component_candidates=1,
                lag_screen_folds=1,
                lag_screen_validation_size=8,
                lag_screen_learning_rate_grid=(1e-3,),
            ),
            output_dir=Path(temp_dir) / "block_proposed_results",
        )
        result = VMDARDLFFNNExperiment(config).run_block_proposed_benchmark(
            path,
            seeds=(1,),
            include_varnn=False,
        )
        assert set(result["ablation_metrics"]["ablation_id"]) == {
            "A0",
            "A1",
            "A2",
            "A3",
            "A4",
            "A5",
            "A6",
            "A7",
        }
        assert "vmd_ardl_block_subset_proposed" in set(
            result["comparison_metrics"]["pipeline"]
        )
        assert "vmd_ardl_joint_select" in set(
            result["comparison_metrics"]["pipeline"]
        )
        assert result["date_coverage"]["same_dates_for_all_models"].all()


def test_block_subset_comparator_locks_mean_seed_winner() -> None:
    """The comparator must select one shared structure, then refit every seed."""
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "data.csv"
        synthetic_data().to_csv(path, index=False)
        seeds = (1, 2)
        config = ExperimentConfig(
            data=DataConfig(target="Y", features=("X1", "X2")),
            vmd=VMDConfig(modes=2, max_iterations=30),
            ardl=ARDLSelectionConfig(
                max_target_lag=2,
                max_exog_lag=2,
                target_lag_strategy="ardl_select_order",
                exog_lag_strategy="ardl_block_subsets",
            ),
            ffnn=FFNNConfig(
                hidden_units_candidates=(3,),
                alpha_grid=(1e-3,),
                seed_grid=seeds,
                activation_grid=("tanh",),
                learning_rate_grid=(1e-3,),
                max_iter=8,
                search_strategy="full_grid",
            ),
            output_dir=Path(temp_dir) / "block_subset",
        )
        result = VMDARDLFFNNExperiment(config).run_block_subset_mean_seed(
            path,
            reporting_seeds=seeds,
            search_strategy="two_stage",
        )
        best = result["best_component_models"]
        assert set(best["seed"]) == set(seeds)
        assert best["selected_in_reconstructed_winner"].all()
        for _, component_rows in best.groupby("component"):
            assert component_rows["lag_spec"].nunique() == 1
            assert component_rows["hidden_layer_sizes"].map(tuple).nunique() == 1
            assert component_rows["selection_seed_count"].eq(len(seeds)).all()
        selected = result["search_results"].query("selected")
        assert len(selected) == 1
        assert selected["seed_count"].eq(len(seeds)).all()
        assert selected["final_selection_scale"].eq("level").all()
        assert selected["selection_protocol"].eq(
            "mean_reconstructed_level_validation_across_reporting_seeds"
        ).all()
        validation_level = result["final_metrics"].query(
            "split == 'validation' and scale == 'level'"
        )
        assert validation_level["RMSE"].mean() == pytest.approx(
            float(selected.iloc[0]["mean_val_level_RMSE"])
        )
        rankings = result["component_signature_rankings"]
        assert rankings["selected_in_reconstructed_winner"].sum() == best[
            "component"
        ].nunique()
        assert result["reconstructed_seed_scores"]["seed"].nunique() == len(seeds)
        fast_metrics = result["block_subset_fast_screen_metrics"]
        fast_summary = result["block_subset_fast_screen_summary"]
        assert not fast_metrics.empty
        assert fast_metrics["alpha"].eq(0.1).all()
        assert fast_metrics["learning_rate_init"].eq(0.001).all()
        assert fast_metrics["seed"].nunique() == len(seeds)
        assert fast_summary.groupby("component")["selected_for_full_grid"].sum().le(4).all()
        assert result["search_results"]["block_subset_search_strategy"].eq(
            "two_stage"
        ).all()
        combos = result["ardl_orders"].query("feature == '__ffnn_combo__'")
        assert combos.groupby("component").size().eq(3).all()
        assert set(result["final_forecasts"]["seed"]) == set(seeds)
        assert set(result["final_metrics"]["seed"]) == set(seeds)


def test_shared_no_vmd_variant_locks_one_signature_across_seeds() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "data.csv"
        synthetic_data().to_csv(path, index=False)
        seeds = (1, 2)
        config = ExperimentConfig(
            data=DataConfig(target="Y", features=("X1", "X2")),
            ardl=ARDLSelectionConfig(
                max_target_lag=2,
                max_exog_lag=2,
                target_lag_strategy="fixed",
                exog_lag_strategy="fixed",
            ),
            ffnn=FFNNConfig(
                hidden_units_candidates=(3,),
                alpha_grid=(1e-3,),
                seed_grid=seeds,
                activation_grid=("tanh",),
                learning_rate_grid=(1e-3,),
                max_iter=8,
                search_strategy="full_grid",
            ),
            output_dir=Path(temp_dir) / "shared_no_vmd",
        )
        result = VMDARDLFFNNExperiment(config).run_no_vmd_shared_mean_seed(
            path,
            reporting_seeds=seeds,
            model_name="Shared_NoVMD",
        )
        best = result["best_component_models"]
        assert set(best["seed"]) == set(seeds)
        assert best["lag_spec"].nunique() == 1
        assert best["hidden_layer_sizes"].map(tuple).nunique() == 1
        assert best["selection_seed_count"].eq(len(seeds)).all()
        assert result["mean_seed_rankings"].query("selected")[
            "seed_count"
        ].eq(len(seeds)).all()
        assert set(result["final_metrics"]["seed"]) == set(seeds)


def test_shared_vmd_component_selection_locks_one_signature_per_component() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "data.csv"
        synthetic_data().to_csv(path, index=False)
        seeds = (1, 2)
        config = ExperimentConfig(
            data=DataConfig(target="Y", features=("X1", "X2")),
            vmd=VMDConfig(modes=2, max_iterations=30),
            ardl=ARDLSelectionConfig(
                max_target_lag=2,
                max_exog_lag=2,
                target_lag_strategy="fixed",
                exog_lag_strategy="fixed",
            ),
            ffnn=FFNNConfig(
                hidden_units_candidates=(3,),
                alpha_grid=(1e-3,),
                seed_grid=seeds,
                activation_grid=("tanh",),
                learning_rate_grid=(1e-3,),
                max_iter=8,
                search_strategy="full_grid",
                top_component_candidates=1,
            ),
            output_dir=Path(temp_dir) / "shared_vmd",
        )
        result = VMDARDLFFNNExperiment(config).run_vmd_shared_mean_seed(
            path,
            reporting_seeds=seeds,
            model_name="Shared_VMD",
            selection_mode="component_mean_seed",
        )
        best = result["best_component_models"]
        assert set(best["seed"]) == set(seeds)
        for _, component_rows in best.groupby("component"):
            assert component_rows["lag_spec"].nunique() == 1
            assert component_rows["hidden_layer_sizes"].map(tuple).nunique() == 1
        assert result["search_results"]["selection_protocol"].eq(
            "component_mean_validation_across_reporting_seeds_then_reconstruct"
        ).all()
        assert set(result["final_metrics"]["seed"]) == set(seeds)


@requires_tensorflow
def test_varnn_tuning() -> None:
    """VARNN tuning mode should search the grid and save the selected hyperparameters."""
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
                learning_rate_grid=(1e-3,),
                max_iter=10,
                fast_hidden_units=3,
                fast_max_iter=10,
            ),
            output_dir=Path(temp_dir) / "results",
        )
        from vmd_ardl_ffnn.models.varnn import run_varnn_pipeline
        varnn_config = VARNNConfig(
            tune=True,
            hidden_units_grid=(4,),
            activation_grid=("sigmoid",),
            learning_rate_grid=(0.01, 0.001),
            epochs=2,
            batch_size=8,
        )
        validation_only = run_varnn_pipeline(
            path,
            replace(config, output_dir=Path(temp_dir) / "validation_only"),
            varnn_config,
            validation_only=True,
        )
        assert "training_history" not in validation_only
        validation_search = validation_only["validation_search"]
        assert set(validation_search["learning_rate"]) == {0.01, 0.001}
        assert {
            "Val Level RMSE",
            "Val Level MAE",
            "Val Level MAPE",
            "seed",
        }.issubset(validation_search.columns)
        assert (
            validation_search[["Val Level RMSE", "Val Level MAE"]]
            .notna()
            .all()
            .all()
        )

        result = run_varnn_pipeline(
            path,
            config,
            varnn_config,
        )
        assert "training_history" in result
        hist = result["training_history"]
        assert "selected_hidden_units" in hist.columns
        assert "selected_activation" in hist.columns
        assert "selected_learning_rate" in hist.columns
        assert hist["selected_hidden_units"].iloc[0] == 4
        assert hist["selected_activation"].iloc[0] == "sigmoid"
        assert hist["selected_learning_rate"].iloc[0] in {0.01, 0.001}
        search = result["validation_search"]
        assert set(search["learning_rate"]) == {0.01, 0.001}
        assert search["selected"].sum() == 1
        assert "Val Level RMSE" in search


@requires_tensorflow
def test_varnn_max_lag() -> None:
    """VARNN should respect the max_lag parameter and keep selected lag within range."""
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
                learning_rate_grid=(1e-3,),
                max_iter=10,
                fast_hidden_units=3,
                fast_max_iter=10,
            ),
            output_dir=Path(temp_dir) / "results",
        )
        from vmd_ardl_ffnn.models.varnn import run_varnn_pipeline
        result = run_varnn_pipeline(
            path,
            config,
            VARNNConfig(
                min_lag=2,
                max_lag=3,
                epochs=2,
                hidden_units=4,
                batch_size=8,
            )
        )
        assert "var_lag_selection" in result
        lag_table = result["var_lag_selection"]
        selected_lag = lag_table["selected_lag_used"].iloc[0]
        assert 2 <= selected_lag <= 3
