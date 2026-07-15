from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vmd_ardl_ffnn.artifacts import prepare_run_directory, write_csv, write_manifest
from vmd_ardl_ffnn.config import ARDLSelectionConfig, DataConfig, ExperimentConfig
from vmd_ardl_ffnn.data import GenericDataLoader
from vmd_ardl_ffnn.experiment import VMDARDLFFNNExperiment
from vmd_ardl_ffnn.features import LagSpec
from vmd_ardl_ffnn.lag_selection import ARDLOrderSelector
from vmd_ardl_ffnn.metrics import evaluate_forecast
from vmd_ardl_ffnn.models.ffnn import NumpyFFNNRegressor
from vmd_ardl_ffnn.models.varnn import _choose_var_lag
from vmd_ardl_ffnn.statistical_tests import holm_adjust, pairwise_dm_tests
from vmd_ardl_ffnn.sensitivity_experiments import sensitivity_profiles


def _runner_module():
    path = Path(__file__).resolve().parents[2] / "run_pipeline_advanced.py"
    spec = importlib.util.spec_from_file_location("run_pipeline_advanced", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _frame(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(9)
    return pd.DataFrame(
        {"Y": rng.normal(size=n).cumsum(), "X": rng.normal(size=n).cumsum()},
        index=pd.date_range("2010-01-01", periods=n, freq="MS"),
    )


def test_runner_profiles_use_sequential_proposed_and_keep_joint_comparator() -> None:
    runner = _runner_module()
    proposed = runner.build_experiment_config("vmd", Path("out"))
    block_proposed = runner.build_experiment_config(
        "block-proposed-benchmark",
        Path("out"),
    )
    select_order = runner.build_experiment_config("vmd-ardl-select-order", Path("out"))
    joint = runner.build_experiment_config("vmd-ardl-joint-select", Path("out"))
    assert proposed.ardl.target_lag_strategy == "ardl_select_order"
    assert select_order.ardl.target_lag_strategy == "ardl_select_order"
    assert joint.ardl.target_lag_strategy == "ardl_select_order_joint"
    assert proposed.ffnn.learning_rate_grid == (0.001,)
    assert block_proposed.ffnn.learning_rate_grid == (0.01, 0.001)
    assert proposed.ffnn.lag_screen_folds == 3
    assert proposed.ffnn.lag_screen_validation_size == 12
    assert proposed.ffnn.lag_screen_learning_rates() == (0.01, 0.001)
    assert proposed.ffnn.lag_screen_selection_rule == "one_standard_error"
    assert proposed.ffnn.final_selection_scale == "level"
    varnn = runner.build_varnn_config()
    assert varnn.tune is True
    assert varnn.hidden_units_grid == (32,)
    assert varnn.activation_grid == ("sigmoid",)
    assert varnn.learning_rate_grid == (0.01, 0.001)


def test_runner_returns_nonzero_for_missing_dataset(tmp_path: Path) -> None:
    runner = _runner_module()
    assert runner.main(["--data", str(tmp_path / "missing.csv")]) == 2


def test_runner_parses_block_subset_two_stage_option() -> None:
    runner = _runner_module()
    args = runner.parse_args(
        ["--pipeline", "benchmark", "--block-subset-search-strategy", "two-stage"]
    )
    assert args.block_subset_search_strategy == "two-stage"


def test_runner_exposes_separate_block_proposed_pipeline() -> None:
    runner = _runner_module()
    args = runner.parse_args(["--pipeline", "block-proposed-benchmark"])
    assert args.pipeline == "block-proposed-benchmark"


def test_block_subset_fast_screen_keeps_exactly_four_best_specs(monkeypatch) -> None:
    experiment = VMDARDLFFNNExperiment(
        ExperimentConfig(
            data=DataConfig(target="Y", features=("X",), log_transform=False)
        )
    )
    specs = [
        LagSpec(target_lags=(1,), exog_lags={"X": tuple(range(1, order + 1))})
        for order in range(1, 6)
    ]
    dates = pd.date_range("2020-01-01", periods=16, freq="MS")
    train = pd.DataFrame({"Y": 10.0, "X": 5.0}, index=dates)
    validation = train.iloc[-8:].copy()

    class FakeModel:
        def __init__(self, spec: LagSpec, seed: int):
            self.spec = spec
            self.seed = seed

    def fake_fit(data, spec, hidden_layer_sizes, alpha, seed, activation, **kwargs):
        assert alpha == 0.1
        assert kwargs["learning_rate_init"] == 0.001
        return FakeModel(spec, seed), len(data)

    def fake_predict(
        model,
        component,
        cache,
        actual_raw,
        spec,
        split,
        candidate_id=None,
        actual_components=None,
    ):
        order = max(spec.exog_lags["X"])
        return pd.DataFrame(
            {
                "date": actual_raw.index,
                "split": split,
                "component": component,
                "actual_component": np.full(len(actual_raw), 10.0),
                "predicted_component": np.full(
                    len(actual_raw), 10.0 + order + model.seed * 0.001
                ),
            }
        )

    monkeypatch.setattr(experiment, "_fit_model", fake_fit)
    monkeypatch.setattr(experiment, "_predict_component_from_cache", fake_predict)
    selected, detail, summary = experiment._fast_screen_block_subset_specs(
        component="VMD1",
        train_data=train,
        specs=specs,
        validation=validation,
        validation_cache={},
        validation_actual_components={},
        seeds=(1, 2),
    )
    assert len(detail) == len(specs) * 2
    assert len(selected) == 4
    assert summary["selected_for_full_grid"].sum() == 4
    assert specs[-1].label not in {spec.label for spec in selected}
    assert summary["fast_screen_seed_aggregation"].eq(
        "mean_across_reporting_seeds"
    ).all()


def test_sensitivity_grid_is_one_factor_at_a_time_and_keeps_base() -> None:
    profiles = sensitivity_profiles(
        ExperimentConfig(data=DataConfig(target="Y", features=("X",), log_transform=False))
    )
    assert "S0_base" in profiles
    assert profiles["S_bic"].ardl.ic == "bic"
    assert profiles["S_full_product_feasible"].ardl.lag_spec_strategy == "full_product"
    assert profiles["S_split_65_20_15"].train_ratio == 0.65


def test_joint_select_keeps_statsmodels_joint_lags(monkeypatch) -> None:
    class Selection:
        ar_lags = [2, 3]
        dl_lags = {"X": [1, 2]}

    monkeypatch.setattr("statsmodels.tsa.ardl.ardl_select_order", lambda **kwargs: Selection())
    selector = ARDLOrderSelector(
        ARDLSelectionConfig(
            max_target_lag=4,
            max_exog_lag=3,
            target_lag_strategy="ardl_select_order_joint",
            force_target_lag_1=True,
        )
    )
    _, specs = selector.select(_frame(), "Y", ["X"])
    assert specs[0].target_lags == (2, 3)
    assert specs[0].exog_lags == {"X": (1, 2)}


def test_joint_select_failure_policy_is_explicit(monkeypatch) -> None:
    def fail(**kwargs):
        raise RuntimeError("selection failed")

    monkeypatch.setattr("statsmodels.tsa.ardl.ardl_select_order", fail)
    strict = ARDLOrderSelector(
        ARDLSelectionConfig(target_lag_strategy="ardl_select_order_joint", selection_failure_policy="raise")
    )
    with pytest.raises(RuntimeError, match="joint selection"):
        strict.select(_frame(), "Y", ["X"])

    permissive = ARDLOrderSelector(
        ARDLSelectionConfig(target_lag_strategy="ardl_select_order_joint", selection_failure_policy="fallback")
    )
    table, _ = permissive.select(_frame(), "Y", ["X"])
    target_rows = table[table["feature"].eq("__target_lag__")]
    assert target_rows["target_lag_selection_fallback"].all()
    assert target_rows["error"].str.contains("selection failed").all()


def test_ardl_candidate_ic_uses_common_sample() -> None:
    selector = ARDLOrderSelector(
        ARDLSelectionConfig(
            max_target_lag=4,
            max_exog_lag=3,
            target_lag_strategy="fixed",
            fixed_target_lags=(1,),
            top_n=3,
        )
    )
    table, _ = selector.select(_frame(), "Y", ["X"])
    candidates = table[table["feature"].eq("X")]
    assert candidates["nobs"].nunique() == 1
    assert candidates["hold_back"].nunique() == 1


def test_fixed_exogenous_lags_bypass_ardl_screening() -> None:
    selector = ARDLOrderSelector(
        ARDLSelectionConfig(
            max_target_lag=3,
            max_exog_lag=3,
            target_lag_strategy="fixed",
            fixed_target_lags=(1,),
            exog_lag_strategy="fixed",
            fixed_exog_lags=(1, 3),
        )
    )
    table, specs = selector.select(_frame(), "Y", ["X"])
    assert len(specs) == 1
    assert specs[0].target_lags == (1,)
    assert specs[0].exog_lags == {"X": (1, 3)}
    assert set(table[table["feature"].eq("X")]["exog_lag"]) == {1, 3}


def test_aic_variable_blocks_generate_all_nonempty_subsets() -> None:
    frame = _frame().assign(X2=lambda data: data["X"] * 0.4 + 0.2)
    selector = ARDLOrderSelector(
        ARDLSelectionConfig(
            max_target_lag=3,
            max_exog_lag=2,
            target_lag_strategy="fixed",
            fixed_target_lags=(1,),
            exog_lag_strategy="ardl_block_subsets",
        )
    )
    table, specs = selector.select(frame, "Y", ["X", "X2"])
    assert len(specs) == 3
    included_sets = {
        frozenset(feature for feature, lags in spec.exog_lags.items() if lags)
        for spec in specs
    }
    assert included_sets == {
        frozenset({"X"}),
        frozenset({"X2"}),
        frozenset({"X", "X2"}),
    }
    combo_rows = table[table["feature"].eq("__ffnn_combo__")]
    assert len(combo_rows) == 3
    assert set(combo_rows["combo_source"]) == {"aic_rank1_variable_block_subset"}
    assert set(combo_rows["included_block_count"]) == {1, 2}
    assert all(
        lags == tuple(range(1, max(lags) + 1))
        for spec in specs
        for lags in spec.exog_lags.values()
        if lags
    )


def test_dm_outputs_holm_adjustment_and_block_bootstrap_interval() -> None:
    dates = pd.date_range("2020-01-01", periods=12, freq="MS")
    actual = np.linspace(1.0, 3.0, len(dates))
    rows = []
    for pipeline, offset in (("better", 0.05), ("middle", 0.25), ("worse", 0.75)):
        for seed in (1, 2):
            for date, value in zip(dates, actual, strict=True):
                rows.append(
                    {
                        "date": date,
                        "split": "test",
                        "actual": value,
                        "predicted": value + offset + seed * 0.001,
                        "pipeline": pipeline,
                        "seed": seed,
                    }
                )
    result = pairwise_dm_tests(pd.DataFrame(rows))
    assert len(result) == 3
    assert result["holm_adjusted_pvalue"].between(0.0, 1.0).all()
    assert result["paired_mean_loss_difference_ci95_lower"].notna().all()
    assert np.allclose(holm_adjust(np.array([0.01, 0.04, 0.03])), [0.03, 0.06, 0.06])


def test_paper_orchestrator_builds_benchmark_ablation_and_dm_tables(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from vmd_ardl_ffnn import paper_experiments as paper

    dates = pd.date_range("2020-01-01", periods=8, freq="MS")

    def frames(pipeline: str, offset: float):
        forecasts = pd.DataFrame(
            {
                "date": list(dates) * 2,
                "split": ["validation"] * len(dates) + ["test"] * len(dates),
                "actual": list(np.arange(len(dates), dtype=float) + 10.0) * 2,
            }
        )
        forecasts["predicted"] = forecasts["actual"] + offset
        forecasts["pipeline"] = pipeline
        forecasts["scale"] = "level"
        metrics = pd.DataFrame(
            [
                {
                    "model": pipeline,
                    "pipeline": pipeline,
                    "target": "Y",
                    "split": split,
                    "scale": "level",
                    "RMSE": offset,
                    "MAE": offset,
                    "MAPE": offset,
                    "Directional_Accuracy": 100.0,
                }
                for split in ("validation", "test")
            ]
        )
        return metrics, forecasts

    def fake_naive(csv_path, config, *, lag, model_name):
        metrics, forecasts = frames(Path(config.output_dir).name, lag / 100.0)
        return {"metrics": metrics, "forecasts": forecasts}

    def fake_ardl(csv_path, config):
        metrics, forecasts = frames("standalone_ardl", 0.04)
        return {"metrics": metrics, "forecasts": forecasts}

    def fake_neural(csv_path, config, *, pipeline, definition, seed, output_dir):
        metrics, forecasts = frames(pipeline, 0.02 + seed / 1000.0)
        metrics["seed"] = seed
        forecasts["seed"] = seed
        result = {
            "ardl_orders": pd.DataFrame({"feature": ["__ffnn_combo__"]}),
            "best_component_models": pd.DataFrame(
                {
                    "component": ["raw"],
                    "target_lags": [(1,)],
                    "exog_lags": [{"X": (1,)}],
                    "lag_spec": ["target:[1] | X:[1]"],
                }
            ),
            "search_results": pd.DataFrame({"candidate_id": [1, 2]}),
        }
        return result, metrics, forecasts

    def fake_block_subset(csv_path, config, *, seeds, output_dir, search_strategy):
        assert search_strategy == "full_grid"
        metric_frames = []
        forecast_frames = []
        best_rows = []
        for seed in seeds:
            metrics, forecasts = frames(
                paper.BLOCK_SUBSET_PIPELINE,
                0.015 + seed / 1000.0,
            )
            metrics["model"] = paper.BLOCK_SUBSET_MODEL
            metrics["seed"] = seed
            forecasts["seed"] = seed
            metric_frames.append(metrics)
            forecast_frames.append(forecasts)
            best_rows.append(
                {
                    "component": "raw",
                    "seed": seed,
                    "target_lags": (1,),
                    "exog_lags": {"X": (1,)},
                    "lag_spec": "target:[1] | X:[1]",
                }
            )
        block_config = ExperimentConfig(
            data=config.data,
            vmd=config.vmd,
            ardl=ARDLSelectionConfig(
                max_target_lag=config.ardl.max_target_lag,
                max_exog_lag=config.ardl.max_exog_lag,
                exog_lag_strategy="ardl_block_subsets",
            ),
            ffnn=replace(config.ffnn, seed_grid=tuple(seeds), search_strategy="full_grid"),
            output_dir=output_dir,
        )
        result = {
            "ardl_orders": pd.DataFrame({"feature": ["__ffnn_combo__"]}),
            "best_component_models": pd.DataFrame(best_rows),
            "component_search_results": pd.DataFrame({"candidate_id": [1, 2]}),
        }
        return (
            result,
            pd.concat(metric_frames, ignore_index=True),
            pd.concat(forecast_frames, ignore_index=True),
            block_config,
        )

    monkeypatch.setattr(paper, "run_naive_baseline", fake_naive)
    monkeypatch.setattr(paper, "run_linear_ardl_baseline", fake_ardl)
    monkeypatch.setattr(paper, "_run_neural_variant", fake_neural)
    monkeypatch.setattr(paper, "_run_block_subset_mean_seed_variant", fake_block_subset)
    csv_path = tmp_path / "study.csv"
    pd.DataFrame(
        {
            "TIME_PERIOD": pd.date_range("2018-01-01", periods=24, freq="MS"),
            "Y": np.linspace(10.0, 20.0, 24),
            "X": np.linspace(5.0, 9.0, 24),
        }
    ).to_csv(csv_path, index=False)
    result = paper.run_paper_experiments(
        csv_path,
        ExperimentConfig(
            data=DataConfig(target="Y", features=("X",), log_transform=False),
            output_dir=tmp_path / "paper",
        ),
        seeds=(1, 2),
        include_varnn=False,
    )
    assert set(result["ablation_metrics"]["ablation_id"]) == {"A0", "A1", "A2", "A3", "A4", "A5"}
    assert not result["dm_tests"].empty
    assert not result["lag_selection"].empty
    assert paper.BLOCK_SUBSET_PIPELINE in set(result["comparison_metrics"]["pipeline"])
    assert paper.BLOCK_SUBSET_PIPELINE not in set(result["ablation_metrics"]["pipeline"])
    block_setting = result["settings"].query(
        "pipeline == @paper.BLOCK_SUBSET_PIPELINE"
    ).iloc[0]
    assert block_setting["final_selection_scale"] == "reconstructed_level"
    assert block_setting["selection_seed_aggregation"] == (
        "mean_reconstructed_level_validation_across_reporting_seeds"
    )
    assert (tmp_path / "paper" / "paper_experiment_settings.csv").exists()
    assert (tmp_path / "paper" / "efficiency_summary.csv").exists()


def test_block_proposed_orchestrator_builds_shared_benchmark_and_a0_a7(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from vmd_ardl_ffnn import block_proposed_experiments as study

    dates = pd.date_range("2020-01-01", periods=8, freq="MS")
    seeds = (1, 2)

    def frames(pipeline: str, model: str, offset: float):
        metric_rows = []
        forecast_rows = []
        for seed in seeds:
            for split in ("validation", "test"):
                metric_rows.append(
                    {
                        "model": model,
                        "pipeline": pipeline,
                        "target": "Y",
                        "split": split,
                        "scale": "level",
                        "RMSE": offset + seed / 1000.0,
                        "MAE": offset,
                        "MAPE": offset,
                        "Directional_Accuracy": 100.0,
                        "seed": seed,
                    }
                )
                for date_index, date in enumerate(dates):
                    actual = float(10 + date_index)
                    forecast_rows.append(
                        {
                            "date": date,
                            "split": split,
                            "actual": actual,
                            "predicted": actual + offset + seed / 1000.0,
                            "pipeline": pipeline,
                            "scale": "level",
                            "seed": seed,
                        }
                    )
        return pd.DataFrame(metric_rows), pd.DataFrame(forecast_rows)

    def fake_result() -> dict[str, pd.DataFrame]:
        return {
            "ardl_orders": pd.DataFrame({"feature": ["__ffnn_combo__"]}),
            "best_component_models": pd.DataFrame(
                {
                    "component": ["VMD1", "VMD1"],
                    "seed": list(seeds),
                    "target_lags": [(1,), (1,)],
                    "exog_lags": [{"X": (1,)}, {"X": (1,)}],
                    "lag_spec": ["target:[1] | X:[1]"] * 2,
                }
            ),
            "component_search_results": pd.DataFrame({"candidate_id": [1, 2]}),
            "lag_screen_fold_metrics": pd.DataFrame(),
            "lag_screen_stability": pd.DataFrame(),
            "training_components": {},
        }

    def fake_naive(csv_path, config, *, lag, model_name):
        pipeline = Path(config.output_dir).name
        metrics, forecasts = frames(pipeline, model_name, lag / 100.0)
        return {
            "metrics": metrics.drop_duplicates(["split"]).drop(columns=["seed"]),
            "forecasts": forecasts.query("seed == 1").drop(columns=["seed"]),
        }

    def fake_ardl(csv_path, config):
        metrics, forecasts = frames("standalone_ardl", "Standalone_ARDL", 0.04)
        return {
            "metrics": metrics.drop_duplicates(["split"]).drop(columns=["seed"]),
            "forecasts": forecasts.query("seed == 1").drop(columns=["seed"]),
        }

    def fake_shared_vmd(csv_path, config, **kwargs):
        metrics, forecasts = frames(kwargs["pipeline"], kwargs["model"], 0.02)
        return fake_result(), metrics, forecasts, replace(config, output_dir=kwargs["output_dir"])

    def fake_shared_no_vmd(csv_path, config, **kwargs):
        metrics, forecasts = frames(kwargs["pipeline"], kwargs["model"], 0.025)
        return fake_result(), metrics, forecasts, replace(config, output_dir=kwargs["output_dir"])

    def fake_shared_block(csv_path, config, **kwargs):
        metrics, forecasts = frames(kwargs["pipeline"], kwargs["model"], 0.015)
        return fake_result(), metrics, forecasts, replace(config, output_dir=kwargs["output_dir"])

    def fake_per_seed_block(csv_path, config, **kwargs):
        metrics, forecasts = frames(kwargs["pipeline"], kwargs["model"], 0.018)
        results = [fake_result(), fake_result()]
        run_metadata = [
            (replace(config, output_dir=kwargs["output_dir"] / f"seed_{seed}"), 0.1)
            for seed in seeds
        ]
        return results, metrics, forecasts, run_metadata

    def unavailable_varnn(*args, **kwargs):
        raise ImportError("TensorFlow unavailable in test")

    monkeypatch.setattr(study, "run_naive_baseline", fake_naive)
    monkeypatch.setattr(study, "run_linear_ardl_baseline", fake_ardl)
    monkeypatch.setattr(study, "_run_shared_vmd_variant", fake_shared_vmd)
    monkeypatch.setattr(study, "_run_shared_no_vmd_variant", fake_shared_no_vmd)
    monkeypatch.setattr(study, "_run_shared_block_variant", fake_shared_block)
    monkeypatch.setattr(study, "_run_per_seed_block_variant", fake_per_seed_block)
    monkeypatch.setattr(study, "_run_varnn_shared_mean_seed", unavailable_varnn)
    monkeypatch.setattr(study, "write_paper_figures", lambda *args, **kwargs: None)

    csv_path = tmp_path / "study.csv"
    pd.DataFrame(
        {
            "TIME_PERIOD": pd.date_range("2018-01-01", periods=24, freq="MS"),
            "Y": np.linspace(10.0, 20.0, 24),
            "X": np.linspace(5.0, 9.0, 24),
        }
    ).to_csv(csv_path, index=False)
    result = study.run_block_proposed_experiments(
        csv_path,
        ExperimentConfig(
            data=DataConfig(target="Y", features=("X",), log_transform=False),
            output_dir=tmp_path / "block_proposed",
        ),
        seeds=seeds,
        include_varnn=True,
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
    assert study.PROPOSED_PIPELINE in set(result["comparison_metrics"]["pipeline"])
    assert study.SEQUENTIAL_PIPELINE in set(result["comparison_metrics"]["pipeline"])
    assert study.JOINT_SELECT_PIPELINE in set(
        result["comparison_metrics"]["pipeline"]
    )
    assert result["date_coverage"]["same_dates_for_all_models"].all()
    assert result["settings"]["ablation_schema_version"].eq(
        study.ABLATION_SCHEMA_VERSION
    ).all()
    varnn_setting = result["settings"].query("pipeline == 'varnn'").iloc[0]
    assert not bool(varnn_setting["enabled"])
    assert result["skipped_models"].iloc[0]["pipeline"] == "varnn"
    joint_setting = result["settings"].query(
        "pipeline == @study.JOINT_SELECT_PIPELINE"
    ).iloc[0]
    assert joint_setting["target_lag_strategy"] == "ardl_select_order_joint"
    assert joint_setting["exog_lag_strategy"] == "ardl_select_order_joint"
    assert joint_setting["selection_protocol"] == (
        "mean_reconstructed_level_validation_across_reporting_seeds"
    )
    assert (tmp_path / "block_proposed" / "ablation_metrics.csv").exists()
    assert (tmp_path / "block_proposed" / "benchmark_metrics_comparison.csv").exists()


def test_directional_accuracy_uses_previous_actual() -> None:
    metrics = evaluate_forecast([10.0, 12.0, 11.0], [10.0, 11.0, 11.5])
    assert metrics["Directional_Accuracy"] == 100.0


def test_ffnn_restores_weights_that_match_best_recorded_loss() -> None:
    x = np.arange(20, dtype=float).reshape(10, 2)
    y = np.linspace(0.0, 1.0, 10)
    model = NumpyFFNNRegressor(hidden_layer_sizes=(3,), max_iter=20, seed=3).fit(x, y)
    scaled_x = model.x_scaler_.transform(x)
    scaled_y = model.y_scaler_.transform(y)
    restored_loss = model._loss(model._forward(scaled_x)[0][-1], scaled_y)
    assert restored_loss == pytest.approx(min(model.loss_curve_))


def test_loader_rejects_missing_calendar_period(tmp_path: Path) -> None:
    path = tmp_path / "gapped.csv"
    pd.DataFrame(
        {
            "TIME_PERIOD": ["2020-01-01", "2020-03-01"],
            "Y": [1.0, 2.0],
            "X": [1.0, 2.0],
        }
    ).to_csv(path, index=False)
    loader = GenericDataLoader(DataConfig(target="Y", features=("X",), log_transform=False))
    with pytest.raises(ValueError, match="missing 1 expected MS periods"):
        loader.load_csv(path)


def test_run_scoped_csv_is_overwritten_not_appended(tmp_path: Path) -> None:
    run_dir = prepare_run_directory(tmp_path, "vmd", {"config": 1}, run_id="run-1")
    output = run_dir / "metrics.csv"
    write_csv(pd.DataFrame({"value": [1, 2]}), output)
    write_csv(pd.DataFrame({"value": [3]}), output)
    assert pd.read_csv(output)["value"].tolist() == [3]
    manifest = write_manifest(
        run_dir,
        pipeline="vmd",
        dataset=tmp_path / "data.csv",
        config={"config": 1},
        status="success",
    )
    assert '"run_id": "run-1"' in manifest.read_text(encoding="utf-8")


def test_var_lag_search_respects_upper_bound() -> None:
    lag, _, searched, _ = _choose_var_lag(_frame(120), min_lag=1, max_lag=2)
    assert searched <= 2
    assert lag <= 2
