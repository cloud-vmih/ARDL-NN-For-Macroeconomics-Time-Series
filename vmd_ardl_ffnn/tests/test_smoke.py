from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from vmd_ardl_ffnn.config import ARDLSelectionConfig, DataConfig, ExperimentConfig, FFNNConfig, VMDConfig
from vmd_ardl_ffnn.decomposition import VariationalModeDecomposer
from vmd_ardl_ffnn.experiment import VMDARDLFFNNExperiment


def synthetic_data(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(2)
    x1 = np.exp(np.cumsum(rng.normal(0.0, 0.02, n))) + 2.0
    x2 = np.exp(np.cumsum(rng.normal(0.0, 0.01, n))) + 3.0
    y = np.exp(2.0 + 0.02 * np.arange(n) / n + 0.15 * np.log(x1) - 0.05 * np.log(x2) + rng.normal(0, 0.01, n))
    return pd.DataFrame({"TIME_PERIOD": pd.date_range("2020-01-01", periods=n, freq="MS"), "Y": y, "X1": x1, "X2": x2})


def test_vmd_decomposition_reconstructs_shape() -> None:
    series = pd.Series(np.sin(np.linspace(0, 10, 80)) + 0.1 * np.arange(80), name="x")
    levels = VariationalModeDecomposer(VMDConfig(modes=2, max_iterations=50)).decompose(series)
    assert set(levels) == {"VMD1", "VMD2", "RES"}
    assert all(len(component) == len(series) for component in levels.values())


def test_experiment_smoke_run() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "data.csv"
        synthetic_data().to_csv(path, index=False)
        config = ExperimentConfig(
            data=DataConfig(target="Y", features=("X1", "X2")),
            vmd=VMDConfig(modes=2, max_iterations=50),
            ardl=ARDLSelectionConfig(max_target_lag=3, max_exog_lag=2, top_n=2),
            ffnn=FFNNConfig(hidden_units_candidates=(3,), alpha_grid=(1e-3,), seed_grid=(1,), max_iter=20),
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
        assert set(result["component_forecasts"]["component"]) == {"VMD1", "VMD2", "RES"}
        assert int(result["best_component_models"].iloc[0]["HR"]) == 3
