from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vmd_ardl_ffnn.flood_ardl import SequentialARDL
from vmd_ardl_ffnn.flood_config import (
    FloodARDLConfig,
    FloodDataConfig,
    FloodFFNNConfig,
    FloodRoutingConfig,
)
from vmd_ardl_ffnn.flood_data import FloodPanelLoader
from vmd_ardl_ffnn.flood_pipeline import run_flood_experiment
from vmd_ardl_ffnn.flood_preparation import (
    area_weighted_basin_aggregation,
    deaccumulate_forecast_fields,
    effective_grid_coverage,
    select_upstream_network,
)
from vmd_ardl_ffnn.flood_routing import BasinRouter, RunoffThresholds


def synthetic_flood_panel(n: int = 240) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(13)
    times = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    records: list[dict[str, object]] = []
    zones = ("101", "102", "103")
    for zone_number, zone in enumerate(zones):
        wet = rng.random(n) < 0.24
        rain = rng.gamma(1.4, 0.5, n) * wet
        temperature = 22.0 + 3.0 * np.sin(np.arange(n) * 2 * np.pi / 24)
        soil = np.clip(0.24 + np.cumsum(rain * 0.0015 - 0.0001), 0.15, 0.48)
        runoff = np.zeros(n, dtype=float)
        for position in range(1, n):
            nonlinear = 0.08 * rain[position] ** 2
            runoff[position] = max(
                0.0,
                0.52 * runoff[position - 1]
                + 0.30 * rain[position]
                + nonlinear
                + 0.08 * soil[position]
                + zone_number * 0.01,
            )
        for position, timestamp in enumerate(times):
            records.append(
                {
                    "HYBAS_ID": zone,
                    "datetime": timestamp,
                    "rain": rain[position],
                    "temperature_2m": temperature[position],
                    "soil_moisture_1_to_3cm": soil[position],
                    "surface_runoff_mm_h": runoff[position],
                }
            )
    observations = pd.DataFrame.from_records(records)
    static = pd.DataFrame(
        {
            "HYBAS_ID": list(zones),
            "NEXT_DOWN": ["103", "103", "0"],
            "SUB_AREA": [100.0, 125.0, 310.0],
            "UP_AREA": [100.0, 125.0, 535.0],
            "ORDER": [1, 1, 2],
            "min_elevation_m": [320.0, 410.0, 180.0],
            "max_elevation_m": [1250.0, 1400.0, 900.0],
            "mean_elevation_m": [700.0, 820.0, 480.0],
            "soil_name": ["loam", "clay loam", "loam"],
        }
    )
    return observations, static


def test_panel_loader_preserves_duplicate_times_across_zones(tmp_path: Path) -> None:
    observations, static = synthetic_flood_panel(48)
    observation_path = tmp_path / "observations.csv"
    static_path = tmp_path / "static.csv"
    observations.to_csv(observation_path, index=False)
    static.to_csv(static_path, index=False)
    config = FloodDataConfig(require_hourly_observations=True)
    loader = FloodPanelLoader(config)
    loaded_observations = loader.load_observations(observation_path)
    loaded_static = loader.load_static(static_path)
    audit = loader.audit(loaded_observations, loaded_static)
    assert audit.observation_rows == 3 * 48
    assert audit.unique_times == 48
    assert audit.duplicate_zone_times == 0
    assert audit.zones_with_non_hourly_gaps == 0


def test_forecast_loader_rejects_issue_time_leakage(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "HYBAS_ID": ["101"],
            "issue_time": ["2024-01-01T01:00:00Z"],
            "valid_time": ["2024-01-01T01:00:00Z"],
            "rain": [1.0],
            "temperature_2m": [20.0],
            "soil_moisture_1_to_3cm": [0.3],
        }
    )
    path = tmp_path / "forecast.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="later than issue_time"):
        FloodPanelLoader(FloodDataConfig()).load_forecasts(path)


def test_sequential_ardl_forecasts_a_contiguous_future() -> None:
    observations, _ = synthetic_flood_panel(180)
    zone = observations.loc[observations["HYBAS_ID"].eq("101")].set_index("datetime")
    columns = [
        "surface_runoff_mm_h",
        "rain",
        "temperature_2m",
        "soil_moisture_1_to_3cm",
    ]
    train = zone.iloc[:150][columns]
    model = SequentialARDL(
        "surface_runoff_mm_h",
        ("rain", "temperature_2m", "soil_moisture_1_to_3cm"),
        FloodARDLConfig(max_target_lag=3, max_exog_lag=2, min_train_rows=100),
    ).fit(train)
    future = zone.iloc[150:156][["rain", "temperature_2m", "soil_moisture_1_to_3cm"]]
    forecast = model.forecast(train, future)
    assert len(forecast) == 6
    assert np.isfinite(forecast).all()
    assert model.target_lags_
    assert set(model.exog_lags_) == {"rain", "temperature_2m", "soil_moisture_1_to_3cm"}


def test_router_conserves_water_and_rejects_cycles() -> None:
    _, static = synthetic_flood_panel(48)
    config = FloodDataConfig()
    issue = pd.Timestamp("2024-01-01T00:00:00Z")
    rows = []
    for zone in static["HYBAS_ID"]:
        for lead in range(1, 7):
            rows.append(
                {
                    "HYBAS_ID": zone,
                    "issue_time": issue,
                    "valid_time": issue + pd.Timedelta(hours=lead),
                    "lead_hour": lead,
                    "runoff_prediction_mm_h": 1.0,
                }
            )
    routed, audit = BasinRouter(
        static, config, FloodRoutingConfig(default_travel_hours=2.0)
    ).route(pd.DataFrame(rows))
    assert len(routed) == 18
    assert audit["relative_mass_balance_error"].max() < 1e-12
    assert routed.loc[routed["HYBAS_ID"].eq("103"), "routing_inflow_m3_s"].max() > routed.loc[
        routed["HYBAS_ID"].eq("103"), "lateral_discharge_m3_s"
    ].max()

    cyclic = static.copy()
    cyclic.loc[cyclic["HYBAS_ID"].eq("103"), "NEXT_DOWN"] = "101"
    with pytest.raises(ValueError, match="cycle"):
        BasinRouter(cyclic, config)


def test_source_preparation_selects_upstream_and_preserves_weighted_depth() -> None:
    _, static = synthetic_flood_panel(48)
    selected = select_upstream_network(static, ["103"])
    assert set(selected["HYBAS_ID"]) == {"101", "102", "103"}
    assert set(selected.loc[selected["HYBAS_ID"].ne("103"), "selection_reason"]) == {
        "upstream_contributor"
    }

    weights = pd.DataFrame(
        {
            "HYBAS_ID": ["101", "101", "101"],
            "grid_id": ["a", "b", "c"],
            "overlap_weight": [0.5, 0.3, 0.2],
        }
    )
    coverage = effective_grid_coverage(weights, minimum_effective_cells=2.5)
    assert coverage.loc[0, "independent_resolution_supported"]
    gridded = pd.DataFrame(
        {
            "grid_id": ["a", "b", "c"],
            "valid_time": [pd.Timestamp("2024-01-01T01:00:00Z")] * 3,
            "rain_mm": [1.0, 2.0, 4.0],
        }
    )
    aggregated = area_weighted_basin_aggregation(
        gridded, weights, value_columns=("rain_mm",)
    )
    assert aggregated.loc[0, "rain_mm"] == pytest.approx(1.9)


def test_deaccumulation_never_crosses_forecast_cycles() -> None:
    frame = pd.DataFrame(
        {
            "HYBAS_ID": ["101"] * 4,
            "issue_time": [
                pd.Timestamp("2024-01-01T00:00:00Z"),
                pd.Timestamp("2024-01-01T00:00:00Z"),
                pd.Timestamp("2024-01-01T06:00:00Z"),
                pd.Timestamp("2024-01-01T06:00:00Z"),
            ],
            "valid_time": pd.to_datetime(
                [
                    "2024-01-01T01:00:00Z",
                    "2024-01-01T02:00:00Z",
                    "2024-01-01T07:00:00Z",
                    "2024-01-01T08:00:00Z",
                ]
            ),
            "total_precipitation_m": [0.001, 0.003, 0.004, 0.005],
        }
    )
    result = deaccumulate_forecast_fields(
        frame, {"total_precipitation_m": "rain_step_m"}
    )
    assert result["rain_step_m"].tolist() == pytest.approx([0.001, 0.002, 0.004, 0.001])


def test_zero_runoff_climatology_remains_green() -> None:
    config = FloodDataConfig()
    observations = pd.DataFrame(
        {
            "HYBAS_ID": ["101"] * 48,
            "datetime": pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC"),
            "surface_runoff_mm_h": np.zeros(48),
        }
    )
    thresholds = RunoffThresholds(config, FloodRoutingConfig()).fit(observations)
    levels = thresholds.levels(
        observations["HYBAS_ID"],
        observations["datetime"],
        observations["surface_runoff_mm_h"],
    )
    assert np.array_equal(levels, np.zeros(48, dtype=int))


def test_end_to_end_oracle_exogenous_pipeline(tmp_path: Path) -> None:
    observations, static = synthetic_flood_panel(240)
    observation_path = tmp_path / "observations.csv"
    static_path = tmp_path / "static.csv"
    observations.to_csv(observation_path, index=False)
    static.to_csv(static_path, index=False)
    times = pd.DatetimeIndex(sorted(observations["datetime"].unique()))
    run_dir = run_flood_experiment(
        static_path=static_path,
        observation_path=observation_path,
        forecast_path=None,
        output_dir=tmp_path / "results",
        train_end=times[159],
        validation_end=times[199],
        test_end=times[229],
        data_config=FloodDataConfig(),
        ardl_config=FloodARDLConfig(
            max_target_lag=3,
            max_exog_lag=2,
            min_train_rows=100,
        ),
        ffnn_config=FloodFFNNConfig(
            hidden_layer_sizes=(8,),
            max_iter=20,
            max_horizon_hours=6,
            issue_stride_hours=6,
            max_residual_samples=5_000,
        ),
        routing_config=FloodRoutingConfig(default_travel_hours=2.0),
        oracle_exogenous=True,
    )
    expected = {
        "manifest.json",
        "data_audit.csv",
        "ardl_lag_selection.csv",
        "runoff_forecasts.csv",
        "routed_proxy_warnings.csv",
        "routing_mass_balance.csv",
        "runoff_metrics.csv",
        "proxy_alert_metrics.csv",
        "acceptance_summary.csv",
    }
    assert expected.issubset({path.name for path in run_dir.iterdir()})
    forecast = pd.read_csv(run_dir / "runoff_forecasts.csv")
    assert {"validation", "test"} == set(forecast["split"])
    assert (forecast["runoff_prediction_mm_h"] >= 0).all()
    routing_audit = pd.read_csv(run_dir / "routing_mass_balance.csv")
    assert routing_audit["relative_mass_balance_error"].max() < 1e-12
    warnings = pd.read_csv(run_dir / "routed_proxy_warnings.csv")
    assert set(warnings["warning_type"]) == {"model_proxy"}
