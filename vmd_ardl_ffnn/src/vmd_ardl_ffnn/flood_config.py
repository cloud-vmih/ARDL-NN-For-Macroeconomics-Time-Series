"""Configuration objects for the Son La flood-proxy forecasting pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FloodDataConfig:
    """Column contract shared by observations, NWP forecasts and basin metadata."""

    zone_col: str = "HYBAS_ID"
    time_col: str = "datetime"
    issue_time_col: str = "issue_time"
    valid_time_col: str = "valid_time"
    target_col: str = "surface_runoff_mm_h"
    exogenous_cols: tuple[str, ...] = (
        "rain",
        "temperature_2m",
        "soil_moisture_1_to_3cm",
    )
    forecast_exogenous_cols: tuple[str, ...] = tuple()
    static_numeric_cols: tuple[str, ...] = (
        "SUB_AREA",
        "UP_AREA",
        "ORDER",
        "min_elevation_m",
        "max_elevation_m",
        "mean_elevation_m",
    )
    static_categorical_cols: tuple[str, ...] = ("soil_name",)
    timezone: str = "Asia/Ho_Chi_Minh"
    require_hourly_observations: bool = True

    @property
    def nwp_cols(self) -> tuple[str, ...]:
        return self.forecast_exogenous_cols or self.exogenous_cols

    def __post_init__(self) -> None:
        names = (
            self.zone_col,
            self.time_col,
            self.issue_time_col,
            self.valid_time_col,
            self.target_col,
        )
        if any(not value for value in names):
            raise ValueError("Flood data column names must not be empty.")
        if not self.exogenous_cols:
            raise ValueError("At least one exogenous column is required.")
        if len(set(self.exogenous_cols)) != len(self.exogenous_cols):
            raise ValueError("exogenous_cols must not contain duplicates.")
        if len(set(self.nwp_cols)) != len(self.nwp_cols):
            raise ValueError("forecast_exogenous_cols must not contain duplicates.")


@dataclass(frozen=True)
class FloodARDLConfig:
    """Sequential AIC/BIC selection bounds for the per-zone ARDL baseline."""

    max_target_lag: int = 24
    max_exog_lag: int = 24
    min_train_rows: int = 24 * 30
    ic: str = "aic"
    include_exog_lag_zero: bool = True

    def __post_init__(self) -> None:
        if self.max_target_lag < 1 or self.max_exog_lag < 0:
            raise ValueError("ARDL lag bounds must be non-negative and target lag >= 1.")
        if self.min_train_rows <= self.max_target_lag + self.max_exog_lag + 2:
            raise ValueError("min_train_rows is too small for the requested lag bounds.")
        if self.ic.lower() not in {"aic", "bic"}:
            raise ValueError("FloodARDLConfig.ic must be 'aic' or 'bic'.")


@dataclass(frozen=True)
class FloodFFNNConfig:
    """Global residual FFNN settings."""

    hidden_layer_sizes: tuple[int, ...] = (32, 16)
    alpha: float = 1e-3
    activation: str = "tanh"
    learning_rate: float = 1e-3
    max_iter: int = 300
    seed: int = 42
    issue_stride_hours: int = 6
    max_horizon_hours: int = 24
    max_residual_samples: int = 50_000
    fixed_target_lags: tuple[int, ...] = (1, 24)

    def __post_init__(self) -> None:
        if not self.hidden_layer_sizes or any(v < 1 for v in self.hidden_layer_sizes):
            raise ValueError("hidden_layer_sizes must contain positive widths.")
        if self.alpha < 0 or self.learning_rate <= 0 or self.max_iter < 1:
            raise ValueError("Invalid FFNN optimization settings.")
        if self.issue_stride_hours < 1 or self.max_horizon_hours < 1:
            raise ValueError("Forecast stride and horizon must be >= 1 hour.")
        if self.max_residual_samples < 100:
            raise ValueError("max_residual_samples must be at least 100.")
        if not self.fixed_target_lags or any(lag < 1 for lag in self.fixed_target_lags):
            raise ValueError("fixed_target_lags must contain positive lags.")


@dataclass(frozen=True)
class FloodRoutingConfig:
    """Linear-reservoir routing and proxy-warning settings."""

    downstream_col: str = "NEXT_DOWN"
    area_col: str = "SUB_AREA"
    travel_hours_col: str = "travel_hours"
    default_travel_hours: float = 3.0
    runoff_quantiles: tuple[float, float, float] = (0.90, 0.95, 0.99)
    river_threshold_cols: tuple[str, str, str] = (
        "q_1_5yr_m3s",
        "q_2yr_m3s",
        "q_5yr_m3s",
    )

    def __post_init__(self) -> None:
        if self.default_travel_hours <= 0:
            raise ValueError("default_travel_hours must be > 0.")
        if tuple(sorted(self.runoff_quantiles)) != self.runoff_quantiles:
            raise ValueError("runoff_quantiles must be ascending.")
        if any(not 0 < value < 1 for value in self.runoff_quantiles):
            raise ValueError("runoff_quantiles must lie in (0, 1).")
