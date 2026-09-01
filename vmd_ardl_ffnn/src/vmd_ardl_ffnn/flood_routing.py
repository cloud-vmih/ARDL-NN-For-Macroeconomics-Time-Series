"""Mass-conserving linear-reservoir routing and transparent proxy warnings."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .flood_config import FloodDataConfig, FloodRoutingConfig


LEVEL_NAMES = np.array(["green", "yellow", "orange", "red"], dtype=object)


def _id(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


@dataclass
class RunoffThresholds:
    """Month-specific train-only P90/P95/P99 thresholds with zone/global fallbacks."""

    data_config: FloodDataConfig
    routing_config: FloodRoutingConfig
    monthly_: dict[tuple[str, int], tuple[float, float, float]] = field(default_factory=dict)
    zone_: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    global_: tuple[float, float, float] | None = None

    def fit(self, observations: pd.DataFrame) -> "RunoffThresholds":
        zone_col = self.data_config.zone_col
        time_col = self.data_config.time_col
        target = self.data_config.target_col
        quantiles = list(self.routing_config.runoff_quantiles)
        frame = observations[[zone_col, time_col, target]].copy()
        local_time = frame[time_col].dt.tz_convert(self.data_config.timezone)
        frame["month"] = local_time.dt.month
        for (zone, month), group in frame.groupby([zone_col, "month"], observed=True):
            self.monthly_[(str(zone), int(month))] = tuple(
                float(v) for v in group[target].quantile(quantiles).to_numpy()
            )
        for zone, group in frame.groupby(zone_col, observed=True):
            self.zone_[str(zone)] = tuple(
                float(v) for v in group[target].quantile(quantiles).to_numpy()
            )
        self.global_ = tuple(float(v) for v in frame[target].quantile(quantiles).to_numpy())
        return self

    def levels(self, zones: pd.Series, times: pd.Series, values: pd.Series) -> np.ndarray:
        if self.global_ is None:
            raise RuntimeError("Call fit before levels.")
        local_month = times.dt.tz_convert(self.data_config.timezone).dt.month
        output = np.zeros(len(values), dtype=int)
        for index, (zone, month, value) in enumerate(zip(zones, local_month, values)):
            thresholds = self.monthly_.get(
                (str(zone), int(month)), self.zone_.get(str(zone), self.global_)
            )
            # Strict exceedance keeps an all-zero/dry climatology at green when value is zero.
            output[index] = int(np.searchsorted(np.asarray(thresholds), float(value), side="left"))
        return output

    def table(self) -> pd.DataFrame:
        rows = [
            {
                self.data_config.zone_col: zone,
                "month": month,
                "runoff_p90_mm_h": values[0],
                "runoff_p95_mm_h": values[1],
                "runoff_p99_mm_h": values[2],
            }
            for (zone, month), values in sorted(self.monthly_.items())
        ]
        return pd.DataFrame.from_records(rows)


class BasinRouter:
    """Route lateral runoff through NEXT_DOWN using one linear reservoir per zone."""

    def __init__(
        self,
        static: pd.DataFrame,
        data_config: FloodDataConfig,
        routing_config: FloodRoutingConfig | None = None,
    ) -> None:
        self.data_config = data_config
        self.config = routing_config or FloodRoutingConfig()
        zone_col = data_config.zone_col
        required = [zone_col, self.config.downstream_col, self.config.area_col]
        missing = [column for column in required if column not in static]
        if missing:
            raise ValueError(f"Routing static table is missing columns: {missing}")
        self.static = static.copy()
        self.static[zone_col] = self.static[zone_col].map(_id)
        self.static[self.config.downstream_col] = self.static[self.config.downstream_col].map(_id)
        self.static[self.config.area_col] = pd.to_numeric(
            self.static[self.config.area_col], errors="raise"
        )
        if (self.static[self.config.area_col] <= 0).any():
            raise ValueError("Routing areas must be positive.")
        self.static = self.static.set_index(zone_col, drop=False)
        self.zones = tuple(self.static.index.astype(str))
        zone_set = set(self.zones)
        self.downstream = {
            zone: downstream
            for zone, downstream in self.static[self.config.downstream_col].items()
            if downstream in zone_set and downstream != zone
        }
        self.order = self._topological_order()

    def route(self, forecasts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        required = [
            self.data_config.zone_col,
            self.data_config.issue_time_col,
            self.data_config.valid_time_col,
            "runoff_prediction_mm_h",
        ]
        missing = [column for column in required if column not in forecasts]
        if missing:
            raise ValueError(f"Runoff forecast table is missing routing columns: {missing}")
        outputs: list[pd.DataFrame] = []
        audits: list[dict[str, object]] = []
        issue_col = self.data_config.issue_time_col
        for issue, group in forecasts.groupby(issue_col, sort=True, observed=True):
            routed, audit = self._route_issue(pd.Timestamp(issue), group)
            outputs.append(routed)
            audits.append(audit)
        return (
            pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame(),
            pd.DataFrame.from_records(audits),
        )

    def _route_issue(
        self, issue: pd.Timestamp, forecasts: pd.DataFrame
    ) -> tuple[pd.DataFrame, dict[str, object]]:
        zone_col = self.data_config.zone_col
        issue_col = self.data_config.issue_time_col
        valid_col = self.data_config.valid_time_col
        forecasts = forecasts.copy()
        forecasts[zone_col] = forecasts[zone_col].map(_id)
        valid_times = pd.DatetimeIndex(sorted(forecasts[valid_col].unique()))
        if valid_times.empty:
            raise ValueError("Cannot route an empty forecast issue.")
        gaps = valid_times.to_series().diff().dropna() / pd.Timedelta(hours=1)
        if not gaps.eq(1.0).all():
            raise ValueError("Routing requires contiguous hourly forecast horizons.")
        n_steps = len(valid_times)
        runoff_by_zone: dict[str, np.ndarray] = {}
        source_rows: dict[tuple[str, pd.Timestamp], pd.Series] = {}
        for _, row in forecasts.iterrows():
            source_rows[(_id(row[zone_col]), pd.Timestamp(row[valid_col]))] = row
        for zone in self.zones:
            zone_rows = forecasts.loc[forecasts[zone_col].eq(zone)].set_index(valid_col)
            if zone_rows.empty:
                runoff_by_zone[zone] = np.zeros(n_steps, dtype=float)
            else:
                runoff_by_zone[zone] = (
                    zone_rows["runoff_prediction_mm_h"].reindex(valid_times).fillna(0.0).to_numpy(float)
                )

        upstream_contribution = {zone: np.zeros(n_steps, dtype=float) for zone in self.zones}
        outflows: dict[str, np.ndarray] = {}
        inflows: dict[str, np.ndarray] = {}
        storages: dict[str, np.ndarray] = {}
        total_lateral_volume = 0.0
        for zone in self.order:
            area = float(self.static.loc[zone, self.config.area_col])
            lateral = np.maximum(0.0, runoff_by_zone[zone]) * area / 3.6
            total_lateral_volume += float(lateral.sum() * 3600.0)
            inflow = lateral + upstream_contribution[zone]
            travel = self.config.default_travel_hours
            if self.config.travel_hours_col in self.static.columns:
                candidate = self.static.loc[zone, self.config.travel_hours_col]
                if pd.notna(candidate) and float(candidate) > 0:
                    travel = float(candidate)
            alpha = 1.0 - np.exp(-1.0 / travel)
            storage = 0.0
            outflow = np.zeros(n_steps, dtype=float)
            storage_path = np.zeros(n_steps, dtype=float)
            for step in range(n_steps):
                available = storage + inflow[step] * 3600.0
                released = available * alpha
                outflow[step] = released / 3600.0
                storage = available - released
                storage_path[step] = storage
            inflows[zone] = inflow
            outflows[zone] = outflow
            storages[zone] = storage_path
            downstream = self.downstream.get(zone)
            if downstream is not None:
                upstream_contribution[downstream] += outflow

        external_volume = 0.0
        storage_end = 0.0
        rows: list[dict[str, object]] = []
        for zone in self.zones:
            if zone not in self.downstream:
                external_volume += float(outflows[zone].sum() * 3600.0)
            storage_end += float(storages[zone][-1])
            for step, valid in enumerate(valid_times):
                source = source_rows.get((zone, valid))
                if source is None:
                    continue
                row = source.to_dict()
                row["lateral_discharge_m3_s"] = float(
                    max(0.0, row["runoff_prediction_mm_h"])
                    * float(self.static.loc[zone, self.config.area_col])
                    / 3.6
                )
                row["routing_inflow_m3_s"] = float(inflows[zone][step])
                row["routed_discharge_m3_s"] = float(outflows[zone][step])
                row["routing_storage_m3"] = float(storages[zone][step])
                rows.append(row)
        balance_error = total_lateral_volume - external_volume - storage_end
        relative_error = abs(balance_error) / max(total_lateral_volume, 1.0)
        audit = {
            issue_col: issue,
            "lateral_input_volume_m3": total_lateral_volume,
            "external_outflow_volume_m3": external_volume,
            "ending_storage_m3": storage_end,
            "mass_balance_error_m3": balance_error,
            "relative_mass_balance_error": relative_error,
        }
        return pd.DataFrame.from_records(rows), audit

    def _topological_order(self) -> tuple[str, ...]:
        indegree = {zone: 0 for zone in self.zones}
        for downstream in self.downstream.values():
            indegree[downstream] += 1
        ready = sorted(zone for zone, degree in indegree.items() if degree == 0)
        order: list[str] = []
        while ready:
            zone = ready.pop(0)
            order.append(zone)
            downstream = self.downstream.get(zone)
            if downstream is not None:
                indegree[downstream] -= 1
                if indegree[downstream] == 0:
                    ready.append(downstream)
                    ready.sort()
        if len(order) != len(self.zones):
            cyclic = sorted(zone for zone, degree in indegree.items() if degree > 0)
            raise ValueError(f"NEXT_DOWN graph contains a cycle: {cyclic[:10]}")
        return tuple(order)


def apply_proxy_alerts(
    routed: pd.DataFrame,
    runoff_thresholds: RunoffThresholds,
    data_config: FloodDataConfig,
    routing_config: FloodRoutingConfig,
    river_thresholds: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach local-runoff, river-flow and combined model-proxy alert levels."""
    output = routed.copy()
    zone_col = data_config.zone_col
    valid_col = data_config.valid_time_col
    runoff_ord = runoff_thresholds.levels(
        output[zone_col], output[valid_col], output["runoff_prediction_mm_h"]
    )
    output["runoff_alert_level"] = LEVEL_NAMES[runoff_ord]
    output["discharge_24h_mean_m3_s"] = output.groupby(
        [zone_col, data_config.issue_time_col], observed=True
    )["routed_discharge_m3_s"].transform("mean")

    river_ord = np.zeros(len(output), dtype=int)
    missing_threshold = np.ones(len(output), dtype=bool)
    if river_thresholds is not None and not river_thresholds.empty:
        required = [zone_col, *routing_config.river_threshold_cols]
        missing = [column for column in required if column not in river_thresholds]
        if missing:
            raise ValueError(f"River threshold table is missing columns: {missing}")
        threshold_map = river_thresholds.copy()
        threshold_map[zone_col] = threshold_map[zone_col].map(_id)
        threshold_map = threshold_map.set_index(zone_col)
        for position, (_, row) in enumerate(output.iterrows()):
            zone = _id(row[zone_col])
            if zone not in threshold_map.index:
                continue
            values = threshold_map.loc[zone, list(routing_config.river_threshold_cols)]
            values = np.asarray(values, dtype=float)
            if not np.isfinite(values).all() or not np.all(np.diff(values) >= 0):
                continue
            river_ord[position] = int(
                np.searchsorted(values, float(row["discharge_24h_mean_m3_s"]), side="left")
            )
            missing_threshold[position] = False
    output["river_alert_level"] = LEVEL_NAMES[river_ord]
    output["alert_level"] = LEVEL_NAMES[np.maximum(runoff_ord, river_ord)]
    output["warning_type"] = "model_proxy"
    output["river_threshold_missing"] = missing_threshold
    return output
