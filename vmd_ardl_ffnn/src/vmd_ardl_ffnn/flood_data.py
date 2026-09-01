"""Leakage-safe panel loading and auditing for flood-proxy experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .flood_config import FloodDataConfig


def _read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table format: {path.suffix}")


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _utc(values: pd.Series, label: str) -> pd.Series:
    try:
        parsed = pd.to_datetime(values, errors="raise", utc=True)
    except Exception as exc:  # pragma: no cover - pandas supplies detailed parser errors
        raise ValueError(f"Unable to parse {label} as timestamps: {exc}") from exc
    return parsed


def _normalise_zone(values: pd.Series) -> pd.Series:
    return values.astype("string").str.strip()


@dataclass(frozen=True)
class PanelAudit:
    observation_rows: int
    static_rows: int
    observation_zones: int
    static_zones: int
    matched_zones: int
    start_time_utc: str
    end_time_utc: str
    unique_times: int
    min_rows_per_zone: int
    max_rows_per_zone: int
    duplicate_zone_times: int
    missing_observation_values: int
    missing_static_values: int
    zones_with_non_hourly_gaps: int
    has_target: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class FloodPanelLoader:
    """Load observation/NWP panels without collapsing duplicated timestamps across zones."""

    def __init__(self, config: FloodDataConfig):
        self.config = config

    def load_static(self, path: str | Path) -> pd.DataFrame:
        frame = _read_table(path).copy()
        required = [self.config.zone_col, *self.config.static_numeric_cols, *self.config.static_categorical_cols]
        _require_columns(frame, required, "Static table")
        frame[self.config.zone_col] = _normalise_zone(frame[self.config.zone_col])
        if frame[self.config.zone_col].isna().any() or frame[self.config.zone_col].eq("").any():
            raise ValueError("Static table contains empty zone identifiers.")
        duplicated = frame[self.config.zone_col].duplicated(keep=False)
        if duplicated.any():
            values = frame.loc[duplicated, self.config.zone_col].head(10).tolist()
            raise ValueError(f"Static table contains duplicate zones: {values}")
        for column in self.config.static_numeric_cols:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        return frame.sort_values(self.config.zone_col).reset_index(drop=True)

    def load_observations(self, path: str | Path, *, require_target: bool = True) -> pd.DataFrame:
        frame = _read_table(path).copy()
        required = [self.config.zone_col, self.config.time_col, *self.config.exogenous_cols]
        if require_target:
            required.append(self.config.target_col)
        _require_columns(frame, required, "Observation table")
        frame[self.config.zone_col] = _normalise_zone(frame[self.config.zone_col])
        frame[self.config.time_col] = _utc(frame[self.config.time_col], self.config.time_col)
        numeric = list(self.config.exogenous_cols)
        if self.config.target_col in frame:
            numeric.append(self.config.target_col)
        for column in numeric:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
            if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
                raise ValueError(f"Observation column {column!r} contains non-finite values.")
        duplicate = frame.duplicated([self.config.zone_col, self.config.time_col], keep=False)
        if duplicate.any():
            rows = frame.loc[duplicate, [self.config.zone_col, self.config.time_col]].head(10)
            raise ValueError(f"Observation table contains duplicate zone/timestamps:\n{rows}")
        frame = frame.sort_values([self.config.zone_col, self.config.time_col]).reset_index(drop=True)
        if self.config.require_hourly_observations:
            self._validate_hourly(frame)
        return frame

    def load_forecasts(self, path: str | Path) -> pd.DataFrame:
        frame = _read_table(path).copy()
        required = [
            self.config.zone_col,
            self.config.issue_time_col,
            self.config.valid_time_col,
            *self.config.nwp_cols,
        ]
        _require_columns(frame, required, "Forecast table")
        frame[self.config.zone_col] = _normalise_zone(frame[self.config.zone_col])
        frame[self.config.issue_time_col] = _utc(frame[self.config.issue_time_col], self.config.issue_time_col)
        frame[self.config.valid_time_col] = _utc(frame[self.config.valid_time_col], self.config.valid_time_col)
        delta = (
            frame[self.config.valid_time_col] - frame[self.config.issue_time_col]
        ).dt.total_seconds() / 3600.0
        if (delta <= 0).any():
            raise ValueError("Every NWP valid_time must be later than issue_time.")
        rounded = np.rint(delta).astype(int)
        if not np.allclose(delta, rounded):
            raise ValueError("Forecast lead times must be whole hours.")
        if "lead_hour" in frame:
            supplied = pd.to_numeric(frame["lead_hour"], errors="raise").astype(int)
            if not np.array_equal(supplied.to_numpy(), rounded):
                raise ValueError("lead_hour does not match valid_time - issue_time.")
        frame["lead_hour"] = rounded
        for column in self.config.nwp_cols:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
            if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
                raise ValueError(f"Forecast column {column!r} contains non-finite values.")
        key = [self.config.zone_col, self.config.issue_time_col, self.config.valid_time_col]
        duplicate = frame.duplicated(key, keep=False)
        if duplicate.any():
            raise ValueError("Forecast table contains duplicate zone/issue/valid records.")
        return frame.sort_values(key).reset_index(drop=True)

    def audit(self, observations: pd.DataFrame, static: pd.DataFrame) -> PanelAudit:
        zone = self.config.zone_col
        time = self.config.time_col
        obs_zones = set(observations[zone].astype(str))
        static_zones = set(static[zone].astype(str))
        row_counts = observations.groupby(zone, observed=True).size()
        duplicate_count = int(observations.duplicated([zone, time]).sum())
        non_hourly = self._zones_with_non_hourly_gaps(observations)
        relevant_obs = [*self.config.exogenous_cols]
        if self.config.target_col in observations:
            relevant_obs.append(self.config.target_col)
        static_columns = [*self.config.static_numeric_cols, *self.config.static_categorical_cols]
        return PanelAudit(
            observation_rows=int(len(observations)),
            static_rows=int(len(static)),
            observation_zones=int(len(obs_zones)),
            static_zones=int(len(static_zones)),
            matched_zones=int(len(obs_zones & static_zones)),
            start_time_utc=observations[time].min().isoformat(),
            end_time_utc=observations[time].max().isoformat(),
            unique_times=int(observations[time].nunique()),
            min_rows_per_zone=int(row_counts.min()),
            max_rows_per_zone=int(row_counts.max()),
            duplicate_zone_times=duplicate_count,
            missing_observation_values=int(observations[relevant_obs].isna().sum().sum()),
            missing_static_values=int(static[static_columns].isna().sum().sum()),
            zones_with_non_hourly_gaps=non_hourly,
            has_target=self.config.target_col in observations,
        )

    def validate_zone_coverage(
        self,
        observations: pd.DataFrame,
        static: pd.DataFrame,
        forecasts: pd.DataFrame | None = None,
    ) -> None:
        zone = self.config.zone_col
        obs_zones = set(observations[zone])
        static_zones = set(static[zone])
        missing_static = sorted(obs_zones - static_zones)
        if missing_static:
            raise ValueError(f"Observation zones missing from static table: {missing_static[:10]}")
        if forecasts is not None:
            forecast_zones = set(forecasts[zone])
            unknown = sorted(forecast_zones - static_zones)
            if unknown:
                raise ValueError(f"Forecast zones missing from static table: {unknown[:10]}")

    def make_oracle_forecasts(
        self,
        observations: pd.DataFrame,
        *,
        max_horizon: int = 24,
        issue_stride_hours: int = 6,
    ) -> pd.DataFrame:
        """Create a clearly labelled perfect-exogenous forecast for upper-bound experiments."""
        zone_col = self.config.zone_col
        time_col = self.config.time_col
        records: list[dict[str, object]] = []
        for zone, group in observations.groupby(zone_col, sort=False, observed=True):
            group = group.sort_values(time_col).reset_index(drop=True)
            values = group.set_index(time_col)
            issues = group.iloc[::issue_stride_hours][time_col]
            for issue in issues:
                for lead in range(1, max_horizon + 1):
                    valid = issue + pd.Timedelta(hours=lead)
                    if valid not in values.index:
                        continue
                    row: dict[str, object] = {
                        zone_col: zone,
                        self.config.issue_time_col: issue,
                        self.config.valid_time_col: valid,
                        "lead_hour": lead,
                        "forecast_mode": "oracle_exogenous",
                    }
                    source = values.loc[valid]
                    for observed_col, forecast_col in zip(self.config.exogenous_cols, self.config.nwp_cols):
                        row[forecast_col] = float(source[observed_col])
                    records.append(row)
        return pd.DataFrame.from_records(records)

    def _validate_hourly(self, frame: pd.DataFrame) -> None:
        count = self._zones_with_non_hourly_gaps(frame)
        if count:
            raise ValueError(f"Observation data contain non-hourly gaps in {count} zone(s).")

    def _zones_with_non_hourly_gaps(self, frame: pd.DataFrame) -> int:
        zone = self.config.zone_col
        time = self.config.time_col
        count = 0
        for _, group in frame.groupby(zone, sort=False, observed=True):
            delta = group.sort_values(time)[time].diff().dropna().dt.total_seconds() / 3600.0
            if not delta.eq(1.0).all():
                count += 1
        return count
