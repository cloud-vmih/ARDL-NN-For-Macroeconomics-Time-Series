"""Source-data preparation helpers independent of a particular GIS backend."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd


def normalise_hybas_id(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def select_upstream_network(
    static: pd.DataFrame,
    seed_zone_ids: Iterable[object],
    *,
    zone_col: str = "HYBAS_ID",
    downstream_col: str = "NEXT_DOWN",
) -> pd.DataFrame:
    """Keep seed polygons (e.g. intersecting Son La) and every contributing upstream zone."""
    required = [zone_col, downstream_col]
    missing = [column for column in required if column not in static]
    if missing:
        raise ValueError(f"Static table is missing topology columns: {missing}")
    frame = static.copy()
    frame[zone_col] = frame[zone_col].map(normalise_hybas_id)
    frame[downstream_col] = frame[downstream_col].map(normalise_hybas_id)
    known = set(frame[zone_col])
    seeds = {normalise_hybas_id(value) for value in seed_zone_ids}
    unknown = sorted(seeds - known)
    if unknown:
        raise ValueError(f"Seed zones are absent from the static table: {unknown[:10]}")
    reverse: dict[str, list[str]] = {zone: [] for zone in known}
    for zone, downstream in frame[[zone_col, downstream_col]].itertuples(index=False):
        if downstream in reverse and downstream != zone:
            reverse[downstream].append(zone)
    selected = set(seeds)
    queue: deque[str] = deque(sorted(seeds))
    while queue:
        downstream = queue.popleft()
        for upstream in reverse.get(downstream, []):
            if upstream not in selected:
                selected.add(upstream)
                queue.append(upstream)
    output = frame.loc[frame[zone_col].isin(selected)].copy()
    output["selection_reason"] = np.where(
        output[zone_col].isin(seeds), "intersects_son_la", "upstream_contributor"
    )
    return output.sort_values(zone_col).reset_index(drop=True)


def effective_grid_coverage(
    weights: pd.DataFrame,
    *,
    zone_col: str = "HYBAS_ID",
    weight_col: str = "overlap_weight",
    minimum_effective_cells: float = 3.0,
) -> pd.DataFrame:
    """Calculate Kish effective cell count and flag unsupported fine-scale model zones."""
    if minimum_effective_cells <= 0:
        raise ValueError("minimum_effective_cells must be > 0.")
    frame = weights[[zone_col, weight_col]].copy()
    frame[zone_col] = frame[zone_col].map(normalise_hybas_id)
    frame[weight_col] = pd.to_numeric(frame[weight_col], errors="raise")
    if (frame[weight_col] < 0).any():
        raise ValueError("Raster overlap weights must be non-negative.")
    grouped = frame.groupby(zone_col, observed=True)[weight_col]
    total = grouped.sum()
    squared = grouped.apply(lambda values: float(np.square(values).sum()))
    effective = np.square(total) / squared.replace(0.0, np.nan)
    return pd.DataFrame(
        {
            zone_col: effective.index,
            "effective_grid_cells": effective.to_numpy(),
            "minimum_effective_grid_cells": float(minimum_effective_cells),
            "independent_resolution_supported": effective.to_numpy() >= minimum_effective_cells,
        }
    )


def area_weighted_basin_aggregation(
    gridded: pd.DataFrame,
    weights: pd.DataFrame,
    *,
    value_columns: Iterable[str],
    time_columns: tuple[str, ...] = ("valid_time",),
    grid_col: str = "grid_id",
    zone_col: str = "HYBAS_ID",
    weight_col: str = "overlap_weight",
) -> pd.DataFrame:
    """Aggregate point/grid values to basins using precomputed polygon-cell overlap weights."""
    value_columns = tuple(value_columns)
    required_grid = [grid_col, *time_columns, *value_columns]
    required_weight = [grid_col, zone_col, weight_col]
    missing_grid = [column for column in required_grid if column not in gridded]
    missing_weight = [column for column in required_weight if column not in weights]
    if missing_grid or missing_weight:
        raise ValueError(
            f"Missing aggregation columns; gridded={missing_grid}, weights={missing_weight}"
        )
    weight_frame = weights[required_weight].copy()
    weight_frame[zone_col] = weight_frame[zone_col].map(normalise_hybas_id)
    weight_frame[weight_col] = pd.to_numeric(weight_frame[weight_col], errors="raise")
    merged = gridded[required_grid].merge(weight_frame, on=grid_col, how="inner", validate="many_to_many")
    if merged.empty:
        raise ValueError("No grid IDs overlap between gridded values and basin weights.")
    group_columns = [zone_col, *time_columns]
    weight_sum = merged.groupby(group_columns, observed=True)[weight_col].transform("sum")
    if (weight_sum <= 0).any():
        raise ValueError("Every basin/time group must have positive overlap weight.")
    merged["__normalised_weight"] = merged[weight_col] / weight_sum
    for column in value_columns:
        merged[column] = pd.to_numeric(merged[column], errors="raise") * merged["__normalised_weight"]
    return (
        merged.groupby(group_columns, as_index=False, observed=True)[list(value_columns)]
        .sum()
        .sort_values(group_columns)
        .reset_index(drop=True)
    )


def deaccumulate_forecast_fields(
    frame: pd.DataFrame,
    cumulative_columns: Mapping[str, str],
    *,
    group_columns: tuple[str, ...] = ("HYBAS_ID", "issue_time"),
    valid_time_col: str = "valid_time",
    negative_tolerance: float = 1e-8,
) -> pd.DataFrame:
    """Convert cumulative NWP fields to per-step amounts without crossing issue cycles.

    ``cumulative_columns`` maps source cumulative names to output per-step names. A large
    negative difference is treated as a source/cycle error instead of silently clipping it.
    """
    required = [*group_columns, valid_time_col, *cumulative_columns]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Forecast frame is missing deaccumulation columns: {missing}")
    output = frame.copy().sort_values([*group_columns, valid_time_col])
    for source, destination in cumulative_columns.items():
        output[source] = pd.to_numeric(output[source], errors="raise")
        increment = output.groupby(list(group_columns), observed=True)[source].diff()
        first = output.groupby(list(group_columns), observed=True).cumcount().eq(0)
        increment.loc[first] = output.loc[first, source]
        if (increment < -negative_tolerance).any():
            example = output.loc[increment < -negative_tolerance, [*group_columns, valid_time_col]].head(5)
            raise ValueError(
                "Cumulative forecast decreases within an issue cycle; verify cycle grouping. "
                f"Examples:\n{example}"
            )
        output[destination] = increment.clip(lower=0.0)
    return output


def convert_water_depth_metres_to_mm(
    frame: pd.DataFrame,
    columns: Mapping[str, str],
) -> pd.DataFrame:
    """Rename ERA5-Land water-depth fields while converting metres to millimetres."""
    missing = [source for source in columns if source not in frame]
    if missing:
        raise ValueError(f"Water-depth source columns are missing: {missing}")
    output = frame.copy()
    for source, destination in columns.items():
        output[destination] = pd.to_numeric(output[source], errors="raise") * 1000.0
    return output
