"""Build the fixed US unemployment study panel from FRED/ALFRED vintages."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from .macro_fred_source import SERIES_SPECS, _http_transport, fetch_series_snapshot


class SourceAvailabilityError(ValueError):
    """A source observation was unavailable at its registered vintage."""


PANEL_NAME = "us_unemployment_awhman_2000_2025_09.csv"
PANEL_FIELDS = ("date", *(spec.field for spec in SERIES_SPECS))


def _snapshot_value(
    frame: pd.DataFrame, series_id: str, month: pd.Timestamp, as_of: pd.Timestamp
) -> float:
    month_end = month + pd.offsets.MonthEnd(1)
    rows = frame.loc[frame["date"].between(month, month_end)].copy()
    if len(rows) != 1 or rows.iloc[0]["date"] != month:
        raise SourceAvailabilityError(
            f"{series_id} {month:%Y-%m}: expected one monthly observation."
        )
    if rows["value"].isna().any() or not np.isfinite(rows["value"].to_numpy(float)).all():
        raise SourceAvailabilityError(f"{series_id} {month:%Y-%m}: source value missing.")
    if rows["realtime_start"].isna().any() or rows["realtime_end"].isna().any():
        raise SourceAvailabilityError(f"{series_id} {month:%Y-%m}: vintage dates missing.")
    as_of_iso = as_of.strftime("%Y-%m-%d")
    if (rows["realtime_start"] > as_of_iso).any() or (rows["realtime_end"] < as_of_iso).any():
        raise SourceAvailabilityError(
            f"{series_id} {month:%Y-%m}: value unavailable by {as_of:%Y-%m-%d}."
        )
    return float(rows.iloc[0]["value"])


def _write_immutable(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"Existing study artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def build_us_unemployment_panel(
    cache_root: Path,
    output_root: Path,
    *,
    start: str = "2000-01-01",
    end: str = "2025-09-01",
    api_key: str,
    transport: Callable[[Mapping[str, Any]], Mapping[str, Any]] = _http_transport,
) -> tuple[Path, Path]:
    """Freeze monthly inputs at the end of the next month and audit availability."""
    first, last = pd.Timestamp(start), pd.Timestamp(end)
    if first.day != 1 or last.day != 1 or first > last:
        raise ValueError("Panel start/end must be ordered month-start dates.")
    if first <= pd.Timestamp("2025-10-01") <= last:
        raise ValueError("2025-10 UNRATE is unavailable; do not fill this month.")
    months = pd.date_range(first, last, freq="MS")
    target = fetch_series_snapshot(
        "UNRATE", None, output_type=4, api_key=api_key, cache_root=cache_root,
        transport=transport, observation_start=first.date(), observation_end=last.date(),
    )
    if target["date"].duplicated().any():
        raise SourceAvailabilityError("UNRATE initial releases contain duplicate months.")
    target_by_date = target.set_index("date")
    panel_rows: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = [
        {"series_id": "UNRATE", "kind": "initial_release", "raw_sha256": target.attrs["raw_sha256"],
         "retrieved_utc": target.attrs["retrieved_utc"]}
    ]
    for month in months:
        if month not in target_by_date.index:
            raise SourceAvailabilityError(f"UNRATE {month:%Y-%m}: initial release missing.")
        label = target_by_date.loc[month]
        if pd.isna(label["value"]) or not np.isfinite(float(label["value"])):
            raise SourceAvailabilityError(f"UNRATE {month:%Y-%m}: initial release value missing.")
        if pd.isna(label["realtime_start"]) or label["realtime_start"] <= (month + pd.offsets.MonthEnd(1)).strftime("%Y-%m-%d"):
            raise SourceAvailabilityError(
                f"UNRATE {month:%Y-%m}: invalid initial release date."
            )
        as_of = month + pd.offsets.MonthEnd(2)
        terminal_delayed = month == last == pd.Timestamp("2025-09-01")
        if terminal_delayed:
            # Shutdown-delayed September exogs cannot affect any forecast in this
            # panel: all feature lags are >= 1 and October has no UNRATE label.
            as_of = pd.Timestamp("2026-01-31")
        if month != last and label["realtime_start"] > as_of.strftime("%Y-%m-%d"):
            raise SourceAvailabilityError(
                f"UNRATE {month:%Y-%m}: first release unavailable by {as_of:%Y-%m-%d} "
                "for the next forecast origin."
            )
        row: dict[str, Any] = {"date": month.strftime("%Y-%m-%d"), "unemployment_rate": float(label["value"])}
        for spec in SERIES_SPECS[1:]:
            snapshot = fetch_series_snapshot(
                spec.series_id, as_of.date(), api_key=api_key, cache_root=cache_root,
                transport=transport, observation_start=month.date(),
                observation_end=(month + pd.offsets.MonthEnd(1)).date(),
            )
            row[spec.field] = _snapshot_value(snapshot, spec.series_id, month, as_of)
            snapshots.append(
                {"series_id": spec.series_id, "reference_month": row["date"],
                 "source_month": row["date"],
                 "source_lag_months": 0,
                 "as_of": as_of.strftime("%Y-%m-%d"),
                 "used_in_forecasts": month != last,
                 "raw_sha256": snapshot.attrs["raw_sha256"],
                 "retrieved_utc": snapshot.attrs["retrieved_utc"]}
            )
        panel_rows.append(row)

    panel = pd.DataFrame(panel_rows, columns=PANEL_FIELDS)
    if panel.isna().any().any() or not np.isfinite(panel.iloc[:, 1:].to_numpy(float)).all():
        raise ValueError("Panel contains missing or nonfinite model values.")
    if (first, last) == (pd.Timestamp("2000-01-01"), pd.Timestamp("2025-09-01")):
        if len(panel) != 309:
            raise ValueError(f"Main panel must have 309 months, got {len(panel)}.")
    csv_bytes = panel.to_csv(index=False).encode()
    panel_hash = hashlib.sha256(csv_bytes).hexdigest()
    audit = {
        "schema_version": 1,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "source_api": "FRED/ALFRED fred/series/observations",
        "forecast_origin": "last calendar day of reference month, 23:59 America/New_York",
        "input_rule": "observations through t-1, frozen at end of m+1; exact reference month required; unused terminal Sep 2025 exogs at Jan 31 2026",
        "target_rule": "UNRATE output_type=4 initial release",
        "series": [asdict(spec) for spec in SERIES_SPECS],
        "start": panel.iloc[0]["date"],
        "end": panel.iloc[-1]["date"],
        "row_count": len(panel),
        "split_counts": {"train": 216, "validation": 46, "test": 47} if len(panel) == 309 else None,
        "missing_counts": panel.isna().sum().to_dict(),
        "panel_sha256": panel_hash,
        "snapshots": snapshots,
        "terminal_exog_exception": (
            {"reference_month": "2025-09-01", "as_of": "2026-01-31",
             "used_in_forecasts": False,
             "reason": "September 2025 INDPRO was released after November 2025; no October UNRATE label",
             "source_urls": [
                 "https://www.bls.gov/news.release/archives/empsit_11202025.htm",
                 "https://www.census.gov/construction/nrc/data/releases.html",
                 "https://www.federalreserve.gov/releases/g17/20251203/default.htm",
             ]}
            if last == pd.Timestamp("2025-09-01") else None
        ),
    }
    output_root = Path(output_root)
    panel_path = output_root / PANEL_NAME
    audit_path = panel_path.with_suffix(".audit.json")
    _write_immutable(panel_path, csv_bytes)
    audit_bytes = (json.dumps(audit, sort_keys=True, indent=2) + "\n").encode()
    # Retrieval time changes between invocations; preserve an existing valid audit.
    if audit_path.exists():
        existing = json.loads(audit_path.read_text())
        if existing.get("panel_sha256") != panel_hash or existing.get("snapshots") != snapshots:
            raise FileExistsError(f"Existing source audit differs: {audit_path}")
    else:
        _write_immutable(audit_path, audit_bytes)
    return panel_path, audit_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    key = os.environ.get("FRED_API_KEY", "")
    if not key:
        parser.error("FRED_API_KEY must be set in the environment.")
    panel_path, audit_path = build_us_unemployment_panel(
        args.cache_root, args.output_root, api_key=key
    )
    print(f"Panel: {panel_path}\nAudit: {audit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
