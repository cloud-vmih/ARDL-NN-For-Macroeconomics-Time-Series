"""Auditable point-in-time FRED/ALFRED observations for the macro study."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
from threading import Lock
from time import monotonic, sleep
from typing import Any, Callable, Mapping
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen
from uuid import uuid4

import pandas as pd


API_URL = "https://api.stlouisfed.org/fred/series/observations"
CACHE_SCHEMA_VERSION = 2
REQUEST_INTERVAL_SECONDS = 0.55  # Below FRED's 120 requests/minute ceiling.
_request_lock = Lock()
_next_request_at = 0.0


@dataclass(frozen=True)
class SeriesSpec:
    field: str
    series_id: str
    frequency: str
    unit: str
    construction: str


SERIES_SPECS = (
    SeriesSpec("unemployment_rate", "UNRATE", "monthly", "percent", "first_release"),
    SeriesSpec("manufacturing_hours", "AWHMAN", "monthly", "hours_seasonally_adjusted", "level"),
    SeriesSpec("nonfarm_payrolls", "PAYEMS", "monthly", "thousands_of_persons", "level"),
    SeriesSpec("cpi_all_items", "CPIAUCSL", "monthly", "index_1982_1984_100", "level"),
    SeriesSpec("industrial_prod", "INDPRO", "monthly", "index_2017_100", "level"),
)


def _http_transport(params: Mapping[str, Any]) -> dict[str, Any]:
    global _next_request_at
    with _request_lock:
        delay = max(0.0, _next_request_at - monotonic())
        if delay:
            sleep(delay)
        _next_request_at = monotonic() + REQUEST_INTERVAL_SECONDS
    with urlopen(f"{API_URL}?{urlencode(params)}", timeout=30) as response:
        return json.load(response)


def _parse_snapshot(payload: Mapping[str, Any], raw_sha256: str, retrieved_utc: str) -> pd.DataFrame:
    if "observations" not in payload or not isinstance(payload["observations"], list):
        raise ValueError("FRED response has no observations list.")
    observations = payload["observations"]
    if "count" in payload and int(payload["count"]) > len(observations):
        raise ValueError("FRED response was truncated; pagination is required.")
    frame = pd.DataFrame(observations)
    for column in ("date", "value", "realtime_start", "realtime_end"):
        if column not in frame:
            frame[column] = pd.Series(dtype="object")
    frame = frame[["date", "value", "realtime_start", "realtime_end"]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame["value"] = pd.to_numeric(frame["value"].replace(".", pd.NA), errors="coerce")
    for column in ("realtime_start", "realtime_end"):
        # ALFRED uses 9999-12-31 for open-ended vintages, beyond pandas'
        # nanosecond timestamp range. ISO date strings remain sortable.
        frame[column] = frame[column].map(
            lambda value: date.fromisoformat(str(value)).isoformat()
            if pd.notna(value) else None
        )
    frame.attrs["raw_sha256"] = raw_sha256
    frame.attrs["retrieved_utc"] = retrieved_utc
    return frame


def fetch_series_snapshot(
    series_id: str,
    as_of: date | None,
    *,
    output_type: int = 1,
    api_key: str,
    cache_root: Path,
    transport: Callable[[Mapping[str, Any]], Mapping[str, Any]] = _http_transport,
    observation_start: date | None = None,
    observation_end: date | None = None,
) -> pd.DataFrame:
    """Fetch one source snapshot, caching raw JSON without the API credential."""
    if not api_key:
        raise ValueError("FRED_API_KEY is required.")
    if as_of is None and output_type != 4:
        raise ValueError("An as-of date is required for a non-initial-release snapshot.")
    if as_of is not None and output_type != 1:
        raise ValueError("As-of snapshots use output_type=1.")
    public_params: dict[str, Any] = {
        "series_id": series_id,
        "file_type": "json",
        "output_type": output_type,
        "realtime_start": as_of.isoformat() if as_of else "1776-07-04",
        "realtime_end": as_of.isoformat() if as_of else "9999-12-31",
    }
    if observation_start is not None:
        public_params["observation_start"] = observation_start.isoformat()
    if observation_end is not None:
        public_params["observation_end"] = observation_end.isoformat()
    cache_identity = {"schema": CACHE_SCHEMA_VERSION, **public_params}
    key = hashlib.sha256(json.dumps(cache_identity, sort_keys=True).encode()).hexdigest()
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    payload_path = cache_root / f"{series_id}_{key}.json"
    metadata_path = cache_root / f"{series_id}_{key}.meta.json"
    if payload_path.exists() and metadata_path.exists():
        raw = payload_path.read_bytes()
        metadata = json.loads(metadata_path.read_text())
        raw_hash = hashlib.sha256(raw).hexdigest()
        if raw_hash != metadata.get("raw_sha256") or metadata.get("request") != cache_identity:
            raise ValueError(f"FRED cache integrity failure for {series_id}.")
        return _parse_snapshot(json.loads(raw), raw_hash, metadata["retrieved_utc"])

    for attempt in range(4):
        try:
            payload = dict(transport({**public_params, "api_key": api_key}))
            break
        except HTTPError as exc:
            if exc.code not in {429, 500, 502, 503, 504} or attempt == 3:
                raise RuntimeError(f"FRED request failed for {series_id} (HTTP {exc.code}).") from None
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            try:
                delay = float(retry_after) if retry_after is not None else (60.0 if exc.code == 429 else 2.0 ** attempt)
            except ValueError:
                delay = 60.0 if exc.code == 429 else 2.0 ** attempt
            sleep(max(0.0, delay))
        except Exception:
            # urllib errors can include the request URL, including the credential.
            raise RuntimeError(f"FRED request failed for {series_id}.") from None
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    raw_hash = hashlib.sha256(raw).hexdigest()
    retrieved_utc = datetime.now(timezone.utc).isoformat()
    frame = _parse_snapshot(payload, raw_hash, retrieved_utc)
    metadata = {"request": cache_identity, "raw_sha256": raw_hash, "retrieved_utc": retrieved_utc}
    payload_tmp = payload_path.with_name(f"{payload_path.name}.{uuid4().hex}.tmp")
    metadata_tmp = metadata_path.with_name(f"{metadata_path.name}.{uuid4().hex}.tmp")
    payload_tmp.write_bytes(raw)
    metadata_tmp.write_text(json.dumps(metadata, sort_keys=True, indent=2) + "\n")
    payload_tmp.replace(payload_path)
    metadata_tmp.replace(metadata_path)
    return frame
