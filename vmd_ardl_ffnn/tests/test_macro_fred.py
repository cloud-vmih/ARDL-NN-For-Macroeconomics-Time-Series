from __future__ import annotations

from datetime import date
from io import BytesIO
import json
from pathlib import Path
from urllib.error import HTTPError

import pandas as pd
import pytest

from vmd_ardl_ffnn.macro_fred_source import fetch_series_snapshot
from vmd_ardl_ffnn.macro_fred_panel import build_us_unemployment_panel, SourceAvailabilityError


def test_snapshot_uses_requested_vintage_and_keeps_key_out_of_cache(tmp_path: Path) -> None:
    requests = []

    def transport(params):
        requests.append(dict(params))
        return {
            "observations": [
                {"date": "2024-01-01", "value": "3.7", "realtime_start": "2024-02-02", "realtime_end": "2024-02-28"},
                {"date": "2024-02-01", "value": ".", "realtime_start": "2024-02-28", "realtime_end": "2024-02-28"},
            ]
        }

    first = fetch_series_snapshot(
        "UNRATE", date(2024, 2, 28), api_key="secret-key",
        cache_root=tmp_path, transport=transport,
    )
    second = fetch_series_snapshot(
        "UNRATE", date(2024, 2, 28), api_key="secret-key",
        cache_root=tmp_path,
        transport=lambda _: pytest.fail("cache miss"),
    )

    assert len(requests) == 1
    assert requests[0]["realtime_start"] == "2024-02-28"
    assert requests[0]["realtime_end"] == "2024-02-28"
    assert requests[0]["output_type"] == 1
    assert requests[0]["file_type"] == "json"
    assert first.iloc[0]["value"] == 3.7
    assert pd.isna(first.iloc[1]["value"])
    pd.testing.assert_frame_equal(first, second)
    assert len(first.attrs["raw_sha256"]) == 64
    assert first.attrs["retrieved_utc"] == second.attrs["retrieved_utc"]
    assert "T" in first.attrs["retrieved_utc"]
    assert "secret-key" not in "".join(p.read_text() for p in tmp_path.glob("*.json"))


def test_initial_release_request_uses_complete_realtime_range(tmp_path: Path) -> None:
    seen = []

    def transport(params):
        seen.append(dict(params))
        return {"observations": [{"date": "2024-01-01", "value": "3.7", "realtime_start": "2024-02-02", "realtime_end": "2024-02-28"}]}

    fetch_series_snapshot("UNRATE", None, output_type=4, api_key="key", cache_root=tmp_path, transport=transport)

    assert seen[0]["realtime_start"] == "1776-07-04"
    assert seen[0]["realtime_end"] == "9999-12-31"
    assert seen[0]["output_type"] == 4


def test_snapshot_accepts_open_ended_realtime_date(tmp_path: Path) -> None:
    frame = fetch_series_snapshot(
        "UNRATE", None, output_type=4, api_key="key", cache_root=tmp_path,
        transport=lambda params: {"observations": [{
            "date": "2025-09-01", "value": "4.4",
            "realtime_start": "2025-11-20", "realtime_end": "9999-12-31",
        }]},
    )
    assert frame.iloc[0]["realtime_end"] == "9999-12-31"


def test_http_transport_spaces_requests_below_fred_rate_limit(monkeypatch) -> None:
    from vmd_ardl_ffnn import macro_fred_source as source

    clock = [0.0]
    delays = []
    monkeypatch.setattr(source, "_next_request_at", 0.0, raising=False)
    monkeypatch.setattr(source, "monotonic", lambda: clock[0], raising=False)
    def fake_sleep(seconds):
        delays.append(seconds)
        clock[0] += seconds
    monkeypatch.setattr(source, "sleep", fake_sleep, raising=False)
    monkeypatch.setattr(source, "urlopen", lambda *args, **kwargs: BytesIO(b'{"observations": []}'))

    source._http_transport({"series_id": "UNRATE", "api_key": "secret"})
    source._http_transport({"series_id": "UNRATE", "api_key": "secret"})

    assert len(delays) == 1
    assert delays[0] >= 0.5


def test_snapshot_retries_http_429_without_exposing_key(tmp_path: Path, monkeypatch) -> None:
    from vmd_ardl_ffnn import macro_fred_source as source

    attempts = []
    monkeypatch.setattr(source, "sleep", lambda seconds: None, raising=False)

    def transport(params):
        attempts.append(1)
        if len(attempts) == 1:
            raise HTTPError("https://example.test/?api_key=secret-key", 429, "Too Many Requests", {"Retry-After": "0"}, None)
        return {"observations": [{
            "date": "2024-01-01", "value": "3.7", "realtime_start": "2024-02-02", "realtime_end": "2024-02-28",
        }]}

    frame = fetch_series_snapshot(
        "UNRATE", date(2024, 2, 28), api_key="secret-key",
        cache_root=tmp_path, transport=transport,
    )
    assert len(attempts) == 2
    assert frame.iloc[0]["value"] == 3.7


def _two_month_transport(params):
    series = params["series_id"]
    if series == "UNRATE":
        return {"observations": [
            {"date": "2024-01-01", "value": "3.7", "realtime_start": "2024-02-02", "realtime_end": "2024-03-07"},
            {"date": "2024-02-01", "value": "3.9", "realtime_start": "2024-03-08", "realtime_end": "2024-04-04"},
        ]}
    month = params["observation_start"][:7]
    as_of = params["realtime_start"]
    value = {"AWHMAN": "41.5" if month == "2024-01" else "41.2", "PAYEMS": "150000", "CPIAUCSL": "300", "INDPRO": "100"}[series]
    return {"observations": [
        {"date": f"{month}-01", "value": value, "realtime_start": as_of, "realtime_end": as_of}
    ]}


def test_panel_maps_manufacturing_hours_and_records_source_audit(tmp_path: Path) -> None:
    panel_path, audit_path = build_us_unemployment_panel(
        tmp_path / "cache", tmp_path / "out", start="2024-01-01", end="2024-02-01",
        api_key="key", transport=_two_month_transport,
    )
    panel = pd.read_csv(panel_path)
    audit = __import__("json").loads(audit_path.read_text())

    assert panel["manufacturing_hours"].tolist() == [41.5, 41.2]
    assert panel["unemployment_rate"].tolist() == [3.7, 3.9]
    assert panel.columns.tolist() == [
        "date", "unemployment_rate", "manufacturing_hours", "nonfarm_payrolls",
        "cpi_all_items", "industrial_prod",
    ]
    assert len(audit["panel_sha256"]) == 64
    assert {item["series_id"] for item in audit["series"]} == {
        "UNRATE", "AWHMAN", "PAYEMS", "CPIAUCSL", "INDPRO"
    }
    assert "weekly_aggregation" not in audit
    assert audit["snapshots"][0]["raw_sha256"]


def test_panel_rejects_late_source_value(tmp_path: Path) -> None:
    def late_payroll(params):
        payload = _two_month_transport(params)
        if params["series_id"] == "PAYEMS":
            payload["observations"][0]["realtime_start"] = "2025-01-01"
        return payload

    with pytest.raises(SourceAvailabilityError, match="PAYEMS.*2024-01"):
        build_us_unemployment_panel(
            tmp_path / "cache", tmp_path / "out", start="2024-01-01", end="2024-02-01",
            api_key="key", transport=late_payroll,
        )


def test_panel_rejects_missing_manufacturing_hours(tmp_path: Path) -> None:
    def absent_hours(params):
        if params["series_id"] == "AWHMAN":
            return {"observations": []}
        return _two_month_transport(params)

    with pytest.raises(SourceAvailabilityError, match="AWHMAN.*2024-01"):
        build_us_unemployment_panel(
            tmp_path / "cache", tmp_path / "out", start="2024-01-01", end="2024-01-01",
            api_key="key", transport=absent_hours,
        )


def test_panel_rejects_target_lag_released_after_next_month(tmp_path: Path) -> None:
    def late_target(params):
        payload = _two_month_transport(params)
        if params["series_id"] == "UNRATE":
            payload["observations"][0]["realtime_start"] = "2024-03-01"
        return payload

    with pytest.raises(SourceAvailabilityError, match="UNRATE.*2024-01"):
        build_us_unemployment_panel(
            tmp_path / "cache", tmp_path / "out", start="2024-01-01", end="2024-02-01",
            api_key="key", transport=late_target,
        )


def test_panel_refuses_missing_october_target(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="2025-10"):
        build_us_unemployment_panel(
            tmp_path / "cache", tmp_path / "out", start="2025-10-01", end="2025-10-01",
            api_key="key", transport=_two_month_transport,
        )


def test_terminal_september_exogs_use_audited_unused_vintage(tmp_path: Path) -> None:
    def delayed_sources(params):
        series = params["series_id"]
        if series == "UNRATE":
            return {"observations": [{
                "date": "2025-09-01", "value": "4.4",
                "realtime_start": "2025-11-20", "realtime_end": "2025-12-01",
            }]}
        assert params["realtime_start"] == "2026-01-31"
        dates = ["2025-09-01"]
        return {"observations": [
            {"date": day, "value": "100", "realtime_start": "2025-11-20",
             "realtime_end": "2026-01-31"}
            for day in dates
        ]}

    panel_path, audit_path = build_us_unemployment_panel(
        tmp_path / "cache", tmp_path / "out", start="2025-09-01", end="2025-09-01",
        api_key="key", transport=delayed_sources,
    )
    audit = json.loads(audit_path.read_text())
    assert pd.read_csv(panel_path)["unemployment_rate"].tolist() == [4.4]
    assert audit["terminal_exog_exception"]["as_of"] == "2026-01-31"
    assert audit["terminal_exog_exception"]["used_in_forecasts"] is False
