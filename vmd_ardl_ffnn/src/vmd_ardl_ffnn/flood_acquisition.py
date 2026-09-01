"""Reproducible acquisition of public hydrometeorological data for Son La.

The module intentionally keeps downloaded source files and processed model tables
separate.  Every network request is cached, so interrupted downloads can be resumed.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
import gzip
import json
import math
from pathlib import Path
import shutil
import time
from typing import Any, Iterable, Sequence
import zipfile

import numpy as np
import pandas as pd
import requests

from .flood_preparation import normalise_hybas_id, select_upstream_network


HYDROBASINS_URL = (
    "https://data.hydrosheds.org/file/hydrobasins/standard/"
    "hybas_as_lev12_v1c.zip"
)
VNM_ADM1_URL = (
    "https://github.com/wmgeolab/geoBoundaries/raw/9469f09/"
    "releaseData/gbOpen/VNM/ADM1/geoBoundaries-VNM-ADM1.geojson"
)
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FLOOD_URL = "https://flood-api.open-meteo.com/v1/flood"
ERA5_BUCKET_URL = "https://nsf-ncar-era5.s3.us-west-2.amazonaws.com"

WEATHER_VARIABLES = (
    "temperature_2m",
    "rain",
    "precipitation",
    "et0_fao_evapotranspiration",
    "soil_moisture_0_to_7cm",
    "soil_moisture_7_to_28cm",
    "soil_moisture_28_to_100cm",
    "soil_moisture_100_to_255cm",
)


@dataclass(frozen=True)
class AcquisitionPaths:
    root: Path

    @property
    def raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def hydro_zip(self) -> Path:
        return self.raw / "hydrobasins" / "hybas_as_lev12_v1c.zip"

    @property
    def hydro_dir(self) -> Path:
        return self.raw / "hydrobasins" / "hybas_as_lev12_v1c"

    @property
    def hydro_shp(self) -> Path:
        return self.hydro_dir / "hybas_as_lev12_v1c.shp"

    @property
    def adm1_geojson(self) -> Path:
        return self.raw / "boundaries" / "geoBoundaries-VNM-ADM1.geojson"

    @property
    def basin_network(self) -> Path:
        return self.processed / "sonla_basin_network.csv"

    @property
    def static_subset(self) -> Path:
        return self.processed / "static_features_sonla_upstream.csv"


def _atomic_download(url: str, destination: Path, *, timeout: int = 300) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    temporary = destination.with_suffix(destination.suffix + ".part")
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        expected = int(response.headers.get("content-length", "0"))
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    if expected and temporary.stat().st_size != expected:
        temporary.unlink(missing_ok=True)
        raise IOError(
            f"Incomplete download for {url}: expected {expected}, got "
            f"{temporary.stat().st_size if temporary.exists() else 0} bytes"
        )
    temporary.replace(destination)
    return destination


def ensure_geometry_sources(paths: AcquisitionPaths) -> None:
    """Download the two open geometry sources when they are absent."""
    _atomic_download(HYDROBASINS_URL, paths.hydro_zip)
    if not paths.hydro_shp.exists():
        paths.hydro_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(paths.hydro_zip) as archive:
            archive.extractall(paths.hydro_dir)
    _atomic_download(VNM_ADM1_URL, paths.adm1_geojson)


def _load_son_la_geometry(path: Path):
    from shapely.geometry import shape

    payload = json.loads(path.read_text(encoding="utf-8"))
    matches = [
        item for item in payload["features"]
        if item["properties"].get("shapeName") in {"Sơn La", "Son La"}
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one Son La ADM1 feature, found {len(matches)}.")
    return shape(matches[0]["geometry"])


def build_basin_network(
    paths: AcquisitionPaths,
    *,
    static_path: str | Path,
) -> pd.DataFrame:
    """Select Son La-intersecting HydroBASINS polygons and all upstream contributors."""
    import shapefile
    from shapely.geometry import shape
    from shapely.ops import nearest_points

    ensure_geometry_sources(paths)
    static = pd.read_csv(static_path)
    static["HYBAS_ID"] = static["HYBAS_ID"].map(normalise_hybas_id)
    wanted = set(static["HYBAS_ID"])
    son_la = _load_son_la_geometry(paths.adm1_geojson)

    geometries: dict[str, Any] = {}
    with shapefile.Reader(str(paths.hydro_shp)) as reader:
        for shape_record in reader.iterShapeRecords():
            zone = normalise_hybas_id(shape_record.record.HYBAS_ID)
            if zone in wanted:
                geometries[zone] = shape(shape_record.shape.__geo_interface__)
    missing = sorted(wanted - set(geometries))
    if missing:
        raise ValueError(f"HydroBASINS geometry is missing static IDs: {missing[:10]}")

    intersecting = sorted(zone for zone, geom in geometries.items() if geom.intersects(son_la))
    selected = select_upstream_network(static, intersecting)
    selected_ids = set(selected["HYBAS_ID"])

    rows: list[dict[str, Any]] = []
    topology = static.set_index("HYBAS_ID")["NEXT_DOWN"].map(normalise_hybas_id)
    for zone in sorted(selected_ids):
        geom = geometries[zone]
        point = geom.representative_point()
        overlap = geom.intersection(son_la).area / geom.area if geom.area else 0.0
        downstream = topology.get(zone, "")
        downstream_geom = geometries.get(downstream)
        if downstream_geom is None:
            outlet = point
            outlet_method = "representative_point"
        else:
            shared = geom.boundary.intersection(downstream_geom.boundary)
            if not shared.is_empty:
                outlet = shared.representative_point()
                outlet_method = "shared_boundary"
            else:
                outlet = nearest_points(geom, downstream_geom)[0]
                outlet_method = "nearest_downstream_polygon"
        # Keep the GloFAS lookup just inside the local basin while remaining near its outlet.
        glofas_lon = 0.95 * outlet.x + 0.05 * point.x
        glofas_lat = 0.95 * outlet.y + 0.05 * point.y
        rows.append(
            {
                "HYBAS_ID": zone,
                "selection_reason": (
                    "intersects_son_la" if zone in intersecting else "upstream_contributor"
                ),
                "son_la_overlap_fraction": overlap,
                "representative_longitude": point.x,
                "representative_latitude": point.y,
                "outlet_longitude": outlet.x,
                "outlet_latitude": outlet.y,
                "glofas_longitude": glofas_lon,
                "glofas_latitude": glofas_lat,
                "outlet_method": outlet_method,
            }
        )
    network = pd.DataFrame(rows)
    paths.processed.mkdir(parents=True, exist_ok=True)
    network.to_csv(paths.basin_network, index=False)
    selected = selected.drop(columns=["selection_reason"]).merge(
        network, on="HYBAS_ID", how="inner", validate="one_to_one"
    )
    selected.to_csv(paths.static_subset, index=False)
    print(
        f"geometry: {len(intersecting)} Son La polygons, {len(network)} including upstream; "
        f"wrote {paths.static_subset}"
    )
    return network


def _load_network(paths: AcquisitionPaths, static_path: str | Path) -> pd.DataFrame:
    if not paths.basin_network.exists():
        return build_basin_network(paths, static_path=static_path)
    frame = pd.read_csv(paths.basin_network, dtype={"HYBAS_ID": "string"})
    frame["HYBAS_ID"] = frame["HYBAS_ID"].map(normalise_hybas_id)
    return frame


def _chunks(values: Sequence[Any], size: int) -> Iterable[tuple[int, Sequence[Any]]]:
    for start in range(0, len(values), size):
        yield start // size, values[start : start + size]


def _request_json(
    url: str,
    params: dict[str, str],
    *,
    attempts: int = 6,
    timeout: int = 300,
) -> Any:
    error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            if response.status_code == 429 or response.status_code >= 500:
                retry_after = float(response.headers.get("Retry-After", "0") or 0)
                time.sleep(max(retry_after, min(60.0, 2.0**attempt)))
                continue
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            error = exc
            if attempt + 1 < attempts:
                time.sleep(min(60.0, 2.0**attempt))
    raise RuntimeError(f"API request failed after {attempts} attempts: {error}")


def _load_or_request_json(cache_path: Path, url: str, params: dict[str, str]) -> Any:
    if cache_path.exists():
        with gzip.open(cache_path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    payload = _request_json(url, params)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".part")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
    temporary.replace(cache_path)
    return payload


def download_weather(
    paths: AcquisitionPaths,
    network: pd.DataFrame,
    *,
    start_year: int,
    end_year: int,
    batch_size: int = 10,
) -> list[Path]:
    """Download hourly ERA5/ERA5-Land weather at basin representative points."""
    outputs: list[Path] = []
    records = network.sort_values("HYBAS_ID").to_dict("records")
    for year in range(start_year, end_year + 1):
        output = paths.processed / f"weather_history_{year}.parquet"
        outputs.append(output)
        if output.exists():
            print(f"weather {year}: cached {output}")
            continue
        frames: list[pd.DataFrame] = []
        for batch_index, batch in _chunks(records, batch_size):
            params = {
                "latitude": ",".join(f"{row['representative_latitude']:.6f}" for row in batch),
                "longitude": ",".join(f"{row['representative_longitude']:.6f}" for row in batch),
                "start_date": f"{year}-01-01",
                "end_date": f"{year}-12-31",
                "hourly": ",".join(WEATHER_VARIABLES),
                "models": "era5_seamless",
                "timezone": "GMT",
            }
            cache = paths.raw / "open_meteo" / f"weather_{year}_batch_{batch_index:03d}.json.gz"
            payload = _load_or_request_json(cache, OPEN_METEO_ARCHIVE_URL, params)
            payloads = payload if isinstance(payload, list) else [payload]
            if len(payloads) != len(batch):
                raise ValueError(
                    f"Open-Meteo returned {len(payloads)} locations for a batch of {len(batch)}."
                )
            for row, location in zip(batch, payloads, strict=True):
                hourly = pd.DataFrame(location["hourly"])
                hourly.insert(0, "HYBAS_ID", row["HYBAS_ID"])
                hourly = hourly.rename(columns={"time": "datetime"})
                hourly["datetime"] = pd.to_datetime(hourly["datetime"], utc=True)
                hourly["soil_moisture_1_to_3cm"] = hourly["soil_moisture_0_to_7cm"]
                hourly["weather_api_latitude"] = location["latitude"]
                hourly["weather_api_longitude"] = location["longitude"]
                frames.append(hourly)
            print(
                f"weather {year}: batch {batch_index + 1}/"
                f"{math.ceil(len(records) / batch_size)}"
            )
        annual = pd.concat(frames, ignore_index=True)
        annual = annual.sort_values(["HYBAS_ID", "datetime"]).reset_index(drop=True)
        if annual.duplicated(["HYBAS_ID", "datetime"]).any():
            raise ValueError(f"Duplicate weather keys detected for {year}.")
        expected = len(network) * (8784 if _is_leap(year) else 8760)
        if len(annual) != expected:
            raise ValueError(f"Weather {year} has {len(annual)} rows; expected {expected}.")
        annual.to_parquet(output, index=False, compression="zstd")
        print(f"weather {year}: wrote {len(annual):,} rows to {output}")
    return outputs


def download_weather_era5_aws(
    paths: AcquisitionPaths,
    network: pd.DataFrame,
    *,
    start_year: int,
    end_year: int,
) -> list[Path]:
    """Extract hourly weather from the public time-chunked ERA5 Icechunk store.

    The compatible public repository currently ends at 2024-12-31. Its precipitation
    arrays have already been flattened to per-hour forecast amounts, so unlike the raw
    SRO files they must not be differenced again.
    """
    import icechunk
    import xarray as xr

    storage = icechunk.s3_storage(
        bucket="earthmover-icechunk-era5",
        prefix="era5_surface_aws",
        region="us-east-1",
        anonymous=True,
    )
    repository = icechunk.Repository.open(storage)
    session = repository.readonly_session("main")
    dataset = xr.open_zarr(
        session.store,
        group="temporal",
        consolidated=False,
        chunks=None,
    )
    available_end = pd.Timestamp(dataset["time"].values[-1]).year
    if end_year > available_end:
        raise ValueError(
            f"The Python-3.11-compatible ERA5 temporal store ends in {available_end}; "
            f"requested {end_year}. Use --end-year {available_end}."
        )
    mapping, lat_index, lon_index = _era5_grid_mapping(dataset, network)
    mapping.to_csv(paths.processed / "era5_weather_grid_mapping.csv", index=False)
    zone_grid_index = mapping["era5_grid_index"].to_numpy(dtype=int)
    zones = mapping["HYBAS_ID"].astype(str).to_numpy()
    outputs: list[Path] = []
    try:
        for year in range(start_year, end_year + 1):
            output = paths.processed / f"weather_history_{year}.parquet"
            outputs.append(output)
            if output.exists():
                print(f"weather {year}: cached {output}")
                continue
            annual = dataset[["t2", "cp", "lsp", "swvl1"]].sel(
                time=slice(f"{year}-01-01", f"{year}-12-31T23:00:00")
            ).isel(
                latitude=xr.DataArray(lat_index, dims="grid"),
                longitude=xr.DataArray(lon_index, dims="grid"),
            ).load()
            times = pd.DatetimeIndex(annual["time"].to_numpy()).tz_localize("UTC")
            temperature = annual["t2"].to_numpy()[:, zone_grid_index] - 273.15
            convective = annual["cp"].to_numpy()[:, zone_grid_index] * 1000.0
            large_scale = annual["lsp"].to_numpy()[:, zone_grid_index] * 1000.0
            rain = np.maximum(convective + large_scale, 0.0)
            soil = annual["swvl1"].to_numpy()[:, zone_grid_index]
            frame = pd.DataFrame(
                {
                    "HYBAS_ID": np.repeat(zones, len(times)),
                    "datetime": np.tile(times.to_numpy(), len(zones)),
                    "temperature_2m": temperature.T.reshape(-1),
                    "rain": rain.T.reshape(-1),
                    "precipitation": rain.T.reshape(-1),
                    "soil_moisture_1_to_3cm": soil.T.reshape(-1),
                    "soil_moisture_0_to_7cm": soil.T.reshape(-1),
                    "weather_source": "NSF-NCAR ERA5 via Earthmover temporal Icechunk",
                }
            )
            if not np.isfinite(
                frame[["temperature_2m", "rain", "soil_moisture_1_to_3cm"]]
            ).all().all():
                raise ValueError(f"ERA5 weather contains non-finite values in {year}.")
            expected = len(network) * (8784 if _is_leap(year) else 8760)
            if len(frame) != expected:
                raise ValueError(f"Weather {year} has {len(frame)} rows; expected {expected}.")
            frame.to_parquet(output, index=False, compression="zstd")
            print(f"weather {year}: wrote {len(frame):,} rows to {output}")
    finally:
        dataset.close()
    return outputs


def _is_leap(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def download_glofas(
    paths: AcquisitionPaths,
    network: pd.DataFrame,
    *,
    start_date: str = "1984-01-01",
    end_date: str = "2025-12-31",
    batch_size: int = 20,
    scope: str = "main-outlets",
) -> tuple[Path, Path]:
    """Download daily GloFAS v4 discharge and derive annual-maximum return thresholds."""
    if scope not in {"main-outlets", "all-basins"}:
        raise ValueError("scope must be 'main-outlets' or 'all-basins'.")
    records_frame = network.sort_values("HYBAS_ID")
    if scope == "main-outlets":
        static = pd.read_csv(paths.static_subset, dtype={"HYBAS_ID": "string"})
        static["HYBAS_ID"] = static["HYBAS_ID"].map(normalise_hybas_id)
        known = set(static["HYBAS_ID"])
        outside = ~static["NEXT_DOWN"].map(normalise_hybas_id).isin(known)
        monitoring = (
            static.loc[outside]
            .sort_values("UP_AREA", ascending=False)
            .groupby("MAIN_BAS", observed=True, as_index=False)
            .first()["HYBAS_ID"]
        )
        records_frame = records_frame.loc[records_frame["HYBAS_ID"].isin(set(monitoring))]
    suffix = "main_outlets" if scope == "main-outlets" else "all_basins"
    output = paths.processed / f"glofas_discharge_daily_{suffix}.parquet"
    threshold_output = paths.processed / f"glofas_return_thresholds_{suffix}.csv"
    if output.exists() and threshold_output.exists():
        print(f"glofas: cached {output}")
        return output, threshold_output
    records = records_frame.to_dict("records")
    frames: list[pd.DataFrame] = []
    for batch_index, batch in _chunks(records, batch_size):
        params = {
            "latitude": ",".join(f"{row['glofas_latitude']:.6f}" for row in batch),
            "longitude": ",".join(f"{row['glofas_longitude']:.6f}" for row in batch),
            "daily": "river_discharge",
            "start_date": start_date,
            "end_date": end_date,
            "models": "seamless_v4",
        }
        cache = (
            paths.raw
            / "glofas"
            / f"discharge_{suffix}_{start_date}_{end_date}_batch_{batch_index:03d}.json.gz"
        )
        payload = _load_or_request_json(cache, OPEN_METEO_FLOOD_URL, params)
        payloads = payload if isinstance(payload, list) else [payload]
        if len(payloads) != len(batch):
            raise ValueError(f"GloFAS returned {len(payloads)} locations for {len(batch)} inputs.")
        for row, location in zip(batch, payloads, strict=True):
            daily = pd.DataFrame(location["daily"])
            daily.insert(0, "HYBAS_ID", row["HYBAS_ID"])
            daily = daily.rename(columns={"time": "date"})
            daily["date"] = pd.to_datetime(daily["date"], utc=True)
            daily["glofas_api_latitude"] = location["latitude"]
            daily["glofas_api_longitude"] = location["longitude"]
            frames.append(daily)
        print(
            f"glofas: batch {batch_index + 1}/{math.ceil(len(records) / batch_size)}"
        )
    discharge = pd.concat(frames, ignore_index=True).sort_values(["HYBAS_ID", "date"])
    discharge.to_parquet(output, index=False, compression="zstd")
    annual = discharge.assign(year=discharge["date"].dt.year).groupby(
        ["HYBAS_ID", "year"], observed=True, as_index=False
    )["river_discharge"].max()
    probabilities = {"q_1_5yr_m3s": 1.0 - 1.0 / 1.5, "q_2yr_m3s": 0.5, "q_5yr_m3s": 0.8}
    threshold_rows: list[dict[str, Any]] = []
    for zone, group in annual.groupby("HYBAS_ID", observed=True):
        values = group["river_discharge"].dropna().to_numpy(dtype=float)
        row: dict[str, Any] = {"HYBAS_ID": zone, "annual_maxima_years": len(values)}
        for column, probability in probabilities.items():
            row[column] = float(np.quantile(values, probability)) if len(values) else np.nan
        threshold_rows.append(row)
    pd.DataFrame(threshold_rows).to_csv(threshold_output, index=False)
    print(f"glofas: wrote {len(discharge):,} daily rows and {len(threshold_rows)} thresholds")
    return output, threshold_output


def _next_month(year: int, month: int) -> tuple[int, int]:
    return (year + 1, 1) if month == 12 else (year, month + 1)


def _era5_objects(year: int) -> list[tuple[str, Path]]:
    objects: list[tuple[str, Path]] = []
    for month in range(1, 13):
        current = f"{year:04d}{month:02d}"
        next_year, next_month = _next_month(year, month)
        following = f"{next_year:04d}{next_month:02d}"
        stems = (
            f"{current}0106_{current}1606",
            f"{current}1606_{following}0106",
        )
        for stem in stems:
            filename = f"e5.oper.fc.sfc.accumu.128_008_sro.ll025sc.{stem}.nc"
            key = f"e5.oper.fc.sfc.accumu/{current}/{filename}"
            objects.append((f"{ERA5_BUCKET_URL}/{key}", Path(str(year)) / filename))
    return objects


def download_era5_runoff(
    paths: AcquisitionPaths,
    *,
    start_year: int,
    end_year: int,
    workers: int = 6,
) -> list[Path]:
    """Download public NSF-NCAR ERA5 surface-runoff NetCDF source files."""
    jobs: list[tuple[str, Path]] = []
    for year in range(start_year, end_year + 1):
        for url, relative in _era5_objects(year):
            jobs.append((url, paths.raw / "era5_sro" / relative))
    completed = 0
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_atomic_download, url, destination): (url, destination)
            for url, destination in jobs
        }
        for future in as_completed(futures):
            url, destination = futures[future]
            try:
                future.result()
                completed += 1
                if completed % 5 == 0 or completed == len(jobs):
                    print(f"era5 download: {completed}/{len(jobs)} files")
            except Exception as exc:  # keep the full batch resumable
                failures.append(f"{destination.name}: {exc}")
    if failures:
        raise RuntimeError("ERA5 downloads failed:\n" + "\n".join(failures[:20]))
    return [destination for _, destination in jobs]


def _era5_grid_mapping(dataset: Any, network: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    latitudes = dataset["latitude"].to_numpy()
    longitudes = dataset["longitude"].to_numpy()
    mapping = network[["HYBAS_ID", "representative_latitude", "representative_longitude"]].copy()
    mapping["era5_latitude_index"] = [
        int(np.abs(latitudes - value).argmin()) for value in mapping["representative_latitude"]
    ]
    mapping["era5_longitude_index"] = [
        int(np.abs(longitudes - value).argmin()) for value in mapping["representative_longitude"]
    ]
    mapping["era5_grid_latitude"] = latitudes[mapping["era5_latitude_index"].to_numpy()]
    mapping["era5_grid_longitude"] = longitudes[mapping["era5_longitude_index"].to_numpy()]
    pairs = list(
        zip(
            mapping["era5_latitude_index"].astype(int),
            mapping["era5_longitude_index"].astype(int),
            strict=True,
        )
    )
    unique_pairs = sorted(set(pairs))
    pair_to_index = {pair: index for index, pair in enumerate(unique_pairs)}
    mapping["era5_grid_index"] = [pair_to_index[pair] for pair in pairs]
    lat_index = np.asarray([pair[0] for pair in unique_pairs], dtype=int)
    lon_index = np.asarray([pair[1] for pair in unique_pairs], dtype=int)
    return mapping, lat_index, lon_index


def _interval_runoff_metres_to_mm(values: np.ndarray) -> np.ndarray:
    """Convert NCAR forecast-hour SRO interval amounts without differencing them."""
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError("ERA5 SRO contains non-finite interval amounts.")
    return np.maximum(array, 0.0) * 1000.0


def process_era5_runoff(
    paths: AcquisitionPaths,
    network: pd.DataFrame,
    *,
    start_year: int,
    end_year: int,
) -> list[Path]:
    """Convert hourly ERA5 SRO to mm and map basins to the nearest 0.25-degree cell.

    Although the NCAR directory is named ``accumu``, its forecast-hour values are
    already interval amounts (they are not monotonic within a 12-hour forecast cycle).
    Differencing them would therefore remove valid runoff pulses.
    """
    import xarray as xr

    first_file = paths.raw / "era5_sro" / _era5_objects(start_year)[0][1]
    if not first_file.exists():
        raise FileNotFoundError(f"Download ERA5 first; missing {first_file}")
    with xr.open_dataset(first_file, engine="h5netcdf") as first_dataset:
        mapping, lat_index, lon_index = _era5_grid_mapping(first_dataset, network)
    mapping.to_csv(paths.processed / "era5_grid_mapping.csv", index=False)
    zone_grid_index = mapping["era5_grid_index"].to_numpy(dtype=int)
    zones = mapping["HYBAS_ID"].astype(str).to_numpy()
    outputs: list[Path] = []
    audit_path = paths.processed / "era5_runoff_processing_audit.json"
    processing_version = 2
    existing_version = None
    if audit_path.exists():
        try:
            existing_version = json.loads(audit_path.read_text(encoding="utf-8")).get(
                "processing_version"
            )
        except (OSError, ValueError):
            existing_version = None
    force_reprocess = existing_version != processing_version
    audit: dict[str, Any] = {
        "processing_version": processing_version,
        "processing": "direct interval amount; clip negative packing noise; metres to mm",
        "source": "NSF-NCAR ERA5 atmospheric surface forecast (accumulated)",
        "source_resolution_degrees": 0.25,
        "unique_grid_cells": int(mapping["era5_grid_index"].nunique()),
        "basins": int(len(mapping)),
        "negative_raw_value_count": 0,
        "minimum_raw_value_m": 0.0,
    }
    for year in range(start_year, end_year + 1):
        output = paths.processed / f"era5_surface_runoff_{year}.parquet"
        outputs.append(output)
        if output.exists() and not force_reprocess:
            print(f"era5 runoff {year}: cached {output}")
            continue
        frames: list[pd.DataFrame] = []
        objects = _era5_objects(year)
        for file_index, (_, relative) in enumerate(objects, start=1):
            source = paths.raw / "era5_sro" / relative
            with xr.open_dataset(source, engine="h5netcdf") as dataset:
                sampled = dataset["SRO"].isel(
                    latitude=xr.DataArray(lat_index, dims="grid"),
                    longitude=xr.DataArray(lon_index, dims="grid"),
                ).load().to_numpy()
                initial = pd.DatetimeIndex(dataset["forecast_initial_time"].to_numpy())
                forecast_hour = dataset["forecast_hour"].to_numpy().astype(int)
            negative = sampled < 0
            if negative.any():
                audit["negative_raw_value_count"] += int(negative.sum())
                audit["minimum_raw_value_m"] = min(
                    float(audit["minimum_raw_value_m"]), float(sampled.min())
                )
            hourly_runoff = _interval_runoff_metres_to_mm(sampled)
            valid_time = (
                initial.to_numpy()[:, None]
                + forecast_hour.astype("timedelta64[h]")[None, :]
            )
            zone_values = hourly_runoff[:, :, zone_grid_index]
            frame = pd.DataFrame(
                {
                    "datetime": np.repeat(valid_time.reshape(-1), len(zones)),
                    "HYBAS_ID": np.tile(zones, valid_time.size),
                    "surface_runoff_mm_h": zone_values.reshape(-1),
                }
            )
            frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True)
            frames.append(frame)
            print(f"era5 runoff {year}: processed {file_index}/{len(objects)}")
        annual = pd.concat(frames, ignore_index=True)
        annual = annual.drop_duplicates(["HYBAS_ID", "datetime"]).sort_values(
            ["HYBAS_ID", "datetime"]
        )
        if not np.isfinite(annual["surface_runoff_mm_h"]).all():
            raise ValueError(f"ERA5 runoff contains non-finite values in {year}.")
        annual.to_parquet(output, index=False, compression="zstd")
        print(f"era5 runoff {year}: wrote {len(annual):,} rows to {output}")
    audit_path.write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    return outputs


def combine_observations(
    paths: AcquisitionPaths,
    *,
    start_year: int,
    end_year: int,
) -> Path:
    """Merge weather and runoff into the observation panel consumed by the model."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    destination = paths.processed / "observations_hourly.parquet"
    temporary = destination.with_suffix(".parquet.part")
    writer: pq.ParquetWriter | None = None
    total_rows = 0
    try:
        for year in range(start_year, end_year + 1):
            weather_path = paths.processed / f"weather_history_{year}.parquet"
            runoff_paths = [paths.processed / f"era5_surface_runoff_{year}.parquet"]
            if year > start_year:
                runoff_paths.insert(0, paths.processed / f"era5_surface_runoff_{year - 1}.parquet")
            weather = pd.read_parquet(weather_path)
            runoff = pd.concat([pd.read_parquet(path) for path in runoff_paths], ignore_index=True)
            start = pd.Timestamp(f"{year}-01-01", tz="UTC")
            end = pd.Timestamp(f"{year + 1}-01-01", tz="UTC")
            runoff = runoff.loc[(runoff["datetime"] >= start) & (runoff["datetime"] < end)]
            runoff = runoff.drop_duplicates(["HYBAS_ID", "datetime"])
            weather["HYBAS_ID"] = weather["HYBAS_ID"].map(normalise_hybas_id)
            runoff["HYBAS_ID"] = runoff["HYBAS_ID"].map(normalise_hybas_id)
            merged = weather.merge(
                runoff,
                on=["HYBAS_ID", "datetime"],
                how="inner",
                validate="one_to_one",
            ).sort_values(["HYBAS_ID", "datetime"])
            if merged.empty:
                raise ValueError(f"Weather/runoff merge is empty for {year}.")
            table = pa.Table.from_pandas(merged, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
            writer.write_table(table, row_group_size=100_000)
            total_rows += len(merged)
            print(f"observations {year}: merged {len(merged):,} rows")
    finally:
        if writer is not None:
            writer.close()
    temporary.replace(destination)
    print(f"observations: wrote {total_rows:,} rows to {destination}")
    return destination


def write_manifest(
    paths: AcquisitionPaths,
    *,
    start_year: int,
    end_year: int,
) -> Path:
    file_paths = {
        "static": paths.static_subset,
        "basin_locations": paths.basin_network,
        "observations": paths.processed / "observations_hourly.parquet",
        "glofas_daily": paths.processed / "glofas_discharge_daily_main_outlets.parquet",
        "river_thresholds": paths.processed / "glofas_return_thresholds_main_outlets.csv",
    }
    manifest = {
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "study_area": "Son La Province and contributing upstream HydroBASINS level-12 zones",
        "period": {"start_year": start_year, "end_year": end_year},
        "files": {
            name: {"path": str(path), "available": path.exists()}
            for name, path in file_paths.items()
        },
        "sources": {
            "hydrobasins": HYDROBASINS_URL,
            "province_boundary": VNM_ADM1_URL,
            "weather": "s3://earthmover-icechunk-era5/era5_surface_aws (temporal group)",
            "runoff": "s3://nsf-ncar-era5/e5.oper.fc.sfc.accumu (parameter 128.008 SRO)",
            "river_discharge": OPEN_METEO_FLOOD_URL,
        },
        "limitations": [
            "ERA5 SRO is 0.25 degree, so multiple level-12 basins share one runoff grid cell.",
            "ERA5 weather and runoff use nearest 0.25-degree cells, not polygon-area averages.",
            "GloFAS is daily river discharge near a basin outlet and is not the hourly runoff target.",
            "The merged history supports oracle-exogenous backtesting; deployable forecasts still need issue-time NWP.",
        ],
    }
    destination = paths.processed / "data_manifest.json"
    destination.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Acquire public Son La flood-model data.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--static", default="static_features.csv")
    parser.add_argument("--start-year", type=int, default=2021)
    parser.add_argument("--end-year", type=int, default=2024)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("geometry")
    weather = subparsers.add_parser("weather")
    weather.add_argument("--batch-size", type=int, default=10)
    weather.add_argument(
        "--source",
        choices=("era5-aws", "open-meteo"),
        default="era5-aws",
    )
    glofas = subparsers.add_parser("glofas")
    glofas.add_argument("--batch-size", type=int, default=20)
    glofas.add_argument(
        "--scope", choices=("main-outlets", "all-basins"), default="main-outlets"
    )
    runoff = subparsers.add_parser("era5-runoff")
    runoff.add_argument("--workers", type=int, default=6)
    subparsers.add_parser("combine")
    all_parser = subparsers.add_parser("all")
    all_parser.add_argument("--weather-batch-size", type=int, default=10)
    all_parser.add_argument("--glofas-batch-size", type=int, default=20)
    all_parser.add_argument("--workers", type=int, default=6)
    return parser


def _validate_years(start_year: int, end_year: int) -> None:
    if start_year < 1950 or end_year < start_year:
        raise ValueError("Require 1950 <= start_year <= end_year.")
    if end_year >= date.today().year:
        raise ValueError("Use completed historical years only.")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    _validate_years(args.start_year, args.end_year)
    root = Path(args.root).resolve()
    paths = AcquisitionPaths(root)
    paths.processed.mkdir(parents=True, exist_ok=True)
    if args.command == "geometry":
        build_basin_network(paths, static_path=root / args.static)
        return
    network = _load_network(paths, root / args.static)
    if args.command == "weather":
        if args.source == "era5-aws":
            download_weather_era5_aws(
                paths,
                network,
                start_year=args.start_year,
                end_year=args.end_year,
            )
        else:
            download_weather(
                paths,
                network,
                start_year=args.start_year,
                end_year=args.end_year,
                batch_size=args.batch_size,
            )
        return
    if args.command == "glofas":
        download_glofas(paths, network, batch_size=args.batch_size, scope=args.scope)
        return
    if args.command == "era5-runoff":
        download_era5_runoff(
            paths,
            start_year=args.start_year,
            end_year=args.end_year,
            workers=args.workers,
        )
        process_era5_runoff(
            paths,
            network,
            start_year=args.start_year,
            end_year=args.end_year,
        )
        return
    if args.command == "combine":
        combine_observations(paths, start_year=args.start_year, end_year=args.end_year)
        write_manifest(paths, start_year=args.start_year, end_year=args.end_year)
        return
    download_weather_era5_aws(
        paths,
        network,
        start_year=args.start_year,
        end_year=args.end_year,
    )
    download_glofas(paths, network, batch_size=args.glofas_batch_size, scope="main-outlets")
    download_era5_runoff(
        paths,
        start_year=args.start_year,
        end_year=args.end_year,
        workers=args.workers,
    )
    process_era5_runoff(
        paths,
        network,
        start_year=args.start_year,
        end_year=args.end_year,
    )
    combine_observations(paths, start_year=args.start_year, end_year=args.end_year)
    write_manifest(paths, start_year=args.start_year, end_year=args.end_year)


if __name__ == "__main__":
    main()
