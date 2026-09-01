# Son La ARDL–FFNN Flood-Proxy Pipeline

This module implements an hourly panel pipeline for catchment runoff and a transparent
flood-risk proxy. It is independent of the existing monthly VMD experiment.

```text
issue-time NWP + observed basin state
→ per-zone sequential ARDL baseline
→ pooled FFNN residual correction
→ non-negative runoff
→ NEXT_DOWN linear-reservoir routing
→ runoff/river proxy alert
```

The output is **not an official flood or water-level warning**. Without local gauge data,
cross-sections and rating curves, it represents modelled runoff, routed discharge and a
model-proxy risk level only.

## Data audits

Run from the repository root:

```bash
./.venv/bin/python3 run_flood_pipeline.py audit \
  --static static_features.csv \
  --observations weather.csv
```

The original input contains 342 zones and 504 hourly timestamps per zone, with no missing
values or duplicate `(HYBAS_ID, datetime)` keys. It does not contain the required target.

The acquired panel is model-ready for an oracle-exogenous historical backtest:

```bash
./.venv/bin/python3 run_flood_pipeline.py audit \
  --static data/processed/static_features_sonla_upstream.csv \
  --observations data/processed/observations_hourly.parquet
```

Its audited coverage is 183 zones × 35,057 contiguous hours from 2021-01-01 07:00 UTC
through 2024-12-31 23:00 UTC: 6,415,431 rows, no missing values, duplicate keys or
non-hourly gaps. The target is NSF-NCAR ERA5 surface runoff rather than a gauge
observation.

## Data contracts

All timestamps are parsed and stored internally as UTC. Local month/hour features use
`Asia/Ho_Chi_Minh`.

### Basin static table

Required by the default configuration:

```text
HYBAS_ID, NEXT_DOWN, SUB_AREA, UP_AREA, ORDER,
min_elevation_m, max_elevation_m, mean_elevation_m, soil_name
```

- `HYBAS_ID` and `NEXT_DOWN` are identifiers/topology, not continuous model features.
- `SUB_AREA` is in km².
- Optional `travel_hours` overrides the default linear-reservoir residence time.
- Add slope, flow length, land cover, hydraulic soil properties and reservoir attributes
  to `--static-numeric-cols` after they have been area-aggregated.

First intersect official HydroBASINS polygons with the Son La administrative boundary.
Pass those intersecting IDs to `select_upstream_network()` to retain every contributing
upstream basin. `effective_grid_coverage()` flags zones represented by fewer than three
effective 0.1° raster cells; merge these zones before claiming independent fine-scale skill.

### Hourly observations

```text
HYBAS_ID, datetime,
surface_runoff_mm_h,
rain, temperature_2m, soil_moisture_1_to_3cm, ...
```

Current source mapping:

- NSF-NCAR ERA5 `SRO` at 0.25° → target in mm/h.
- NSF-NCAR ERA5 convective + large-scale precipitation → `rain` in mm/h.
- NSF-NCAR ERA5 2 m temperature and soil-water layer 1 → land-state inputs.

Use `convert_water_depth_metres_to_mm()` for water-depth unit conversion. Accumulated
fields must be de-accumulated per source forecast cycle; never difference across cycles.
Check actual source semantics first: the NCAR forecast-hour SRO files used here already
contain interval amounts despite residing under an `accumu` directory, so they are
converted directly from metres to millimetres without differencing.

### Issue-time NWP forecasts

```text
HYBAS_ID, issue_time, valid_time, lead_hour,
rain, temperature_2m, soil_moisture_1_to_3cm, ...
```

Each `(HYBAS_ID, issue_time)` must contain a contiguous hourly path from +1 through the
requested horizon. `valid_time` must be strictly later than `issue_time`; a supplied
`lead_hour` must equal their difference. Use `deaccumulate_forecast_fields()` for GFS/ECMWF
cumulative precipitation.

For method-development only, `--oracle-exogenous` constructs the future exogenous path
from observations. The run manifest marks this mode and it must not be reported as an
operational 1–24 hour forecast.

### Optional river thresholds

```text
HYBAS_ID,q_1_5yr_m3s,q_2yr_m3s,q_5yr_m3s
```

Derive all columns from one pinned GloFAS version. GloFAS is daily, so the pipeline compares
them against mean routed discharge over the 24-hour forecast path. Missing thresholds are
flagged and do not silently create a river warning.

## Running an experiment

The acquired history can first be run in explicitly labelled oracle mode:

```bash
./.venv/bin/python3 run_flood_pipeline.py run \
  --static data/processed/static_features_sonla_upstream.csv \
  --observations data/processed/observations_hourly.parquet \
  --oracle-exogenous \
  --train-end 2022-12-31T23:00:00Z \
  --validation-end 2023-12-31T23:00:00Z \
  --test-end 2024-12-31T23:00:00Z \
  --output-dir results/flood-proxy-oracle
```

This is an upper-bound retrospective experiment, not an operational forecast.

For deployment-like evaluation, supply issue-time NWP instead:

```bash
./.venv/bin/python3 run_flood_pipeline.py run \
  --static data/basin_static.csv \
  --observations data/basin_observation_hourly.csv \
  --forecasts data/basin_nwp_hourly.csv \
  --train-end 2023-12-31T23:00:00+07:00 \
  --validation-end 2024-12-31T23:00:00+07:00 \
  --test-end 2025-12-31T23:00:00+07:00 \
  --river-thresholds data/glofas_thresholds.csv \
  --output-dir results/flood-proxy
```

Important options:

- `--exog-cols`: logical observation feature names used by ARDL.
- `--nwp-cols`: forecast columns mapped positionally to `--exog-cols`; omit when names match.
- `--max-target-lag` / `--max-exog-lag`: sequential AIC selection bounds.
- `--max-horizon`: defaults to 24 hourly steps.
- `--state-mode recursive`: primary deployment-like test; predicted runoff becomes the next
  issue's autoregressive state. `observed` is a teacher-forced diagnostic only.

Every execution creates a timestamped run directory containing:

- data and leakage audit;
- per-zone ARDL order-selection table;
- residual FFNN training summary;
- runoff forecasts with validation-calibrated 90% intervals;
- routed proxy warnings and a mass-balance audit;
- train-only runoff thresholds;
- runoff/peak metrics for persistence, standalone ARDL, fixed-lag FFNN,
  ARDL-guided direct FFNN and the residual hybrid;
- proxy-alert POD, FAR, CSI, F1 and AUPRC;
- an acceptance table and a full manifest.

The hybrid passes each model gate only when it has lower mean test RMSE and higher mean
test KGE than that comparator. Failure is retained in the output; the experiment does not
relabel a losing model as the production winner.

## Open-data acquisition order

The current public-source acquisition is reproducible and resumable:

```bash
./.venv/bin/python3 download_sonla_flood_data.py geometry
./.venv/bin/python3 download_sonla_flood_data.py \
  --start-year 2021 --end-year 2024 weather --source era5-aws
./.venv/bin/python3 download_sonla_flood_data.py \
  --start-year 2021 --end-year 2025 era5-runoff --workers 6
./.venv/bin/python3 download_sonla_flood_data.py \
  --start-year 2021 --end-year 2024 combine
```

The 183 level-12 zones map to only 42 distinct 0.25° ERA5 cells. Results must therefore
be described as spatially correlated runoff proxies; they do not support claims of 183
independent fine-resolution measurements.

1. HydroBASINS polygons/pour points and a Son La boundary for spatial selection.
2. ERA5-Land hourly surface/subsurface runoff and soil-water layers.
3. IMERG half-hourly rainfall for observed precipitation.
4. Archived issue-time GFS forecasts for train/validation/test and live ingestion.
5. One pinned GloFAS historical version for daily outlet calibration and return thresholds.
6. HydroATLAS/DEM, SoilGrids and ESA WorldCover static attributes.

Install optional GIS readers only when building polygon/raster overlap weights:

```bash
./.venv/bin/pip install -e './vmd_ardl_ffnn[flood-geo]'
```

The core model and all tests do not require those large geospatial dependencies.
