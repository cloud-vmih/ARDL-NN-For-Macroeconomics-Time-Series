# Son La flood data

The processed training panel covers the 140 HydroBASINS level-12 polygons that
intersect Sơn La plus 43 contributing upstream polygons. All model timestamps are
UTC and all identifiers use `HYBAS_ID`.

## Processed files

- `processed/static_features_sonla_upstream.csv`: 183 basin records, topology,
  original terrain/soil features, representative points and outlet proxies.
- `processed/weather_history_YYYY.parquet`: hourly ERA5 temperature, precipitation
  and top-layer soil moisture for 2021–2024.
- `processed/era5_surface_runoff_YYYY.parquet`: hourly surface runoff in mm/h for
  2021–2025. The extra 2025 target is retained for later use when matching weather
  becomes available.
- `processed/observations_hourly.parquet`: the model-ready weather/runoff merge for
  2021–2024.
- `processed/era5_grid_mapping.csv` and `era5_weather_grid_mapping.csv`: explicit
  basin-to-grid mappings used to audit spatial resolution.
- `processed/glofas_*_main_outlets.*`: optional daily discharge and return-period
  proxies for the two main provincial outlets. These files are absent until the
  Open-Meteo Flood API quota permits the optional download.

## Reproduce or resume

```bash
./.venv/bin/python3 download_sonla_flood_data.py geometry
./.venv/bin/python3 download_sonla_flood_data.py \
  --start-year 2021 --end-year 2024 weather --source era5-aws
./.venv/bin/python3 download_sonla_flood_data.py \
  --start-year 2021 --end-year 2025 era5-runoff --workers 6
./.venv/bin/python3 download_sonla_flood_data.py \
  --start-year 2021 --end-year 2024 combine
```

Optional GloFAS history:

```bash
./.venv/bin/python3 download_sonla_flood_data.py glofas --scope main-outlets
```

## Important limitations

ERA5 has a 0.25-degree grid here: the 183 small basin polygons map to only 42
distinct cells. Treat level-12 basin predictions as spatially correlated proxies,
not independent high-resolution runoff measurements. The history is suitable for
an oracle-exogenous backtest. A deployable 1–24 hour warning system still requires
an issue-time NWP panel keyed by `(HYBAS_ID, issue_time, valid_time)`.
