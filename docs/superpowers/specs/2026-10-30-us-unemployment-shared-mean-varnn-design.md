# US Unemployment Shared Mean Benchmark Design

## Objective

Rebuild the US monthly unemployment panel from auditable BLS/FRED series and
run the existing shared mean benchmark with VAR-NN and a naïve reference. The
invalid `df_final.csv` and its benchmark artifacts are outside this study.

## Forecast contract

- Target: US seasonally adjusted unemployment rate, FRED `UNRATE`, in percentage
  points. Forecast the value for reference month `t` at 23:59 New York time on
  the last calendar day of `t`, before its Employment Situation release.
- Inputs: target and exogenous observations from months no later than `t-1`.
  The benchmark forbids lag zero. A source value is usable only if its release
  or vintage was available by the forecast origin.
- Main panel: January 2000 through September 2025, 309 calendar months. Keep
  the existing 216/46/47 train/validation/test split. Do not fill October 2025:
  BLS did not collect its household-survey estimates. Extending past this gap
  requires a separate missing-month design and new split registration.
- Main metric: level RMSE on identical forecast dates. Report MAE, MAPE and
  directional accuracy too. All model and feature decisions use train and
  validation only; test is opened after signatures are locked.

## Source contract

| Model field | FRED series | Frequency and unit | Construction |
| --- | --- | --- | --- |
| `unemployment_rate` | `UNRATE` | Monthly, percent, seasonally adjusted | First published value for each reference month |
| `manufacturing_hours` | `AWHMAN` | Monthly, hours, seasonally adjusted | Level |
| `nonfarm_payrolls` | `PAYEMS` | Monthly, thousands of persons, seasonally adjusted | Level |
| `cpi_all_items` | `CPIAUCSL` | Monthly, 1982–84=100, seasonally adjusted | Level |
| `industrial_prod` | `INDPRO` | Monthly, 2017=100, seasonally adjusted | Level |

Use FRED/ALFRED observation vintages via `fred/series/observations`. `AWHMAN`
replaces `HOUST` because housing starts for September–October 2013 and
December 2018–January 2019 were unavailable at their forecast origins. `ICSA`
also cannot support point-in-time features for the 2000–2009 training period:
ALFRED's release history begins in May 2009.
Freeze each
exogenous observation for month `m` at the vintage visible at the end of month
`m+1`, its earliest permitted use in this one-month-lag design. For target
labels, use the initial-release value. The builder must verify that each
`t-1` value was actually available at the forecast origin for `t`. Require
the exact reference month for every exogenous series. Fail on missing
observations; do not substitute an older month, interpolate, or use a later
vintage. Record source
series IDs, units, aggregation, as-of dates, retrieval time, raw-response hashes
and panel hash. Read `FRED_API_KEY` from the environment; never store it in
artifacts or logs.

Exception for the final September 2025 row: its exogenous values were delayed
by the federal shutdown, so fetch them at the January 31, 2026 vintage and
mark them **unused in forecasts** in the audit. All exogenous lags are at least
one month, and October 2025 has no target label; therefore this trailing row
cannot enter a forecast in the registered horizon. The September target label
is likewise evaluated only after its first release. Use paced API requests
and retry 429/5xx responses because the cold cache requires over 1,200 calls.

Source references:

- FRED series: `https://fred.stlouisfed.org/series/UNRATE`, `/AWHMAN`, `/PAYEMS`,
  `/CPIAUCSL`, `/INDPRO`.
- FRED real-time API: `https://fred.stlouisfed.org/docs/api/fred/realtime_period.html`
  and `https://fred.stlouisfed.org/docs/api/fred/series_observations.html`.
- BLS missing October 2025: `https://www.bls.gov/cps/methods/2025-federal-government-shutdown-impact-cps.htm`.
- BLS delayed September 2025 release: `https://www.bls.gov/news.release/archives/empsit_11202025.htm`.
- ALFRED ICSA history: `https://alfred.stlouisfed.org/series?seid=ICSA`.
- BLS manufacturing workweek as a leading indicator: `https://www.bls.gov/opub/mlr/2020/article/employment-expansion-continued-in-2019-but-growth-slowed-in-several-industries.htm`.
- Census 2013 housing-start delay: `https://www.census.gov/construction/nrc/announcements.html`.
- Census 2018 shutdown release: `https://www.census.gov/construction/nrc/pdf/newresconst_201812.pdf`.
- Census 2025 release history: `https://www.census.gov/construction/nrc/data/releases.html`.
- FRED rate limit: `https://fred.stlouisfed.org/docs/api/fred/errors.html`.

## Benchmark contract

Use `data/macro_us/processed/us_unemployment_awhman_2000_2025_09.csv` as the immutable
run input with columns `date`, `unemployment_rate`, `manufacturing_hours`,
`nonfarm_payrolls`, `cpi_all_items`, `industrial_prod`. Keep four existing
shared-mean workflows: ARDL–FFNN no VMD, VMD block-subset two-stage Proposed,
VMD sequential screen, and VMD joint select. Add `varnn` using the existing
`_run_varnn_shared_mean_seed` helper. Add `naive_last_observation` as a
deterministic reference, reported separately from the five stochastic models.

All stochastic workflows use the same reporting seeds `(7, 42, 123, 2024,
2025)`. VAR-NN must select one architecture/learning-rate signature by mean
level validation RMSE over these seeds, then refit that signature separately
for each seed. The naïve forecast uses `unemployment_rate[t-1]`; it has no seed
or tuning. Include all six pipelines in forecast coverage, metric tables and
pairwise DM tests. Compute VAR-NN directional accuracy from its per-seed level
forecasts so the metric is populated in the same reporting table. A run fails
if a required model lacks a reporting seed or test date. Do not silently skip
VAR-NN when TensorFlow is unavailable.

## CLI and artifacts

Add an explicit `--dataset-profile us-unemployment` branch to
`shared-mean-benchmark`; do not infer profile from filename. It constructs
the macro `ExperimentConfig` with no log transform and the fixed four source
fields. The run requires an explicit `--data` path and records panel hash,
source-manifest hash, profile, forecast-origin rule and model set in its
manifest. Keep legacy CLI profiles callable, but do not use old `df_final.csv`
results as evidence for this study.

Keep current shared-mean output names for metrics, forecasts, settings,
coverage and DM tests. Write VAR-NN search/selection artifacts under
`<run>/varnn/`; write the data audit next to the run manifest. Update
`macro-benchmark`'s advertised model set if it inherits the expanded shared
runner, so manifest metadata matches the actual outputs.

## Acceptance

1. An offline fixture proves source-to-field mapping, monthly AWHMAN values,
   vintage cutoffs, manifest hashes and missing-month rejection.
2. A tiny benchmark fixture proves five stochastic models plus naïve appear in
   tables, each stochastic model covers every requested seed, and all six cover
   the same validation/test dates.
3. VAR-NN's selected signature is identical across reporting seeds; selection
   reads validation only. The test set is not used to choose a method.
4. The full test suite passes and one small smoke run completes before a full
   five-seed production run is started.

## Research interpretation

The fixed validation period contains the COVID-19 shock while the test period
is much calmer. Report this shift and compare the Proposed with the naïve
reference. After the main run, pre-register additional rolling-origin periods
as a robustness study; do not choose a favorable period after seeing its test
score. No claim of Proposed superiority follows from the invalid export panel.
