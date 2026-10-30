# US Unemployment Shared Mean Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an auditable BLS/FRED unemployment panel and compare the four existing shared-mean workflows with VAR-NN and a naïve reference on identical dates.

**Architecture:** A focused source module fetches and caches FRED/ALFRED vintages, then writes an immutable six-column monthly panel and source audit. The existing shared-mean orchestrator reuses its VAR-NN helper, adds a deterministic naïve baseline, and writes all six models into the same reporting tables. An explicit CLI dataset profile selects the unemployment configuration without changing the legacy export profile.

**Tech Stack:** Python 3.11, pandas, NumPy, statsmodels, TensorFlow/Keras, sktime, FRED/ALFRED HTTPS API, pytest.

**Spec:** `docs/superpowers/specs/2026-10-30-us-unemployment-shared-mean-varnn-design.md`

## Global Constraints

- `df_final.csv` and all results derived from it are excluded from the new study.
- Target is FRED `UNRATE`; exogenous series are `ICSA`, `PAYEMS`, `CPIAUCSL`, `INDPRO`.
- Forecast origin is the end of reference month `t` in New York; all model inputs must have been available by that origin and be dated no later than `t-1`.
- Main panel is monthly January 2000–September 2025, with 216 train, 46 validation and 47 test observations. Do not impute October 2025.
- Five stochastic workflows use seeds `(7, 42, 123, 2024, 2025)` and one shared validation-selected signature per workflow. The naïve reference has no seed.
- Do not select a Proposed method or revise feature rules using test results.
- Preserve the current uncommitted notebook, runner change, input files and results; stage only files owned by the implementation task.

---

## File map

| File | Responsibility |
| --- | --- |
| `vmd_ardl_ffnn/src/vmd_ardl_ffnn/macro_fred_source.py` | FRED request/parse/cache and source-series contract |
| `vmd_ardl_ffnn/src/vmd_ardl_ffnn/macro_fred_panel.py` | Monthly vintage panel, weekly ICSA aggregation, availability audit |
| `vmd_ardl_ffnn/src/vmd_ardl_ffnn/shared_mean_benchmark_experiments.py` | Six-model comparison, coverage and reporting |
| `vmd_ardl_ffnn/src/vmd_ardl_ffnn/block_proposed_experiments.py` | Shared VAR-NN helper and per-seed directional accuracy |
| `run_pipeline_advanced.py` | Explicit dataset profile and VARNN config wiring |
| `vmd_ardl_ffnn/src/vmd_ardl_ffnn/experiment_fixed.py` | Forward VAR-NN config into shared-mean orchestrator |
| `vmd_ardl_ffnn/src/vmd_ardl_ffnn/macro_benchmark.py` | Keep macro profile's advertised model set in sync |
| `vmd_ardl_ffnn/tests/test_macro_fred.py` | Offline source and panel tests |
| `vmd_ardl_ffnn/tests/test_refactor.py`, `tests/test_macro_benchmark.py` | Benchmark/CLI integration tests |
| `README.md`, `vmd_ardl_ffnn/docs/MACRO_BENCHMARK_PLAN.md` | Reproducible commands and interpretation |

## Task 1: Fetch and cache the exact source series

**Files:** Create `macro_fred_source.py`, `tests/test_macro_fred.py`; modify `vmd_ardl_ffnn/pyproject.toml` only if an additional HTTP dependency is chosen.

**Interfaces:** Produce `SERIES_SPECS: tuple[SeriesSpec, ...]` and `fetch_series_snapshot(series_id: str, as_of: date | None, *, output_type: int = 1, api_key: str, cache_root: Path, transport: Callable) -> pd.DataFrame`. `as_of=None, output_type=4` retrieves the target's initial release; a specified `as_of` sets both `realtime_start` and `realtime_end` for exogenous vintages.

- [ ] Write an offline test whose fake transport checks `series_id`, `realtime_start`, `realtime_end`, `output_type` and `file_type=json`; return one `observations` JSON fixture with date/value/vintage fields. Assert values parse as floats, `.` becomes missing, and the API key is absent from cached JSON and errors.
- [ ] Run `cd vmd_ardl_ffnn && ../.venv/bin/python3 -m pytest tests/test_macro_fred.py -q`; confirm the test fails because `macro_fred_source.py` is absent.
- [ ] Implement the five fixed `SeriesSpec` entries, the API request, parser and file cache. Use `FRED_API_KEY` only from the environment at the CLI boundary. The cache key must include series ID, as-of date, output type and schema version. Write payload and SHA-256 metadata through sibling temporary files followed by `Path.replace()`.
- [ ] Re-run the focused test and add a second case proving a cache hit makes zero transport calls. Commit only the new source and test files if the checkout permits commits.

## Task 2: Construct and audit the point-in-time panel

**Files:** Create `macro_fred_panel.py`; expand `tests/test_macro_fred.py`.

**Interfaces:** Produce `build_us_unemployment_panel(cache_root: Path, output_root: Path, *, start: str = "2000-01-01", end: str = "2025-09-01") -> tuple[Path, Path]`, returning panel CSV and audit JSON paths. The output columns are exactly `date,unemployment_rate,initial_claims,nonfarm_payrolls,cpi_all_items,industrial_prod`. The module's `main()` reads `FRED_API_KEY`, fills the source cache via `fetch_series_snapshot`, then builds the panel; invoke it as `python -m vmd_ardl_ffnn.macro_fred_panel --cache-root data/macro_us/raw --output-root data/macro_us/processed`.

- [ ] Add a failing fixture with two months of UNRATE, PAYEMS, CPIAUCSL and INDPRO plus four ICSA week-ending Saturdays. Assert `initial_claims` is their arithmetic mean for the reference month and all source units/series IDs appear in the audit.
- [ ] Add a failing availability fixture: a PAYEMS value first visible after the forecast origin must cause a named exception with series ID and month. Add a missing-UNRATE fixture for October 2025 and assert the builder refuses to fill it.
- [ ] Run `cd vmd_ardl_ffnn && ../.venv/bin/python3 -m pytest tests/test_macro_fred.py -q`; confirm the new cases fail for the intended missing behavior.
- [ ] Implement month-start indexing and the end-of-month New York origin. Freeze exogenous observation `m` from the vintage visible at the end of `m+1`; use the initial-release UNRATE label. Verify all `t-1` values were published by origin `t`. Reject duplicate months, missing model values, nonfinite values and mismatched calendars. Do not interpolate or forward-fill.
- [ ] Write `data/macro_us/processed/us_unemployment_2000_2025_09.csv` plus a JSON audit with hashes for raw snapshots and panel, retrieval timestamp, series IDs, units, ICSA aggregation, as-of dates, missing counts and first/last dates. Verify 309 rows and the three registered split counts. Re-run focused tests and commit task files if permitted.

## Task 3: Add VAR-NN under the shared mean selector

**Files:** Modify `shared_mean_benchmark_experiments.py`, `block_proposed_experiments.py`, `experiment_fixed.py`, `run_pipeline_advanced.py`; expand `tests/test_refactor.py`.

**Interfaces:** Extend `run_shared_mean_benchmark_experiments(..., varnn_config: VARNNConfig | None = None)` and the forwarding methods. Reuse `_run_varnn_shared_mean_seed` from `block_proposed_experiments.py`, which returns `(results, metrics, forecasts, ranking, fit_count)`.

- [ ] Add a failing orchestrator test with four mocked existing runners and a mocked `_run_varnn_shared_mean_seed`. Assert `varnn` appears exactly once in seed metrics, forecasts, settings, efficiency and `variant_results`; assert every requested seed is present.
- [ ] Add a failing test where VAR-NN omits one seed or validation/test date; assert the benchmark raises instead of writing a successful comparison.
- [ ] Add a failing helper test proving VAR-NN's level `Directional_Accuracy` is computed per seed from its normalized level forecasts. Its current `metrics` table leaves this column absent.
- [ ] Run `cd vmd_ardl_ffnn && ../.venv/bin/python3 -m pytest tests/test_refactor.py -k 'shared_mean or varnn' -q`; confirm failures match the new expectations.
- [ ] Add `varnn` to `SHARED_MEAN_BENCHMARK_PIPELINES`. Call the existing shared VAR-NN helper after the four FFNN workflows, append its level metrics and normalized forecasts, record its shared selection rule and fit count, and include its selection table under `<run>/varnn/`. In the helper, calculate per-seed level `Directional_Accuracy` with the project `evaluate_forecast` function on each split's normalized `actual`/`predicted` arrays. Pass `VARNNConfig` through `VMDARDLFFNNExperiment.run_shared_mean_benchmark()` and `run_pipeline_advanced.run_pipeline()`.
- [ ] Preserve complete-seed and same-date checks. Raise on TensorFlow import failure for this six-model study; do not mark it successful with VAR-NN missing. Re-run focused tests and commit task files if permitted.

## Task 4: Add the naïve reference and enforce equal forecast coverage

**Files:** Modify `shared_mean_benchmark_experiments.py`; expand `tests/test_refactor.py`.

**Interfaces:** Reuse `run_naive_baseline(csv_path, replace(config, output_dir=root / "naive_last_observation"), lag=1, model_name="Naive_LastObservation")` from `paper_baselines.py`.

- [ ] Add a failing fixture proving the naïve prediction on date `t` equals actual target at `t-1` and that its `seed` is empty. Assert it appears once in comparison metrics, forecasts, coverage and pairwise DM tests.
- [ ] Run the focused shared-mean tests and confirm the naïve row is missing.
- [ ] Register the deterministic baseline separately from the five-model seed-coverage loop. Include it in `paper_seed_metrics.csv`, benchmark metrics/forecasts, figures, settings and DM inputs. Validate that all six models cover exactly the registered validation and test date sets; fail with missing-date details otherwise.
- [ ] Run focused tests. Confirm the mean/std aggregate treats the deterministic model as one observation and does not invent five seeds. Commit task files if permitted.

## Task 5: Expose an explicit unemployment profile and align macro metadata

**Files:** Modify `run_pipeline_advanced.py`, `macro_benchmark.py`, `README.md`, `vmd_ardl_ffnn/docs/MACRO_BENCHMARK_PLAN.md`; expand `tests/test_refactor.py`, `tests/test_macro_benchmark.py`.

**Interfaces:** Add `--dataset-profile us-unemployment` for `shared-mean-benchmark`. It requires explicit `--data`, builds `DataConfig(date_col="date", target="unemployment_rate", features=("initial_claims", "nonfarm_payrolls", "cpi_all_items", "industrial_prod"), freq="MS", log_transform=False)`, and uses the 216/309 and 46/309 split ratios. Keep legacy export invocation separate.

- [ ] Add parser tests rejecting `--dataset-profile us-unemployment` without `--data`, with an incompatible pipeline, or with a panel missing any of the six required columns. Add a test proving the macro profile uses `unemployment_rate` and never `Export_US`.
- [ ] Add manifest tests requiring panel SHA-256, source-audit SHA-256, profile name, forecast-origin rule, five stochastic seeds and six model IDs before status can be `success`.
- [ ] Run the parser/manifest tests and confirm failures for the intended missing profile handling.
- [ ] Implement the explicit profile branch using `build_macro_experiment_config` as the starting configuration, with the fixed four source fields. Read and verify the adjacent source-audit JSON before creating a run. Pass the profile config and VAR-NN config to the shared runner. Keep the current `df_final.csv` default in legacy paths but reject it for the unemployment profile.
- [ ] Update `macro_benchmark.CORE_PIPELINES` to advertise its inherited VAR-NN and naïve workflows; remove naïve from `FULL_ONLY_PIPELINES` so names remain unique. Document that exact Kaggle reproduction remains isolated from the fair comparison. Re-run focused tests and commit only these files if permitted.

## Task 6: Verify data and benchmark before drawing conclusions

**Files:** No source edits unless a test exposes a defect; final run artifacts live under `results/shared-mean-benchmark/<run-id>/`.

- [ ] Run all source, panel, parser and shared-mean unit tests. Run the complete suite with `cd vmd_ardl_ffnn && ../.venv/bin/python3 -m pytest -q` and `git diff --check`.
- [ ] With `FRED_API_KEY` set in the environment, run `cd vmd_ardl_ffnn && ../.venv/bin/python3 -m vmd_ardl_ffnn.macro_fred_panel --cache-root ../data/macro_us/raw --output-root ../data/macro_us/processed`. Inspect audit entries for every source, the ICSA weekly-to-monthly rule, 309 monthly rows, zero model-column gaps and exact 216/46/47 split boundaries. Check the known October 2025 absence without filling it.
- [ ] Run a one-seed smoke benchmark on a separate output root. Check that VAR-NN selection/refit succeeds, the naïve reference is present, all six models share forecast dates, and the manifest records the source and panel hashes.
- [ ] Run the registered five-seed benchmark with a new run ID. Read `benchmark_metrics_comparison.csv`, `benchmark_ensemble_metrics_comparison.csv`, `benchmark_forecasts_comparison.csv`, `paper_forecast_date_coverage.csv`, `paper_experiment_settings.csv` and `diebold_mariano_tests.csv`. Confirm no result from `df_final.csv` enters these tables.
- [ ] Report validation and test metrics separately, including naïve. State explicitly whether A7 beats each comparator; do not infer significance from rank alone. Record the COVID-period validation/test distribution shift and pre-register any additional rolling-origin robustness run before examining its test scores.
- [ ] Inspect `git status --short`; preserve all pre-existing user changes and generated results. Commit only implementation and documentation files if permitted.
