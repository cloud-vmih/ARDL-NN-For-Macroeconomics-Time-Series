# Macro Sequential Cache and Resume Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the sequential macro VMD–ARDL–FFNN profile resumable and memory-bounded while using one tuning seed and four explicitly configured macro features.

**Architecture:** The CLI owns the fixed sequential profile and resume identity. VMD results use a content-addressed disk cache, forecast-origin decompositions are accessed through a bounded LRU, and completed FFNN candidates are persisted in run-scoped checkpoints. Existing non-sequential workflows retain their current behavior.

**Tech Stack:** Python 3.11, pandas, NumPy, TensorFlow 2.15/Keras, sktime VMD, statsmodels ARDL, pytest, JSON and pickle artifacts.

**Spec:** `docs/superpowers/specs/2026-09-03-macro-sequential-cache-resume-design.md`

## Constraints

- Apply the changes only to the sequential macro profile.
- Use exactly `nonfarm_payrolls`, `initial_claims`, `industrial_prod`, and `retail_sales` by default.
- Put the editable feature tuple in one obvious place in `run_pipeline_advanced.py`.
- Tune with seed `7` only; do not change other grids, epochs, callbacks, folds, or scoring rules.
- Do not rerun macro feature screening in the sequential branch.
- Resume only when both `--resume` and an explicit `--run-id` are supplied.
- Reject resume if the dataset/configuration identity differs.
- Preserve causal VMD forecast origins: predictions for time `t` use history through `t-1`.
- Preserve all unrelated user changes and never stage generated data/results/notebook outputs.

## Task 1: Lock the sequential profile and define resume identity

**Files:**

- Modify: `run_pipeline_advanced.py`
- Modify: `vmd_ardl_ffnn/src/vmd_ardl_ffnn/artifacts.py`
- Modify: `vmd_ardl_ffnn/tests/test_refactor.py`
- Create or modify: `vmd_ardl_ffnn/tests/test_artifacts.py`

- [ ] Add failing tests proving that the sequential macro branch does not call feature screening, passes the fixed four-feature tuple, and passes `(7,)` as its seed list.
- [ ] Add failing parser tests proving `--resume` requires an explicit `--run-id`.
- [ ] Add failing artifact tests for exact resume-manifest validation and mismatch rejection.
- [ ] Run the focused tests and confirm they fail for the expected missing behavior.
- [ ] Add `SEQUENTIAL_MACRO_FEATURES`, `SEQUENTIAL_TUNING_SEED`, and a checkpoint schema version near the top of `run_pipeline_advanced.py`.
- [ ] Add `--resume`; reject it without `--run-id` before loading TensorFlow or data.
- [ ] Add manifest identity helpers using the normalized dataset path, dataset fingerprint, config hash, selected-feature order, seed, and schema version.
- [ ] Replace sequential feature screening with validation and selection of the fixed tuple. Leave the benchmark branch unchanged.
- [ ] Run focused tests until green, then commit only task files.

## Task 2: Add a content-addressed VMD disk cache

**Files:**

- Create: `vmd_ardl_ffnn/src/vmd_ardl_ffnn/vmd_cache.py`
- Create: `vmd_ardl_ffnn/tests/test_vmd_cache.py`

- [ ] Write failing tests for stable keys, column-order sensitivity, configuration invalidation, cache hits, partial-entry rejection, and corrupted-payload recomputation.
- [ ] Run the new tests and confirm they fail because the cache module is absent.
- [ ] Implement `VMDCacheRef`, cache statistics, and `ContentAddressedVMDCache`.
- [ ] Hash the exact index, ordered columns, dtypes, values, VMD configuration, and cache schema.
- [ ] Persist each component separately plus metadata; write a completion marker only after all payloads are durable.
- [ ] Use temporary sibling files and atomic `Path.replace` operations.
- [ ] Treat incomplete or unreadable entries as misses and safely overwrite only that exact cache key.
- [ ] Run cache tests until green, then commit only task files.

## Task 3: Bound forecast-origin memory with disk-backed access

**Files:**

- Modify: `vmd_ardl_ffnn/src/vmd_ardl_ffnn/experiment_fixed.py`
- Modify: `vmd_ardl_ffnn/tests/test_experiment_fixed.py`
- Modify: `vmd_ardl_ffnn/tests/test_vmd_cache.py`

- [ ] Add failing tests showing an origin cache stores references rather than every decomposed DataFrame, evicts old materializations, and preserves `t-1` causality.
- [ ] Add an equivalence test comparing legacy in-memory predictions with disk-backed predictions on a tiny deterministic decomposition.
- [ ] Run focused tests and confirm the new expectations fail.
- [ ] Implement a bounded `DiskBackedOriginCache` LRU over VMD cache references.
- [ ] Allow `VMDARDLFFNNExperiment` to receive an optional VMD cache while keeping the legacy path as the default.
- [ ] Make `decompose`, origin-cache creation, actual-component creation, and cached prediction use content-addressed entries when configured.
- [ ] Materialize only the requested component/fold, evict old entries, and release fold-local objects after use.
- [ ] Run focused tests until green, then commit only task files.

## Task 4: Add atomic FFNN candidate checkpoints

**Files:**

- Create: `vmd_ardl_ffnn/src/vmd_ardl_ffnn/checkpoints.py`
- Create: `vmd_ardl_ffnn/tests/test_checkpoints.py`
- Modify: `vmd_ardl_ffnn/src/vmd_ardl_ffnn/experiment_fixed.py`
- Modify: `vmd_ardl_ffnn/tests/test_experiment_fixed.py`

- [ ] Write failing tests for stable candidate signatures, successful round trips, incomplete/corrupt checkpoint misses, and identity isolation.
- [ ] Add a failing experiment test proving a completed candidate is loaded without calling model fitting and failures are not marked complete.
- [ ] Run focused tests and confirm failure for missing checkpoint behavior.
- [ ] Implement `CandidateCheckpointStore` with run identity, candidate identity, fold/stage context, atomic payload files, and a final completion marker.
- [ ] Include every hyperparameter and data/fold/config identity that can affect a candidate result in its signature.
- [ ] Inject the optional store into `VMDARDLFFNNExperiment`.
- [ ] Load checkpoints at the start of each FFNN candidate evaluation; persist row and prediction artifacts only after success.
- [ ] Apply the same mechanism to staged-halving screening folds without altering ranking or candidate IDs.
- [ ] Run focused tests until green, then commit only task files.

## Task 5: Wire cache and resume into the sequential runner

**Files:**

- Modify: `run_pipeline_advanced.py`
- Modify: `vmd_ardl_ffnn/src/vmd_ardl_ffnn/artifacts.py`
- Modify: `vmd_ardl_ffnn/tests/test_refactor.py`
- Modify: `README.md`
- Modify: `MACRO_BENCHMARK_PLAN.md`

- [ ] Add failing CLI/integration tests for new-run directory creation, explicit resume reuse, manifest mismatch rejection before TensorFlow bootstrap, and cache/checkpoint wiring.
- [ ] Run focused tests and confirm they fail for the expected missing wiring.
- [ ] For a new sequential run, create the run directory and write the strict identity manifest before training.
- [ ] For resume, require the existing directory and validate the manifest before TensorFlow bootstrap or training.
- [ ] Use `<output-root>/_cache/vmd` for reusable VMD artifacts and `<run-dir>/checkpoints/ffnn` for run-specific candidates.
- [ ] Record hit/miss/resumed-candidate counters in run metadata without changing model selection semantics.
- [ ] Document the single edit point for the four features and show new-run/resume CLI examples.
- [ ] Run focused tests until green, then commit only task files.

## Task 6: Verify resource safety and preserve the worktree

- [ ] Run all targeted cache, checkpoint, artifact, CLI, and experiment tests.
- [ ] Run the complete project test suite.
- [ ] Run `git diff --check`.
- [ ] Run a tiny synthetic sequential job twice: first to populate artifacts, second with the same run ID and `--resume` to verify cache/checkpoint reuse.
- [ ] Inspect effective configuration output to confirm four fixed features and seed `7`.
- [ ] Inspect `git status --short` and ensure `macro_indicators.csv`, generated results, and the modified notebook remain unstaged.
- [ ] Commit any final documentation/test adjustment separately; do not start a full production training run.

