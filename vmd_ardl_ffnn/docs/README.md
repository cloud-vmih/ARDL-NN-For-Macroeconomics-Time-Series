# VMD–ARDL–FFNN reproduction

Time-series forecasting package with chronological train/validation/test splits,
train-only lag selection, rolling one-step VMD decomposition caches and FFNN
forecasting.

FFNN production is implemented with TensorFlow/Keras and VMD production uses
`sktime.libs.vmdpy`. The legacy self-contained VMD backend remains an explicit
compatibility option.

The macro workflow for `macro_indicators.csv` defaults to the lightweight
five-model internal comparison and does not run notebook MLP/LSTM/GRU models:

```bash
python ../run_pipeline_advanced.py --pipeline macro-benchmark \
  --data ../macro_indicators.csv --macro-profile core-only
```

Use `--macro-profile full` only when the Kaggle reproduction plus Naive, MLP,
LSTM and GRU are required. Macro production runs require a TensorFlow GPU by
default; `--tensorflow-device cpu` is an explicit smoke/test override. See
`MACRO_BENCHMARK_PLAN.md` for the feature and split policy.

The staged FFNN workflow evaluates every ARDL lag specification with both
learning rates over three train-only expanding folds. It retains the raw
mean-RMSE winner plus the parsimonious one-standard-error winner before the
expensive outer-validation grid. Final reconstructed winners are selected on
level RMSE, matching the primary reported paper metric.

The recommended project entrypoint is:

```bash
python ../run_pipeline_advanced.py --pipeline vmd
```

The paper study is available from the same entrypoint:

```bash
python ../run_pipeline_advanced.py --pipeline benchmark \
  --paper-seeds 7 42 123 2024 2025
```

It runs naïve and standalone-ARDL baselines, fixed/adaptive FFNN benchmarks,
A0–A5 ablations, seed aggregation, Diebold–Mariano tests with Holm correction,
lag-selection summaries, runtime/search-efficiency tables and paper figures.
The hard-config comparator is included as B7 without changing A0–A5. Use
`--skip-varnn` when the VARNN comparator is not required.

Both benchmark profiles report two distinct metric tables.
`benchmark_metrics_comparison.csv` is the primary table and summarizes metrics
computed separately per seed with mean, standard deviation, median and count.
`benchmark_seed_robustness_metrics.csv` is an explicit compatibility alias of
that primary table. `benchmark_ensemble_metrics_comparison.csv` is secondary:
it scores the date-wise mean prediction across seeds and is the estimand used
by DM/Holm tests.

A separate study promotes the AIC block-subset two-stage workflow to the main
Proposed model:

```bash
python ../run_pipeline_advanced.py --pipeline block-proposed-benchmark \
  --paper-seeds 7 42 123 2024 2025
```

This profile preserves the legacy `benchmark` pipeline and writes the distinct
`block_subset_proposed_v1` ablation schema. Its stochastic main benchmarks lock
one model signature using mean validation performance over all reporting seeds:
level RMSE for no-VMD FFNN/VARNN and reconstructed-level RMSE for VMD models.
It then refits the locked winner at every seed; it never selects a best seed or
uses test loss for selection.

For a smaller direct comparison without baselines, VARNN or ablations, run:

```bash
python ../run_pipeline_advanced.py --pipeline shared-mean-benchmark \
  --paper-seeds 7 42 123 2024 2025
```

This profile contains exactly `ardl_ffnn_no_vmd`, the current block-subset
Proposed, the legacy sequential Proposed, and `vmd_ardl_joint_select`. Each
workflow locks one FFNN signature by mean validation RMSE across all reporting
seeds before per-seed refit. Its primary table is the per-seed mean/std summary;
the mean-prediction ensemble is retained only as a secondary table and for DM.
For the current block-subset Proposed in this profile, the lightweight
`alpha=0.1`, learning-rate `0.001` screen runs on train-only expanding folds.
Outer validation is reserved for the subsequent full FFNN grid and reconstructed
winner selection. The separate `block-proposed-benchmark` retains its historical
outer-validation fast-screen protocol.

The new A0–A7 study isolates fixed/no-VMD baselines, ARDL target selection, AIC
variable blocks, component versus reconstructed objectives, shared versus
per-seed selection, and full-grid versus two-stage search. A7 is the Proposed
model; A5 is its shared exhaustive reference and A6 is the per-seed
counterfactual. The main benchmark also includes both the previous sequential
target-then-exogenous workflow and `vmd_ardl_joint_select`; both use shared
reconstructed-level mean-seed FFNN selection. The final FFNN learning-rate grid
is `0.01, 0.001`; A7's
lightweight screen remains fixed at `alpha=0.1` and learning rate `0.001`.

Because this command runs A5, A6 and A7 in one study, smoke-test first with
`--paper-seeds 7 --skip-varnn`. See
`../Experimental_Plan_BlockSubset_Proposed.md` for the exact comparison rules.

The train-only hard-configuration comparator can also run independently:

```bash
python ../run_pipeline_advanced.py --pipeline vmd-ardl-hard-ffnn \
  --paper-seeds 7 42 123 2024 2025
```

It keeps the full Cartesian product of the top three ARDL distributed-lag
orders per exogenous feature. Each component/lag specification receives exactly
one architecture, activation, learning rate and alpha from the preregistered
IMF frequency/entropy/roughness and lag-load policy. It does not run an FFNN
hyperparameter grid. The top three candidates/component are still combined and
selected by shared mean reconstructed-level validation RMSE. The same model is
included as B7 in `benchmark` and B8 in `block-proposed-benchmark`.

The benchmark also includes `vmd_ardl_block_subset_mean_seed`: ARDL locks target
lags, AIC selects one distributed-lag block per exogenous variable, every non-empty
block subset is crossed with the full FFNN grid, and shortlisted component
signatures are combined and selected by mean reconstructed level validation RMSE
across reporting seeds. This comparator is separate from, and does not replace,
the A4 proposed pipeline.

Use `--block-subset-search-strategy two-stage` on a benchmark run to fast-screen
all block specs with `alpha=0.1`, learning rate `0.001`, and mean score across
reporting seeds, retaining four specs/component for the full grid. The default is
`full-grid`; both modes keep reconstructed level mean-seed final selection.

FFNN and VARNN validation compare learning rates `0.01` and `0.001`. VARNN keeps
32 hidden units and sigmoid fixed so this validation changes learning rate only.

Run the predeclared one-factor-at-a-time robustness grid separately with
`--pipeline sensitivity`; use `--paper-seeds 7` for a cheap smoke run before the
full five-seed study.

Available pipelines can be listed with `python ../run_pipeline_advanced.py --help`.
TensorFlow and sktime are installed with the production package. The legacy
extra remains accepted for compatibility:

```bash
pip install -e '.[varnn]'
```

Every invocation writes to a unique `results/<pipeline>/<run_id>/` directory and
records its resolved configuration and status in `manifest.json`. Forecast metrics
use the `rolling_one_step` protocol.

Lag-screen audit files are written as `ffnn_lag_screen_folds_*.csv` and
`ffnn_lag_screen_stability_*.csv`; benchmark runs additionally aggregate them as
`lag_screen_fold_metrics.csv` and `lag_screen_stability.csv`.
