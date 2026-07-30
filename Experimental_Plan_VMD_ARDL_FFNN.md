# Experimental Plan for the VMD–ARDL–FFNN Paper

## 1. Study Objective

The experiments evaluate whether a sequential, component-specific lag-selection
framework improves rolling one-step-ahead forecasts. The proposed model is defined
as **VMD–ARDL(Target SelectOrder + Exogenous Screening)–FFNN**.

The study addresses four research questions:

1. **RQ1 — VMD contribution:** Does VMD decomposition improve forecast accuracy
   relative to an otherwise comparable no-VMD model?
2. **RQ2 — Target-lag contribution:** Does component-specific ARDL target-lag
   selection improve accuracy relative to fixed target lags?
3. **RQ3 — Exogenous-lag contribution:** After target lags are locked, does
   ARDL-guided exogenous-lag screening improve accuracy and search efficiency?
4. **RQ4 — Heterogeneous dynamics:** Do different VMD components select materially
   different target and exogenous lag structures?

Claims about superiority or component heterogeneity must be stated as hypotheses
until supported by the reported results.

## 2. Proposed Pipeline Definition

All selection and model fitting below are performed separately for each target VMD
component (`VMD1`, ..., `VMDK`, and `RES`). VMD decomposition of evaluation data uses
an expanding-origin cache so that the decomposition at time `t` only sees history
available before `t`.

### 2.1 Sequential ARDL-guided lag selection

For component `k`, the proposed lag-selection procedure is:

1. **Select target lags on train only.** Run `statsmodels.ardl_select_order` with:
   - component `k` of the target as the endogenous series;
   - component `k` of all exogenous variables as candidate regressors;
   - `max_target_lag` and `max_exog_lag` as the search bounds;
   - causal exogenous terms only;
   - the predeclared information criterion, primarily AIC and secondarily BIC.
2. Retain only the target autoregressive lags returned by `ardl_select_order`.
   These target lags are now locked. Do not add, remove, or filter them with ACF/PACF.
3. **Screen each exogenous variable conditionally on the locked target lags.** For
   each exogenous variable `x_j` and candidate order `q = 1, ..., Q`, fit:

   ```text
   y_k(t) ~ selected target lags + x_j(t-1), ..., x_j(t-q)
   ```

4. Rank candidate exogenous orders using the same AIC/BIC criterion and a common
   `hold_back`, ensuring that all information criteria are computed on the same
   effective sample.
5. Retain the top `N` exogenous orders for each variable and construct candidate
   multivariate lag specifications using the predeclared staged-combination budget.
6. Evaluate every retained lag specification on three expanding folds contained
   entirely within train. Each fold uses a 12-month validation block and the fast
   FFNN compares learning rates `{0.01, 0.001}`.
7. Shortlist lag specifications using mean fold RMSE. Always retain the raw
   mean-RMSE winner and the simplest candidate within one standard error of that
   winner; fill the remaining predeclared top-`K` budget by mean-RMSE rank.
8. Search FFNN architecture, activation, regularization, and learning rate on the
   outer validation split only after the lag shortlist has been locked.
9. Select the combination of component models using reconstructed **level-scale**
   validation RMSE, with level MAE, level MAPE, and input dimension as tie-breakers.
10. Refit the locked winners on train plus validation and evaluate once on test.

This is a **sequential selection procedure**, not joint ARDL order selection. The
`ardl_select_order_joint` pipeline is retained only as an alternative benchmark or
robustness model.

### 2.2 FFNN inputs

For every candidate lag specification, FFNN input at time `t` contains only:

```text
selected target component lags at t-p
selected exogenous component lags at t-q, q >= 1
```

Lag 0 is prohibited. The FFNN standardizes X and y using training data only.

## 3. Data and Forecasting Protocol

- Use the same variables, observation period, transformations, and sample for every
  model comparison.
- Split chronologically into 70% train, 15% validation, and 15% test.
- Determine log and stationarity transformations without using validation or test.
- Select all lags on train only.
- Use train-only expanding folds for lag-spec fast screening.
- Use the outer validation split for architecture and hyperparameter selection.
- Use test exactly once after all model and lag decisions are locked.
- The primary protocol is **rolling one-step-ahead forecasting**:
  after forecasting time `t`, the actual observation at `t` becomes available before
  forecasting `t+1`.
- Clearly distinguish this protocol from recursive multi-step forecasting; do not
  claim multi-step performance from the rolling one-step results.
- Refit every compared trainable model on train plus validation before test whenever
  its protocol permits it.

Primary metrics must be reported on the original level scale:

- RMSE
- MAE
- MAPE
- Directional Accuracy, computed relative to the previous actual observation

Transformed-scale metrics may be included as diagnostics but must not be the primary
cross-model evidence, because MAPE on differenced or near-zero component series can
be unstable or uninterpretable.

## 4. Benchmark Models

All benchmarks must use the same split, rolling-origin information set, level-scale
evaluation dates, and target definition.

| ID | Model | Purpose |
|---|---|---|
| B0 | Seasonal naïve / last-observation naïve | Minimal forecasting reference |
| B1 | Standalone ARDL | Classical linear dynamic-regression reference |
| B2 | FFNN with fixed target/exogenous lags | Neural baseline without ARDL screening or VMD |
| B3 | ARDL-selected FFNN without VMD | Isolate the contribution of VMD |
| B4 | VMD + FFNN with fixed lags | Isolate VMD without adaptive lag selection |
| B5 | VARNN | Multivariate neural time-series benchmark |
| B6 | VMD + AIC variable-block subsets + FFNN | Exhaustive block-subset comparator with mean reconstructed-level validation selection across seeds |
| B7 | VMD + ARDL top-3 full-product + hard-config FFNN | Lag-combination comparator without an FFNN hyperparameter grid |
| P | Proposed sequential VMD–ARDL–FFNN | Full proposed framework |

The implementation now includes all B0–B7 baselines/comparators and the proposed sequential
model. VARNN remains an optional TensorFlow dependency; if it cannot run, the study
records this explicitly in `skipped_models.csv` and must not describe the resulting
table as including B5.

### 4.1 B6 variable-block subset protocol

B6 is a benchmark comparator, not a replacement for the proposed A4 model and not
an additional ablation. For each VMD component it:

1. selects target lags on train using ARDL SelectOrder and locks them;
2. selects one AIC-minimizing distributed-lag order `q_j` independently for each
   exogenous variable conditional on the locked target lags;
3. defines variable block `j` as lags `1..q_j` and enumerates every non-empty
   subset of the blocks (`2^m - 1` candidates for `m` exogenous variables);
4. evaluates the complete FFNN architecture/alpha/activation/learning-rate grid
   for every subset and every predeclared reporting seed;
5. ranks shared signatures by mean component validation score across seeds and
   retains `top_component_candidates` per component only for the tractable final
   Cartesian combination search;
6. reconstructs each cross-component combination for every seed and selects one
   shared combination using mean reconstructed level validation RMSE across all
   seeds, followed by level MAE, MAPE, and total input dimension;
7. refits that locked combination separately for every seed and reports
   reconstructed level-scale validation/test metrics.

No rolling top-k lag shortlist is applied in the default B6 `full_grid` mode; the
exhaustive subset grid is the point of that comparator. With four exogenous
variables, this creates 15 lag specs per VMD component before crossing the FFNN
grid.

An explicitly labeled computational variant, `two_stage`, is also implemented.
It fast-screens all 15 specs/component using a lightweight one-hidden-layer FFNN,
`alpha=0.1`, learning rate `0.001`, `fast_max_iter=150`, and mean component
validation metrics across all reporting seeds. The best four lag specs/component
then enter the complete declared FFNN grid and the same mean reconstructed-level
combination selection described above. The default `full_grid` variant remains
available, and papers must report which strategy generated the presented results.

### 4.2 B7 hard-config full-product protocol

B7 is a benchmark comparator, not an ablation and not a replacement for Proposed
A4. For each train-only VMD component it:

1. selects and locks target lags with ARDL SelectOrder;
2. ranks distributed-lag orders independently for every exogenous variable by AIC
   conditional on those target lags, retaining the top three orders;
3. expands order `q` into block `1..q` and evaluates the full Cartesian product
   (up to `3^4 = 81` lag specs/component for four exogenous variables);
4. deterministically assigns one architecture, activation, learning rate and alpha
   from train-only IMF frequency, spectral entropy, roughness and lag load;
5. fits that fixed configuration for every reporting seed, with no FFNN
   hyperparameter grid;
6. retains the top three shared signatures/component and chooses the final
   cross-component combination by mean reconstructed level validation RMSE across
   the reporting seeds;
7. locks the winner, refits on train plus validation and reports test metrics for
   every seed.

B7 forecasts enter the same benchmark metrics, date-coverage audit and pairwise
DM/Holm tests as B0–B6 and Proposed.

## 5. Ablation Study

The ablation must change one module at a time while keeping data, VMD settings,
FFNN search budget, seeds, and forecasting protocol fixed.

| ID | VMD | Target lags | Exogenous lags | Purpose |
|---|---|---|---|---|
| A0 | No | Fixed | Fixed | Basic FFNN reference |
| A1 | Yes | Fixed | Fixed | VMD contribution |
| A2 | Yes | ARDL SelectOrder | Fixed | Target-lag selection contribution |
| A3 | Yes | Fixed | ARDL screening | Exogenous-screening contribution |
| A4 | Yes | ARDL SelectOrder | ARDL screening | Full proposed sequential model |
| A5 | Yes | Joint ARDL SelectOrder | Joint ARDL SelectOrder | Alternative joint-selection comparator |

The primary comparisons are:

- `A1 − A0`: contribution of VMD;
- `A2 − A1`: contribution of component-specific target-lag selection;
- `A4 − A2`: incremental contribution of exogenous-lag screening;
- `A5 − A4`: joint versus sequential lag selection.

Do not label `validation_screen` as the proposed target-lag method in the main paper.
It may be retained as an additional ablation comparing validation-driven and
information-criterion-driven target-lag selection.

## 6. Random Seeds and Model Selection

Random seed must not be treated as a hyperparameter from which the single best seed
is selected. Use a predeclared set of 5–10 seeds.

Recommended procedure:

1. Select structural hyperparameters using mean validation RMSE across the declared
   seeds, with MAE as the first tie-breaker.
2. Lock the selected lag structure and hyperparameters.
3. Refit and evaluate the locked configuration for every seed.
4. Report test mean, standard deviation, median, and the individual seed results
   as the primary expected single-model performance analysis.
5. As a secondary deployment analysis, average predictions across seeds at each
   forecast date and score that ensemble prediction once. Use this ensemble
   prediction in DM/Holm tests and label its estimand explicitly.

If computational constraints require a single seed during structural screening,
declare that seed in advance, then rerun the locked winner using all reporting seeds.

The implemented study never selects the best reporting seed. It runs every declared
seed independently, exports every result, and reports mean, standard deviation,
median, and count. This includes both parameter-initialization variation and any
validation-selected-structure variation across seeds; that distinction must be
stated when interpreting the standard deviation.

B6 is the exception to independent structural selection: it reconstructs each
shortlisted component combination for every reporting seed, averages level
validation metrics across seeds, locks one common cross-component structure, then
refits/evaluates every seed. Its exported seed variation therefore isolates
initialization variation under a shared selected structure more directly.

For VARNN, validation changes only the learning rate `{0.01, 0.001}` in the main
runner; hidden units remain 32 and activation remains sigmoid. For FFNN variants,
the same two learning rates are crossed with the declared architecture and alpha
grid. The validation winner is locked and reused during train-plus-validation refit.

## 7. Statistical Validation

- Apply the Diebold-Mariano test to paired one-step forecast loss differentials on
  the common test dates.
- Use mean/std/median/count of separately scored seed runs as the primary accuracy
  estimand, representing expected single-model performance across initializations.
- Report ensemble-mean prediction metrics as a secondary deployment estimand.
  Use that single forecast series for DM/Holm and do not present its DM result as
  a test of the per-seed metric distribution.
- Report the DM statistic, p-value, loss function, and compared models.
- Correct for multiple pairwise comparisons using Holm's procedure.
- Report paired error differences and bootstrap confidence intervals where feasible.
- Treat random-seed variation and time-series forecast uncertainty as distinct:
  seed mean ± standard deviation does not replace a paired forecast-accuracy test.

## 8. Robustness and Sensitivity Analysis

Predeclare a compact sensitivity grid:

- VMD mode count `K`;
- VMD bandwidth parameter `alpha`;
- maximum target and exogenous lag bounds;
- AIC versus BIC;
- staged versus full-product exogenous combination search for a feasible subset;
- expanding versus fixed-length rolling training windows, if computationally feasible;
- alternative train/validation/test cut points.

For each sensitivity run, preserve the test protocol and report both accuracy and
runtime. Avoid selecting the best sensitivity setting using test performance.

The implemented `sensitivity` pipeline uses a one-factor-at-a-time grid around S0:
`K={2,4}`, `alpha={0.5×,2×}`, compact lag bounds, BIC, a feasible full-product
subset, and 65/20/15 plus 75/10/15 splits. It is run separately from the main
benchmark so its cost is explicit:

```bash
python run_pipeline_advanced.py --pipeline sensitivity \
  --paper-seeds 7 42 123 2024 2025
```

Train-only expanding folds are implemented for lag-spec screening. A separate
expanding-versus-fixed-length **final forecasting-window robustness comparison** is
not implemented and must not be claimed as completed evidence.

## 9. Lag-Selection Analysis

For every component and seed/configuration, export:

- target lags selected by `ardl_select_order`;
- ranked exogenous orders and information-criterion scores;
- final FFNN lag specification;
- selection frequency across seeds or rolling windows;
- number of retained features and total FFNN input dimension.
- fold-level lag-screen RMSE/MAE/MAPE for both learning rates;
- mean, standard deviation, standard error, raw winner, one-SE winner, and shortlist flag.

Recommended tables and figures:

- component-by-lag target heatmap;
- component-by-variable exogenous-order heatmap;
- lag-selection frequency plot;
- VMD center-frequency or component-order summary;
- relationship between component frequency and selected maximum lag.

Any claim that high- and low-frequency components exhibit different dependencies
must be supported by these results rather than assumed in advance.

## 10. Efficiency Analysis

Report measured, not hypothetical, values for:

- theoretical full-product lag combinations;
- candidate count after ARDL screening;
- number of FFNN fits in screening and final search stages;
- ARDL selection time, VMD cache time, FFNN search time, and total runtime;
- peak memory if available.

Search-space reduction is calculated as:

```text
100 × (full candidate count − evaluated candidate count) / full candidate count
```

Accuracy and efficiency must be reported together so that search reduction is not
presented as beneficial if it materially degrades forecasts.

The implementation records total wall-clock runtime plus ARDL selection, VMD cache,
FFNN search, and refit/forecast phases in `efficiency_summary.csv`. The full-product
count and retained lag-spec count refer to lag structures; `evaluated_ffnn_fits`
reports actual fitted FFNN candidates and should not be confused with lag-spec count.

## 11. Required Tables and Figures

### Main tables

1. Dataset variables, frequency, transformations, and split dates.
2. Complete benchmark results on validation and test.
3. Ablation results.
4. Mean ± standard deviation across seeds.
5. DM tests with multiplicity-adjusted p-values.
6. Selected target and exogenous lag structures by component.
7. Runtime and candidate-count comparison.

### Main figures

1. VMD decomposition of the training target.
2. Test actual versus rolling one-step forecasts.
3. Model error distribution.
4. Target- and exogenous-lag heatmaps.
5. RMSE/MAE comparison with uncertainty bars.
6. Accuracy-versus-runtime plot.

## 12. Implementation Alignment Checklist

Before producing the final paper tables, verify that:

- [x] The benchmark labels the sequential `ardl_select_order` target-lag pipeline as
      the proposed model.
- [x] `validation_screen` is moved to an ablation/alternative model rather than the
      proposed benchmark profile.
- [x] The four missing classical/fixed-lag baselines are implemented.
- [x] Seeds are aggregated rather than selecting the best individual seed.
- [x] Every candidate ARDL comparison uses a common `hold_back`.
- [x] Every model reports the same rolling one-step test dates on the level scale.
- [x] Test data are not used for lag, VMD parameter, architecture, seed, or model
      selection.
- [x] Benchmark artifacts are stored in separate model directories with a run
      manifest and full resolved configuration.

The checked items describe implemented safeguards. They do not mean the empirical
study has already been run to completion on the final dataset; verify the generated
manifest has `status: success` and inspect `skipped_models.csv` before using results.

## 13. Reproduction Command and Artifacts

Run the complete benchmark and ablation study from the project root:

```bash
python run_pipeline_advanced.py \
  --pipeline benchmark \
  --paper-seeds 7 42 123 2024 2025
```

Use `--skip-varnn` only when intentionally omitting B5. The main machine-readable
outputs are `paper_seed_metrics.csv`, `paper_metrics_summary.csv`,
`paper_ensemble_metrics_summary.csv`, `benchmark_metrics_comparison.csv`,
`benchmark_ensemble_metrics_comparison.csv`,
`benchmark_seed_robustness_metrics.csv`, `ablation_metrics.csv`,
`diebold_mariano_tests.csv`, `lag_selection_frequency.csv`, and
`efficiency_summary.csv`. `paper_forecast_date_coverage.csv` must show identical
validation/test date sets before any pairwise accuracy claim is made.

B6 additionally writes `ardl_variable_blocks_train_only.csv`,
`ffnn_block_subset_component_search.csv`,
`ffnn_block_subset_fast_screen_metrics.csv` and
`ffnn_block_subset_fast_screen_summary.csv` when `two_stage` is enabled,
`ffnn_block_subset_component_mean_seed_ranking.csv`,
`ffnn_block_subset_reconstructed_seed_scores.csv`,
`ffnn_block_subset_mean_seed_selection.csv`, and the locked per-seed component
and reconstructed forecast/metric tables under
`vmd_ardl_block_subset_mean_seed/`.

B7 writes its train-only descriptors, policy assignments, component shortlist,
reconstructed shared selection and locked forecasts under
`vmd_ardl_hard_config_ffnn/`.

## 14. Paper Contributions to Test

1. A sequential component-specific lag-selection procedure that first identifies
   target dynamics using ARDL SelectOrder and then screens exogenous lag orders
   conditionally on those locked target lags.
2. Empirical evidence on whether component-specific target and exogenous lag
   structures improve forecast accuracy beyond VMD alone.
3. Evidence on the accuracy–efficiency trade-off achieved by ARDL-guided screening.
4. A leakage-controlled rolling one-step VMD–ARDL–FFNN forecasting framework.
