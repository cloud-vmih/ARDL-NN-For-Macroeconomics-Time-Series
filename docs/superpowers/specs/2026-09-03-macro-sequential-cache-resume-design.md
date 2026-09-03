# Macro Sequential Cache and Resume Design

## Objective

Make `sequential-vmd-ardl-ffnn` practical on a machine with 16 GB RAM and a
4 GB GTX 1650 without changing its forecasting protocol. The implementation
will:

- tune FFNN candidates with one fixed seed;
- stop rerunning macro feature screening and expose one obvious feature tuple;
- persist VMD decompositions in a content-addressed disk cache;
- avoid retaining all VMD fold/origin data in memory simultaneously;
- checkpoint each completed FFNN candidate and resume an interrupted run safely.

The changes apply to the sequential macro profile only unless a cache primitive
is shared internally without changing other profiles' behavior.

## Fixed Sequential Macro Configuration

`run_pipeline_advanced.py` will expose these adjacent constants near the other
pipeline-level constants:

```python
SEQUENTIAL_MACRO_FEATURES = (
    "nonfarm_payrolls",
    "initial_claims",
    "industrial_prod",
    "retail_sales",
)
SEQUENTIAL_TUNING_SEED = 7
```

This is the single user-editable location for the sequential macro feature set.
The sequential branch will validate that every named feature exists, pass the
tuple directly to `build_macro_experiment_config`, and record it in the
manifest. It will not call `screen_macro_features`.

Only `seed_grid` changes to `(SEQUENTIAL_TUNING_SEED,)` for this profile. The
existing staged-halving strategy, ARDL settings, VMD settings, architectures,
alpha grid, activation grid, learning-rate grid, epochs, and callbacks remain
unchanged.

## Content-Addressed VMD Disk Cache

A focused cache component will sit below `VMDARDLFFNNExperiment.decompose`.
Every decomposition will have a cache key derived from:

- a cache schema version;
- the ordered timestamp index;
- ordered column names and numeric values after preprocessing;
- the complete `VMDConfig` that affects numerical output;
- the VMD backend/package compatibility marker.

The stable cache root will be:

```text
results/_cache/vmd/<cache-key>/
```

Each entry contains metadata plus serialized component arrays/index/columns.
Files are first written with a temporary suffix and atomically renamed only
after serialization completes. A cache entry is accepted only when its schema,
key, shapes, index, columns, and component names validate. Corrupt or partial
entries are ignored and recomputed.

The cache is reusable across run IDs when data and numerical configuration are
identical. It contains no model-selection result and therefore cannot leak
validation/test decisions between runs.

## Memory-Bounded VMD Access

The current implementation builds Python objects for all rolling folds and all
forecast origins before candidate evaluation. It will be replaced with disk
descriptors and component-scoped materialization:

1. Generate or locate each fold/origin decomposition in the disk cache.
2. Keep lightweight cache keys/paths in the fold context.
3. When evaluating one VMD component, materialize only that component's train
   and origin frames.
4. Release component data before proceeding to the next component or fold.
5. Explicitly drop references and run garbage collection at these boundaries.

Forecast-origin decomposition remains strictly causal: the prediction cache at
date `t` is built from history through `t-1`; the actual-component cache used
for scoring may include the actual at `t` but no later timestamp. Disk caching
must not merge these two distinct keys.

This design reduces retained Python/Pandas objects. It does not move VMD or
ARDL to GPU and does not change numerical parameters.

## Candidate Checkpointing

The CLI will add `--resume`. It is valid only with an explicit `--run-id`.

```bash
./.venv/bin/python3 run_pipeline_advanced.py \
  --pipeline sequential-vmd-ardl-ffnn \
  --data macro_indicators.csv \
  --run-id macro-sequential-01 \
  --resume
```

Without `--resume`, an existing run directory remains an error. With
`--resume`, the runner opens that directory and validates the existing
manifest before any computation. Resume is allowed only when all of these
match:

- pipeline name;
- resolved dataset path and dataset SHA-256;
- configuration hash excluding output directory;
- fixed selected-feature tuple;
- checkpoint schema version.

Each completed FFNN candidate is identified by a stable signature containing
the component, lag spec, architecture, alpha, seed, activation, learning rate,
epoch budget, scaler, and selection stage. Its checkpoint stores:

- candidate metadata and validation metrics;
- validation prediction rows required for later reconstruction;
- completion marker written last.

Writes are atomic. On resume, complete matching candidates are loaded and
skipped; incomplete candidates are retrained. The selected winner and final
test artifacts continue to be produced by the existing pipeline logic, so
resuming cannot open test data early.

The manifest will record `resume_requested`, `resumed_from_checkpoint`, counts
of loaded/computed VMD entries, and counts of loaded/trained FFNN candidates.

## Failure and Interruption Behavior

Ordinary Python exceptions continue to mark the manifest `failed`. A SIGKILL or
power loss can leave it `running`; `--resume` treats that state as interrupted
only after all identity checks pass. No checkpoint file is trusted without its
completion marker.

The implementation will not automatically select the latest run. Requiring an
explicit run ID prevents accidental mixing of experiments. Changing a feature,
seed, data value, VMD parameter, or FFNN candidate definition invalidates the
affected cache/checkpoint through its hash.

## User Flow

Start a resumable run:

```bash
./.venv/bin/python3 run_pipeline_advanced.py \
  --pipeline sequential-vmd-ardl-ffnn \
  --data macro_indicators.csv \
  --tensorflow-device gpu \
  --run-id macro-sequential-01
```

Resume after interruption:

```bash
./.venv/bin/python3 run_pipeline_advanced.py \
  --pipeline sequential-vmd-ardl-ffnn \
  --data macro_indicators.csv \
  --tensorflow-device gpu \
  --run-id macro-sequential-01 \
  --resume
```

To change model inputs, edit only `SEQUENTIAL_MACRO_FEATURES` and start a new
run ID. Existing incompatible checkpoints will not be reused.

## Testing

Tests will verify:

- the sequential macro branch uses the fixed tuple and never invokes feature
  screening;
- its FFNN seed grid contains only seed 7;
- identical frames/configurations hit the same VMD cache entry;
- a changed data value, column order, index, or VMD parameter misses the cache;
- partial/corrupt cache writes are rejected;
- disk-backed origin access preserves forecast-origin causality;
- fold/component data is released rather than accumulated in one context list;
- a completed candidate is skipped on resume and yields the same row and
  prediction as a fresh run;
- incompatible dataset/config/feature hashes reject resume;
- `--resume` without `--run-id` fails before creating or modifying artifacts;
- the existing non-resume behavior and other pipeline profiles remain intact.

No full macro training run is required in the automated suite. Small synthetic
frames and lightweight test doubles will exercise cache identity, atomic files,
candidate checkpoint restoration, and CLI validation; existing integration
tests continue to cover real VMD/TensorFlow boundaries.

## Non-Goals

- Moving ARDL or VMD to GPU.
- Changing the statistical split or forecast protocol.
- Changing VMD modes, tolerance, or iteration limit.
- Reducing FFNN grids other than using one tuning seed.
- Automatically applying the six-feature research-notebook recommendation.
- Resuming an old run created before this checkpoint schema exists.
