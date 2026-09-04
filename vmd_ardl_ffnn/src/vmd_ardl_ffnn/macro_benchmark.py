from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.ardl import ARDL

from .artifacts import write_csv
from .block_proposed_experiments import _normalized_result
from .config import (
    ARDLSelectionConfig,
    DataConfig,
    ExperimentConfig,
    FFNNConfig,
    StationarityConfig,
    TensorFlowRuntimeConfig,
    VMDConfig,
)
from .experiment import VMDARDLFFNNExperiment
from .metrics import evaluate_forecast
from .models.tensorflow_runtime import bootstrap_tensorflow
from .paper_experiments import _aggregate_seed_metrics
from .shared_mean_benchmark_experiments import run_shared_mean_benchmark_experiments


TARGET = "unemployment_rate"
DATE_COLUMN = "date"
MACRO_FEATURE_CANDIDATES = (
    "initial_claims",
    "nonfarm_payrolls",
    "consumer_sentiment",
    "industrial_prod",
    "retail_sales",
    "building_permits",
)
MACRO_MODEL_FEATURES = (*MACRO_FEATURE_CANDIDATES, "cpi_all_items")
CORE_PIPELINES = (
    "ardl_ffnn_no_vmd",
    "vmd_ardl_block_subset_proposed",
    "vmd_ardl_sequential_screen",
    "vmd_ardl_joint_select",
    "vmd_ardl_univariate_shared_mean",
)
FULL_ONLY_PIPELINES = (
    "notebook_exact_reproduction",
    "naive_last_observation",
    "notebook_mlp_fair",
    "notebook_lstm_fair",
    "notebook_gru_fair",
)
NOTEBOOK_SHA256 = "a708ae71684090890c8353b2b90a2e0a4f9fe96e48abc4a36325477b17a0728d"
NOTEBOOK_URL = "https://www.kaggle.com/code/awing11/forecasting-a-convenient-macro-indicator"


@dataclass(frozen=True)
class MacroDataset:
    raw: pd.DataFrame
    model_data: pd.DataFrame
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    audit: pd.DataFrame


def pipelines_for_profile(profile: str) -> tuple[str, ...]:
    normalized = profile.lower().replace("_", "-")
    if normalized == "core-only":
        return CORE_PIPELINES
    if normalized == "full":
        return CORE_PIPELINES + FULL_ONLY_PIPELINES
    raise ValueError("macro profile must be 'core-only' or 'full'.")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_macro_dataset(path: str | Path) -> MacroDataset:
    """Load the complete monthly target prefix and apply the fixed fair split."""
    source = Path(path)
    raw = pd.read_csv(source)
    required = [DATE_COLUMN, TARGET, *MACRO_MODEL_FEATURES]
    missing = [column for column in required if column not in raw]
    if missing:
        raise ValueError(f"Macro CSV is missing required columns: {missing}")
    raw[DATE_COLUMN] = pd.to_datetime(raw[DATE_COLUMN], errors="raise")
    if raw[DATE_COLUMN].duplicated().any():
        raise ValueError("Macro CSV contains duplicate dates.")
    raw = raw.sort_values(DATE_COLUMN).set_index(DATE_COLUMN)
    expected = pd.date_range(raw.index.min(), raw.index.max(), freq="MS")
    if not raw.index.equals(expected):
        raise ValueError("Macro CSV must have a contiguous monthly calendar.")
    raw = raw.asfreq("MS")

    target_missing = np.flatnonzero(raw[TARGET].isna().to_numpy())
    prefix_end = int(target_missing[0]) if len(target_missing) else len(raw)
    if prefix_end == 0:
        raise ValueError("Macro target is missing at the first timestamp.")
    model_data = raw.iloc[:prefix_end][required[1:]].apply(
        pd.to_numeric, errors="coerce"
    )
    missing_counts = model_data.isna().sum()
    if missing_counts.any():
        details = {key: int(value) for key, value in missing_counts.items() if value}
        raise ValueError(f"Macro model prefix contains missing values: {details}")

    train = model_data.loc[:"2017-12-01"].copy()
    validation = model_data.loc["2018-01-01":"2021-10-01"].copy()
    test = model_data.loc["2021-11-01":"2025-09-01"].copy()
    combined = pd.concat([train, validation, test])
    if not combined.index.equals(model_data.index):
        raise ValueError(
            "Macro data coverage does not match the fixed 2000-01 through 2025-09 protocol."
        )
    audit = pd.DataFrame(
        [
            {
                "split": name,
                "rows": len(frame),
                "start": frame.index.min(),
                "end": frame.index.max(),
            }
            for name, frame in (
                ("train", train),
                ("validation", validation),
                ("test", test),
            )
        ]
    )
    return MacroDataset(raw, model_data, train, validation, test, audit)


def _fit_candidate_aic(
    data: pd.DataFrame,
    target: str,
    feature: str,
    target_lag: int,
    exog_lag: int,
) -> float:
    model = ARDL(
        data[target],
        lags=list(range(1, int(target_lag) + 1)),
        exog=data[[feature]],
        order={feature: list(range(1, int(exog_lag) + 1))},
        causal=True,
        trend="c",
        hold_back=12,
        missing="raise",
    ).fit()
    return float(model.aic)


def screen_macro_features(
    train: pd.DataFrame,
    *,
    candidates: tuple[str, ...] = MACRO_FEATURE_CANDIDATES,
    top_n: int = 4,
) -> tuple[tuple[str, ...], pd.DataFrame]:
    """Rank fixed exogenous candidates by their best train-only ARDL AIC."""
    if top_n < 1 or top_n > len(candidates):
        raise ValueError("top_n must select between one and all macro candidates.")
    rows: list[dict[str, Any]] = []
    for feature in candidates:
        if feature not in train:
            raise ValueError(f"Macro feature candidate is missing: {feature}")
        for target_lag in range(1, 13):
            for exog_lag in range(1, 7):
                try:
                    aic = _fit_candidate_aic(
                        train, TARGET, feature, target_lag, exog_lag
                    )
                    error = None
                except Exception as exc:
                    aic = float("nan")
                    error = f"{type(exc).__name__}: {exc}"
                rows.append(
                    {
                        "feature": feature,
                        "target_max_lag": target_lag,
                        "exog_max_lag": exog_lag,
                        "AIC": aic,
                        "error": error,
                        "selection_scope": "train_only",
                    }
                )
    audit = pd.DataFrame(rows)
    best = (
        audit.dropna(subset=["AIC"])
        .sort_values(["AIC", "feature", "target_max_lag", "exog_max_lag"])
        .groupby("feature", as_index=False)
        .first()
        .sort_values(["AIC", "feature"])
        .reset_index(drop=True)
    )
    if len(best) < top_n:
        raise ValueError("Too few macro features produced a valid ARDL AIC score.")
    selected = tuple(best.head(top_n)["feature"].astype(str))
    rank = {feature: index + 1 for index, feature in enumerate(best["feature"])}
    audit["feature_rank"] = audit["feature"].map(rank)
    audit["used_for_model"] = audit["feature"].isin(selected)
    return selected, audit


def build_macro_experiment_config(
    features: tuple[str, ...],
    output_dir: str | Path,
    *,
    tensorflow: TensorFlowRuntimeConfig | None = None,
    scaler: str = "standard",
    seeds: tuple[int, ...] = (7, 42, 123, 2024, 2025),
) -> ExperimentConfig:
    return ExperimentConfig(
        data=DataConfig(
            date_col=DATE_COLUMN,
            target=TARGET,
            features=features,
            freq="MS",
            log_transform=False,
        ),
        vmd=VMDConfig(modes=3, backend="sktime"),
        ardl=ARDLSelectionConfig(
            max_target_lag=12,
            max_exog_lag=6,
            top_n=3,
            ic="aic",
            causal=True,
            target_lag_strategy="ardl_select_order",
            exog_lag_strategy="ardl_screen",
            lag_spec_strategy="staged",
            max_lag_specs=16,
            selection_failure_policy="raise",
        ),
        ffnn=FFNNConfig(
            hidden_layer_candidates=(1, 2),
            hidden_width_multipliers=(1.0, 0.5),
            alpha_grid=(1e-4, 1e-3, 1e-2),
            seed_grid=seeds,
            activation="tanh",
            activation_grid=("tanh",),
            learning_rate_init=0.001,
            learning_rate_grid=(0.001,),
            max_iter=150,
            search_strategy="staged_halving",
            fast_max_iter=75,
            top_k_lag_specs=4,
            top_component_candidates=3,
            scaler=scaler,
            batch_size=16,
            validation_fraction=0.20,
            early_stopping_patience=15,
            reduce_lr_patience=6,
            reduce_lr_factor=0.5,
            min_learning_rate=1e-5,
            dropout_rate=0.10,
        ),
        stationarity=StationarityConfig(enabled=True),
        tensorflow=tensorflow or TensorFlowRuntimeConfig(device="gpu"),
        train_ratio=216 / 309,
        val_ratio=46 / 309,
        output_dir=Path(output_dir),
    )


def _write_prepared_csv(dataset: MacroDataset, path: Path) -> None:
    frame = dataset.model_data.reset_index()
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _run_project_workflows(
    prepared_csv: Path,
    config: ExperimentConfig,
    *,
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    shared = run_shared_mean_benchmark_experiments(prepared_csv, config, seeds=seeds)
    univariate_pipeline = "vmd_ardl_univariate_shared_mean"
    univariate_model = "VMD_ARDL_Univariate_SharedMeanSeed_FFNN"
    univariate_config = replace(
        config,
        data=replace(config.data, features=()),
        ardl=replace(
            config.ardl,
            target_lag_strategy="ic_topk",
            exog_lag_strategy="fixed",
            target_lag_top_n=2,
        ),
        output_dir=Path(config.output_dir) / univariate_pipeline,
    )
    started = perf_counter()
    univariate_result = VMDARDLFFNNExperiment(
        univariate_config
    ).run_vmd_shared_mean_seed(
        prepared_csv,
        reporting_seeds=seeds,
        model_name=univariate_model,
        selection_mode="reconstructed_mean_seed",
        screening_mode="none",
        artifact_stem=univariate_pipeline,
    )
    univariate_metrics, univariate_forecasts = _normalized_result(
        univariate_result,
        pipeline=univariate_pipeline,
        model=univariate_model,
        uses_vmd=True,
    )
    seed_metrics = pd.concat(
        [shared["seed_metrics"], univariate_metrics], ignore_index=True, sort=False
    )
    forecasts = pd.concat(
        [shared["comparison_forecasts"], univariate_forecasts],
        ignore_index=True,
        sort=False,
    )
    aggregate = _aggregate_seed_metrics(seed_metrics)
    settings = pd.concat(
        [
            shared["settings"],
            pd.DataFrame(
                [
                    {
                        "pipeline": univariate_pipeline,
                        "model": univariate_model,
                        "included_in_benchmark": True,
                        "uses_vmd": True,
                        "target_lag_strategy": "ic_topk",
                        "exog_lag_strategy": "none_target_only",
                        "selection_protocol": "mean_reconstructed_level_validation_across_reporting_seeds",
                        "selection_seed_aggregation": "shared_mean_validation_across_reporting_seeds",
                        "shared_configuration_across_seeds": True,
                        "search_strategy": "full_grid",
                        "reporting_seeds": seeds,
                        "ffnn_learning_rate_grid": config.ffnn.learning_rate_grid,
                        "final_selection_scale": "reconstructed_level",
                        "enabled": True,
                        "runtime_seconds": perf_counter() - started,
                    }
                ]
            ),
        ],
        ignore_index=True,
        sort=False,
    )
    return {
        "seed_metrics": seed_metrics,
        "comparison_metrics": aggregate,
        "comparison_forecasts": forecasts,
        "settings": settings,
        "shared_four": shared,
        "univariate": univariate_result,
    }


def _windowed(values: np.ndarray, dates: pd.DatetimeIndex, lookback: int = 12):
    X, y, target_dates = [], [], []
    for index in range(lookback, len(values)):
        X.append(values[index - lookback:index])
        y.append(values[index])
        target_dates.append(dates[index])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32), pd.DatetimeIndex(target_dates)


def _build_notebook_model(tf: Any, kind: str, lookback: int = 12):
    kind = kind.lower()
    if kind == "mlp":
        return tf.keras.Sequential(
            [
                tf.keras.layers.Input((lookback,)),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dense(1),
            ]
        )
    recurrent = tf.keras.layers.LSTM if kind == "lstm" else tf.keras.layers.GRU
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input((lookback, 1)),
            recurrent(32),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )


def _notebook_inner_split(
    X: np.ndarray,
    y: np.ndarray,
    validation_fraction: float = 0.20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Chronologically reserve validation from notebook training windows."""
    cut = int(len(X) * (1.0 - validation_fraction))
    if cut < 1 or cut >= len(X):
        raise ValueError("Notebook training data is too short for inner validation.")
    return X[:cut], y[:cut], X[cut:], y[cut:]


def _fit_notebook_model(
    kind: str,
    train_X: np.ndarray,
    train_y: np.ndarray,
    validation_X: np.ndarray,
    validation_y: np.ndarray,
    *,
    runtime: TensorFlowRuntimeConfig,
    seed: int = 42,
):
    tf, device, runtime_metadata = bootstrap_tensorflow(runtime)
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    recurrent = kind in {"lstm", "gru"}
    if recurrent:
        train_X = train_X[..., np.newaxis]
        validation_X = validation_X[..., np.newaxis]
    with tf.device(device):
        model = _build_notebook_model(tf, kind)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse"
        )
        history = model.fit(
            train_X,
            train_y,
            validation_data=(validation_X, validation_y),
            epochs=150,
            batch_size=16,
            shuffle=False,
            verbose=0,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=15, restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=6, min_lr=1e-5
                ),
            ],
        )
    return model, history.history, runtime_metadata


def _predict_notebook_model(model: Any, kind: str, X: np.ndarray) -> np.ndarray:
    model_X = X[..., np.newaxis] if kind in {"lstm", "gru"} else X
    return np.asarray(model.predict(model_X, verbose=0)).reshape(-1)


def _metric_row(model: str, split: str, actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    return {"pipeline": model, "model": model, "split": split, "scale": "level", **evaluate_forecast(actual, predicted)}


def _run_notebook_benchmarks(
    dataset: MacroDataset,
    output_dir: Path,
    *,
    runtime: TensorFlowRuntimeConfig,
) -> dict[str, pd.DataFrame]:
    """Run exact/leaky notebook reproduction and fair chronological comparators."""
    exact_dir = output_dir / "notebook_reproduction"
    fair_dir = output_dir / "notebook_fair_models"
    exact_dir.mkdir(parents=True, exist_ok=False)
    fair_dir.mkdir(parents=True, exist_ok=False)

    # Exact notebook policy: drop missing target rows, 80/20 split, MinMax, and
    # select the neural architecture on test MAPE. This is isolated and labelled.
    exact_series = dataset.raw[TARGET].dropna().astype(float)
    exact_cut = int(len(exact_series) * 0.8)
    exact_scaler = MinMaxScaler().fit(exact_series.iloc[:exact_cut].to_numpy().reshape(-1, 1))
    exact_scaled = exact_scaler.transform(exact_series.to_numpy().reshape(-1, 1)).reshape(-1)
    exact_X, exact_y, exact_dates = _windowed(exact_scaled, exact_series.index)
    exact_train_mask = exact_dates < exact_series.index[exact_cut]
    exact_fit_X, exact_fit_y, exact_validation_X, exact_validation_y = (
        _notebook_inner_split(exact_X[exact_train_mask], exact_y[exact_train_mask])
    )
    exact_rows, exact_forecasts = [], []
    for kind in ("mlp", "lstm", "gru"):
        model, history, _ = _fit_notebook_model(
            kind,
            exact_fit_X,
            exact_fit_y,
            exact_validation_X,
            exact_validation_y,
            runtime=runtime,
        )
        scaled_prediction = _predict_notebook_model(model, kind, exact_X[~exact_train_mask])
        actual = exact_scaler.inverse_transform(exact_y[~exact_train_mask, None]).reshape(-1)
        predicted = exact_scaler.inverse_transform(scaled_prediction[:, None]).reshape(-1)
        row = _metric_row(f"notebook_exact_{kind}", "test", actual, predicted)
        row["selection_warning"] = "architecture_selected_on_test_mape"
        row["epochs_trained"] = len(history.get("loss", ()))
        exact_rows.append(row)
        exact_forecasts.append(pd.DataFrame({"date": exact_dates[~exact_train_mask], "actual": actual, "predicted": predicted, "model": kind}))
    exact_metrics = pd.DataFrame(exact_rows).sort_values("MAPE").reset_index(drop=True)
    exact_metrics["selected_by_notebook"] = exact_metrics.index == 0
    write_csv(exact_metrics, exact_dir / "metrics.csv")
    write_csv(pd.concat(exact_forecasts, ignore_index=True), exact_dir / "forecasts.csv")

    values = dataset.model_data[TARGET].to_numpy(float)
    dates = dataset.model_data.index
    train_end = len(dataset.train)
    scaler = MinMaxScaler().fit(values[:train_end, None])
    scaled = scaler.transform(values[:, None]).reshape(-1)
    X, y, target_dates = _windowed(scaled, dates)
    train_mask = target_dates <= dataset.train.index.max()
    val_mask = target_dates.isin(dataset.validation.index)
    test_mask = target_dates.isin(dataset.test.index)
    metric_rows: list[dict[str, Any]] = []
    forecast_frames: list[pd.DataFrame] = []
    for kind in ("mlp", "lstm", "gru"):
        model, history, _ = _fit_notebook_model(
            kind, X[train_mask], y[train_mask], X[val_mask], y[val_mask], runtime=runtime
        )
        for split, mask in (("validation", val_mask), ("test", test_mask)):
            scaled_prediction = _predict_notebook_model(model, kind, X[mask])
            actual = scaler.inverse_transform(y[mask, None]).reshape(-1)
            predicted = scaler.inverse_transform(scaled_prediction[:, None]).reshape(-1)
            row = _metric_row(f"notebook_{kind}_fair", split, actual, predicted)
            row["epochs_trained"] = len(history.get("loss", ()))
            metric_rows.append(row)
            forecast_frames.append(pd.DataFrame({"date": target_dates[mask], "split": split, "actual": actual, "predicted": predicted, "model": kind}))
    for split, frame in (("validation", dataset.validation), ("test", dataset.test)):
        actual = frame[TARGET].to_numpy(float)
        previous = dataset.model_data[TARGET].shift(1).loc[frame.index].to_numpy(float)
        metric_rows.append(_metric_row("naive_last_observation", split, actual, previous))
        forecast_frames.append(pd.DataFrame({"date": frame.index, "split": split, "actual": actual, "predicted": previous, "model": "naive"}))
    fair_metrics = pd.DataFrame(metric_rows)
    fair_forecasts = pd.concat(forecast_frames, ignore_index=True)
    write_csv(fair_metrics, fair_dir / "metrics.csv")
    write_csv(fair_forecasts, fair_dir / "forecasts.csv")
    return {"exact_metrics": exact_metrics, "fair_metrics": fair_metrics, "fair_forecasts": fair_forecasts}


def run_macro_benchmark(
    csv_path: str | Path,
    output_dir: str | Path,
    *,
    profile: str = "core-only",
    tensorflow: TensorFlowRuntimeConfig | None = None,
    scaler: str = "standard",
    seeds: tuple[int, ...] = (7, 42, 123, 2024, 2025),
) -> dict[str, Any]:
    enabled = pipelines_for_profile(profile)
    runtime = tensorflow or TensorFlowRuntimeConfig(device="gpu")
    # Device policy is checked before data preparation or artifact creation so
    # a production GPU request can never silently fall back to CPU.
    bootstrap_tensorflow(runtime)
    output = Path(output_dir)
    dataset = load_macro_dataset(csv_path)
    selected_features, feature_audit = screen_macro_features(dataset.train)
    config = build_macro_experiment_config(
        selected_features,
        output,
        tensorflow=runtime,
        scaler=scaler,
        seeds=seeds,
    )
    output.mkdir(parents=True, exist_ok=True)
    prepared_csv = output / "macro_model_prefix.csv"
    _write_prepared_csv(dataset, prepared_csv)
    write_csv(dataset.audit, output / "macro_split_audit.csv")
    write_csv(feature_audit, output / "macro_feature_screen_train_only.csv")
    project = _run_project_workflows(prepared_csv, config, seeds=seeds)
    write_csv(project["comparison_metrics"], output / "macro_internal_workflow_comparison.csv")
    write_csv(project["comparison_forecasts"], output / "macro_internal_workflow_forecasts.csv")
    write_csv(project["settings"], output / "macro_experiment_settings.csv")
    notebook = None
    if profile.lower().replace("_", "-") == "full":
        notebook = _run_notebook_benchmarks(dataset, output, runtime=config.tensorflow)
    return {
        "profile": profile.lower().replace("_", "-"),
        "enabled_pipelines": enabled,
        "skipped_pipelines": tuple(
            pipeline for pipeline in CORE_PIPELINES + FULL_ONLY_PIPELINES if pipeline not in enabled
        ),
        "selected_features": selected_features,
        "dataset_sha256": file_sha256(csv_path),
        "notebook_source_sha256": NOTEBOOK_SHA256,
        "notebook_url": NOTEBOOK_URL,
        "config": config,
        "project": project,
        "notebook": notebook,
    }
