from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.api import VAR

from ..config import ExperimentConfig
from ..experiment import VMDARDLFFNNExperiment


@dataclass(frozen=True)
class VARNNConfig:
    """Runtime options for the notebook-style VARNN replacement step."""

    lag_criterion: str = "aic"
    maxlags: int | None = None
    min_lag: int = 1
    fixed_lag: int | None = 2
    hidden_units: int = 32
    epochs: int = 100
    batch_size: int = 32
    patience: int = 12
    seed: int = 7


def _log(message: str) -> None:
    print(f"[vmd_ardl_ffnn][varnn] {message}", flush=True)


def _choose_var_lag(
    frame: pd.DataFrame,
    *,
    criterion: str = "aic",
    maxlags: int | None = None,
    min_lag: int = 1,
    fixed_lag: int | None = 2,
) -> tuple[int, dict[str, int], int, pd.DataFrame]:
    clean_frame = frame.dropna()
    if len(clean_frame) < 8:
        raise ValueError("Need at least 8 train observations to select a VAR lag.")

    if maxlags is None:
        n_obs = len(clean_frame)
        n_vars = clean_frame.shape[1]
        maxlags = min(12, max(min_lag, (n_obs - n_vars - 1) // (n_vars + 1)))

    maxlags = int(max(min_lag, min(maxlags, len(clean_frame) - 2)))
    order_result = VAR(clean_frame).select_order(maxlags=maxlags)
    selected_orders = {key: int(value) for key, value in order_result.selected_orders.items()}
    if criterion not in selected_orders:
        raise ValueError(f"Unknown VAR lag criterion: {criterion}")

    selected_lag = int(fixed_lag) if fixed_lag is not None else max(min_lag, int(selected_orders[criterion]))
    selected_lag = int(max(min_lag, min(selected_lag, maxlags)))

    lag_selection_table = pd.DataFrame(order_result.ics)
    lag_selection_table.insert(0, "lag", range(len(lag_selection_table)))
    lag_selection_table["selected_by"] = ""
    for key, value in selected_orders.items():
        if value < len(lag_selection_table):
            marker = lag_selection_table.loc[value, "selected_by"]
            lag_selection_table.loc[value, "selected_by"] = f"{marker},{key}" if marker else key
    lag_selection_table["selected_lag_used"] = selected_lag
    lag_selection_table["lag_selection_mode"] = "fixed" if fixed_lag is not None else criterion

    return selected_lag, selected_orders, maxlags, lag_selection_table


def _scale_frame(frame: pd.DataFrame, columns: list[str], scalers: dict[str, MinMaxScaler]) -> pd.DataFrame:
    scaled = frame[columns].copy()
    for col, scaler in scalers.items():
        scaled[col] = scaler.transform(frame[[col]]).reshape(-1)
    return scaled


def _to_sequences_multivariate_by_split(
    dataset: pd.DataFrame,
    lag: int,
    train_index: pd.Index,
    val_index: pd.Index,
    test_index: pd.Index,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, np.ndarray]:
    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    dates: list[pd.Timestamp] = []
    split_labels: list[str] = []
    for i in range(lag, len(dataset)):
        date = dataset.index[i]
        if date in train_index:
            split = "train"
        elif date in val_index:
            split = "validation"
        elif date in test_index:
            split = "test"
        else:
            continue
        x_rows.append(dataset.iloc[i - lag : i, :].to_numpy(float))
        y_rows.append(dataset.iloc[i : i + 1, :].to_numpy(float).reshape(-1))
        dates.append(pd.Timestamp(date))
        split_labels.append(split)
    return np.asarray(x_rows), np.asarray(y_rows), pd.DatetimeIndex(dates), np.asarray(split_labels)


def _evaluate_forecast(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> pd.Series:
    actual_arr = np.asarray(actual, dtype=float)
    pred_arr = np.asarray(predicted, dtype=float)
    denominator = np.where(np.abs(actual_arr) < 1e-12, np.nan, np.abs(actual_arr))
    return pd.Series(
        {
            "RMSE": float(np.sqrt(mean_squared_error(actual_arr, pred_arr))),
            "MAE": float(mean_absolute_error(actual_arr, pred_arr)),
            "MAPE": float(np.nanmean(np.abs((actual_arr - pred_arr) / denominator)) * 100.0),
        }
    )


def _to_level(
    forecasts: pd.DataFrame,
    *,
    raw_data: pd.DataFrame,
    target: str,
    transform_info: dict[str, str],
    log_transform: bool,
) -> tuple[np.ndarray, np.ndarray]:
    actual_level: list[float] = []
    predicted_level: list[float] = []
    target_transform = transform_info.get(target, "level")
    for row in forecasts.itertuples(index=False):
        date = pd.Timestamp(row.date)
        if target_transform == "diff1":
            pos = raw_data.index.get_loc(date)
            if isinstance(pos, slice) or isinstance(pos, np.ndarray):
                raise ValueError(f"Date is not unique in raw_data index: {date}")
            if pos <= 0:
                raise ValueError(f"Cannot invert diff1 for first timestamp: {date}")
            previous_base = float(raw_data.iloc[pos - 1][target])
            actual_base = previous_base + float(row.actual_raw_transformed)
            predicted_base = previous_base + float(row.predicted_raw_transformed)
        else:
            actual_base = float(row.actual_raw_transformed)
            predicted_base = float(row.predicted_raw_transformed)

        if log_transform:
            actual_base = float(np.exp(actual_base))
            predicted_base = float(np.exp(predicted_base))
        actual_level.append(actual_base)
        predicted_level.append(predicted_base)
    return np.asarray(actual_level), np.asarray(predicted_level)


def _plot_forecasts(forecast_df: pd.DataFrame, output_dir: Path, target: str) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)
    for ax, split in zip(axes, ["train", "validation", "test"]):
        split_df = forecast_df[forecast_df["split"].eq(split)].sort_values("date")
        ax.plot(split_df["date"], split_df["actual_level"], label="Actual level", linewidth=2)
        ax.plot(split_df["date"], split_df["predicted_level"], label="Predicted level", linewidth=2)
        ax.set_title(f"VARNN level actual vs predicted - {split}")
        ax.set_xlabel("Time")
        ax.set_ylabel(f"{target} level")
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_dir / "varnn_level_actual_vs_predicted_train_val_test.png", dpi=160)
    plt.close(fig)


def _tensorflow() -> Any:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError("TensorFlow/Keras is required to train the VARNN model.") from exc
    return tf


def run_varnn_pipeline(
    csv_path: str | Path,
    experiment_config: ExperimentConfig,
    varnn_config: VARNNConfig = VARNNConfig(),
) -> dict[str, pd.DataFrame]:
    """Run the CLI preprocessing flow, replacing ARDL lag + FFNN with VARNN."""

    tf = _tensorflow()
    output_dir = Path("../") / experiment_config.output_dir
    # output_dir.mkdir(parents=True, exist_ok=True)

    experiment = VMDARDLFFNNExperiment(experiment_config)
    _log("Starting VARNN pipeline.")
    raw_data = experiment.load_data(csv_path)
    base_train, base_val, base_test = experiment.split_raw_data(raw_data)
    transformed = experiment._apply_stationarity_transform(raw_data, base_train)

    train = transformed.loc[transformed.index.intersection(base_train.index)].copy()
    validation = transformed.loc[transformed.index.intersection(base_val.index)].copy()
    test = transformed.loc[transformed.index.intersection(base_test.index)].copy()
    _log(f"Stationarity-adjusted rows: train={len(train)}, validation={len(validation)}, test={len(test)}")

    model_columns = [experiment_config.data.target, *experiment_config.data.features]
    target_idx = model_columns.index(experiment_config.data.target)

    scalers = {col: MinMaxScaler(feature_range=(-1, 1)).fit(train[[col]]) for col in model_columns}
    scaled_full = pd.concat(
        [
            _scale_frame(train, model_columns, scalers),
            _scale_frame(validation, model_columns, scalers),
            _scale_frame(test, model_columns, scalers),
        ]
    ).sort_index()

    lag, selected_orders, maxlags_used, lag_selection_table = _choose_var_lag(
        train[model_columns],
        criterion=varnn_config.lag_criterion,
        maxlags=varnn_config.maxlags,
        min_lag=varnn_config.min_lag,
        fixed_lag=varnn_config.fixed_lag,
    )
    _log(
        f"Selected VAR lag P={lag} using {varnn_config.lag_criterion.upper()} "
        f"(searched maxlags={maxlags_used}; raw selected orders={selected_orders})"
    )

    x_all, y_all, dates, splits = _to_sequences_multivariate_by_split(
        scaled_full,
        lag,
        train.index,
        validation.index,
        test.index,
    )
    train_mask = splits == "train"
    val_mask = splits == "validation"
    test_mask = splits == "test"
    train_x, train_y = x_all[train_mask], y_all[train_mask]
    val_x, val_y = x_all[val_mask], y_all[val_mask]
    test_x, test_y = x_all[test_mask], y_all[test_mask]
    if len(train_x) == 0 or len(val_x) == 0 or len(test_x) == 0:
        raise ValueError("VARNN sequence construction produced an empty train/validation/test split.")

    shape_table = pd.DataFrame(
        [
            {"split": "all", "selected_lag": lag, "lag_criterion": varnn_config.lag_criterion, "X_shape": tuple(x_all.shape), "Y_shape": tuple(y_all.shape)},
            {"split": "train", "selected_lag": lag, "lag_criterion": varnn_config.lag_criterion, "X_shape": tuple(train_x.shape), "Y_shape": tuple(train_y.shape)},
            {"split": "validation", "selected_lag": lag, "lag_criterion": varnn_config.lag_criterion, "X_shape": tuple(val_x.shape), "Y_shape": tuple(val_y.shape)},
            {"split": "test", "selected_lag": lag, "lag_criterion": varnn_config.lag_criterion, "X_shape": tuple(test_x.shape), "Y_shape": tuple(test_y.shape)},
        ]
    )

    tf.keras.utils.set_random_seed(varnn_config.seed)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(varnn_config.hidden_units, activation="sigmoid"),
            tf.keras.layers.Dense(len(model_columns)),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=varnn_config.patience,
        min_delta=1e-5,
        restore_best_weights=True,
    )

    start = time.time()
    history = model.fit(
        train_x,
        train_y,
        verbose=2,
        epochs=varnn_config.epochs,
        batch_size=varnn_config.batch_size,
        validation_data=(val_x, val_y),
        callbacks=[early_stopping],
    )
    training_seconds = time.time() - start

    pred_scaled = model.predict(x_all, batch_size=varnn_config.batch_size, verbose=0)
    target_scaler = scalers[experiment_config.data.target]
    actual_target = target_scaler.inverse_transform(y_all[:, target_idx].reshape(-1, 1)).reshape(-1)
    predicted_target = target_scaler.inverse_transform(pred_scaled[:, target_idx].reshape(-1, 1)).reshape(-1)
    forecast_df = pd.DataFrame(
        {
            "date": dates,
            "split": splits,
            "actual_raw_transformed": actual_target,
            "predicted_raw_transformed": predicted_target,
        }
    )

    stationarity_screen = experiment.stationarity_screen_.copy()
    transform_info = dict(zip(stationarity_screen["Variable"], stationarity_screen["Transform used"]))
    forecast_df["actual_level"], forecast_df["predicted_level"] = _to_level(
        forecast_df,
        raw_data=raw_data,
        target=experiment_config.data.target,
        transform_info=transform_info,
        log_transform=experiment_config.data.log_transform,
    )

    metric_rows: list[pd.DataFrame] = []
    for scale, actual_col, predicted_col in [
        ("raw_transformed", "actual_raw_transformed", "predicted_raw_transformed"),
        ("level", "actual_level", "predicted_level"),
    ]:
        rows = (
            forecast_df.groupby("split", sort=False)
            .apply(lambda group: _evaluate_forecast(group[actual_col], group[predicted_col]))
            .reset_index()
        )
        rows.insert(0, "model", "VARNN")
        rows.insert(2, "target", experiment_config.data.target)
        rows.insert(3, "scale", scale)
        metric_rows.append(rows)
    metrics_table = pd.concat(metric_rows, ignore_index=True)
    metrics_wide = metrics_table.pivot(index=["model", "target", "split"], columns="scale", values=["RMSE", "MAE", "MAPE"])
    metrics_wide.columns = [f"{metric}_{scale}" for metric, scale in metrics_wide.columns]
    metrics_wide = metrics_wide.reset_index()

    history_df = pd.DataFrame(history.history)
    history_df["training_seconds"] = training_seconds
    # forecast_df.to_csv(output_dir / "varnn_forecasts.csv", index=False)
    # metrics_table.to_csv(output_dir / "varnn_metrics.csv", index=False)
    metrics_wide.to_csv(output_dir / "varnn_metrics_wide.csv", index=False)
    # shape_table.to_csv(output_dir / "varnn_input_shapes.csv", index=False)
    # lag_selection_table.to_csv(output_dir / "varnn_var_lag_selection.csv", index=False)
    # history_df.to_csv(output_dir / "varnn_training_history.csv", index=False)
    # stationarity_screen.to_csv(output_dir / "varnn_stationarity_screen_train_only.csv", index=False)
    _plot_forecasts(forecast_df, output_dir, experiment_config.data.target)

    return {
        "forecasts": forecast_df,
        "metrics": metrics_table,
        "metrics_wide": metrics_wide,
        "input_shapes": shape_table,
        "var_lag_selection": lag_selection_table,
        "training_history": history_df,
        "stationarity_screen": stationarity_screen,
    }
