from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

try:
    from statsmodels.tsa.stattools import adfuller, kpss
except ImportError:  # pragma: no cover - stationarity can be disabled.
    adfuller = None
    kpss = None


CPI_FEATURES = ("broad_money", "ppi_qoq", "wti", "gold", "policy_rate", "VNINDEX", "NIKKEI225", "USDVND")
DEFAULT_CPI_DATA = Path(__file__).resolve().parents[1] / "cpi_forecast_selected_variables.csv"


@dataclass(frozen=True)
class RNNDataConfig:
    date_col: str = "date"
    target: str = "cpi_mom_inflation"
    features: tuple[str, ...] = CPI_FEATURES
    freq: str | None = "MS"
    log_transform: bool = False
    stationarity: bool = True
    train_ratio: float = 0.70
    val_ratio: float = 0.15


@dataclass(frozen=True)
class RNNTrainConfig:
    model_type: str = "LSTM"
    model_label: str = ""
    time_steps: int = 12
    units: int = 32
    dense_units: int = 16
    dropout: float = 0.2
    learning_rate: float = 1e-3
    epochs: int = 100
    batch_size: int = 16
    patience: int = 12
    seed: int = 7


def _coerce_features(value: Iterable[str] | str) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    return tuple(value)


def load_processed_like_ardl(csv_path: str | Path, config: RNNDataConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load CSV with the same basic preprocessing contract as the ARDL-FFNN CLI."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    df = pd.read_csv(csv_path)
    required = [config.date_col, config.target, *config.features]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    df[config.date_col] = pd.to_datetime(df[config.date_col], errors="raise")
    df = df.sort_values(config.date_col).drop_duplicates(config.date_col).set_index(config.date_col)
    if config.freq:
        df = df.asfreq(config.freq)

    cols = [config.target, *config.features]
    out = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if config.log_transform:
        non_positive = [col for col in cols if (out[col] <= 0).any()]
        if non_positive:
            raise ValueError(f"log_transform=True requires positive values: {non_positive}")
        out = np.log(out)
    return out.sort_index(), out.copy()


def chronological_split(
    data: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    if not (0 < train_end < val_end < n):
        raise ValueError("Invalid chronological train/validation/test split.")
    return data.iloc[:train_end].copy(), data.iloc[train_end:val_end].copy(), data.iloc[val_end:].copy()


def _safe_adf_pvalue(series: pd.Series, min_obs: int = 12) -> float:
    series = series.dropna()
    if adfuller is None or len(series) < min_obs or series.nunique() <= 1:
        return float("nan")
    try:
        return float(adfuller(series, autolag="AIC")[1])
    except Exception:
        return float("nan")


def _safe_kpss_pvalue(series: pd.Series, min_obs: int = 12) -> float:
    series = series.dropna()
    if kpss is None or len(series) < min_obs or series.nunique() <= 1:
        return float("nan")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(kpss(series, regression="c", nlags="auto")[1])
    except Exception:
        return float("nan")


def apply_train_only_stationarity_transform(
    base_data: pd.DataFrame,
    base_train: pd.DataFrame,
    enabled: bool = True,
    alpha: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match ARDL-FFNN's train-only level/diff1 stationarity decision."""
    rows: list[dict[str, Any]] = []
    transformed = base_data.copy()
    for col in base_data.columns:
        if not enabled:
            transform = "level"
            level_adf = level_kpss = diff_adf = diff_kpss = float("nan")
        else:
            level_adf = _safe_adf_pvalue(base_train[col])
            level_kpss = _safe_kpss_pvalue(base_train[col])
            diff_adf = _safe_adf_pvalue(base_train[col].diff())
            diff_kpss = _safe_kpss_pvalue(base_train[col].diff())
            is_stationary = pd.notna(level_adf) and pd.notna(level_kpss) and level_adf < alpha and level_kpss > alpha
            transform = "level" if is_stationary else "diff1"
        if transform == "diff1":
            transformed[col] = base_data[col].diff()
        rows.append(
            {
                "Variable": col,
                "ADF level p": level_adf,
                "KPSS level p": level_kpss,
                "ADF diff1 p": diff_adf,
                "KPSS diff1 p": diff_kpss,
                "Transform used": transform,
                "Decision basis": "train_only" if enabled else "disabled",
            }
        )
    return transformed.dropna().copy(), pd.DataFrame(rows)


class SequenceDataset:
    def __init__(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
        target: str,
        time_steps: int,
        input_columns: Iterable[str] | None = None,
    ) -> None:
        self.target = target
        self.time_steps = int(time_steps)
        self.columns = list(input_columns) if input_columns is not None else list(train.columns)
        missing = [col for col in self.columns if col not in train.columns]
        if missing:
            raise ValueError(f"Input columns are missing from transformed data: {missing}")
        self.feature_scaler = StandardScaler().fit(train[self.columns])
        self.target_scaler = StandardScaler().fit(train[[target]])

        full = pd.concat([train, val, test]).sort_index()
        scaled = pd.DataFrame(self.feature_scaler.transform(full[self.columns]), index=full.index, columns=self.columns)
        target_scaled = pd.Series(
            self.target_scaler.transform(full[[target]]).reshape(-1),
            index=full.index,
            name=target,
        )
        self.X, self.y, self.dates, self.splits = self._make_sequences(scaled, target_scaled, train.index, val.index, test.index)
        self.X_train, self.y_train = self._select("train")
        self.X_val, self.y_val = self._select("validation")
        self.X_test, self.y_test = self._select("test")

    def _make_sequences(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        train_index: pd.Index,
        val_index: pd.Index,
        test_index: pd.Index,
    ) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, np.ndarray]:
        X_rows: list[np.ndarray] = []
        y_rows: list[float] = []
        dates: list[pd.Timestamp] = []
        splits: list[str] = []
        for pos in range(self.time_steps, len(features)):
            date = features.index[pos]
            if date in train_index:
                split = "train"
            elif date in val_index:
                split = "validation"
            elif date in test_index:
                split = "test"
            else:
                continue
            X_rows.append(features.iloc[pos - self.time_steps : pos].to_numpy(float))
            y_rows.append(float(target.iloc[pos]))
            dates.append(pd.Timestamp(date))
            splits.append(split)
        return np.asarray(X_rows), np.asarray(y_rows).reshape(-1, 1), pd.DatetimeIndex(dates), np.asarray(splits)

    def _select(self, split: str) -> tuple[np.ndarray, np.ndarray]:
        mask = self.splits == split
        return self.X[mask], self.y[mask]

    def frame_for_predictions(self, y_pred_scaled: np.ndarray) -> pd.DataFrame:
        actual = self.target_scaler.inverse_transform(self.y).reshape(-1)
        predicted = self.target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(-1)
        return pd.DataFrame(
            {
                "date": self.dates,
                "split": self.splits,
                "actual_raw_transformed": actual,
                "predicted_raw_transformed": predicted,
            }
        )


def _keras_objects():
    try:
        from tensorflow.keras import backend as K
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        from tensorflow.keras.layers import Dense, Dropout, GRU, LSTM, SimpleRNN
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.optimizers import Adam
    except ImportError:
        from keras import backend as K
        from keras.callbacks import EarlyStopping, ReduceLROnPlateau
        from keras.layers import Dense, Dropout, GRU, LSTM, SimpleRNN
        from keras.models import Sequential
        from keras.optimizers import Adam
    return K, EarlyStopping, ReduceLROnPlateau, Dense, Dropout, GRU, LSTM, SimpleRNN, Sequential, Adam


def rmse(y_true, y_pred):
    K = _keras_objects()[0]
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def mda(y_true, y_pred):
    K = _keras_objects()[0]
    direction_match = K.equal(K.sign(y_true[1:] - y_true[:-1]), K.sign(y_pred[1:] - y_pred[:-1]))
    return K.mean(K.cast(direction_match, K.floatx()))


def build_recurrent_model(input_shape: tuple[int, int], params: RNNTrainConfig):
    _, _, _, Dense, Dropout, GRU, LSTM, SimpleRNN, Sequential, Adam = _keras_objects()
    layer_map = {"RNN": SimpleRNN, "SIMPLERNN": SimpleRNN, "GRU": GRU, "LSTM": LSTM}
    layer_cls = layer_map.get(params.model_type.upper())
    if layer_cls is None:
        raise ValueError("model_type must be one of: RNN, GRU, LSTM")
    model = Sequential(
        [
            layer_cls(units=params.units, input_shape=input_shape),
            Dropout(params.dropout),
            Dense(params.dense_units, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(
        loss="mse",
        optimizer=Adam(learning_rate=params.learning_rate),
        metrics=[rmse, mda],
    )
    return model


def build_SimpleRNN(input_shape, parameters):
    params = _params_from_dict("RNN", parameters)
    shape = tuple(input_shape[-2:]) if len(input_shape) == 3 else tuple(input_shape)
    return build_recurrent_model(shape, params)


def build_GRU(input_shape, parameters):
    params = _params_from_dict("GRU", parameters)
    shape = tuple(input_shape[-2:]) if len(input_shape) == 3 else tuple(input_shape)
    return build_recurrent_model(shape, params)


def build_LSTM(input_shape, parameters):
    params = _params_from_dict("LSTM", parameters)
    shape = tuple(input_shape[-2:]) if len(input_shape) == 3 else tuple(input_shape)
    return build_recurrent_model(shape, params)


def _params_from_dict(model_type: str, parameters: dict[str, Any]) -> RNNTrainConfig:
    return RNNTrainConfig(
        model_type=model_type,
        model_label=str(parameters.get("model_label", "")),
        time_steps=int(parameters.get("time_steps", 12)),
        units=int(parameters.get("RNN_size", parameters.get("units", 64))),
        dense_units=int(parameters.get("FC_size", parameters.get("dense_units", 32))),
        dropout=float(parameters.get("dropout", 0.2)),
        learning_rate=float(parameters.get("lr", parameters.get("learning_rate", 1e-3))),
        epochs=int(parameters.get("epochs", 80)),
        batch_size=int(parameters.get("batch_size", 16)),
        patience=int(parameters.get("earlystop", {}).get("patience", 12)),
    )


def training_callbacks(callback_list, params, filepath=None):
    _, EarlyStopping, ReduceLROnPlateau, *_ = _keras_objects()
    callbacks = []
    if "es" in callback_list:
        early = params.get("earlystop", {})
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                patience=int(early.get("patience", 12)),
                min_delta=float(early.get("min_delta", 1e-5)),
                restore_best_weights=True,
            )
        )
    if "reduce_lr" in callback_list:
        reduce = params.get("reduce_lr", {})
        callbacks.append(
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=float(reduce.get("factor", 0.2)),
                patience=int(reduce.get("patience", 6)),
                min_delta=float(reduce.get("min_delta", 1e-5)),
                min_lr=0.0,
            )
        )
    if "mcp" in callback_list and filepath:
        try:
            from tensorflow.keras.callbacks import ModelCheckpoint
        except ImportError:
            from keras.callbacks import ModelCheckpoint
        callbacks.append(ModelCheckpoint(filepath=filepath, monitor="val_loss", save_best_only=True, mode="min"))
    return callbacks


def evaluate_forecast(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> dict[str, float]:
    actual_arr = np.asarray(actual, dtype=float)
    pred_arr = np.asarray(predicted, dtype=float)
    rmse_value = float(np.sqrt(mean_squared_error(actual_arr, pred_arr)))
    mae_value = float(mean_absolute_error(actual_arr, pred_arr))
    denominator = np.where(np.abs(actual_arr) < 1e-12, np.nan, np.abs(actual_arr))
    mape_value = float(np.nanmean(np.abs((actual_arr - pred_arr) / denominator)) * 100.0)
    return {"RMSE": rmse_value, "MAE": mae_value, "MAPE": mape_value}


class ModelPredictions:
    def __init__(self, prediction_frame: pd.DataFrame):
        self.frame = prediction_frame.copy()
        self.metrics = (
            self.frame.groupby("split")
            .apply(lambda g: pd.Series(evaluate_forecast(g["actual_raw_transformed"], g["predicted_raw_transformed"])))
            .reset_index()
        )

    def plot_predictions(self, output_path: str | Path | None = None, title: str | None = None):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        for split, group in self.frame.groupby("split", sort=False):
            group = group.sort_values("date")
            ax.plot(group["date"], group["actual_raw_transformed"], color="black", linewidth=1.8)
            ax.plot(group["date"], group["predicted_raw_transformed"], label=f"{split} predicted", linewidth=1.5)
        ax.set_title(title or "RNN Predictions")
        ax.set_xlabel("Time")
        ax.set_ylabel("Transformed target")
        ax.legend()
        ax.grid(True, alpha=0.25)
        fig.autofmt_xdate()
        fig.tight_layout()
        if output_path:
            fig.savefig(output_path, dpi=160)
        return fig


def run_rnn_comparison(
    csv_path: str | Path,
    out_dir: str | Path = "compare_models/rnn_results",
    data_config: RNNDataConfig = RNNDataConfig(),
    train_config: RNNTrainConfig = RNNTrainConfig(),
    input_columns: Iterable[str] | None = None,
) -> dict[str, pd.DataFrame]:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    np.random.seed(train_config.seed)
    raw_data, _ = load_processed_like_ardl(csv_path, data_config)
    base_train, base_val, base_test = chronological_split(raw_data, data_config.train_ratio, data_config.val_ratio)
    transformed, stationarity_screen = apply_train_only_stationarity_transform(
        raw_data,
        base_train,
        enabled=data_config.stationarity,
    )
    train = transformed.loc[transformed.index.intersection(base_train.index)].copy()
    val = transformed.loc[transformed.index.intersection(base_val.index)].copy()
    test = transformed.loc[transformed.index.intersection(base_test.index)].copy()
    dataset = SequenceDataset(train, val, test, data_config.target, train_config.time_steps, input_columns=input_columns)
    if min(len(dataset.X_train), len(dataset.X_val), len(dataset.X_test)) == 0:
        raise ValueError("Not enough rows after chronological split and sequence creation.")

    model = build_recurrent_model((dataset.X.shape[1], dataset.X.shape[2]), train_config)
    model_name = train_config.model_label or train_config.model_type.upper()
    file_stem = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace(",", "_")
    callbacks = training_callbacks(
        ["es", "reduce_lr"],
        {
            "earlystop": {"patience": train_config.patience, "min_delta": 1e-5},
            "reduce_lr": {"factor": 0.2, "patience": max(3, train_config.patience // 2), "min_delta": 1e-5},
        },
    )
    history = model.fit(
        dataset.X_train,
        dataset.y_train,
        epochs=train_config.epochs,
        batch_size=train_config.batch_size,
        shuffle=False,
        validation_data=(dataset.X_val, dataset.y_val),
        callbacks=callbacks,
        verbose=2,
    )
    predictions = dataset.frame_for_predictions(model.predict(dataset.X, batch_size=train_config.batch_size, verbose=0))
    metrics = (
        predictions[predictions["split"].isin(["validation", "test"])]
        .groupby("split", sort=False)
        .apply(lambda g: pd.Series(evaluate_forecast(g["actual_raw_transformed"], g["predicted_raw_transformed"])))
        .reset_index()
    )
    metrics.insert(1, "scale", "raw_transformed")
    metrics.insert(0, "model", model_name)
    metrics.insert(1, "input_shape", str(tuple(dataset.X.shape)))
    metrics.insert(2, "input_columns", ",".join(dataset.columns))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out_dir / f"{file_stem}_forecasts.csv", index=False)
    metrics.to_csv(out_dir / f"{file_stem}_metrics.csv", index=False)
    stationarity_screen.to_csv(out_dir / "rnn_stationarity_screen_train_only.csv", index=False)
    pd.DataFrame(history.history).to_csv(out_dir / f"{file_stem}_training_history.csv", index=False)
    try:
        ModelPredictions(predictions[predictions["split"].isin(["validation", "test"])]).plot_predictions(
            out_dir / f"{file_stem}_actual_vs_predicted.png",
            title=f"{model_name} Actual vs Predicted",
        )
    except ImportError:
        pass
    return {
        "predictions": predictions,
        "metrics": metrics,
        "stationarity_screen": stationarity_screen,
        "history": pd.DataFrame(history.history),
        "input_shape": pd.DataFrame(
            [
                {
                    "model": model_name,
                    "input_shape": tuple(dataset.X.shape),
                    "train_shape": tuple(dataset.X_train.shape),
                    "validation_shape": tuple(dataset.X_val.shape),
                    "test_shape": tuple(dataset.X_test.shape),
                    "input_columns": dataset.columns,
                }
            ]
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RNN/GRU/LSTM on ARDL-FFNN preprocessed macro time-series data.")
    parser.add_argument("--data", default=str(DEFAULT_CPI_DATA))
    parser.add_argument("--out", default="compare_models/rnn_results")
    parser.add_argument("--date-col", default="date")
    parser.add_argument("--target", default="cpi_mom_inflation")
    parser.add_argument("--features", nargs="+", default=list(CPI_FEATURES))
    parser.add_argument("--freq", default="MS")
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--no-stationarity", action="store_true")
    parser.add_argument("--model", choices=["RNN", "GRU", "LSTM"], default="LSTM")
    parser.add_argument("--time-steps", type=int, default=12)
    parser.add_argument("--units", type=int, default=64)
    parser.add_argument("--dense-units", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    data_config = RNNDataConfig(
        date_col=args.date_col,
        target=args.target,
        features=_coerce_features(args.features),
        freq=args.freq if args.freq.lower() != "none" else None,
        log_transform=not args.no_log,
        stationarity=not args.no_stationarity,
    )
    train_config = RNNTrainConfig(
        model_type=args.model,
        time_steps=args.time_steps,
        units=args.units,
        dense_units=args.dense_units,
        dropout=args.dropout,
        learning_rate=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        seed=args.seed,
    )
    result = run_rnn_comparison(args.data, args.out, data_config, train_config)
    print(result["metrics"].to_string(index=False))


def load_this_module():
    spec = importlib.util.spec_from_file_location("rnn_models", Path(__file__))
    if spec is None or spec.loader is None:
        raise ImportError("Could not load rnn-models.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    main()
