"""Generic ARDL + FFNN lag-selection pipeline for economic time series.

The module can model any positive-valued economic target against a list of
economic features. It keeps the old notebook-friendly API by allowing multiple
named model configs through the ``markets`` argument.
"""

from __future__ import annotations

import itertools
import os
import warnings
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera as sm_jarque_bera
from statsmodels.tsa.ardl import ARDL
from statsmodels.tsa.stattools import adfuller, kpss


DEFAULT_TARGET_LAGS = (1, 12)
DEFAULT_ARDL_TARGET_LAGS = (1, 10)
DEFAULT_LAG_CANDIDATES = range(0, 7)

DEFAULT_MARKETS = {
    "US": {
        "target": "Export_US",
        "features": ["PCE", "US_Retail", "US_Sentiment", "USD_VND", "Import_CN"],
        "lag_candidates": DEFAULT_LAG_CANDIDATES,
    },
    "EU": {
        "target": "Export_EU",
        "features": ["EU_RETAIL", "EUR_VND", "Import_CN"],
        "lag_candidates": DEFAULT_LAG_CANDIDATES,
    },
}


def safe_adf_pvalue(series: pd.Series) -> float:
    series = series.dropna()
    if len(series) < 12 or series.nunique() <= 1:
        return np.nan
    try:
        return adfuller(series, autolag="AIC")[1]
    except Exception:
        return np.nan


def safe_kpss_pvalue(series: pd.Series) -> float:
    series = series.dropna()
    if len(series) < 12 or series.nunique() <= 1:
        return np.nan
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return kpss(series, regression="c", nlags="auto")[1]
    except Exception:
        return np.nan


def is_stationary_by_adf_kpss(adf_p: float, kpss_p: float, alpha: float = 0.05) -> bool:
    return pd.notna(adf_p) and pd.notna(kpss_p) and adf_p < alpha and kpss_p > alpha


def stationarity_label(adf_p: float, kpss_p: float, alpha: float = 0.05) -> str:
    if is_stationary_by_adf_kpss(adf_p, kpss_p, alpha):
        return "Stationary"
    if pd.notna(adf_p) and pd.notna(kpss_p) and adf_p >= alpha and kpss_p <= alpha:
        return "Non-stationary"
    return "Mixed/unclear"


def normalize_model_configs(
    *,
    target: Optional[str] = None,
    features: Optional[Sequence[str]] = None,
    feature: Optional[str | Sequence[str]] = None,
    lag_candidates: Iterable[int] = DEFAULT_LAG_CANDIDATES,
    name: str = "Model",
    model_configs: Optional[Mapping[str, Mapping[str, object]]] = None,
    markets: Optional[Mapping[str, Mapping[str, object]]] = None,
) -> Dict[str, Dict[str, object]]:
    """Create one normalized config dictionary from either single or multi-model input."""
    raw_configs = model_configs if model_configs is not None else markets
    if raw_configs is None:
        if features is None and feature is not None:
            features = [feature] if isinstance(feature, str) else list(feature)
        if target is None or features is None:
            raise ValueError("Provide either target + features/feature, or model_configs/markets.")
        raw_configs = {
            name: {
                "target": target,
                "features": list(features),
                "lag_candidates": lag_candidates,
            }
        }

    configs: Dict[str, Dict[str, object]] = {}
    for config_name, cfg in raw_configs.items():
        cfg_target = cfg["target"]
        cfg_features = cfg.get("features", cfg.get("variables"))
        if not cfg_features:
            raise ValueError(f"{config_name}: missing features/variables.")
        configs[str(config_name)] = {
            "target": str(cfg_target),
            "features": list(cfg_features),
            "variables": list(cfg_features),  # Backward-compatible notebook alias.
            "lag_candidates": list(cfg.get("lag_candidates", lag_candidates)),
        }
    return configs


def model_columns_from_configs(configs: Mapping[str, Mapping[str, object]]) -> list:
    cols = []
    for cfg in configs.values():
        for col in [cfg["target"], *cfg["features"]]:
            if col not in cols:
                cols.append(col)
    return cols


def build_lagged_design(
    data: pd.DataFrame,
    target: str,
    lag_map: Mapping[str, int],
    target_lags: Sequence[int] = DEFAULT_TARGET_LAGS,
) -> pd.DataFrame:
    out = pd.DataFrame(index=data.index)
    out[target] = data[target]

    for lag in target_lags:
        out[f"{target}_lag{int(lag)}"] = data[target].shift(int(lag))

    for var, lag in lag_map.items():
        lag = int(lag)
        if lag < 0:
            raise ValueError(f"Feature lag must be >= 0. Received {var}_lag{lag}.")
        out[f"{var}_lag{lag}"] = data[var].shift(lag)

    return out.dropna()


def split_supervised(
    data: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return data.iloc[:train_end], data.iloc[train_end:val_end], data.iloc[val_end:]


def values_as_level_logs(
    values: Sequence[float],
    index: pd.Index,
    target: str,
    transform_map: Mapping[str, str],
    level_log_data: pd.DataFrame,
) -> pd.Series:
    values = pd.Series(np.asarray(values, dtype=float), index=index)
    if transform_map.get(target) == "diff1":
        base = level_log_data[target].shift(1).reindex(index)
        return base + values
    return values


def regression_metrics_level(
    y_true_model: Sequence[float],
    y_pred_model: Sequence[float],
    index: pd.Index,
    target: str,
    transform_map: Mapping[str, str],
    level_log_data: pd.DataFrame,
    inverse_log: bool = True,
) -> Dict[str, float]:
    true_level = values_as_level_logs(y_true_model, index, target, transform_map, level_log_data)
    pred_level = values_as_level_logs(y_pred_model, index, target, transform_map, level_log_data)

    valid = true_level.notna() & pred_level.notna()
    if inverse_log:
        y_true = np.exp(true_level[valid].to_numpy(dtype=float))
        y_pred = np.exp(pred_level[valid].to_numpy(dtype=float))
    else:
        y_true = true_level[valid].to_numpy(dtype=float)
        y_pred = pred_level[valid].to_numpy(dtype=float)

    denom = np.where(np.abs(y_true) < 1e-12, np.nan, y_true)
    metrics = {
        "RMSE": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "MAE": float(np.mean(np.abs(y_true - y_pred))),
        "MAPE": float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100),
    }

    if transform_map.get(target) == "diff1":
        actual_dir = np.sign(np.asarray(y_true_model, dtype=float))[valid.to_numpy()]
        pred_dir = np.sign(np.asarray(y_pred_model, dtype=float))[valid.to_numpy()]
    else:
        prev_level = level_log_data[target].shift(1).reindex(index)
        dir_valid = valid & prev_level.notna()
        actual_dir = np.sign(true_level[dir_valid] - prev_level[dir_valid]).to_numpy(dtype=float)
        pred_dir = np.sign(pred_level[dir_valid] - prev_level[dir_valid]).to_numpy(dtype=float)

    metrics["Directional_Accuracy"] = float((actual_dir == pred_dir).mean() * 100)
    return metrics


def fit_single_variable_ardl(
    data: pd.DataFrame,
    target: str,
    feature: str,
    lag: int,
    target_lags: Sequence[int] = DEFAULT_ARDL_TARGET_LAGS,
):
    lag = int(lag)
    temp = data[[target, feature]].dropna()
    try:
        return ARDL(
            temp[target],
            lags=list(target_lags),
            exog=temp[[feature]],
            order={feature: [lag]},
            trend="c",
            missing="drop",
        ).fit()
    except Exception:
        return None


def ardl_top_lags_by_feature(
    data: pd.DataFrame,
    target: str,
    features: Sequence[str],
    candidate_lags: Iterable[int],
    top_n: int = 3,
    target_lags: Sequence[int] = DEFAULT_ARDL_TARGET_LAGS,
) -> Tuple[pd.DataFrame, Dict[str, list]]:
    rows = []
    for feature in features:
        for lag in candidate_lags:
            result = fit_single_variable_ardl(data, target, feature, int(lag), target_lags)
            if result is None:
                continue
            coef_name = f"{feature}.L{int(lag)}"
            rows.append(
                {
                    "feature": feature,
                    "variable": feature,
                    "lag": int(lag),
                    "AIC": float(result.aic),
                    "BIC": float(result.bic),
                    "coef": float(result.params.get(coef_name, np.nan)),
                    "pvalue": float(result.pvalues.get(coef_name, np.nan)),
                    "nobs": int(result.nobs),
                }
            )

    if not rows:
        raise ValueError(f"No ARDL models could be fitted for target={target}.")

    table = pd.DataFrame(rows).sort_values(["feature", "AIC", "BIC"]).reset_index(drop=True)
    top_lags = {
        feature: table[table["feature"].eq(feature)].head(top_n)["lag"].astype(int).tolist()
        for feature in features
    }
    return table, top_lags


def ardl_top_lags_by_variable(*args, **kwargs):
    """Backward-compatible alias for older notebook code."""
    return ardl_top_lags_by_feature(*args, **kwargs)


def train_eval_ffnn(
    data: pd.DataFrame,
    level_data: pd.DataFrame,
    transform_map: Mapping[str, str],
    target: str,
    lag_map: Mapping[str, int],
    hidden_layer_sizes: Tuple[int, ...] = (8,),
    alpha: float = 1e-3,
    seed: int = 7,
    target_lags: Sequence[int] = DEFAULT_TARGET_LAGS,
    inverse_log: bool = True,
):
    supervised = build_lagged_design(data, target, lag_map, target_lags=target_lags)
    train, val, test = split_supervised(supervised)

    if len(train) < 40 or len(val) < 10 or len(test) < 10:
        return None

    feature_cols = [c for c in supervised.columns if c != target]
    X_train_raw = train[feature_cols].to_numpy(dtype=float)
    X_val_raw = val[feature_cols].to_numpy(dtype=float)
    X_test_raw = test[feature_cols].to_numpy(dtype=float)
    y_train_raw = train[[target]].to_numpy(dtype=float)
    y_val = val[target].to_numpy(dtype=float)
    y_test = test[target].to_numpy(dtype=float)

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_train = x_scaler.fit_transform(X_train_raw)
    X_val = x_scaler.transform(X_val_raw)
    X_test = x_scaler.transform(X_test_raw)
    y_train = y_scaler.fit_transform(y_train_raw).ravel()

    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        alpha=alpha,
        learning_rate_init=0.01,
        max_iter=500,
        early_stopping=False,
        n_iter_no_change=30,
        random_state=seed,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(X_train, y_train)

    pred_val_model = y_scaler.inverse_transform(model.predict(X_val).reshape(-1, 1)).ravel()
    pred_test_model = y_scaler.inverse_transform(model.predict(X_test).reshape(-1, 1)).ravel()

    return {
        "lag_map": dict(lag_map),
        "hidden_layer_sizes": hidden_layer_sizes,
        "alpha": alpha,
        "seed": seed,
        "n_iter": int(model.n_iter_),
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "val_metrics": regression_metrics_level(
            y_val, pred_val_model, val.index, target, transform_map, level_data, inverse_log
        ),
        "test_metrics": regression_metrics_level(
            y_test, pred_test_model, test.index, target, transform_map, level_data, inverse_log
        ),
    }


def ffnn_grid_search_for_lag_maps(
    data: pd.DataFrame,
    level_data: pd.DataFrame,
    transform_map: Mapping[str, str],
    target: str,
    lag_maps: Sequence[Mapping[str, int]],
    hidden_grid: Sequence[Tuple[int, ...]] = ((8,),),
    alpha_grid: Sequence[float] = (1e-3,),
    seed_grid: Sequence[int] = (7,),
    target_lags: Sequence[int] = DEFAULT_TARGET_LAGS,
    inverse_log: bool = True,
) -> pd.DataFrame:
    rows = []
    for lag_map in lag_maps:
        for hidden in hidden_grid:
            for alpha in alpha_grid:
                for seed in seed_grid:
                    result = train_eval_ffnn(
                        data=data,
                        level_data=level_data,
                        transform_map=transform_map,
                        target=target,
                        lag_map=lag_map,
                        hidden_layer_sizes=hidden,
                        alpha=alpha,
                        seed=seed,
                        target_lags=target_lags,
                        inverse_log=inverse_log,
                    )
                    if result is None:
                        continue
                    row = {
                        "lag_map": result["lag_map"],
                        "hidden_layer_sizes": result["hidden_layer_sizes"],
                        "alpha": result["alpha"],
                        "seed": result["seed"],
                        "n_iter": result["n_iter"],
                        "n_train": result["n_train"],
                        "n_val": result["n_val"],
                        "n_test": result["n_test"],
                        **{f"Val {k}": v for k, v in result["val_metrics"].items()},
                        **{f"Test {k}": v for k, v in result["test_metrics"].items()},
                    }
                    for feature, lag in result["lag_map"].items():
                        row[f"{feature}_lag"] = lag
                    rows.append(row)

    if not rows:
        raise ValueError("No FFNN models could be trained. Check sample size, lags, and split ratios.")

    return pd.DataFrame(rows).sort_values(["Val RMSE", "Val MAE", "Val MAPE"]).reset_index(drop=True)


def make_lag_maps_from_top_lags(top_lags: Mapping[str, Sequence[int]]) -> list:
    features = list(top_lags.keys())
    return [dict(zip(features, map(int, combo))) for combo in itertools.product(*[top_lags[f] for f in features])]


def make_common_lag_maps(features: Sequence[str], candidate_lags: Iterable[int]) -> list:
    lag_list = [int(lag) for lag in candidate_lags if int(lag) >= 1]
    if not lag_list:
        raise ValueError("candidate_lags must contain at least one lag >= 1 for the common-lag baseline.")
    return [{feature: int(lag) for feature in features} for lag in lag_list]


def compact_result_rows(rows: Sequence[pd.Series]) -> pd.DataFrame:
    keep_cols = [
        "Market",
        "Name",
        "Target",
        "Model",
        "lag_map",
        "hidden_layer_sizes",
        "alpha",
        "seed",
        "n_train",
        "n_val",
        "n_test",
        "Val RMSE",
        "Val MAE",
        "Val MAPE",
        "Val Directional_Accuracy",
        "Test RMSE",
        "Test MAE",
        "Test MAPE",
        "Test Directional_Accuracy",
    ]
    out = pd.DataFrame(rows)
    for col in ["Market", "Name"]:
        if col not in out.columns and ("Market" in out.columns or "Name" in out.columns):
            out[col] = out.get("Market", out.get("Name"))
    missing_cols = [col for col in keep_cols if col not in out.columns]
    if missing_cols:
        raise KeyError(f"Missing columns in result table: {missing_cols}")
    return out[keep_cols].sort_values(["Name", "Model"]).reset_index(drop=True)


def compare_individual_vs_common(comparison_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    individual_name = "Lag riêng - ARDL top3 + FFNN grid"
    common_name = "Lag chung - FFNN grid"
    group_col = "Name" if "Name" in comparison_table.columns else "Market"

    for name in comparison_table[group_col].unique():
        temp = comparison_table[comparison_table[group_col].eq(name)].set_index("Model")
        if individual_name not in temp.index or common_name not in temp.index:
            raise KeyError(f"{name}: missing one of the two models for comparison.")

        individual = temp.loc[individual_name]
        common = temp.loc[common_name]
        rows.append(
            {
                group_col: name,
                "Market": name,
                "Target": individual["Target"],
                "Lag chung map": common["lag_map"],
                "Lag riêng map": individual["lag_map"],
                "Lag chung Test RMSE": common["Test RMSE"],
                "Lag riêng Test RMSE": individual["Test RMSE"],
                "Better Test RMSE": individual_name
                if individual["Test RMSE"] < common["Test RMSE"]
                else common_name,
                "RMSE improvement lag riêng vs lag chung %": (
                    common["Test RMSE"] - individual["Test RMSE"]
                )
                / common["Test RMSE"]
                * 100,
                "Lag chung Test MAE": common["Test MAE"],
                "Lag riêng Test MAE": individual["Test MAE"],
                "Better Test MAE": individual_name
                if individual["Test MAE"] < common["Test MAE"]
                else common_name,
                "MAE improvement lag riêng vs lag chung %": (
                    common["Test MAE"] - individual["Test MAE"]
                )
                / common["Test MAE"]
                * 100,
                "Lag chung Test MAPE": common["Test MAPE"],
                "Lag riêng Test MAPE": individual["Test MAPE"],
                "Better Test MAPE": individual_name
                if individual["Test MAPE"] < common["Test MAPE"]
                else common_name,
                "MAPE improvement lag riêng vs lag chung %": (
                    common["Test MAPE"] - individual["Test MAPE"]
                )
                / common["Test MAPE"]
                * 100,
                "DA change point lag riêng vs lag chung": individual["Test Directional_Accuracy"]
                - common["Test Directional_Accuracy"],
            }
        )
    return pd.DataFrame(rows)


def ardl_diagnostics(
    data: pd.DataFrame,
    target: str,
    lag_map: Mapping[str, int],
    target_lags: Sequence[int] = DEFAULT_TARGET_LAGS,
):
    features = list(lag_map.keys())
    result = ARDL(
        data[target],
        lags=list(target_lags),
        exog=data[features],
        order={feature: [int(lag)] for feature, lag in lag_map.items()},
        trend="c",
        missing="drop",
    ).fit()

    resid = pd.Series(result.resid).dropna()
    lb = acorr_ljungbox(resid, lags=[12], return_df=True)
    arch_stat, arch_pvalue, _, _ = het_arch(resid, nlags=12)
    jb_stat, jb_pvalue, skewness, kurt = sm_jarque_bera(resid)
    diag = pd.DataFrame(
        {
            "Test": ["Ljung-Box lag 12", "ARCH-LM lag 12", "Jarque-Bera"],
            "Statistic": [float(lb["lb_stat"].iloc[0]), float(arch_stat), float(jb_stat)],
            "p-value": [float(lb["lb_pvalue"].iloc[0]), float(arch_pvalue), float(jb_pvalue)],
        }
    )
    return result, diag


class ARDLFFNNModel:
    """ARDL-based lag preselection followed by FFNN lag-map evaluation.

    Basic usage:
        model = ARDLFFNNModel(data=df, target="GDP", features=["CPI", "Interest"])
        model.run_all()

    Multiple models can be passed through ``model_configs`` or the legacy
    ``markets`` argument. Each config needs ``target`` and ``features``.
    """

    def __init__(
        self,
        data: Optional[pd.DataFrame | str | Path] = None,
        target: Optional[str] = None,
        features: Optional[Sequence[str]] = None,
        *,
        feature: Optional[str | Sequence[str]] = None,
        data_path: Optional[str | Path] = None,
        date_col: str = "TIME_PERIOD",
        freq: Optional[str] = "MS",
        name: str = "Model",
        model_configs: Optional[Mapping[str, Mapping[str, object]]] = None,
        markets: Optional[Mapping[str, Mapping[str, object]]] = None,
        lag_candidates: Iterable[int] = DEFAULT_LAG_CANDIDATES,
        model_cols: Optional[Sequence[str]] = None,
        log_transform: bool = True,
        target_lags: Sequence[int] = DEFAULT_TARGET_LAGS,
        ardl_target_lags: Sequence[int] = DEFAULT_ARDL_TARGET_LAGS,
        top_n: int = 3,
        hidden_grid: Sequence[Tuple[int, ...]] = ((8,),),
        alpha_grid: Sequence[float] = (1e-3,),
        seed_grid: Sequence[int] = (7,),
    ) -> None:
        if isinstance(data, (str, Path)) and data_path is None:
            data_path = data
            data = None

        self.input_data = data
        self.data_path = Path(data_path) if data_path is not None else None
        self.date_col = date_col
        self.freq = freq
        self.log_transform = log_transform
        self.target_lags = tuple(target_lags)
        self.ardl_target_lags = tuple(ardl_target_lags)
        self.top_n = top_n
        self.hidden_grid = tuple(hidden_grid)
        self.alpha_grid = tuple(alpha_grid)
        self.seed_grid = tuple(seed_grid)

        self.configs = normalize_model_configs(
            target=target,
            features=features,
            feature=feature,
            lag_candidates=lag_candidates,
            name=name,
            model_configs=model_configs,
            markets=markets,
        )
        self.markets = self.configs  # Backward-compatible notebook alias.
        self.model_cols = list(model_cols or model_columns_from_configs(self.configs))

        self.df: Optional[pd.DataFrame] = None
        self.df_model: Optional[pd.DataFrame] = None
        self.level_df: Optional[pd.DataFrame] = None
        self.log_df: Optional[pd.DataFrame] = None
        self.model_df: Optional[pd.DataFrame] = None
        self.transform_info: Dict[str, str] = {}
        self.stationarity_screen: Optional[pd.DataFrame] = None
        self.ardl_lag_tables: Dict[str, pd.DataFrame] = {}
        self.top_lags_by_model: Dict[str, Dict[str, list]] = {}
        self.top_lags_by_market: Dict[str, Dict[str, list]] = self.top_lags_by_model
        self.ffnn_lag_search_tables: Dict[str, pd.DataFrame] = {}
        self.common_lag_search_tables: Dict[str, pd.DataFrame] = {}
        self.best_separate_lag_rows: list = []
        self.best_common_lag_rows: list = []
        self.comparison: Optional[pd.DataFrame] = None
        self.final_summary: Optional[pd.DataFrame] = None
        self.best_lag_map_by_model: Dict[str, Dict[str, int]] = {}
        self.best_lag_map_by_market: Dict[str, Dict[str, int]] = self.best_lag_map_by_model
        self.best_lag_map: Optional[Dict[str, int]] = None
        self.diagnostics: Dict[str, Tuple[object, pd.DataFrame]] = {}

    def _read_data(self) -> pd.DataFrame:
        if self.input_data is not None:
            return self.input_data.copy()
        if self.data_path is None:
            raise ValueError("Provide data as a DataFrame or data_path as a CSV path.")
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        return pd.read_csv(self.data_path)

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        self.df = self._read_data()
        missing_cols = [col for col in [self.date_col, *self.model_cols] if col not in self.df.columns]
        if missing_cols:
            raise KeyError(f"Missing columns in input data: {missing_cols}")

        self.df[self.date_col] = pd.to_datetime(self.df[self.date_col])
        self.df_model = self.df.set_index(self.date_col).sort_index()
        if self.freq:
            self.df_model = self.df_model.asfreq(self.freq)

        raw_level = self.df_model[self.model_cols].apply(pd.to_numeric, errors="coerce").dropna().copy()
        if self.log_transform:
            non_positive = [col for col in raw_level.columns if (raw_level[col] <= 0).any()]
            if non_positive:
                raise ValueError(
                    "log_transform=True requires all model columns to be positive. "
                    f"Non-positive columns: {non_positive}"
                )
            self.level_df = np.log(raw_level).copy()
            self.log_df = self.level_df
        else:
            self.level_df = raw_level.copy()
            self.log_df = self.level_df
        return self.df, self.df_model, self.level_df

    def run_stationarity_screen(self) -> pd.DataFrame:
        if self.level_df is None:
            self.load_data()

        screen_rows = []
        self.transform_info = {}
        self.model_df = self.level_df.copy()

        for col in self.level_df.columns:
            level_adf = safe_adf_pvalue(self.level_df[col])
            level_kpss = safe_kpss_pvalue(self.level_df[col])
            diff_series = self.level_df[col].diff()
            diff_adf = safe_adf_pvalue(diff_series)
            diff_kpss = safe_kpss_pvalue(diff_series)

            transform = "level" if is_stationary_by_adf_kpss(level_adf, level_kpss) else "diff1"
            if transform == "diff1":
                self.model_df[col] = diff_series

            self.transform_info[col] = transform
            screen_rows.append(
                {
                    "Variable": col,
                    "ADF level p": level_adf,
                    "KPSS level p": level_kpss,
                    "Level decision": stationarity_label(level_adf, level_kpss),
                    "ADF diff1 p": diff_adf,
                    "KPSS diff1 p": diff_kpss,
                    "Diff1 decision": stationarity_label(diff_adf, diff_kpss),
                    "Transform used": transform,
                    "Likely I(2)?": "Yes"
                    if not is_stationary_by_adf_kpss(diff_adf, diff_kpss)
                    else "No",
                }
            )

        self.stationarity_screen = pd.DataFrame(screen_rows)
        self.model_df = self.model_df.dropna().copy()
        return self.stationarity_screen

    def select_ardl_top_lags(self, top_n: Optional[int] = None):
        if self.model_df is None:
            self.run_stationarity_screen()

        top_n = int(top_n or self.top_n)
        self.ardl_lag_tables = {}
        self.top_lags_by_model = {}
        for model_name, cfg in self.configs.items():
            table, top_lags = ardl_top_lags_by_feature(
                data=self.model_df,
                target=cfg["target"],
                features=cfg["features"],
                candidate_lags=cfg["lag_candidates"],
                top_n=top_n,
                target_lags=self.ardl_target_lags,
            )
            self.ardl_lag_tables[model_name] = table
            self.top_lags_by_model[model_name] = top_lags
        self.top_lags_by_market = self.top_lags_by_model
        return self.ardl_lag_tables, self.top_lags_by_model

    def run_separate_lag_ffnn_search(self):
        if not self.top_lags_by_model:
            self.select_ardl_top_lags()

        self.ffnn_lag_search_tables = {}
        self.best_separate_lag_rows = []
        for model_name, cfg in self.configs.items():
            lag_maps = make_lag_maps_from_top_lags(self.top_lags_by_model[model_name])
            table = ffnn_grid_search_for_lag_maps(
                data=self.model_df,
                level_data=self.level_df,
                transform_map=self.transform_info,
                target=cfg["target"],
                lag_maps=lag_maps,
                hidden_grid=self.hidden_grid,
                alpha_grid=self.alpha_grid,
                seed_grid=self.seed_grid,
                target_lags=self.target_lags,
                inverse_log=self.log_transform,
            )
            self.ffnn_lag_search_tables[model_name] = table
            best = table.iloc[0].copy()
            best["Market"] = model_name
            best["Name"] = model_name
            best["Target"] = cfg["target"]
            best["Model"] = "Lag riêng - ARDL top3 + FFNN grid"
            self.best_separate_lag_rows.append(best)
        return self.ffnn_lag_search_tables, self.best_separate_lag_rows

    def run_common_lag_ffnn_search(self):
        if self.model_df is None:
            self.run_stationarity_screen()

        self.common_lag_search_tables = {}
        self.best_common_lag_rows = []
        for model_name, cfg in self.configs.items():
            lag_maps = make_common_lag_maps(cfg["features"], cfg["lag_candidates"])
            table = ffnn_grid_search_for_lag_maps(
                data=self.model_df,
                level_data=self.level_df,
                transform_map=self.transform_info,
                target=cfg["target"],
                lag_maps=lag_maps,
                hidden_grid=self.hidden_grid,
                alpha_grid=self.alpha_grid,
                seed_grid=self.seed_grid,
                target_lags=self.target_lags,
                inverse_log=self.log_transform,
            )
            self.common_lag_search_tables[model_name] = table
            best = table.iloc[0].copy()
            best["Market"] = model_name
            best["Name"] = model_name
            best["Target"] = cfg["target"]
            best["Model"] = "Lag chung - FFNN grid"
            self.best_common_lag_rows.append(best)
        return self.common_lag_search_tables, self.best_common_lag_rows

    def build_comparison(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if not self.best_separate_lag_rows:
            self.run_separate_lag_ffnn_search()
        if not self.best_common_lag_rows:
            self.run_common_lag_ffnn_search()

        self.comparison = compact_result_rows(self.best_common_lag_rows + self.best_separate_lag_rows)
        self.final_summary = compare_individual_vs_common(self.comparison)
        return self.comparison, self.final_summary

    def extract_best_lag_maps(self) -> Dict[str, Dict[str, int]]:
        if self.comparison is None:
            self.build_comparison()

        self.best_lag_map_by_model = {}
        for model_name in self.configs:
            best_row = self.comparison[
                self.comparison["Name"].eq(model_name)
                & self.comparison["Model"].eq("Lag riêng - ARDL top3 + FFNN grid")
            ].sort_values("Val RMSE").iloc[0]
            self.best_lag_map_by_model[model_name] = dict(best_row["lag_map"])

        self.best_lag_map_by_market = self.best_lag_map_by_model
        self.best_lag_map = next(iter(self.best_lag_map_by_model.values()))
        return self.best_lag_map_by_model

    def run_diagnostics(self):
        if not self.best_lag_map_by_model:
            self.extract_best_lag_maps()

        self.diagnostics = {}
        for model_name, cfg in self.configs.items():
            lag_map = self.best_lag_map_by_model[model_name]
            result, diag = ardl_diagnostics(
                self.model_df,
                cfg["target"],
                lag_map,
                target_lags=self.target_lags,
            )
            self.diagnostics[model_name] = (result, diag)
        return self.diagnostics

    def run_all(self) -> "ARDLFFNNModel":
        self.load_data()
        self.run_stationarity_screen()
        self.select_ardl_top_lags()
        self.run_separate_lag_ffnn_search()
        self.run_common_lag_ffnn_search()
        self.build_comparison()
        self.extract_best_lag_maps()
        self.run_diagnostics()
        return self


def main() -> ARDLFFNNModel:
    """Run a sample model from df_final.csv using target + features parameters."""
    sample = ARDLFFNNModel(
        data_path="df_final.csv",
        target="Export_US",
        features=["PCE", "US_Retail", "US_Sentiment", "USD_VND", "Import_CN"],
        name="Sample_US_Export",
        lag_candidates=range(0, 7),
    ).run_all()

    print("Best lag map:")
    print(sample.best_lag_map)
    print("\nComparison:")
    print(sample.comparison.round(4).to_string(index=False))
    print("\nFinal summary:")
    print(sample.final_summary.round(4).to_string(index=False))
    return sample


if __name__ == "__main__":
    main()
