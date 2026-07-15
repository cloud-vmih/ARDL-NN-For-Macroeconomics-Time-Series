from __future__ import annotations

from dataclasses import dataclass
import warnings

import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

from .config import StationarityConfig


@dataclass(frozen=True)
class StationarityResult:
    transformed: pd.DataFrame
    screen: pd.DataFrame
    transforms: dict[str, str]


def _adf_pvalue(series: pd.Series, min_obs: int) -> float:
    clean = series.dropna()
    if len(clean) < min_obs or clean.nunique() <= 1:
        return float("nan")
    try:
        return float(adfuller(clean, autolag="AIC")[1])
    except (ValueError, TypeError, FloatingPointError):
        return float("nan")


def _kpss_pvalue(series: pd.Series, min_obs: int) -> float:
    clean = series.dropna()
    if len(clean) < min_obs or clean.nunique() <= 1:
        return float("nan")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(kpss(clean, regression="c", nlags="auto")[1])
    except (ValueError, TypeError, FloatingPointError):
        return float("nan")


def _is_stationary(adf_pvalue: float, kpss_pvalue: float, alpha: float) -> bool:
    return pd.notna(adf_pvalue) and pd.notna(kpss_pvalue) and adf_pvalue < alpha and kpss_pvalue > alpha


def _label(adf_pvalue: float, kpss_pvalue: float, alpha: float) -> str:
    if _is_stationary(adf_pvalue, kpss_pvalue, alpha):
        return "Stationary"
    if pd.notna(adf_pvalue) and pd.notna(kpss_pvalue) and adf_pvalue >= alpha and kpss_pvalue <= alpha:
        return "Non-stationary"
    return "Mixed/unclear"


def apply_stationarity_transform(
    base_data: pd.DataFrame,
    train_data: pd.DataFrame,
    config: StationarityConfig,
) -> StationarityResult:
    if not config.enabled:
        transforms = {column: "level" for column in base_data.columns}
        screen = pd.DataFrame(
            {
                "Variable": list(base_data.columns),
                "Transform used": ["level"] * len(base_data.columns),
                "Decision basis": ["disabled"] * len(base_data.columns),
            }
        )
        return StationarityResult(base_data.copy(), screen, transforms)

    transformed = base_data.copy()
    transforms: dict[str, str] = {}
    rows: list[dict[str, object]] = []
    for column in base_data.columns:
        level_adf = _adf_pvalue(train_data[column], config.min_obs)
        level_kpss = _kpss_pvalue(train_data[column], config.min_obs)
        diff_adf = _adf_pvalue(train_data[column].diff(), config.min_obs)
        diff_kpss = _kpss_pvalue(train_data[column].diff(), config.min_obs)
        transform = "level" if _is_stationary(level_adf, level_kpss, config.alpha) else "diff1"
        if transform == "diff1":
            transformed[column] = base_data[column].diff()
        transforms[column] = transform
        rows.append(
            {
                "Variable": column,
                "ADF level p": level_adf,
                "KPSS level p": level_kpss,
                "Level decision": _label(level_adf, level_kpss, config.alpha),
                "ADF diff1 p": diff_adf,
                "KPSS diff1 p": diff_kpss,
                "Diff1 decision": _label(diff_adf, diff_kpss, config.alpha),
                "Transform used": transform,
                "Decision basis": "train_only",
                "Likely I(2)?": "No" if _is_stationary(diff_adf, diff_kpss, config.alpha) else "Yes",
            }
        )
    return StationarityResult(transformed.dropna().copy(), pd.DataFrame(rows), transforms)
