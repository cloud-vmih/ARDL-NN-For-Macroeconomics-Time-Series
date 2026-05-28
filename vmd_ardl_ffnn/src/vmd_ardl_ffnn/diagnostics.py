from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.ardl import ARDL
from statsmodels.tsa.stattools import adfuller, grangercausalitytests, kpss

from .config import DiagnosticsConfig
from .features import LagSpec


def _safe_adf(series: pd.Series) -> tuple[float, float]:
    clean = series.dropna().astype(float)
    if len(clean) < 12 or clean.nunique() <= 1:
        return np.nan, np.nan
    try:
        stat, pvalue, *_ = adfuller(clean, autolag="AIC")
        return float(stat), float(pvalue)
    except Exception:
        return np.nan, np.nan


def _safe_kpss(series: pd.Series) -> tuple[float, float]:
    clean = series.dropna().astype(float)
    if len(clean) < 12 or clean.nunique() <= 1:
        return np.nan, np.nan
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, pvalue, *_ = kpss(clean, regression="c", nlags="auto")
        return float(stat), float(pvalue)
    except Exception:
        return np.nan, np.nan


def stationarity_decision(adf_pvalue: float, kpss_pvalue: float, alpha: float) -> str:
    if pd.notna(adf_pvalue) and pd.notna(kpss_pvalue) and adf_pvalue < alpha and kpss_pvalue > alpha:
        return "Stationary"
    if pd.notna(adf_pvalue) and pd.notna(kpss_pvalue) and adf_pvalue >= alpha and kpss_pvalue <= alpha:
        return "Non-stationary"
    return "Mixed/unclear"


@dataclass
class ForecastDiagnostics:
    config: DiagnosticsConfig = DiagnosticsConfig()

    def describe_series(self, data: pd.DataFrame, group: str) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for col in data.columns:
            clean = data[col].dropna().astype(float)
            adf_stat, adf_pvalue = _safe_adf(clean)
            kpss_stat, kpss_pvalue = _safe_kpss(clean)
            diff_adf_stat, diff_adf_pvalue = _safe_adf(clean.diff())
            diff_kpss_stat, diff_kpss_pvalue = _safe_kpss(clean.diff())

            jb_stat = jb_pvalue = arch_stat = arch_pvalue = lb_stat = lb_pvalue = np.nan
            if len(clean) >= max(20, self.config.lags + 5) and clean.nunique() > 1:
                centered = clean - clean.mean()
                try:
                    jb_stat, jb_pvalue, _, _ = jarque_bera(clean)
                except Exception:
                    pass
                try:
                    arch_stat, arch_pvalue, _, _ = het_arch(centered, nlags=self.config.lags)
                except Exception:
                    pass
                try:
                    lb = acorr_ljungbox(centered, lags=[self.config.lags], return_df=True).iloc[0]
                    lb_stat = float(lb["lb_stat"])
                    lb_pvalue = float(lb["lb_pvalue"])
                except Exception:
                    pass

            rows.append(
                {
                    "group": group,
                    "series": col,
                    "nobs": int(len(clean)),
                    "mean": float(clean.mean()) if len(clean) else np.nan,
                    "std": float(clean.std()) if len(clean) else np.nan,
                    "ADF_stat": adf_stat,
                    "ADF_pvalue": adf_pvalue,
                    "KPSS_stat": kpss_stat,
                    "KPSS_pvalue": kpss_pvalue,
                    "stationarity_decision": stationarity_decision(adf_pvalue, kpss_pvalue, self.config.alpha),
                    "Diff1_ADF_stat": diff_adf_stat,
                    "Diff1_ADF_pvalue": diff_adf_pvalue,
                    "Diff1_KPSS_stat": diff_kpss_stat,
                    "Diff1_KPSS_pvalue": diff_kpss_pvalue,
                    "diff1_stationarity_decision": stationarity_decision(
                        diff_adf_pvalue, diff_kpss_pvalue, self.config.alpha
                    ),
                    "Jarque_Bera_stat": float(jb_stat),
                    "Jarque_Bera_pvalue": float(jb_pvalue),
                    "ARCH_LM_stat": float(arch_stat),
                    "ARCH_LM_pvalue": float(arch_pvalue),
                    f"Ljung_Box_lag{self.config.lags}_stat": float(lb_stat),
                    f"Ljung_Box_lag{self.config.lags}_pvalue": float(lb_pvalue),
                }
            )
        return pd.DataFrame(rows)

    def granger_causality(self, data: pd.DataFrame, target: str, features: list[str]) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for feature in features:
            pair = data[[target, feature]].dropna()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = grangercausalitytests(pair, maxlag=self.config.granger_maxlag, verbose=False)
                f_stat, pvalue, _, _ = result[self.config.granger_maxlag][0]["ssr_ftest"]
                rows.append(
                    {
                        "cause": feature,
                        "effect": target,
                        "lag": self.config.granger_maxlag,
                        "F_stat": float(f_stat),
                        "pvalue": float(pvalue),
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "cause": feature,
                        "effect": target,
                        "lag": self.config.granger_maxlag,
                        "F_stat": np.nan,
                        "pvalue": np.nan,
                        "error": str(exc),
                    }
                )
        return pd.DataFrame(rows)

    def ardl_residual_diagnostics(
        self,
        data: pd.DataFrame,
        target: str,
        features: list[str],
        spec: LagSpec,
        component: str,
        rank: int,
        trend: str = "c",
    ) -> dict[str, object]:
        exog_lags = {feature: list(spec.exog_lags.get(feature, ())) for feature in features}
        active_exog = [feature for feature, lags in exog_lags.items() if lags]
        clean = data[[target, *active_exog]].dropna()
        try:
            model = ARDL(
                clean[target],
                lags=list(spec.target_lags) or None,
                exog=clean[active_exog] if active_exog else None,
                order={feature: exog_lags[feature] for feature in active_exog} if active_exog else None,
                trend=trend,
                missing="drop",
            ).fit()
            residuals = pd.Series(model.resid).dropna()
            lb = acorr_ljungbox(residuals, lags=[self.config.lags], return_df=True).iloc[0]
            arch_stat, arch_pvalue, _, _ = het_arch(residuals, nlags=self.config.lags)
            jb_stat, jb_pvalue, skewness, kurtosis = jarque_bera(residuals)
            return {
                "component": component,
                "rank": rank,
                "lag_spec": spec.label,
                "nobs": int(model.nobs),
                "AIC": float(model.aic),
                "BIC": float(model.bic),
                "HQIC": float(model.hqic),
                f"resid_Ljung_Box_lag{self.config.lags}_stat": float(lb["lb_stat"]),
                f"resid_Ljung_Box_lag{self.config.lags}_pvalue": float(lb["lb_pvalue"]),
                "resid_ARCH_LM_stat": float(arch_stat),
                "resid_ARCH_LM_pvalue": float(arch_pvalue),
                "resid_Jarque_Bera_stat": float(jb_stat),
                "resid_Jarque_Bera_pvalue": float(jb_pvalue),
                "resid_skewness": float(skewness),
                "resid_kurtosis": float(kurtosis),
            }
        except Exception as exc:
            return {
                "component": component,
                "rank": rank,
                "lag_spec": spec.label,
                "error": str(exc),
            }
