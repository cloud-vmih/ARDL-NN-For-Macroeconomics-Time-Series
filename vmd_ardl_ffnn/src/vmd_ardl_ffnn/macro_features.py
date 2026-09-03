from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm


DEFAULT_BASE_EXOG_COLUMNS = (
    "initial_claims",
    "retail_sales",
    "consumer_sentiment",
    "cpi_all_items",
    "core_cpi",
    "core_pce",
    "m2_money_supply",
    "building_permits",
    "industrial_prod",
    "nonfarm_payrolls",
    "avg_hourly_earnings",
)

TARGET_DERIVED_COLUMNS = (
    "unemp_3m_avg",
    "unemp_3m_avg_12m_min",
    "sahm_rule_indicator",
    "sahm_triggered",
)


@dataclass(frozen=True)
class FeatureResearchConfig:
    date_column: str = "date"
    target: str = "unemployment_rate"
    base_exog_columns: tuple[str, ...] = DEFAULT_BASE_EXOG_COLUMNS
    target_lags: tuple[int, ...] = tuple(range(1, 13))
    exog_lags: tuple[int, ...] = (1, 3, 6, 12)
    rolling_windows: tuple[int, ...] = (3, 6)
    train_end: pd.Timestamp = pd.Timestamp("2017-12-01")
    validation_end: pd.Timestamp = pd.Timestamp("2021-10-01")
    cv_splits: int = 5
    ridge_alpha: float = 1.0
    minimum_improvement_rate: float = 0.60
    redundancy_threshold: float = 0.90
    max_selected_features: int = 6

    def __post_init__(self) -> None:
        object.__setattr__(self, "train_end", pd.Timestamp(self.train_end))
        object.__setattr__(self, "validation_end", pd.Timestamp(self.validation_end))


@dataclass(frozen=True)
class FeatureScreeningResult:
    scores: pd.DataFrame
    catalog: pd.DataFrame
    eligibility: pd.DataFrame
    selected_signals: tuple[str, ...]
    selected_base_features: tuple[str, ...]
    research_matrix: pd.DataFrame
    split_audit: pd.DataFrame


def _validate_config(config: FeatureResearchConfig) -> None:
    if not config.target_lags or min(config.target_lags) < 1:
        raise ValueError("target_lags must contain only positive lags.")
    if not config.exog_lags or min(config.exog_lags) < 1:
        raise ValueError("exog_lags must contain only positive lags.")
    if config.cv_splits < 2:
        raise ValueError("cv_splits must be at least 2.")
    if not 0.0 <= config.minimum_improvement_rate <= 1.0:
        raise ValueError("minimum_improvement_rate must be between 0 and 1.")
    if not 0.0 <= config.redundancy_threshold <= 1.0:
        raise ValueError("redundancy_threshold must be between 0 and 1.")


def load_macro_frame(path: str | Path, config: FeatureResearchConfig) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {config.date_column, config.target}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Macro CSV is missing required columns: {missing}")
    frame[config.date_column] = pd.to_datetime(frame[config.date_column], errors="raise")
    if frame[config.date_column].duplicated().any():
        raise ValueError("Macro CSV contains duplicate dates.")
    frame = frame.sort_values(config.date_column).reset_index(drop=True)
    return frame


def build_eligibility_table(
    frame: pd.DataFrame, config: FeatureResearchConfig
) -> pd.DataFrame:
    rows = []
    requested = set(config.base_exog_columns)
    target_derived = set(TARGET_DERIVED_COLUMNS)
    for column in frame.columns:
        if column == config.date_column:
            status, reason = "excluded", "date_index"
        elif column == config.target:
            status, reason = "excluded", "forecast_target"
        elif column in target_derived:
            status, reason = "excluded", "derived_from_target_leakage_risk"
        elif column in requested:
            status, reason = "eligible", "raw_macro_candidate"
        else:
            status, reason = "excluded", "derived_or_out_of_scope"
        rows.append({"column": column, "status": status, "reason": reason})
    return pd.DataFrame(rows)


def engineer_forecast_safe_features(
    frame: pd.DataFrame,
    config: FeatureResearchConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create signals that use information dated strictly before the target month."""
    config = config or FeatureResearchConfig()
    _validate_config(config)
    required = {config.date_column, config.target}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Macro frame is missing required columns: {missing}")

    indexed = frame.copy()
    indexed[config.date_column] = pd.to_datetime(indexed[config.date_column], errors="raise")
    indexed = indexed.sort_values(config.date_column).set_index(config.date_column)
    target_values = pd.to_numeric(indexed[config.target], errors="coerce")
    engineered_columns: dict[str, pd.Series] = {config.target: target_values}
    catalog_rows: list[dict[str, object]] = []

    for base in config.base_exog_columns:
        if base not in indexed or base in TARGET_DERIVED_COLUMNS:
            continue
        values = pd.to_numeric(indexed[base], errors="coerce")
        for lag in sorted(set(config.exog_lags)):
            signal = f"{base}__lag_{lag}"
            engineered_columns[signal] = values.shift(lag)
            catalog_rows.append(
                {
                    "signal": signal,
                    "base_feature": base,
                    "transformation": f"level_lag_{lag}",
                    "availability_lag": lag,
                    "leakage_safe": True,
                }
            )

        changes = {
            f"{base}__mom_1_lag_1": (values.pct_change(1, fill_method=None).shift(1), "mom_1"),
            f"{base}__yoy_12_lag_1": (values.pct_change(12, fill_method=None).shift(1), "yoy_12"),
        }
        for signal, (series, transformation) in changes.items():
            engineered_columns[signal] = series
            catalog_rows.append(
                {
                    "signal": signal,
                    "base_feature": base,
                    "transformation": transformation,
                    "availability_lag": 1,
                    "leakage_safe": True,
                }
            )

        shifted = values.shift(1)
        for window in sorted(set(config.rolling_windows)):
            for statistic, series in (
                ("mean", shifted.rolling(window).mean()),
                ("std", shifted.rolling(window).std()),
            ):
                signal = f"{base}__roll_{statistic}_{window}_lag_1"
                engineered_columns[signal] = series
                catalog_rows.append(
                    {
                        "signal": signal,
                        "base_feature": base,
                        "transformation": f"rolling_{statistic}_{window}",
                        "availability_lag": 1,
                        "leakage_safe": True,
                    }
                )

    for lag in sorted(set(config.target_lags)):
        engineered_columns[f"{config.target}__lag_{lag}"] = target_values.shift(lag)
    engineered = pd.DataFrame(engineered_columns, index=indexed.index)
    engineered = engineered.replace([np.inf, -np.inf], np.nan)
    catalog = pd.DataFrame(catalog_rows)
    return engineered, catalog


def _regression_metrics(actual: pd.Series, predicted: np.ndarray) -> tuple[float, float]:
    return (
        float(mean_squared_error(actual, predicted) ** 0.5),
        float(mean_absolute_error(actual, predicted)),
    )


def _aic_gain(y: pd.Series, baseline: pd.DataFrame, candidate: pd.Series) -> float:
    baseline_fit = sm.OLS(y, sm.add_constant(baseline, has_constant="add")).fit()
    candidate_fit = sm.OLS(
        y,
        sm.add_constant(pd.concat([baseline, candidate], axis=1), has_constant="add"),
    ).fit()
    return float(baseline_fit.aic - candidate_fit.aic)


def _score_signals(
    train_matrix: pd.DataFrame,
    catalog: pd.DataFrame,
    config: FeatureResearchConfig,
) -> pd.DataFrame:
    target_lag_columns = [f"{config.target}__lag_{lag}" for lag in config.target_lags]
    required = [config.target, *target_lag_columns, *catalog["signal"].tolist()]
    clean = train_matrix[required].dropna()
    if len(clean) <= config.cv_splits + 1:
        raise ValueError("Too few complete train rows for rolling feature screening.")

    splitter = TimeSeriesSplit(n_splits=config.cv_splits)
    fold_indices = list(splitter.split(clean))
    rows: list[dict[str, object]] = []
    baseline_X = clean[target_lag_columns]
    y = clean[config.target]
    baseline_by_fold: dict[int, tuple[float, float]] = {}
    for fold, (fit_idx, score_idx) in enumerate(fold_indices, start=1):
        model = make_pipeline(StandardScaler(), Ridge(alpha=config.ridge_alpha))
        model.fit(baseline_X.iloc[fit_idx], y.iloc[fit_idx])
        baseline_by_fold[fold] = _regression_metrics(
            y.iloc[score_idx], model.predict(baseline_X.iloc[score_idx])
        )

    for catalog_row in catalog.itertuples(index=False):
        signal = str(catalog_row.signal)
        candidate_X = clean[[*target_lag_columns, signal]]
        rmse_gains: list[float] = []
        mae_gains: list[float] = []
        candidate_rmses: list[float] = []
        for fold, (fit_idx, score_idx) in enumerate(fold_indices, start=1):
            model = make_pipeline(StandardScaler(), Ridge(alpha=config.ridge_alpha))
            model.fit(candidate_X.iloc[fit_idx], y.iloc[fit_idx])
            candidate_rmse, candidate_mae = _regression_metrics(
                y.iloc[score_idx], model.predict(candidate_X.iloc[score_idx])
            )
            baseline_rmse, baseline_mae = baseline_by_fold[fold]
            candidate_rmses.append(candidate_rmse)
            rmse_gains.append(baseline_rmse - candidate_rmse)
            mae_gains.append(baseline_mae - candidate_mae)
        rows.append(
            {
                "signal": signal,
                "base_feature": str(catalog_row.base_feature),
                "transformation": str(catalog_row.transformation),
                "mean_candidate_rmse": float(np.mean(candidate_rmses)),
                "mean_rmse_gain_vs_target_lags": float(np.mean(rmse_gains)),
                "mean_mae_gain_vs_target_lags": float(np.mean(mae_gains)),
                "improvement_rate": float(np.mean(np.asarray(rmse_gains) > 0.0)),
                "rmse_gain_std": float(np.std(rmse_gains, ddof=1)),
                "aic_gain_vs_target_lags": _aic_gain(
                    y, baseline_X, clean[signal].rename(signal)
                ),
                "cv_folds": config.cv_splits,
                "selection_scope": "train_only_expanding_window",
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["mean_rmse_gain_vs_target_lags", "improvement_rate", "aic_gain_vs_target_lags"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def _select_nonredundant_signals(
    scores: pd.DataFrame,
    train_matrix: pd.DataFrame,
    config: FeatureResearchConfig,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    selected_signals: list[str] = []
    selected_bases: list[str] = []
    for row in scores.itertuples(index=False):
        if float(row.mean_rmse_gain_vs_target_lags) <= 0.0:
            continue
        if float(row.improvement_rate) < config.minimum_improvement_rate:
            continue
        signal = str(row.signal)
        base = str(row.base_feature)
        if base in selected_bases:
            continue
        if selected_signals:
            correlations = train_matrix[[signal, *selected_signals]].corr().loc[
                signal, selected_signals
            ]
            if bool((correlations.abs() >= config.redundancy_threshold).any()):
                continue
        selected_signals.append(signal)
        selected_bases.append(base)
        if len(selected_bases) >= config.max_selected_features:
            break
    return tuple(selected_signals), tuple(selected_bases)


def run_feature_screening(
    frame: pd.DataFrame,
    config: FeatureResearchConfig | None = None,
) -> FeatureScreeningResult:
    config = config or FeatureResearchConfig()
    _validate_config(config)
    engineered, catalog = engineer_forecast_safe_features(frame, config)
    eligibility = build_eligibility_table(frame, config)
    train_matrix = engineered.loc[: config.train_end]
    scores = _score_signals(train_matrix, catalog, config)
    selected_signals, selected_bases = _select_nonredundant_signals(
        scores, train_matrix, config
    )
    split_audit = pd.DataFrame(
        [
            {
                "split": "train_selection",
                "start": engineered.index.min(),
                "end": min(config.train_end, engineered.index.max()),
                "used_for_selection": True,
            },
            {
                "split": "validation_locked",
                "start": config.train_end + pd.offsets.MonthBegin(1),
                "end": min(config.validation_end, engineered.index.max()),
                "used_for_selection": False,
            },
            {
                "split": "test_locked",
                "start": config.validation_end + pd.offsets.MonthBegin(1),
                "end": engineered.index.max(),
                "used_for_selection": False,
            },
        ]
    )
    return FeatureScreeningResult(
        scores=scores,
        catalog=catalog,
        eligibility=eligibility,
        selected_signals=selected_signals,
        selected_base_features=selected_bases,
        research_matrix=engineered,
        split_audit=split_audit,
    )


def write_feature_research_artifacts(
    result: FeatureScreeningResult,
    output_dir: str | Path,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result.eligibility.to_csv(output / "feature_eligibility.csv", index=False)
    result.catalog.to_csv(output / "engineered_feature_catalog.csv", index=False)
    result.scores.to_csv(output / "feature_screening_scores.csv", index=False)
    result.split_audit.to_csv(output / "feature_split_audit.csv", index=False)
    selected_matrix = result.research_matrix[
        [column for column in result.research_matrix if column in result.selected_signals]
    ]
    selected_matrix.reset_index().to_csv(
        output / "selected_engineered_signals.csv", index=False
    )
    payload = {
        "selected_base_features": list(result.selected_base_features),
        "selected_engineered_signals": list(result.selected_signals),
        "selection_scope": "train_only_expanding_window",
        "validation_and_test_locked": True,
    }
    (output / "selected_features.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return output


def run_feature_research(
    csv_path: str | Path,
    output_dir: str | Path,
    config: FeatureResearchConfig | None = None,
) -> FeatureScreeningResult:
    config = config or FeatureResearchConfig()
    frame = load_macro_frame(csv_path, config)
    result = run_feature_screening(frame, config)
    write_feature_research_artifacts(result, output_dir)
    return result
