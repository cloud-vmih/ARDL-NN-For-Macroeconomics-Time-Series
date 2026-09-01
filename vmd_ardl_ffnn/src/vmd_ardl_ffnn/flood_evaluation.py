"""Hydrology-aware evaluation and split-conformal uncertainty for flood forecasts."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .flood_config import FloodDataConfig
from .flood_routing import LEVEL_NAMES, RunoffThresholds


def _continuous_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = predicted - actual
    rmse = float(np.sqrt(np.mean(np.square(error))))
    mae = float(np.mean(np.abs(error)))
    denominator = float(np.sum(np.square(actual - actual.mean())))
    nse = float(1.0 - np.sum(np.square(error)) / denominator) if denominator > 0 else float("nan")
    if len(actual) > 1 and np.std(actual) > 0 and np.std(predicted) > 0 and actual.mean() != 0:
        correlation = float(np.corrcoef(actual, predicted)[0, 1])
        alpha = float(np.std(predicted) / np.std(actual))
        beta = float(np.mean(predicted) / np.mean(actual))
        kge = float(1.0 - np.sqrt((correlation - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))
    else:
        kge = float("nan")
    return {"RMSE": rmse, "MAE": mae, "NSE": nse, "KGE": kge}


def attach_actuals(
    forecasts: pd.DataFrame,
    observations: pd.DataFrame,
    config: FloodDataConfig,
) -> pd.DataFrame:
    actual = observations[
        [config.zone_col, config.time_col, config.target_col]
    ].rename(
        columns={config.time_col: config.valid_time_col, config.target_col: "actual_runoff_mm_h"}
    )
    return forecasts.merge(
        actual,
        on=[config.zone_col, config.valid_time_col],
        how="left",
        validate="many_to_one",
    )


def score_runoff_forecasts(
    forecasts_with_actuals: pd.DataFrame,
    config: FloodDataConfig,
) -> pd.DataFrame:
    frame = forecasts_with_actuals.dropna(
        subset=["actual_runoff_mm_h", "runoff_prediction_mm_h"]
    ).copy()
    if frame.empty:
        return pd.DataFrame()
    group_columns = ["split", "lead_hour"] if "split" in frame else ["lead_hour"]
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(group_columns, sort=True, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        row.update(
            _continuous_metrics(
                group["actual_runoff_mm_h"].to_numpy(),
                group["runoff_prediction_mm_h"].to_numpy(),
            )
        )
        row["count"] = int(len(group))
        rows.append(row)

    peak_groups = [config.zone_col, config.issue_time_col]
    if "split" in frame:
        peak_groups.insert(0, "split")
    peak_rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(peak_groups, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        actual_idx = group["actual_runoff_mm_h"].idxmax()
        predicted_idx = group["runoff_prediction_mm_h"].idxmax()
        actual_peak = float(frame.loc[actual_idx, "actual_runoff_mm_h"])
        predicted_peak = float(frame.loc[predicted_idx, "runoff_prediction_mm_h"])
        actual_time = pd.Timestamp(frame.loc[actual_idx, config.valid_time_col])
        predicted_time = pd.Timestamp(frame.loc[predicted_idx, config.valid_time_col])
        peak_rows.append(
            {
                "split": group["split"].iloc[0] if "split" in group else "all",
                "peak_magnitude_abs_error": abs(predicted_peak - actual_peak),
                "peak_timing_abs_error_hours": abs(
                    (predicted_time - actual_time).total_seconds() / 3600.0
                ),
            }
        )
    peaks = pd.DataFrame.from_records(peak_rows)
    metrics = pd.DataFrame.from_records(rows)
    if not peaks.empty:
        peak_summary = peaks.groupby("split", observed=True).agg(
            peak_magnitude_MAE=("peak_magnitude_abs_error", "mean"),
            peak_timing_MAE_hours=("peak_timing_abs_error_hours", "mean"),
        )
        if "split" in metrics:
            metrics = metrics.merge(peak_summary, left_on="split", right_index=True, how="left")
        else:
            metrics["peak_magnitude_MAE"] = peaks["peak_magnitude_abs_error"].mean()
            metrics["peak_timing_MAE_hours"] = peaks["peak_timing_abs_error_hours"].mean()
    return metrics


def score_proxy_alerts(
    forecasts_with_actuals: pd.DataFrame,
    thresholds: RunoffThresholds,
    config: FloodDataConfig,
) -> pd.DataFrame:
    frame = forecasts_with_actuals.dropna(subset=["actual_runoff_mm_h"]).copy()
    if frame.empty:
        return pd.DataFrame()
    actual_level = thresholds.levels(
        frame[config.zone_col], frame[config.valid_time_col], frame["actual_runoff_mm_h"]
    )
    predicted_level = thresholds.levels(
        frame[config.zone_col], frame[config.valid_time_col], frame["runoff_prediction_mm_h"]
    )
    frame["actual_level_ord"] = actual_level
    frame["predicted_level_ord"] = predicted_level
    split_values = frame["split"].unique() if "split" in frame else ["all"]
    rows: list[dict[str, object]] = []
    for split in split_values:
        group = frame.loc[frame["split"].eq(split)] if "split" in frame else frame
        for cutoff, level_name in [(1, "yellow_plus"), (2, "orange_plus"), (3, "red")]:
            actual_event = group["actual_level_ord"].to_numpy() >= cutoff
            predicted_event = group["predicted_level_ord"].to_numpy() >= cutoff
            tp = int(np.sum(actual_event & predicted_event))
            fp = int(np.sum(~actual_event & predicted_event))
            fn = int(np.sum(actual_event & ~predicted_event))
            pod = tp / (tp + fn) if tp + fn else float("nan")
            far = fp / (tp + fp) if tp + fp else float("nan")
            csi = tp / (tp + fp + fn) if tp + fp + fn else float("nan")
            f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else float("nan")
            try:
                auprc = float(
                    average_precision_score(
                        actual_event.astype(int), group["runoff_prediction_mm_h"].to_numpy()
                    )
                )
            except ValueError:
                auprc = float("nan")
            rows.append(
                {
                    "split": split,
                    "event_level": level_name,
                    "TP": tp,
                    "FP": fp,
                    "FN": fn,
                    "POD": pod,
                    "FAR": far,
                    "CSI": csi,
                    "F1": f1,
                    "AUPRC": auprc,
                }
            )
    return pd.DataFrame.from_records(rows)


@dataclass
class LeadConformalIntervals:
    """Symmetric split-conformal intervals calibrated independently by lead hour."""

    coverage: float = 0.90
    radius_: dict[int, float] = field(default_factory=dict, init=False)
    fallback_: float | None = field(default=None, init=False)

    def fit(self, validation_forecasts: pd.DataFrame) -> "LeadConformalIntervals":
        if not 0 < self.coverage < 1:
            raise ValueError("coverage must be in (0, 1).")
        frame = validation_forecasts.dropna(
            subset=["actual_runoff_mm_h", "runoff_prediction_mm_h"]
        ).copy()
        if frame.empty:
            raise ValueError("Validation forecasts are required to calibrate intervals.")
        frame["absolute_error"] = (
            frame["actual_runoff_mm_h"] - frame["runoff_prediction_mm_h"]
        ).abs()
        self.radius_ = {}
        for lead, group in frame.groupby("lead_hour", observed=True):
            finite_sample_quantile = min(
                1.0, np.ceil((len(group) + 1) * self.coverage) / len(group)
            )
            self.radius_[int(lead)] = float(
                group["absolute_error"].quantile(
                    finite_sample_quantile, interpolation="higher"
                )
            )
        fallback_quantile = min(
            1.0, np.ceil((len(frame) + 1) * self.coverage) / len(frame)
        )
        self.fallback_ = float(
            frame["absolute_error"].quantile(fallback_quantile, interpolation="higher")
        )
        return self

    def apply(self, forecasts: pd.DataFrame) -> pd.DataFrame:
        if self.fallback_ is None:
            raise RuntimeError("Call fit before apply.")
        output = forecasts.copy()
        radius = output["lead_hour"].astype(int).map(self.radius_).fillna(self.fallback_)
        lower_percentile = int(round((1 - self.coverage) * 50))
        upper_percentile = int(round(100 - (1 - self.coverage) * 50))
        output[f"runoff_lower_p{lower_percentile:02d}"] = np.maximum(
            0.0, output["runoff_prediction_mm_h"] - radius
        )
        output[f"runoff_upper_p{upper_percentile:02d}"] = (
            output["runoff_prediction_mm_h"] + radius
        )
        return output

    def table(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "lead_hour": list(sorted(self.radius_)),
                "absolute_error_radius": [self.radius_[key] for key in sorted(self.radius_)],
                "coverage": self.coverage,
            }
        )
