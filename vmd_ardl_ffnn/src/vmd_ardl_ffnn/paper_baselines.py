from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .artifacts import write_csv
from .config import ExperimentConfig
from .experiment import VMDARDLFFNNExperiment
from .features import LagSpec
from .lag_selection import ARDLOrderSelector
from .metrics import evaluate_forecast


def _level_values(values: np.ndarray, log_transform: bool) -> np.ndarray:
    return np.exp(values) if log_transform else values


def run_naive_baseline(
    csv_path: str | Path,
    config: ExperimentConfig,
    *,
    lag: int = 1,
    model_name: str = "Naive_LastObservation",
) -> dict[str, pd.DataFrame]:
    experiment = VMDARDLFFNNExperiment(config)
    base = experiment.load_data(csv_path)
    train, validation, test = experiment.split_raw_data(base)
    predicted_base = base[experiment.target].shift(lag)
    rows: list[pd.DataFrame] = []
    for split, evaluation in (("validation", validation), ("test", test)):
        actual = base.loc[evaluation.index, experiment.target].to_numpy(float)
        predicted = predicted_base.reindex(evaluation.index).to_numpy(float)
        rows.append(
            pd.DataFrame(
                {
                    "date": evaluation.index,
                    "split": split,
                    "actual": _level_values(actual, config.data.log_transform),
                    "predicted": _level_values(predicted, config.data.log_transform),
                    "pipeline": model_name,
                    "scale": "level",
                    "forecast_protocol": config.forecast_protocol,
                }
            )
        )
    forecasts = pd.concat(rows, ignore_index=True)
    metrics_rows = []
    for split, group in forecasts.groupby("split", sort=False):
        metrics_rows.append(
            {
                "model": model_name,
                "pipeline": model_name,
                "target": experiment.target,
                "split": split,
                "scale": "level",
                "forecast_protocol": config.forecast_protocol,
                **evaluate_forecast(group["actual"], group["predicted"]),
            }
        )
    metrics = pd.DataFrame(metrics_rows)
    write_csv(forecasts, Path(config.output_dir) / "forecasts.csv")
    write_csv(metrics, Path(config.output_dir) / "metrics.csv")
    return {"forecasts": forecasts, "metrics": metrics}


def run_linear_ardl_baseline(
    csv_path: str | Path,
    config: ExperimentConfig,
) -> dict[str, pd.DataFrame]:
    """Fit a linear ARDL-equivalent regression under rolling one-step inputs."""
    experiment = VMDARDLFFNNExperiment(config)
    base = experiment.load_data(csv_path)
    base_train, base_validation, base_test = experiment.split_raw_data(base)
    transformed = experiment._apply_stationarity_transform(base, base_train)
    train = transformed.loc[transformed.index.intersection(base_train.index)]
    validation = transformed.loc[transformed.index.intersection(base_validation.index)]
    test = transformed.loc[transformed.index.intersection(base_test.index)]
    selector_config = replace(
        config.ardl,
        target_lag_strategy="ardl_select_order_joint",
        selection_failure_policy="raise",
    )
    order_table, specs = ARDLOrderSelector(selector_config).select(
        train,
        experiment.target,
        experiment.features,
    )
    try:
        spec = experiment._safe_spec(specs[0])
    except ValueError:
        spec = LagSpec(target_lags=(1,), exog_lags={feature: tuple() for feature in experiment.features})
        order_table = pd.concat(
            [
                order_table,
                pd.DataFrame(
                    [
                        {
                            "feature": "__standalone_ardl_fallback__",
                            "target_lag": 1,
                            "target_lags": (1,),
                            "selected_for_target_lags": True,
                            "reason": "joint ARDL selected an intercept-only model",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    def fit_predict(fit_data: pd.DataFrame, observed: pd.DataFrame, evaluation: pd.DataFrame, split: str):
        supervised = experiment._build_supervised(fit_data, spec)
        columns = experiment._feature_names(spec)
        model = sm.OLS(
            supervised[experiment.target].to_numpy(float),
            sm.add_constant(supervised[columns].to_numpy(float), has_constant="add"),
        ).fit()
        design = experiment._build_supervised(observed, spec)
        design = design.loc[design.index.intersection(evaluation.index)]
        prediction = model.predict(sm.add_constant(design[columns].to_numpy(float), has_constant="add"))
        return pd.DataFrame(
            {
                "date": design.index,
                "split": split,
                "actual_raw_transformed": design[experiment.target].to_numpy(float),
                "predicted_raw_transformed": np.asarray(prediction, dtype=float),
            }
        )

    validation_forecasts = fit_predict(train, pd.concat([train, validation]), validation, "validation")
    test_forecasts = fit_predict(
        pd.concat([train, validation]),
        pd.concat([train, validation, test]),
        test,
        "test",
    )
    internal = pd.concat([validation_forecasts, test_forecasts], ignore_index=True)
    metrics = experiment._metrics_rows(
        internal,
        "actual_raw_transformed",
        "predicted_raw_transformed",
        "raw_transformed",
    )
    if "actual_level" not in internal:
        internal["actual_level"] = internal["actual_raw_transformed"]
        internal["predicted_level"] = internal["predicted_raw_transformed"]
        metrics.loc[metrics["scale"].eq("raw_transformed"), "scale"] = "level"
    metrics = metrics[metrics["scale"].eq("level")].copy()
    metrics.insert(0, "model", "Standalone_ARDL")
    metrics.insert(1, "pipeline", "standalone_ardl")
    metrics.insert(2, "target", experiment.target)
    forecasts = internal[["date", "split", "actual_level", "predicted_level"]].rename(
        columns={"actual_level": "actual", "predicted_level": "predicted"}
    )
    forecasts["pipeline"] = "standalone_ardl"
    forecasts["scale"] = "level"
    forecasts["forecast_protocol"] = config.forecast_protocol
    write_csv(forecasts, Path(config.output_dir) / "forecasts.csv")
    write_csv(metrics, Path(config.output_dir) / "metrics.csv")
    write_csv(order_table, Path(config.output_dir) / "ardl_orders.csv")
    return {"forecasts": forecasts, "metrics": metrics, "ardl_orders": order_table}
