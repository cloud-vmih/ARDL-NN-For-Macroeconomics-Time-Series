"""Forecast-safe SARIMAX baseline for monthly one-step forecasts."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


@dataclass(frozen=True)
class SARIMAXConfig:
    """Small validation grid suitable for the audited monthly macro panel."""

    order_grid: tuple[tuple[int, int, int], ...] = (
        (1, 0, 0),
        (2, 0, 0),
        (0, 1, 1),
        (1, 1, 0),
        (1, 1, 1),
    )
    seasonal_order_grid: tuple[tuple[int, int, int, int], ...] = (
        (0, 0, 0, 12),
        (1, 0, 0, 12),
        (0, 1, 1, 12),
    )
    exog_lag: int = 1
    trend: str = "c"
    maxiter: int = 100

    def __post_init__(self) -> None:
        if not self.order_grid:
            raise ValueError("SARIMAX order grid must not be empty.")
        if not self.seasonal_order_grid:
            raise ValueError("SARIMAX seasonal-order grid must not be empty.")
        if any(len(order) != 3 or min(order) < 0 for order in self.order_grid):
            raise ValueError("Every SARIMAX order must be a nonnegative (p, d, q) tuple.")
        if any(
            len(order) != 4 or min(order[:3]) < 0 or order[3] < 2
            for order in self.seasonal_order_grid
        ):
            raise ValueError(
                "Every seasonal SARIMAX order must be a valid (P, D, Q, s) tuple."
            )
        if self.exog_lag < 1:
            raise ValueError("SARIMAX exogenous lag must be >= 1 for forecast safety.")
        if self.maxiter < 1:
            raise ValueError("SARIMAX maxiter must be positive.")


def fit_predict_sarimax(
    frame: pd.DataFrame,
    *,
    target: str,
    features: tuple[str, ...],
    fit_end: int,
    prediction_start: int,
    prediction_end: int,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    config: SARIMAXConfig,
) -> tuple[np.ndarray, float, bool, int]:
    """Fit before ``fit_end`` and issue rolling one-step level forecasts.

    Exogenous values are shifted by at least one month. After each forecast the
    observed target is appended to the state without re-estimating parameters,
    matching the benchmark's rolling one-step information set.
    """
    required = (target, *features)
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"SARIMAX input is missing columns: {missing}")
    if not (
        config.exog_lag < fit_end <= prediction_start < prediction_end <= len(frame)
    ):
        raise ValueError("Invalid chronological SARIMAX fit/prediction boundaries.")
    values = frame.loc[:, required].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("SARIMAX input contains missing or nonfinite values.")

    endog = frame[target].to_numpy(dtype=float)
    exog = (
        frame.loc[:, features].shift(config.exog_lag).to_numpy(dtype=float)
        if features
        else None
    )
    first = config.exog_lag
    fit_endog = endog[first:fit_end]
    fit_exog = exog[first:fit_end] if exog is not None else None
    if len(fit_endog) < 24:
        raise ValueError("SARIMAX requires at least 24 forecast-safe fit rows.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = SARIMAX(
            fit_endog,
            exog=fit_exog,
            order=order,
            seasonal_order=seasonal_order,
            trend=config.trend,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False, maxiter=config.maxiter)

    predictions: list[float] = []
    state = fitted
    for index in range(prediction_start, prediction_end):
        next_exog = exog[index : index + 1] if exog is not None else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            predicted = np.asarray(
                state.get_forecast(steps=1, exog=next_exog).predicted_mean,
                dtype=float,
            ).reshape(-1)
        if predicted.size != 1 or not np.isfinite(predicted[0]):
            raise ValueError(f"SARIMAX produced a nonfinite forecast at row {index}.")
        predictions.append(float(predicted[0]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            state = state.append(
                np.asarray([endog[index]], dtype=float),
                exog=next_exog,
                refit=False,
            )

    mle_retvals = getattr(fitted, "mle_retvals", {}) or {}
    return (
        np.asarray(predictions, dtype=float),
        float(fitted.aic),
        bool(mle_retvals.get("converged", True)),
        int(mle_retvals.get("iterations", 0)),
    )
