from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


def _loss_differential(
    actual: np.ndarray,
    first_prediction: np.ndarray,
    second_prediction: np.ndarray,
    loss: str,
) -> np.ndarray:
    first_error = actual - first_prediction
    second_error = actual - second_prediction
    if loss == "squared":
        return np.square(first_error) - np.square(second_error)
    if loss == "absolute":
        return np.abs(first_error) - np.abs(second_error)
    raise ValueError("loss must be 'squared' or 'absolute'.")


def moving_block_bootstrap_mean_ci(
    values: np.ndarray,
    *,
    confidence: float = 0.95,
    repetitions: int = 2000,
    block_length: int | None = None,
    seed: int = 1729,
) -> tuple[float, float]:
    """Return a reproducible moving-block bootstrap interval for a paired mean."""
    values = np.asarray(values, dtype=float)
    if len(values) < 3:
        raise ValueError("Bootstrap confidence interval requires at least three values.")
    if not 0.0 < confidence < 1.0 or repetitions < 100:
        raise ValueError("confidence must be in (0, 1) and repetitions must be >= 100.")
    length = block_length or max(1, int(np.ceil(len(values) ** (1.0 / 3.0))))
    length = min(int(length), len(values))
    blocks = int(np.ceil(len(values) / length))
    offsets = np.arange(length)
    rng = np.random.default_rng(seed)
    means = np.empty(repetitions, dtype=float)
    for repetition in range(repetitions):
        starts = rng.integers(0, len(values), size=blocks)
        indices = ((starts[:, None] + offsets[None, :]) % len(values)).ravel()[: len(values)]
        means[repetition] = float(values[indices].mean())
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(means, [alpha, 1.0 - alpha])
    return float(lower), float(upper)


def diebold_mariano(
    actual: np.ndarray,
    first_prediction: np.ndarray,
    second_prediction: np.ndarray,
    *,
    loss: str = "squared",
    horizon: int = 1,
) -> dict[str, float | int | str]:
    """Two-sided Diebold–Mariano test with a Newey–West variance estimate."""
    actual = np.asarray(actual, dtype=float)
    first = np.asarray(first_prediction, dtype=float)
    second = np.asarray(second_prediction, dtype=float)
    if not (len(actual) == len(first) == len(second)) or len(actual) < 3:
        raise ValueError("DM test requires aligned arrays with at least three observations.")
    if horizon < 1:
        raise ValueError("horizon must be >= 1.")
    differential = _loss_differential(actual, first, second, loss)

    nobs = len(differential)
    centered = differential - differential.mean()
    long_run_variance = float(np.dot(centered, centered) / nobs)
    for lag in range(1, min(horizon - 1, nobs - 1) + 1):
        covariance = float(np.dot(centered[lag:], centered[:-lag]) / nobs)
        long_run_variance += 2.0 * (1.0 - lag / horizon) * covariance
    if long_run_variance <= np.finfo(float).eps:
        statistic = 0.0 if abs(float(differential.mean())) <= 1e-12 else float(np.sign(differential.mean()) * np.inf)
        pvalue = 1.0 if statistic == 0.0 else 0.0
    else:
        statistic = float(differential.mean() / np.sqrt(long_run_variance / nobs))
        pvalue = float(2.0 * student_t.sf(abs(statistic), df=nobs - 1))
    return {
        "loss": loss,
        "horizon": horizon,
        "nobs": nobs,
        "mean_loss_differential_first_minus_second": float(differential.mean()),
        "DM_statistic": statistic,
        "pvalue": pvalue,
    }


def holm_adjust(pvalues: np.ndarray) -> np.ndarray:
    values = np.asarray(pvalues, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running_max = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * values[index])
        running_max = max(running_max, candidate)
        adjusted[index] = running_max
    return adjusted


def pairwise_dm_tests(
    forecasts: pd.DataFrame,
    *,
    split: str = "test",
    loss: str = "squared",
    horizon: int = 1,
) -> pd.DataFrame:
    """Compare ensemble-mean predictions for every model pair on common dates."""
    frame = forecasts[forecasts["split"].eq(split)].copy()
    if frame.empty:
        return pd.DataFrame()
    frame["date"] = pd.to_datetime(frame["date"])
    ensemble = (
        frame.groupby(["pipeline", "date"], as_index=False)
        .agg(actual=("actual", "first"), predicted=("predicted", "mean"))
    )
    rows: list[dict[str, object]] = []
    for first_model, second_model in combinations(sorted(ensemble["pipeline"].unique()), 2):
        first = ensemble[ensemble["pipeline"].eq(first_model)][["date", "actual", "predicted"]]
        second = ensemble[ensemble["pipeline"].eq(second_model)][["date", "actual", "predicted"]]
        aligned = first.merge(second, on="date", suffixes=("_first", "_second"), validate="one_to_one")
        if not np.allclose(
            aligned["actual_first"].to_numpy(float),
            aligned["actual_second"].to_numpy(float),
            rtol=1e-9,
            atol=1e-9,
        ):
            raise ValueError(f"Actual values differ between {first_model} and {second_model}.")
        result = diebold_mariano(
            aligned["actual_first"].to_numpy(float),
            aligned["predicted_first"].to_numpy(float),
            aligned["predicted_second"].to_numpy(float),
            loss=loss,
            horizon=horizon,
        )
        differential = _loss_differential(
            aligned["actual_first"].to_numpy(float),
            aligned["predicted_first"].to_numpy(float),
            aligned["predicted_second"].to_numpy(float),
            loss,
        )
        ci_lower, ci_upper = moving_block_bootstrap_mean_ci(differential)
        rows.append(
            {
                "split": split,
                "first_model": first_model,
                "second_model": second_model,
                **result,
                "paired_mean_loss_difference_ci95_lower": ci_lower,
                "paired_mean_loss_difference_ci95_upper": ci_upper,
            }
        )
    output = pd.DataFrame(rows)
    if not output.empty:
        output["holm_adjusted_pvalue"] = holm_adjust(output["pvalue"].to_numpy(float))
    return output
