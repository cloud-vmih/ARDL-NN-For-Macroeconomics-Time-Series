from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _literal(value: Any, fallback: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return fallback


def lag_selection_frequency(lag_selection: pd.DataFrame) -> pd.DataFrame:
    """Convert final component lag structures into across-seed frequencies."""
    rows: list[dict[str, Any]] = []
    for _, selected in lag_selection.iterrows():
        common = {
            "pipeline": selected["pipeline"],
            "seed": selected["seed"],
            "component": selected["component"],
        }
        target_lags = _literal(selected.get("target_lags", tuple()), tuple())
        for lag in tuple(target_lags):
            rows.append({**common, "lag_type": "target", "variable": "target", "lag": int(lag)})
        exogenous_lags = _literal(selected.get("exog_lags", {}), {})
        for variable, lags in dict(exogenous_lags).items():
            for lag in tuple(lags):
                rows.append(
                    {**common, "lag_type": "exogenous", "variable": str(variable), "lag": int(lag)}
                )
    raw = pd.DataFrame(rows)
    if raw.empty:
        return raw
    denominators = (
        lag_selection.groupby(["pipeline", "component"], dropna=False)["seed"]
        .nunique()
        .rename("reporting_seed_count")
        .reset_index()
    )
    frequency = (
        raw.groupby(
            ["pipeline", "component", "lag_type", "variable", "lag"],
            as_index=False,
            dropna=False,
        )["seed"]
        .nunique()
        .rename(columns={"seed": "selected_seed_count"})
        .merge(denominators, on=["pipeline", "component"], how="left", validate="many_to_one")
    )
    frequency["selection_frequency"] = (
        frequency["selected_seed_count"] / frequency["reporting_seed_count"]
    )
    return frequency.sort_values(
        ["pipeline", "component", "lag_type", "variable", "lag"]
    ).reset_index(drop=True)


def _matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    return plt


def _save_forecast_plot(plt, forecasts: pd.DataFrame, output_dir: Path) -> None:
    test = forecasts[forecasts["split"].eq("test")].copy()
    if test.empty:
        return
    test["date"] = pd.to_datetime(test["date"])
    ensemble = (
        test.groupby(["pipeline", "date"], as_index=False)
        .agg(actual=("actual", "first"), predicted=("predicted", "mean"))
    )
    actual = ensemble[["date", "actual"]].drop_duplicates("date").sort_values("date")
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.plot(actual["date"], actual["actual"], color="black", linewidth=2.2, label="Actual")
    for pipeline, group in ensemble.groupby("pipeline", sort=True):
        group = group.sort_values("date")
        ax.plot(group["date"], group["predicted"], linewidth=1.5, label=pipeline)
    ax.set(title="Rolling one-step test forecasts", xlabel="Time", ylabel="Target level")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_dir / "paper_test_forecasts.png", dpi=180)
    plt.close(fig)


def _save_error_distribution(plt, forecasts: pd.DataFrame, output_dir: Path) -> None:
    test = forecasts[forecasts["split"].eq("test")].copy()
    if test.empty:
        return
    test["absolute_error"] = (test["actual"] - test["predicted"]).abs()
    pipelines = sorted(test["pipeline"].unique())
    values = [test.loc[test["pipeline"].eq(pipeline), "absolute_error"].to_numpy() for pipeline in pipelines]
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.boxplot(values, tick_labels=pipelines, showfliers=False)
    ax.set(title="Test absolute-error distribution", ylabel="Absolute error")
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "paper_test_error_distribution.png", dpi=180)
    plt.close(fig)


def _save_uncertainty_plot(plt, metrics: pd.DataFrame, output_dir: Path) -> None:
    test = metrics[metrics["split"].eq("test")].sort_values("RMSE_mean")
    if test.empty or "RMSE_mean" not in test:
        return
    fig, axes = plt.subplots(2, 1, figsize=(12, 11))
    for axis, metric in zip(axes, ("RMSE", "MAE"), strict=True):
        mean_column = f"{metric}_mean"
        if mean_column not in test:
            axis.set_visible(False)
            continue
        error = test.get(
            f"{metric}_std", pd.Series(np.zeros(len(test)), index=test.index)
        ).fillna(0.0)
        axis.bar(test["pipeline"], test[mean_column], yerr=error, capsize=4)
        axis.set(
            title=f"Test {metric} across reporting seeds",
            ylabel=f"{metric} mean ± standard deviation",
        )
        axis.tick_params(axis="x", rotation=35, labelsize=8)
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "paper_rmse_mae_seed_uncertainty.png", dpi=180)
    plt.close(fig)


def _save_accuracy_runtime_plot(
    plt,
    metrics: pd.DataFrame,
    efficiency: pd.DataFrame,
    output_dir: Path,
) -> None:
    test = metrics[metrics["split"].eq("test")][["pipeline", "RMSE_mean"]]
    runtime = efficiency.groupby("pipeline", as_index=False)["runtime_seconds"].mean()
    merged = test.merge(runtime, on="pipeline", how="inner")
    if merged.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(merged["runtime_seconds"], merged["RMSE_mean"], s=55)
    for _, row in merged.iterrows():
        ax.annotate(row["pipeline"], (row["runtime_seconds"], row["RMSE_mean"]), fontsize=8)
    ax.set(title="Accuracy–runtime trade-off", xlabel="Mean runtime (seconds)", ylabel="Test RMSE")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "paper_accuracy_runtime.png", dpi=180)
    plt.close(fig)


def _save_lag_heatmap(
    plt,
    frequency: pd.DataFrame,
    output_dir: Path,
    lag_type: str,
    proposed_pipeline: str,
) -> None:
    selected = frequency[
        frequency["pipeline"].eq(proposed_pipeline)
        & frequency["lag_type"].eq(lag_type)
    ].copy()
    if selected.empty:
        return
    selected["row"] = selected["component"].astype(str)
    if lag_type == "exogenous":
        selected["row"] += " / " + selected["variable"].astype(str)
    matrix = selected.pivot_table(
        index="row", columns="lag", values="selection_frequency", aggfunc="max", fill_value=0.0
    ).sort_index()
    fig_height = max(4.0, 0.35 * len(matrix) + 2.0)
    fig, ax = plt.subplots(figsize=(11, fig_height))
    image = ax.imshow(matrix.to_numpy(float), aspect="auto", cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(matrix.columns)), labels=[str(value) for value in matrix.columns])
    ax.set_yticks(np.arange(len(matrix.index)), labels=list(matrix.index), fontsize=8)
    ax.set(xlabel="Lag", title=f"Proposed-model {lag_type} lag selection frequency")
    fig.colorbar(image, ax=ax, label="Frequency across seeds")
    fig.tight_layout()
    fig.savefig(output_dir / f"paper_{lag_type}_lag_heatmap.png", dpi=180)
    plt.close(fig)


def _save_vmd_decomposition(
    plt,
    training_components: dict[str, pd.DataFrame] | None,
    target: str | None,
    output_dir: Path,
) -> None:
    if not training_components or not target:
        return
    component_names = list(training_components)
    fig, axes = plt.subplots(
        len(component_names),
        1,
        figsize=(12, max(5.0, 2.2 * len(component_names))),
        sharex=True,
    )
    axes = np.atleast_1d(axes)
    for axis, component in zip(axes, component_names, strict=True):
        frame = training_components[component]
        axis.plot(frame.index, frame[target], linewidth=1.3)
        axis.set_ylabel(component)
        axis.grid(alpha=0.2)
    axes[0].set_title("Train-only VMD decomposition of the target")
    axes[-1].set_xlabel("Time")
    fig.tight_layout()
    fig.savefig(output_dir / "paper_train_vmd_decomposition.png", dpi=180)
    plt.close(fig)


def write_paper_figures(
    benchmark_forecasts: pd.DataFrame,
    benchmark_metrics: pd.DataFrame,
    efficiency: pd.DataFrame,
    lag_frequency: pd.DataFrame,
    output_dir: str | Path,
    *,
    training_components: dict[str, pd.DataFrame] | None = None,
    target: str | None = None,
    proposed_pipeline: str = "vmd_ardl_ffnn_proposed",
) -> None:
    """Write the core paper figures when matplotlib is available."""
    plt = _matplotlib()
    if plt is None:
        return
    root = Path(output_dir)
    _save_forecast_plot(plt, benchmark_forecasts, root)
    _save_error_distribution(plt, benchmark_forecasts, root)
    _save_uncertainty_plot(plt, benchmark_metrics, root)
    _save_accuracy_runtime_plot(plt, benchmark_metrics, efficiency, root)
    _save_lag_heatmap(
        plt,
        lag_frequency,
        root,
        "target",
        proposed_pipeline,
    )
    _save_lag_heatmap(
        plt,
        lag_frequency,
        root,
        "exogenous",
        proposed_pipeline,
    )
    _save_vmd_decomposition(plt, training_components, target, root)
