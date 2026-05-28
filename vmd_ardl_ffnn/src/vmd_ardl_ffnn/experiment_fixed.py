from __future__ import annotations

from dataclasses import replace
from itertools import product
import os
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

from .config import DataConfig, ExperimentConfig
from .data import GenericDataLoader
from .decomposition import VariationalModeDecomposer
from .diagnostics import ForecastDiagnostics
from .features import LagSpec
from .lag_selection import ARDLOrderSelector
from .metrics import evaluate_forecast
from .models.ffnn import SklearnFFNNRegressor


class VMDARDLFFNNExperiment:
    """Leakage-safe VMD/no-VMD ARDL-FFNN experiments with cached validation inputs."""

    def __init__(self, config: ExperimentConfig = ExperimentConfig()) -> None:
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.raw_model_data_: pd.DataFrame | None = None
        self.order_tables_: dict[str, pd.DataFrame] = {}
        self.selected_specs_: dict[str, list[LagSpec]] = {}
        self.search_results_: pd.DataFrame | None = None
        self.best_component_models_: pd.DataFrame | None = None
        self.component_forecasts_: pd.DataFrame | None = None
        self.final_forecasts_: pd.DataFrame | None = None
        self.final_metrics_: pd.DataFrame | None = None
        self.input_audit_: pd.DataFrame | None = None
        self.base_model_data_: pd.DataFrame | None = None
        self.stationarity_screen_: pd.DataFrame | None = None
        self.transform_info_: dict[str, str] = {}

    @property
    def target(self) -> str:
        return self.config.data.target

    @property
    def features(self) -> list[str]:
        return list(self.config.data.features)

    def _log(self, message: str) -> None:
        print(f"[vmd_ardl_ffnn] {message}", flush=True)

    def load_data(self, csv_path: str | Path) -> pd.DataFrame:
        self._log(f"Loading data: {csv_path}")
        loader = GenericDataLoader(self.config.data)
        self.raw_model_data_ = loader.load_csv(csv_path).sort_index()
        if not self.raw_model_data_.index.is_monotonic_increasing:
            raise ValueError(f"Data must be sorted by {self.config.data.date_col}.")
        return self.raw_model_data_

    def split_raw_data(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        n = len(data)
        train_end = int(n * self.config.train_ratio)
        val_end = int(n * (self.config.train_ratio + self.config.val_ratio))
        if not (0 < train_end < val_end < n):
            raise ValueError("Invalid chronological train/validation/test split.")
        train = data.iloc[:train_end].copy()
        val = data.iloc[train_end:val_end].copy()
        test = data.iloc[val_end:].copy()
        self._log(f"Split rows: train={len(train)}, validation={len(val)}, test={len(test)}")
        return train, val, test

    def _safe_adf_pvalue(self, series: pd.Series) -> float:
        series = series.dropna()
        if len(series) < self.config.stationarity.min_obs or series.nunique() <= 1:
            return float("nan")
        try:
            return float(adfuller(series, autolag="AIC")[1])
        except Exception:
            return float("nan")

    def _safe_kpss_pvalue(self, series: pd.Series) -> float:
        series = series.dropna()
        if len(series) < self.config.stationarity.min_obs or series.nunique() <= 1:
            return float("nan")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return float(kpss(series, regression="c", nlags="auto")[1])
        except Exception:
            return float("nan")

    def _is_stationary(self, adf_p: float, kpss_p: float) -> bool:
        alpha = float(self.config.stationarity.alpha)
        return pd.notna(adf_p) and pd.notna(kpss_p) and adf_p < alpha and kpss_p > alpha

    def _stationarity_label(self, adf_p: float, kpss_p: float) -> str:
        alpha = float(self.config.stationarity.alpha)
        if self._is_stationary(adf_p, kpss_p):
            return "Stationary"
        if pd.notna(adf_p) and pd.notna(kpss_p) and adf_p >= alpha and kpss_p <= alpha:
            return "Non-stationary"
        return "Mixed/unclear"

    def _apply_stationarity_transform(
        self,
        base_data: pd.DataFrame,
        base_train: pd.DataFrame,
    ) -> pd.DataFrame:
        self.base_model_data_ = base_data.copy()
        self.transform_info_ = {}
        if not self.config.stationarity.enabled:
            self.transform_info_ = {col: "level" for col in base_data.columns}
            self.stationarity_screen_ = pd.DataFrame(
                {
                    "Variable": list(base_data.columns),
                    "Transform used": ["level"] * len(base_data.columns),
                    "Decision basis": ["disabled"] * len(base_data.columns),
                }
            )
            return base_data.copy()

        self._log("Running train-only stationarity screen (ADF/KPSS).")
        transformed = base_data.copy()
        rows: list[dict[str, Any]] = []
        for col in base_data.columns:
            level_adf = self._safe_adf_pvalue(base_train[col])
            level_kpss = self._safe_kpss_pvalue(base_train[col])
            train_diff = base_train[col].diff()
            diff_adf = self._safe_adf_pvalue(train_diff)
            diff_kpss = self._safe_kpss_pvalue(train_diff)
            transform = "level" if self._is_stationary(level_adf, level_kpss) else "diff1"
            if transform == "diff1":
                transformed[col] = base_data[col].diff()
            self.transform_info_[col] = transform
            rows.append(
                {
                    "Variable": col,
                    "ADF level p": level_adf,
                    "KPSS level p": level_kpss,
                    "Level decision": self._stationarity_label(level_adf, level_kpss),
                    "ADF diff1 p": diff_adf,
                    "KPSS diff1 p": diff_kpss,
                    "Diff1 decision": self._stationarity_label(diff_adf, diff_kpss),
                    "Transform used": transform,
                    "Decision basis": "train_only",
                    "Likely I(2)?": "Yes" if not self._is_stationary(diff_adf, diff_kpss) else "No",
                }
            )
        self.stationarity_screen_ = pd.DataFrame(rows)
        return transformed.dropna().copy()

    def decompose(self, data: pd.DataFrame) -> dict[str, pd.DataFrame]:
        return VariationalModeDecomposer(self.config.vmd).decompose_frame(data)

    def _has_exog_lag_zero(self, spec: LagSpec) -> bool:
        return any(0 in tuple(lags) for lags in spec.exog_lags.values())

    def _safe_spec(self, spec: LagSpec) -> LagSpec:
        target_lags = tuple(sorted({int(lag) for lag in spec.target_lags if int(lag) >= 1}))
        exog_lags = {
            feature: tuple(sorted({int(lag) for lag in spec.exog_lags.get(feature, ()) if int(lag) >= 1}))
            for feature in self.features
        }
        safe = LagSpec(target_lags=target_lags, exog_lags=exog_lags)
        if not self._feature_names(safe):
            raise ValueError("Forecast-safe specs require at least one lagged feature.")
        return safe

    def _feature_names(self, spec: LagSpec) -> list[str]:
        names = [f"{self.target}__lag_{lag}" for lag in spec.target_lags]
        for feature in self.features:
            names.extend(f"{feature}__lag_{lag}" for lag in spec.exog_lags.get(feature, ()))
        return names

    def _build_supervised(self, data: pd.DataFrame, spec: LagSpec) -> pd.DataFrame:
        if any(lag < 1 for lag in spec.target_lags) or self._has_exog_lag_zero(spec):
            raise ValueError("Forecast design only allows target/exogenous lags >= 1.")
        out = pd.DataFrame(index=data.index)
        out[self.target] = data[self.target].astype(float)
        for lag in spec.target_lags:
            out[f"{self.target}__lag_{lag}"] = data[self.target].shift(lag)
        for feature in self.features:
            for lag in spec.exog_lags.get(feature, ()):
                out[f"{feature}__lag_{lag}"] = data[feature].shift(lag)
        return out.dropna()

    def _fit_model(
        self,
        data: pd.DataFrame,
        spec: LagSpec,
        hidden_units: int,
        alpha: float,
        seed: int,
    ) -> tuple[SklearnFFNNRegressor, int]:
        supervised = self._build_supervised(data, spec)
        if len(supervised) < self.config.ffnn.min_train:
            raise ValueError(f"Only {len(supervised)} training rows after lagging.")
        model = SklearnFFNNRegressor(
            hidden_layer_sizes=self.config.ffnn.architecture_for(hidden_units),
            alpha=alpha,
            seed=seed,
            activation=self.config.ffnn.activation,
            learning_rate_init=self.config.ffnn.learning_rate_init,
            max_iter=self.config.ffnn.max_iter,
        ).fit(supervised[self._feature_names(spec)].to_numpy(float), supervised[self.target].to_numpy(float))
        return model, len(supervised)

    def _batch_predict_observed(
        self,
        model: SklearnFFNNRegressor,
        observed: pd.DataFrame,
        eval_index: pd.Index,
        spec: LagSpec,
        split: str,
        component: str,
        predicted_col: str = "predicted",
    ) -> pd.DataFrame:
        design = self._build_supervised(observed, spec)
        design = design.loc[design.index.intersection(eval_index)].copy()
        pred = model.predict(design[self._feature_names(spec)].to_numpy(float))
        return pd.DataFrame(
            {
                "date": design.index,
                "split": split,
                "component": component,
                "actual": design[self.target].to_numpy(float),
                predicted_col: pred,
            }
        )

    def _next_x(self, component_history: pd.DataFrame, spec: LagSpec) -> np.ndarray:
        values: list[float] = []
        for lag in spec.target_lags:
            values.append(float(component_history[self.target].iloc[-lag]))
        for feature in self.features:
            for lag in spec.exog_lags.get(feature, ()):
                values.append(float(component_history[feature].iloc[-lag]))
        return np.asarray(values, dtype=float).reshape(1, -1)

    def _select_specs(self, frames: dict[str, pd.DataFrame]) -> None:
        selector = ARDLOrderSelector(self.config.ardl)
        self.order_tables_ = {}
        self.selected_specs_ = {}
        for name, frame in frames.items():
            self._log(f"Selecting ARDL lags on train only: {name}")
            table, raw_specs = selector.select(frame, self.target, self.features)
            safe_specs: list[LagSpec] = []
            seen: set[str] = set()
            for raw_spec in raw_specs:
                try:
                    safe = self._safe_spec(raw_spec)
                except ValueError:
                    continue
                if safe.label not in seen:
                    safe_specs.append(safe)
                    seen.add(safe.label)
            if not safe_specs:
                raise ValueError(f"No forecast-safe lag specs for {name}.")
            self.order_tables_[name] = table
            self.selected_specs_[name] = safe_specs

    def _audit_rows(
        self,
        pipeline: str,
        stage: str,
        split: str,
        data: pd.DataFrame,
        spec: LagSpec | None = None,
        feature_columns: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        has_nan = bool(data.isna().any().any())
        if "date" in data.columns:
            start_time = pd.to_datetime(data["date"]).min()
            end_time = pd.to_datetime(data["date"]).max()
        else:
            start_time = data.index.min() if len(data) else pd.NaT
            end_time = data.index.max() if len(data) else pd.NaT
        return [
            {
                "pipeline": pipeline,
                "stage": stage,
                "split": split,
                "n_rows": int(len(data)),
                "start_time": start_time,
                "end_time": end_time,
                "feature_columns": feature_columns or list(data.columns),
                "target_lags": tuple(spec.target_lags) if spec else tuple(),
                "exogenous_lags": spec.exog_lags if spec else {},
                "has_lag_0": self._has_exog_lag_zero(spec) if spec else False,
                "has_nan": has_nan,
                "stationarity_transforms": self.transform_info_,
            }
        ]

    def _metrics_rows(
        self,
        forecasts: pd.DataFrame,
        actual_col: str,
        predicted_col: str,
        transformed_scale: str,
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for split, group in forecasts.groupby("split", sort=False):
            row: dict[str, Any] = {"split": split, "scale": transformed_scale}
            row.update(evaluate_forecast(group[actual_col], group[predicted_col]))
            rows.append(row)
            if self.config.data.log_transform or self.transform_info_.get(self.target) == "diff1":
                actual_level, predicted_level = self._to_level(group, actual_col, predicted_col)
                forecasts.loc[group.index, "actual_level"] = actual_level
                forecasts.loc[group.index, "predicted_level"] = predicted_level
                level_row: dict[str, Any] = {"split": split, "scale": "level"}
                level_row.update(evaluate_forecast(actual_level, predicted_level))
                rows.append(level_row)
        return pd.DataFrame(rows)

    def _to_level(
        self,
        frame: pd.DataFrame,
        actual_col: str,
        predicted_col: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.base_model_data_ is None:
            raise RuntimeError("base_model_data_ is required to invert stationarity transforms.")
        dates = pd.Index(frame["date"])
        target_transform = self.transform_info_.get(self.target, "level")
        if target_transform == "diff1":
            previous_base = self.base_model_data_[self.target].shift(1).reindex(dates).to_numpy(float)
            actual_base = previous_base + frame[actual_col].to_numpy(float)
            predicted_base = previous_base + frame[predicted_col].to_numpy(float)
        else:
            actual_base = frame[actual_col].to_numpy(float)
            predicted_base = frame[predicted_col].to_numpy(float)

        if self.config.data.log_transform:
            return np.exp(actual_base), np.exp(predicted_base)
        return actual_base, predicted_base

    def _plot_actual_predicted(
        self,
        forecasts: pd.DataFrame,
        split: str,
        actual_col: str,
        predicted_col: str,
        filename: str,
    ) -> None:
        try:
            os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
            import matplotlib.pyplot as plt
        except ImportError:
            self._log(f"Skipping plot {filename}: matplotlib is not installed.")
            return
        group = forecasts[forecasts["split"].eq(split)].copy()
        if group.empty:
            return
        if self.config.data.log_transform and {"actual_level", "predicted_level"}.issubset(group.columns):
            actual_col, predicted_col = "actual_level", "predicted_level"
        group = group.sort_values("date")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(group["date"], group[actual_col], label="Actual", linewidth=2)
        ax.plot(group["date"], group[predicted_col], label="Predicted", linewidth=2)
        ax.set_title(f"Actual vs Predicted - {split}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Level" if actual_col.endswith("_level") else "Transformed")
        ax.legend()
        ax.grid(True, alpha=0.25)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(self.output_dir / filename, dpi=160)
        plt.close(fig)

    def _diagnostics(self, train: pd.DataFrame, frames: dict[str, pd.DataFrame], suffix: str) -> dict[str, pd.DataFrame]:
        diagnostics = ForecastDiagnostics(self.config.diagnostics)
        series = [diagnostics.describe_series(train, f"train_raw{suffix}")]
        series.extend(diagnostics.describe_series(frame, f"train_{name}{suffix}") for name, frame in frames.items())
        series_df = pd.concat(series, ignore_index=True)
        granger_df = diagnostics.granger_causality(train, self.target, self.features)
        residual_rows = []
        for name, specs in self.selected_specs_.items():
            frame = frames.get(name, train)
            for rank, spec in enumerate(specs, start=1):
                residual_rows.append(
                    diagnostics.ardl_residual_diagnostics(
                        frame, self.target, self.features, spec, name, rank, trend=self.config.ardl.trend
                    )
                )
        residual_df = pd.DataFrame(residual_rows)
        return {
            "series_diagnostics": series_df,
            "granger_causality": granger_df,
            "ardl_residual_diagnostics": residual_df,
        }

    def run_without_vmd(self, csv_path: str | Path) -> dict[str, pd.DataFrame]:
        self._log("Starting fixed no-VMD pipeline.")
        base_data = self.load_data(csv_path)
        base_train, base_val, base_test = self.split_raw_data(base_data)
        data = self._apply_stationarity_transform(base_data, base_train)
        train = data.loc[data.index.intersection(base_train.index)].copy()
        val = data.loc[data.index.intersection(base_val.index)].copy()
        test = data.loc[data.index.intersection(base_test.index)].copy()
        self._log(
            "Stationarity-adjusted rows: "
            f"train={len(train)}, validation={len(val)}, test={len(test)}"
        )
        component = "raw_no_vmd"
        self._select_specs({component: train})
        audit = []
        for split, frame in [("train", train), ("validation", val), ("test", test)]:
            audit.extend(self._audit_rows("no_vmd", "chronological_split", split, frame))

        rows: list[dict[str, Any]] = []
        val_frames: dict[int, pd.DataFrame] = {}
        row_id = 0
        for spec in self.selected_specs_[component]:
            for hidden_units in self.config.ffnn.hidden_units_candidates:
                for alpha in self.config.ffnn.alpha_grid:
                    for seed in self.config.ffnn.seed_grid:
                        try:
                            model, n_train = self._fit_model(train, spec, int(hidden_units), float(alpha), int(seed))
                            val_pred = self._batch_predict_observed(
                                model, pd.concat([train, val]), val.index, spec, "validation", component
                            )
                        except ValueError:
                            continue
                        if len(val_pred) < self.config.ffnn.min_val:
                            continue
                        row = {
                            "candidate_id": row_id,
                            "component": component,
                            "lag_spec": spec.label,
                            "target_lags": spec.target_lags,
                            "exog_lags": spec.exog_lags,
                            "HR": int(hidden_units),
                            "hidden_layer_sizes": self.config.ffnn.architecture_for(int(hidden_units)),
                            "alpha": float(alpha),
                            "seed": int(seed),
                            "n_train": n_train,
                            "n_val": len(val_pred),
                            "forecast_safe_no_exog_lag0": True,
                            "uses_vmd": False,
                        }
                        row.update({f"Val {k}": v for k, v in evaluate_forecast(val_pred["actual"], val_pred["predicted"]).items()})
                        rows.append(row)
                        val_frames[row_id] = val_pred
                        audit.extend(self._audit_rows("no_vmd", "validation_design", "validation", val_pred, spec, self._feature_names(spec)))
                        row_id += 1
        if not rows:
            raise ValueError("No no-VMD candidates could be trained.")
        self.search_results_ = pd.DataFrame(rows).sort_values(["Val RMSE", "Val MAE", "Val MAPE"]).reset_index(drop=True)
        best = self.search_results_.iloc[0]
        best_spec = LagSpec(tuple(best["target_lags"]), {k: tuple(v) for k, v in best["exog_lags"].items()})
        self._log("Refitting locked no-VMD winner on train+validation.")
        best_model, n_refit = self._fit_model(pd.concat([train, val]), best_spec, int(best["HR"]), float(best["alpha"]), int(best["seed"]))
        val_forecasts = val_frames[int(best["candidate_id"])].rename(columns={"actual": "actual_raw_transformed", "predicted": "predicted_raw_transformed"})
        test_forecasts = self._batch_predict_observed(
            best_model, pd.concat([train, val, test]), test.index, best_spec, "test", component
        ).rename(columns={"actual": "actual_raw_transformed", "predicted": "predicted_raw_transformed"})
        final = pd.concat([val_forecasts, test_forecasts], ignore_index=True)
        self.final_metrics_ = self._metrics_rows(final, "actual_raw_transformed", "predicted_raw_transformed", "raw_transformed")
        self.final_forecasts_ = final.sort_values(["split", "date"]).reset_index(drop=True)
        best_model_df = pd.DataFrame([{**best.to_dict(), "n_refit_train_val": n_refit}])
        self.best_component_models_ = best_model_df
        audit.extend(self._audit_rows("no_vmd", "test_design", "test", test_forecasts, best_spec, self._feature_names(best_spec)))
        self.input_audit_ = pd.DataFrame(audit)

        diagnostics = self._diagnostics(train, {component: train}, "_no_vmd")
        ardl_orders = pd.concat([table.assign(component=name) for name, table in self.order_tables_.items()], ignore_index=True)
        self.search_results_.to_csv(self.output_dir / "ffnn_validation_search_fixed_no_vmd.csv", index=False)
        best_model_df.to_csv(self.output_dir / "best_model_fixed_no_vmd.csv", index=False)
        self.final_forecasts_.to_csv(self.output_dir / "final_forecasts_fixed_no_vmd.csv", index=False)
        self.final_metrics_.to_csv(self.output_dir / "final_metrics_fixed_no_vmd.csv", index=False)
        self.input_audit_.to_csv(self.output_dir / "input_audit_no_vmd.csv", index=False)
        if self.stationarity_screen_ is not None:
            self.stationarity_screen_.to_csv(
                self.output_dir / "stationarity_screen_train_only_no_vmd.csv",
                index=False,
            )
        ardl_orders.to_csv(self.output_dir / "ardl_selected_orders_train_only_no_vmd.csv", index=False)
        for name, df in diagnostics.items():
            df.to_csv(self.output_dir / f"{name}_train_only_no_vmd.csv", index=False)
        self._plot_actual_predicted(self.final_forecasts_, "validation", "actual_raw_transformed", "predicted_raw_transformed", "actual_vs_predicted_val_no_vmd.png")
        self._plot_actual_predicted(self.final_forecasts_, "test", "actual_raw_transformed", "predicted_raw_transformed", "actual_vs_predicted_test_no_vmd.png")
        return {
            "search_results": self.search_results_,
            "ardl_orders": ardl_orders,
            "best_component_models": best_model_df,
            "final_forecasts": self.final_forecasts_,
            "final_metrics": self.final_metrics_,
            "input_audit": self.input_audit_,
            "stationarity_screen": self.stationarity_screen_,
            **diagnostics,
        }

    def _make_origin_cache(
        self,
        initial_history: pd.DataFrame,
        evaluation: pd.DataFrame,
        split: str,
    ) -> dict[pd.Timestamp, dict[str, pd.DataFrame]]:
        self._log(f"Creating {split} VMD origin cache ({len(evaluation)} timestamps).")
        history = initial_history.copy()
        cache: dict[pd.Timestamp, dict[str, pd.DataFrame]] = {}
        for date, row in evaluation.iterrows():
            cache[pd.Timestamp(date)] = self.decompose(history)
            history = pd.concat([history, row.to_frame().T], axis=0)
        return cache

    def _predict_component_from_cache(
        self,
        model: SklearnFFNNRegressor,
        component: str,
        cache: dict[pd.Timestamp, dict[str, pd.DataFrame]],
        actual_raw: pd.DataFrame,
        spec: LagSpec,
        split: str,
        candidate_id: int | None = None,
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for date in actual_raw.index:
            component_history = cache[pd.Timestamp(date)][component]
            predicted = float(model.predict(self._next_x(component_history, spec))[0])
            row = {
                "date": date,
                "split": split,
                "component": component,
                "predicted_component": predicted,
            }
            if candidate_id is not None:
                row["candidate_id"] = candidate_id
            rows.append(row)
        return pd.DataFrame(rows)

    def _reconstruct(self, component_predictions: pd.DataFrame, actual_raw: pd.DataFrame) -> pd.DataFrame:
        out = (
            component_predictions.groupby(["split", "date"], as_index=False)
            .agg(predicted_reconstructed=("predicted_component", "sum"), component_count=("component", "nunique"))
            .sort_values(["split", "date"])
        )
        out["actual_raw_transformed"] = out["date"].map(actual_raw[self.target]).astype(float)
        return out

    def run(self, csv_path: str | Path) -> dict[str, pd.DataFrame]:
        self._log("Starting cached VMD pipeline.")
        base_data = self.load_data(csv_path)
        base_train, base_val, base_test = self.split_raw_data(base_data)
        data = self._apply_stationarity_transform(base_data, base_train)
        train = data.loc[data.index.intersection(base_train.index)].copy()
        val = data.loc[data.index.intersection(base_val.index)].copy()
        test = data.loc[data.index.intersection(base_test.index)].copy()
        self._log(
            "Stationarity-adjusted rows: "
            f"train={len(train)}, validation={len(val)}, test={len(test)}"
        )
        audit = []
        for split, frame in [("train", train), ("validation", val), ("test", test)]:
            audit.extend(self._audit_rows("cached_vmd", "chronological_split", split, frame))

        self._log("Running VMD on train only.")
        train_components = self.decompose(train)
        self._select_specs(train_components)
        val_cache = self._make_origin_cache(train, val, "validation")

        candidate_predictions: dict[str, list[pd.DataFrame]] = {}
        candidate_rows: list[dict[str, Any]] = []
        candidate_id = 0
        for component, train_component in train_components.items():
            candidate_predictions[component] = []
            for spec in self.selected_specs_[component]:
                for hidden_units in self.config.ffnn.hidden_units_candidates:
                    for alpha in self.config.ffnn.alpha_grid:
                        for seed in self.config.ffnn.seed_grid:
                            try:
                                model, n_train = self._fit_model(train_component, spec, int(hidden_units), float(alpha), int(seed))
                                pred = self._predict_component_from_cache(model, component, val_cache, val, spec, "validation", candidate_id)
                            except (ValueError, KeyError, IndexError):
                                continue
                            row = {
                                "candidate_id": candidate_id,
                                "component": component,
                                "lag_spec": spec.label,
                                "target_lags": spec.target_lags,
                                "exog_lags": spec.exog_lags,
                                "HR": int(hidden_units),
                                "hidden_layer_sizes": self.config.ffnn.architecture_for(int(hidden_units)),
                                "alpha": float(alpha),
                                "seed": int(seed),
                                "n_train": n_train,
                                "n_val": len(pred),
                                "forecast_safe_no_exog_lag0": True,
                            }
                            candidate_rows.append(row)
                            candidate_predictions[component].append(pred)
                            audit.extend(self._audit_rows("cached_vmd", "validation_cache", "validation", pred, spec, self._feature_names(spec)))
                            candidate_id += 1
            if not candidate_predictions[component]:
                raise ValueError(f"No cached VMD validation candidates for {component}.")

        component_ranked: dict[str, pd.DataFrame] = {}
        for component, frames in candidate_predictions.items():
            scored = []
            for frame in frames:
                reconstructed = self._reconstruct(frame, val)
                metrics = evaluate_forecast(reconstructed["actual_raw_transformed"], reconstructed["predicted_reconstructed"])
                scored.append({"candidate_id": int(frame["candidate_id"].iloc[0]), **{f"Component Val {k}": v for k, v in metrics.items()}})
            component_ranked[component] = pd.DataFrame(scored).sort_values(["Component Val RMSE", "Component Val MAE", "Component Val MAPE"])

        top_by_component = {
            component: ranked.head(min(5, len(ranked)))["candidate_id"].astype(int).tolist()
            for component, ranked in component_ranked.items()
        }
        pred_by_id = {
            int(frame["candidate_id"].iloc[0]): frame
            for frames in candidate_predictions.values()
            for frame in frames
        }
        combo_rows: list[dict[str, Any]] = []
        component_order = list(top_by_component)
        for combo_id, ids in enumerate(product(*top_by_component.values())):
            component_frames = [pred_by_id[int(candidate)] for candidate in ids]
            reconstructed = self._reconstruct(pd.concat(component_frames, ignore_index=True), val)
            metric = evaluate_forecast(reconstructed["actual_raw_transformed"], reconstructed["predicted_reconstructed"])
            row = {"combo_id": combo_id, "component_candidate_ids": tuple(int(x) for x in ids)}
            row.update({f"{component}_candidate_id": int(candidate) for component, candidate in zip(component_order, ids)})
            row.update({f"Val {k}": v for k, v in metric.items()})
            combo_rows.append(row)
        combo_df = pd.DataFrame(combo_rows).sort_values(["Val RMSE", "Val MAE", "Val MAPE"]).reset_index(drop=True)

        candidates_df = pd.DataFrame(candidate_rows)
        self.search_results_ = combo_df
        best_ids = tuple(int(x) for x in combo_df.iloc[0]["component_candidate_ids"])
        best_candidates = candidates_df[candidates_df["candidate_id"].isin(best_ids)].copy()
        best_candidates["combo_id"] = int(combo_df.iloc[0]["combo_id"])

        self._log("Refitting cached VMD winners on train+validation.")
        history_before_test = pd.concat([train, val], axis=0)
        history_components = self.decompose(history_before_test)
        test_cache = self._make_origin_cache(history_before_test, test, "test")
        selected_component_predictions: list[pd.DataFrame] = []
        val_selected = [pred_by_id[candidate_id] for candidate_id in best_ids]
        selected_component_predictions.extend(val_selected)
        for _, best in best_candidates.iterrows():
            component = str(best["component"])
            spec = LagSpec(tuple(best["target_lags"]), {k: tuple(v) for k, v in best["exog_lags"].items()})
            model, n_refit = self._fit_model(history_components[component], spec, int(best["HR"]), float(best["alpha"]), int(best["seed"]))
            test_pred = self._predict_component_from_cache(model, component, test_cache, test, spec, "test", int(best["candidate_id"]))
            selected_component_predictions.append(test_pred)
            best_candidates.loc[best_candidates["candidate_id"].eq(best["candidate_id"]), "n_refit_train_val"] = n_refit
            audit.extend(self._audit_rows("cached_vmd", "test_cache", "test", test_pred, spec, self._feature_names(spec)))

        self.component_forecasts_ = pd.concat(selected_component_predictions, ignore_index=True)
        actual_all = pd.concat([val, test], axis=0)
        self.final_forecasts_ = self._reconstruct(self.component_forecasts_, actual_all).reset_index(drop=True)
        self.final_metrics_ = self._metrics_rows(
            self.final_forecasts_,
            "actual_raw_transformed",
            "predicted_reconstructed",
            "reconstructed_transformed",
        )
        self.best_component_models_ = best_candidates.sort_values("component").reset_index(drop=True)
        self.input_audit_ = pd.DataFrame(audit)
        diagnostics = self._diagnostics(train, train_components, "_cached_vmd")
        ardl_orders = pd.concat([table.assign(component=name) for name, table in self.order_tables_.items()], ignore_index=True)

        combo_df.to_csv(self.output_dir / "ffnn_validation_search_cached_vmd.csv", index=False)
        self.best_component_models_.to_csv(self.output_dir / "best_component_models_cached_vmd.csv", index=False)
        self.component_forecasts_.to_csv(self.output_dir / "component_predictions_cached_vmd.csv", index=False)
        self.final_forecasts_.to_csv(self.output_dir / "final_reconstructed_forecasts_cached_vmd.csv", index=False)
        self.final_metrics_.to_csv(self.output_dir / "final_reconstructed_metrics_cached_vmd.csv", index=False)
        self.input_audit_.to_csv(self.output_dir / "input_audit_cached_vmd.csv", index=False)
        if self.stationarity_screen_ is not None:
            self.stationarity_screen_.to_csv(
                self.output_dir / "stationarity_screen_train_only_cached_vmd.csv",
                index=False,
            )
        ardl_orders.to_csv(self.output_dir / "ardl_selected_orders_train_only.csv", index=False)
        for name, df in diagnostics.items():
            df.to_csv(self.output_dir / f"{name}_train_only.csv", index=False)
        self._plot_actual_predicted(self.final_forecasts_, "validation", "actual_raw_transformed", "predicted_reconstructed", "actual_vs_predicted_val_cached_vmd.png")
        self._plot_actual_predicted(self.final_forecasts_, "test", "actual_raw_transformed", "predicted_reconstructed", "actual_vs_predicted_test_cached_vmd.png")
        return {
            "search_results": combo_df,
            "ardl_orders": ardl_orders,
            "best_component_models": self.best_component_models_,
            "component_forecasts": self.component_forecasts_,
            "final_forecasts": self.final_forecasts_,
            "final_metrics": self.final_metrics_,
            "input_audit": self.input_audit_,
            "stationarity_screen": self.stationarity_screen_,
            **diagnostics,
        }


def config_for_columns(
    target: str,
    features: tuple[str, ...],
    date_col: str = "TIME_PERIOD",
    output_dir: str | Path = "results",
) -> ExperimentConfig:
    return replace(
        ExperimentConfig(),
        data=DataConfig(date_col=date_col, target=target, features=features),
        output_dir=Path(output_dir),
    )
