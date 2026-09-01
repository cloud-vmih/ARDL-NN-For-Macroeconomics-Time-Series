"""Per-zone ARDL baseline plus a pooled nonlinear FFNN residual corrector."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .flood_ardl import SequentialARDL
from .flood_config import FloodARDLConfig, FloodDataConfig, FloodFFNNConfig
from .models.ffnn import TensorFlowFFNNRegressor


@dataclass
class StaticFeatureEncoder:
    zone_col: str
    numeric_cols: tuple[str, ...]
    categorical_cols: tuple[str, ...]
    categories_: dict[str, tuple[str, ...]] = field(default_factory=dict, init=False)
    rows_: dict[str, dict[str, float]] = field(default_factory=dict, init=False)
    feature_names_: tuple[str, ...] = field(default_factory=tuple, init=False)

    def fit(self, static: pd.DataFrame) -> "StaticFeatureEncoder":
        feature_names = [f"static__{column}" for column in self.numeric_cols]
        for column in self.categorical_cols:
            categories = tuple(sorted(static[column].astype("string").fillna("__missing__").unique()))
            self.categories_[column] = categories
            feature_names.extend(f"static__{column}={category}" for category in categories)
        self.feature_names_ = tuple(feature_names)
        self.rows_ = {}
        for _, row in static.iterrows():
            zone = str(row[self.zone_col])
            encoded: dict[str, float] = {}
            for column in self.numeric_cols:
                encoded[f"static__{column}"] = float(row[column])
            for column, categories in self.categories_.items():
                value = str(row[column]) if pd.notna(row[column]) else "__missing__"
                for category in categories:
                    encoded[f"static__{column}={category}"] = float(value == category)
            self.rows_[zone] = encoded
        return self

    def transform_zone(self, zone: str) -> dict[str, float]:
        if zone not in self.rows_:
            raise ValueError(f"No static features available for zone {zone!r}.")
        return self.rows_[zone].copy()


class ARDLFFNNResidualForecaster:
    """Genuine hybrid: ARDL forecasts the linear response, FFNN forecasts its residual."""

    def __init__(
        self,
        data_config: FloodDataConfig,
        ardl_config: FloodARDLConfig | None = None,
        ffnn_config: FloodFFNNConfig | None = None,
    ) -> None:
        self.data_config = data_config
        self.ardl_config = ardl_config or FloodARDLConfig()
        self.ffnn_config = ffnn_config or FloodFFNNConfig()
        if len(self.data_config.exogenous_cols) != len(self.data_config.nwp_cols):
            raise ValueError(
                "Observed and forecast exogenous columns must have equal length for explicit mapping."
            )
        self.nwp_to_observed = dict(zip(self.data_config.nwp_cols, self.data_config.exogenous_cols))
        self.models_: dict[str, SequentialARDL] = {}
        self.static_encoder_ = StaticFeatureEncoder(
            self.data_config.zone_col,
            self.data_config.static_numeric_cols,
            self.data_config.static_categorical_cols,
        )
        self.ffnn_: TensorFlowFFNNRegressor | None = None
        self.fixed_lag_ffnn_: TensorFlowFFNNRegressor | None = None
        self.ardl_guided_ffnn_: TensorFlowFFNNRegressor | None = None
        self.residual_feature_names_: tuple[str, ...] = tuple()
        self.fixed_lag_feature_names_: tuple[str, ...] = tuple()
        self.ardl_guided_feature_names_: tuple[str, ...] = tuple()
        self.residual_training_summary_: dict[str, object] = {}
        self.train_end_: pd.Timestamp | None = None

    def fit(
        self,
        observations: pd.DataFrame,
        static: pd.DataFrame,
        forecasts: pd.DataFrame,
        *,
        train_end: str | pd.Timestamp,
    ) -> "ARDLFFNNResidualForecaster":
        train_end = pd.Timestamp(train_end)
        if train_end.tzinfo is None:
            train_end = train_end.tz_localize("UTC")
        else:
            train_end = train_end.tz_convert("UTC")
        self.train_end_ = train_end
        self.static_encoder_.fit(static)

        zone_col = self.data_config.zone_col
        time_col = self.data_config.time_col
        target = self.data_config.target_col
        train = observations.loc[observations[time_col] <= train_end].copy()
        self.models_ = {}
        failures: list[str] = []
        for zone, group in train.groupby(zone_col, sort=True, observed=True):
            zone = str(zone)
            series = group.set_index(time_col)[[target, *self.data_config.exogenous_cols]].sort_index()
            try:
                model = SequentialARDL(
                    target=target,
                    exogenous=self.data_config.exogenous_cols,
                    config=self.ardl_config,
                ).fit(series)
            except ValueError as exc:
                failures.append(f"{zone}: {exc}")
                continue
            self.models_[zone] = model
        if failures:
            preview = "; ".join(failures[:5])
            raise ValueError(f"Unable to fit ARDL for {len(failures)} zone(s): {preview}")
        if not self.models_:
            raise ValueError("No ARDL zone models were fitted.")

        forecast_train = forecasts.loc[
            (forecasts[self.data_config.issue_time_col] <= train_end)
            & (forecasts[self.data_config.valid_time_col] <= train_end)
            & (forecasts["lead_hour"] <= self.ffnn_config.max_horizon_hours)
        ].copy()
        residual_rows = self._build_residual_training_rows(train, forecast_train)
        if residual_rows.empty:
            raise ValueError(
                "No residual-training samples could be built. Check train dates, issue times and hourly horizons."
            )
        residual_rows = self._cap_training_rows(residual_rows)
        non_features = {
            zone_col,
            self.data_config.issue_time_col,
            self.data_config.valid_time_col,
            "actual_runoff",
            "residual_target",
        }
        feature_names = tuple(column for column in residual_rows.columns if column not in non_features)
        X = residual_rows.loc[:, feature_names].to_numpy(dtype=float)
        y = residual_rows["residual_target"].to_numpy(dtype=float)
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("Residual FFNN design contains non-finite values.")
        self.residual_feature_names_ = feature_names
        self.ffnn_ = TensorFlowFFNNRegressor(
            hidden_layer_sizes=self.ffnn_config.hidden_layer_sizes,
            alpha=self.ffnn_config.alpha,
            seed=self.ffnn_config.seed,
            activation=self.ffnn_config.activation,
            learning_rate_init=self.ffnn_config.learning_rate,
            max_iter=self.ffnn_config.max_iter,
        ).fit(X, y)
        self.fixed_lag_feature_names_ = tuple(
            feature for feature in feature_names if feature != "ardl_prediction"
        )
        self.ardl_guided_feature_names_ = feature_names
        direct_target = residual_rows["actual_runoff"].to_numpy(dtype=float)
        self.fixed_lag_ffnn_ = self._new_ffnn(seed_offset=101).fit(
            residual_rows.loc[:, self.fixed_lag_feature_names_].to_numpy(dtype=float),
            direct_target,
        )
        self.ardl_guided_ffnn_ = self._new_ffnn(seed_offset=202).fit(X, direct_target)
        self.residual_training_summary_ = {
            "samples": int(len(residual_rows)),
            "zones": int(residual_rows[zone_col].nunique()),
            "issues": int(residual_rows[self.data_config.issue_time_col].nunique()),
            "feature_count": int(len(feature_names)),
            "target_mean": float(residual_rows["actual_runoff"].mean()),
            "target_p90": float(residual_rows["actual_runoff"].quantile(0.90)),
            "residual_rmse_before_ffnn": float(
                np.sqrt(np.mean(np.square(residual_rows["residual_target"])))
            ),
            "benchmarks": "FFNN_FIXED_LAGS,ARDL_GUIDED_FFNN_DIRECT",
        }
        return self

    def forecast_issues(
        self,
        observations: pd.DataFrame,
        forecasts: pd.DataFrame,
        *,
        state_mode: str = "recursive",
    ) -> pd.DataFrame:
        if self.ffnn_ is None:
            raise RuntimeError("Call fit before forecast_issues.")
        mode = state_mode.replace("-", "_").lower()
        if mode not in {"recursive", "observed"}:
            raise ValueError("state_mode must be 'recursive' or 'observed'.")
        output: list[pd.DataFrame] = []
        zone_col = self.data_config.zone_col
        for zone, zone_forecasts in forecasts.groupby(zone_col, sort=True, observed=True):
            zone = str(zone)
            if zone not in self.models_:
                continue
            zone_obs = observations.loc[observations[zone_col].eq(zone)].copy()
            if mode == "observed":
                output.extend(self._forecast_zone_observed(zone, zone_obs, zone_forecasts))
            else:
                output.extend(self._forecast_zone_recursive(zone, zone_obs, zone_forecasts))
        if not output:
            return pd.DataFrame()
        return pd.concat(output, ignore_index=True).sort_values(
            [zone_col, self.data_config.issue_time_col, self.data_config.valid_time_col]
        ).reset_index(drop=True)

    def ardl_selection_table(self) -> pd.DataFrame:
        rows: list[pd.DataFrame] = []
        for zone, model in self.models_.items():
            table = model.selection_table_.copy()
            table.insert(0, self.data_config.zone_col, zone)
            rows.append(table)
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    def _build_residual_training_rows(
        self,
        observations: pd.DataFrame,
        forecasts: pd.DataFrame,
    ) -> pd.DataFrame:
        records: list[dict[str, object]] = []
        zone_col = self.data_config.zone_col
        issue_col = self.data_config.issue_time_col
        time_col = self.data_config.time_col
        target = self.data_config.target_col
        actual_lookup = observations.set_index([zone_col, time_col])[target]
        for zone, zone_forecasts in forecasts.groupby(zone_col, sort=False, observed=True):
            zone = str(zone)
            if zone not in self.models_:
                continue
            zone_obs = observations.loc[observations[zone_col].eq(zone)].copy()
            issues = sorted(zone_forecasts[issue_col].unique())
            kept_issues = self._stride_issues(issues)
            for issue in kept_issues:
                issue = pd.Timestamp(issue)
                future_raw = zone_forecasts.loc[zone_forecasts[issue_col].eq(issue)]
                try:
                    base, future, history = self._ardl_forecast_for_issue(zone, zone_obs, future_raw, issue)
                    features = self._residual_design(zone, issue, history, future, base)
                except ValueError:
                    continue
                for row in features.to_dict("records"):
                    key = (zone, row[self.data_config.valid_time_col])
                    if key not in actual_lookup.index:
                        continue
                    actual = float(actual_lookup.loc[key])
                    row["actual_runoff"] = actual
                    row["residual_target"] = actual - float(row["ardl_prediction"])
                    records.append(row)
        return pd.DataFrame.from_records(records)

    def _forecast_zone_observed(
        self,
        zone: str,
        observations: pd.DataFrame,
        forecasts: pd.DataFrame,
    ) -> list[pd.DataFrame]:
        issue_col = self.data_config.issue_time_col
        output: list[pd.DataFrame] = []
        for issue in sorted(forecasts[issue_col].unique()):
            issue = pd.Timestamp(issue)
            future_raw = forecasts.loc[forecasts[issue_col].eq(issue)]
            try:
                base, future, history = self._ardl_forecast_for_issue(
                    zone, observations, future_raw, issue
                )
            except ValueError:
                continue
            output.append(self._hybrid_prediction(zone, issue, history, future, base, "observed"))
        return output

    def _forecast_zone_recursive(
        self,
        zone: str,
        observations: pd.DataFrame,
        forecasts: pd.DataFrame,
    ) -> list[pd.DataFrame]:
        issue_col = self.data_config.issue_time_col
        time_col = self.data_config.time_col
        target = self.data_config.target_col
        issues = [pd.Timestamp(value) for value in sorted(forecasts[issue_col].unique())]
        if not issues:
            return []
        first_issue = issues[0]
        initial_state = observations.loc[
            observations[time_col] <= first_issue, [time_col, target]
        ].copy()
        initial_state = initial_state.set_index(time_col)[target]
        state_by_model = {
            model_name: initial_state.copy()
            for model_name in ("hybrid", "ardl", "fixed_ffnn", "guided_ffnn")
        }
        output: list[pd.DataFrame] = []
        for position, issue in enumerate(issues):
            future_raw = forecasts.loc[forecasts[issue_col].eq(issue)]
            predictions_by_model: dict[str, pd.DataFrame] = {}
            failed = False
            for model_name, state in state_by_model.items():
                history = observations.loc[observations[time_col] <= issue].copy()
                history[target] = history[time_col].map(state)
                history = history.dropna(subset=[target])
                try:
                    base, future, history_indexed = self._ardl_forecast_for_issue(
                        zone, history, future_raw, issue
                    )
                except ValueError:
                    failed = True
                    break
                predictions_by_model[model_name] = self._hybrid_prediction(
                    zone, issue, history_indexed, future, base, "recursive"
                )
            if failed:
                continue
            prediction = predictions_by_model["hybrid"].copy()
            prediction["ardl_prediction"] = predictions_by_model["ardl"][
                "ardl_prediction"
            ].to_numpy()
            prediction["ffnn_fixed_lag_prediction_mm_h"] = predictions_by_model[
                "fixed_ffnn"
            ]["ffnn_fixed_lag_prediction_mm_h"].to_numpy()
            prediction["ardl_guided_ffnn_prediction_mm_h"] = predictions_by_model[
                "guided_ffnn"
            ]["ardl_guided_ffnn_prediction_mm_h"].to_numpy()
            output.append(prediction)
            next_issue = issues[position + 1] if position + 1 < len(issues) else prediction[
                self.data_config.valid_time_col
            ].max()
            commit_columns = {
                "hybrid": "runoff_prediction_mm_h",
                "ardl": "ardl_prediction",
                "fixed_ffnn": "ffnn_fixed_lag_prediction_mm_h",
                "guided_ffnn": "ardl_guided_ffnn_prediction_mm_h",
            }
            for model_name, column in commit_columns.items():
                source = predictions_by_model[model_name]
                committed = source.loc[
                    source[self.data_config.valid_time_col] <= next_issue,
                    [self.data_config.valid_time_col, column],
                ]
                for _, row in committed.iterrows():
                    state_by_model[model_name].loc[row[self.data_config.valid_time_col]] = max(
                        0.0, float(row[column])
                    )
        return output

    def _ardl_forecast_for_issue(
        self,
        zone: str,
        observations: pd.DataFrame,
        future_raw: pd.DataFrame,
        issue: pd.Timestamp,
    ) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
        time_col = self.data_config.time_col
        target = self.data_config.target_col
        history = observations.loc[observations[time_col] <= issue].set_index(time_col)
        history = history[[target, *self.data_config.exogenous_cols]].sort_index()
        if history.empty or history.index.max() != issue:
            raise ValueError(f"No complete observation state at issue time {issue}.")
        future = future_raw.loc[
            future_raw["lead_hour"].between(1, self.ffnn_config.max_horizon_hours)
        ].copy()
        future = future.sort_values(self.data_config.valid_time_col).set_index(
            self.data_config.valid_time_col
        )
        if future.empty:
            raise ValueError("Forecast issue has no supported horizon rows.")
        expected = pd.date_range(
            issue + pd.Timedelta(hours=1), periods=int(future["lead_hour"].max()), freq="h"
        )
        if not future.index.equals(expected):
            raise ValueError("Forecast issue must contain a contiguous hourly path starting at +1h.")
        logical = future[list(self.data_config.nwp_cols)].rename(columns=self.nwp_to_observed)
        base = self.models_[zone].forecast(history, logical)
        return base, logical, history

    def _residual_design(
        self,
        zone: str,
        issue: pd.Timestamp,
        history: pd.DataFrame,
        future: pd.DataFrame,
        base: pd.Series,
    ) -> pd.DataFrame:
        issue_col = self.data_config.issue_time_col
        valid_col = self.data_config.valid_time_col
        rain_col = next(
            (
                column
                for column in self.data_config.exogenous_cols
                if "rain" in column.lower() or "precip" in column.lower()
            ),
            self.data_config.exogenous_cols[0],
        )
        static_values = self.static_encoder_.transform_zone(zone)
        cumulative_rain = future[rain_col].cumsum()
        records: list[dict[str, object]] = []
        for lead, valid in enumerate(future.index, start=1):
            hour_angle = 2.0 * np.pi * valid.hour / 24.0
            day_angle = 2.0 * np.pi * valid.dayofyear / 366.0
            row: dict[str, object] = {
                self.data_config.zone_col: zone,
                issue_col: issue,
                valid_col: valid,
                "ardl_prediction": float(base.loc[valid]),
                "lead_hour": float(lead),
                "lead_fraction": float(lead / self.ffnn_config.max_horizon_hours),
                "forecast_rain_cumulative": float(cumulative_rain.loc[valid]),
                "valid_hour_sin": float(np.sin(hour_angle)),
                "valid_hour_cos": float(np.cos(hour_angle)),
                "day_of_year_sin": float(np.sin(day_angle)),
                "day_of_year_cos": float(np.cos(day_angle)),
            }
            for lag in self.ffnn_config.fixed_target_lags:
                lag_time = issue - pd.Timedelta(hours=lag)
                if lag_time not in history.index:
                    raise ValueError(f"Missing target history at fixed FFNN lag {lag}.")
                row[f"fixed_target__lag_{lag}"] = float(
                    history.loc[lag_time, self.data_config.target_col]
                )
            for column in self.data_config.exogenous_cols:
                row[f"current__{column}"] = float(history[column].iloc[-1])
                row[f"forecast__{column}"] = float(future.loc[valid, column])
            row.update(static_values)
            records.append(row)
        return pd.DataFrame.from_records(records)

    def _hybrid_prediction(
        self,
        zone: str,
        issue: pd.Timestamp,
        history: pd.DataFrame,
        future: pd.DataFrame,
        base: pd.Series,
        state_mode: str,
    ) -> pd.DataFrame:
        assert self.ffnn_ is not None
        assert self.fixed_lag_ffnn_ is not None
        assert self.ardl_guided_ffnn_ is not None
        design = self._residual_design(zone, issue, history, future, base)
        X = design.loc[:, self.residual_feature_names_].to_numpy(dtype=float)
        residual = self.ffnn_.predict(X)
        raw = design["ardl_prediction"].to_numpy(dtype=float) + residual
        result = design[
            [
                self.data_config.zone_col,
                self.data_config.issue_time_col,
                self.data_config.valid_time_col,
                "lead_hour",
                "ardl_prediction",
            ]
        ].copy()
        result["ffnn_residual_prediction"] = residual
        result["runoff_prediction_mm_h"] = np.maximum(0.0, raw)
        result["ffnn_fixed_lag_prediction_mm_h"] = np.maximum(
            0.0,
            self.fixed_lag_ffnn_.predict(
                design.loc[:, self.fixed_lag_feature_names_].to_numpy(dtype=float)
            ),
        )
        result["ardl_guided_ffnn_prediction_mm_h"] = np.maximum(
            0.0,
            self.ardl_guided_ffnn_.predict(
                design.loc[:, self.ardl_guided_feature_names_].to_numpy(dtype=float)
            ),
        )
        result["state_mode"] = state_mode
        return result

    def _new_ffnn(self, *, seed_offset: int = 0) -> TensorFlowFFNNRegressor:
        return TensorFlowFFNNRegressor(
            hidden_layer_sizes=self.ffnn_config.hidden_layer_sizes,
            alpha=self.ffnn_config.alpha,
            seed=self.ffnn_config.seed + seed_offset,
            activation=self.ffnn_config.activation,
            learning_rate_init=self.ffnn_config.learning_rate,
            max_iter=self.ffnn_config.max_iter,
        )

    def _stride_issues(self, issues: list[object]) -> list[pd.Timestamp]:
        selected: list[pd.Timestamp] = []
        minimum = pd.Timedelta(hours=self.ffnn_config.issue_stride_hours)
        for value in issues:
            issue = pd.Timestamp(value)
            if not selected or issue - selected[-1] >= minimum:
                selected.append(issue)
        return selected

    def _cap_training_rows(self, rows: pd.DataFrame) -> pd.DataFrame:
        maximum = self.ffnn_config.max_residual_samples
        if len(rows) <= maximum:
            return rows.reset_index(drop=True)
        threshold = rows["actual_runoff"].quantile(0.90)
        extremes = rows.loc[rows["actual_runoff"] >= threshold]
        remaining_budget = max(0, maximum - len(extremes))
        ordinary = rows.loc[rows["actual_runoff"] < threshold]
        if len(extremes) >= maximum:
            sampled = extremes.nlargest(maximum, "actual_runoff")
        else:
            sampled_ordinary = ordinary.sample(
                n=min(remaining_budget, len(ordinary)), random_state=self.ffnn_config.seed
            )
            sampled = pd.concat([extremes, sampled_ordinary], ignore_index=True)
        return sampled.sample(frac=1.0, random_state=self.ffnn_config.seed).reset_index(drop=True)
