"""A small, explicit ARDL implementation for hourly per-catchment runoff series."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .flood_config import FloodARDLConfig


@dataclass
class SequentialARDL:
    """Select target/exogenous lag orders sequentially, then fit one linear ARDL.

    The implementation intentionally exposes the lagged design.  That makes recursive
    forecasts with issue-time NWP paths deterministic and keeps the same effective sample
    (`hold_back`) for every AIC/BIC comparison.
    """

    target: str
    exogenous: tuple[str, ...]
    config: FloodARDLConfig = field(default_factory=FloodARDLConfig)
    target_lags_: tuple[int, ...] = field(default_factory=tuple, init=False)
    exog_lags_: dict[str, tuple[int, ...]] = field(default_factory=dict, init=False)
    coefficients_: pd.Series | None = field(default=None, init=False)
    selection_table_: pd.DataFrame = field(default_factory=pd.DataFrame, init=False)
    fit_start_: pd.Timestamp | None = field(default=None, init=False)
    fit_end_: pd.Timestamp | None = field(default=None, init=False)

    def fit(self, frame: pd.DataFrame) -> "SequentialARDL":
        required = [self.target, *self.exogenous]
        missing = [column for column in required if column not in frame]
        if missing:
            raise ValueError(f"ARDL frame is missing columns: {missing}")
        clean = frame[required].astype(float).dropna().sort_index()
        if not isinstance(clean.index, pd.DatetimeIndex):
            raise ValueError("ARDL frame must use a DatetimeIndex.")
        if clean.index.has_duplicates:
            raise ValueError("ARDL frame contains duplicate timestamps.")
        if len(clean) < self.config.min_train_rows:
            raise ValueError(
                f"ARDL requires at least {self.config.min_train_rows} training rows; got {len(clean)}."
            )
        gaps = clean.index.to_series().diff().dropna() / pd.Timedelta(hours=1)
        if not gaps.eq(1.0).all():
            raise ValueError("ARDL training data must be a complete hourly series.")

        hold_back = max(self.config.max_target_lag, self.config.max_exog_lag)
        base_exog_lags = self._order_to_lags(0)
        rows: list[dict[str, object]] = []

        target_candidates: list[tuple[float, int]] = []
        for order in range(1, self.config.max_target_lag + 1):
            target_lags = tuple(range(1, order + 1))
            exog_lags = {name: base_exog_lags for name in self.exogenous}
            score, _, nobs = self._fit_spec(clean, target_lags, exog_lags, hold_back)
            rows.append(
                {
                    "stage": "target",
                    "feature": "__target__",
                    "order": order,
                    self.config.ic.lower(): score,
                    "nobs": nobs,
                }
            )
            target_candidates.append((score, order))
        _, selected_target_order = min(target_candidates, key=lambda value: (value[0], value[1]))
        self.target_lags_ = tuple(range(1, selected_target_order + 1))

        self.exog_lags_ = {}
        for feature in self.exogenous:
            candidates: list[tuple[float, int]] = []
            for order in range(0, self.config.max_exog_lag + 1):
                lags = self._order_to_lags(order)
                score, _, nobs = self._fit_spec(
                    clean,
                    self.target_lags_,
                    {feature: lags},
                    hold_back,
                )
                rows.append(
                    {
                        "stage": "exogenous",
                        "feature": feature,
                        "order": order,
                        self.config.ic.lower(): score,
                        "nobs": nobs,
                    }
                )
                candidates.append((score, order))
            _, selected_order = min(candidates, key=lambda value: (value[0], value[1]))
            self.exog_lags_[feature] = self._order_to_lags(selected_order)

        _, coefficients, _ = self._fit_spec(
            clean,
            self.target_lags_,
            self.exog_lags_,
            hold_back,
        )
        self.coefficients_ = coefficients
        selected_rows = rows.copy()
        for row in selected_rows:
            if row["stage"] == "target":
                row["selected"] = row["order"] == selected_target_order
            else:
                feature = str(row["feature"])
                selected_lags = self.exog_lags_[feature]
                selected_order = max(selected_lags) if selected_lags else 0
                row["selected"] = row["order"] == selected_order
        self.selection_table_ = pd.DataFrame.from_records(selected_rows)
        self.fit_start_ = clean.index.min()
        self.fit_end_ = clean.index.max()
        return self

    def forecast(self, history: pd.DataFrame, future_exog: pd.DataFrame) -> pd.Series:
        if self.coefficients_ is None:
            raise RuntimeError("Call fit before forecast.")
        history = history[[self.target, *self.exogenous]].astype(float).sort_index()
        future = future_exog[list(self.exogenous)].astype(float).sort_index()
        if history.empty or future.empty:
            raise ValueError("ARDL forecast requires non-empty history and future exogenous data.")
        if history.index.has_duplicates or future.index.has_duplicates:
            raise ValueError("ARDL history/future timestamps must be unique.")
        issue_time = history.index.max()
        expected_first = issue_time + pd.Timedelta(hours=1)
        if future.index.min() != expected_first:
            raise ValueError(
                f"Future exogenous path must start one hour after history ({expected_first})."
            )
        steps = future.index.to_series().diff().dropna() / pd.Timedelta(hours=1)
        if not steps.eq(1.0).all():
            raise ValueError("Future exogenous path must be contiguous hourly data.")

        target_state = history[self.target].copy()
        exog_state = pd.concat([history[list(self.exogenous)], future], axis=0)
        exog_state = exog_state[~exog_state.index.duplicated(keep="last")].sort_index()
        predictions: dict[pd.Timestamp, float] = {}
        for valid_time in future.index:
            values: dict[str, float] = {"const": 1.0}
            for lag in self.target_lags_:
                timestamp = valid_time - pd.Timedelta(hours=lag)
                if timestamp not in target_state.index:
                    raise ValueError(f"Missing target state at {timestamp} for ARDL lag {lag}.")
                values[f"{self.target}__lag_{lag}"] = float(target_state.loc[timestamp])
            for feature, lags in self.exog_lags_.items():
                for lag in lags:
                    timestamp = valid_time - pd.Timedelta(hours=lag)
                    if timestamp not in exog_state.index:
                        raise ValueError(f"Missing {feature} state at {timestamp} for ARDL lag {lag}.")
                    values[f"{feature}__lag_{lag}"] = float(exog_state.loc[timestamp, feature])
            vector = pd.Series(values).reindex(self.coefficients_.index)
            if vector.isna().any():
                raise RuntimeError("ARDL forecast design does not match fitted coefficients.")
            prediction = float(vector.to_numpy() @ self.coefficients_.to_numpy())
            predictions[valid_time] = prediction
            target_state.loc[valid_time] = prediction
        return pd.Series(predictions, name="ardl_prediction", dtype=float)

    def fitted_values(self, frame: pd.DataFrame) -> pd.Series:
        if self.coefficients_ is None:
            raise RuntimeError("Call fit before fitted_values.")
        hold_back = max(self.config.max_target_lag, self.config.max_exog_lag)
        X, _, index = self._design(frame, self.target_lags_, self.exog_lags_, hold_back)
        prediction = X.reindex(columns=self.coefficients_.index).to_numpy() @ self.coefficients_.to_numpy()
        return pd.Series(prediction, index=index, name="ardl_fitted")

    def _order_to_lags(self, order: int) -> tuple[int, ...]:
        start = 0 if self.config.include_exog_lag_zero else 1
        if order < start:
            return tuple()
        return tuple(range(start, order + 1))

    def _fit_spec(
        self,
        frame: pd.DataFrame,
        target_lags: tuple[int, ...],
        exog_lags: dict[str, tuple[int, ...]],
        hold_back: int,
    ) -> tuple[float, pd.Series, int]:
        X, y, _ = self._design(frame, target_lags, exog_lags, hold_back)
        if len(y) <= X.shape[1] + 1:
            return float("inf"), pd.Series(dtype=float), len(y)
        coefficients, *_ = np.linalg.lstsq(X.to_numpy(), y.to_numpy(), rcond=None)
        residual = y.to_numpy() - X.to_numpy() @ coefficients
        rss = max(float(residual @ residual), np.finfo(float).tiny)
        nobs = len(y)
        parameters = X.shape[1]
        if self.config.ic.lower() == "bic":
            score = nobs * np.log(rss / nobs) + parameters * np.log(nobs)
        else:
            score = nobs * np.log(rss / nobs) + 2.0 * parameters
        return float(score), pd.Series(coefficients, index=X.columns), nobs

    def _design(
        self,
        frame: pd.DataFrame,
        target_lags: tuple[int, ...],
        exog_lags: dict[str, tuple[int, ...]],
        hold_back: int,
    ) -> tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex]:
        clean = frame[[self.target, *self.exogenous]].astype(float).sort_index()
        design = pd.DataFrame(index=clean.index)
        design["const"] = 1.0
        for lag in target_lags:
            design[f"{self.target}__lag_{lag}"] = clean[self.target].shift(lag)
        for feature, lags in exog_lags.items():
            for lag in lags:
                design[f"{feature}__lag_{lag}"] = clean[feature].shift(lag)
        design[self.target] = clean[self.target]
        design = design.iloc[hold_back:].dropna()
        y = design.pop(self.target)
        return design, y, pd.DatetimeIndex(design.index)
