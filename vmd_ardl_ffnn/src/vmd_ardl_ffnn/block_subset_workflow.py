"""BlockSubsetWorkflow methods for the forecasting experiment."""

from __future__ import annotations

from itertools import product
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import numpy as np
import pandas as pd

from .artifacts import write_csv
from .features import LagSpec
from .metrics import evaluate_forecast


class BlockSubsetWorkflowMixin:
    def _component_signature_key(self, row: pd.Series) -> tuple[Any, ...]:
        """Return the seed-independent identity of one component candidate."""
        return (
            str(row["component"]),
            str(row["lag_spec"]),
            self._hidden_layer_sizes_from_row(row),
            float(row["alpha"]),
            str(row["activation"]),
            float(row["learning_rate_init"]),
        )

    def _fast_screen_block_subset_specs(
        self,
        *,
        component: str,
        train_data: pd.DataFrame,
        specs: list[LagSpec],
        validation: pd.DataFrame,
        validation_cache: dict[pd.Timestamp, dict[str, pd.DataFrame]],
        validation_actual_components: dict[pd.Timestamp, dict[str, float]],
        seeds: tuple[int, ...],
        top_k: int = 4,
        alpha: float = 0.1,
        learning_rate_init: float = 0.001,
    ) -> tuple[list[LagSpec], pd.DataFrame, pd.DataFrame]:
        """Fast-screen every block LagSpec with one lightweight FFNN per seed."""
        rows: list[dict[str, Any]] = []
        for spec in specs:
            hidden_layer_sizes = self.config.ffnn.fast_architecture_for(
                self._lag_spec_input_dimension(spec)
            )
            for seed in seeds:
                try:
                    model, n_train = self._fit_model(
                        train_data,
                        spec,
                        hidden_layer_sizes,
                        float(alpha),
                        int(seed),
                        self.config.ffnn.activation,
                        learning_rate_init=float(learning_rate_init),
                        max_iter=int(self.config.ffnn.fast_max_iter),
                    )
                    prediction = self._predict_component_from_cache(
                        model,
                        component,
                        validation_cache,
                        validation,
                        spec,
                        "validation",
                        actual_components=validation_actual_components,
                    )
                    if len(prediction) < int(self.config.ffnn.min_val):
                        raise ValueError(
                            f"Only {len(prediction)} validation rows in fast screen."
                        )
                    metrics = self._score_component_prediction(prediction, validation)
                    row: dict[str, Any] = {
                        "component": component,
                        "lag_spec": spec.label,
                        "target_lags": spec.target_lags,
                        "exog_lags": spec.exog_lags,
                        "input_dimension": self._lag_spec_input_dimension(spec),
                        "seed": int(seed),
                        "hidden_layer_sizes": hidden_layer_sizes,
                        "alpha": float(alpha),
                        "activation": self.config.ffnn.activation,
                        "learning_rate_init": float(learning_rate_init),
                        "max_iter": int(self.config.ffnn.fast_max_iter),
                        "n_train": int(n_train),
                        "n_val": int(len(prediction)),
                        "screen_stage": "block_subset_two_stage_fast_screen",
                        "screen_scale": "component_transformed",
                        "screen_validation_scope": "outer_validation",
                        "selection_data_scope": "outer_validation",
                        "error": None,
                    }
                    row.update({f"Val {metric}": value for metric, value in metrics.items()})
                except (ValueError, KeyError, IndexError) as exc:
                    row = {
                        "component": component,
                        "lag_spec": spec.label,
                        "target_lags": spec.target_lags,
                        "exog_lags": spec.exog_lags,
                        "input_dimension": self._lag_spec_input_dimension(spec),
                        "seed": int(seed),
                        "hidden_layer_sizes": hidden_layer_sizes,
                        "alpha": float(alpha),
                        "activation": self.config.ffnn.activation,
                        "learning_rate_init": float(learning_rate_init),
                        "max_iter": int(self.config.ffnn.fast_max_iter),
                        "n_train": 0,
                        "n_val": 0,
                        "screen_stage": "block_subset_two_stage_fast_screen",
                        "screen_scale": "component_transformed",
                        "screen_validation_scope": "outer_validation",
                        "selection_data_scope": "outer_validation",
                        "error": str(exc),
                        "Val RMSE": float("nan"),
                        "Val MAE": float("nan"),
                        "Val MAPE": float("nan"),
                        "Val Directional_Accuracy": float("nan"),
                    }
                rows.append(row)

        detail = pd.DataFrame(rows)
        valid = detail.dropna(subset=["Val RMSE"]).copy()
        summary = (
            valid.groupby("lag_spec", as_index=False)
            .agg(
                mean_val_RMSE=("Val RMSE", "mean"),
                std_val_RMSE=("Val RMSE", "std"),
                mean_val_MAE=("Val MAE", "mean"),
                mean_val_MAPE=("Val MAPE", "mean"),
                seed_count=("seed", "nunique"),
                input_dimension=("input_dimension", "first"),
            )
        )
        summary = summary[summary["seed_count"].eq(len(seeds))].copy()
        if summary.empty:
            raise ValueError(
                f"No complete two-stage fast-screen result for component={component}."
            )
        summary["std_val_RMSE"] = summary["std_val_RMSE"].fillna(0.0)
        summary = summary.sort_values(
            [
                "mean_val_RMSE",
                "mean_val_MAE",
                "mean_val_MAPE",
                "input_dimension",
                "lag_spec",
            ]
        ).reset_index(drop=True)
        summary.insert(0, "component", component)
        summary.insert(1, "fast_screen_rank", np.arange(1, len(summary) + 1))
        keep_count = min(max(1, int(top_k)), len(summary))
        summary["selected_for_full_grid"] = summary["fast_screen_rank"].le(keep_count)
        summary["fast_screen_alpha"] = float(alpha)
        summary["fast_screen_learning_rate_init"] = float(learning_rate_init)
        summary["fast_screen_seed_aggregation"] = "mean_across_reporting_seeds"
        summary["screen_validation_scope"] = "outer_validation"

        selected_labels = set(
            summary.loc[summary["selected_for_full_grid"], "lag_spec"].astype(str)
        )
        selected_specs = [spec for spec in specs if spec.label in selected_labels]
        detail["selected_for_full_grid"] = detail["lag_spec"].isin(selected_labels)
        return selected_specs, detail, summary

    def _fast_screen_block_subset_specs_inner_rolling(
        self,
        *,
        component: str,
        specs: list[LagSpec],
        contexts: list[dict[str, Any]],
        seeds: tuple[int, ...],
        top_k: int = 4,
        alpha: float = 0.1,
        learning_rate_init: float = 0.001,
    ) -> tuple[list[LagSpec], pd.DataFrame, pd.DataFrame]:
        """Fast-screen block specs on train-only expanding validation folds."""
        if not contexts:
            raise ValueError(
                "Block-subset inner rolling screen requires at least one fold."
            )
        rows: list[dict[str, Any]] = []
        for spec in specs:
            hidden_layer_sizes = self.config.ffnn.fast_architecture_for(
                self._lag_spec_input_dimension(spec)
            )
            for seed in seeds:
                for context in contexts:
                    fold = int(context["fold"])
                    fold_train = context["train"]
                    fold_validation = context["validation"]
                    try:
                        component_train = context["train_components"][component]
                        model, n_train = self._fit_model(
                            component_train,
                            spec,
                            hidden_layer_sizes,
                            float(alpha),
                            int(seed),
                            self.config.ffnn.activation,
                            learning_rate_init=float(learning_rate_init),
                            max_iter=int(self.config.ffnn.fast_max_iter),
                        )
                        prediction = self._predict_component_from_cache(
                            model,
                            component,
                            context["origin_cache"],
                            fold_validation,
                            spec,
                            f"block_subset_inner_fold_{fold}",
                            actual_components=context["actual_components"],
                        )
                        if len(prediction) < int(self.config.ffnn.min_val):
                            raise ValueError(
                                f"Only {len(prediction)} validation rows in "
                                "block-subset inner screen."
                            )
                        metrics = self._score_component_prediction(
                            prediction,
                            fold_validation,
                        )
                        row: dict[str, Any] = {
                            "component": component,
                            "lag_spec": spec.label,
                            "target_lags": spec.target_lags,
                            "exog_lags": spec.exog_lags,
                            "input_dimension": self._lag_spec_input_dimension(spec),
                            "seed": int(seed),
                            "fold": fold,
                            "train_start": fold_train.index.min(),
                            "train_end": fold_train.index.max(),
                            "validation_start": fold_validation.index.min(),
                            "validation_end": fold_validation.index.max(),
                            "hidden_layer_sizes": hidden_layer_sizes,
                            "alpha": float(alpha),
                            "activation": self.config.ffnn.activation,
                            "learning_rate_init": float(learning_rate_init),
                            "max_iter": int(self.config.ffnn.fast_max_iter),
                            "n_train": int(n_train),
                            "n_val": int(len(prediction)),
                            "screen_stage": (
                                "block_subset_two_stage_fast_screen"
                            ),
                            "screen_scale": "component_transformed",
                            "screen_validation_scope": "train_inner_rolling",
                            "selection_data_scope": "train_only_inner_rolling",
                            "error": None,
                        }
                        row.update(
                            {
                                f"Val {metric}": value
                                for metric, value in metrics.items()
                            }
                        )
                    except (ValueError, KeyError, IndexError) as exc:
                        row = {
                            "component": component,
                            "lag_spec": spec.label,
                            "target_lags": spec.target_lags,
                            "exog_lags": spec.exog_lags,
                            "input_dimension": self._lag_spec_input_dimension(spec),
                            "seed": int(seed),
                            "fold": fold,
                            "train_start": fold_train.index.min(),
                            "train_end": fold_train.index.max(),
                            "validation_start": fold_validation.index.min(),
                            "validation_end": fold_validation.index.max(),
                            "hidden_layer_sizes": hidden_layer_sizes,
                            "alpha": float(alpha),
                            "activation": self.config.ffnn.activation,
                            "learning_rate_init": float(learning_rate_init),
                            "max_iter": int(self.config.ffnn.fast_max_iter),
                            "n_train": 0,
                            "n_val": 0,
                            "screen_stage": (
                                "block_subset_two_stage_fast_screen"
                            ),
                            "screen_scale": "component_transformed",
                            "screen_validation_scope": "train_inner_rolling",
                            "selection_data_scope": "train_only_inner_rolling",
                            "error": str(exc),
                            "Val RMSE": float("nan"),
                            "Val MAE": float("nan"),
                            "Val MAPE": float("nan"),
                            "Val Directional_Accuracy": float("nan"),
                        }
                    rows.append(row)

        detail = pd.DataFrame(rows)
        valid = detail.dropna(subset=["Val RMSE"]).copy()
        expected_fold_count = len(contexts)
        completeness = (
            valid.groupby("lag_spec", as_index=False)
            .agg(
                seed_count=("seed", "nunique"),
                fold_count=("fold", "nunique"),
                valid_row_count=("Val RMSE", "count"),
            )
        )
        completeness = completeness[
            completeness["seed_count"].eq(len(seeds))
            & completeness["fold_count"].eq(expected_fold_count)
            & completeness["valid_row_count"].eq(
                len(seeds) * expected_fold_count
            )
        ].copy()
        complete_labels = set(completeness["lag_spec"].astype(str))
        valid = valid[valid["lag_spec"].astype(str).isin(complete_labels)]
        summary = (
            valid.groupby("lag_spec", as_index=False)
            .agg(
                mean_val_RMSE=("Val RMSE", "mean"),
                std_val_RMSE=("Val RMSE", "std"),
                mean_val_MAE=("Val MAE", "mean"),
                mean_val_MAPE=("Val MAPE", "mean"),
                seed_count=("seed", "nunique"),
                fold_count=("fold", "nunique"),
                input_dimension=("input_dimension", "first"),
            )
        )
        if summary.empty:
            raise ValueError(
                f"No complete inner rolling fast-screen result for "
                f"component={component}."
            )
        summary["std_val_RMSE"] = summary["std_val_RMSE"].fillna(0.0)
        summary = summary.sort_values(
            [
                "mean_val_RMSE",
                "mean_val_MAE",
                "mean_val_MAPE",
                "input_dimension",
                "lag_spec",
            ]
        ).reset_index(drop=True)
        summary.insert(0, "component", component)
        summary.insert(1, "fast_screen_rank", np.arange(1, len(summary) + 1))
        keep_count = min(max(1, int(top_k)), len(summary))
        summary["selected_for_full_grid"] = summary["fast_screen_rank"].le(
            keep_count
        )
        summary["fast_screen_alpha"] = float(alpha)
        summary["fast_screen_learning_rate_init"] = float(
            learning_rate_init
        )
        summary["fast_screen_seed_aggregation"] = (
            "mean_across_reporting_seeds_and_inner_folds"
        )
        summary["screen_validation_scope"] = "train_inner_rolling"

        selected_labels = set(
            summary.loc[
                summary["selected_for_full_grid"],
                "lag_spec",
            ].astype(str)
        )
        selected_specs = [spec for spec in specs if spec.label in selected_labels]
        detail["selected_for_full_grid"] = detail["lag_spec"].isin(
            selected_labels
        )
        return selected_specs, detail, summary

    def _fast_screen_specs_rolling_mean_seed(
        self,
        *,
        component: str,
        specs: list[LagSpec],
        seeds: tuple[int, ...],
        evaluate: Callable[[LagSpec, float, int], list[dict[str, Any]]],
    ) -> tuple[list[LagSpec], pd.DataFrame, pd.DataFrame]:
        """Create one train-only rolling shortlist shared by all reporting seeds."""
        fold_rows: list[dict[str, Any]] = []
        stability_start = len(self._lag_screen_stability_rows)
        for spec in specs:
            for learning_rate in self.config.ffnn.lag_screen_learning_rates():
                for seed in seeds:
                    for raw_row in evaluate(spec, float(learning_rate), int(seed)):
                        row = {
                            **raw_row,
                            "component": component,
                            "lag_spec": spec.label,
                            "target_lags": spec.target_lags,
                            "exog_lags": spec.exog_lags,
                            "input_dimension": self._lag_spec_input_dimension(spec),
                            "learning_rate_init": float(learning_rate),
                            "activation": self.config.ffnn.activation,
                            "alpha": float(self.config.ffnn.fast_alpha),
                            "max_iter": int(self.config.ffnn.fast_max_iter),
                            "seed": int(seed),
                            "selection_data_scope": "train_only_inner_rolling",
                            "selection_seed_aggregation": "mean_across_reporting_seeds",
                        }
                        fold_rows.append(row)
                        self._lag_screen_fold_rows.append(row)

        detail = pd.DataFrame(fold_rows)
        valid = detail.dropna(subset=["screen_RMSE"]).copy()
        expected_fold_count = int(detail["fold"].nunique()) if not detail.empty else 0
        complete = (
            valid.groupby(["lag_spec", "learning_rate_init"], as_index=False)
            .agg(
                seed_count=("seed", "nunique"),
                valid_row_count=("screen_RMSE", "count"),
            )
        )
        complete = complete[
            complete["seed_count"].eq(len(seeds))
            & complete["valid_row_count"].eq(len(seeds) * expected_fold_count)
        ]
        complete_keys = set(
            zip(
                complete["lag_spec"].astype(str),
                complete["learning_rate_init"].astype(float),
                strict=True,
            )
        )
        complete_rows = [
            row
            for row in fold_rows
            if (str(row["lag_spec"]), float(row["learning_rate_init"]))
            in complete_keys
        ]
        selected_labels, _ = self._summarize_lag_screen(
            component,
            specs,
            complete_rows,
        )
        if not selected_labels:
            raise ValueError(
                f"No complete mean-seed rolling lag screen for component={component}."
            )
        selected_specs = [spec for spec in specs if spec.label in selected_labels]
        detail["selected_for_full_grid"] = detail["lag_spec"].isin(selected_labels)
        summary = pd.DataFrame(self._lag_screen_stability_rows[stability_start:])
        if not summary.empty:
            summary["selection_seed_aggregation"] = "mean_across_reporting_seeds"
            summary["reporting_seed_count"] = len(seeds)
            summary["selected_for_full_grid"] = summary[
                "selected_for_hyperparam_search"
            ]
        return selected_specs, detail, summary

    def _rank_mean_seed_component_signatures(
        self,
        component_candidates: pd.DataFrame,
        seeds: tuple[int, ...],
        top_component_candidates: int | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Rank shared component signatures and flag the reconstructed shortlist."""
        if component_candidates.empty:
            raise ValueError("Cannot rank an empty component-candidate table.")
        component = str(component_candidates["component"].iloc[0])
        signature_columns = [
            "lag_spec",
            "hidden_layer_sizes",
            "alpha",
            "activation",
            "learning_rate_init",
        ]
        summary = (
            component_candidates.groupby(signature_columns, as_index=False, dropna=False)
            .agg(
                mean_component_val_RMSE=("Val RMSE", "mean"),
                std_component_val_RMSE=("Val RMSE", "std"),
                mean_component_val_MAE=("Val MAE", "mean"),
                mean_component_val_MAPE=("Val MAPE", "mean"),
                seed_count=("seed", "nunique"),
            )
        )
        summary = summary[summary["seed_count"].eq(len(seeds))].copy()
        if summary.empty:
            raise ValueError(
                f"No complete mean-seed candidate signature for component={component}."
            )
        summary["std_component_val_RMSE"] = summary[
            "std_component_val_RMSE"
        ].fillna(0.0)
        summary = summary.sort_values(
            [
                "mean_component_val_RMSE",
                "mean_component_val_MAE",
                "mean_component_val_MAPE",
                "lag_spec",
                "hidden_layer_sizes",
                "alpha",
                "learning_rate_init",
            ]
        ).reset_index(drop=True)
        summary.insert(0, "component", component)
        summary.insert(
            1,
            "component_mean_seed_rank",
            np.arange(1, len(summary) + 1),
        )
        summary.insert(
            2,
            "component_signature_id",
            [f"{component}_S{rank:04d}" for rank in summary["component_mean_seed_rank"]],
        )
        top_n = min(
            max(
                1,
                int(
                    top_component_candidates
                    if top_component_candidates is not None
                    else self.config.ffnn.top_component_candidates
                ),
            ),
            len(summary),
        )
        summary["shortlisted_for_reconstructed_search"] = summary[
            "component_mean_seed_rank"
        ].le(top_n)
        summary["selected_in_reconstructed_winner"] = False

        signature_id_by_key = {
            self._component_signature_key(row): str(row["component_signature_id"])
            for _, row in summary.iterrows()
        }
        ranked_candidates = component_candidates.copy()
        ranked_candidates["component_signature_id"] = [
            signature_id_by_key.get(self._component_signature_key(row))
            for _, row in ranked_candidates.iterrows()
        ]
        shortlisted_ids = set(
            summary.loc[
                summary["shortlisted_for_reconstructed_search"],
                "component_signature_id",
            ].astype(str)
        )
        ranked_candidates["eligible_shared_seed_selection"] = ranked_candidates[
            "component_signature_id"
        ].notna()
        ranked_candidates["shortlisted_for_reconstructed_search"] = ranked_candidates[
            "component_signature_id"
        ].isin(shortlisted_ids)
        ranked_candidates["selected_in_reconstructed_winner"] = False
        return ranked_candidates, summary

    def _score_mean_seed_reconstructed_combinations(
        self,
        candidates: pd.DataFrame,
        component_rankings: pd.DataFrame,
        shortlisted_predictions: dict[int, pd.DataFrame],
        validation: pd.DataFrame,
        seeds: tuple[int, ...],
        component_order: tuple[str, ...],
        top_component_candidates: int | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Score shared component-signature combinations on reconstructed level."""
        shortlisted = candidates[
            candidates["shortlisted_for_reconstructed_search"]
        ].copy()
        lookup_columns = ["component", "component_signature_id", "seed"]
        if shortlisted.duplicated(lookup_columns).any():
            raise ValueError("Duplicate candidate for a component signature and seed.")
        candidate_id_by_signature_seed = {
            (str(row["component"]), str(row["component_signature_id"]), int(row["seed"])): int(
                row["candidate_id"]
            )
            for _, row in shortlisted.iterrows()
        }

        top_signatures: dict[str, list[str]] = {}
        complexity_by_signature: dict[tuple[str, str], int] = {}
        for component in component_order:
            component_top = component_rankings[
                component_rankings["component"].eq(component)
                & component_rankings["shortlisted_for_reconstructed_search"]
            ].sort_values("component_mean_seed_rank")
            if component_top.empty:
                raise ValueError(
                    f"No shortlisted shared signature for component={component}."
                )
            top_signatures[component] = component_top[
                "component_signature_id"
            ].astype(str).tolist()
            for signature_id in top_signatures[component]:
                row = shortlisted[
                    shortlisted["component"].eq(component)
                    & shortlisted["component_signature_id"].eq(signature_id)
                ].iloc[0]
                complexity_by_signature[(component, signature_id)] = (
                    len(tuple(row["target_lags"]))
                    + sum(len(tuple(lags)) for lags in row["exog_lags"].values())
                )

        combo_metadata: list[dict[str, Any]] = []
        seed_score_rows: list[dict[str, Any]] = []
        for combo_id, signature_ids in enumerate(
            product(*(top_signatures[component] for component in component_order))
        ):
            signature_by_component = dict(zip(component_order, signature_ids, strict=True))
            metadata: dict[str, Any] = {
                "combo_id": int(combo_id),
                "component_signature_ids": tuple(signature_ids),
                "top_component_candidates": int(
                    top_component_candidates
                    if top_component_candidates is not None
                    else self.config.ffnn.top_component_candidates
                ),
                "total_input_dimension": sum(
                    complexity_by_signature[(component, signature_by_component[component])]
                    for component in component_order
                ),
            }
            metadata.update(
                {
                    f"{component}_signature_id": signature_by_component[component]
                    for component in component_order
                }
            )
            combo_metadata.append(metadata)

            for seed in seeds:
                candidate_ids = tuple(
                    candidate_id_by_signature_seed[
                        (component, signature_by_component[component], int(seed))
                    ]
                    for component in component_order
                )
                component_frames = [
                    shortlisted_predictions[candidate_id]
                    for candidate_id in candidate_ids
                ]
                reconstructed = self._reconstruct(
                    pd.concat(component_frames, ignore_index=True, sort=False),
                    validation,
                )
                transformed_metrics = evaluate_forecast(
                    reconstructed["actual_raw_transformed"],
                    reconstructed["predicted_reconstructed"],
                )
                actual_level, predicted_level = self._to_level(
                    reconstructed,
                    "actual_raw_transformed",
                    "predicted_reconstructed",
                )
                level_metrics = evaluate_forecast(actual_level, predicted_level)
                score_row: dict[str, Any] = {
                    "combo_id": int(combo_id),
                    "seed": int(seed),
                    "component_candidate_ids": candidate_ids,
                }
                score_row.update(
                    {f"Val {metric}": value for metric, value in transformed_metrics.items()}
                )
                score_row.update(
                    {f"Val Level {metric}": value for metric, value in level_metrics.items()}
                )
                seed_score_rows.append(score_row)

        seed_scores = pd.DataFrame(seed_score_rows)
        combo_summary = (
            seed_scores.groupby("combo_id", as_index=False)
            .agg(
                mean_val_level_RMSE=("Val Level RMSE", "mean"),
                std_val_level_RMSE=("Val Level RMSE", "std"),
                mean_val_level_MAE=("Val Level MAE", "mean"),
                mean_val_level_MAPE=("Val Level MAPE", "mean"),
                mean_val_level_Directional_Accuracy=(
                    "Val Level Directional_Accuracy",
                    "mean",
                ),
                mean_val_transformed_RMSE=("Val RMSE", "mean"),
                seed_count=("seed", "nunique"),
            )
            .merge(pd.DataFrame(combo_metadata), on="combo_id", how="left")
        )
        combo_summary["std_val_level_RMSE"] = combo_summary[
            "std_val_level_RMSE"
        ].fillna(0.0)
        combo_summary = combo_summary.sort_values(
            [
                "mean_val_level_RMSE",
                "mean_val_level_MAE",
                "mean_val_level_MAPE",
                "total_input_dimension",
                "combo_id",
            ]
        ).reset_index(drop=True)
        combo_summary.insert(
            1,
            "mean_seed_reconstructed_rank",
            np.arange(1, len(combo_summary) + 1),
        )
        combo_summary["selected"] = combo_summary[
            "mean_seed_reconstructed_rank"
        ].eq(1)
        combo_summary["final_selection_scale"] = "level"
        combo_summary["selection_protocol"] = (
            "mean_reconstructed_level_validation_across_reporting_seeds"
        )
        return combo_summary, seed_scores

    def run_block_subset_mean_seed(
        self,
        csv_path: str | Path,
        *,
        reporting_seeds: tuple[int, ...],
        search_strategy: str = "full_grid",
        fast_screen_validation_scope: str = "outer_validation",
    ) -> dict[str, pd.DataFrame]:
        """Run the AIC variable-block subset comparator with shared seed selection.

        Target lags are selected once per VMD component. AIC then selects one
        distributed-lag order for every exogenous variable, all non-empty subsets
        of those variable blocks are enumerated, and the full FFNN grid is scored.
        Candidate signatures are first shortlisted per component using mean
        component validation metrics across seeds. Their cross-component
        combinations are then ranked by mean reconstructed level validation RMSE
        across the same seeds, never by the best individual seed. The locked
        reconstructed winner is refit/evaluated for every seed.
        """
        seeds = tuple(dict.fromkeys(int(seed) for seed in reporting_seeds))
        block_search_strategy = search_strategy.replace("-", "_")
        fast_screen_scope = fast_screen_validation_scope.replace("-", "_")
        if not seeds:
            raise ValueError("Block-subset mean-seed selection requires reporting seeds.")
        if block_search_strategy not in {"full_grid", "two_stage"}:
            raise ValueError(
                "Block-subset search_strategy must be 'full_grid' or 'two_stage'."
            )
        if fast_screen_scope not in {
            "outer_validation",
            "train_inner_rolling",
        }:
            raise ValueError(
                "fast_screen_validation_scope must be 'outer_validation' or "
                "'train_inner_rolling'."
            )
        if self.config.ardl.exog_lag_strategy.replace("-", "_") != "ardl_block_subsets":
            raise ValueError(
                "run_block_subset_mean_seed requires exog_lag_strategy='ardl_block_subsets'."
            )
        if self.config.ardl.ic.lower() != "aic":
            raise ValueError("Block-subset variable orders must be selected with AIC.")
        if self.config.ffnn.search_strategy.replace("-", "_") != "full_grid":
            raise ValueError(
                "Block-subset mean-seed selection requires search_strategy='full_grid'."
            )
        if tuple(self.config.ffnn.seed_grid) != seeds:
            raise ValueError("FFNN seed_grid must exactly match reporting_seeds.")

        total_started = perf_counter()
        self._log(
            "Starting AIC variable-block subset comparator with mean reconstructed "
            f"level validation selection across seeds={seeds}; "
            f"search_strategy={block_search_strategy}; "
            f"fast_screen_validation_scope={fast_screen_scope}."
        )
        base_data = self.load_data(csv_path)
        base_train, base_val, base_test = self.split_raw_data(base_data)
        data = self._apply_stationarity_transform(base_data, base_train)
        train = data.loc[data.index.intersection(base_train.index)].copy()
        val = data.loc[data.index.intersection(base_val.index)].copy()
        test = data.loc[data.index.intersection(base_test.index)].copy()

        audit: list[dict[str, Any]] = []
        for split, frame in (("train", train), ("validation", val), ("test", test)):
            audit.extend(self._audit_rows("block_subset_mean_seed", "chronological_split", split, frame))

        vmd_cache_started = perf_counter()
        train_components = self.decompose(train)
        val_cache = self._make_origin_cache(train, val, "block-subset-validation")
        val_actual_components = self._make_actual_component_cache(
            train,
            val,
            "block-subset-validation",
        )
        vmd_cache_seconds = perf_counter() - vmd_cache_started

        selection_started = perf_counter()
        self._select_specs(train_components)
        ardl_selection_seconds = perf_counter() - selection_started

        ffnn_search_started = perf_counter()
        candidate_frames: list[pd.DataFrame] = []
        component_ranking_frames: list[pd.DataFrame] = []
        shortlisted_predictions: dict[int, pd.DataFrame] = {}
        fast_screen_detail_frames: list[pd.DataFrame] = []
        fast_screen_summary_frames: list[pd.DataFrame] = []
        candidate_id = 0
        inner_screen_contexts = (
            self._prepare_vmd_lag_screen_contexts(train)
            if block_search_strategy == "two_stage"
            and fast_screen_scope == "train_inner_rolling"
            else []
        )
        if (
            block_search_strategy == "two_stage"
            and fast_screen_scope == "train_inner_rolling"
            and not inner_screen_contexts
        ):
            raise ValueError(
                "No train-only inner rolling folds are available for the "
                "block-subset fast screen."
            )
        for component, train_component in train_components.items():
            component_specs = self.selected_specs_[component]
            if block_search_strategy == "two_stage":
                if fast_screen_scope == "train_inner_rolling":
                    component_specs, fast_detail, fast_summary = (
                        self._fast_screen_block_subset_specs_inner_rolling(
                            component=component,
                            specs=component_specs,
                            contexts=inner_screen_contexts,
                            seeds=seeds,
                            top_k=4,
                            alpha=0.1,
                            learning_rate_init=0.001,
                        )
                    )
                else:
                    component_specs, fast_detail, fast_summary = (
                        self._fast_screen_block_subset_specs(
                            component=component,
                            train_data=train_component,
                            specs=component_specs,
                            validation=val,
                            validation_cache=val_cache,
                            validation_actual_components=val_actual_components,
                            seeds=seeds,
                            top_k=4,
                            alpha=0.1,
                            learning_rate_init=0.001,
                        )
                    )
                fast_screen_detail_frames.append(fast_detail)
                fast_screen_summary_frames.append(fast_summary)
            rows, predictions, _, candidate_id = self._ffnn_search_candidates(
                component=component,
                train_data=train_component,
                specs=component_specs,
                prediction_fn=lambda model, spec, row_id, component=component: self._predict_component_from_cache(
                    model,
                    component,
                    val_cache,
                    val,
                    spec,
                    "validation",
                    row_id,
                    val_actual_components,
                ),
                scoring_fn=lambda pred: self._score_component_prediction(pred, val),
                uses_vmd=True,
                candidate_id_start=candidate_id,
            )
            component_candidates = pd.DataFrame(rows)
            ranked_candidates, component_ranking = (
                self._rank_mean_seed_component_signatures(
                    component_candidates,
                    seeds,
                )
            )
            candidate_frames.append(ranked_candidates)
            component_ranking_frames.append(component_ranking)
            shortlisted_candidate_ids = set(
                ranked_candidates.loc[
                    ranked_candidates["shortlisted_for_reconstructed_search"],
                    "candidate_id",
                ].astype(int)
            )
            shortlisted_predictions.update(
                {
                    candidate: predictions[candidate]
                    for candidate in shortlisted_candidate_ids
                }
            )

        fast_screen_detail_df = (
            pd.concat(fast_screen_detail_frames, ignore_index=True, sort=False)
            if fast_screen_detail_frames
            else pd.DataFrame()
        )
        fast_screen_summary_df = (
            pd.concat(fast_screen_summary_frames, ignore_index=True, sort=False)
            if fast_screen_summary_frames
            else pd.DataFrame()
        )
        candidates_df = pd.concat(candidate_frames, ignore_index=True, sort=False)
        if candidates_df.empty:
            raise ValueError("No FFNN candidate could be scored for block-subset selection.")
        component_rankings_df = pd.concat(
            component_ranking_frames,
            ignore_index=True,
            sort=False,
        )
        selection_df, reconstructed_seed_scores = (
            self._score_mean_seed_reconstructed_combinations(
                candidates_df,
                component_rankings_df,
                shortlisted_predictions,
                val,
                seeds,
                tuple(train_components),
            )
        )
        for frame in (
            candidates_df,
            component_rankings_df,
            reconstructed_seed_scores,
            selection_df,
        ):
            frame["block_subset_search_strategy"] = block_search_strategy
            frame["block_subset_fast_screen_top_k"] = (
                4 if block_search_strategy == "two_stage" else pd.NA
            )
            frame["block_subset_fast_screen_alpha"] = (
                0.1 if block_search_strategy == "two_stage" else pd.NA
            )
            frame["block_subset_fast_screen_learning_rate_init"] = (
                0.001 if block_search_strategy == "two_stage" else pd.NA
            )
            frame["block_subset_fast_screen_validation_scope"] = (
                fast_screen_scope
                if block_search_strategy == "two_stage"
                else pd.NA
            )
        winning_combo = selection_df.iloc[0]
        selected_signature_by_component = {
            component: str(winning_combo[f"{component}_signature_id"])
            for component in train_components
        }
        selected_mask = pd.Series(False, index=candidates_df.index)
        for component, signature_id in selected_signature_by_component.items():
            selected_mask |= candidates_df["component"].eq(component) & candidates_df[
                "component_signature_id"
            ].eq(signature_id)
        best_candidates = candidates_df[selected_mask].copy()
        if len(best_candidates) != len(train_components) * len(seeds):
            raise ValueError(
                "Reconstructed mean-seed winner does not cover every component and seed."
            )
        candidates_df["selected_in_reconstructed_winner"] = selected_mask
        best_candidates["selected_in_reconstructed_winner"] = True
        for component, signature_id in selected_signature_by_component.items():
            component_rankings_df.loc[
                component_rankings_df["component"].eq(component)
                & component_rankings_df["component_signature_id"].eq(signature_id),
                "selected_in_reconstructed_winner",
            ] = True
        best_candidates["combo_id"] = int(winning_combo["combo_id"])
        best_candidates["final_selection_scale"] = "level"
        best_candidates["combo_mean_seed_val_level_RMSE"] = float(
            winning_combo["mean_val_level_RMSE"]
        )
        best_candidates["combo_std_seed_val_level_RMSE"] = float(
            winning_combo["std_val_level_RMSE"]
        )
        best_candidates["combo_mean_seed_val_level_MAE"] = float(
            winning_combo["mean_val_level_MAE"]
        )
        best_candidates["combo_mean_seed_val_level_MAPE"] = float(
            winning_combo["mean_val_level_MAPE"]
        )
        best_candidates["combo_total_input_dimension"] = int(
            winning_combo["total_input_dimension"]
        )
        best_candidates["selection_seed_count"] = len(seeds)
        best_candidates["block_subset_search_strategy"] = block_search_strategy
        best_candidates["block_subset_fast_screen_validation_scope"] = (
            fast_screen_scope
            if block_search_strategy == "two_stage"
            else pd.NA
        )
        best_candidates["selection_protocol"] = (
            "mean_reconstructed_level_validation_across_reporting_seeds"
        )
        ffnn_search_seconds = perf_counter() - ffnn_search_started

        refit_started = perf_counter()
        history_before_test = pd.concat([train, val], axis=0)
        history_components = self.decompose(history_before_test)
        test_cache = self._make_origin_cache(
            history_before_test,
            test,
            "block-subset-test",
        )
        test_actual_components = self._make_actual_component_cache(
            history_before_test,
            test,
            "block-subset-test",
        )
        selected_component_predictions: list[pd.DataFrame] = []
        for row_index, best in best_candidates.iterrows():
            component = str(best["component"])
            seed = int(best["seed"])
            spec = LagSpec(
                tuple(best["target_lags"]),
                {feature: tuple(lags) for feature, lags in best["exog_lags"].items()},
            )
            validation_prediction = shortlisted_predictions[
                int(best["candidate_id"])
            ].copy()
            validation_prediction["seed"] = seed
            selected_component_predictions.append(validation_prediction)

            test_model, n_refit = self._fit_model(
                history_components[component],
                spec,
                self._hidden_layer_sizes_from_row(best),
                float(best["alpha"]),
                seed,
                str(best["activation"]),
                learning_rate_init=float(best["learning_rate_init"]),
            )
            test_prediction = self._predict_component_from_cache(
                test_model,
                component,
                test_cache,
                test,
                spec,
                "test",
                actual_components=test_actual_components,
            )
            test_prediction["seed"] = seed
            selected_component_predictions.append(test_prediction)
            best_candidates.loc[row_index, "n_refit_train_val"] = int(n_refit)

        self.component_forecasts_ = pd.concat(
            selected_component_predictions,
            ignore_index=True,
            sort=False,
        )
        actual_all = pd.concat([val, test], axis=0)
        self.final_forecasts_ = self._reconstruct(
            self.component_forecasts_,
            actual_all,
        ).reset_index(drop=True)
        self.final_forecasts_["forecast_protocol"] = self.config.forecast_protocol

        metric_frames: list[pd.DataFrame] = []
        for seed in seeds:
            index = self.final_forecasts_.index[self.final_forecasts_["seed"].eq(seed)]
            seed_forecasts = self.final_forecasts_.loc[index].copy()
            seed_best = best_candidates[best_candidates["seed"].eq(seed)].copy()
            metrics = self._annotate_final_metrics(
                self._metrics_rows(
                    seed_forecasts,
                    "actual_raw_transformed",
                    "predicted_reconstructed",
                    "reconstructed_transformed",
                ),
                csv_path,
                "VMD_ARDL_BlockSubset_MeanSeed_FFNN",
                True,
                seed_best,
            )
            metrics["seed"] = seed
            metric_frames.append(metrics)
            for column in ("actual_level", "predicted_level"):
                if column in seed_forecasts:
                    self.final_forecasts_.loc[index, column] = seed_forecasts[column].to_numpy()

        self.final_metrics_ = pd.concat(metric_frames, ignore_index=True, sort=False)
        self.best_component_models_ = best_candidates.sort_values(
            ["seed", "component"]
        ).reset_index(drop=True)
        self.input_audit_ = pd.DataFrame(audit)
        refit_forecast_seconds = perf_counter() - refit_started

        ardl_orders = pd.concat(
            [table.assign(component=name) for name, table in self.order_tables_.items()],
            ignore_index=True,
        )
        diagnostics = self._diagnostics(
            train,
            train_components,
            "_block_subset_mean_seed",
        )
        timing = pd.DataFrame(
            [
                {
                    "ardl_selection_seconds": ardl_selection_seconds,
                    "vmd_cache_seconds": vmd_cache_seconds,
                    "ffnn_search_seconds": ffnn_search_seconds,
                    "refit_forecast_seconds": refit_forecast_seconds,
                    "total_runtime_seconds": perf_counter() - total_started,
                }
            ]
        )

        write_csv(candidates_df, self.output_dir / "ffnn_block_subset_component_search.csv")
        if not fast_screen_detail_df.empty:
            write_csv(
                fast_screen_detail_df,
                self.output_dir / "ffnn_block_subset_fast_screen_metrics.csv",
            )
        if not fast_screen_summary_df.empty:
            write_csv(
                fast_screen_summary_df,
                self.output_dir / "ffnn_block_subset_fast_screen_summary.csv",
            )
        write_csv(
            component_rankings_df,
            self.output_dir / "ffnn_block_subset_component_mean_seed_ranking.csv",
        )
        write_csv(
            reconstructed_seed_scores,
            self.output_dir / "ffnn_block_subset_reconstructed_seed_scores.csv",
        )
        write_csv(selection_df, self.output_dir / "ffnn_block_subset_mean_seed_selection.csv")
        write_csv(
            self.best_component_models_,
            self.output_dir / "best_component_models_block_subset_mean_seed.csv",
        )
        write_csv(
            self.component_forecasts_,
            self.output_dir / "component_predictions_block_subset_mean_seed.csv",
        )
        write_csv(
            self.final_forecasts_,
            self.output_dir / "final_reconstructed_forecasts_block_subset_mean_seed.csv",
        )
        write_csv(
            self.final_metrics_,
            self.output_dir / "final_reconstructed_metrics_block_subset_mean_seed.csv",
        )
        write_csv(ardl_orders, self.output_dir / "ardl_variable_blocks_train_only.csv")
        write_csv(self.input_audit_, self.output_dir / "input_audit_block_subset_mean_seed.csv")
        if self.stationarity_screen_ is not None:
            write_csv(
                self.stationarity_screen_,
                self.output_dir / "stationarity_screen_train_only_block_subset_mean_seed.csv",
            )
        for name, frame in diagnostics.items():
            write_csv(frame, self.output_dir / f"{name}_train_only.csv")
        write_csv(timing, self.output_dir / "runtime_breakdown.csv")

        return {
            "search_results": selection_df,
            "component_search_results": candidates_df,
            "component_signature_rankings": component_rankings_df,
            "reconstructed_seed_scores": reconstructed_seed_scores,
            "block_subset_fast_screen_metrics": fast_screen_detail_df,
            "block_subset_fast_screen_summary": fast_screen_summary_df,
            "ardl_orders": ardl_orders,
            "best_component_models": self.best_component_models_,
            "component_forecasts": self.component_forecasts_,
            "training_components": train_components,
            "final_forecasts": self.final_forecasts_,
            "final_metrics": self.final_metrics_,
            "input_audit": self.input_audit_,
            "stationarity_screen": self.stationarity_screen_,
            "lag_screen_fold_metrics": pd.DataFrame(),
            "lag_screen_stability": pd.DataFrame(),
            "timing": timing,
            **diagnostics,
        }
