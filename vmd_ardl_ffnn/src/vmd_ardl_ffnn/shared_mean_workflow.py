"""SharedMeanWorkflow methods for the forecasting experiment."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import write_csv
from .features import LagSpec


class SharedMeanWorkflowMixin:
    def run_vmd_shared_mean_seed(
        self,
        csv_path: str | Path,
        *,
        reporting_seeds: tuple[int, ...],
        model_name: str,
        selection_mode: str = "reconstructed_mean_seed",
        lag_spec_scope: str = "all",
        screening_mode: str = "none",
        artifact_stem: str = "shared_mean_seed",
    ) -> dict[str, pd.DataFrame]:
        """Run a VMD variant with one FFNN/lag structure shared across seeds.

        This entry point is used by the block-proposed paper profile for fixed,
        target-selected, all-block, component-wise block-subset, and sequential
        benchmark variants.  Candidate signatures never include the seed.
        """
        seeds = tuple(dict.fromkeys(int(seed) for seed in reporting_seeds))
        selection_mode = selection_mode.replace("-", "_")
        lag_spec_scope = lag_spec_scope.replace("-", "_")
        screening_mode = screening_mode.replace("-", "_")
        if not seeds:
            raise ValueError("Shared mean-seed selection requires reporting seeds.")
        if selection_mode not in {
            "component_mean_seed",
            "reconstructed_mean_seed",
        }:
            raise ValueError(
                "selection_mode must be 'component_mean_seed' or "
                "'reconstructed_mean_seed'."
            )
        if lag_spec_scope not in {"all", "all_blocks_only"}:
            raise ValueError("lag_spec_scope must be 'all' or 'all_blocks_only'.")
        if screening_mode not in {"none", "rolling_mean_seed"}:
            raise ValueError(
                "screening_mode must be 'none' or 'rolling_mean_seed'."
            )
        if self.config.ffnn.search_strategy.replace("-", "_") != "full_grid":
            raise ValueError("Shared mean-seed variants require FFNN full_grid search.")
        if tuple(self.config.ffnn.seed_grid) != seeds:
            raise ValueError("FFNN seed_grid must exactly match reporting_seeds.")

        total_started = perf_counter()
        self._lag_screen_fold_rows = []
        self._lag_screen_stability_rows = []
        self._log(
            f"Starting shared mean-seed VMD variant model={model_name}; "
            f"selection={selection_mode}; screening={screening_mode}; seeds={seeds}."
        )
        base_data = self.load_data(csv_path)
        base_train, base_val, base_test = self.split_raw_data(base_data)
        data = self._apply_stationarity_transform(base_data, base_train)
        train = data.loc[data.index.intersection(base_train.index)].copy()
        val = data.loc[data.index.intersection(base_val.index)].copy()
        test = data.loc[data.index.intersection(base_test.index)].copy()

        audit: list[dict[str, Any]] = []
        for split, frame in (("train", train), ("validation", val), ("test", test)):
            audit.extend(
                self._audit_rows(
                    artifact_stem,
                    "chronological_split",
                    split,
                    frame,
                )
            )

        vmd_cache_started = perf_counter()
        train_components = self.decompose(train)
        val_cache = self._make_origin_cache(train, val, f"{artifact_stem}-validation")
        val_actual_components = self._make_actual_component_cache(
            train,
            val,
            f"{artifact_stem}-validation",
        )
        vmd_cache_seconds = perf_counter() - vmd_cache_started

        selection_started = perf_counter()
        self._select_specs(train_components)
        if lag_spec_scope == "all_blocks_only":
            for component, specs in self.selected_specs_.items():
                all_block_specs = [
                    spec
                    for spec in specs
                    if all(tuple(spec.exog_lags.get(feature, ())) for feature in self.features)
                ]
                if len(all_block_specs) != 1:
                    raise ValueError(
                        "AIC all-block ablation requires exactly one all-block "
                        f"LagSpec for component={component}; got {len(all_block_specs)}."
                    )
                self.selected_specs_[component] = all_block_specs
        ardl_selection_seconds = perf_counter() - selection_started

        rolling_contexts = (
            self._prepare_vmd_lag_screen_contexts(train)
            if screening_mode == "rolling_mean_seed"
            else []
        )
        if screening_mode == "rolling_mean_seed" and not rolling_contexts:
            raise ValueError("No train-only rolling contexts are available for shared screening.")

        ffnn_search_started = perf_counter()
        candidate_frames: list[pd.DataFrame] = []
        component_ranking_frames: list[pd.DataFrame] = []
        retained_predictions: dict[int, pd.DataFrame] = {}
        screen_detail_frames: list[pd.DataFrame] = []
        screen_summary_frames: list[pd.DataFrame] = []
        candidate_id = 0
        for component, train_component in train_components.items():
            specs = self.selected_specs_[component]
            if screening_mode == "rolling_mean_seed" and len(specs) > 1:
                specs, screen_detail, screen_summary = (
                    self._fast_screen_specs_rolling_mean_seed(
                        component=component,
                        specs=specs,
                        seeds=seeds,
                        evaluate=lambda spec, learning_rate, seed, component=component: self._screen_vmd_lag_spec(
                            component,
                            spec,
                            learning_rate,
                            rolling_contexts,
                            seed=seed,
                        ),
                    )
                )
                screen_detail_frames.append(screen_detail)
                screen_summary_frames.append(screen_summary)

            rows, predictions, _, candidate_id = self._ffnn_search_candidates(
                component=component,
                train_data=train_component,
                specs=specs,
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
                scoring_fn=lambda prediction: self._score_component_prediction(
                    prediction,
                    val,
                ),
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
            retained_candidate_ids = set(
                ranked_candidates.loc[
                    ranked_candidates["shortlisted_for_reconstructed_search"],
                    "candidate_id",
                ].astype(int)
            )
            retained_predictions.update(
                {
                    candidate: predictions[candidate]
                    for candidate in retained_candidate_ids
                }
            )

        candidates_df = pd.concat(candidate_frames, ignore_index=True, sort=False)
        component_rankings_df = pd.concat(
            component_ranking_frames,
            ignore_index=True,
            sort=False,
        )
        if selection_mode == "component_mean_seed":
            component_rankings_df["shortlisted_for_reconstructed_search"] = (
                component_rankings_df["component_mean_seed_rank"].eq(1)
            )
            selected_signature_keys = set(
                zip(
                    component_rankings_df.loc[
                        component_rankings_df["shortlisted_for_reconstructed_search"],
                        "component",
                    ].astype(str),
                    component_rankings_df.loc[
                        component_rankings_df["shortlisted_for_reconstructed_search"],
                        "component_signature_id",
                    ].astype(str),
                    strict=True,
                )
            )
            candidates_df["shortlisted_for_reconstructed_search"] = [
                (str(component), str(signature_id)) in selected_signature_keys
                for component, signature_id in zip(
                    candidates_df["component"],
                    candidates_df["component_signature_id"],
                    strict=True,
                )
            ]

        selection_df, reconstructed_seed_scores = (
            self._score_mean_seed_reconstructed_combinations(
                candidates_df,
                component_rankings_df,
                retained_predictions,
                val,
                seeds,
                tuple(train_components),
            )
        )
        selection_protocol = (
            "component_mean_validation_across_reporting_seeds_then_reconstruct"
            if selection_mode == "component_mean_seed"
            else "mean_reconstructed_level_validation_across_reporting_seeds"
        )
        selection_df["selection_protocol"] = selection_protocol
        selection_df["shared_selection_mode"] = selection_mode
        selection_df["shared_screening_mode"] = screening_mode
        reconstructed_seed_scores["shared_selection_mode"] = selection_mode
        candidates_df["shared_selection_mode"] = selection_mode
        candidates_df["shared_screening_mode"] = screening_mode
        component_rankings_df["shared_selection_mode"] = selection_mode

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
                "Shared VMD winner does not cover every component and reporting seed."
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
        best_candidates["selection_protocol"] = selection_protocol
        ffnn_search_seconds = perf_counter() - ffnn_search_started

        refit_started = perf_counter()
        history_before_test = pd.concat([train, val], axis=0)
        history_components = self.decompose(history_before_test)
        test_cache = self._make_origin_cache(
            history_before_test,
            test,
            f"{artifact_stem}-test",
        )
        test_actual_components = self._make_actual_component_cache(
            history_before_test,
            test,
            f"{artifact_stem}-test",
        )
        selected_component_predictions: list[pd.DataFrame] = []
        for row_index, best in best_candidates.iterrows():
            component = str(best["component"])
            seed = int(best["seed"])
            spec = LagSpec(
                tuple(best["target_lags"]),
                {
                    feature: tuple(lags)
                    for feature, lags in best["exog_lags"].items()
                },
            )
            validation_prediction = retained_predictions[
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
        self.final_forecasts_ = self._reconstruct(
            self.component_forecasts_,
            pd.concat([val, test], axis=0),
        ).reset_index(drop=True)
        self.final_forecasts_["forecast_protocol"] = self.config.forecast_protocol

        metric_frames: list[pd.DataFrame] = []
        for seed in seeds:
            index = self.final_forecasts_.index[
                self.final_forecasts_["seed"].eq(seed)
            ]
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
                model_name,
                True,
                seed_best,
            )
            metrics["seed"] = seed
            metric_frames.append(metrics)
            for column in ("actual_level", "predicted_level"):
                if column in seed_forecasts:
                    self.final_forecasts_.loc[index, column] = seed_forecasts[
                        column
                    ].to_numpy()

        self.final_metrics_ = pd.concat(metric_frames, ignore_index=True, sort=False)
        self.best_component_models_ = best_candidates.sort_values(
            ["seed", "component"]
        ).reset_index(drop=True)
        self.input_audit_ = pd.DataFrame(audit)
        self.lag_screen_fold_metrics_ = pd.DataFrame(self._lag_screen_fold_rows)
        self.lag_screen_stability_ = pd.DataFrame(self._lag_screen_stability_rows)
        refit_forecast_seconds = perf_counter() - refit_started

        ardl_orders = pd.concat(
            [
                table.assign(component=component)
                for component, table in self.order_tables_.items()
            ],
            ignore_index=True,
        )
        diagnostics = self._diagnostics(
            train,
            train_components,
            f"_{artifact_stem}",
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
        screen_detail_df = (
            pd.concat(screen_detail_frames, ignore_index=True, sort=False)
            if screen_detail_frames
            else pd.DataFrame()
        )
        screen_summary_df = (
            pd.concat(screen_summary_frames, ignore_index=True, sort=False)
            if screen_summary_frames
            else pd.DataFrame()
        )

        write_csv(candidates_df, self.output_dir / "ffnn_shared_component_search.csv")
        write_csv(
            component_rankings_df,
            self.output_dir / "ffnn_shared_component_mean_seed_ranking.csv",
        )
        write_csv(
            reconstructed_seed_scores,
            self.output_dir / "ffnn_shared_reconstructed_seed_scores.csv",
        )
        write_csv(selection_df, self.output_dir / "ffnn_shared_mean_seed_selection.csv")
        if not screen_detail_df.empty:
            write_csv(screen_detail_df, self.output_dir / "ffnn_shared_screen_metrics.csv")
        if not screen_summary_df.empty:
            write_csv(screen_summary_df, self.output_dir / "ffnn_shared_screen_summary.csv")
        write_csv(
            self.best_component_models_,
            self.output_dir / "best_component_models_shared_mean_seed.csv",
        )
        write_csv(
            self.component_forecasts_,
            self.output_dir / "component_predictions_shared_mean_seed.csv",
        )
        write_csv(
            self.final_forecasts_,
            self.output_dir / "final_reconstructed_forecasts_shared_mean_seed.csv",
        )
        write_csv(
            self.final_metrics_,
            self.output_dir / "final_reconstructed_metrics_shared_mean_seed.csv",
        )
        write_csv(ardl_orders, self.output_dir / "ardl_orders_train_only.csv")
        write_csv(self.input_audit_, self.output_dir / "input_audit_shared_mean_seed.csv")
        if self.stationarity_screen_ is not None:
            write_csv(
                self.stationarity_screen_,
                self.output_dir / "stationarity_screen_train_only_shared_mean_seed.csv",
            )
        for name, frame in diagnostics.items():
            write_csv(frame, self.output_dir / f"{name}_train_only.csv")
        write_csv(timing, self.output_dir / "runtime_breakdown.csv")

        return {
            "search_results": selection_df,
            "component_search_results": candidates_df,
            "component_signature_rankings": component_rankings_df,
            "reconstructed_seed_scores": reconstructed_seed_scores,
            "shared_screen_metrics": screen_detail_df,
            "shared_screen_summary": screen_summary_df,
            "ardl_orders": ardl_orders,
            "best_component_models": self.best_component_models_,
            "component_forecasts": self.component_forecasts_,
            "training_components": train_components,
            "final_forecasts": self.final_forecasts_,
            "final_metrics": self.final_metrics_,
            "input_audit": self.input_audit_,
            "stationarity_screen": self.stationarity_screen_,
            "lag_screen_fold_metrics": self.lag_screen_fold_metrics_,
            "lag_screen_stability": self.lag_screen_stability_,
            "timing": timing,
            **diagnostics,
        }

    def run_no_vmd_shared_mean_seed(
        self,
        csv_path: str | Path,
        *,
        reporting_seeds: tuple[int, ...],
        model_name: str,
        screening_mode: str = "none",
        artifact_stem: str = "no_vmd_shared_mean_seed",
    ) -> dict[str, pd.DataFrame]:
        """Run one no-VMD model with a validation winner shared across seeds."""
        seeds = tuple(dict.fromkeys(int(seed) for seed in reporting_seeds))
        screening_mode = screening_mode.replace("-", "_")
        if not seeds:
            raise ValueError("Shared mean-seed selection requires reporting seeds.")
        if screening_mode not in {"none", "rolling_mean_seed"}:
            raise ValueError(
                "screening_mode must be 'none' or 'rolling_mean_seed'."
            )
        if self.config.ffnn.search_strategy.replace("-", "_") != "full_grid":
            raise ValueError("Shared mean-seed variants require FFNN full_grid search.")
        if tuple(self.config.ffnn.seed_grid) != seeds:
            raise ValueError("FFNN seed_grid must exactly match reporting_seeds.")

        total_started = perf_counter()
        self._lag_screen_fold_rows = []
        self._lag_screen_stability_rows = []
        base_data = self.load_data(csv_path)
        base_train, base_val, base_test = self.split_raw_data(base_data)
        data = self._apply_stationarity_transform(base_data, base_train)
        train = data.loc[data.index.intersection(base_train.index)].copy()
        val = data.loc[data.index.intersection(base_val.index)].copy()
        test = data.loc[data.index.intersection(base_test.index)].copy()
        component = "raw_no_vmd"

        audit: list[dict[str, Any]] = []
        for split, frame in (("train", train), ("validation", val), ("test", test)):
            audit.extend(
                self._audit_rows(
                    artifact_stem,
                    "chronological_split",
                    split,
                    frame,
                )
            )

        selection_started = perf_counter()
        self._select_specs({component: train})
        ardl_selection_seconds = perf_counter() - selection_started
        specs = self.selected_specs_[component]
        screen_detail = pd.DataFrame()
        screen_summary = pd.DataFrame()
        if screening_mode == "rolling_mean_seed" and len(specs) > 1:
            folds = self._expanding_lag_screen_splits(train)
            if not folds:
                raise ValueError(
                    "No train-only rolling folds are available for shared screening."
                )
            specs, screen_detail, screen_summary = (
                self._fast_screen_specs_rolling_mean_seed(
                    component=component,
                    specs=specs,
                    seeds=seeds,
                    evaluate=lambda spec, learning_rate, seed: self._screen_no_vmd_lag_spec(
                        spec,
                        learning_rate,
                        folds,
                        component,
                        seed=seed,
                    ),
                )
            )

        ffnn_search_started = perf_counter()
        observed_val = pd.concat([train, val], axis=0)
        rows, validation_predictions, _, _ = self._ffnn_search_candidates(
            component=component,
            train_data=train,
            specs=specs,
            prediction_fn=lambda model, spec, row_id: self._batch_predict_observed(
                model,
                observed_val,
                val.index,
                spec,
                "validation",
                component,
            ),
            scoring_fn=lambda prediction: self._score_raw_prediction_on_both_scales(
                prediction,
                "actual",
                "predicted",
            ),
            uses_vmd=False,
        )
        candidates = pd.DataFrame(rows)
        if candidates.empty:
            raise ValueError("No no-VMD shared candidate could be trained.")
        signature_columns = [
            "lag_spec",
            "hidden_layer_sizes",
            "alpha",
            "activation",
            "learning_rate_init",
        ]
        ranking = (
            candidates.groupby(signature_columns, as_index=False, dropna=False)
            .agg(
                mean_val_level_RMSE=("Val Level RMSE", "mean"),
                std_val_level_RMSE=("Val Level RMSE", "std"),
                mean_val_level_MAE=("Val Level MAE", "mean"),
                mean_val_level_MAPE=("Val Level MAPE", "mean"),
                seed_count=("seed", "nunique"),
            )
        )
        ranking = ranking[ranking["seed_count"].eq(len(seeds))].copy()
        if ranking.empty:
            raise ValueError("No complete no-VMD signature covers every reporting seed.")
        ranking["std_val_level_RMSE"] = ranking["std_val_level_RMSE"].fillna(0.0)
        ranking = ranking.sort_values(
            [
                "mean_val_level_RMSE",
                "mean_val_level_MAE",
                "mean_val_level_MAPE",
                "lag_spec",
                "hidden_layer_sizes",
                "alpha",
                "learning_rate_init",
            ]
        ).reset_index(drop=True)
        ranking.insert(0, "mean_seed_rank", np.arange(1, len(ranking) + 1))
        ranking["selected"] = ranking["mean_seed_rank"].eq(1)
        ranking["selection_protocol"] = (
            "mean_level_validation_across_reporting_seeds"
        )
        winning_signature = ranking.iloc[0]

        def no_vmd_signature_key(row: pd.Series) -> tuple[Any, ...]:
            return (
                str(row["lag_spec"]),
                self._hidden_layer_sizes_from_row(row),
                float(row["alpha"]),
                str(row["activation"]),
                float(row["learning_rate_init"]),
            )

        signature_key = no_vmd_signature_key(winning_signature)
        selected_mask = pd.Series(
            [
                no_vmd_signature_key(row) == signature_key
                for _, row in candidates.iterrows()
            ],
            index=candidates.index,
        )
        best_candidates = candidates[selected_mask].copy()
        if best_candidates["seed"].nunique() != len(seeds):
            raise ValueError("Shared no-VMD winner does not cover every reporting seed.")
        candidates["selected_shared_mean_seed"] = selected_mask
        best_candidates["selected_shared_mean_seed"] = True
        best_candidates["selection_seed_count"] = len(seeds)
        best_candidates["selection_protocol"] = (
            "mean_level_validation_across_reporting_seeds"
        )
        best_candidates["mean_seed_val_level_RMSE"] = float(
            winning_signature["mean_val_level_RMSE"]
        )
        ffnn_search_seconds = perf_counter() - ffnn_search_started

        refit_started = perf_counter()
        selected_forecasts: list[pd.DataFrame] = []
        history = pd.concat([train, val], axis=0)
        observed_test = pd.concat([train, val, test], axis=0)
        for row_index, best in best_candidates.iterrows():
            seed = int(best["seed"])
            spec = LagSpec(
                tuple(best["target_lags"]),
                {
                    feature: tuple(lags)
                    for feature, lags in best["exog_lags"].items()
                },
            )
            validation_prediction = validation_predictions[
                int(best["candidate_id"])
            ].rename(
                columns={
                    "actual": "actual_raw_transformed",
                    "predicted": "predicted_raw_transformed",
                }
            )
            validation_prediction["seed"] = seed
            selected_forecasts.append(validation_prediction)

            test_model, n_refit = self._fit_model(
                history,
                spec,
                self._hidden_layer_sizes_from_row(best),
                float(best["alpha"]),
                seed,
                str(best["activation"]),
                learning_rate_init=float(best["learning_rate_init"]),
            )
            test_prediction = self._batch_predict_observed(
                test_model,
                observed_test,
                test.index,
                spec,
                "test",
                component,
            ).rename(
                columns={
                    "actual": "actual_raw_transformed",
                    "predicted": "predicted_raw_transformed",
                }
            )
            test_prediction["seed"] = seed
            selected_forecasts.append(test_prediction)
            best_candidates.loc[row_index, "n_refit_train_val"] = int(n_refit)

        self.final_forecasts_ = pd.concat(
            selected_forecasts,
            ignore_index=True,
            sort=False,
        ).sort_values(["seed", "split", "date"]).reset_index(drop=True)
        self.final_forecasts_["forecast_protocol"] = self.config.forecast_protocol
        metric_frames: list[pd.DataFrame] = []
        for seed in seeds:
            index = self.final_forecasts_.index[
                self.final_forecasts_["seed"].eq(seed)
            ]
            seed_forecasts = self.final_forecasts_.loc[index].copy()
            seed_best = best_candidates[best_candidates["seed"].eq(seed)].copy()
            metrics = self._annotate_final_metrics(
                self._metrics_rows(
                    seed_forecasts,
                    "actual_raw_transformed",
                    "predicted_raw_transformed",
                    "raw_transformed",
                ),
                csv_path,
                model_name,
                False,
                seed_best,
            )
            metrics["seed"] = seed
            metric_frames.append(metrics)
            for column in ("actual_level", "predicted_level"):
                if column in seed_forecasts:
                    self.final_forecasts_.loc[index, column] = seed_forecasts[
                        column
                    ].to_numpy()

        self.final_metrics_ = pd.concat(metric_frames, ignore_index=True, sort=False)
        self.best_component_models_ = best_candidates.sort_values("seed").reset_index(
            drop=True
        )
        self.input_audit_ = pd.DataFrame(audit)
        self.lag_screen_fold_metrics_ = pd.DataFrame(self._lag_screen_fold_rows)
        self.lag_screen_stability_ = pd.DataFrame(self._lag_screen_stability_rows)
        refit_forecast_seconds = perf_counter() - refit_started
        ardl_orders = pd.concat(
            [
                table.assign(component=name)
                for name, table in self.order_tables_.items()
            ],
            ignore_index=True,
        )
        diagnostics = self._diagnostics(
            train,
            {component: train},
            f"_{artifact_stem}",
        )
        timing = pd.DataFrame(
            [
                {
                    "ardl_selection_seconds": ardl_selection_seconds,
                    "vmd_cache_seconds": 0.0,
                    "ffnn_search_seconds": ffnn_search_seconds,
                    "refit_forecast_seconds": refit_forecast_seconds,
                    "total_runtime_seconds": perf_counter() - total_started,
                }
            ]
        )

        write_csv(candidates, self.output_dir / "ffnn_shared_validation_search.csv")
        write_csv(ranking, self.output_dir / "ffnn_shared_mean_seed_ranking.csv")
        if not screen_detail.empty:
            write_csv(screen_detail, self.output_dir / "ffnn_shared_screen_metrics.csv")
        if not screen_summary.empty:
            write_csv(screen_summary, self.output_dir / "ffnn_shared_screen_summary.csv")
        write_csv(
            self.best_component_models_,
            self.output_dir / "best_model_shared_mean_seed.csv",
        )
        write_csv(
            self.final_forecasts_,
            self.output_dir / "final_forecasts_shared_mean_seed.csv",
        )
        write_csv(
            self.final_metrics_,
            self.output_dir / "final_metrics_shared_mean_seed.csv",
        )
        write_csv(ardl_orders, self.output_dir / "ardl_orders_train_only.csv")
        write_csv(self.input_audit_, self.output_dir / "input_audit_shared_mean_seed.csv")
        if self.stationarity_screen_ is not None:
            write_csv(
                self.stationarity_screen_,
                self.output_dir / "stationarity_screen_train_only_shared_mean_seed.csv",
            )
        for name, frame in diagnostics.items():
            write_csv(frame, self.output_dir / f"{name}_train_only.csv")
        write_csv(timing, self.output_dir / "runtime_breakdown.csv")

        return {
            "search_results": candidates,
            "mean_seed_rankings": ranking,
            "ardl_orders": ardl_orders,
            "best_component_models": self.best_component_models_,
            "final_forecasts": self.final_forecasts_,
            "final_metrics": self.final_metrics_,
            "input_audit": self.input_audit_,
            "stationarity_screen": self.stationarity_screen_,
            "lag_screen_fold_metrics": self.lag_screen_fold_metrics_,
            "lag_screen_stability": self.lag_screen_stability_,
            "timing": timing,
            **diagnostics,
        }
