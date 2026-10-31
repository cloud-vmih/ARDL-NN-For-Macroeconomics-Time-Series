"""HardFFNNWorkflow methods for the forecasting experiment."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd

from .artifacts import write_csv
from .features import LagSpec
from .hard_ffnn_policy import HardFFNNPolicy, describe_target_imfs


class HardFFNNWorkflowMixin:
    def run_hard_ffnn_comparator(
        self,
        csv_path: str | Path,
        *,
        reporting_seeds: tuple[int, ...],
        model_name: str = "VMD_ARDL_HardConfig_FFNN",
    ) -> dict[str, pd.DataFrame]:
        """Run full lag-combination search with one train-only FFNN config/spec."""
        seeds = tuple(dict.fromkeys(int(seed) for seed in reporting_seeds))
        if not seeds:
            raise ValueError("Hard-FFNN comparator requires reporting seeds.")
        if tuple(self.config.ffnn.seed_grid) != seeds:
            raise ValueError("FFNN seed_grid must exactly match reporting_seeds.")
        ardl = self.config.ardl
        if ardl.target_lag_strategy.replace("-", "_") != "ardl_select_order":
            raise ValueError(
                "Hard-FFNN comparator requires target_lag_strategy="
                "'ardl_select_order'."
            )
        if ardl.exog_lag_strategy.replace("-", "_") != "ardl_screen":
            raise ValueError(
                "Hard-FFNN comparator requires exog_lag_strategy='ardl_screen'."
            )
        if ardl.lag_spec_strategy.replace("-", "_") != "full_product":
            raise ValueError(
                "Hard-FFNN comparator requires lag_spec_strategy='full_product'."
            )
        if int(ardl.top_n) != 3:
            raise ValueError("Hard-FFNN comparator requires exactly top_n=3.")
        if ardl.ic.lower() != "aic":
            raise ValueError("Hard-FFNN comparator requires AIC lag ranking.")

        total_started = perf_counter()
        self._log(
            "Starting train-only frequency/lag hard-FFNN comparator with "
            f"full-product lag search across seeds={seeds}."
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
                    "hard_ffnn",
                    "chronological_split",
                    split,
                    frame,
                )
            )

        vmd_cache_started = perf_counter()
        train_components = self.decompose(train)
        descriptors = describe_target_imfs(train_components, self.target)
        descriptor_by_component = {
            str(row["component"]): row
            for _, row in descriptors.iterrows()
        }
        val_cache = self._make_origin_cache(train, val, "hard-ffnn-validation")
        val_actual_components = self._make_actual_component_cache(
            train,
            val,
            "hard-ffnn-validation",
        )
        vmd_cache_seconds = perf_counter() - vmd_cache_started

        selection_started = perf_counter()
        self._select_specs(train_components)
        ardl_selection_seconds = perf_counter() - selection_started
        policy = HardFFNNPolicy(self.config.hard_ffnn)
        top_component_candidates = int(
            self.config.hard_ffnn.top_component_candidates
        )

        ffnn_search_started = perf_counter()
        candidate_frames: list[pd.DataFrame] = []
        component_ranking_frames: list[pd.DataFrame] = []
        retained_predictions: dict[int, pd.DataFrame] = {}
        assignment_rows: list[dict[str, Any]] = []
        fit_error_rows: list[dict[str, Any]] = []
        candidate_id = 0
        for component, train_component in train_components.items():
            descriptor = descriptor_by_component[component]
            rows: list[dict[str, Any]] = []
            predictions: dict[int, pd.DataFrame] = {}
            specs = self.selected_specs_[component]
            for spec_rank, spec in enumerate(specs, start=1):
                supervised = self._build_supervised(train_component, spec)
                input_dimension = self._lag_spec_input_dimension(spec)
                all_lags = [
                    *spec.target_lags,
                    *(
                        lag
                        for feature_lags in spec.exog_lags.values()
                        for lag in feature_lags
                    ),
                ]
                max_lag = max(int(lag) for lag in all_lags)
                decision = policy.decide(
                    descriptor,
                    input_dimension=input_dimension,
                    max_lag=max_lag,
                    effective_train_samples=len(supervised),
                )
                assignment_rows.append(
                    {
                        **descriptor.to_dict(),
                        **decision.as_row(),
                        "lag_spec_rank": int(spec_rank),
                        "lag_spec": spec.label,
                        "target_lags": spec.target_lags,
                        "exog_lags": spec.exog_lags,
                        "ffnn_config_source": "train_only_frequency_lag_policy",
                    }
                )
                for seed in seeds:
                    current_id = candidate_id
                    try:
                        model, n_train = self._fit_model(
                            train_component,
                            spec,
                            decision.hidden_layer_sizes,
                            decision.alpha,
                            seed,
                            decision.activation,
                            learning_rate_init=decision.learning_rate_init,
                            max_iter=int(self.config.ffnn.max_iter),
                        )
                        prediction = self._predict_component_from_cache(
                            model,
                            component,
                            val_cache,
                            val,
                            spec,
                            "validation",
                            current_id,
                            val_actual_components,
                        )
                        if len(prediction) < int(self.config.ffnn.min_val):
                            raise ValueError(
                                "Hard-FFNN validation prediction has fewer rows "
                                f"than min_val={self.config.ffnn.min_val}."
                            )
                    except (ValueError, KeyError, IndexError) as exc:
                        fit_error_rows.append(
                            {
                                "component": component,
                                "lag_spec": spec.label,
                                "seed": int(seed),
                                "error": str(exc),
                            }
                        )
                        candidate_id += 1
                        continue
                    metrics = self._score_component_prediction(prediction, val)
                    row = {
                        "candidate_id": int(current_id),
                        "component": component,
                        "lag_spec": spec.label,
                        "target_lags": spec.target_lags,
                        "exog_lags": spec.exog_lags,
                        "HR": int(decision.hidden_layer_sizes[0]),
                        "hidden_layers": len(decision.hidden_layer_sizes),
                        "hidden_units_per_layer": int(
                            decision.hidden_layer_sizes[0]
                        ),
                        "hidden_layer_sizes": decision.hidden_layer_sizes,
                        "alpha": float(decision.alpha),
                        "seed": int(seed),
                        "activation": decision.activation,
                        "learning_rate_init": float(
                            decision.learning_rate_init
                        ),
                        "max_iter": int(self.config.ffnn.max_iter),
                        "n_train": int(n_train),
                        "n_val": int(len(prediction)),
                        "input_dimension": int(decision.input_dimension),
                        "max_lag": int(decision.max_lag),
                        "effective_train_samples": int(
                            decision.effective_train_samples
                        ),
                        "input_sample_ratio": float(
                            decision.input_sample_ratio
                        ),
                        "hard_ffnn_regime": decision.regime,
                        "hard_ffnn_policy_version": decision.policy_version,
                        "hard_ffnn_policy_reason": decision.policy_reason,
                        "heavy_lag_load": bool(decision.heavy_lag_load),
                        "ffnn_config_source": "train_only_frequency_lag_policy",
                        "search_strategy": "hard_config_full_lag_product",
                        "search_stage": "hard_config_validation",
                        "selected_for_hyperparam_search": False,
                        "forecast_safe_no_exog_lag0": True,
                        "uses_vmd": True,
                    }
                    row.update(
                        {f"Val {metric}": value for metric, value in metrics.items()}
                    )
                    rows.append(row)
                    predictions[current_id] = prediction
                    candidate_id += 1

            component_candidates = pd.DataFrame(rows)
            if component_candidates.empty:
                raise ValueError(
                    f"No hard-FFNN candidate could be scored for {component}."
                )
            ranked_candidates, component_ranking = (
                self._rank_mean_seed_component_signatures(
                    component_candidates,
                    seeds,
                    top_component_candidates=top_component_candidates,
                )
            )
            candidate_frames.append(ranked_candidates)
            component_ranking_frames.append(component_ranking)
            retained_ids = set(
                ranked_candidates.loc[
                    ranked_candidates["shortlisted_for_reconstructed_search"],
                    "candidate_id",
                ].astype(int)
            )
            retained_predictions.update(
                {
                    retained_id: predictions[retained_id]
                    for retained_id in retained_ids
                }
            )

        candidates_df = pd.concat(candidate_frames, ignore_index=True, sort=False)
        component_rankings_df = pd.concat(
            component_ranking_frames,
            ignore_index=True,
            sort=False,
        )
        assignments = pd.DataFrame(assignment_rows)
        fit_errors = pd.DataFrame(fit_error_rows)
        policy_config = pd.DataFrame([asdict(self.config.hard_ffnn)])
        selection_df, reconstructed_seed_scores = (
            self._score_mean_seed_reconstructed_combinations(
                candidates_df,
                component_rankings_df,
                retained_predictions,
                val,
                seeds,
                tuple(train_components),
                top_component_candidates=top_component_candidates,
            )
        )
        for frame in (
            candidates_df,
            component_rankings_df,
            reconstructed_seed_scores,
            selection_df,
        ):
            frame["ffnn_config_source"] = "train_only_frequency_lag_policy"
            frame["hard_ffnn_policy_version"] = (
                self.config.hard_ffnn.policy_version
            )
            frame["lag_spec_search_strategy"] = "full_product_top3_per_feature"

        winning_combo = selection_df.iloc[0]
        selected_signature_by_component = {
            component: str(winning_combo[f"{component}_signature_id"])
            for component in train_components
        }
        selected_mask = pd.Series(False, index=candidates_df.index)
        for component, signature_id in selected_signature_by_component.items():
            selected_mask |= candidates_df["component"].eq(component) & (
                candidates_df["component_signature_id"].eq(signature_id)
            )
            component_rankings_df.loc[
                component_rankings_df["component"].eq(component)
                & component_rankings_df["component_signature_id"].eq(
                    signature_id
                ),
                "selected_in_reconstructed_winner",
            ] = True
        best_candidates = candidates_df[selected_mask].copy()
        if len(best_candidates) != len(train_components) * len(seeds):
            raise ValueError(
                "Hard-FFNN reconstructed winner does not cover every "
                "component and reporting seed."
            )
        candidates_df["selected_in_reconstructed_winner"] = selected_mask
        best_candidates["selected_in_reconstructed_winner"] = True
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
            "hard-ffnn-test",
        )
        test_actual_components = self._make_actual_component_cache(
            history_before_test,
            test,
            "hard-ffnn-test",
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
                max_iter=int(best["max_iter"]),
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
            seed_best = best_candidates[
                best_candidates["seed"].eq(seed)
            ].copy()
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
            metrics["ffnn_config_source"] = (
                "train_only_frequency_lag_policy"
            )
            metrics["hard_ffnn_policy_version"] = (
                self.config.hard_ffnn.policy_version
            )
            metric_frames.append(metrics)
            for column in ("actual_level", "predicted_level"):
                if column in seed_forecasts:
                    self.final_forecasts_.loc[index, column] = (
                        seed_forecasts[column].to_numpy()
                    )

        self.final_metrics_ = pd.concat(
            metric_frames,
            ignore_index=True,
            sort=False,
        )
        self.best_component_models_ = best_candidates.sort_values(
            ["seed", "component"]
        ).reset_index(drop=True)
        self.input_audit_ = pd.DataFrame(audit)
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
            "_hard_ffnn",
        )
        timing = pd.DataFrame(
            [
                {
                    "ardl_selection_seconds": ardl_selection_seconds,
                    "vmd_cache_seconds": vmd_cache_seconds,
                    "ffnn_search_seconds": ffnn_search_seconds,
                    "refit_forecast_seconds": refit_forecast_seconds,
                    "total_runtime_seconds": perf_counter() - total_started,
                    "evaluated_ffnn_fits": int(
                        len(candidates_df) + len(fit_errors)
                    ),
                    "hard_policy_assignments": int(len(assignments)),
                    "reconstructed_combinations": int(len(selection_df)),
                }
            ]
        )

        write_csv(descriptors, self.output_dir / "hard_ffnn_imf_descriptors.csv")
        write_csv(
            policy_config,
            self.output_dir / "hard_ffnn_policy_config.csv",
        )
        write_csv(assignments, self.output_dir / "hard_ffnn_policy_assignments.csv")
        if not fit_errors.empty:
            write_csv(fit_errors, self.output_dir / "hard_ffnn_fit_errors.csv")
        write_csv(
            candidates_df,
            self.output_dir / "hard_ffnn_component_candidates.csv",
        )
        write_csv(
            component_rankings_df,
            self.output_dir / "hard_ffnn_component_shortlist.csv",
        )
        write_csv(
            reconstructed_seed_scores,
            self.output_dir / "hard_ffnn_reconstructed_seed_scores.csv",
        )
        write_csv(
            selection_df,
            self.output_dir / "hard_ffnn_shared_selection.csv",
        )
        write_csv(
            self.best_component_models_,
            self.output_dir / "best_component_models_hard_config.csv",
        )
        write_csv(
            self.component_forecasts_,
            self.output_dir / "component_predictions_hard_config.csv",
        )
        write_csv(
            self.final_forecasts_,
            self.output_dir / "final_reconstructed_forecasts_hard_config.csv",
        )
        write_csv(
            self.final_metrics_,
            self.output_dir / "final_reconstructed_metrics_hard_config.csv",
        )
        write_csv(
            ardl_orders,
            self.output_dir / "ardl_top3_full_product_train_only.csv",
        )
        write_csv(
            self.input_audit_,
            self.output_dir / "input_audit_hard_config.csv",
        )
        if self.stationarity_screen_ is not None:
            write_csv(
                self.stationarity_screen_,
                self.output_dir / "stationarity_screen_train_only_hard_config.csv",
            )
        for name, frame in diagnostics.items():
            write_csv(frame, self.output_dir / f"{name}_train_only.csv")
        write_csv(timing, self.output_dir / "runtime_breakdown.csv")

        return {
            "search_results": selection_df,
            "component_search_results": candidates_df,
            "component_signature_rankings": component_rankings_df,
            "reconstructed_seed_scores": reconstructed_seed_scores,
            "imf_descriptors": descriptors,
            "hard_ffnn_policy_config": policy_config,
            "hard_ffnn_policy_assignments": assignments,
            "hard_ffnn_fit_errors": fit_errors,
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
