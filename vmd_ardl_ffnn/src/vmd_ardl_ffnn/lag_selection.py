from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import logging

import numpy as np
import pandas as pd
from statsmodels.tsa.ardl import ARDL
from statsmodels.tsa.stattools import acf, pacf

from .config import ARDLSelectionConfig
from .features import LagSpec

logger = logging.getLogger(__name__)


@dataclass
class ARDLOrderSelector:
    """Chọn tổ hợp độ trễ ARDL để đưa vào lưới FFNN."""

    config: ARDLSelectionConfig = ARDLSelectionConfig()

    def _fallback_target_lags(self) -> tuple[int, ...]:
        """Return validated fixed target lags for fallback/compatibility."""
        max_target_lag = int(self.config.max_target_lag)
        lags = tuple(
            sorted({int(lag) for lag in self.config.fixed_target_lags if 1 <= int(lag) <= max_target_lag})
        )
        if not lags:
            lags = (1,)
        return lags

    def _normalize_target_lag_candidates(
        self,
        raw_candidates: tuple[tuple[int, ...], ...] | list[tuple[int, ...]],
    ) -> list[tuple[int, ...]]:
        """Validate, bound and de-duplicate target-lag candidate sets."""
        max_target_lag = int(self.config.max_target_lag)
        candidates: list[tuple[int, ...]] = []
        seen: set[tuple[int, ...]] = set()
        for candidate in raw_candidates:
            lags = {int(lag) for lag in candidate if 1 <= int(lag) <= max_target_lag}
            if self.config.force_target_lag_1:
                lags.add(1)
            normalized = tuple(sorted(lags))
            if normalized and normalized not in seen:
                candidates.append(normalized)
                seen.add(normalized)
        if not candidates:
            candidates.append(self._fallback_target_lags())
        return candidates

    def _acf_pacf_lag_rows(self, data: pd.DataFrame, target: str) -> list[dict[str, object]]:
        """Score target lags with train-only ACF/PACF for candidate generation."""
        if target not in data:
            raise ValueError(f"Target column {target!r} is required for acf-pacf target-lag preset.")
        series = pd.to_numeric(data[target], errors="coerce").dropna()
        max_target_lag = int(self.config.max_target_lag)
        if len(series) < 6:
            raise ValueError("At least 6 target observations are required for acf-pacf target-lag screening.")
        effective_max_lag = min(max_target_lag, max(1, len(series) // 2 - 1))
        values = series.to_numpy(dtype=float)
        if np.nanstd(values) <= 1e-12:
            raise ValueError("ACF/PACF target-lag screening requires a non-constant target series.")

        threshold = 1.96 / np.sqrt(len(values))
        acf_values = acf(values, nlags=effective_max_lag, fft=False, missing="drop")
        pacf_values = pacf(values, nlags=effective_max_lag, method="ywm")

        rows: list[dict[str, object]] = []
        for lag in range(1, effective_max_lag + 1):
            acf_value = float(acf_values[lag])
            pacf_value = float(pacf_values[lag])
            source: list[str] = []
            if abs(pacf_value) >= threshold:
                source.append("pacf")
            if abs(acf_value) >= threshold:
                source.append("acf")
            if self.config.force_target_lag_1 and lag == 1:
                source.append("forced")
            rows.append(
                {
                    "target_lag": lag,
                    "acf_value": acf_value,
                    "pacf_value": pacf_value,
                    "significance_threshold": float(threshold),
                    "target_lag_source": "+".join(dict.fromkeys(source)) or "ranked_fallback",
                    "pacf_significant": bool(abs(pacf_value) >= threshold),
                    "acf_significant": bool(abs(acf_value) >= threshold),
                    "abs_pacf": abs(pacf_value),
                    "abs_acf": abs(acf_value),
                    "nobs": int(len(values)),
                }
            )
        return rows

    def _acf_pacf_candidate_sets(self, data: pd.DataFrame, target: str) -> list[tuple[int, ...]]:
        """Create compact PACF-core target-lag candidates, with ACF as supplement."""
        rows = self._acf_pacf_lag_rows(data, target)
        top_n = max(1, int(self.config.target_lag_acf_pacf_top_n))
        max_lags_per_set = max(1, int(self.config.target_lag_acf_pacf_max_lags_per_set))

        pacf_ranked = sorted(
            [row for row in rows if row["pacf_significant"]],
            key=lambda row: float(row["abs_pacf"]),
            reverse=True,
        )
        if not pacf_ranked:
            pacf_ranked = sorted(rows, key=lambda row: float(row["abs_pacf"]), reverse=True)
        acf_ranked = sorted(
            [row for row in rows if row["acf_significant"]],
            key=lambda row: float(row["abs_acf"]),
            reverse=True,
        )
        if not acf_ranked:
            acf_ranked = sorted(rows, key=lambda row: float(row["abs_acf"]), reverse=True)

        pacf_lags = [int(row["target_lag"]) for row in pacf_ranked[:top_n]]
        acf_lags = [int(row["target_lag"]) for row in acf_ranked[:top_n]]

        raw_candidates: list[tuple[int, ...]] = [(1,)]
        if pacf_lags:
            raw_candidates.append((1, pacf_lags[0]))
            raw_candidates.append(tuple([1, *pacf_lags[:2]]))
        for acf_lag in acf_lags:
            if acf_lag not in pacf_lags[:2]:
                raw_candidates.append(tuple([1, *pacf_lags[:1], acf_lag]))
                break
        compact = tuple([1, *pacf_lags, *[lag for lag in acf_lags if lag not in pacf_lags]][:max_lags_per_set])
        raw_candidates.append(compact)
        return self._normalize_target_lag_candidates(raw_candidates)

    def target_lag_candidate_metadata(self, data: pd.DataFrame, target: str) -> dict[tuple[int, ...], dict[str, object]]:
        """Return ACF/PACF audit metadata for each generated target-lag candidate."""
        preset = self.config.target_lag_preset.lower().replace("-", "_")
        if preset != "acf_pacf":
            return {}
        rows = self._acf_pacf_lag_rows(data, target)
        rows_by_lag = {int(row["target_lag"]): row for row in rows}
        metadata: dict[tuple[int, ...], dict[str, object]] = {}
        for candidate in self._acf_pacf_candidate_sets(data, target):
            candidate_rows = [rows_by_lag[lag] for lag in candidate if lag in rows_by_lag]
            metadata[candidate] = {
                "target_lag_source": {
                    int(row["target_lag"]): row["target_lag_source"] for row in candidate_rows
                },
                "acf_value": {int(row["target_lag"]): row["acf_value"] for row in candidate_rows},
                "pacf_value": {int(row["target_lag"]): row["pacf_value"] for row in candidate_rows},
                "significance_threshold": float(candidate_rows[0]["significance_threshold"]) if candidate_rows else np.nan,
            }
        return metadata

    def target_lag_candidate_sets(
        self,
        data: pd.DataFrame | None = None,
        target: str | None = None,
    ) -> list[tuple[int, ...]]:
        """Return bounded target-lag candidate sets for validation screening."""
        configured = tuple(self.config.target_lag_candidates)
        if configured:
            return self._normalize_target_lag_candidates(configured)
        preset = self.config.target_lag_preset.lower().replace("-", "_")
        if preset == "acf_pacf":
            if data is None or target is None:
                return [self._fallback_target_lags()]
            try:
                return self._acf_pacf_candidate_sets(data, target)
            except ValueError as exc:
                logger.warning("ACF/PACF target-lag candidate generation failed: %s", exc)
                return [self._fallback_target_lags()]
        if preset == "daily":
            raw_candidates = ((1,), (1, 2), (1, 3), (1, 5), (1, 7), (1, 14), (1, 7, 14), (1, 7, 30))
        elif preset == "yearly":
            raw_candidates = ((1,), (1, 2), (1, 3), (1, 5))
        elif preset == "monthly":
            raw_candidates = ((1,), (1, 2), (1, 3), (1, 6), (1, 12), (1, 3, 12), (1, 6, 12))
        else:
            raise ValueError(
                "ARDLSelectionConfig.target_lag_preset must be 'acf_pacf', 'monthly', 'daily', or 'yearly'."
            )
        return self._normalize_target_lag_candidates(raw_candidates)

    def _score_single_target_lag(self, clean: pd.DataFrame, target: str, target_lag: int) -> dict[str, object]:
        """Score one target autoregressive lag using ARDL's IC."""
        model = ARDL(
            clean[target],
            lags=[int(target_lag)],
            trend=self.config.trend,
            missing="raise",
        ).fit()
        ic_name = self.config.ic.lower()
        return {
            "feature": "__target_lag__",
            "target_lag": int(target_lag),
            "target_lags": (int(target_lag),),
            "target_lag_strategy": self.config.target_lag_strategy.lower().replace("-", "_"),
            "AIC": float(model.aic),
            "BIC": float(model.bic),
            "selected_ic": ic_name,
            "target_lag_score": float(getattr(model, ic_name)),
            "selected_score": float(getattr(model, ic_name)),
            "nobs": int(model.nobs),
            "selected_for_target_lags": False,
            "target_lag_selection_fallback": False,
        }

    def _select_target_lags(self, clean: pd.DataFrame, target: str) -> tuple[tuple[int, ...], list[dict[str, object]]]:
        """Select target lags once before exogenous lag and FFNN search."""
        strategy = self.config.target_lag_strategy.lower().replace("-", "_")
        if strategy not in {"fixed", "ic_topk", "validation_screen"}:
            raise ValueError("ARDLSelectionConfig.target_lag_strategy must be 'fixed', 'ic_topk', or 'validation_screen'.")

        fallback_lags = self._fallback_target_lags()
        if strategy == "fixed":
            return fallback_lags, [
                {
                    "feature": "__target_lag__",
                    "target_lag": int(lag),
                    "target_lags": fallback_lags,
                    "target_lag_strategy": strategy,
                    "selected_ic": self.config.ic.lower(),
                    "selected_for_target_lags": True,
                    "target_lag_selection_fallback": False,
                }
                for lag in fallback_lags
            ]

        effective_strategy = "ic_topk" if strategy == "validation_screen" else strategy
        rows: list[dict[str, object]] = []
        for target_lag in range(1, int(self.config.max_target_lag) + 1):
            try:
                row = self._score_single_target_lag(clean, target, int(target_lag))
                row["target_lag_strategy"] = effective_strategy
                rows.append(row)
            except Exception as exc:
                logger.warning(
                    "ARDL target-lag candidate failed: target=%s target_lag=%d error=%s",
                    target,
                    int(target_lag),
                    exc,
                )
                logger.debug("ARDL target-lag candidate failure traceback", exc_info=True)
                rows.append(
                    {
                        "feature": "__target_lag__",
                        "target_lag": int(target_lag),
                        "target_lags": (int(target_lag),),
                        "target_lag_strategy": effective_strategy,
                        "AIC": float("nan"),
                        "BIC": float("nan"),
                        "selected_ic": self.config.ic.lower(),
                        "target_lag_score": float("nan"),
                        "selected_score": float("nan"),
                        "nobs": 0,
                        "selected_for_target_lags": False,
                        "target_lag_selection_fallback": False,
                        "error": str(exc),
                    }
                )

        valid = [row for row in rows if pd.notna(row["target_lag_score"])]
        if valid:
            ranked = sorted(valid, key=lambda row: float(row["target_lag_score"]))
            selected = {int(row["target_lag"]) for row in ranked[: max(1, int(self.config.target_lag_top_n))]}
            if self.config.force_target_lag_1:
                selected.add(1)
            selected = {lag for lag in selected if 1 <= lag <= int(self.config.max_target_lag)}
            target_lags = tuple(sorted(selected))
            selected_set = set(target_lags)
            for row in rows:
                row["selected_for_target_lags"] = int(row["target_lag"]) in selected_set
                row["target_lags"] = target_lags
            return target_lags, rows

        fallback_rows = []
        for lag in fallback_lags:
            fallback_rows.append(
                {
                    "feature": "__target_lag__",
                    "target_lag": int(lag),
                    "target_lags": fallback_lags,
                    "target_lag_strategy": effective_strategy,
                    "selected_ic": self.config.ic.lower(),
                    "selected_for_target_lags": True,
                    "target_lag_selection_fallback": True,
                    "error": "No target-lag candidate could be scored.",
                }
            )
        return fallback_lags, rows + fallback_rows

    def _score_single_exog_lag(
        self,
        clean: pd.DataFrame,
        target: str,
        feature: str,
        exog_lag: int,
        target_lags: tuple[int, ...],
    ) -> dict[str, object]:
        """Chấm điểm một lag ngoại sinh bằng tiêu chí AIC/BIC."""
        model = ARDL(
            clean[target],
            lags=list(target_lags),
            exog=clean[[feature]],
            order={feature: [int(exog_lag)]},
            trend=self.config.trend,
            missing="raise",
        ).fit()
        ic_name = self.config.ic.lower()
        if ic_name not in {"aic", "bic"}:
            raise ValueError("ARDLSelectionConfig.ic must be either 'aic' or 'bic' for lag scoring.")

        return {
            "feature": feature,
            "exog_lag": int(exog_lag),
            "target_lags": target_lags,
            "AIC": float(model.aic),
            "BIC": float(model.bic),
            "selected_ic": ic_name,
            "selected_score": float(getattr(model, ic_name)),
            "nobs": int(model.nobs),
        }

    def _candidate_lag_combos(
        self,
        features: list[str],
        top_lags_by_feature: dict[str, list[int]],
        scores_by_feature_lag: dict[tuple[str, int], float],
    ) -> list[tuple[tuple[int, ...], float, str]]:
        """Create lag combinations under the configured search budget."""
        strategy = self.config.lag_spec_strategy.lower().replace("-", "_")
        max_lag_specs = max(1, int(self.config.max_lag_specs))

        all_combos: list[tuple[tuple[int, ...], float, str]] = []
        for lag_combo in product(*(top_lags_by_feature[feature] for feature in features)):
            combo = tuple(int(lag) for lag in lag_combo)
            combo_score = sum(
                float(scores_by_feature_lag[(feature, int(exog_lag))])
                for feature, exog_lag in zip(features, combo, strict=True)
            )
            all_combos.append((combo, combo_score, "full_product"))

        all_combos = sorted(all_combos, key=lambda item: item[1])
        if strategy == "full_product":
            return all_combos
        if strategy != "staged":
            raise ValueError("ARDLSelectionConfig.lag_spec_strategy must be 'staged' or 'full_product'.")

        selected: dict[tuple[int, ...], tuple[tuple[int, ...], float, str]] = {}
        baseline = tuple(int(top_lags_by_feature[feature][0]) for feature in features)
        baseline_score = sum(
            float(scores_by_feature_lag[(feature, lag)])
            for feature, lag in zip(features, baseline, strict=True)
        )
        selected[baseline] = (baseline, baseline_score, "baseline_rank1")

        for feature_index, feature in enumerate(features):
            for lag in top_lags_by_feature[feature][1:]:
                combo = list(baseline)
                combo[feature_index] = int(lag)
                combo_tuple = tuple(combo)
                combo_score = sum(
                    float(scores_by_feature_lag[(combo_feature, combo_lag)])
                    for combo_feature, combo_lag in zip(features, combo_tuple, strict=True)
                )
                selected.setdefault(combo_tuple, (combo_tuple, combo_score, "one_feature_perturbation"))

        for combo, combo_score, _ in all_combos:
            if len(selected) >= max_lag_specs:
                break
            selected.setdefault(combo, (combo, combo_score, "best_combo_score"))

        return sorted(selected.values(), key=lambda item: item[1])[:max_lag_specs]

    def select(
        self,
        data: pd.DataFrame,
        target: str,
        features: list[str],
        target_lags_override: tuple[int, ...] | None = None,
    ) -> tuple[pd.DataFrame, list[LagSpec]]:
        """Chọn top lag cho từng feature và tạo các LagSpec ứng viên."""
        if self.config.ic.lower() not in {"aic", "bic"}:
            raise ValueError("ARDLSelectionConfig.ic must be either 'aic' or 'bic' for lag scoring.")

        clean = data[[target, *features]].dropna()
        rows: list[dict[str, object]] = []
        top_lags_by_feature: dict[str, list[int]] = {}
        scores_by_feature_lag: dict[tuple[str, int], float] = {}
        if target_lags_override is None:
            target_lags, target_lag_rows = self._select_target_lags(clean, target)
        else:
            target_lags = tuple(sorted({int(lag) for lag in target_lags_override if int(lag) >= 1}))
            if not target_lags:
                target_lags = self._fallback_target_lags()
            target_lag_rows = [
                {
                    "feature": "__target_lag__",
                    "target_lag": int(lag),
                    "target_lags": target_lags,
                    "target_lag_strategy": self.config.target_lag_strategy.lower().replace("-", "_"),
                    "selected_ic": self.config.ic.lower(),
                    "selected_for_target_lags": True,
                    "target_lag_selection_fallback": False,
                }
                for lag in target_lags
            ]
        rows.extend(target_lag_rows)

        if len(clean) <= max(target_lags) + 5:
            raise ValueError(
                f"Not enough observations to fit ARDL with target_lags={target_lags}."
            )

        logger.info(
            "Starting ARDL lag selection: target=%s features=%s nobs=%d target_lags=%s max_exog_lag=%d top_n=%d ic=%s",
            target,
            features,
            len(clean),
            target_lags,
            int(self.config.max_exog_lag),
            int(self.config.top_n),
            self.config.ic.lower(),
        )

        candidate_lags = range(1, int(self.config.max_exog_lag) + 1)
        for feature in features:
            logger.info("Scoring ARDL exogenous lags: target=%s feature=%s", target, feature)
            feature_rows: list[dict[str, object]] = []
            for exog_lag in candidate_lags:
                try:
                    row = self._score_single_exog_lag(clean, target, feature, int(exog_lag), target_lags)
                    logger.debug(
                        "ARDL candidate scored: target=%s feature=%s exog_lag=%d AIC=%.6f BIC=%.6f %s=%.6f nobs=%d",
                        target,
                        feature,
                        int(exog_lag),
                        float(row["AIC"]),
                        float(row["BIC"]),
                        str(row["selected_ic"]).upper(),
                        float(row["selected_score"]),
                        int(row["nobs"]),
                    )
                    feature_rows.append(row)
                except Exception as exc:
                    logger.warning(
                        "ARDL candidate failed: target=%s feature=%s exog_lag=%d error=%s",
                        target,
                        feature,
                        int(exog_lag),
                        exc,
                    )
                    logger.debug("ARDL candidate failure traceback", exc_info=True)
                    feature_rows.append(
                        {
                            "feature": feature,
                            "exog_lag": int(exog_lag),
                            "target_lags": target_lags,
                            "AIC": float("nan"),
                            "BIC": float("nan"),
                            "selected_ic": self.config.ic.lower(),
                            "selected_score": float("nan"),
                            "nobs": 0,
                            "error": str(exc),
                        }
                    )

            valid = [row for row in feature_rows if pd.notna(row["selected_score"])]
            if not valid:
                raise ValueError(f"No ARDL lag could be scored for exogenous feature {feature}.")

            ranked = sorted(valid, key=lambda row: float(row["selected_score"]))
            top = ranked[: int(self.config.top_n)]
            top_lags_by_feature[feature] = [int(row["exog_lag"]) for row in top]
            for row in top:
                scores_by_feature_lag[(feature, int(row["exog_lag"]))] = float(row["selected_score"])
            logger.info(
                "Selected ARDL top lags: target=%s feature=%s top_lags=%s scores=%s",
                target,
                feature,
                top_lags_by_feature[feature],
                [float(row["selected_score"]) for row in top],
            )

            top_rank = {int(row["exog_lag"]): rank for rank, row in enumerate(top, start=1)}
            for row in feature_rows:
                exog_lag = int(row["exog_lag"])
                row = dict(row)
                row["rank_within_feature"] = top_rank.get(exog_lag)
                row["selected_for_ffnn_grid"] = exog_lag in top_rank
                rows.append(row)

        lag_combos = self._candidate_lag_combos(features, top_lags_by_feature, scores_by_feature_lag)

        specs: list[LagSpec] = []
        for combo_rank, (lag_combo, combo_score, combo_source) in enumerate(lag_combos, start=1):
            exog_lags = {
                feature: tuple(range(1, int(exog_lag) + 1))
                for feature, exog_lag in zip(features, lag_combo, strict=True)
            }
            spec = LagSpec(target_lags=target_lags, exog_lags=exog_lags)
            rows.append(
                {
                    "feature": "__ffnn_combo__",
                    "combo_rank": combo_rank,
                    "combo_score_sum": combo_score,
                    "combo_source": combo_source,
                    "lag_spec_strategy": self.config.lag_spec_strategy.lower().replace("-", "_"),
                    "target_lags": spec.target_lags,
                    "exog_lags": spec.exog_lags,
                    "lag_spec": spec.label,
                    "selected_ic": self.config.ic.lower(),
                }
            )
            specs.append(spec)
            logger.info(
                "Prepared FFNN lag-spec candidate: target=%s combo_rank=%d combo_score_sum=%.6f lag_spec=%s",
                target,
                combo_rank,
                combo_score,
                spec.label,
            )

        if not specs:
            raise ValueError(f"No ARDL order could be selected for target={target}.")
        logger.info("Completed ARDL lag selection: target=%s n_specs=%d", target, len(specs))
        return pd.DataFrame(rows), specs
