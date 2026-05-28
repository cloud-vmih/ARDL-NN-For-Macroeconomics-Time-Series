from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import logging

import pandas as pd
from statsmodels.tsa.ardl import ARDL

from .config import ARDLSelectionConfig
from .features import LagSpec

logger = logging.getLogger(__name__)


@dataclass
class ARDLOrderSelector:
    config: ARDLSelectionConfig = ARDLSelectionConfig()

    @staticmethod
    def _fixed_target_lags() -> tuple[int, ...]:
        # Temporary policy requested for this experiment. This can later be
        # replaced with a target-lag selection routine.
        return (1, 12)

    def _score_single_exog_lag(
        self,
        clean: pd.DataFrame,
        target: str,
        feature: str,
        exog_lag: int,
    ) -> dict[str, object]:
        target_lags = self._fixed_target_lags()
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

    def select(self, data: pd.DataFrame, target: str, features: list[str]) -> tuple[pd.DataFrame, list[LagSpec]]:
        if self.config.ic.lower() not in {"aic", "bic"}:
            raise ValueError("ARDLSelectionConfig.ic must be either 'aic' or 'bic' for lag scoring.")

        clean = data[[target, *features]].dropna()
        rows: list[dict[str, object]] = []
        top_lags_by_feature: dict[str, list[int]] = {}
        target_lags = self._fixed_target_lags()

        if len(clean) <= max(target_lags) + 5:
            raise ValueError(
                f"Not enough observations to fit ARDL with fixed target_lags={target_lags}."
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
                    row = self._score_single_exog_lag(clean, target, feature, int(exog_lag))
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

        specs: list[LagSpec] = []
        for combo_rank, lag_combo in enumerate(
            product(*(top_lags_by_feature[feature] for feature in features)),
            start=1,
        ):
            exog_lags = {
                feature: (int(exog_lag),)
                for feature, exog_lag in zip(features, lag_combo, strict=True)
            }
            spec = LagSpec(target_lags=target_lags, exog_lags=exog_lags)
            combo_score = sum(
                float(
                    next(
                        row["selected_score"]
                        for row in rows
                        if row["feature"] == feature and int(row["exog_lag"]) == exog_lag
                    )
                )
                for feature, exog_lag in zip(features, lag_combo, strict=True)
            )
            rows.append(
                {
                    "feature": "__ffnn_combo__",
                    "combo_rank": combo_rank,
                    "combo_score_sum": combo_score,
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
