from __future__ import annotations

import numpy as np
import pandas as pd

from vmd_ardl_ffnn.macro_features import (
    FeatureResearchConfig,
    engineer_forecast_safe_features,
    run_feature_screening,
)


def _macro_frame(n: int = 96) -> pd.DataFrame:
    dates = pd.date_range("2010-01-01", periods=n, freq="MS")
    time = np.arange(n, dtype=float)
    return pd.DataFrame(
        {
            "date": dates,
            "unemployment_rate": 8.0 - 0.02 * time + 0.2 * np.sin(time / 4),
            "initial_claims": 300.0 + 2.0 * time,
            "retail_sales": 100.0 + time,
            "unemp_3m_avg": 8.0 - 0.02 * time,
            "sahm_rule_indicator": np.zeros(n),
        }
    )


def test_engineering_uses_only_past_exog_and_excludes_target_derivatives() -> None:
    frame = _macro_frame(30)
    engineered, catalog = engineer_forecast_safe_features(
        frame,
        FeatureResearchConfig(
            base_exog_columns=("initial_claims", "retail_sales"),
            exog_lags=(1, 3),
            rolling_windows=(3,),
        ),
    )

    assert engineered.loc[pd.Timestamp("2010-02-01"), "initial_claims__lag_1"] == 300.0
    assert engineered.loc[pd.Timestamp("2010-04-01"), "initial_claims__lag_3"] == 300.0
    assert not any("unemp_3m_avg" in name for name in catalog["signal"])
    assert not any("sahm" in name for name in catalog["signal"])
    assert catalog["availability_lag"].min() >= 1


def test_screening_is_unchanged_when_locked_test_values_change() -> None:
    frame = _macro_frame()
    config = FeatureResearchConfig(
        base_exog_columns=("initial_claims", "retail_sales"),
        exog_lags=(1, 3),
        rolling_windows=(3,),
        train_end="2015-12-01",
        validation_end="2016-12-01",
        cv_splits=3,
        max_selected_features=2,
    )
    first = run_feature_screening(frame, config)

    changed = frame.copy()
    changed.loc[changed["date"] > config.validation_end, "unemployment_rate"] = 9999.0
    changed.loc[changed["date"] > config.validation_end, "initial_claims"] = -9999.0
    second = run_feature_screening(changed, config)

    pd.testing.assert_frame_equal(first.scores, second.scores)
    assert first.selected_base_features == second.selected_base_features

