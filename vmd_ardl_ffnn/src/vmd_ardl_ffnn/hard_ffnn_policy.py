from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil

import numpy as np
import pandas as pd

from .config import HardFFNNPolicyConfig


@dataclass(frozen=True)
class IMFDescriptor:
    """Train-only spectral and roughness metadata for one target IMF."""

    component: str
    observation_count: int
    dominant_frequency: float
    dominant_period: float
    spectral_centroid: float
    spectral_bandwidth: float
    spectral_entropy: float
    roughness: float
    energy: float
    energy_ratio: float

    def as_row(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class HardFFNNDecision:
    """One seed-independent FFNN configuration assigned to one lag spec."""

    regime: str
    hidden_layer_sizes: tuple[int, ...]
    alpha: float
    activation: str
    learning_rate_init: float
    input_dimension: int
    max_lag: int
    effective_train_samples: int
    input_sample_ratio: float
    heavy_lag_load: bool
    policy_version: str
    policy_reason: str

    def as_row(self) -> dict[str, object]:
        return asdict(self)


def describe_target_imfs(
    components: dict[str, pd.DataFrame],
    target: str,
) -> pd.DataFrame:
    """Describe every target IMF using train-only, positive-frequency power."""
    raw_rows: list[dict[str, object]] = []
    for component, frame in components.items():
        values = frame[target].dropna().to_numpy(dtype=float)
        if len(values) == 0:
            raise ValueError(f"Cannot describe empty target IMF: {component}.")
        centered = values - float(np.mean(values))
        spectrum = np.fft.rfft(centered)
        frequencies = np.fft.rfftfreq(len(centered), d=1.0)
        positive = frequencies > 0
        positive_frequency = frequencies[positive]
        power = np.abs(spectrum[positive]) ** 2
        total_power = float(np.sum(power))
        if total_power > 0 and len(power):
            probability = power / total_power
            dominant_frequency = float(
                positive_frequency[int(np.argmax(power))]
            )
            spectral_centroid = float(
                np.sum(positive_frequency * probability)
            )
            spectral_bandwidth = float(
                np.sqrt(
                    np.sum(
                        ((positive_frequency - spectral_centroid) ** 2)
                        * probability
                    )
                )
            )
            spectral_entropy = (
                float(
                    -np.sum(probability * np.log(probability + 1e-15))
                    / np.log(len(probability))
                )
                if len(probability) > 1
                else 0.0
            )
        else:
            dominant_frequency = 0.0
            spectral_centroid = 0.0
            spectral_bandwidth = 0.0
            spectral_entropy = 0.0
        standard_deviation = float(np.std(values))
        roughness = (
            float(np.std(np.diff(values)) / standard_deviation)
            if len(values) > 1 and standard_deviation > 0
            else 0.0
        )
        raw_rows.append(
            {
                "component": str(component),
                "observation_count": int(len(values)),
                "dominant_frequency": dominant_frequency,
                "dominant_period": (
                    float(1.0 / dominant_frequency)
                    if dominant_frequency > 0
                    else float("inf")
                ),
                "spectral_centroid": spectral_centroid,
                "spectral_bandwidth": spectral_bandwidth,
                "spectral_entropy": spectral_entropy,
                "roughness": roughness,
                "energy": float(np.sum(np.square(values))),
            }
        )
    total_energy = sum(float(row["energy"]) for row in raw_rows)
    rows = [
        {
            **row,
            "energy_ratio": (
                float(row["energy"]) / total_energy if total_energy > 0 else 0.0
            ),
        }
        for row in raw_rows
    ]
    return pd.DataFrame(rows)


class HardFFNNPolicy:
    """Map one train-only IMF descriptor and lag load to one FFNN config."""

    _REGIME_PARAMETERS = {
        "low": {
            "multipliers": (0.5,),
            "activation": "tanh",
            "learning_rate_init": 0.001,
            "alpha": 1e-3,
        },
        "mid": {
            "multipliers": (1.0, 0.5),
            "activation": "tanh",
            "learning_rate_init": 0.001,
            "alpha": 1e-2,
        },
        "high_structured": {
            "multipliers": (0.5, 0.25),
            "activation": "tanh",
            "learning_rate_init": 0.001,
            "alpha": 1e-2,
        },
        "irregular": {
            "multipliers": (0.25,),
            "activation": "tanh",
            "learning_rate_init": 0.001,
            "alpha": 1e-1,
        },
    }

    def __init__(self, config: HardFFNNPolicyConfig) -> None:
        self.config = config

    def _regime(self, descriptor: pd.Series) -> tuple[str, str]:
        component = str(descriptor["component"])
        entropy = float(descriptor["spectral_entropy"])
        roughness = float(descriptor["roughness"])
        centroid = float(descriptor["spectral_centroid"])
        if (
            component.upper() == "RES"
            or entropy >= self.config.high_entropy_min
            or (
                roughness >= self.config.irregular_roughness_min
                and entropy >= self.config.irregular_entropy_min
            )
        ):
            return (
                "irregular",
                "residual_or_high_entropy_roughness",
            )
        if centroid < self.config.low_frequency_max:
            return "low", "spectral_centroid_below_low_threshold"
        if centroid < self.config.mid_frequency_max:
            return "mid", "spectral_centroid_in_mid_band"
        return "high_structured", "high_frequency_but_not_irregular"

    def _width(self, multiplier: float, input_dimension: int) -> int:
        return min(
            self.config.max_width,
            max(self.config.min_width, int(ceil(multiplier * input_dimension))),
        )

    def _promote_alpha(self, alpha: float) -> float:
        levels = self.config.alpha_levels
        for index, value in enumerate(levels):
            if np.isclose(float(value), float(alpha)):
                return float(levels[min(index + 1, len(levels) - 1)])
        larger = [float(value) for value in levels if float(value) > float(alpha)]
        return min(larger) if larger else float(levels[-1])

    def decide(
        self,
        descriptor: pd.Series,
        *,
        input_dimension: int,
        max_lag: int,
        effective_train_samples: int,
    ) -> HardFFNNDecision:
        if input_dimension < 1:
            raise ValueError("Hard FFNN input_dimension must be >= 1.")
        if max_lag < 1:
            raise ValueError("Hard FFNN max_lag must be >= 1.")
        if effective_train_samples < 1:
            raise ValueError("Hard FFNN effective_train_samples must be >= 1.")
        regime, regime_reason = self._regime(descriptor)
        parameters = self._REGIME_PARAMETERS[regime]
        ratio = float(input_dimension / effective_train_samples)
        heavy = ratio > self.config.heavy_input_ratio
        widths = tuple(
            self._width(float(multiplier), input_dimension)
            for multiplier in parameters["multipliers"]
        )
        alpha = float(parameters["alpha"])
        if heavy:
            widths = tuple(
                min(
                    self.config.max_width,
                    max(
                        self.config.min_width,
                        int(ceil(width * self.config.heavy_width_scale)),
                    ),
                )
                for width in widths
            )
            alpha = self._promote_alpha(alpha)
        reason = (
            f"{regime_reason};"
            f"input_ratio={ratio:.6f};"
            f"heavy_adjustment={str(heavy).lower()}"
        )
        return HardFFNNDecision(
            regime=regime,
            hidden_layer_sizes=widths,
            alpha=alpha,
            activation=str(parameters["activation"]),
            learning_rate_init=float(parameters["learning_rate_init"]),
            input_dimension=int(input_dimension),
            max_lag=int(max_lag),
            effective_train_samples=int(effective_train_samples),
            input_sample_ratio=ratio,
            heavy_lag_load=heavy,
            policy_version=self.config.policy_version,
            policy_reason=reason,
        )
