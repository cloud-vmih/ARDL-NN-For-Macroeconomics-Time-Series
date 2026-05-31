from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class LagSpec:
    """Mô tả các độ trễ dùng cho target và biến ngoại sinh."""

    target_lags: tuple[int, ...]
    exog_lags: dict[str, tuple[int, ...]]

    @property
    def label(self) -> str:
        """Tạo nhãn đọc được cho cấu hình độ trễ."""
        exog = ", ".join(f"{key}:{list(value)}" for key, value in self.exog_lags.items())
        return f"target:{list(self.target_lags)} | {exog}"


def expand_lags(order: int | list[int] | tuple[int, ...] | None, include_zero: bool = False) -> tuple[int, ...]:
    """Mở rộng bậc trễ thành tuple các lag duy nhất."""
    if order is None:
        return tuple()
    if isinstance(order, (list, tuple)):
        values = [int(lag) for lag in order]
    else:
        start = 0 if include_zero else 1
        values = list(range(start, int(order) + 1))
    return tuple(sorted({lag for lag in values if lag >= 0}))


def build_lagged_design(data: pd.DataFrame, target: str, spec: LagSpec) -> pd.DataFrame:
    """Tạo bảng supervised learning từ dữ liệu và cấu hình lag."""
    out = pd.DataFrame(index=data.index)
    out[target] = data[target]
    for lag in spec.target_lags:
        if lag <= 0:
            continue
        out[f"{target}_lag{lag}"] = data[target].shift(lag)
    for feature, lags in spec.exog_lags.items():
        for lag in lags:
            if lag < 0:
                raise ValueError(f"Lag must be >= 0, got {feature}_lag{lag}.")
            out[f"{feature}_lag{lag}"] = data[feature].shift(lag)
    return out.dropna()


def time_split(data: pd.DataFrame, train_ratio: float, val_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chia dữ liệu theo thứ tự thời gian thành train/validation/test."""
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return data.iloc[:train_end], data.iloc[train_end:val_end], data.iloc[val_end:]
