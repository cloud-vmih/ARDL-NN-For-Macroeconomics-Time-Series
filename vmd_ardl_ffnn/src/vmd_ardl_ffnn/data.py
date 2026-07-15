from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DataConfig


@dataclass
class GenericDataLoader:
    """Nạp và chuẩn hóa dữ liệu CSV theo DataConfig."""

    config: DataConfig = DataConfig()

    def load_csv(self, path: str | Path) -> pd.DataFrame:
        """Đọc CSV, kiểm tra cột bắt buộc và trả về dữ liệu mô hình."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        df = pd.read_csv(path)
        required = [self.config.date_col, self.config.target, *self.config.features]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")

        df[self.config.date_col] = pd.to_datetime(df[self.config.date_col], errors="raise")
        duplicated_dates = df[self.config.date_col].duplicated(keep=False)
        if duplicated_dates.any():
            duplicates = df.loc[duplicated_dates, self.config.date_col].dt.strftime("%Y-%m-%d").tolist()
            raise ValueError(f"CSV contains duplicate timestamps: {duplicates[:10]}")
        df = df.sort_values(self.config.date_col)
        df = df.set_index(self.config.date_col)
        if self.config.freq:
            original_index = df.index.copy()
            df = df.asfreq(self.config.freq)
            missing_periods = df.index.difference(original_index)
            if len(missing_periods) and self.config.missing_period_policy == "raise":
                formatted = [pd.Timestamp(value).strftime("%Y-%m-%d") for value in missing_periods[:10]]
                raise ValueError(
                    f"CSV is missing {len(missing_periods)} expected {self.config.freq} periods: {formatted}"
                )

        cols = [self.config.target, *self.config.features]
        out = df[cols].apply(pd.to_numeric, errors="coerce")
        if self.config.missing_period_policy == "drop":
            out = out.dropna()
        missing_counts = out.isna().sum()
        if missing_counts.any() and self.config.missing_value_policy == "raise":
            details = {col: int(count) for col, count in missing_counts.items() if count}
            raise ValueError(f"CSV contains missing/non-numeric model values: {details}")
        if self.config.missing_value_policy == "drop":
            out = out.dropna()
        if self.config.log_transform:
            non_positive = [col for col in cols if (out[col] <= 0).any()]
            if non_positive:
                raise ValueError(f"log_transform=True requires positive values: {non_positive}")
            out = np.log(out)
        return out
