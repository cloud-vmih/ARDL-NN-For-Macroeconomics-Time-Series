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
        df = df.sort_values(self.config.date_col).drop_duplicates(self.config.date_col)
        df = df.set_index(self.config.date_col)
        if self.config.freq:
            df = df.asfreq(self.config.freq)

        cols = [self.config.target, *self.config.features]
        out = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
        if self.config.log_transform:
            non_positive = [col for col in cols if (out[col] <= 0).any()]
            if non_positive:
                raise ValueError(f"log_transform=True requires positive values: {non_positive}")
            out = np.log(out)
        return out
