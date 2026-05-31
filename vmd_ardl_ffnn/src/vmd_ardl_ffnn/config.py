from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    """Cấu hình cột dữ liệu đầu vào và biến đổi log."""

    date_col: str = "TIME_PERIOD"
    target: str = "Export_US"
    features: tuple[str, ...] = ("PCE", "US_Retail", "US_Sentiment", "USD_VND", "Import_CN")
    freq: str | None = "MS"
    log_transform: bool = True


@dataclass(frozen=True)
class VMDConfig:
    """Cấu hình thuật toán phân rã VMD."""

    modes: int = 3
    alpha: float = 2000.0
    tau: float = 0.0
    dc: bool = False
    init: int = 1
    tolerance: float = 1e-7
    max_iterations: int = 500


@dataclass(frozen=True)
class ARDLSelectionConfig:
    """Cấu hình chọn độ trễ ARDL cho từng biến ngoại sinh."""

    max_target_lag: int = 12
    max_exog_lag: int = 6
    top_n: int = 3
    ic: str = "aic"
    trend: str = "c"
    causal: bool = True 


@dataclass(frozen=True)
class FFNNConfig:
    """Cấu hình lưới siêu tham số cho mô hình FFNN."""

    hidden_units_candidates: tuple[int, ...] = (1, 4, 8, 12, 16)
    alpha_grid: tuple[float, ...] = (1e-4, 1e-3, 1e-2)
    seed_grid: tuple[int, ...] = (7,)
    activation: str = "relu"
    learning_rate_init: float = 0.01
    max_iter: int = 500
    min_train: int = 30
    min_val: int = 8
    min_test: int = 8

    def architecture_for(self, hidden_units: int) -> tuple[int, ...]:
        """Trả về kiến trúc FFNN một lớp ẩn."""
        return (int(hidden_units),)


@dataclass(frozen=True)
class DiagnosticsConfig:
    """Cấu hình kiểm định chẩn đoán chuỗi và phần dư."""

    lags: int = 12
    granger_maxlag: int = 2
    alpha: float = 0.05


@dataclass(frozen=True)
class StationarityConfig:
    """Cấu hình kiểm tra và xử lý tính dừng."""

    enabled: bool = True
    alpha: float = 0.05
    min_obs: int = 12


@dataclass(frozen=True)
class ExperimentConfig:
    """Cấu hình tổng hợp cho toàn bộ pipeline thực nghiệm."""

    data: DataConfig = field(default_factory=DataConfig)
    vmd: VMDConfig = field(default_factory=VMDConfig)
    ardl: ARDLSelectionConfig = field(default_factory=ARDLSelectionConfig)
    ffnn: FFNNConfig = field(default_factory=FFNNConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    stationarity: StationarityConfig = field(default_factory=StationarityConfig)
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    output_dir: Path = Path("results")
