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
    log_transform: bool = False


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
    target_lag_strategy: str = "validation_screen"
    fixed_target_lags: tuple[int, ...] = (1, 12)
    target_lag_top_n: int = 2
    force_target_lag_1: bool = True
    target_lag_preset: str = "acf_pacf"
    target_lag_candidates: tuple[tuple[int, ...], ...] = tuple()
    target_lag_acf_pacf_top_n: int = 3
    target_lag_acf_pacf_max_lags_per_set: int = 3
    lag_spec_strategy: str = "staged"
    max_lag_specs: int = 16


@dataclass(frozen=True)
class FFNNConfig:
    """Cấu hình lưới siêu tham số cho mô hình FFNN."""

    hidden_layer_candidates: tuple[int, ...] = (1, 2, 3)
    hidden_width_multipliers: tuple[float, ...] = (1.0, 0.5, 0.25)
    hidden_units_candidates: tuple[int, ...] = tuple()
    alpha_grid: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1)
    seed_grid: tuple[int, ...] = (7, 42, 123)
    activation: str = "tanh"
    activation_grid: tuple[str, ...] = ("relu", "tanh")
    learning_rate_init: float = 0.01
    max_iter: int = 500
    min_train: int = 30
    min_val: int = 8
    min_test: int = 8
    search_strategy: str = "staged_halving"
    fast_hidden_units: int | None = None
    fast_hidden_layers: int = 1
    fast_hidden_width_multiplier: float = 1.0
    fast_alpha: float = 1e-3
    fast_max_iter: int = 150
    top_k_lag_specs: int = 4
    top_component_candidates: int = 3

    def architecture_for(self, hidden_units: int) -> tuple[int, ...]:
        """Trả về kiến trúc legacy một lớp ẩn từ số neuron tuyệt đối."""
        return (max(1, int(hidden_units)),)

    def architectures_for(self, n_features: int) -> tuple[tuple[int, ...], ...]:
        """Sinh lưới kiến trúc theo số feature đầu vào của từng LagSpec."""
        n_features = max(1, int(n_features))
        if self.hidden_units_candidates:
            return tuple(self.architecture_for(hidden_units) for hidden_units in self.hidden_units_candidates)

        architectures: list[tuple[int, ...]] = []
        seen: set[tuple[int, ...]] = set()
        for layer_count in self.hidden_layer_candidates:
            layers = max(1, int(layer_count))
            for multiplier in self.hidden_width_multipliers:
                width = max(1, int(round(n_features * float(multiplier))))
                architecture = tuple([width] * layers)
                if architecture not in seen:
                    seen.add(architecture)
                    architectures.append(architecture)
        return tuple(architectures)

    def fast_architecture_for(self, n_features: int) -> tuple[int, ...]:
        """Kiến trúc nhẹ dùng ở các bước screening nhanh."""
        if self.fast_hidden_units is not None:
            return self.architecture_for(int(self.fast_hidden_units))
        width = max(1, int(round(max(1, int(n_features)) * float(self.fast_hidden_width_multiplier))))
        return tuple([width] * max(1, int(self.fast_hidden_layers)))


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
