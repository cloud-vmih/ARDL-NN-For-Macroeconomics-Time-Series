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
    missing_period_policy: str = "raise"
    missing_value_policy: str = "raise"

    def __post_init__(self) -> None:
        if not self.target:
            raise ValueError("DataConfig.target must not be empty.")
        if not self.features:
            raise ValueError("DataConfig.features must contain at least one feature.")
        if self.target in self.features:
            raise ValueError("DataConfig.target must not also appear in features.")
        if len(set(self.features)) != len(self.features):
            raise ValueError("DataConfig.features must not contain duplicates.")
        if self.missing_period_policy not in {"raise", "drop"}:
            raise ValueError("missing_period_policy must be 'raise' or 'drop'.")
        if self.missing_value_policy not in {"raise", "drop"}:
            raise ValueError("missing_value_policy must be 'raise' or 'drop'.")


@dataclass(frozen=True)
class VMDConfig:
    """Cấu hình thuật toán phân rã VMD."""

    modes: int = 3
    backend: str = "self_contained"
    alpha: float = 2000.0
    tau: float = 0.0
    dc: bool = False
    init: int = 1
    tolerance: float = 1e-7
    max_iterations: int = 500

    def __post_init__(self) -> None:
        if self.modes < 1:
            raise ValueError("VMDConfig.modes must be >= 1.")
        if self.alpha <= 0:
            raise ValueError("VMDConfig.alpha must be > 0.")
        if self.tolerance <= 0 or self.max_iterations < 2:
            raise ValueError("VMD tolerance must be > 0 and max_iterations must be >= 2.")
        if self.backend.replace("-", "_").lower() not in {"self_contained", "library"}:
            raise ValueError("VMD backend must be 'self_contained' or 'library'.")
        if self.init not in {0, 1, 2}:
            raise ValueError("VMDConfig.init must be 0, 1, or 2.")


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
    selection_failure_policy: str = "raise"
    exog_lag_strategy: str = "ardl_screen"
    fixed_exog_lags: tuple[int, ...] = (1,)

    def __post_init__(self) -> None:
        if self.max_target_lag < 1 or self.max_exog_lag < 1:
            raise ValueError("ARDL maximum lags must be >= 1.")
        if self.top_n < 1 or self.target_lag_top_n < 1 or self.max_lag_specs < 1:
            raise ValueError("ARDL selection counts must be >= 1.")
        if self.ic.lower() not in {"aic", "bic"}:
            raise ValueError("ARDLSelectionConfig.ic must be 'aic' or 'bic'.")
        if self.selection_failure_policy not in {"raise", "fallback"}:
            raise ValueError("selection_failure_policy must be 'raise' or 'fallback'.")
        allowed_strategies = {
            "fixed", "ic_topk", "validation_screen", "joint_validation_screen",
            "ardl_select_order", "ardl_select_order_joint",
        }
        if self.target_lag_strategy.replace("-", "_") not in allowed_strategies:
            raise ValueError(f"Unsupported target_lag_strategy: {self.target_lag_strategy}")
        if self.lag_spec_strategy.replace("-", "_") not in {"staged", "full_product"}:
            raise ValueError("lag_spec_strategy must be 'staged' or 'full_product'.")
        if self.exog_lag_strategy.replace("-", "_") not in {
            "ardl_screen", "ardl_block_subsets", "fixed"
        }:
            raise ValueError(
                "exog_lag_strategy must be 'ardl_screen', 'ardl_block_subsets', or 'fixed'."
            )
        if not self.fixed_exog_lags or any(
            lag < 1 or lag > self.max_exog_lag for lag in self.fixed_exog_lags
        ):
            raise ValueError("fixed_exog_lags must be within 1..max_exog_lag.")


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
    learning_rate_grid: tuple[float, ...] = (0.01, 0.001)
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
    lag_screen_folds: int = 3
    lag_screen_validation_size: int = 12
    lag_screen_learning_rate_grid: tuple[float, ...] = tuple()
    lag_screen_selection_rule: str = "one_standard_error"
    final_selection_scale: str = "level"

    def __post_init__(self) -> None:
        if not self.seed_grid or not self.alpha_grid:
            raise ValueError("FFNN seed_grid and alpha_grid must not be empty.")
        if not self.activation_grid or not self.learning_rate_grid:
            raise ValueError("FFNN activation_grid and learning_rate_grid must not be empty.")
        if self.learning_rate_init <= 0 or any(value <= 0 for value in self.learning_rate_grid):
            raise ValueError("FFNN learning rates must be > 0.")
        allowed = {"relu", "tanh", "logistic", "sigmoid", "identity"}
        if self.activation.lower() not in allowed or any(str(value).lower() not in allowed for value in self.activation_grid):
            raise ValueError(f"FFNN activations must be one of {sorted(allowed)}.")
        if self.max_iter < 1 or self.fast_max_iter < 1:
            raise ValueError("FFNN iteration limits must be >= 1.")
        if min(self.min_train, self.min_val, self.min_test) < 1:
            raise ValueError("FFNN minimum split sizes must be >= 1.")
        if self.lag_screen_folds < 1 or self.lag_screen_validation_size < 1:
            raise ValueError("Lag-screen folds and validation size must be >= 1.")
        if any(value <= 0 for value in self.lag_screen_learning_rate_grid):
            raise ValueError("Lag-screen learning rates must be > 0.")
        if self.lag_screen_selection_rule.replace("-", "_") not in {
            "top_k", "one_standard_error"
        }:
            raise ValueError(
                "lag_screen_selection_rule must be 'top_k' or 'one_standard_error'."
            )
        if self.final_selection_scale.replace("-", "_") not in {"level", "transformed"}:
            raise ValueError("final_selection_scale must be 'level' or 'transformed'.")
        if self.search_strategy.replace("-", "_") not in {"staged_halving", "full_grid"}:
            raise ValueError("FFNN search_strategy must be 'staged_halving' or 'full_grid'.")
        if not self.hidden_units_candidates and (not self.hidden_layer_candidates or not self.hidden_width_multipliers):
            raise ValueError("FFNN architecture search must contain at least one candidate.")

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

    def lag_screen_learning_rates(self) -> tuple[float, ...]:
        """Return the explicit fast-screen grid or reuse the full FFNN grid."""
        values = self.lag_screen_learning_rate_grid or self.learning_rate_grid
        return tuple(dict.fromkeys(float(value) for value in values))


@dataclass(frozen=True)
class DiagnosticsConfig:
    """Cấu hình kiểm định chẩn đoán chuỗi và phần dư."""

    lags: int = 12
    granger_maxlag: int = 2
    alpha: float = 0.05

    def __post_init__(self) -> None:
        if self.lags < 1 or self.granger_maxlag < 1:
            raise ValueError("Diagnostic lags must be >= 1.")
        if not 0 < self.alpha < 1:
            raise ValueError("Diagnostics alpha must be between 0 and 1.")


@dataclass(frozen=True)
class StationarityConfig:
    """Cấu hình kiểm tra và xử lý tính dừng."""

    enabled: bool = True
    alpha: float = 0.05
    min_obs: int = 12

    def __post_init__(self) -> None:
        if not 0 < self.alpha < 1:
            raise ValueError("Stationarity alpha must be between 0 and 1.")
        if self.min_obs < 4:
            raise ValueError("Stationarity min_obs must be >= 4.")


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
    forecast_protocol: str = "rolling_one_step"

    def __post_init__(self) -> None:
        if not 0 < self.train_ratio < 1:
            raise ValueError("train_ratio must be between 0 and 1.")
        if not 0 < self.val_ratio < 1:
            raise ValueError("val_ratio must be between 0 and 1.")
        if self.train_ratio + self.val_ratio >= 1:
            raise ValueError("train_ratio + val_ratio must be < 1.")
        if self.forecast_protocol != "rolling_one_step":
            raise ValueError("Only forecast_protocol='rolling_one_step' is currently supported.")
