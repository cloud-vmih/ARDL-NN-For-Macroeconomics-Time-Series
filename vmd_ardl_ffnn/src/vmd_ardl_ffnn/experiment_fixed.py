from __future__ import annotations

import ast
from dataclasses import asdict, replace
from itertools import product
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Callable
import numpy as np
import pandas as pd

from .config import DataConfig, ExperimentConfig
from .artifacts import write_csv
from .data import GenericDataLoader
from .decomposition import VariationalModeDecomposer
from .diagnostics import ForecastDiagnostics
from .evaluation import compare_two_metric_tables, normalize_forecasts
from .features import LagSpec
from .hard_ffnn_policy import HardFFNNPolicy, describe_target_imfs
from .lag_selection import ARDLOrderSelector
from .metrics import evaluate_forecast
from .models.ffnn import NumpyFFNNRegressor
from .stationarity import apply_stationarity_transform


class VMDARDLFFNNExperiment:
    """Điều phối thực nghiệm VMD/no-VMD + ARDL + FFNN theo thứ tự thời gian.
    Lớp này giữ trạng thái trung gian, chọn lag trên train-only, cache VMD theo
    từng forecast origin và xuất các bảng forecast/metric/audit ra output_dir.
    """

    def __init__(self, config: ExperimentConfig = ExperimentConfig()) -> None:
        """Khởi tạo cấu hình, thư mục kết quả và các bảng trạng thái rỗng.
        Các thuộc tính có hậu tố `_` được điền trong quá trình chạy pipeline.
        """
        self.config = config
        # Tạo thư mục chứa toàn bộ CSV/PNG kết quả trước khi pipeline ghi file.
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Lưu bản dữ liệu đã đọc từ CSV để các bước sau có thể tái sử dụng.
        self.raw_model_data_: pd.DataFrame | None = None
        # Lưu bảng điểm ARDL theo từng component sau bước chọn lag.
        self.order_tables_: dict[str, pd.DataFrame] = {}
        # Lưu các LagSpec đã lọc an toàn để dùng trong dự báo out-of-sample.
        self.selected_specs_: dict[str, list[LagSpec]] = {}
        # Các bảng kết quả này được điền dần sau grid-search và refit.
        self.search_results_: pd.DataFrame | None = None
        self.best_component_models_: pd.DataFrame | None = None
        self.component_forecasts_: pd.DataFrame | None = None
        self.final_forecasts_: pd.DataFrame | None = None
        self.final_metrics_: pd.DataFrame | None = None
        self.input_audit_: pd.DataFrame | None = None
        # Giữ dữ liệu level/log gốc để đảo sai phân hoặc log khi tính metric level.
        self.base_model_data_: pd.DataFrame | None = None
        self.stationarity_screen_: pd.DataFrame | None = None
        # Ghi transform của từng cột: "level" hoặc "diff1".
        self.transform_info_: dict[str, str] = {}
        # Fast-screen lag được giữ riêng khỏi full FFNN validation search để
        # audit được từng rolling fold mà không trộn candidate_id của outer validation.
        self.lag_screen_fold_metrics_: pd.DataFrame | None = None
        self.lag_screen_stability_: pd.DataFrame | None = None
        self._lag_screen_fold_rows: list[dict[str, Any]] = []
        self._lag_screen_stability_rows: list[dict[str, Any]] = []

    @property
    def target(self) -> str:
        """Trả về tên biến mục tiêu được khai báo trong cấu hình dữ liệu."""
        return self.config.data.target

    @property
    def features(self) -> list[str]:
        """Trả về danh sách biến ngoại sinh dùng làm input dự báo."""
        return list(self.config.data.features)

    def _log(self, message: str) -> None:
        """In thông báo tiến trình với prefix thống nhất của package."""
        print(f"[vmd_ardl_ffnn] {message}", flush=True)

    def load_data(self, csv_path: str | Path) -> pd.DataFrame:
        """Đọc CSV, chuẩn hóa theo DataConfig và lưu vào raw_model_data_.
        Hàm đảm bảo index thời gian tăng dần trước khi pipeline tiếp tục.
        """
        self._log(f"Loading data: {csv_path}")
        loader = GenericDataLoader(self.config.data)
        # Loader xử lý parse date, chọn cột, ép numeric và log-transform nếu bật.
        self.raw_model_data_ = loader.load_csv(csv_path).sort_index()
        if not self.raw_model_data_.index.is_monotonic_increasing:
            # Chặn dữ liệu sai thứ tự vì mọi split và lag đều giả định thời gian tăng dần.
            raise ValueError(f"Data must be sorted by {self.config.data.date_col}.")
        return self.raw_model_data_
    
    def split_raw_data(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Chia dữ liệu thành train/validation/test không xáo trộn thời gian.
        Tỷ lệ lấy từ ExperimentConfig và được kiểm tra để mỗi split có dữ liệu.
        """
        n = len(data)
        # Tính ranh giới split bằng index vị trí để không trộn thứ tự thời gian.
        train_end = int(n * self.config.train_ratio)
        val_end = int(n * (self.config.train_ratio + self.config.val_ratio))
        if not (0 < train_end < val_end < n):
            raise ValueError("Invalid chronological train/validation/test split.")
        # Copy từng split để các phép biến đổi sau không ghi ngược vào data gốc.
        train = data.iloc[:train_end].copy()
        val = data.iloc[train_end:val_end].copy()
        test = data.iloc[val_end:].copy()
        self._log(f"Split rows: train={len(train)}, validation={len(val)}, test={len(test)}")
        return train, val, test

    def _apply_stationarity_transform(
        self,
        base_data: pd.DataFrame,
        base_train: pd.DataFrame,
    ) -> pd.DataFrame:
        """Chọn level hoặc sai phân bậc một cho từng biến bằng train-only screen.
        Quyết định biến đổi chỉ dựa vào tập train để tránh leakage; kết quả
        được lưu vào transform_info_ và stationarity_screen_.
        """
        # Lưu dữ liệu trước khi sai phân để sau này có thể đảo dự báo về level.
        self.base_model_data_ = base_data.copy()
        if self.config.stationarity.enabled:
            self._log("Running train-only stationarity screen (ADF/KPSS).")
        result = apply_stationarity_transform(base_data, base_train, self.config.stationarity)
        self.transform_info_ = result.transforms
        self.stationarity_screen_ = result.screen
        return result.transformed

    def decompose(self, data: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Phân rã từng cột dữ liệu thành các component VMD và phần dư."""
        return VariationalModeDecomposer(self.config.vmd).decompose_frame(data)

    def _has_exog_lag_zero(self, spec: LagSpec) -> bool:
        """Kiểm tra LagSpec có chứa lag 0 của biến ngoại sinh hay không."""
        return any(0 in tuple(lags) for lags in spec.exog_lags.values())

    def _safe_spec(self, spec: LagSpec) -> LagSpec:
        """Chuẩn hóa LagSpec để chỉ còn các lag >= 1 dùng được cho forecast.

        Lag 0 bị loại vì tại thời điểm dự báo tương lai không được biết trước
        giá trị đồng thời của biến ngoại sinh.
        """
        # Target lag phải bắt đầu từ 1 để dự báo không nhìn thấy y hiện tại.
        target_lags = tuple(sorted({int(lag) for lag in spec.target_lags if int(lag) >= 1}))
        # Exogenous lag 0 bị loại vì gây leakage khi forecast tương lai.
        exog_lags = {
            feature: tuple(sorted({int(lag) for lag in spec.exog_lags.get(feature, ()) if int(lag) >= 1}))
            for feature in self.features
        }
        safe = LagSpec(target_lags=target_lags, exog_lags=exog_lags)
        if not self._feature_names(safe):
            raise ValueError("Forecast-safe specs require at least one lagged feature.")
        return safe

    def _feature_names(self, spec: LagSpec) -> list[str]:
        """Sinh danh sách tên cột feature theo thứ tự target lag rồi exog lag."""
        names = [f"{self.target}__lag_{lag}" for lag in spec.target_lags]
        for feature in self.features:
            names.extend(f"{feature}__lag_{lag}" for lag in spec.exog_lags.get(feature, ()))
        return names

    def _build_supervised(self, data: pd.DataFrame, spec: LagSpec) -> pd.DataFrame:
        """Tạo bảng supervised gồm y hiện tại và các feature bị trễ.
        Hàm reject mọi lag không an toàn, shift dữ liệu theo LagSpec và bỏ các
        dòng thiếu sinh ra ở đầu chuỗi.
        """
        # Chặn spec có lag 0 trước khi tạo design matrix để bảo toàn causal forecast.
        if any(lag < 1 for lag in spec.target_lags) or self._has_exog_lag_zero(spec):
            raise ValueError("Forecast design only allows target/exogenous lags >= 1.")
        out = pd.DataFrame(index=data.index)
        # Cột target là y tại thời điểm hiện tại cần dự báo.
        out[self.target] = data[self.target].astype(float)
        for lag in spec.target_lags:
            # Target lag dùng lịch sử y làm feature tự hồi quy.
            out[f"{self.target}__lag_{lag}"] = data[self.target].shift(lag)
        for feature in self.features:
            for lag in spec.exog_lags.get(feature, ()):
                # Feature ngoại sinh cũng bị shift để chỉ dùng thông tin quá khứ.
                out[f"{feature}__lag_{lag}"] = data[feature].shift(lag)
        # Bỏ các dòng đầu chuỗi không đủ lịch sử cho lag lớn nhất.
        return out.dropna()

    def _fit_model(
        self,
        data: pd.DataFrame,
        spec: LagSpec,
        hidden_layer_sizes: tuple[int, ...],
        alpha: float,
        seed: int,
        activation: str | None = None,
        learning_rate_init: float | None = None,
        max_iter: int | None = None,
    ) -> tuple[NumpyFFNNRegressor, int]:
        """Fit FFNN cho một LagSpec và trả về model kèm số dòng train hợp lệ.
        Dữ liệu được chuyển sang ma trận supervised trước khi kiểm tra min_train.
        """
        # Chuyển chuỗi thời gian thành X/y theo LagSpec trước khi fit FFNN.
        supervised = self._build_supervised(data, spec)
        if len(supervised) < self.config.ffnn.min_train:
            raise ValueError(f"Only {len(supervised)} training rows after lagging.")
        # FFNN NumPy wrapper tự chuẩn hóa X và y để huấn luyện ổn định hơn.
        model = NumpyFFNNRegressor(
            hidden_layer_sizes=tuple(int(size) for size in hidden_layer_sizes),
            alpha=alpha,
            seed=seed,
            activation=(activation or self.config.ffnn.activation),
            learning_rate_init=(
                self.config.ffnn.learning_rate_init
                if learning_rate_init is None
                else float(learning_rate_init)
            ),
            max_iter=self.config.ffnn.max_iter if max_iter is None else int(max_iter),
        ).fit(supervised[self._feature_names(spec)].to_numpy(float), supervised[self.target].to_numpy(float))
        # Trả kèm số dòng train thật sự sau khi mất dữ liệu do lag.
        return model, len(supervised)

    @staticmethod
    def _hidden_layer_sizes_from_row(row: pd.Series) -> tuple[int, ...]:
        """Khôi phục tuple kiến trúc từ dòng kết quả search."""
        value = row.get("hidden_layer_sizes")
        if isinstance(value, tuple):
            return tuple(max(1, int(size)) for size in value)
        if isinstance(value, list):
            return tuple(max(1, int(size)) for size in value)
        if isinstance(value, str):
            parsed = ast.literal_eval(value)
            if isinstance(parsed, int):
                return (max(1, int(parsed)),)
            return tuple(max(1, int(size)) for size in parsed)
        return (max(1, int(row["HR"])),)

    def _lag_spec_input_dimension(self, spec: LagSpec) -> int:
        """Count target and exogenous lagged inputs in one LagSpec."""
        return len(tuple(spec.target_lags)) + sum(
            len(tuple(lags)) for lags in spec.exog_lags.values()
        )

    def _expanding_lag_screen_splits(
        self,
        data: pd.DataFrame,
    ) -> list[tuple[int, pd.DataFrame, pd.DataFrame]]:
        """Build non-overlapping validation blocks at the end of train-only data.

        The requested fold count is reduced automatically for short datasets,
        but every retained fold must leave enough pre-lag training observations
        and at least ``min_val`` validation timestamps.
        """
        requested_folds = int(self.config.ffnn.lag_screen_folds)
        requested_size = int(self.config.ffnn.lag_screen_validation_size)
        largest_lag = max(
            int(self.config.ardl.max_target_lag),
            int(self.config.ardl.max_exog_lag),
        )
        minimum_history = int(self.config.ffnn.min_train) + largest_lag
        available = len(data) - minimum_history
        if available < int(self.config.ffnn.min_val):
            return []

        validation_size = min(requested_size, available)
        if validation_size < int(self.config.ffnn.min_val):
            validation_size = int(self.config.ffnn.min_val)
        fold_count = min(requested_folds, available // validation_size)
        if fold_count < 1:
            return []

        first_validation_start = len(data) - fold_count * validation_size
        folds: list[tuple[int, pd.DataFrame, pd.DataFrame]] = []
        for fold in range(fold_count):
            validation_start = first_validation_start + fold * validation_size
            validation_end = validation_start + validation_size
            fold_train = data.iloc[:validation_start].copy()
            fold_validation = data.iloc[validation_start:validation_end].copy()
            if (
                len(fold_train) >= minimum_history
                and len(fold_validation) >= int(self.config.ffnn.min_val)
            ):
                folds.append((fold + 1, fold_train, fold_validation))
        return folds

    def _summarize_lag_screen(
        self,
        component: str,
        specs: list[LagSpec],
        fold_rows: list[dict[str, Any]],
    ) -> tuple[set[str], dict[str, float]]:
        """Aggregate rolling folds and shortlist robust lag specifications.

        All lag specs are evaluated before pruning. The raw mean-RMSE winner is
        always retained. Under the one-standard-error rule, the simplest spec
        statistically tied with that winner is retained as well; remaining
        slots are filled by mean RMSE rank.
        """
        if not fold_rows:
            return set(), {}

        fold_frame = pd.DataFrame(fold_rows)
        valid = fold_frame.dropna(subset=["screen_RMSE"]).copy()
        if valid.empty:
            return set(), {}

        grouped = (
            valid.groupby(["lag_spec", "learning_rate_init"], as_index=False)
            .agg(
                rmse_mean=("screen_RMSE", "mean"),
                rmse_std=("screen_RMSE", "std"),
                rmse_count=("screen_RMSE", "count"),
                mae_mean=("screen_MAE", "mean"),
                mape_mean=("screen_MAPE", "mean"),
            )
        )
        grouped["rmse_std"] = grouped["rmse_std"].fillna(0.0)
        grouped["rmse_se"] = grouped["rmse_std"] / np.sqrt(grouped["rmse_count"].clip(lower=1))

        spec_by_label = {spec.label: spec for spec in specs}
        grouped["input_dimension"] = grouped["lag_spec"].map(
            lambda label: self._lag_spec_input_dimension(spec_by_label[str(label)])
        )
        grouped = grouped.sort_values(
            ["lag_spec", "rmse_mean", "mae_mean", "mape_mean", "learning_rate_init"]
        ).reset_index(drop=True)
        grouped["best_learning_rate_for_spec"] = ~grouped.duplicated("lag_spec")
        best_by_spec = grouped[grouped["best_learning_rate_for_spec"]].copy()
        best_by_spec = best_by_spec.sort_values(
            ["rmse_mean", "mae_mean", "mape_mean", "input_dimension", "lag_spec"]
        ).reset_index(drop=True)
        best_by_spec["mean_rmse_rank"] = np.arange(1, len(best_by_spec) + 1)

        raw_best = best_by_spec.iloc[0]
        rule = self.config.ffnn.lag_screen_selection_rule.replace("-", "_")
        threshold = float(raw_best["rmse_mean"])
        if rule == "one_standard_error":
            threshold += float(raw_best["rmse_se"])
        best_by_spec["one_se_threshold"] = threshold
        best_by_spec["within_one_standard_error"] = best_by_spec["rmse_mean"].le(threshold)
        eligible = best_by_spec[best_by_spec["within_one_standard_error"]].copy()
        one_se_winner = eligible.sort_values(
            ["input_dimension", "rmse_mean", "mae_mean", "mape_mean", "lag_spec"]
        ).iloc[0]

        top_k = min(max(1, int(self.config.ffnn.top_k_lag_specs)), len(best_by_spec))
        selected_labels: list[str] = [str(raw_best["lag_spec"])]
        if rule == "one_standard_error":
            selected_labels.append(str(one_se_winner["lag_spec"]))
        for label in best_by_spec["lag_spec"].astype(str):
            if len(set(selected_labels)) >= top_k:
                break
            selected_labels.append(label)
        selected_set = set(selected_labels)

        annotations = best_by_spec.set_index("lag_spec")[
            [
                "mean_rmse_rank",
                "one_se_threshold",
                "within_one_standard_error",
            ]
        ].to_dict("index")
        grouped["mean_rmse_rank"] = grouped["lag_spec"].map(
            lambda label: annotations[str(label)]["mean_rmse_rank"]
        )
        grouped["one_se_threshold"] = grouped["lag_spec"].map(
            lambda label: annotations[str(label)]["one_se_threshold"]
        )
        grouped["within_one_standard_error"] = grouped["lag_spec"].map(
            lambda label: annotations[str(label)]["within_one_standard_error"]
        )
        grouped["one_se_winner"] = grouped["lag_spec"].eq(str(one_se_winner["lag_spec"]))
        grouped["raw_mean_rmse_winner"] = grouped["lag_spec"].eq(str(raw_best["lag_spec"]))
        grouped["selected_for_hyperparam_search"] = grouped["lag_spec"].isin(selected_set)
        grouped.insert(0, "component", component)
        grouped["selection_rule"] = rule
        self._lag_screen_stability_rows.extend(grouped.to_dict("records"))

        best_rate_rows = grouped[grouped["best_learning_rate_for_spec"]]
        mean_rmse_by_label = dict(
            zip(best_rate_rows["lag_spec"], best_rate_rows["rmse_mean"], strict=False)
        )
        return selected_set, {str(label): float(value) for label, value in mean_rmse_by_label.items()}

    def _ffnn_search_candidates(
        self,
        component: str,
        train_data: pd.DataFrame,
        specs: list[LagSpec],
        prediction_fn: Callable[[NumpyFFNNRegressor, LagSpec, int], pd.DataFrame],
        scoring_fn: Callable[[pd.DataFrame], dict[str, float]],
        uses_vmd: bool,
        candidate_id_start: int = 0,
        lag_screen_fn: Callable[[LagSpec, float], list[dict[str, Any]]] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[int, pd.DataFrame], list[int], int]:
        """Run either full-grid or staged FFNN search for one component."""
        strategy = self.config.ffnn.search_strategy.lower().replace("-", "_")
        if strategy not in {"staged_halving", "full_grid"}:
            raise ValueError("FFNNConfig.search_strategy must be 'staged_halving' or 'full_grid'.")

        rows: list[dict[str, Any]] = []
        pred_by_id: dict[int, pd.DataFrame] = {}
        final_candidate_ids: list[int] = []
        candidate_id = int(candidate_id_start)

        def fit_score_candidate(
            spec: LagSpec,
            hidden_layer_sizes: tuple[int, ...],
            alpha: float,
            seed: int,
            activation: str,
            learning_rate_init: float,
            max_iter: int,
            search_stage: str,
            selected_for_hyperparam_search: bool,
            fast_screen_rmse: float | None = None,
        ) -> tuple[int | None, dict[str, Any] | None, pd.DataFrame | None]:
            nonlocal candidate_id
            try:
                model, n_train = self._fit_model(
                    train_data,
                    spec,
                    hidden_layer_sizes,
                    float(alpha),
                    int(seed),
                    activation,
                    learning_rate_init=float(learning_rate_init),
                    max_iter=int(max_iter),
                )
                pred = prediction_fn(model, spec, candidate_id)
            except (ValueError, KeyError, IndexError):
                return None, None, None
            if len(pred) < self.config.ffnn.min_val:
                return None, None, None

            row = {
                "candidate_id": candidate_id,
                "component": component,
                "lag_spec": spec.label,
                "target_lags": spec.target_lags,
                "exog_lags": spec.exog_lags,
                "HR": int(hidden_layer_sizes[0]),
                "hidden_layers": len(hidden_layer_sizes),
                "hidden_units_per_layer": int(hidden_layer_sizes[0]),
                "hidden_layer_sizes": hidden_layer_sizes,
                "alpha": float(alpha),
                "seed": int(seed),
                "activation": activation,
                "learning_rate_init": float(learning_rate_init),
                "max_iter": int(max_iter),
                "n_train": n_train,
                "n_val": len(pred),
                "forecast_safe_no_exog_lag0": True,
                "uses_vmd": bool(uses_vmd),
                "search_strategy": strategy,
                "search_stage": search_stage,
                "selected_for_hyperparam_search": bool(selected_for_hyperparam_search),
                "fast_screen_rmse": fast_screen_rmse,
            }
            row.update({f"Val {k}": v for k, v in scoring_fn(pred).items()})
            current_id = candidate_id
            candidate_id += 1
            return current_id, row, pred

        if strategy == "staged_halving" and len(specs) > 1 and lag_screen_fn is not None:
            component_fold_rows: list[dict[str, Any]] = []
            for spec in specs:
                for learning_rate in self.config.ffnn.lag_screen_learning_rates():
                    screen_rows = lag_screen_fn(spec, float(learning_rate))
                    for screen_row in screen_rows:
                        enriched = {
                            **screen_row,
                            "component": component,
                            "lag_spec": spec.label,
                            "target_lags": spec.target_lags,
                            "exog_lags": spec.exog_lags,
                            "input_dimension": self._lag_spec_input_dimension(spec),
                            "learning_rate_init": float(learning_rate),
                            "activation": self.config.ffnn.activation,
                            "alpha": float(self.config.ffnn.fast_alpha),
                            "max_iter": int(self.config.ffnn.fast_max_iter),
                            "selection_data_scope": "train_only_inner_rolling",
                        }
                        component_fold_rows.append(enriched)
                        self._lag_screen_fold_rows.append(enriched)

            shortlisted_labels, fast_rmse_by_label = self._summarize_lag_screen(
                component,
                specs,
                component_fold_rows,
            )
            if shortlisted_labels:
                search_specs = [spec for spec in specs if spec.label in shortlisted_labels]
            else:
                self._log(
                    f"Rolling lag screen produced no valid candidate for {component}; "
                    "falling back to outer-validation fast screen."
                )
                lag_screen_fn = None

        if strategy == "staged_halving" and len(specs) > 1 and lag_screen_fn is None:
            screen_rows: list[dict[str, Any]] = []
            for spec in specs:
                n_features = len(self._feature_names(spec))
                current_id, row, pred = fit_score_candidate(
                    spec,
                    self.config.ffnn.fast_architecture_for(n_features),
                    self.config.ffnn.fast_alpha,
                    self.config.ffnn.seed_grid[0],
                    self.config.ffnn.activation,
                    self.config.ffnn.learning_rate_init,
                    self.config.ffnn.fast_max_iter,
                    "lag_screen",
                    False,
                )
                if current_id is None or row is None or pred is None:
                    continue
                rows.append(row)
                pred_by_id[current_id] = pred
                screen_rows.append(row)

            if not screen_rows:
                return rows, pred_by_id, final_candidate_ids, candidate_id

            screen_df = pd.DataFrame(screen_rows).sort_values(["Val RMSE", "Val MAE", "Val MAPE"])
            top_k = max(1, int(self.config.ffnn.top_k_lag_specs))
            shortlisted_labels = set(screen_df.head(top_k)["lag_spec"].tolist())
            fast_rmse_by_label = dict(zip(screen_df["lag_spec"], screen_df["Val RMSE"], strict=False))
            search_specs = [spec for spec in specs if spec.label in shortlisted_labels]
        elif strategy != "staged_halving" or len(specs) == 1:
            search_specs = specs
            fast_rmse_by_label = {}

        for spec in search_specs:
            n_features = len(self._feature_names(spec))
            for hidden_layer_sizes in self.config.ffnn.architectures_for(n_features):
                for alpha in self.config.ffnn.alpha_grid:
                    for seed in self.config.ffnn.seed_grid:
                        for activation in self.config.ffnn.activation_grid:
                            for learning_rate in self.config.ffnn.learning_rate_grid:
                                current_id, row, pred = fit_score_candidate(
                                    spec,
                                    hidden_layer_sizes,
                                    float(alpha),
                                    int(seed),
                                    str(activation),
                                    float(learning_rate),
                                    self.config.ffnn.max_iter,
                                    "hyperparam_search" if strategy == "staged_halving" else "full_grid",
                                    True,
                                    fast_rmse_by_label.get(spec.label),
                                )
                                if current_id is None or row is None or pred is None:
                                    continue
                                rows.append(row)
                                pred_by_id[current_id] = pred
                                final_candidate_ids.append(current_id)

        return rows, pred_by_id, final_candidate_ids, candidate_id

    def _batch_predict_observed(
        self,
        model: NumpyFFNNRegressor,
        observed: pd.DataFrame,
        eval_index: pd.Index,
        spec: LagSpec,
        split: str,
        component: str,
        predicted_col: str = "predicted",
    ) -> pd.DataFrame:
        """Dự báo các timestamp đã quan sát cho pipeline no-VMD.
        Hàm chỉ lấy các dòng thuộc eval_index và trả về DataFrame chuẩn gồm
        date, split, component, actual và cột dự báo.
        """
        # Tạo design từ chuỗi đã quan sát gồm train + split cần đánh giá.
        design = self._build_supervised(observed, spec)
        # Giữ đúng các timestamp thuộc validation/test để không chấm nhầm train.
        design = design.loc[design.index.intersection(eval_index)].copy()
        # Dự báo bằng đúng thứ tự feature đã dùng lúc fit.
        pred = model.predict(design[self._feature_names(spec)].to_numpy(float))
        return pd.DataFrame(
            {
                "date": design.index,
                "split": split,
                "component": component,
                "actual": design[self.target].to_numpy(float),
                predicted_col: pred,
            }
        )

    def _next_x(self, component_history: pd.DataFrame, spec: LagSpec) -> np.ndarray:
        """Tạo vector input một bước từ lịch sử component tại forecast origin."""
        values: list[float] = []
        for lag in spec.target_lags:
            # Lấy target lag từ lịch sử component ngay trước forecast origin.
            values.append(float(component_history[self.target].iloc[-lag]))
        for feature in self.features:
            for lag in spec.exog_lags.get(feature, ()):
                # Lấy exogenous lag cùng thứ tự với _feature_names để model nhận đúng cột.
                values.append(float(component_history[feature].iloc[-lag]))
        return np.asarray(values, dtype=float).reshape(1, -1)

    def _uses_validation_target_lag_screen(self) -> bool:
        """Return whether target lags should be selected by FFNN validation screening."""
        return self.config.ardl.target_lag_strategy.lower().replace("-", "_") == "validation_screen"

    def _uses_joint_target_lag_spec_screen(self) -> bool:
        """Return whether target and exogenous lags should be screened jointly."""
        return self.config.ardl.target_lag_strategy.lower().replace("-", "_") == "joint_validation_screen"

    def _target_lag_screen_spec(self, train_frame: pd.DataFrame, target_lags: tuple[int, ...]) -> LagSpec:
        """Build one cheap ARDL rank-1 exogenous spec for a target-lag candidate."""
        screen_config = replace(
            self.config.ardl,
            target_lag_strategy="fixed",
            fixed_target_lags=tuple(int(lag) for lag in target_lags),
            top_n=1,
            lag_spec_strategy="staged",
            max_lag_specs=1,
        )
        table, raw_specs = ARDLOrderSelector(screen_config).select(
            train_frame,
            self.target,
            self.features,
            target_lags_override=tuple(int(lag) for lag in target_lags),
        )
        del table
        for raw_spec in raw_specs:
            try:
                return self._safe_spec(raw_spec)
            except ValueError:
                continue
        raise ValueError(f"No forecast-safe validation-screen spec for target_lags={target_lags}.")

    def _validation_screen_target_lags(
        self,
        component: str,
        train_frame: pd.DataFrame,
        prediction_fn: Callable[[NumpyFFNNRegressor, LagSpec, int], pd.DataFrame],
        scoring_fn: Callable[[pd.DataFrame], dict[str, float]],
    ) -> tuple[tuple[int, ...], pd.DataFrame]:
        """Select target lags by cheap FFNN validation screening."""
        selector = ARDLOrderSelector(self.config.ardl)
        candidates = selector.target_lag_candidate_sets(train_frame, self.target)
        target_lag_metadata = selector.target_lag_candidate_metadata(train_frame, self.target)
        rows: list[dict[str, Any]] = []
        for candidate_id, target_lags in enumerate(candidates):
            metadata = target_lag_metadata.get(tuple(target_lags), {})
            row: dict[str, Any] = {
                "feature": "__target_lag_validation_screen__",
                "component": component,
                "target_lag_candidate_id": candidate_id,
                "target_lags": tuple(target_lags),
                "target_lag_strategy": "validation_screen",
                "target_lag_preset": self.config.ardl.target_lag_preset,
                "target_lag_source": metadata.get("target_lag_source"),
                "acf_value": metadata.get("acf_value"),
                "pacf_value": metadata.get("pacf_value"),
                "significance_threshold": metadata.get("significance_threshold"),
                "selected_for_target_lags": False,
                "target_lag_selection_fallback": False,
            }
            try:
                spec = self._target_lag_screen_spec(train_frame, tuple(target_lags))
                n_features = len(self._feature_names(spec))
                model, n_train = self._fit_model(
                    train_frame,
                    spec,
                    self.config.ffnn.fast_architecture_for(n_features),
                    float(self.config.ffnn.fast_alpha),
                    int(self.config.ffnn.seed_grid[0]),
                    self.config.ffnn.activation,
                    max_iter=int(self.config.ffnn.fast_max_iter),
                )
                pred = prediction_fn(model, spec, candidate_id)
                if len(pred) < self.config.ffnn.min_val:
                    raise ValueError(f"Only {len(pred)} validation rows after target-lag screening.")
                row.update(
                    {
                        "lag_spec": spec.label,
                        "exog_lags": spec.exog_lags,
                        "n_train": n_train,
                        "n_val": len(pred),
                    }
                )
                row.update({f"Val {key}": value for key, value in scoring_fn(pred).items()})
            except (ValueError, KeyError, IndexError) as exc:
                row["error"] = str(exc)
            rows.append(row)

        screen = pd.DataFrame(rows)
        valid = screen.dropna(subset=["Val RMSE"]) if "Val RMSE" in screen.columns else pd.DataFrame()
        if valid.empty:
            fallback = ARDLOrderSelector(self.config.ardl)._fallback_target_lags()
            fallback_row = {
                "feature": "__target_lag_validation_screen__",
                "component": component,
                "target_lags": fallback,
                "target_lag_strategy": "validation_screen",
                "target_lag_preset": self.config.ardl.target_lag_preset,
                "selected_for_target_lags": True,
                "target_lag_selection_fallback": True,
                "error": "No target-lag validation-screen candidate could be scored.",
            }
            screen = pd.concat([screen, pd.DataFrame([fallback_row])], ignore_index=True)
            return fallback, screen

        best_idx = valid.sort_values(["Val RMSE", "Val MAE", "Val MAPE"]).index[0]
        selected = tuple(int(lag) for lag in screen.loc[best_idx, "target_lags"])
        screen.loc[best_idx, "selected_for_target_lags"] = True
        return selected, screen

    def _joint_target_lag_candidate_specs(self, component: str, train_frame: pd.DataFrame) -> tuple[pd.DataFrame, list[LagSpec]]:
        """Build LagSpecs by pairing each PACF/ACF target candidate with ARDL exog ranks.

        This strategy validates target-lag candidates and their matching ARDL
        exogenous-lag candidates in the same FFNN search instead of first
        selecting target lags and then running ARDL again for exogenous lags.
        """
        selector = ARDLOrderSelector(self.config.ardl)
        candidates = selector.target_lag_candidate_sets(train_frame, self.target)
        target_lag_metadata = selector.target_lag_candidate_metadata(train_frame, self.target)
        tables: list[pd.DataFrame] = []
        raw_specs: list[LagSpec] = []

        for candidate_id, target_lags in enumerate(candidates):
            target_lags = tuple(int(lag) for lag in target_lags)
            metadata = target_lag_metadata.get(target_lags, {})
            candidate_row = pd.DataFrame(
                [
                    {
                        "feature": "__target_lag_joint_validation_screen__",
                        "component": component,
                        "target_lag_candidate_id": candidate_id,
                        "target_lags": target_lags,
                        "target_lag_strategy": "joint_validation_screen",
                        "target_lag_preset": self.config.ardl.target_lag_preset,
                        "target_lag_source": metadata.get("target_lag_source"),
                        "acf_value": metadata.get("acf_value"),
                        "pacf_value": metadata.get("pacf_value"),
                        "significance_threshold": metadata.get("significance_threshold"),
                        "selected_for_joint_validation": True,
                    }
                ]
            )

            # Joint mode intentionally ranks the exogenous lags by AIC top-3
            # for each target-lag candidate before FFNN validation chooses the winner.
            joint_config = replace(
                self.config.ardl,
                target_lag_strategy="fixed",
                fixed_target_lags=target_lags,
                ic="aic",
                top_n=3,
            )
            table, candidate_specs = ARDLOrderSelector(joint_config).select(
                train_frame,
                self.target,
                self.features,
                target_lags_override=target_lags,
            )
            table = table.copy()
            table["component"] = component
            table["target_lag_candidate_id"] = candidate_id
            table["target_lag_strategy"] = "joint_validation_screen"
            table["target_lag_preset"] = self.config.ardl.target_lag_preset
            table["selected_ic"] = "aic"
            tables.extend([candidate_row, table])
            raw_specs.extend(candidate_specs)

        if not raw_specs:
            raise ValueError(f"No joint target/exogenous lag candidates for {component}.")
        return pd.concat(tables, ignore_index=True), raw_specs

    def _select_specs(
        self,
        frames: dict[str, pd.DataFrame],
        target_lags_by_frame: dict[str, tuple[int, ...]] | None = None,
        target_lag_screen_tables: dict[str, pd.DataFrame] | None = None,
    ) -> None:
        """Chạy ARDLOrderSelector cho từng frame và lưu các spec forecast-safe.
        Bảng điểm ARDL được giữ trong order_tables_, còn danh sách LagSpec đã
        lọc trùng được giữ trong selected_specs_.
        """
        selector = ARDLOrderSelector(self.config.ardl)
        # Reset kết quả chọn lag để mỗi lần run không dùng lẫn state cũ.
        self.order_tables_ = {}
        self.selected_specs_ = {}
        target_lags_by_frame = target_lags_by_frame or {}
        target_lag_screen_tables = target_lag_screen_tables or {}
        for name, frame in frames.items():
            self._log(f"Selecting ARDL lags on train only: {name}")
            # ARDL chỉ chạy trên frame train/component tương ứng.
            if self._uses_joint_target_lag_spec_screen() and name not in target_lags_by_frame:
                table, raw_specs = self._joint_target_lag_candidate_specs(name, frame)
            else:
                table, raw_specs = selector.select(
                    frame,
                    self.target,
                    self.features,
                    target_lags_override=target_lags_by_frame.get(name),
                )
            if name in target_lag_screen_tables:
                table = pd.concat([target_lag_screen_tables[name], table], ignore_index=True)
            safe_specs: list[LagSpec] = []
            seen: set[str] = set()
            for raw_spec in raw_specs:
                try:
                    # Loại lag 0 và các spec không có feature dùng được.
                    safe = self._safe_spec(raw_spec)
                except ValueError:
                    continue
                if safe.label not in seen:
                    # Tránh train trùng cùng một cấu hình lag.
                    safe_specs.append(safe)
                    seen.add(safe.label)
            if not safe_specs:
                raise ValueError(f"No forecast-safe lag specs for {name}.")
            self.order_tables_[name] = table
            self.selected_specs_[name] = safe_specs

    def _audit_rows(
        self,
        pipeline: str,
        stage: str,
        split: str,
        data: pd.DataFrame,
        spec: LagSpec | None = None,
        feature_columns: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Tạo bản ghi audit cho dữ liệu đi qua một stage của pipeline.
        Audit lưu kích thước, khoảng thời gian, danh sách feature, lag được dùng,
        trạng thái NaN và các biến đổi dừng đã áp dụng.
        """
        # Cờ NaN giúp phát hiện stage nào còn thiếu dữ liệu sau split/lag/cache.
        has_nan = bool(data.isna().any().any())
        if "date" in data.columns:
            # Forecast DataFrame dùng cột date thay vì DatetimeIndex.
            start_time = pd.to_datetime(data["date"]).min()
            end_time = pd.to_datetime(data["date"]).max()
        else:
            # Dữ liệu raw/train/component vẫn dùng index thời gian.
            start_time = data.index.min() if len(data) else pd.NaT
            end_time = data.index.max() if len(data) else pd.NaT
        return [
            {
                "pipeline": pipeline,
                "stage": stage,
                "split": split,
                "n_rows": int(len(data)),
                "start_time": start_time,
                "end_time": end_time,
                "feature_columns": feature_columns or list(data.columns),
                "target_lags": tuple(spec.target_lags) if spec else tuple(),
                "exogenous_lags": spec.exog_lags if spec else {},
                "has_lag_0": self._has_exog_lag_zero(spec) if spec else False,
                "has_nan": has_nan,
                "stationarity_transforms": self.transform_info_,
                "forecast_protocol": self.config.forecast_protocol,
            }
        ]

    def _metrics_rows(
        self,
        forecasts: pd.DataFrame,
        actual_col: str,
        predicted_col: str,
        transformed_scale: str,
    ) -> pd.DataFrame:
        """Tính metric theo split trên thang mô hình và thang level nếu cần.
        Khi có log transform hoặc diff1 ở target, hàm bổ sung actual_level và
        predicted_level vào forecasts trước khi tính metric level.
        """
        rows: list[dict[str, Any]] = []
        for split, group in forecasts.groupby("split", sort=False):
            # Metric đầu tiên luôn tính trên thang mà model trực tiếp dự báo.
            row: dict[str, Any] = {
                "split": split,
                "scale": transformed_scale,
                "forecast_protocol": self.config.forecast_protocol,
            }
            row.update(evaluate_forecast(group[actual_col], group[predicted_col]))
            rows.append(row)
            if self.config.data.log_transform or self.transform_info_.get(self.target) == "diff1":
                # Nếu pipeline đã log hoặc sai phân target, cần đổi về level để báo cáo dễ hiểu.
                actual_level, predicted_level = self._to_level(group, actual_col, predicted_col)
                forecasts.loc[group.index, "actual_level"] = actual_level
                forecasts.loc[group.index, "predicted_level"] = predicted_level
                level_row: dict[str, Any] = {
                    "split": split,
                    "scale": "level",
                    "forecast_protocol": self.config.forecast_protocol,
                }
                level_row.update(evaluate_forecast(actual_level, predicted_level))
                rows.append(level_row)
        return pd.DataFrame(rows)

    def _annotate_final_metrics(
        self,
        metrics: pd.DataFrame,
        csv_path: str | Path,
        model: str,
        uses_vmd: bool,
        best_models: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Gắn metadata run vào từng dòng final metric trước khi ghi/return."""
        out = metrics.copy()
        out.insert(0, "dataset", Path(csv_path).stem)
        out.insert(1, "model", model)
        out.insert(2, "target", self.target)
        out.insert(3, "uses_vmd", bool(uses_vmd))
        out.insert(4, "vmd_setting", "cached_vmd" if uses_vmd else "no_vmd")
        out.insert(5, "vmd_modes", int(self.config.vmd.modes) if uses_vmd else 0)
        if best_models is not None and "activation" in best_models.columns:
            activations = sorted({str(value) for value in best_models["activation"].dropna().tolist()})
            out.insert(6, "activation", ",".join(activations))
        else:
            out.insert(6, "activation", self.config.ffnn.activation)
        return out

    def _append_csv(self, frame: pd.DataFrame, path: Path) -> None:
        """Write a run-scoped CSV without mixing results from earlier runs."""
        write_csv(frame, path)

    def _to_level(
        self,
        frame: pd.DataFrame,
        actual_col: str,
        predicted_col: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Đảo sai phân và log transform để đưa actual/predicted về thang level.
        Với diff1, giá trị gốc liền trước được lấy từ base_model_data_ theo date.
        """
        if self.base_model_data_ is None:
            raise RuntimeError("base_model_data_ is required to invert stationarity transforms.")
        # Dùng date của forecast để map về quan sát gốc liền trước.
        dates = pd.Index(frame["date"])
        target_transform = self.transform_info_.get(self.target, "level")
        if target_transform == "diff1":
            # Đảo diff1 bằng cách cộng dự báo sai phân với giá trị gốc t-1.
            previous_base = self.base_model_data_[self.target].shift(1).reindex(dates).to_numpy(float)
            actual_base = previous_base + frame[actual_col].to_numpy(float)
            predicted_base = previous_base + frame[predicted_col].to_numpy(float)
        else:
            # Nếu target không sai phân, giá trị forecast đã ở base scale trước bước exp.
            actual_base = frame[actual_col].to_numpy(float)
            predicted_base = frame[predicted_col].to_numpy(float)

        if self.config.data.log_transform:
            # Đảo log-transform để đưa kết quả về đơn vị ban đầu.
            return np.exp(actual_base), np.exp(predicted_base)
        return actual_base, predicted_base

    def _plot_actual_predicted(
        self,
        forecasts: pd.DataFrame,
        split: str,
        actual_col: str,
        predicted_col: str,
        filename: str,
    ) -> None:
        """Vẽ và lưu biểu đồ actual so với predicted cho một split.
        Nếu có cột level sau khi đảo biến đổi, biểu đồ ưu tiên hiển thị thang đó.
        """
        try:
            # Dùng MPLCONFIGDIR trong /tmp để matplotlib không cần ghi vào home.
            os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
            import matplotlib.pyplot as plt
        except ImportError:
            self._log(f"Skipping plot {filename}: matplotlib is not installed.")
            return
        group = forecasts[forecasts["split"].eq(split)].copy()
        if group.empty:
            return
        if self.config.data.log_transform and {"actual_level", "predicted_level"}.issubset(group.columns):
            # Khi đã có cột level, biểu đồ ưu tiên hiển thị đơn vị gốc.
            actual_col, predicted_col = "actual_level", "predicted_level"
        group = group.sort_values("date")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(group["date"], group[actual_col], label="Actual", linewidth=2)
        ax.plot(group["date"], group[predicted_col], label="Predicted", linewidth=2)
        ax.set_title(f"Actual vs Predicted - {split}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Level" if actual_col.endswith("_level") else "Transformed")
        ax.legend()
        ax.grid(True, alpha=0.25)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(self.output_dir / filename, dpi=160)
        plt.close(fig)

    def _comparison_table(
        self,
        no_vmd_metrics: pd.DataFrame,
        vmd_metrics: pd.DataFrame,
    ) -> pd.DataFrame:
        """Tạo bảng tổng hợp so sánh metric giữa baseline no-VMD và cached VMD."""
        return compare_two_metric_tables(
            no_vmd_metrics.assign(pipeline="no_vmd"),
            vmd_metrics.assign(pipeline="cached_vmd"),
            first_key="no_vmd",
            second_key="cached_vmd",
            key_column="pipeline",
        )

    def _comparison_forecast_frame(
        self,
        forecasts: pd.DataFrame,
        pipeline: str,
        predicted_col: str,
    ) -> pd.DataFrame:
        """Chuẩn hóa forecast của một pipeline để vẽ chung trên cùng biểu đồ."""
        return normalize_forecasts(forecasts, pipeline=pipeline, predicted_col=predicted_col)

    def _model_metrics_comparison_table(
        self,
        first_metrics: pd.DataFrame,
        second_metrics: pd.DataFrame,
        first_model: str,
        second_model: str,
    ) -> pd.DataFrame:
        """Compare two model metric tables on common split/scale rows."""
        return compare_two_metric_tables(
            first_metrics,
            second_metrics,
            first_key=first_model,
            second_key=second_model,
            key_column="model",
        )

    def _varnn_forecast_comparison_frame(self, forecasts: pd.DataFrame) -> pd.DataFrame:
        """Normalize VARNN forecast columns for side-by-side model comparison."""
        return normalize_forecasts(
            forecasts,
            pipeline="varnn",
            predicted_col="predicted_raw_transformed",
        )

    def _plot_comparison_predictions(
        self,
        comparison_forecasts: pd.DataFrame,
        split: str,
        filename: str,
    ) -> None:
        """Vẽ actual, no-VMD predicted và cached VMD predicted trên cùng biểu đồ."""
        try:
            os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
            import matplotlib.pyplot as plt
        except ImportError:
            self._log(f"Skipping plot {filename}: matplotlib is not installed.")
            return
        group = comparison_forecasts[comparison_forecasts["split"].eq(split)].copy()
        if group.empty:
            return
        group["date"] = pd.to_datetime(group["date"])
        actual = (
            group[["date", "actual"]]
            .drop_duplicates(subset=["date"])
            .sort_values("date")
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(actual["date"], actual["actual"], label="Actual", linewidth=2.2, color="black")
        for pipeline, label in [("no_vmd", "No VMD predicted"), ("cached_vmd", "VMD predicted")]:
            pred = group[group["pipeline"].eq(pipeline)].sort_values("date")
            if pred.empty:
                continue
            ax.plot(pred["date"], pred["predicted"], label=label, linewidth=2)
        scale = group["scale"].dropna().iloc[0] if "scale" in group.columns and not group["scale"].dropna().empty else "transformed"
        ax.set_title(f"VMD vs No-VMD Predictions - {split}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Level" if scale == "level" else "Transformed")
        ax.legend()
        ax.grid(True, alpha=0.25)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(self.output_dir / filename, dpi=160)
        plt.close(fig)

    def _plot_model_comparison_predictions(
        self,
        comparison_forecasts: pd.DataFrame,
        split: str,
        filename: str,
    ) -> None:
        """Plot actual and each model prediction from a normalized comparison frame."""
        try:
            os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
            import matplotlib.pyplot as plt
        except ImportError:
            self._log(f"Skipping plot {filename}: matplotlib is not installed.")
            return
        group = comparison_forecasts[comparison_forecasts["split"].eq(split)].copy()
        if group.empty:
            return
        group["date"] = pd.to_datetime(group["date"])
        actual = group[["date", "actual"]].drop_duplicates(subset=["date"]).sort_values("date")
        label_map = {
            "vmd_ardl_ffnn": "VMD-ARDL-FFNN predicted",
            "varnn": "VARNN predicted",
        }
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(actual["date"], actual["actual"], label="Actual", linewidth=2.2, color="black")
        for pipeline in sorted(group["pipeline"].dropna().unique()):
            pred = group[group["pipeline"].eq(pipeline)].sort_values("date")
            ax.plot(pred["date"], pred["predicted"], label=label_map.get(str(pipeline), str(pipeline)), linewidth=2)
        scale = group["scale"].dropna().iloc[0] if "scale" in group.columns and not group["scale"].dropna().empty else "transformed"
        ax.set_title(f"VMD-ARDL-FFNN vs VARNN Predictions - {split}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Level" if scale == "level" else "Transformed")
        ax.legend()
        ax.grid(True, alpha=0.25)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(self.output_dir / filename, dpi=160)
        plt.close(fig)

    def _diagnostics(self, train: pd.DataFrame, frames: dict[str, pd.DataFrame], suffix: str) -> dict[str, pd.DataFrame]:
        """Tạo các bảng chẩn đoán train-only cho chuỗi, Granger và phần dư ARDL."""
        diagnostics = ForecastDiagnostics(self.config.diagnostics)
        # Chẩn đoán raw train trước, sau đó thêm từng component VMD/no-VMD.
        series = [diagnostics.describe_series(train, f"train_raw{suffix}")]
        series.extend(diagnostics.describe_series(frame, f"train_{name}{suffix}") for name, frame in frames.items())
        series_df = pd.concat(series, ignore_index=True)
        # Granger chỉ chạy trên train để không nhìn thấy validation/test.
        granger_df = diagnostics.granger_causality(train, self.target, self.features)
        residual_rows = []
        for name, specs in self.selected_specs_.items():
            frame = frames.get(name, train)
            for rank, spec in enumerate(specs, start=1):
                # Kiểm tra phần dư ARDL cho từng spec đã được chọn.
                residual_rows.append(
                    diagnostics.ardl_residual_diagnostics(
                        frame, self.target, self.features, spec, name, rank, trend=self.config.ardl.trend
                    )
                )
        residual_df = pd.DataFrame(residual_rows)
        return {
            "series_diagnostics": series_df,
            "granger_causality": granger_df,
            "ardl_residual_diagnostics": residual_df,
        }

    def run_without_vmd(self, csv_path: str | Path) -> dict[str, pd.DataFrame]:
        """Chạy pipeline baseline không VMD trên chuỗi đã biến đổi dừng.
        Quy trình gồm chọn lag trên train, grid-search FFNN theo validation,
        refit model thắng trên train+validation, dự báo test và ghi các CSV/PNG.
        """
        total_started = perf_counter()
        self._log("Starting fixed no-VMD pipeline.")
        self._lag_screen_fold_rows = []
        self._lag_screen_stability_rows = []
        # Nạp dữ liệu rồi chia thời gian trước khi bất kỳ model nào được fit.
        base_data = self.load_data(csv_path)
        base_train, base_val, base_test = self.split_raw_data(base_data)
        # Biến đổi dừng được quyết định bằng train nhưng áp dụng nhất quán cho cả chuỗi.
        data = self._apply_stationarity_transform(base_data, base_train)
        # Căn lại split sau khi drop NaN do diff1 để train/val/test vẫn đúng mốc thời gian.
        train = data.loc[data.index.intersection(base_train.index)].copy()
        val = data.loc[data.index.intersection(base_val.index)].copy()
        test = data.loc[data.index.intersection(base_test.index)].copy()
        self._log(
            "Stationarity-adjusted rows: "
            f"train={len(train)}, validation={len(val)}, test={len(test)}"
        )
        component = "raw_no_vmd"
        observed_val = pd.concat([train, val])
        target_lags_by_frame: dict[str, tuple[int, ...]] = {}
        target_lag_screen_tables: dict[str, pd.DataFrame] = {}
        selection_started = perf_counter()
        if self._uses_validation_target_lag_screen():
            self._log(f"Validation-screening target lags: {component}")
            selected_target_lags, screen_table = self._validation_screen_target_lags(
                component=component,
                train_frame=train,
                prediction_fn=lambda model, spec, candidate_id: self._batch_predict_observed(
                    model, observed_val, val.index, spec, "validation", component
                ),
                scoring_fn=lambda pred: evaluate_forecast(pred["actual"], pred["predicted"]),
            )
            target_lags_by_frame[component] = selected_target_lags
            target_lag_screen_tables[component] = screen_table
        # Với baseline no-VMD, toàn bộ chuỗi được xem như một component duy nhất.
        self._select_specs({component: train}, target_lags_by_frame, target_lag_screen_tables)
        ardl_selection_seconds = perf_counter() - selection_started
        audit = []
        for split, frame in [("train", train), ("validation", val), ("test", test)]:
            # Ghi audit split trước khi tạo lag để biết kích thước dữ liệu gốc của từng split.
            audit.extend(self._audit_rows("no_vmd", "chronological_split", split, frame))

        ffnn_search_started = perf_counter()
        lag_screen_folds = self._expanding_lag_screen_splits(train)
        rows, val_frames, final_candidate_ids, _ = self._ffnn_search_candidates(
            component=component,
            train_data=train,
            specs=self.selected_specs_[component],
            prediction_fn=lambda model, spec, candidate_id: self._batch_predict_observed(
                model, observed_val, val.index, spec, "validation", component
            ),
            scoring_fn=lambda pred: self._score_raw_prediction_on_both_scales(
                pred,
                "actual",
                "predicted",
            ),
            uses_vmd=False,
            lag_screen_fn=(
                lambda spec, learning_rate: self._screen_no_vmd_lag_spec(
                    spec,
                    learning_rate,
                    lag_screen_folds,
                    component,
                )
            )
            if lag_screen_folds
            else None,
        )
        for row in rows:
            spec = LagSpec(tuple(row["target_lags"]), {k: tuple(v) for k, v in row["exog_lags"].items()})
            audit.extend(
                self._audit_rows(
                    "no_vmd",
                    f"validation_design_{row['search_stage']}",
                    "validation",
                    val_frames[int(row["candidate_id"])],
                    spec,
                    self._feature_names(spec),
                )
            )
        if not rows:
            raise ValueError("No no-VMD candidates could be trained.")
        # Chọn candidate tốt nhất theo RMSE, rồi MAE và MAPE làm tie-breaker.
        search_df = pd.DataFrame(rows)
        search_df["eligible_final_selection"] = (
            search_df["candidate_id"].isin(final_candidate_ids) if final_candidate_ids else True
        )
        selection_df = search_df[search_df["eligible_final_selection"]].copy()
        if final_candidate_ids:
            selection_df = search_df[search_df["candidate_id"].isin(final_candidate_ids)].copy()
        selection_scale = self.config.ffnn.final_selection_scale.replace("-", "_")
        selection_prefix = "Val Level" if selection_scale == "level" else "Val"
        selection_columns = [
            f"{selection_prefix} RMSE",
            f"{selection_prefix} MAE",
            f"{selection_prefix} MAPE",
        ]
        self.search_results_ = search_df.sort_values(
            ["eligible_final_selection", *selection_columns],
            ascending=[False, True, True, True],
        ).reset_index(drop=True)
        best = selection_df.sort_values(selection_columns).iloc[0]
        best = best.copy()
        best["final_selection_scale"] = selection_scale
        ffnn_search_seconds = perf_counter() - ffnn_search_started
        # Khôi phục LagSpec từ dòng winner để refit và dự báo test.
        best_spec = LagSpec(tuple(best["target_lags"]), {k: tuple(v) for k, v in best["exog_lags"].items()})
        self._log("Refitting locked no-VMD winner on train+validation.")
        # Refit winner trên train+validation sau khi hyperparameter đã khóa bằng validation.
        refit_started = perf_counter()
        best_model, n_refit = self._fit_model(
            pd.concat([train, val]),
            best_spec,
            self._hidden_layer_sizes_from_row(best),
            float(best["alpha"]),
            int(best["seed"]),
            str(best.get("activation", self.config.ffnn.activation)),
            learning_rate_init=float(
                best.get("learning_rate_init", self.config.ffnn.learning_rate_init)
            ),
        )
        # Giữ nguyên validation forecast của winner để final_forecasts gồm cả val và test.
        val_forecasts = val_frames[int(best["candidate_id"])].rename(columns={"actual": "actual_raw_transformed", "predicted": "predicted_raw_transformed"})
        # Test forecast dùng model đã refit và lịch sử train+validation+test quan sát.
        test_forecasts = self._batch_predict_observed(
            best_model, pd.concat([train, val, test]), test.index, best_spec, "test", component
        ).rename(columns={"actual": "actual_raw_transformed", "predicted": "predicted_raw_transformed"})
        refit_forecast_seconds = perf_counter() - refit_started
        final = pd.concat([val_forecasts, test_forecasts], ignore_index=True)
        final["forecast_protocol"] = self.config.forecast_protocol
        best_model_df = pd.DataFrame([{**best.to_dict(), "n_refit_train_val": n_refit}])
        self.best_component_models_ = best_model_df
        # Metric có thể thêm cột level vào final khi cần đảo log/diff.
        self.final_metrics_ = self._annotate_final_metrics(
            self._metrics_rows(final, "actual_raw_transformed", "predicted_raw_transformed", "raw_transformed"),
            csv_path,
            "ARDL_FFNN",
            False,
            best_model_df,
        )
        self.final_forecasts_ = final.sort_values(["split", "date"]).reset_index(drop=True)
        audit.extend(self._audit_rows("no_vmd", "test_design", "test", test_forecasts, best_spec, self._feature_names(best_spec)))
        self.input_audit_ = pd.DataFrame(audit)
        self.lag_screen_fold_metrics_ = pd.DataFrame(self._lag_screen_fold_rows)
        self.lag_screen_stability_ = pd.DataFrame(self._lag_screen_stability_rows)

        diagnostics = self._diagnostics(train, {component: train}, "_no_vmd")
        # Gộp bảng chọn lag ARDL từ từng component để xuất một file duy nhất.
        ardl_orders = pd.concat([table.assign(component=name) for name, table in self.order_tables_.items()], ignore_index=True)
        # Xuất toàn bộ artifact của baseline no-VMD để đối chiếu lại sau chạy.
        self.search_results_.to_csv(self.output_dir / "ffnn_validation_search_fixed_no_vmd.csv", index=False)
        if not self.lag_screen_fold_metrics_.empty:
            self.lag_screen_fold_metrics_.to_csv(
                self.output_dir / "ffnn_lag_screen_folds_no_vmd.csv",
                index=False,
            )
        if not self.lag_screen_stability_.empty:
            self.lag_screen_stability_.to_csv(
                self.output_dir / "ffnn_lag_screen_stability_no_vmd.csv",
                index=False,
            )
        best_model_df.to_csv(self.output_dir / "best_model_fixed_no_vmd.csv", index=False)
        self.final_forecasts_.to_csv(self.output_dir / "final_forecasts_fixed_no_vmd.csv", index=False)
        self._append_csv(self.final_metrics_, self.output_dir / "final_metrics_fixed_no_vmd.csv")
        self.input_audit_.to_csv(self.output_dir / "input_audit_no_vmd.csv", index=False)
        if self.stationarity_screen_ is not None:
            self.stationarity_screen_.to_csv(
                self.output_dir / "stationarity_screen_train_only_no_vmd.csv",
                index=False,
            )
        ardl_orders.to_csv(self.output_dir / "ardl_selected_orders_train_only_no_vmd.csv", index=False)
        for name, df in diagnostics.items():
            df.to_csv(self.output_dir / f"{name}_train_only_no_vmd.csv", index=False)
        self._plot_actual_predicted(self.final_forecasts_, "validation", "actual_raw_transformed", "predicted_raw_transformed", "actual_vs_predicted_val_no_vmd.png")
        self._plot_actual_predicted(self.final_forecasts_, "test", "actual_raw_transformed", "predicted_raw_transformed", "actual_vs_predicted_test_no_vmd.png")
        timing = pd.DataFrame(
            [
                {
                    "ardl_selection_seconds": ardl_selection_seconds,
                    "vmd_cache_seconds": 0.0,
                    "ffnn_search_seconds": ffnn_search_seconds,
                    "refit_forecast_seconds": refit_forecast_seconds,
                    "total_runtime_seconds": perf_counter() - total_started,
                }
            ]
        )
        write_csv(timing, self.output_dir / "runtime_breakdown.csv")
        return {
            "search_results": self.search_results_,
            "ardl_orders": ardl_orders,
            "best_component_models": best_model_df,
            "final_forecasts": self.final_forecasts_,
            "final_metrics": self.final_metrics_,
            "input_audit": self.input_audit_,
            "stationarity_screen": self.stationarity_screen_,
            "lag_screen_fold_metrics": self.lag_screen_fold_metrics_,
            "lag_screen_stability": self.lag_screen_stability_,
            "timing": timing,
            **diagnostics,
        }

    def _make_origin_cache(
        self,
        initial_history: pd.DataFrame,
        evaluation: pd.DataFrame,
        split: str,
    ) -> dict[pd.Timestamp, dict[str, pd.DataFrame]]:
        """Tạo cache VMD cho từng forecast origin trong validation/test.
        Mỗi origin chỉ phân rã lịch sử đã có trước thời điểm cần dự báo, sau đó
        mới append quan sát thật để chuẩn bị origin tiếp theo.
        """
        self._log(f"Creating {split} VMD origin cache ({len(evaluation)} timestamps).")
        # History bắt đầu bằng train hoặc train+validation tùy split đang cache.
        history = initial_history.copy()
        cache: dict[pd.Timestamp, dict[str, pd.DataFrame]] = {}
        for date, row in evaluation.iterrows():
            # Cache tại date chỉ thấy lịch sử trước date, vì row hiện tại chưa được append.
            cache[pd.Timestamp(date)] = self.decompose(history)
            # Sau khi tạo cache cho origin hiện tại, thêm actual để origin kế tiếp có lịch sử mới.
            history = pd.concat([history, row.to_frame().T], axis=0)
        return cache

    def _make_actual_component_cache(
        self,
        initial_history: pd.DataFrame,
        evaluation: pd.DataFrame,
        split: str,
    ) -> dict[pd.Timestamp, dict[str, float]]:
        """Lấy actual target component tại từng origin bằng expanding-history VMD.

        Với date hiện tại, cache này chỉ phân rã history + actual của chính date đó.
        Nó không nhìn các dòng tương lai nhưng vẫn tạo được nhãn component để chấm
        component-level forecast đúng thang đo.
        """
        self._log(f"Creating {split} VMD actual-component cache ({len(evaluation)} timestamps).")
        history = initial_history.copy()
        actual_components: dict[pd.Timestamp, dict[str, float]] = {}
        for date, row in evaluation.iterrows():
            current_history = pd.concat([history, row.to_frame().T], axis=0)
            decomposed = self.decompose(current_history)
            actual_components[pd.Timestamp(date)] = {
                component: float(component_frame.loc[date, self.target])
                for component, component_frame in decomposed.items()
            }
            history = current_history
        return actual_components

    def _predict_component_from_cache(
        self,
        model: NumpyFFNNRegressor,
        component: str,
        cache: dict[pd.Timestamp, dict[str, pd.DataFrame]],
        actual_raw: pd.DataFrame,
        spec: LagSpec,
        split: str,
        candidate_id: int | None = None,
        actual_components: dict[pd.Timestamp, dict[str, float]] | None = None,
    ) -> pd.DataFrame:
        """Dự báo một component VMD bằng lịch sử đã cache tại từng timestamp.
        Kết quả có thể gắn candidate_id để phục vụ chấm điểm và ghép tổ hợp.
        """
        rows: list[dict[str, Any]] = []
        for date in actual_raw.index:
            # Lấy đúng component history đã được phân rã tại forecast origin này.
            component_history = cache[pd.Timestamp(date)][component]
            # _next_x tạo một hàng feature từ các lag cuối cùng trong component_history.
            predicted = float(model.predict(self._next_x(component_history, spec))[0])
            row = {
                "date": date,
                "split": split,
                "component": component,
                "predicted_component": predicted,
            }
            if actual_components is not None:
                row["actual_component"] = actual_components[pd.Timestamp(date)][component]
            if candidate_id is not None:
                row["candidate_id"] = candidate_id
            rows.append(row)
        return pd.DataFrame(rows)

    def _reconstruct(self, component_predictions: pd.DataFrame, actual_raw: pd.DataFrame) -> pd.DataFrame:
        """Cộng dự báo component theo date để tái tạo forecast của target.
        Actual được lấy từ actual_raw trên cùng thang transformed của mô hình.
        """
        # Tổng các predicted_component chính là dự báo reconstructed của target.
        group_columns = ["split", "date"]
        if "seed" in component_predictions.columns:
            group_columns.append("seed")
        out = (
            component_predictions.groupby(group_columns, as_index=False)
            .agg(predicted_reconstructed=("predicted_component", "sum"), component_count=("component", "nunique"))
            .sort_values(["split", "date"])
        )
        # Gắn actual target để tính metric reconstructed trên cùng timestamp.
        out["actual_raw_transformed"] = out["date"].map(actual_raw[self.target]).astype(float)
        return out

    def _score_component_prediction(self, prediction: pd.DataFrame, actual_raw: pd.DataFrame) -> dict[str, float]:
        """Score one VMD component candidate on its own component scale."""
        del actual_raw
        return evaluate_forecast(prediction["actual_component"], prediction["predicted_component"])

    def _score_raw_prediction_on_both_scales(
        self,
        prediction: pd.DataFrame,
        actual_col: str,
        predicted_col: str,
    ) -> dict[str, float]:
        """Score a target forecast on model and final reported level scales."""
        transformed = evaluate_forecast(prediction[actual_col], prediction[predicted_col])
        actual_level, predicted_level = self._to_level(prediction, actual_col, predicted_col)
        level = evaluate_forecast(actual_level, predicted_level)
        return {
            **transformed,
            **{f"Level {metric}": value for metric, value in level.items()},
        }

    def _screen_no_vmd_lag_spec(
        self,
        spec: LagSpec,
        learning_rate: float,
        folds: list[tuple[int, pd.DataFrame, pd.DataFrame]],
        component: str,
        seed: int | None = None,
    ) -> list[dict[str, Any]]:
        """Evaluate one no-VMD lag spec on every train-only expanding fold."""
        rows: list[dict[str, Any]] = []
        architecture = self.config.ffnn.fast_architecture_for(
            self._lag_spec_input_dimension(spec)
        )
        for fold, fold_train, fold_validation in folds:
            row: dict[str, Any] = {
                "fold": int(fold),
                "train_start": fold_train.index.min(),
                "train_end": fold_train.index.max(),
                "validation_start": fold_validation.index.min(),
                "validation_end": fold_validation.index.max(),
                "n_train_before_lag": len(fold_train),
                "n_validation": len(fold_validation),
                "screen_metric_scale": self.config.ffnn.final_selection_scale.replace("-", "_"),
            }
            try:
                model, n_train = self._fit_model(
                    fold_train,
                    spec,
                    architecture,
                    float(self.config.ffnn.fast_alpha),
                    int(self.config.ffnn.seed_grid[0] if seed is None else seed),
                    self.config.ffnn.activation,
                    learning_rate_init=float(learning_rate),
                    max_iter=int(self.config.ffnn.fast_max_iter),
                )
                prediction = self._batch_predict_observed(
                    model,
                    pd.concat([fold_train, fold_validation]),
                    fold_validation.index,
                    spec,
                    f"lag_screen_fold_{fold}",
                    component,
                )
                metrics = self._score_raw_prediction_on_both_scales(
                    prediction,
                    "actual",
                    "predicted",
                )
                prefix = "Level " if self.config.ffnn.final_selection_scale == "level" else ""
                row.update(
                    {
                        "n_train_after_lag": int(n_train),
                        "screen_RMSE": metrics[f"{prefix}RMSE"],
                        "screen_MAE": metrics[f"{prefix}MAE"],
                        "screen_MAPE": metrics[f"{prefix}MAPE"],
                        "transformed_RMSE": metrics["RMSE"],
                        "level_RMSE": metrics["Level RMSE"],
                        "error": None,
                    }
                )
            except (ValueError, KeyError, IndexError) as exc:
                row.update(
                    {
                        "n_train_after_lag": 0,
                        "screen_RMSE": float("nan"),
                        "screen_MAE": float("nan"),
                        "screen_MAPE": float("nan"),
                        "transformed_RMSE": float("nan"),
                        "level_RMSE": float("nan"),
                        "error": str(exc),
                    }
                )
            rows.append(row)
        return rows

    def _prepare_vmd_lag_screen_contexts(
        self,
        train: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        """Precompute leakage-safe VMD histories for train-only rolling folds."""
        contexts: list[dict[str, Any]] = []
        for fold, fold_train, fold_validation in self._expanding_lag_screen_splits(train):
            split = f"lag-screen-fold-{fold}"
            contexts.append(
                {
                    "fold": int(fold),
                    "train": fold_train,
                    "validation": fold_validation,
                    "train_components": self.decompose(fold_train),
                    "origin_cache": self._make_origin_cache(fold_train, fold_validation, split),
                    "actual_components": self._make_actual_component_cache(
                        fold_train,
                        fold_validation,
                        split,
                    ),
                }
            )
        return contexts

    def _screen_vmd_lag_spec(
        self,
        component: str,
        spec: LagSpec,
        learning_rate: float,
        contexts: list[dict[str, Any]],
        seed: int | None = None,
    ) -> list[dict[str, Any]]:
        """Evaluate one component lag spec over cached train-only VMD folds."""
        rows: list[dict[str, Any]] = []
        architecture = self.config.ffnn.fast_architecture_for(
            self._lag_spec_input_dimension(spec)
        )
        for context in contexts:
            fold = int(context["fold"])
            fold_train = context["train"]
            fold_validation = context["validation"]
            row: dict[str, Any] = {
                "fold": fold,
                "train_start": fold_train.index.min(),
                "train_end": fold_train.index.max(),
                "validation_start": fold_validation.index.min(),
                "validation_end": fold_validation.index.max(),
                "n_train_before_lag": len(fold_train),
                "n_validation": len(fold_validation),
                "screen_metric_scale": "component_transformed",
            }
            try:
                component_train = context["train_components"][component]
                model, n_train = self._fit_model(
                    component_train,
                    spec,
                    architecture,
                    float(self.config.ffnn.fast_alpha),
                    int(self.config.ffnn.seed_grid[0] if seed is None else seed),
                    self.config.ffnn.activation,
                    learning_rate_init=float(learning_rate),
                    max_iter=int(self.config.ffnn.fast_max_iter),
                )
                prediction = self._predict_component_from_cache(
                    model,
                    component,
                    context["origin_cache"],
                    fold_validation,
                    spec,
                    f"lag_screen_fold_{fold}",
                    actual_components=context["actual_components"],
                )
                metrics = self._score_component_prediction(prediction, fold_validation)
                row.update(
                    {
                        "n_train_after_lag": int(n_train),
                        "screen_RMSE": metrics["RMSE"],
                        "screen_MAE": metrics["MAE"],
                        "screen_MAPE": metrics["MAPE"],
                        "transformed_RMSE": metrics["RMSE"],
                        "level_RMSE": float("nan"),
                        "error": None,
                    }
                )
            except (ValueError, KeyError, IndexError) as exc:
                row.update(
                    {
                        "n_train_after_lag": 0,
                        "screen_RMSE": float("nan"),
                        "screen_MAE": float("nan"),
                        "screen_MAPE": float("nan"),
                        "transformed_RMSE": float("nan"),
                        "level_RMSE": float("nan"),
                        "error": str(exc),
                    }
                )
            rows.append(row)
        return rows

    def run(self, csv_path: str | Path) -> dict[str, pd.DataFrame]:
        """Chạy pipeline VMD đầy đủ với cache walk-forward chống leakage.
        Hàm chọn model tốt nhất cho từng component, tìm tổ hợp component tốt
        trên validation, refit trên train+validation, dự báo test và xuất kết quả.
        """
        total_started = perf_counter()
        self._log("Starting cached VMD pipeline.")
        self._lag_screen_fold_rows = []
        self._lag_screen_stability_rows = []
        # Chuẩn bị dữ liệu giống baseline để so sánh công bằng giữa VMD và no-VMD.
        base_data = self.load_data(csv_path)
        base_train, base_val, base_test = self.split_raw_data(base_data)
        data = self._apply_stationarity_transform(base_data, base_train)
        # Gắn split theo index gốc sau khi dữ liệu đã được biến đổi dừng.
        train = data.loc[data.index.intersection(base_train.index)].copy()
        val = data.loc[data.index.intersection(base_val.index)].copy()
        test = data.loc[data.index.intersection(base_test.index)].copy()
        self._log(
            "Stationarity-adjusted rows: "
            f"train={len(train)}, validation={len(val)}, test={len(test)}"
        )
        audit = []
        for split, frame in [("train", train), ("validation", val), ("test", test)]:
            audit.extend(self._audit_rows("cached_vmd", "chronological_split", split, frame))

        self._log("Running VMD on train only.")
        vmd_cache_started = perf_counter()
        # VMD ban đầu chỉ chạy trên train để chọn lag và fit candidate component.
        train_components = self.decompose(train)
        # Cache validation theo origin để VMD không thấy điểm validation tương lai.
        val_cache = self._make_origin_cache(train, val, "validation")
        val_actual_components = self._make_actual_component_cache(train, val, "validation")
        vmd_cache_seconds = perf_counter() - vmd_cache_started
        target_lags_by_frame: dict[str, tuple[int, ...]] = {}
        target_lag_screen_tables: dict[str, pd.DataFrame] = {}
        selection_started = perf_counter()
        if self._uses_validation_target_lag_screen():
            for component, train_component in train_components.items():
                self._log(f"Validation-screening target lags: {component}")
                selected_target_lags, screen_table = self._validation_screen_target_lags(
                    component=component,
                    train_frame=train_component,
                    prediction_fn=lambda model, spec, candidate_id, component=component: self._predict_component_from_cache(
                        model, component, val_cache, val, spec, "validation", candidate_id, val_actual_components
                    ),
                    scoring_fn=lambda pred: self._score_component_prediction(pred, val),
                )
                target_lags_by_frame[component] = selected_target_lags
                target_lag_screen_tables[component] = screen_table
        self._select_specs(train_components, target_lags_by_frame, target_lag_screen_tables)
        ardl_selection_seconds = perf_counter() - selection_started

        ffnn_search_started = perf_counter()
        needs_rolling_lag_screen = (
            self.config.ffnn.search_strategy.replace("-", "_") == "staged_halving"
            and any(len(specs) > 1 for specs in self.selected_specs_.values())
        )
        lag_screen_contexts = (
            self._prepare_vmd_lag_screen_contexts(train)
            if needs_rolling_lag_screen
            else []
        )
        candidate_predictions: dict[str, list[pd.DataFrame]] = {}
        pred_by_id: dict[int, pd.DataFrame] = {}
        candidate_rows: list[dict[str, Any]] = []
        candidate_id = 0
        for component, train_component in train_components.items():
            rows_for_component, preds_for_component, final_ids, candidate_id = self._ffnn_search_candidates(
                component=component,
                train_data=train_component,
                specs=self.selected_specs_[component],
                prediction_fn=lambda model, spec, row_id, component=component: self._predict_component_from_cache(
                    model, component, val_cache, val, spec, "validation", row_id, val_actual_components
                ),
                scoring_fn=lambda pred: self._score_component_prediction(pred, val),
                uses_vmd=True,
                candidate_id_start=candidate_id,
                lag_screen_fn=(
                    lambda spec, learning_rate, component=component: self._screen_vmd_lag_spec(
                        component,
                        spec,
                        learning_rate,
                        lag_screen_contexts,
                    )
                )
                if lag_screen_contexts
                else None,
            )
            candidate_rows.extend(rows_for_component)
            pred_by_id.update(preds_for_component)
            candidate_predictions[component] = [preds_for_component[row_id] for row_id in final_ids]
            for row in rows_for_component:
                spec = LagSpec(tuple(row["target_lags"]), {k: tuple(v) for k, v in row["exog_lags"].items()})
                audit.extend(
                    self._audit_rows(
                        "cached_vmd",
                        f"validation_cache_{row['search_stage']}",
                        "validation",
                        preds_for_component[int(row["candidate_id"])],
                        spec,
                        self._feature_names(spec),
                    )
                )
            if not candidate_predictions[component]:
                raise ValueError(f"No cached VMD validation candidates for {component}.")

        component_ranked: dict[str, pd.DataFrame] = {}
        for component, frames in candidate_predictions.items():
            scored = []
            for frame in frames:
                # Chấm candidate component trên đúng actual component, không so với target tổng.
                metrics = evaluate_forecast(frame["actual_component"], frame["predicted_component"])
                scored.append({"candidate_id": int(frame["candidate_id"].iloc[0]), **{f"Component Val {k}": v for k, v in metrics.items()}})
            component_ranked[component] = pd.DataFrame(scored).sort_values(["Component Val RMSE", "Component Val MAE", "Component Val MAPE"])

        # Giữ một số ít candidate tốt nhất mỗi component để giảm số tổ hợp cần thử.
        top_component_candidates = max(1, int(self.config.ffnn.top_component_candidates))
        top_by_component = {
            component: ranked.head(min(top_component_candidates, len(ranked)))["candidate_id"].astype(int).tolist()
            for component, ranked in component_ranked.items()
        }
        final_component_candidate_ids = {
            int(frame["candidate_id"].iloc[0])
            for frames in candidate_predictions.values()
            for frame in frames
        }
        combo_rows: list[dict[str, Any]] = []
        component_order = list(top_by_component)
        complexity_by_candidate = {
            int(row["candidate_id"]): len(tuple(row["target_lags"]))
            + sum(len(tuple(lags)) for lags in row["exog_lags"].values())
            for row in candidate_rows
        }
        for combo_id, ids in enumerate(product(*top_by_component.values())):
            # Ghép một candidate từ mỗi component để tạo dự báo reconstructed hoàn chỉnh.
            component_frames = [pred_by_id[int(candidate)] for candidate in ids]
            reconstructed = self._reconstruct(pd.concat(component_frames, ignore_index=True), val)
            transformed_metric = evaluate_forecast(
                reconstructed["actual_raw_transformed"],
                reconstructed["predicted_reconstructed"],
            )
            actual_level, predicted_level = self._to_level(
                reconstructed,
                "actual_raw_transformed",
                "predicted_reconstructed",
            )
            level_metric = evaluate_forecast(actual_level, predicted_level)
            row = {"combo_id": combo_id, "component_candidate_ids": tuple(int(x) for x in ids)}
            row.update({f"{component}_candidate_id": int(candidate) for component, candidate in zip(component_order, ids)})
            row["search_strategy"] = self.config.ffnn.search_strategy.lower().replace("-", "_")
            row["top_component_candidates"] = top_component_candidates
            row["total_input_dimension"] = sum(
                complexity_by_candidate[int(candidate)] for candidate in ids
            )
            row.update({f"Val {metric}": value for metric, value in transformed_metric.items()})
            row.update({f"Val Level {metric}": value for metric, value in level_metric.items()})
            combo_rows.append(row)
        # Chọn combo theo cùng scale được dùng trong bảng kết quả paper.
        selection_scale = self.config.ffnn.final_selection_scale.replace("-", "_")
        selection_prefix = "Val Level" if selection_scale == "level" else "Val"
        combo_df = pd.DataFrame(combo_rows).sort_values(
            [
                f"{selection_prefix} RMSE",
                f"{selection_prefix} MAE",
                f"{selection_prefix} MAPE",
                "total_input_dimension",
            ]
        ).reset_index(drop=True)
        combo_df["final_selection_scale"] = selection_scale

        candidates_df = pd.DataFrame(candidate_rows)
        candidates_df["eligible_final_selection"] = candidates_df["candidate_id"].isin(final_component_candidate_ids)
        self.search_results_ = combo_df
        # Lấy danh sách candidate_id thắng, mỗi id tương ứng một component model.
        winning_combo = combo_df.iloc[0]
        best_ids = tuple(int(x) for x in winning_combo["component_candidate_ids"])
        best_candidates = candidates_df[candidates_df["candidate_id"].isin(best_ids)].copy()
        best_candidates["combo_id"] = int(winning_combo["combo_id"])
        best_candidates["final_selection_scale"] = selection_scale
        best_candidates["combo_val_rmse"] = float(winning_combo[f"{selection_prefix} RMSE"])
        best_candidates["combo_val_mae"] = float(winning_combo[f"{selection_prefix} MAE"])
        best_candidates["combo_val_mape"] = float(winning_combo[f"{selection_prefix} MAPE"])
        best_candidates["combo_total_input_dimension"] = int(winning_combo["total_input_dimension"])
        ffnn_search_seconds = perf_counter() - ffnn_search_started

        self._log("Refitting cached VMD winners on train+validation.")
        # Trước khi dự báo test, refit VMD và model trên toàn bộ train+validation.
        history_before_test = pd.concat([train, val], axis=0)
        vmd_test_cache_started = perf_counter()
        history_components = self.decompose(history_before_test)
        # Test cache cũng walk-forward: mỗi test date chỉ thấy lịch sử trước date.
        test_cache = self._make_origin_cache(history_before_test, test, "test")
        test_actual_components = self._make_actual_component_cache(history_before_test, test, "test")
        vmd_cache_seconds += perf_counter() - vmd_test_cache_started
        refit_started = perf_counter()
        selected_component_predictions: list[pd.DataFrame] = []
        # Forecast validation của winner được giữ lại để báo cáo cùng test.
        val_selected = [pred_by_id[candidate_id] for candidate_id in best_ids]
        selected_component_predictions.extend(val_selected)
        for _, best in best_candidates.iterrows():
            component = str(best["component"])
            # Khôi phục LagSpec và hyperparameter của component winner.
            spec = LagSpec(tuple(best["target_lags"]), {k: tuple(v) for k, v in best["exog_lags"].items()})
            # Refit component winner trên component history train+validation.
            model, n_refit = self._fit_model(
                history_components[component],
                spec,
                self._hidden_layer_sizes_from_row(best),
                float(best["alpha"]),
                int(best["seed"]),
                str(best.get("activation", self.config.ffnn.activation)),
                learning_rate_init=float(
                    best.get("learning_rate_init", self.config.ffnn.learning_rate_init)
                ),
            )
            # Dự báo test component bằng model refit và cache VMD test.
            test_pred = self._predict_component_from_cache(
                model,
                component,
                test_cache,
                test,
                spec,
                "test",
                int(best["candidate_id"]),
                test_actual_components,
            )
            selected_component_predictions.append(test_pred)
            best_candidates.loc[best_candidates["candidate_id"].eq(best["candidate_id"]), "n_refit_train_val"] = n_refit
            audit.extend(self._audit_rows("cached_vmd", "test_cache", "test", test_pred, spec, self._feature_names(spec)))
        refit_forecast_seconds = perf_counter() - refit_started

        # Gom forecast component validation/test rồi reconstruct thành forecast cuối.
        self.component_forecasts_ = pd.concat(selected_component_predictions, ignore_index=True)
        actual_all = pd.concat([val, test], axis=0)
        self.final_forecasts_ = self._reconstruct(self.component_forecasts_, actual_all).reset_index(drop=True)
        self.final_forecasts_["forecast_protocol"] = self.config.forecast_protocol
        # Metric cuối được tính trên reconstructed transformed và level nếu có thể đảo.
        self.best_component_models_ = best_candidates.sort_values("component").reset_index(drop=True)
        self.final_metrics_ = self._annotate_final_metrics(
            self._metrics_rows(
                self.final_forecasts_,
                "actual_raw_transformed",
                "predicted_reconstructed",
                "reconstructed_transformed",
            ),
            csv_path,
            "VMD_ARDL_FFNN",
            True,
            self.best_component_models_,
        )
        self.input_audit_ = pd.DataFrame(audit)
        self.lag_screen_fold_metrics_ = pd.DataFrame(self._lag_screen_fold_rows)
        self.lag_screen_stability_ = pd.DataFrame(self._lag_screen_stability_rows)
        diagnostics = self._diagnostics(train, train_components, "_cached_vmd")
        # Gộp mọi bảng ARDL order theo component để xuất audit chọn lag.
        ardl_orders = pd.concat([table.assign(component=name) for name, table in self.order_tables_.items()], ignore_index=True)

        # Ghi artifact chính của pipeline cached VMD ra output_dir.
        combo_df.to_csv(self.output_dir / "ffnn_validation_search_cached_vmd.csv", index=False)
        candidates_df.to_csv(self.output_dir / "ffnn_component_validation_search_cached_vmd.csv", index=False)
        if not self.lag_screen_fold_metrics_.empty:
            self.lag_screen_fold_metrics_.to_csv(
                self.output_dir / "ffnn_lag_screen_folds_cached_vmd.csv",
                index=False,
            )
        if not self.lag_screen_stability_.empty:
            self.lag_screen_stability_.to_csv(
                self.output_dir / "ffnn_lag_screen_stability_cached_vmd.csv",
                index=False,
            )
        self.best_component_models_.to_csv(self.output_dir / "best_component_models_cached_vmd.csv", index=False)
        self.component_forecasts_.to_csv(self.output_dir / "component_predictions_cached_vmd.csv", index=False)
        self.final_forecasts_.to_csv(self.output_dir / "final_reconstructed_forecasts_cached_vmd.csv", index=False)
        self._append_csv(self.final_metrics_, self.output_dir / "final_reconstructed_metrics_cached_vmd.csv")
        self.input_audit_.to_csv(self.output_dir / "input_audit_cached_vmd.csv", index=False)
        if self.stationarity_screen_ is not None:
            self.stationarity_screen_.to_csv(
                self.output_dir / "stationarity_screen_train_only_cached_vmd.csv",
                index=False,
            )
        ardl_orders.to_csv(self.output_dir / "ardl_selected_orders_train_only.csv", index=False)
        for name, df in diagnostics.items():
            df.to_csv(self.output_dir / f"{name}_train_only.csv", index=False)
        self._plot_actual_predicted(self.final_forecasts_, "validation", "actual_raw_transformed", "predicted_reconstructed", "actual_vs_predicted_val_cached_vmd.png")
        self._plot_actual_predicted(self.final_forecasts_, "test", "actual_raw_transformed", "predicted_reconstructed", "actual_vs_predicted_test_cached_vmd.png")
        timing = pd.DataFrame(
            [
                {
                    "ardl_selection_seconds": ardl_selection_seconds,
                    "vmd_cache_seconds": vmd_cache_seconds,
                    "ffnn_search_seconds": ffnn_search_seconds,
                    "refit_forecast_seconds": refit_forecast_seconds,
                    "total_runtime_seconds": perf_counter() - total_started,
                }
            ]
        )
        write_csv(timing, self.output_dir / "runtime_breakdown.csv")
        return {
            "search_results": combo_df,
            "component_search_results": candidates_df,
            "ardl_orders": ardl_orders,
            "best_component_models": self.best_component_models_,
            "component_forecasts": self.component_forecasts_,
            "training_components": train_components,
            "final_forecasts": self.final_forecasts_,
            "final_metrics": self.final_metrics_,
            "input_audit": self.input_audit_,
            "stationarity_screen": self.stationarity_screen_,
            "lag_screen_fold_metrics": self.lag_screen_fold_metrics_,
            "lag_screen_stability": self.lag_screen_stability_,
            "timing": timing,
            **diagnostics,
        }

    def _component_signature_key(self, row: pd.Series) -> tuple[Any, ...]:
        """Return the seed-independent identity of one component candidate."""
        return (
            str(row["component"]),
            str(row["lag_spec"]),
            self._hidden_layer_sizes_from_row(row),
            float(row["alpha"]),
            str(row["activation"]),
            float(row["learning_rate_init"]),
        )

    def _fast_screen_block_subset_specs(
        self,
        *,
        component: str,
        train_data: pd.DataFrame,
        specs: list[LagSpec],
        validation: pd.DataFrame,
        validation_cache: dict[pd.Timestamp, dict[str, pd.DataFrame]],
        validation_actual_components: dict[pd.Timestamp, dict[str, float]],
        seeds: tuple[int, ...],
        top_k: int = 4,
        alpha: float = 0.1,
        learning_rate_init: float = 0.001,
    ) -> tuple[list[LagSpec], pd.DataFrame, pd.DataFrame]:
        """Fast-screen every block LagSpec with one lightweight FFNN per seed."""
        rows: list[dict[str, Any]] = []
        for spec in specs:
            hidden_layer_sizes = self.config.ffnn.fast_architecture_for(
                self._lag_spec_input_dimension(spec)
            )
            for seed in seeds:
                try:
                    model, n_train = self._fit_model(
                        train_data,
                        spec,
                        hidden_layer_sizes,
                        float(alpha),
                        int(seed),
                        self.config.ffnn.activation,
                        learning_rate_init=float(learning_rate_init),
                        max_iter=int(self.config.ffnn.fast_max_iter),
                    )
                    prediction = self._predict_component_from_cache(
                        model,
                        component,
                        validation_cache,
                        validation,
                        spec,
                        "validation",
                        actual_components=validation_actual_components,
                    )
                    if len(prediction) < int(self.config.ffnn.min_val):
                        raise ValueError(
                            f"Only {len(prediction)} validation rows in fast screen."
                        )
                    metrics = self._score_component_prediction(prediction, validation)
                    row: dict[str, Any] = {
                        "component": component,
                        "lag_spec": spec.label,
                        "target_lags": spec.target_lags,
                        "exog_lags": spec.exog_lags,
                        "input_dimension": self._lag_spec_input_dimension(spec),
                        "seed": int(seed),
                        "hidden_layer_sizes": hidden_layer_sizes,
                        "alpha": float(alpha),
                        "activation": self.config.ffnn.activation,
                        "learning_rate_init": float(learning_rate_init),
                        "max_iter": int(self.config.ffnn.fast_max_iter),
                        "n_train": int(n_train),
                        "n_val": int(len(prediction)),
                        "screen_stage": "block_subset_two_stage_fast_screen",
                        "screen_scale": "component_transformed",
                        "screen_validation_scope": "outer_validation",
                        "selection_data_scope": "outer_validation",
                        "error": None,
                    }
                    row.update({f"Val {metric}": value for metric, value in metrics.items()})
                except (ValueError, KeyError, IndexError) as exc:
                    row = {
                        "component": component,
                        "lag_spec": spec.label,
                        "target_lags": spec.target_lags,
                        "exog_lags": spec.exog_lags,
                        "input_dimension": self._lag_spec_input_dimension(spec),
                        "seed": int(seed),
                        "hidden_layer_sizes": hidden_layer_sizes,
                        "alpha": float(alpha),
                        "activation": self.config.ffnn.activation,
                        "learning_rate_init": float(learning_rate_init),
                        "max_iter": int(self.config.ffnn.fast_max_iter),
                        "n_train": 0,
                        "n_val": 0,
                        "screen_stage": "block_subset_two_stage_fast_screen",
                        "screen_scale": "component_transformed",
                        "screen_validation_scope": "outer_validation",
                        "selection_data_scope": "outer_validation",
                        "error": str(exc),
                        "Val RMSE": float("nan"),
                        "Val MAE": float("nan"),
                        "Val MAPE": float("nan"),
                        "Val Directional_Accuracy": float("nan"),
                    }
                rows.append(row)

        detail = pd.DataFrame(rows)
        valid = detail.dropna(subset=["Val RMSE"]).copy()
        summary = (
            valid.groupby("lag_spec", as_index=False)
            .agg(
                mean_val_RMSE=("Val RMSE", "mean"),
                std_val_RMSE=("Val RMSE", "std"),
                mean_val_MAE=("Val MAE", "mean"),
                mean_val_MAPE=("Val MAPE", "mean"),
                seed_count=("seed", "nunique"),
                input_dimension=("input_dimension", "first"),
            )
        )
        summary = summary[summary["seed_count"].eq(len(seeds))].copy()
        if summary.empty:
            raise ValueError(
                f"No complete two-stage fast-screen result for component={component}."
            )
        summary["std_val_RMSE"] = summary["std_val_RMSE"].fillna(0.0)
        summary = summary.sort_values(
            [
                "mean_val_RMSE",
                "mean_val_MAE",
                "mean_val_MAPE",
                "input_dimension",
                "lag_spec",
            ]
        ).reset_index(drop=True)
        summary.insert(0, "component", component)
        summary.insert(1, "fast_screen_rank", np.arange(1, len(summary) + 1))
        keep_count = min(max(1, int(top_k)), len(summary))
        summary["selected_for_full_grid"] = summary["fast_screen_rank"].le(keep_count)
        summary["fast_screen_alpha"] = float(alpha)
        summary["fast_screen_learning_rate_init"] = float(learning_rate_init)
        summary["fast_screen_seed_aggregation"] = "mean_across_reporting_seeds"
        summary["screen_validation_scope"] = "outer_validation"

        selected_labels = set(
            summary.loc[summary["selected_for_full_grid"], "lag_spec"].astype(str)
        )
        selected_specs = [spec for spec in specs if spec.label in selected_labels]
        detail["selected_for_full_grid"] = detail["lag_spec"].isin(selected_labels)
        return selected_specs, detail, summary

    def _fast_screen_block_subset_specs_inner_rolling(
        self,
        *,
        component: str,
        specs: list[LagSpec],
        contexts: list[dict[str, Any]],
        seeds: tuple[int, ...],
        top_k: int = 4,
        alpha: float = 0.1,
        learning_rate_init: float = 0.001,
    ) -> tuple[list[LagSpec], pd.DataFrame, pd.DataFrame]:
        """Fast-screen block specs on train-only expanding validation folds."""
        if not contexts:
            raise ValueError(
                "Block-subset inner rolling screen requires at least one fold."
            )
        rows: list[dict[str, Any]] = []
        for spec in specs:
            hidden_layer_sizes = self.config.ffnn.fast_architecture_for(
                self._lag_spec_input_dimension(spec)
            )
            for seed in seeds:
                for context in contexts:
                    fold = int(context["fold"])
                    fold_train = context["train"]
                    fold_validation = context["validation"]
                    try:
                        component_train = context["train_components"][component]
                        model, n_train = self._fit_model(
                            component_train,
                            spec,
                            hidden_layer_sizes,
                            float(alpha),
                            int(seed),
                            self.config.ffnn.activation,
                            learning_rate_init=float(learning_rate_init),
                            max_iter=int(self.config.ffnn.fast_max_iter),
                        )
                        prediction = self._predict_component_from_cache(
                            model,
                            component,
                            context["origin_cache"],
                            fold_validation,
                            spec,
                            f"block_subset_inner_fold_{fold}",
                            actual_components=context["actual_components"],
                        )
                        if len(prediction) < int(self.config.ffnn.min_val):
                            raise ValueError(
                                f"Only {len(prediction)} validation rows in "
                                "block-subset inner screen."
                            )
                        metrics = self._score_component_prediction(
                            prediction,
                            fold_validation,
                        )
                        row: dict[str, Any] = {
                            "component": component,
                            "lag_spec": spec.label,
                            "target_lags": spec.target_lags,
                            "exog_lags": spec.exog_lags,
                            "input_dimension": self._lag_spec_input_dimension(spec),
                            "seed": int(seed),
                            "fold": fold,
                            "train_start": fold_train.index.min(),
                            "train_end": fold_train.index.max(),
                            "validation_start": fold_validation.index.min(),
                            "validation_end": fold_validation.index.max(),
                            "hidden_layer_sizes": hidden_layer_sizes,
                            "alpha": float(alpha),
                            "activation": self.config.ffnn.activation,
                            "learning_rate_init": float(learning_rate_init),
                            "max_iter": int(self.config.ffnn.fast_max_iter),
                            "n_train": int(n_train),
                            "n_val": int(len(prediction)),
                            "screen_stage": (
                                "block_subset_two_stage_fast_screen"
                            ),
                            "screen_scale": "component_transformed",
                            "screen_validation_scope": "train_inner_rolling",
                            "selection_data_scope": "train_only_inner_rolling",
                            "error": None,
                        }
                        row.update(
                            {
                                f"Val {metric}": value
                                for metric, value in metrics.items()
                            }
                        )
                    except (ValueError, KeyError, IndexError) as exc:
                        row = {
                            "component": component,
                            "lag_spec": spec.label,
                            "target_lags": spec.target_lags,
                            "exog_lags": spec.exog_lags,
                            "input_dimension": self._lag_spec_input_dimension(spec),
                            "seed": int(seed),
                            "fold": fold,
                            "train_start": fold_train.index.min(),
                            "train_end": fold_train.index.max(),
                            "validation_start": fold_validation.index.min(),
                            "validation_end": fold_validation.index.max(),
                            "hidden_layer_sizes": hidden_layer_sizes,
                            "alpha": float(alpha),
                            "activation": self.config.ffnn.activation,
                            "learning_rate_init": float(learning_rate_init),
                            "max_iter": int(self.config.ffnn.fast_max_iter),
                            "n_train": 0,
                            "n_val": 0,
                            "screen_stage": (
                                "block_subset_two_stage_fast_screen"
                            ),
                            "screen_scale": "component_transformed",
                            "screen_validation_scope": "train_inner_rolling",
                            "selection_data_scope": "train_only_inner_rolling",
                            "error": str(exc),
                            "Val RMSE": float("nan"),
                            "Val MAE": float("nan"),
                            "Val MAPE": float("nan"),
                            "Val Directional_Accuracy": float("nan"),
                        }
                    rows.append(row)

        detail = pd.DataFrame(rows)
        valid = detail.dropna(subset=["Val RMSE"]).copy()
        expected_fold_count = len(contexts)
        completeness = (
            valid.groupby("lag_spec", as_index=False)
            .agg(
                seed_count=("seed", "nunique"),
                fold_count=("fold", "nunique"),
                valid_row_count=("Val RMSE", "count"),
            )
        )
        completeness = completeness[
            completeness["seed_count"].eq(len(seeds))
            & completeness["fold_count"].eq(expected_fold_count)
            & completeness["valid_row_count"].eq(
                len(seeds) * expected_fold_count
            )
        ].copy()
        complete_labels = set(completeness["lag_spec"].astype(str))
        valid = valid[valid["lag_spec"].astype(str).isin(complete_labels)]
        summary = (
            valid.groupby("lag_spec", as_index=False)
            .agg(
                mean_val_RMSE=("Val RMSE", "mean"),
                std_val_RMSE=("Val RMSE", "std"),
                mean_val_MAE=("Val MAE", "mean"),
                mean_val_MAPE=("Val MAPE", "mean"),
                seed_count=("seed", "nunique"),
                fold_count=("fold", "nunique"),
                input_dimension=("input_dimension", "first"),
            )
        )
        if summary.empty:
            raise ValueError(
                f"No complete inner rolling fast-screen result for "
                f"component={component}."
            )
        summary["std_val_RMSE"] = summary["std_val_RMSE"].fillna(0.0)
        summary = summary.sort_values(
            [
                "mean_val_RMSE",
                "mean_val_MAE",
                "mean_val_MAPE",
                "input_dimension",
                "lag_spec",
            ]
        ).reset_index(drop=True)
        summary.insert(0, "component", component)
        summary.insert(1, "fast_screen_rank", np.arange(1, len(summary) + 1))
        keep_count = min(max(1, int(top_k)), len(summary))
        summary["selected_for_full_grid"] = summary["fast_screen_rank"].le(
            keep_count
        )
        summary["fast_screen_alpha"] = float(alpha)
        summary["fast_screen_learning_rate_init"] = float(
            learning_rate_init
        )
        summary["fast_screen_seed_aggregation"] = (
            "mean_across_reporting_seeds_and_inner_folds"
        )
        summary["screen_validation_scope"] = "train_inner_rolling"

        selected_labels = set(
            summary.loc[
                summary["selected_for_full_grid"],
                "lag_spec",
            ].astype(str)
        )
        selected_specs = [spec for spec in specs if spec.label in selected_labels]
        detail["selected_for_full_grid"] = detail["lag_spec"].isin(
            selected_labels
        )
        return selected_specs, detail, summary

    def _fast_screen_specs_rolling_mean_seed(
        self,
        *,
        component: str,
        specs: list[LagSpec],
        seeds: tuple[int, ...],
        evaluate: Callable[[LagSpec, float, int], list[dict[str, Any]]],
    ) -> tuple[list[LagSpec], pd.DataFrame, pd.DataFrame]:
        """Create one train-only rolling shortlist shared by all reporting seeds."""
        fold_rows: list[dict[str, Any]] = []
        stability_start = len(self._lag_screen_stability_rows)
        for spec in specs:
            for learning_rate in self.config.ffnn.lag_screen_learning_rates():
                for seed in seeds:
                    for raw_row in evaluate(spec, float(learning_rate), int(seed)):
                        row = {
                            **raw_row,
                            "component": component,
                            "lag_spec": spec.label,
                            "target_lags": spec.target_lags,
                            "exog_lags": spec.exog_lags,
                            "input_dimension": self._lag_spec_input_dimension(spec),
                            "learning_rate_init": float(learning_rate),
                            "activation": self.config.ffnn.activation,
                            "alpha": float(self.config.ffnn.fast_alpha),
                            "max_iter": int(self.config.ffnn.fast_max_iter),
                            "seed": int(seed),
                            "selection_data_scope": "train_only_inner_rolling",
                            "selection_seed_aggregation": "mean_across_reporting_seeds",
                        }
                        fold_rows.append(row)
                        self._lag_screen_fold_rows.append(row)

        detail = pd.DataFrame(fold_rows)
        valid = detail.dropna(subset=["screen_RMSE"]).copy()
        expected_fold_count = int(detail["fold"].nunique()) if not detail.empty else 0
        complete = (
            valid.groupby(["lag_spec", "learning_rate_init"], as_index=False)
            .agg(
                seed_count=("seed", "nunique"),
                valid_row_count=("screen_RMSE", "count"),
            )
        )
        complete = complete[
            complete["seed_count"].eq(len(seeds))
            & complete["valid_row_count"].eq(len(seeds) * expected_fold_count)
        ]
        complete_keys = set(
            zip(
                complete["lag_spec"].astype(str),
                complete["learning_rate_init"].astype(float),
                strict=True,
            )
        )
        complete_rows = [
            row
            for row in fold_rows
            if (str(row["lag_spec"]), float(row["learning_rate_init"]))
            in complete_keys
        ]
        selected_labels, _ = self._summarize_lag_screen(
            component,
            specs,
            complete_rows,
        )
        if not selected_labels:
            raise ValueError(
                f"No complete mean-seed rolling lag screen for component={component}."
            )
        selected_specs = [spec for spec in specs if spec.label in selected_labels]
        detail["selected_for_full_grid"] = detail["lag_spec"].isin(selected_labels)
        summary = pd.DataFrame(self._lag_screen_stability_rows[stability_start:])
        if not summary.empty:
            summary["selection_seed_aggregation"] = "mean_across_reporting_seeds"
            summary["reporting_seed_count"] = len(seeds)
            summary["selected_for_full_grid"] = summary[
                "selected_for_hyperparam_search"
            ]
        return selected_specs, detail, summary

    def _rank_mean_seed_component_signatures(
        self,
        component_candidates: pd.DataFrame,
        seeds: tuple[int, ...],
        top_component_candidates: int | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Rank shared component signatures and flag the reconstructed shortlist."""
        if component_candidates.empty:
            raise ValueError("Cannot rank an empty component-candidate table.")
        component = str(component_candidates["component"].iloc[0])
        signature_columns = [
            "lag_spec",
            "hidden_layer_sizes",
            "alpha",
            "activation",
            "learning_rate_init",
        ]
        summary = (
            component_candidates.groupby(signature_columns, as_index=False, dropna=False)
            .agg(
                mean_component_val_RMSE=("Val RMSE", "mean"),
                std_component_val_RMSE=("Val RMSE", "std"),
                mean_component_val_MAE=("Val MAE", "mean"),
                mean_component_val_MAPE=("Val MAPE", "mean"),
                seed_count=("seed", "nunique"),
            )
        )
        summary = summary[summary["seed_count"].eq(len(seeds))].copy()
        if summary.empty:
            raise ValueError(
                f"No complete mean-seed candidate signature for component={component}."
            )
        summary["std_component_val_RMSE"] = summary[
            "std_component_val_RMSE"
        ].fillna(0.0)
        summary = summary.sort_values(
            [
                "mean_component_val_RMSE",
                "mean_component_val_MAE",
                "mean_component_val_MAPE",
                "lag_spec",
                "hidden_layer_sizes",
                "alpha",
                "learning_rate_init",
            ]
        ).reset_index(drop=True)
        summary.insert(0, "component", component)
        summary.insert(
            1,
            "component_mean_seed_rank",
            np.arange(1, len(summary) + 1),
        )
        summary.insert(
            2,
            "component_signature_id",
            [f"{component}_S{rank:04d}" for rank in summary["component_mean_seed_rank"]],
        )
        top_n = min(
            max(
                1,
                int(
                    top_component_candidates
                    if top_component_candidates is not None
                    else self.config.ffnn.top_component_candidates
                ),
            ),
            len(summary),
        )
        summary["shortlisted_for_reconstructed_search"] = summary[
            "component_mean_seed_rank"
        ].le(top_n)
        summary["selected_in_reconstructed_winner"] = False

        signature_id_by_key = {
            self._component_signature_key(row): str(row["component_signature_id"])
            for _, row in summary.iterrows()
        }
        ranked_candidates = component_candidates.copy()
        ranked_candidates["component_signature_id"] = [
            signature_id_by_key.get(self._component_signature_key(row))
            for _, row in ranked_candidates.iterrows()
        ]
        shortlisted_ids = set(
            summary.loc[
                summary["shortlisted_for_reconstructed_search"],
                "component_signature_id",
            ].astype(str)
        )
        ranked_candidates["eligible_shared_seed_selection"] = ranked_candidates[
            "component_signature_id"
        ].notna()
        ranked_candidates["shortlisted_for_reconstructed_search"] = ranked_candidates[
            "component_signature_id"
        ].isin(shortlisted_ids)
        ranked_candidates["selected_in_reconstructed_winner"] = False
        return ranked_candidates, summary

    def _score_mean_seed_reconstructed_combinations(
        self,
        candidates: pd.DataFrame,
        component_rankings: pd.DataFrame,
        shortlisted_predictions: dict[int, pd.DataFrame],
        validation: pd.DataFrame,
        seeds: tuple[int, ...],
        component_order: tuple[str, ...],
        top_component_candidates: int | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Score shared component-signature combinations on reconstructed level."""
        shortlisted = candidates[
            candidates["shortlisted_for_reconstructed_search"]
        ].copy()
        lookup_columns = ["component", "component_signature_id", "seed"]
        if shortlisted.duplicated(lookup_columns).any():
            raise ValueError("Duplicate candidate for a component signature and seed.")
        candidate_id_by_signature_seed = {
            (str(row["component"]), str(row["component_signature_id"]), int(row["seed"])): int(
                row["candidate_id"]
            )
            for _, row in shortlisted.iterrows()
        }

        top_signatures: dict[str, list[str]] = {}
        complexity_by_signature: dict[tuple[str, str], int] = {}
        for component in component_order:
            component_top = component_rankings[
                component_rankings["component"].eq(component)
                & component_rankings["shortlisted_for_reconstructed_search"]
            ].sort_values("component_mean_seed_rank")
            if component_top.empty:
                raise ValueError(
                    f"No shortlisted shared signature for component={component}."
                )
            top_signatures[component] = component_top[
                "component_signature_id"
            ].astype(str).tolist()
            for signature_id in top_signatures[component]:
                row = shortlisted[
                    shortlisted["component"].eq(component)
                    & shortlisted["component_signature_id"].eq(signature_id)
                ].iloc[0]
                complexity_by_signature[(component, signature_id)] = (
                    len(tuple(row["target_lags"]))
                    + sum(len(tuple(lags)) for lags in row["exog_lags"].values())
                )

        combo_metadata: list[dict[str, Any]] = []
        seed_score_rows: list[dict[str, Any]] = []
        for combo_id, signature_ids in enumerate(
            product(*(top_signatures[component] for component in component_order))
        ):
            signature_by_component = dict(zip(component_order, signature_ids, strict=True))
            metadata: dict[str, Any] = {
                "combo_id": int(combo_id),
                "component_signature_ids": tuple(signature_ids),
                "top_component_candidates": int(
                    top_component_candidates
                    if top_component_candidates is not None
                    else self.config.ffnn.top_component_candidates
                ),
                "total_input_dimension": sum(
                    complexity_by_signature[(component, signature_by_component[component])]
                    for component in component_order
                ),
            }
            metadata.update(
                {
                    f"{component}_signature_id": signature_by_component[component]
                    for component in component_order
                }
            )
            combo_metadata.append(metadata)

            for seed in seeds:
                candidate_ids = tuple(
                    candidate_id_by_signature_seed[
                        (component, signature_by_component[component], int(seed))
                    ]
                    for component in component_order
                )
                component_frames = [
                    shortlisted_predictions[candidate_id]
                    for candidate_id in candidate_ids
                ]
                reconstructed = self._reconstruct(
                    pd.concat(component_frames, ignore_index=True, sort=False),
                    validation,
                )
                transformed_metrics = evaluate_forecast(
                    reconstructed["actual_raw_transformed"],
                    reconstructed["predicted_reconstructed"],
                )
                actual_level, predicted_level = self._to_level(
                    reconstructed,
                    "actual_raw_transformed",
                    "predicted_reconstructed",
                )
                level_metrics = evaluate_forecast(actual_level, predicted_level)
                score_row: dict[str, Any] = {
                    "combo_id": int(combo_id),
                    "seed": int(seed),
                    "component_candidate_ids": candidate_ids,
                }
                score_row.update(
                    {f"Val {metric}": value for metric, value in transformed_metrics.items()}
                )
                score_row.update(
                    {f"Val Level {metric}": value for metric, value in level_metrics.items()}
                )
                seed_score_rows.append(score_row)

        seed_scores = pd.DataFrame(seed_score_rows)
        combo_summary = (
            seed_scores.groupby("combo_id", as_index=False)
            .agg(
                mean_val_level_RMSE=("Val Level RMSE", "mean"),
                std_val_level_RMSE=("Val Level RMSE", "std"),
                mean_val_level_MAE=("Val Level MAE", "mean"),
                mean_val_level_MAPE=("Val Level MAPE", "mean"),
                mean_val_level_Directional_Accuracy=(
                    "Val Level Directional_Accuracy",
                    "mean",
                ),
                mean_val_transformed_RMSE=("Val RMSE", "mean"),
                seed_count=("seed", "nunique"),
            )
            .merge(pd.DataFrame(combo_metadata), on="combo_id", how="left")
        )
        combo_summary["std_val_level_RMSE"] = combo_summary[
            "std_val_level_RMSE"
        ].fillna(0.0)
        combo_summary = combo_summary.sort_values(
            [
                "mean_val_level_RMSE",
                "mean_val_level_MAE",
                "mean_val_level_MAPE",
                "total_input_dimension",
                "combo_id",
            ]
        ).reset_index(drop=True)
        combo_summary.insert(
            1,
            "mean_seed_reconstructed_rank",
            np.arange(1, len(combo_summary) + 1),
        )
        combo_summary["selected"] = combo_summary[
            "mean_seed_reconstructed_rank"
        ].eq(1)
        combo_summary["final_selection_scale"] = "level"
        combo_summary["selection_protocol"] = (
            "mean_reconstructed_level_validation_across_reporting_seeds"
        )
        return combo_summary, seed_scores

    def run_block_subset_mean_seed(
        self,
        csv_path: str | Path,
        *,
        reporting_seeds: tuple[int, ...],
        search_strategy: str = "full_grid",
        fast_screen_validation_scope: str = "outer_validation",
    ) -> dict[str, pd.DataFrame]:
        """Run the AIC variable-block subset comparator with shared seed selection.

        Target lags are selected once per VMD component. AIC then selects one
        distributed-lag order for every exogenous variable, all non-empty subsets
        of those variable blocks are enumerated, and the full FFNN grid is scored.
        Candidate signatures are first shortlisted per component using mean
        component validation metrics across seeds. Their cross-component
        combinations are then ranked by mean reconstructed level validation RMSE
        across the same seeds, never by the best individual seed. The locked
        reconstructed winner is refit/evaluated for every seed.
        """
        seeds = tuple(dict.fromkeys(int(seed) for seed in reporting_seeds))
        block_search_strategy = search_strategy.replace("-", "_")
        fast_screen_scope = fast_screen_validation_scope.replace("-", "_")
        if not seeds:
            raise ValueError("Block-subset mean-seed selection requires reporting seeds.")
        if block_search_strategy not in {"full_grid", "two_stage"}:
            raise ValueError(
                "Block-subset search_strategy must be 'full_grid' or 'two_stage'."
            )
        if fast_screen_scope not in {
            "outer_validation",
            "train_inner_rolling",
        }:
            raise ValueError(
                "fast_screen_validation_scope must be 'outer_validation' or "
                "'train_inner_rolling'."
            )
        if self.config.ardl.exog_lag_strategy.replace("-", "_") != "ardl_block_subsets":
            raise ValueError(
                "run_block_subset_mean_seed requires exog_lag_strategy='ardl_block_subsets'."
            )
        if self.config.ardl.ic.lower() != "aic":
            raise ValueError("Block-subset variable orders must be selected with AIC.")
        if self.config.ffnn.search_strategy.replace("-", "_") != "full_grid":
            raise ValueError(
                "Block-subset mean-seed selection requires search_strategy='full_grid'."
            )
        if tuple(self.config.ffnn.seed_grid) != seeds:
            raise ValueError("FFNN seed_grid must exactly match reporting_seeds.")

        total_started = perf_counter()
        self._log(
            "Starting AIC variable-block subset comparator with mean reconstructed "
            f"level validation selection across seeds={seeds}; "
            f"search_strategy={block_search_strategy}; "
            f"fast_screen_validation_scope={fast_screen_scope}."
        )
        base_data = self.load_data(csv_path)
        base_train, base_val, base_test = self.split_raw_data(base_data)
        data = self._apply_stationarity_transform(base_data, base_train)
        train = data.loc[data.index.intersection(base_train.index)].copy()
        val = data.loc[data.index.intersection(base_val.index)].copy()
        test = data.loc[data.index.intersection(base_test.index)].copy()

        audit: list[dict[str, Any]] = []
        for split, frame in (("train", train), ("validation", val), ("test", test)):
            audit.extend(self._audit_rows("block_subset_mean_seed", "chronological_split", split, frame))

        vmd_cache_started = perf_counter()
        train_components = self.decompose(train)
        val_cache = self._make_origin_cache(train, val, "block-subset-validation")
        val_actual_components = self._make_actual_component_cache(
            train,
            val,
            "block-subset-validation",
        )
        vmd_cache_seconds = perf_counter() - vmd_cache_started

        selection_started = perf_counter()
        self._select_specs(train_components)
        ardl_selection_seconds = perf_counter() - selection_started

        ffnn_search_started = perf_counter()
        candidate_frames: list[pd.DataFrame] = []
        component_ranking_frames: list[pd.DataFrame] = []
        shortlisted_predictions: dict[int, pd.DataFrame] = {}
        fast_screen_detail_frames: list[pd.DataFrame] = []
        fast_screen_summary_frames: list[pd.DataFrame] = []
        candidate_id = 0
        inner_screen_contexts = (
            self._prepare_vmd_lag_screen_contexts(train)
            if block_search_strategy == "two_stage"
            and fast_screen_scope == "train_inner_rolling"
            else []
        )
        if (
            block_search_strategy == "two_stage"
            and fast_screen_scope == "train_inner_rolling"
            and not inner_screen_contexts
        ):
            raise ValueError(
                "No train-only inner rolling folds are available for the "
                "block-subset fast screen."
            )
        for component, train_component in train_components.items():
            component_specs = self.selected_specs_[component]
            if block_search_strategy == "two_stage":
                if fast_screen_scope == "train_inner_rolling":
                    component_specs, fast_detail, fast_summary = (
                        self._fast_screen_block_subset_specs_inner_rolling(
                            component=component,
                            specs=component_specs,
                            contexts=inner_screen_contexts,
                            seeds=seeds,
                            top_k=4,
                            alpha=0.1,
                            learning_rate_init=0.001,
                        )
                    )
                else:
                    component_specs, fast_detail, fast_summary = (
                        self._fast_screen_block_subset_specs(
                            component=component,
                            train_data=train_component,
                            specs=component_specs,
                            validation=val,
                            validation_cache=val_cache,
                            validation_actual_components=val_actual_components,
                            seeds=seeds,
                            top_k=4,
                            alpha=0.1,
                            learning_rate_init=0.001,
                        )
                    )
                fast_screen_detail_frames.append(fast_detail)
                fast_screen_summary_frames.append(fast_summary)
            rows, predictions, _, candidate_id = self._ffnn_search_candidates(
                component=component,
                train_data=train_component,
                specs=component_specs,
                prediction_fn=lambda model, spec, row_id, component=component: self._predict_component_from_cache(
                    model,
                    component,
                    val_cache,
                    val,
                    spec,
                    "validation",
                    row_id,
                    val_actual_components,
                ),
                scoring_fn=lambda pred: self._score_component_prediction(pred, val),
                uses_vmd=True,
                candidate_id_start=candidate_id,
            )
            component_candidates = pd.DataFrame(rows)
            ranked_candidates, component_ranking = (
                self._rank_mean_seed_component_signatures(
                    component_candidates,
                    seeds,
                )
            )
            candidate_frames.append(ranked_candidates)
            component_ranking_frames.append(component_ranking)
            shortlisted_candidate_ids = set(
                ranked_candidates.loc[
                    ranked_candidates["shortlisted_for_reconstructed_search"],
                    "candidate_id",
                ].astype(int)
            )
            shortlisted_predictions.update(
                {
                    candidate: predictions[candidate]
                    for candidate in shortlisted_candidate_ids
                }
            )

        fast_screen_detail_df = (
            pd.concat(fast_screen_detail_frames, ignore_index=True, sort=False)
            if fast_screen_detail_frames
            else pd.DataFrame()
        )
        fast_screen_summary_df = (
            pd.concat(fast_screen_summary_frames, ignore_index=True, sort=False)
            if fast_screen_summary_frames
            else pd.DataFrame()
        )
        candidates_df = pd.concat(candidate_frames, ignore_index=True, sort=False)
        if candidates_df.empty:
            raise ValueError("No FFNN candidate could be scored for block-subset selection.")
        component_rankings_df = pd.concat(
            component_ranking_frames,
            ignore_index=True,
            sort=False,
        )
        selection_df, reconstructed_seed_scores = (
            self._score_mean_seed_reconstructed_combinations(
                candidates_df,
                component_rankings_df,
                shortlisted_predictions,
                val,
                seeds,
                tuple(train_components),
            )
        )
        for frame in (
            candidates_df,
            component_rankings_df,
            reconstructed_seed_scores,
            selection_df,
        ):
            frame["block_subset_search_strategy"] = block_search_strategy
            frame["block_subset_fast_screen_top_k"] = (
                4 if block_search_strategy == "two_stage" else pd.NA
            )
            frame["block_subset_fast_screen_alpha"] = (
                0.1 if block_search_strategy == "two_stage" else pd.NA
            )
            frame["block_subset_fast_screen_learning_rate_init"] = (
                0.001 if block_search_strategy == "two_stage" else pd.NA
            )
            frame["block_subset_fast_screen_validation_scope"] = (
                fast_screen_scope
                if block_search_strategy == "two_stage"
                else pd.NA
            )
        winning_combo = selection_df.iloc[0]
        selected_signature_by_component = {
            component: str(winning_combo[f"{component}_signature_id"])
            for component in train_components
        }
        selected_mask = pd.Series(False, index=candidates_df.index)
        for component, signature_id in selected_signature_by_component.items():
            selected_mask |= candidates_df["component"].eq(component) & candidates_df[
                "component_signature_id"
            ].eq(signature_id)
        best_candidates = candidates_df[selected_mask].copy()
        if len(best_candidates) != len(train_components) * len(seeds):
            raise ValueError(
                "Reconstructed mean-seed winner does not cover every component and seed."
            )
        candidates_df["selected_in_reconstructed_winner"] = selected_mask
        best_candidates["selected_in_reconstructed_winner"] = True
        for component, signature_id in selected_signature_by_component.items():
            component_rankings_df.loc[
                component_rankings_df["component"].eq(component)
                & component_rankings_df["component_signature_id"].eq(signature_id),
                "selected_in_reconstructed_winner",
            ] = True
        best_candidates["combo_id"] = int(winning_combo["combo_id"])
        best_candidates["final_selection_scale"] = "level"
        best_candidates["combo_mean_seed_val_level_RMSE"] = float(
            winning_combo["mean_val_level_RMSE"]
        )
        best_candidates["combo_std_seed_val_level_RMSE"] = float(
            winning_combo["std_val_level_RMSE"]
        )
        best_candidates["combo_mean_seed_val_level_MAE"] = float(
            winning_combo["mean_val_level_MAE"]
        )
        best_candidates["combo_mean_seed_val_level_MAPE"] = float(
            winning_combo["mean_val_level_MAPE"]
        )
        best_candidates["combo_total_input_dimension"] = int(
            winning_combo["total_input_dimension"]
        )
        best_candidates["selection_seed_count"] = len(seeds)
        best_candidates["block_subset_search_strategy"] = block_search_strategy
        best_candidates["block_subset_fast_screen_validation_scope"] = (
            fast_screen_scope
            if block_search_strategy == "two_stage"
            else pd.NA
        )
        best_candidates["selection_protocol"] = (
            "mean_reconstructed_level_validation_across_reporting_seeds"
        )
        ffnn_search_seconds = perf_counter() - ffnn_search_started

        refit_started = perf_counter()
        history_before_test = pd.concat([train, val], axis=0)
        history_components = self.decompose(history_before_test)
        test_cache = self._make_origin_cache(
            history_before_test,
            test,
            "block-subset-test",
        )
        test_actual_components = self._make_actual_component_cache(
            history_before_test,
            test,
            "block-subset-test",
        )
        selected_component_predictions: list[pd.DataFrame] = []
        for row_index, best in best_candidates.iterrows():
            component = str(best["component"])
            seed = int(best["seed"])
            spec = LagSpec(
                tuple(best["target_lags"]),
                {feature: tuple(lags) for feature, lags in best["exog_lags"].items()},
            )
            validation_prediction = shortlisted_predictions[
                int(best["candidate_id"])
            ].copy()
            validation_prediction["seed"] = seed
            selected_component_predictions.append(validation_prediction)

            test_model, n_refit = self._fit_model(
                history_components[component],
                spec,
                self._hidden_layer_sizes_from_row(best),
                float(best["alpha"]),
                seed,
                str(best["activation"]),
                learning_rate_init=float(best["learning_rate_init"]),
            )
            test_prediction = self._predict_component_from_cache(
                test_model,
                component,
                test_cache,
                test,
                spec,
                "test",
                actual_components=test_actual_components,
            )
            test_prediction["seed"] = seed
            selected_component_predictions.append(test_prediction)
            best_candidates.loc[row_index, "n_refit_train_val"] = int(n_refit)

        self.component_forecasts_ = pd.concat(
            selected_component_predictions,
            ignore_index=True,
            sort=False,
        )
        actual_all = pd.concat([val, test], axis=0)
        self.final_forecasts_ = self._reconstruct(
            self.component_forecasts_,
            actual_all,
        ).reset_index(drop=True)
        self.final_forecasts_["forecast_protocol"] = self.config.forecast_protocol

        metric_frames: list[pd.DataFrame] = []
        for seed in seeds:
            index = self.final_forecasts_.index[self.final_forecasts_["seed"].eq(seed)]
            seed_forecasts = self.final_forecasts_.loc[index].copy()
            seed_best = best_candidates[best_candidates["seed"].eq(seed)].copy()
            metrics = self._annotate_final_metrics(
                self._metrics_rows(
                    seed_forecasts,
                    "actual_raw_transformed",
                    "predicted_reconstructed",
                    "reconstructed_transformed",
                ),
                csv_path,
                "VMD_ARDL_BlockSubset_MeanSeed_FFNN",
                True,
                seed_best,
            )
            metrics["seed"] = seed
            metric_frames.append(metrics)
            for column in ("actual_level", "predicted_level"):
                if column in seed_forecasts:
                    self.final_forecasts_.loc[index, column] = seed_forecasts[column].to_numpy()

        self.final_metrics_ = pd.concat(metric_frames, ignore_index=True, sort=False)
        self.best_component_models_ = best_candidates.sort_values(
            ["seed", "component"]
        ).reset_index(drop=True)
        self.input_audit_ = pd.DataFrame(audit)
        refit_forecast_seconds = perf_counter() - refit_started

        ardl_orders = pd.concat(
            [table.assign(component=name) for name, table in self.order_tables_.items()],
            ignore_index=True,
        )
        diagnostics = self._diagnostics(
            train,
            train_components,
            "_block_subset_mean_seed",
        )
        timing = pd.DataFrame(
            [
                {
                    "ardl_selection_seconds": ardl_selection_seconds,
                    "vmd_cache_seconds": vmd_cache_seconds,
                    "ffnn_search_seconds": ffnn_search_seconds,
                    "refit_forecast_seconds": refit_forecast_seconds,
                    "total_runtime_seconds": perf_counter() - total_started,
                }
            ]
        )

        write_csv(candidates_df, self.output_dir / "ffnn_block_subset_component_search.csv")
        if not fast_screen_detail_df.empty:
            write_csv(
                fast_screen_detail_df,
                self.output_dir / "ffnn_block_subset_fast_screen_metrics.csv",
            )
        if not fast_screen_summary_df.empty:
            write_csv(
                fast_screen_summary_df,
                self.output_dir / "ffnn_block_subset_fast_screen_summary.csv",
            )
        write_csv(
            component_rankings_df,
            self.output_dir / "ffnn_block_subset_component_mean_seed_ranking.csv",
        )
        write_csv(
            reconstructed_seed_scores,
            self.output_dir / "ffnn_block_subset_reconstructed_seed_scores.csv",
        )
        write_csv(selection_df, self.output_dir / "ffnn_block_subset_mean_seed_selection.csv")
        write_csv(
            self.best_component_models_,
            self.output_dir / "best_component_models_block_subset_mean_seed.csv",
        )
        write_csv(
            self.component_forecasts_,
            self.output_dir / "component_predictions_block_subset_mean_seed.csv",
        )
        write_csv(
            self.final_forecasts_,
            self.output_dir / "final_reconstructed_forecasts_block_subset_mean_seed.csv",
        )
        write_csv(
            self.final_metrics_,
            self.output_dir / "final_reconstructed_metrics_block_subset_mean_seed.csv",
        )
        write_csv(ardl_orders, self.output_dir / "ardl_variable_blocks_train_only.csv")
        write_csv(self.input_audit_, self.output_dir / "input_audit_block_subset_mean_seed.csv")
        if self.stationarity_screen_ is not None:
            write_csv(
                self.stationarity_screen_,
                self.output_dir / "stationarity_screen_train_only_block_subset_mean_seed.csv",
            )
        for name, frame in diagnostics.items():
            write_csv(frame, self.output_dir / f"{name}_train_only.csv")
        write_csv(timing, self.output_dir / "runtime_breakdown.csv")

        return {
            "search_results": selection_df,
            "component_search_results": candidates_df,
            "component_signature_rankings": component_rankings_df,
            "reconstructed_seed_scores": reconstructed_seed_scores,
            "block_subset_fast_screen_metrics": fast_screen_detail_df,
            "block_subset_fast_screen_summary": fast_screen_summary_df,
            "ardl_orders": ardl_orders,
            "best_component_models": self.best_component_models_,
            "component_forecasts": self.component_forecasts_,
            "training_components": train_components,
            "final_forecasts": self.final_forecasts_,
            "final_metrics": self.final_metrics_,
            "input_audit": self.input_audit_,
            "stationarity_screen": self.stationarity_screen_,
            "lag_screen_fold_metrics": pd.DataFrame(),
            "lag_screen_stability": pd.DataFrame(),
            "timing": timing,
            **diagnostics,
        }

    def run_hard_ffnn_comparator(
        self,
        csv_path: str | Path,
        *,
        reporting_seeds: tuple[int, ...],
        model_name: str = "VMD_ARDL_HardConfig_FFNN",
    ) -> dict[str, pd.DataFrame]:
        """Run full lag-combination search with one train-only FFNN config/spec."""
        seeds = tuple(dict.fromkeys(int(seed) for seed in reporting_seeds))
        if not seeds:
            raise ValueError("Hard-FFNN comparator requires reporting seeds.")
        if tuple(self.config.ffnn.seed_grid) != seeds:
            raise ValueError("FFNN seed_grid must exactly match reporting_seeds.")
        ardl = self.config.ardl
        if ardl.target_lag_strategy.replace("-", "_") != "ardl_select_order":
            raise ValueError(
                "Hard-FFNN comparator requires target_lag_strategy="
                "'ardl_select_order'."
            )
        if ardl.exog_lag_strategy.replace("-", "_") != "ardl_screen":
            raise ValueError(
                "Hard-FFNN comparator requires exog_lag_strategy='ardl_screen'."
            )
        if ardl.lag_spec_strategy.replace("-", "_") != "full_product":
            raise ValueError(
                "Hard-FFNN comparator requires lag_spec_strategy='full_product'."
            )
        if int(ardl.top_n) != 3:
            raise ValueError("Hard-FFNN comparator requires exactly top_n=3.")
        if ardl.ic.lower() != "aic":
            raise ValueError("Hard-FFNN comparator requires AIC lag ranking.")

        total_started = perf_counter()
        self._log(
            "Starting train-only frequency/lag hard-FFNN comparator with "
            f"full-product lag search across seeds={seeds}."
        )
        base_data = self.load_data(csv_path)
        base_train, base_val, base_test = self.split_raw_data(base_data)
        data = self._apply_stationarity_transform(base_data, base_train)
        train = data.loc[data.index.intersection(base_train.index)].copy()
        val = data.loc[data.index.intersection(base_val.index)].copy()
        test = data.loc[data.index.intersection(base_test.index)].copy()

        audit: list[dict[str, Any]] = []
        for split, frame in (("train", train), ("validation", val), ("test", test)):
            audit.extend(
                self._audit_rows(
                    "hard_ffnn",
                    "chronological_split",
                    split,
                    frame,
                )
            )

        vmd_cache_started = perf_counter()
        train_components = self.decompose(train)
        descriptors = describe_target_imfs(train_components, self.target)
        descriptor_by_component = {
            str(row["component"]): row
            for _, row in descriptors.iterrows()
        }
        val_cache = self._make_origin_cache(train, val, "hard-ffnn-validation")
        val_actual_components = self._make_actual_component_cache(
            train,
            val,
            "hard-ffnn-validation",
        )
        vmd_cache_seconds = perf_counter() - vmd_cache_started

        selection_started = perf_counter()
        self._select_specs(train_components)
        ardl_selection_seconds = perf_counter() - selection_started
        policy = HardFFNNPolicy(self.config.hard_ffnn)
        top_component_candidates = int(
            self.config.hard_ffnn.top_component_candidates
        )

        ffnn_search_started = perf_counter()
        candidate_frames: list[pd.DataFrame] = []
        component_ranking_frames: list[pd.DataFrame] = []
        retained_predictions: dict[int, pd.DataFrame] = {}
        assignment_rows: list[dict[str, Any]] = []
        fit_error_rows: list[dict[str, Any]] = []
        candidate_id = 0
        for component, train_component in train_components.items():
            descriptor = descriptor_by_component[component]
            rows: list[dict[str, Any]] = []
            predictions: dict[int, pd.DataFrame] = {}
            specs = self.selected_specs_[component]
            for spec_rank, spec in enumerate(specs, start=1):
                supervised = self._build_supervised(train_component, spec)
                input_dimension = self._lag_spec_input_dimension(spec)
                all_lags = [
                    *spec.target_lags,
                    *(
                        lag
                        for feature_lags in spec.exog_lags.values()
                        for lag in feature_lags
                    ),
                ]
                max_lag = max(int(lag) for lag in all_lags)
                decision = policy.decide(
                    descriptor,
                    input_dimension=input_dimension,
                    max_lag=max_lag,
                    effective_train_samples=len(supervised),
                )
                assignment_rows.append(
                    {
                        **descriptor.to_dict(),
                        **decision.as_row(),
                        "lag_spec_rank": int(spec_rank),
                        "lag_spec": spec.label,
                        "target_lags": spec.target_lags,
                        "exog_lags": spec.exog_lags,
                        "ffnn_config_source": "train_only_frequency_lag_policy",
                    }
                )
                for seed in seeds:
                    current_id = candidate_id
                    try:
                        model, n_train = self._fit_model(
                            train_component,
                            spec,
                            decision.hidden_layer_sizes,
                            decision.alpha,
                            seed,
                            decision.activation,
                            learning_rate_init=decision.learning_rate_init,
                            max_iter=int(self.config.ffnn.max_iter),
                        )
                        prediction = self._predict_component_from_cache(
                            model,
                            component,
                            val_cache,
                            val,
                            spec,
                            "validation",
                            current_id,
                            val_actual_components,
                        )
                        if len(prediction) < int(self.config.ffnn.min_val):
                            raise ValueError(
                                "Hard-FFNN validation prediction has fewer rows "
                                f"than min_val={self.config.ffnn.min_val}."
                            )
                    except (ValueError, KeyError, IndexError) as exc:
                        fit_error_rows.append(
                            {
                                "component": component,
                                "lag_spec": spec.label,
                                "seed": int(seed),
                                "error": str(exc),
                            }
                        )
                        candidate_id += 1
                        continue
                    metrics = self._score_component_prediction(prediction, val)
                    row = {
                        "candidate_id": int(current_id),
                        "component": component,
                        "lag_spec": spec.label,
                        "target_lags": spec.target_lags,
                        "exog_lags": spec.exog_lags,
                        "HR": int(decision.hidden_layer_sizes[0]),
                        "hidden_layers": len(decision.hidden_layer_sizes),
                        "hidden_units_per_layer": int(
                            decision.hidden_layer_sizes[0]
                        ),
                        "hidden_layer_sizes": decision.hidden_layer_sizes,
                        "alpha": float(decision.alpha),
                        "seed": int(seed),
                        "activation": decision.activation,
                        "learning_rate_init": float(
                            decision.learning_rate_init
                        ),
                        "max_iter": int(self.config.ffnn.max_iter),
                        "n_train": int(n_train),
                        "n_val": int(len(prediction)),
                        "input_dimension": int(decision.input_dimension),
                        "max_lag": int(decision.max_lag),
                        "effective_train_samples": int(
                            decision.effective_train_samples
                        ),
                        "input_sample_ratio": float(
                            decision.input_sample_ratio
                        ),
                        "hard_ffnn_regime": decision.regime,
                        "hard_ffnn_policy_version": decision.policy_version,
                        "hard_ffnn_policy_reason": decision.policy_reason,
                        "heavy_lag_load": bool(decision.heavy_lag_load),
                        "ffnn_config_source": "train_only_frequency_lag_policy",
                        "search_strategy": "hard_config_full_lag_product",
                        "search_stage": "hard_config_validation",
                        "selected_for_hyperparam_search": False,
                        "forecast_safe_no_exog_lag0": True,
                        "uses_vmd": True,
                    }
                    row.update(
                        {f"Val {metric}": value for metric, value in metrics.items()}
                    )
                    rows.append(row)
                    predictions[current_id] = prediction
                    candidate_id += 1

            component_candidates = pd.DataFrame(rows)
            if component_candidates.empty:
                raise ValueError(
                    f"No hard-FFNN candidate could be scored for {component}."
                )
            ranked_candidates, component_ranking = (
                self._rank_mean_seed_component_signatures(
                    component_candidates,
                    seeds,
                    top_component_candidates=top_component_candidates,
                )
            )
            candidate_frames.append(ranked_candidates)
            component_ranking_frames.append(component_ranking)
            retained_ids = set(
                ranked_candidates.loc[
                    ranked_candidates["shortlisted_for_reconstructed_search"],
                    "candidate_id",
                ].astype(int)
            )
            retained_predictions.update(
                {
                    retained_id: predictions[retained_id]
                    for retained_id in retained_ids
                }
            )

        candidates_df = pd.concat(candidate_frames, ignore_index=True, sort=False)
        component_rankings_df = pd.concat(
            component_ranking_frames,
            ignore_index=True,
            sort=False,
        )
        assignments = pd.DataFrame(assignment_rows)
        fit_errors = pd.DataFrame(fit_error_rows)
        policy_config = pd.DataFrame([asdict(self.config.hard_ffnn)])
        selection_df, reconstructed_seed_scores = (
            self._score_mean_seed_reconstructed_combinations(
                candidates_df,
                component_rankings_df,
                retained_predictions,
                val,
                seeds,
                tuple(train_components),
                top_component_candidates=top_component_candidates,
            )
        )
        for frame in (
            candidates_df,
            component_rankings_df,
            reconstructed_seed_scores,
            selection_df,
        ):
            frame["ffnn_config_source"] = "train_only_frequency_lag_policy"
            frame["hard_ffnn_policy_version"] = (
                self.config.hard_ffnn.policy_version
            )
            frame["lag_spec_search_strategy"] = "full_product_top3_per_feature"

        winning_combo = selection_df.iloc[0]
        selected_signature_by_component = {
            component: str(winning_combo[f"{component}_signature_id"])
            for component in train_components
        }
        selected_mask = pd.Series(False, index=candidates_df.index)
        for component, signature_id in selected_signature_by_component.items():
            selected_mask |= candidates_df["component"].eq(component) & (
                candidates_df["component_signature_id"].eq(signature_id)
            )
            component_rankings_df.loc[
                component_rankings_df["component"].eq(component)
                & component_rankings_df["component_signature_id"].eq(
                    signature_id
                ),
                "selected_in_reconstructed_winner",
            ] = True
        best_candidates = candidates_df[selected_mask].copy()
        if len(best_candidates) != len(train_components) * len(seeds):
            raise ValueError(
                "Hard-FFNN reconstructed winner does not cover every "
                "component and reporting seed."
            )
        candidates_df["selected_in_reconstructed_winner"] = selected_mask
        best_candidates["selected_in_reconstructed_winner"] = True
        best_candidates["combo_id"] = int(winning_combo["combo_id"])
        best_candidates["final_selection_scale"] = "level"
        best_candidates["combo_mean_seed_val_level_RMSE"] = float(
            winning_combo["mean_val_level_RMSE"]
        )
        best_candidates["combo_std_seed_val_level_RMSE"] = float(
            winning_combo["std_val_level_RMSE"]
        )
        best_candidates["combo_mean_seed_val_level_MAE"] = float(
            winning_combo["mean_val_level_MAE"]
        )
        best_candidates["combo_mean_seed_val_level_MAPE"] = float(
            winning_combo["mean_val_level_MAPE"]
        )
        best_candidates["combo_total_input_dimension"] = int(
            winning_combo["total_input_dimension"]
        )
        best_candidates["selection_seed_count"] = len(seeds)
        best_candidates["selection_protocol"] = (
            "mean_reconstructed_level_validation_across_reporting_seeds"
        )
        ffnn_search_seconds = perf_counter() - ffnn_search_started

        refit_started = perf_counter()
        history_before_test = pd.concat([train, val], axis=0)
        history_components = self.decompose(history_before_test)
        test_cache = self._make_origin_cache(
            history_before_test,
            test,
            "hard-ffnn-test",
        )
        test_actual_components = self._make_actual_component_cache(
            history_before_test,
            test,
            "hard-ffnn-test",
        )
        selected_component_predictions: list[pd.DataFrame] = []
        for row_index, best in best_candidates.iterrows():
            component = str(best["component"])
            seed = int(best["seed"])
            spec = LagSpec(
                tuple(best["target_lags"]),
                {
                    feature: tuple(lags)
                    for feature, lags in best["exog_lags"].items()
                },
            )
            validation_prediction = retained_predictions[
                int(best["candidate_id"])
            ].copy()
            validation_prediction["seed"] = seed
            selected_component_predictions.append(validation_prediction)
            test_model, n_refit = self._fit_model(
                history_components[component],
                spec,
                self._hidden_layer_sizes_from_row(best),
                float(best["alpha"]),
                seed,
                str(best["activation"]),
                learning_rate_init=float(best["learning_rate_init"]),
                max_iter=int(best["max_iter"]),
            )
            test_prediction = self._predict_component_from_cache(
                test_model,
                component,
                test_cache,
                test,
                spec,
                "test",
                actual_components=test_actual_components,
            )
            test_prediction["seed"] = seed
            selected_component_predictions.append(test_prediction)
            best_candidates.loc[row_index, "n_refit_train_val"] = int(n_refit)

        self.component_forecasts_ = pd.concat(
            selected_component_predictions,
            ignore_index=True,
            sort=False,
        )
        self.final_forecasts_ = self._reconstruct(
            self.component_forecasts_,
            pd.concat([val, test], axis=0),
        ).reset_index(drop=True)
        self.final_forecasts_["forecast_protocol"] = self.config.forecast_protocol

        metric_frames: list[pd.DataFrame] = []
        for seed in seeds:
            index = self.final_forecasts_.index[
                self.final_forecasts_["seed"].eq(seed)
            ]
            seed_forecasts = self.final_forecasts_.loc[index].copy()
            seed_best = best_candidates[
                best_candidates["seed"].eq(seed)
            ].copy()
            metrics = self._annotate_final_metrics(
                self._metrics_rows(
                    seed_forecasts,
                    "actual_raw_transformed",
                    "predicted_reconstructed",
                    "reconstructed_transformed",
                ),
                csv_path,
                model_name,
                True,
                seed_best,
            )
            metrics["seed"] = seed
            metrics["ffnn_config_source"] = (
                "train_only_frequency_lag_policy"
            )
            metrics["hard_ffnn_policy_version"] = (
                self.config.hard_ffnn.policy_version
            )
            metric_frames.append(metrics)
            for column in ("actual_level", "predicted_level"):
                if column in seed_forecasts:
                    self.final_forecasts_.loc[index, column] = (
                        seed_forecasts[column].to_numpy()
                    )

        self.final_metrics_ = pd.concat(
            metric_frames,
            ignore_index=True,
            sort=False,
        )
        self.best_component_models_ = best_candidates.sort_values(
            ["seed", "component"]
        ).reset_index(drop=True)
        self.input_audit_ = pd.DataFrame(audit)
        refit_forecast_seconds = perf_counter() - refit_started

        ardl_orders = pd.concat(
            [
                table.assign(component=component)
                for component, table in self.order_tables_.items()
            ],
            ignore_index=True,
        )
        diagnostics = self._diagnostics(
            train,
            train_components,
            "_hard_ffnn",
        )
        timing = pd.DataFrame(
            [
                {
                    "ardl_selection_seconds": ardl_selection_seconds,
                    "vmd_cache_seconds": vmd_cache_seconds,
                    "ffnn_search_seconds": ffnn_search_seconds,
                    "refit_forecast_seconds": refit_forecast_seconds,
                    "total_runtime_seconds": perf_counter() - total_started,
                    "evaluated_ffnn_fits": int(
                        len(candidates_df) + len(fit_errors)
                    ),
                    "hard_policy_assignments": int(len(assignments)),
                    "reconstructed_combinations": int(len(selection_df)),
                }
            ]
        )

        write_csv(descriptors, self.output_dir / "hard_ffnn_imf_descriptors.csv")
        write_csv(
            policy_config,
            self.output_dir / "hard_ffnn_policy_config.csv",
        )
        write_csv(assignments, self.output_dir / "hard_ffnn_policy_assignments.csv")
        if not fit_errors.empty:
            write_csv(fit_errors, self.output_dir / "hard_ffnn_fit_errors.csv")
        write_csv(
            candidates_df,
            self.output_dir / "hard_ffnn_component_candidates.csv",
        )
        write_csv(
            component_rankings_df,
            self.output_dir / "hard_ffnn_component_shortlist.csv",
        )
        write_csv(
            reconstructed_seed_scores,
            self.output_dir / "hard_ffnn_reconstructed_seed_scores.csv",
        )
        write_csv(
            selection_df,
            self.output_dir / "hard_ffnn_shared_selection.csv",
        )
        write_csv(
            self.best_component_models_,
            self.output_dir / "best_component_models_hard_config.csv",
        )
        write_csv(
            self.component_forecasts_,
            self.output_dir / "component_predictions_hard_config.csv",
        )
        write_csv(
            self.final_forecasts_,
            self.output_dir / "final_reconstructed_forecasts_hard_config.csv",
        )
        write_csv(
            self.final_metrics_,
            self.output_dir / "final_reconstructed_metrics_hard_config.csv",
        )
        write_csv(
            ardl_orders,
            self.output_dir / "ardl_top3_full_product_train_only.csv",
        )
        write_csv(
            self.input_audit_,
            self.output_dir / "input_audit_hard_config.csv",
        )
        if self.stationarity_screen_ is not None:
            write_csv(
                self.stationarity_screen_,
                self.output_dir / "stationarity_screen_train_only_hard_config.csv",
            )
        for name, frame in diagnostics.items():
            write_csv(frame, self.output_dir / f"{name}_train_only.csv")
        write_csv(timing, self.output_dir / "runtime_breakdown.csv")

        return {
            "search_results": selection_df,
            "component_search_results": candidates_df,
            "component_signature_rankings": component_rankings_df,
            "reconstructed_seed_scores": reconstructed_seed_scores,
            "imf_descriptors": descriptors,
            "hard_ffnn_policy_config": policy_config,
            "hard_ffnn_policy_assignments": assignments,
            "hard_ffnn_fit_errors": fit_errors,
            "ardl_orders": ardl_orders,
            "best_component_models": self.best_component_models_,
            "component_forecasts": self.component_forecasts_,
            "training_components": train_components,
            "final_forecasts": self.final_forecasts_,
            "final_metrics": self.final_metrics_,
            "input_audit": self.input_audit_,
            "stationarity_screen": self.stationarity_screen_,
            "lag_screen_fold_metrics": pd.DataFrame(),
            "lag_screen_stability": pd.DataFrame(),
            "timing": timing,
            **diagnostics,
        }

    def run_vmd_shared_mean_seed(
        self,
        csv_path: str | Path,
        *,
        reporting_seeds: tuple[int, ...],
        model_name: str,
        selection_mode: str = "reconstructed_mean_seed",
        lag_spec_scope: str = "all",
        screening_mode: str = "none",
        artifact_stem: str = "shared_mean_seed",
    ) -> dict[str, pd.DataFrame]:
        """Run a VMD variant with one FFNN/lag structure shared across seeds.

        This entry point is used by the block-proposed paper profile for fixed,
        target-selected, all-block, component-wise block-subset, and sequential
        benchmark variants.  Candidate signatures never include the seed.
        """
        seeds = tuple(dict.fromkeys(int(seed) for seed in reporting_seeds))
        selection_mode = selection_mode.replace("-", "_")
        lag_spec_scope = lag_spec_scope.replace("-", "_")
        screening_mode = screening_mode.replace("-", "_")
        if not seeds:
            raise ValueError("Shared mean-seed selection requires reporting seeds.")
        if selection_mode not in {
            "component_mean_seed",
            "reconstructed_mean_seed",
        }:
            raise ValueError(
                "selection_mode must be 'component_mean_seed' or "
                "'reconstructed_mean_seed'."
            )
        if lag_spec_scope not in {"all", "all_blocks_only"}:
            raise ValueError("lag_spec_scope must be 'all' or 'all_blocks_only'.")
        if screening_mode not in {"none", "rolling_mean_seed"}:
            raise ValueError(
                "screening_mode must be 'none' or 'rolling_mean_seed'."
            )
        if self.config.ffnn.search_strategy.replace("-", "_") != "full_grid":
            raise ValueError("Shared mean-seed variants require FFNN full_grid search.")
        if tuple(self.config.ffnn.seed_grid) != seeds:
            raise ValueError("FFNN seed_grid must exactly match reporting_seeds.")

        total_started = perf_counter()
        self._lag_screen_fold_rows = []
        self._lag_screen_stability_rows = []
        self._log(
            f"Starting shared mean-seed VMD variant model={model_name}; "
            f"selection={selection_mode}; screening={screening_mode}; seeds={seeds}."
        )
        base_data = self.load_data(csv_path)
        base_train, base_val, base_test = self.split_raw_data(base_data)
        data = self._apply_stationarity_transform(base_data, base_train)
        train = data.loc[data.index.intersection(base_train.index)].copy()
        val = data.loc[data.index.intersection(base_val.index)].copy()
        test = data.loc[data.index.intersection(base_test.index)].copy()

        audit: list[dict[str, Any]] = []
        for split, frame in (("train", train), ("validation", val), ("test", test)):
            audit.extend(
                self._audit_rows(
                    artifact_stem,
                    "chronological_split",
                    split,
                    frame,
                )
            )

        vmd_cache_started = perf_counter()
        train_components = self.decompose(train)
        val_cache = self._make_origin_cache(train, val, f"{artifact_stem}-validation")
        val_actual_components = self._make_actual_component_cache(
            train,
            val,
            f"{artifact_stem}-validation",
        )
        vmd_cache_seconds = perf_counter() - vmd_cache_started

        selection_started = perf_counter()
        self._select_specs(train_components)
        if lag_spec_scope == "all_blocks_only":
            for component, specs in self.selected_specs_.items():
                all_block_specs = [
                    spec
                    for spec in specs
                    if all(tuple(spec.exog_lags.get(feature, ())) for feature in self.features)
                ]
                if len(all_block_specs) != 1:
                    raise ValueError(
                        "AIC all-block ablation requires exactly one all-block "
                        f"LagSpec for component={component}; got {len(all_block_specs)}."
                    )
                self.selected_specs_[component] = all_block_specs
        ardl_selection_seconds = perf_counter() - selection_started

        rolling_contexts = (
            self._prepare_vmd_lag_screen_contexts(train)
            if screening_mode == "rolling_mean_seed"
            else []
        )
        if screening_mode == "rolling_mean_seed" and not rolling_contexts:
            raise ValueError("No train-only rolling contexts are available for shared screening.")

        ffnn_search_started = perf_counter()
        candidate_frames: list[pd.DataFrame] = []
        component_ranking_frames: list[pd.DataFrame] = []
        retained_predictions: dict[int, pd.DataFrame] = {}
        screen_detail_frames: list[pd.DataFrame] = []
        screen_summary_frames: list[pd.DataFrame] = []
        candidate_id = 0
        for component, train_component in train_components.items():
            specs = self.selected_specs_[component]
            if screening_mode == "rolling_mean_seed" and len(specs) > 1:
                specs, screen_detail, screen_summary = (
                    self._fast_screen_specs_rolling_mean_seed(
                        component=component,
                        specs=specs,
                        seeds=seeds,
                        evaluate=lambda spec, learning_rate, seed, component=component: self._screen_vmd_lag_spec(
                            component,
                            spec,
                            learning_rate,
                            rolling_contexts,
                            seed=seed,
                        ),
                    )
                )
                screen_detail_frames.append(screen_detail)
                screen_summary_frames.append(screen_summary)

            rows, predictions, _, candidate_id = self._ffnn_search_candidates(
                component=component,
                train_data=train_component,
                specs=specs,
                prediction_fn=lambda model, spec, row_id, component=component: self._predict_component_from_cache(
                    model,
                    component,
                    val_cache,
                    val,
                    spec,
                    "validation",
                    row_id,
                    val_actual_components,
                ),
                scoring_fn=lambda prediction: self._score_component_prediction(
                    prediction,
                    val,
                ),
                uses_vmd=True,
                candidate_id_start=candidate_id,
            )
            component_candidates = pd.DataFrame(rows)
            ranked_candidates, component_ranking = (
                self._rank_mean_seed_component_signatures(
                    component_candidates,
                    seeds,
                )
            )
            candidate_frames.append(ranked_candidates)
            component_ranking_frames.append(component_ranking)
            retained_candidate_ids = set(
                ranked_candidates.loc[
                    ranked_candidates["shortlisted_for_reconstructed_search"],
                    "candidate_id",
                ].astype(int)
            )
            retained_predictions.update(
                {
                    candidate: predictions[candidate]
                    for candidate in retained_candidate_ids
                }
            )

        candidates_df = pd.concat(candidate_frames, ignore_index=True, sort=False)
        component_rankings_df = pd.concat(
            component_ranking_frames,
            ignore_index=True,
            sort=False,
        )
        if selection_mode == "component_mean_seed":
            component_rankings_df["shortlisted_for_reconstructed_search"] = (
                component_rankings_df["component_mean_seed_rank"].eq(1)
            )
            selected_signature_keys = set(
                zip(
                    component_rankings_df.loc[
                        component_rankings_df["shortlisted_for_reconstructed_search"],
                        "component",
                    ].astype(str),
                    component_rankings_df.loc[
                        component_rankings_df["shortlisted_for_reconstructed_search"],
                        "component_signature_id",
                    ].astype(str),
                    strict=True,
                )
            )
            candidates_df["shortlisted_for_reconstructed_search"] = [
                (str(component), str(signature_id)) in selected_signature_keys
                for component, signature_id in zip(
                    candidates_df["component"],
                    candidates_df["component_signature_id"],
                    strict=True,
                )
            ]

        selection_df, reconstructed_seed_scores = (
            self._score_mean_seed_reconstructed_combinations(
                candidates_df,
                component_rankings_df,
                retained_predictions,
                val,
                seeds,
                tuple(train_components),
            )
        )
        selection_protocol = (
            "component_mean_validation_across_reporting_seeds_then_reconstruct"
            if selection_mode == "component_mean_seed"
            else "mean_reconstructed_level_validation_across_reporting_seeds"
        )
        selection_df["selection_protocol"] = selection_protocol
        selection_df["shared_selection_mode"] = selection_mode
        selection_df["shared_screening_mode"] = screening_mode
        reconstructed_seed_scores["shared_selection_mode"] = selection_mode
        candidates_df["shared_selection_mode"] = selection_mode
        candidates_df["shared_screening_mode"] = screening_mode
        component_rankings_df["shared_selection_mode"] = selection_mode

        winning_combo = selection_df.iloc[0]
        selected_signature_by_component = {
            component: str(winning_combo[f"{component}_signature_id"])
            for component in train_components
        }
        selected_mask = pd.Series(False, index=candidates_df.index)
        for component, signature_id in selected_signature_by_component.items():
            selected_mask |= candidates_df["component"].eq(component) & candidates_df[
                "component_signature_id"
            ].eq(signature_id)
        best_candidates = candidates_df[selected_mask].copy()
        if len(best_candidates) != len(train_components) * len(seeds):
            raise ValueError(
                "Shared VMD winner does not cover every component and reporting seed."
            )
        candidates_df["selected_in_reconstructed_winner"] = selected_mask
        best_candidates["selected_in_reconstructed_winner"] = True
        for component, signature_id in selected_signature_by_component.items():
            component_rankings_df.loc[
                component_rankings_df["component"].eq(component)
                & component_rankings_df["component_signature_id"].eq(signature_id),
                "selected_in_reconstructed_winner",
            ] = True
        best_candidates["combo_id"] = int(winning_combo["combo_id"])
        best_candidates["final_selection_scale"] = "level"
        best_candidates["combo_mean_seed_val_level_RMSE"] = float(
            winning_combo["mean_val_level_RMSE"]
        )
        best_candidates["combo_std_seed_val_level_RMSE"] = float(
            winning_combo["std_val_level_RMSE"]
        )
        best_candidates["combo_mean_seed_val_level_MAE"] = float(
            winning_combo["mean_val_level_MAE"]
        )
        best_candidates["combo_mean_seed_val_level_MAPE"] = float(
            winning_combo["mean_val_level_MAPE"]
        )
        best_candidates["combo_total_input_dimension"] = int(
            winning_combo["total_input_dimension"]
        )
        best_candidates["selection_seed_count"] = len(seeds)
        best_candidates["selection_protocol"] = selection_protocol
        ffnn_search_seconds = perf_counter() - ffnn_search_started

        refit_started = perf_counter()
        history_before_test = pd.concat([train, val], axis=0)
        history_components = self.decompose(history_before_test)
        test_cache = self._make_origin_cache(
            history_before_test,
            test,
            f"{artifact_stem}-test",
        )
        test_actual_components = self._make_actual_component_cache(
            history_before_test,
            test,
            f"{artifact_stem}-test",
        )
        selected_component_predictions: list[pd.DataFrame] = []
        for row_index, best in best_candidates.iterrows():
            component = str(best["component"])
            seed = int(best["seed"])
            spec = LagSpec(
                tuple(best["target_lags"]),
                {
                    feature: tuple(lags)
                    for feature, lags in best["exog_lags"].items()
                },
            )
            validation_prediction = retained_predictions[
                int(best["candidate_id"])
            ].copy()
            validation_prediction["seed"] = seed
            selected_component_predictions.append(validation_prediction)

            test_model, n_refit = self._fit_model(
                history_components[component],
                spec,
                self._hidden_layer_sizes_from_row(best),
                float(best["alpha"]),
                seed,
                str(best["activation"]),
                learning_rate_init=float(best["learning_rate_init"]),
            )
            test_prediction = self._predict_component_from_cache(
                test_model,
                component,
                test_cache,
                test,
                spec,
                "test",
                actual_components=test_actual_components,
            )
            test_prediction["seed"] = seed
            selected_component_predictions.append(test_prediction)
            best_candidates.loc[row_index, "n_refit_train_val"] = int(n_refit)

        self.component_forecasts_ = pd.concat(
            selected_component_predictions,
            ignore_index=True,
            sort=False,
        )
        self.final_forecasts_ = self._reconstruct(
            self.component_forecasts_,
            pd.concat([val, test], axis=0),
        ).reset_index(drop=True)
        self.final_forecasts_["forecast_protocol"] = self.config.forecast_protocol

        metric_frames: list[pd.DataFrame] = []
        for seed in seeds:
            index = self.final_forecasts_.index[
                self.final_forecasts_["seed"].eq(seed)
            ]
            seed_forecasts = self.final_forecasts_.loc[index].copy()
            seed_best = best_candidates[best_candidates["seed"].eq(seed)].copy()
            metrics = self._annotate_final_metrics(
                self._metrics_rows(
                    seed_forecasts,
                    "actual_raw_transformed",
                    "predicted_reconstructed",
                    "reconstructed_transformed",
                ),
                csv_path,
                model_name,
                True,
                seed_best,
            )
            metrics["seed"] = seed
            metric_frames.append(metrics)
            for column in ("actual_level", "predicted_level"):
                if column in seed_forecasts:
                    self.final_forecasts_.loc[index, column] = seed_forecasts[
                        column
                    ].to_numpy()

        self.final_metrics_ = pd.concat(metric_frames, ignore_index=True, sort=False)
        self.best_component_models_ = best_candidates.sort_values(
            ["seed", "component"]
        ).reset_index(drop=True)
        self.input_audit_ = pd.DataFrame(audit)
        self.lag_screen_fold_metrics_ = pd.DataFrame(self._lag_screen_fold_rows)
        self.lag_screen_stability_ = pd.DataFrame(self._lag_screen_stability_rows)
        refit_forecast_seconds = perf_counter() - refit_started

        ardl_orders = pd.concat(
            [
                table.assign(component=component)
                for component, table in self.order_tables_.items()
            ],
            ignore_index=True,
        )
        diagnostics = self._diagnostics(
            train,
            train_components,
            f"_{artifact_stem}",
        )
        timing = pd.DataFrame(
            [
                {
                    "ardl_selection_seconds": ardl_selection_seconds,
                    "vmd_cache_seconds": vmd_cache_seconds,
                    "ffnn_search_seconds": ffnn_search_seconds,
                    "refit_forecast_seconds": refit_forecast_seconds,
                    "total_runtime_seconds": perf_counter() - total_started,
                }
            ]
        )
        screen_detail_df = (
            pd.concat(screen_detail_frames, ignore_index=True, sort=False)
            if screen_detail_frames
            else pd.DataFrame()
        )
        screen_summary_df = (
            pd.concat(screen_summary_frames, ignore_index=True, sort=False)
            if screen_summary_frames
            else pd.DataFrame()
        )

        write_csv(candidates_df, self.output_dir / "ffnn_shared_component_search.csv")
        write_csv(
            component_rankings_df,
            self.output_dir / "ffnn_shared_component_mean_seed_ranking.csv",
        )
        write_csv(
            reconstructed_seed_scores,
            self.output_dir / "ffnn_shared_reconstructed_seed_scores.csv",
        )
        write_csv(selection_df, self.output_dir / "ffnn_shared_mean_seed_selection.csv")
        if not screen_detail_df.empty:
            write_csv(screen_detail_df, self.output_dir / "ffnn_shared_screen_metrics.csv")
        if not screen_summary_df.empty:
            write_csv(screen_summary_df, self.output_dir / "ffnn_shared_screen_summary.csv")
        write_csv(
            self.best_component_models_,
            self.output_dir / "best_component_models_shared_mean_seed.csv",
        )
        write_csv(
            self.component_forecasts_,
            self.output_dir / "component_predictions_shared_mean_seed.csv",
        )
        write_csv(
            self.final_forecasts_,
            self.output_dir / "final_reconstructed_forecasts_shared_mean_seed.csv",
        )
        write_csv(
            self.final_metrics_,
            self.output_dir / "final_reconstructed_metrics_shared_mean_seed.csv",
        )
        write_csv(ardl_orders, self.output_dir / "ardl_orders_train_only.csv")
        write_csv(self.input_audit_, self.output_dir / "input_audit_shared_mean_seed.csv")
        if self.stationarity_screen_ is not None:
            write_csv(
                self.stationarity_screen_,
                self.output_dir / "stationarity_screen_train_only_shared_mean_seed.csv",
            )
        for name, frame in diagnostics.items():
            write_csv(frame, self.output_dir / f"{name}_train_only.csv")
        write_csv(timing, self.output_dir / "runtime_breakdown.csv")

        return {
            "search_results": selection_df,
            "component_search_results": candidates_df,
            "component_signature_rankings": component_rankings_df,
            "reconstructed_seed_scores": reconstructed_seed_scores,
            "shared_screen_metrics": screen_detail_df,
            "shared_screen_summary": screen_summary_df,
            "ardl_orders": ardl_orders,
            "best_component_models": self.best_component_models_,
            "component_forecasts": self.component_forecasts_,
            "training_components": train_components,
            "final_forecasts": self.final_forecasts_,
            "final_metrics": self.final_metrics_,
            "input_audit": self.input_audit_,
            "stationarity_screen": self.stationarity_screen_,
            "lag_screen_fold_metrics": self.lag_screen_fold_metrics_,
            "lag_screen_stability": self.lag_screen_stability_,
            "timing": timing,
            **diagnostics,
        }

    def run_no_vmd_shared_mean_seed(
        self,
        csv_path: str | Path,
        *,
        reporting_seeds: tuple[int, ...],
        model_name: str,
        screening_mode: str = "none",
        artifact_stem: str = "no_vmd_shared_mean_seed",
    ) -> dict[str, pd.DataFrame]:
        """Run one no-VMD model with a validation winner shared across seeds."""
        seeds = tuple(dict.fromkeys(int(seed) for seed in reporting_seeds))
        screening_mode = screening_mode.replace("-", "_")
        if not seeds:
            raise ValueError("Shared mean-seed selection requires reporting seeds.")
        if screening_mode not in {"none", "rolling_mean_seed"}:
            raise ValueError(
                "screening_mode must be 'none' or 'rolling_mean_seed'."
            )
        if self.config.ffnn.search_strategy.replace("-", "_") != "full_grid":
            raise ValueError("Shared mean-seed variants require FFNN full_grid search.")
        if tuple(self.config.ffnn.seed_grid) != seeds:
            raise ValueError("FFNN seed_grid must exactly match reporting_seeds.")

        total_started = perf_counter()
        self._lag_screen_fold_rows = []
        self._lag_screen_stability_rows = []
        base_data = self.load_data(csv_path)
        base_train, base_val, base_test = self.split_raw_data(base_data)
        data = self._apply_stationarity_transform(base_data, base_train)
        train = data.loc[data.index.intersection(base_train.index)].copy()
        val = data.loc[data.index.intersection(base_val.index)].copy()
        test = data.loc[data.index.intersection(base_test.index)].copy()
        component = "raw_no_vmd"

        audit: list[dict[str, Any]] = []
        for split, frame in (("train", train), ("validation", val), ("test", test)):
            audit.extend(
                self._audit_rows(
                    artifact_stem,
                    "chronological_split",
                    split,
                    frame,
                )
            )

        selection_started = perf_counter()
        self._select_specs({component: train})
        ardl_selection_seconds = perf_counter() - selection_started
        specs = self.selected_specs_[component]
        screen_detail = pd.DataFrame()
        screen_summary = pd.DataFrame()
        if screening_mode == "rolling_mean_seed" and len(specs) > 1:
            folds = self._expanding_lag_screen_splits(train)
            if not folds:
                raise ValueError(
                    "No train-only rolling folds are available for shared screening."
                )
            specs, screen_detail, screen_summary = (
                self._fast_screen_specs_rolling_mean_seed(
                    component=component,
                    specs=specs,
                    seeds=seeds,
                    evaluate=lambda spec, learning_rate, seed: self._screen_no_vmd_lag_spec(
                        spec,
                        learning_rate,
                        folds,
                        component,
                        seed=seed,
                    ),
                )
            )

        ffnn_search_started = perf_counter()
        observed_val = pd.concat([train, val], axis=0)
        rows, validation_predictions, _, _ = self._ffnn_search_candidates(
            component=component,
            train_data=train,
            specs=specs,
            prediction_fn=lambda model, spec, row_id: self._batch_predict_observed(
                model,
                observed_val,
                val.index,
                spec,
                "validation",
                component,
            ),
            scoring_fn=lambda prediction: self._score_raw_prediction_on_both_scales(
                prediction,
                "actual",
                "predicted",
            ),
            uses_vmd=False,
        )
        candidates = pd.DataFrame(rows)
        if candidates.empty:
            raise ValueError("No no-VMD shared candidate could be trained.")
        signature_columns = [
            "lag_spec",
            "hidden_layer_sizes",
            "alpha",
            "activation",
            "learning_rate_init",
        ]
        ranking = (
            candidates.groupby(signature_columns, as_index=False, dropna=False)
            .agg(
                mean_val_level_RMSE=("Val Level RMSE", "mean"),
                std_val_level_RMSE=("Val Level RMSE", "std"),
                mean_val_level_MAE=("Val Level MAE", "mean"),
                mean_val_level_MAPE=("Val Level MAPE", "mean"),
                seed_count=("seed", "nunique"),
            )
        )
        ranking = ranking[ranking["seed_count"].eq(len(seeds))].copy()
        if ranking.empty:
            raise ValueError("No complete no-VMD signature covers every reporting seed.")
        ranking["std_val_level_RMSE"] = ranking["std_val_level_RMSE"].fillna(0.0)
        ranking = ranking.sort_values(
            [
                "mean_val_level_RMSE",
                "mean_val_level_MAE",
                "mean_val_level_MAPE",
                "lag_spec",
                "hidden_layer_sizes",
                "alpha",
                "learning_rate_init",
            ]
        ).reset_index(drop=True)
        ranking.insert(0, "mean_seed_rank", np.arange(1, len(ranking) + 1))
        ranking["selected"] = ranking["mean_seed_rank"].eq(1)
        ranking["selection_protocol"] = (
            "mean_level_validation_across_reporting_seeds"
        )
        winning_signature = ranking.iloc[0]

        def no_vmd_signature_key(row: pd.Series) -> tuple[Any, ...]:
            return (
                str(row["lag_spec"]),
                self._hidden_layer_sizes_from_row(row),
                float(row["alpha"]),
                str(row["activation"]),
                float(row["learning_rate_init"]),
            )

        signature_key = no_vmd_signature_key(winning_signature)
        selected_mask = pd.Series(
            [
                no_vmd_signature_key(row) == signature_key
                for _, row in candidates.iterrows()
            ],
            index=candidates.index,
        )
        best_candidates = candidates[selected_mask].copy()
        if best_candidates["seed"].nunique() != len(seeds):
            raise ValueError("Shared no-VMD winner does not cover every reporting seed.")
        candidates["selected_shared_mean_seed"] = selected_mask
        best_candidates["selected_shared_mean_seed"] = True
        best_candidates["selection_seed_count"] = len(seeds)
        best_candidates["selection_protocol"] = (
            "mean_level_validation_across_reporting_seeds"
        )
        best_candidates["mean_seed_val_level_RMSE"] = float(
            winning_signature["mean_val_level_RMSE"]
        )
        ffnn_search_seconds = perf_counter() - ffnn_search_started

        refit_started = perf_counter()
        selected_forecasts: list[pd.DataFrame] = []
        history = pd.concat([train, val], axis=0)
        observed_test = pd.concat([train, val, test], axis=0)
        for row_index, best in best_candidates.iterrows():
            seed = int(best["seed"])
            spec = LagSpec(
                tuple(best["target_lags"]),
                {
                    feature: tuple(lags)
                    for feature, lags in best["exog_lags"].items()
                },
            )
            validation_prediction = validation_predictions[
                int(best["candidate_id"])
            ].rename(
                columns={
                    "actual": "actual_raw_transformed",
                    "predicted": "predicted_raw_transformed",
                }
            )
            validation_prediction["seed"] = seed
            selected_forecasts.append(validation_prediction)

            test_model, n_refit = self._fit_model(
                history,
                spec,
                self._hidden_layer_sizes_from_row(best),
                float(best["alpha"]),
                seed,
                str(best["activation"]),
                learning_rate_init=float(best["learning_rate_init"]),
            )
            test_prediction = self._batch_predict_observed(
                test_model,
                observed_test,
                test.index,
                spec,
                "test",
                component,
            ).rename(
                columns={
                    "actual": "actual_raw_transformed",
                    "predicted": "predicted_raw_transformed",
                }
            )
            test_prediction["seed"] = seed
            selected_forecasts.append(test_prediction)
            best_candidates.loc[row_index, "n_refit_train_val"] = int(n_refit)

        self.final_forecasts_ = pd.concat(
            selected_forecasts,
            ignore_index=True,
            sort=False,
        ).sort_values(["seed", "split", "date"]).reset_index(drop=True)
        self.final_forecasts_["forecast_protocol"] = self.config.forecast_protocol
        metric_frames: list[pd.DataFrame] = []
        for seed in seeds:
            index = self.final_forecasts_.index[
                self.final_forecasts_["seed"].eq(seed)
            ]
            seed_forecasts = self.final_forecasts_.loc[index].copy()
            seed_best = best_candidates[best_candidates["seed"].eq(seed)].copy()
            metrics = self._annotate_final_metrics(
                self._metrics_rows(
                    seed_forecasts,
                    "actual_raw_transformed",
                    "predicted_raw_transformed",
                    "raw_transformed",
                ),
                csv_path,
                model_name,
                False,
                seed_best,
            )
            metrics["seed"] = seed
            metric_frames.append(metrics)
            for column in ("actual_level", "predicted_level"):
                if column in seed_forecasts:
                    self.final_forecasts_.loc[index, column] = seed_forecasts[
                        column
                    ].to_numpy()

        self.final_metrics_ = pd.concat(metric_frames, ignore_index=True, sort=False)
        self.best_component_models_ = best_candidates.sort_values("seed").reset_index(
            drop=True
        )
        self.input_audit_ = pd.DataFrame(audit)
        self.lag_screen_fold_metrics_ = pd.DataFrame(self._lag_screen_fold_rows)
        self.lag_screen_stability_ = pd.DataFrame(self._lag_screen_stability_rows)
        refit_forecast_seconds = perf_counter() - refit_started
        ardl_orders = pd.concat(
            [
                table.assign(component=name)
                for name, table in self.order_tables_.items()
            ],
            ignore_index=True,
        )
        diagnostics = self._diagnostics(
            train,
            {component: train},
            f"_{artifact_stem}",
        )
        timing = pd.DataFrame(
            [
                {
                    "ardl_selection_seconds": ardl_selection_seconds,
                    "vmd_cache_seconds": 0.0,
                    "ffnn_search_seconds": ffnn_search_seconds,
                    "refit_forecast_seconds": refit_forecast_seconds,
                    "total_runtime_seconds": perf_counter() - total_started,
                }
            ]
        )

        write_csv(candidates, self.output_dir / "ffnn_shared_validation_search.csv")
        write_csv(ranking, self.output_dir / "ffnn_shared_mean_seed_ranking.csv")
        if not screen_detail.empty:
            write_csv(screen_detail, self.output_dir / "ffnn_shared_screen_metrics.csv")
        if not screen_summary.empty:
            write_csv(screen_summary, self.output_dir / "ffnn_shared_screen_summary.csv")
        write_csv(
            self.best_component_models_,
            self.output_dir / "best_model_shared_mean_seed.csv",
        )
        write_csv(
            self.final_forecasts_,
            self.output_dir / "final_forecasts_shared_mean_seed.csv",
        )
        write_csv(
            self.final_metrics_,
            self.output_dir / "final_metrics_shared_mean_seed.csv",
        )
        write_csv(ardl_orders, self.output_dir / "ardl_orders_train_only.csv")
        write_csv(self.input_audit_, self.output_dir / "input_audit_shared_mean_seed.csv")
        if self.stationarity_screen_ is not None:
            write_csv(
                self.stationarity_screen_,
                self.output_dir / "stationarity_screen_train_only_shared_mean_seed.csv",
            )
        for name, frame in diagnostics.items():
            write_csv(frame, self.output_dir / f"{name}_train_only.csv")
        write_csv(timing, self.output_dir / "runtime_breakdown.csv")

        return {
            "search_results": candidates,
            "mean_seed_rankings": ranking,
            "ardl_orders": ardl_orders,
            "best_component_models": self.best_component_models_,
            "final_forecasts": self.final_forecasts_,
            "final_metrics": self.final_metrics_,
            "input_audit": self.input_audit_,
            "stationarity_screen": self.stationarity_screen_,
            "lag_screen_fold_metrics": self.lag_screen_fold_metrics_,
            "lag_screen_stability": self.lag_screen_stability_,
            "timing": timing,
            **diagnostics,
        }

    def run_comparison(self, csv_path: str | Path) -> dict[str, pd.DataFrame]:
        """Chạy cả no-VMD và cached VMD, rồi xuất bảng/biểu đồ so sánh chung."""
        self._log("Starting no-VMD vs cached VMD comparison run.")
        # Dùng hai instance riêng để state chọn lag/model/forecast không ghi đè lẫn nhau.
        no_vmd_config = replace(self.config, output_dir=self.output_dir / "no_vmd")
        vmd_config = replace(self.config, output_dir=self.output_dir / "cached_vmd")
        no_vmd_experiment = VMDARDLFFNNExperiment(no_vmd_config)
        no_vmd_result = no_vmd_experiment.run_without_vmd(csv_path)
        vmd_experiment = VMDARDLFFNNExperiment(vmd_config)
        vmd_result = vmd_experiment.run(csv_path)

        comparison_table = self._comparison_table(
            no_vmd_result["final_metrics"],
            vmd_result["final_metrics"],
        )
        comparison_forecasts = pd.concat(
            [
                self._comparison_forecast_frame(
                    no_vmd_result["final_forecasts"],
                    "no_vmd",
                    "predicted_raw_transformed",
                ),
                self._comparison_forecast_frame(
                    vmd_result["final_forecasts"],
                    "cached_vmd",
                    "predicted_reconstructed",
                ),
            ],
            ignore_index=True,
        ).sort_values(["split", "date", "pipeline"]).reset_index(drop=True)

        comparison_table.to_csv(self.output_dir / "vmd_vs_no_vmd_metrics_comparison.csv", index=False)
        comparison_forecasts.to_csv(self.output_dir / "vmd_vs_no_vmd_forecasts_comparison.csv", index=False)
        self._plot_comparison_predictions(
            comparison_forecasts,
            "validation",
            "actual_vs_predicted_val_vmd_vs_no_vmd.png",
        )
        self._plot_comparison_predictions(
            comparison_forecasts,
            "test",
            "actual_vs_predicted_test_vmd_vs_no_vmd.png",
        )

        return {
            "comparison_metrics": comparison_table,
            "comparison_forecasts": comparison_forecasts,
            "no_vmd": no_vmd_result,
            "cached_vmd": vmd_result,
        }

    def run_model_comparison(
        self,
        csv_path: str | Path,
        varnn_config: Any | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Run cached VMD-ARDL-FFNN and VARNN, then export comparison artifacts."""
        self._log("Starting VMD-ARDL-FFNN vs VARNN comparison run.")
        vmd_config = replace(self.config, output_dir=self.output_dir / "vmd_ardl_ffnn")
        vmd_experiment = VMDARDLFFNNExperiment(vmd_config)
        vmd_result = vmd_experiment.run(csv_path)

        from .models.varnn import VARNNConfig, run_varnn_pipeline

        varnn_result = run_varnn_pipeline(
            csv_path,
            replace(self.config, output_dir=self.output_dir / "varnn"),
            varnn_config if varnn_config is not None else VARNNConfig(),
        )
        vmd_model = "VMD_ARDL_FFNN"
        varnn_model = "VARNN"
        comparison_table = self._model_metrics_comparison_table(
            vmd_result["final_metrics"],
            varnn_result["metrics"],
            vmd_model,
            varnn_model,
        )
        comparison_forecasts = pd.concat(
            [
                self._comparison_forecast_frame(
                    vmd_result["final_forecasts"],
                    "vmd_ardl_ffnn",
                    "predicted_reconstructed",
                ),
                self._varnn_forecast_comparison_frame(varnn_result["forecasts"]),
            ],
            ignore_index=True,
        ).sort_values(["split", "date", "pipeline"]).reset_index(drop=True)

        comparison_table.to_csv(self.output_dir / "vmd_ardl_ffnn_vs_varnn_metrics_comparison.csv", index=False)
        comparison_forecasts.to_csv(self.output_dir / "vmd_ardl_ffnn_vs_varnn_forecasts_comparison.csv", index=False)
        self._plot_model_comparison_predictions(
            comparison_forecasts,
            "validation",
            "actual_vs_predicted_val_vmd_ardl_ffnn_vs_varnn.png",
        )
        self._plot_model_comparison_predictions(
            comparison_forecasts,
            "test",
            "actual_vs_predicted_test_vmd_ardl_ffnn_vs_varnn.png",
        )

        return {
            "comparison_metrics": comparison_table,
            "comparison_forecasts": comparison_forecasts,
            "vmd_ardl_ffnn": vmd_result,
            "varnn": varnn_result,
        }

    def _plot_benchmark_predictions(
        self,
        comparison_forecasts: pd.DataFrame,
        split: str,
        filename: str,
    ) -> None:
        """Plot actual and each benchmark model prediction from a normalized comparison frame."""
        try:
            os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
            import matplotlib.pyplot as plt
        except ImportError:
            self._log(f"Skipping plot {filename}: matplotlib is not installed.")
            return
        group = comparison_forecasts[comparison_forecasts["split"].eq(split)].copy()
        if group.empty:
            return
        group["date"] = pd.to_datetime(group["date"])
        actual = group[["date", "actual"]].drop_duplicates(subset=["date"]).sort_values("date")
        label_map = {
            "vmd_ardl_ffnn": "VMD-ARDL-FFNN (Proposed)",
            "vmd_ardl_select_order_ffnn": "VMD-ARDL (SelectOrder)-FFNN",
            "no_vmd_ardl_ffnn": "ARDL-FFNN (No VMD)",
            "varnn": "VARNN",
        }
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(actual["date"], actual["actual"], label="Actual", linewidth=2.2, color="black")
        for pipeline in ["vmd_ardl_ffnn", "vmd_ardl_select_order_ffnn", "no_vmd_ardl_ffnn", "varnn"]:
            if pipeline not in group["pipeline"].unique():
                continue
            pred = group[group["pipeline"].eq(pipeline)].sort_values("date")
            ax.plot(pred["date"], pred["predicted"], label=label_map.get(pipeline, pipeline), linewidth=2)
        scale = group["scale"].dropna().iloc[0] if "scale" in group.columns and not group["scale"].dropna().empty else "transformed"
        ax.set_title(f"Model Predictions Comparison ({split.capitalize()})")
        ax.set_xlabel("Time")
        ax.set_ylabel("Level" if scale == "level" else "Transformed")
        ax.legend()
        ax.grid(True, alpha=0.25)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(self.output_dir / filename, dpi=160)
        plt.close(fig)

    def run_benchmark(
        self,
        csv_path: str | Path,
        varnn_config: Any | None = None,
        *,
        seeds: tuple[int, ...] | None = None,
        include_varnn: bool = True,
        block_subset_search_strategy: str = "full_grid",
    ) -> dict[str, Any]:
        """Run the B0--B7 benchmark and A0--A5 paper study."""
        from .models.varnn import VARNNConfig
        from .paper_experiments import run_paper_experiments

        reporting_seeds = tuple(seeds or self.config.ffnn.seed_grid)
        self._log(
            "Starting paper benchmark/ablation study with seeds "
            f"{reporting_seeds}."
        )
        return run_paper_experiments(
            csv_path,
            self.config,
            varnn_config if varnn_config is not None else VARNNConfig(),
            seeds=reporting_seeds,
            include_varnn=include_varnn,
            block_subset_search_strategy=block_subset_search_strategy,
        )

    def run_block_proposed_benchmark(
        self,
        csv_path: str | Path,
        varnn_config: Any | None = None,
        *,
        seeds: tuple[int, ...] | None = None,
        include_varnn: bool = True,
    ) -> dict[str, Any]:
        """Run the shared-selection benchmark with block-subset A7 as Proposed."""
        from .block_proposed_experiments import run_block_proposed_experiments
        from .models.varnn import VARNNConfig

        reporting_seeds = tuple(seeds or self.config.ffnn.seed_grid)
        self._log(
            "Starting block-proposed benchmark/A0-A7 study with shared "
            f"selection across seeds={reporting_seeds}."
        )
        return run_block_proposed_experiments(
            csv_path,
            self.config,
            varnn_config if varnn_config is not None else VARNNConfig(),
            seeds=reporting_seeds,
            include_varnn=include_varnn,
        )

    def run_shared_mean_benchmark(
        self,
        csv_path: str | Path,
        *,
        seeds: tuple[int, ...] | None = None,
    ) -> dict[str, Any]:
        """Run the four-workflow benchmark with shared mean-seed selection."""
        from .shared_mean_benchmark_experiments import (
            run_shared_mean_benchmark_experiments,
        )

        reporting_seeds = tuple(seeds or self.config.ffnn.seed_grid)
        self._log(
            "Starting four-workflow shared-mean benchmark with seeds "
            f"{reporting_seeds}."
        )
        return run_shared_mean_benchmark_experiments(
            csv_path,
            self.config,
            seeds=reporting_seeds,
        )


def config_for_columns(
    target: str,
    features: tuple[str, ...],
    date_col: str = "TIME_PERIOD",
    output_dir: str | Path = "results",
) -> ExperimentConfig:
    """Tạo ExperimentConfig tối giản khi chỉ cần đổi target, feature và output."""
    # Chỉ thay cấu hình cột và output, các tham số VMD/ARDL/FFNN giữ mặc định.
    return replace(
        ExperimentConfig(),
        data=DataConfig(date_col=date_col, target=target, features=features),
        output_dir=Path(output_dir),
    )
