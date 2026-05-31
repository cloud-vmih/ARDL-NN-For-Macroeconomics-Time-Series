from __future__ import annotations

from dataclasses import replace
from itertools import product
import os
from pathlib import Path
from typing import Any
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from .config import DataConfig, ExperimentConfig
from .data import GenericDataLoader
from .decomposition import VariationalModeDecomposer
from .diagnostics import ForecastDiagnostics
from .features import LagSpec
from .lag_selection import ARDLOrderSelector
from .metrics import evaluate_forecast
from .models.ffnn import SklearnFFNNRegressor


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

    def _safe_adf_pvalue(self, series: pd.Series) -> float:
        """Tính p-value ADF cho một chuỗi đủ dài và không hằng.
        Trả về NaN khi dữ liệu không đạt điều kiện hoặc statsmodels báo lỗi.
        """
        series = series.dropna()
        # ADF không có ý nghĩa với chuỗi quá ngắn hoặc không có biến thiên.
        if len(series) < self.config.stationarity.min_obs or series.nunique() <= 1:
            return float("nan")
        try:
            return float(adfuller(series, autolag="AIC")[1])
        except Exception:
            return float("nan")

    def _safe_kpss_pvalue(self, series: pd.Series) -> float:
        """Tính p-value KPSS với cảnh báo được chặn trong lúc kiểm định.
        Trả về NaN cho chuỗi quá ngắn, chuỗi hằng hoặc lỗi số học/thống kê.
        """
        series = series.dropna()
        # KPSS cũng cần đủ quan sát và phương sai khác 0 để chạy ổn định.
        if len(series) < self.config.stationarity.min_obs or series.nunique() <= 1:
            return float("nan")
        try:
            with warnings.catch_warnings():
                # KPSS thường cảnh báo khi thống kê nằm ngoài bảng tới hạn; ở đây chỉ cần p-value.
                warnings.simplefilter("ignore")
                return float(kpss(series, regression="c", nlags="auto")[1])
        except Exception:
            return float("nan")

    def _is_stationary(self, adf_p: float, kpss_p: float) -> bool:
        """Kết luận dừng khi ADF bác bỏ unit root và KPSS không bác bỏ dừng."""
        alpha = float(self.config.stationarity.alpha)
        return pd.notna(adf_p) and pd.notna(kpss_p) and adf_p < alpha and kpss_p > alpha

    def _stationarity_label(self, adf_p: float, kpss_p: float) -> str:
        """Chuyển cặp p-value ADF/KPSS thành nhãn đọc được trong báo cáo."""
        alpha = float(self.config.stationarity.alpha)
        if self._is_stationary(adf_p, kpss_p):
            return "Stationary"
        if pd.notna(adf_p) and pd.notna(kpss_p) and adf_p >= alpha and kpss_p <= alpha:
            return "Non-stationary"
        return "Mixed/unclear"

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
        self.transform_info_ = {}
        if not self.config.stationarity.enabled:
            # Khi tắt kiểm định tính dừng, mọi biến được giữ nguyên ở dạng level.
            self.transform_info_ = {col: "level" for col in base_data.columns}
            self.stationarity_screen_ = pd.DataFrame(
                {
                    "Variable": list(base_data.columns),
                    "Transform used": ["level"] * len(base_data.columns),
                    "Decision basis": ["disabled"] * len(base_data.columns),
                }
            )
            return base_data.copy()

        self._log("Running train-only stationarity screen (ADF/KPSS).")
        # Biến đổi sẽ áp dụng lên toàn bộ chuỗi, nhưng quyết định chỉ dựa trên train.
        transformed = base_data.copy()
        rows: list[dict[str, Any]] = []
        for col in base_data.columns:
            # Kiểm định level trên train để tránh dùng thông tin validation/test.
            level_adf = self._safe_adf_pvalue(base_train[col])
            level_kpss = self._safe_kpss_pvalue(base_train[col])
            # Kiểm định diff1 trên train dùng làm bằng chứng phụ cho báo cáo I(2).
            train_diff = base_train[col].diff()
            diff_adf = self._safe_adf_pvalue(train_diff)
            diff_kpss = self._safe_kpss_pvalue(train_diff)
            # Nếu level chưa đạt tiêu chí dừng kép thì dùng sai phân bậc một.
            transform = "level" if self._is_stationary(level_adf, level_kpss) else "diff1"
            if transform == "diff1":
                # Sai phân toàn chuỗi sau khi đã chốt quyết định bằng train-only.
                transformed[col] = base_data[col].diff()
            # Ghi lại transform để metric và audit biết dữ liệu đang ở thang nào.
            self.transform_info_[col] = transform
            rows.append(
                {
                    "Variable": col,
                    "ADF level p": level_adf,
                    "KPSS level p": level_kpss,
                    "Level decision": self._stationarity_label(level_adf, level_kpss),
                    "ADF diff1 p": diff_adf,
                    "KPSS diff1 p": diff_kpss,
                    "Diff1 decision": self._stationarity_label(diff_adf, diff_kpss),
                    "Transform used": transform,
                    "Decision basis": "train_only",
                    "Likely I(2)?": "Yes" if not self._is_stationary(diff_adf, diff_kpss) else "No",
                }
            )
        # Bảng này giúp truy vết vì sao mỗi biến được giữ level hoặc chuyển diff1.
        self.stationarity_screen_ = pd.DataFrame(rows)
        # Drop NaN sinh ra bởi diff1 trước khi tạo lag/model.
        return transformed.dropna().copy()

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
        hidden_units: int,
        alpha: float,
        seed: int,
    ) -> tuple[SklearnFFNNRegressor, int]:
        """Fit FFNN cho một LagSpec và trả về model kèm số dòng train hợp lệ.
        Dữ liệu được chuyển sang ma trận supervised trước khi kiểm tra min_train.
        """
        # Chuyển chuỗi thời gian thành X/y theo LagSpec trước khi fit FFNN.
        supervised = self._build_supervised(data, spec)
        if len(supervised) < self.config.ffnn.min_train:
            raise ValueError(f"Only {len(supervised)} training rows after lagging.")
        # FFNN wrapper tự chuẩn hóa X và y để MLPRegressor học ổn định hơn.
        model = SklearnFFNNRegressor(
            hidden_layer_sizes=self.config.ffnn.architecture_for(hidden_units),
            alpha=alpha,
            seed=seed,
            activation=self.config.ffnn.activation,
            learning_rate_init=self.config.ffnn.learning_rate_init,
            max_iter=self.config.ffnn.max_iter,
        ).fit(supervised[self._feature_names(spec)].to_numpy(float), supervised[self.target].to_numpy(float))
        # Trả kèm số dòng train thật sự sau khi mất dữ liệu do lag.
        return model, len(supervised)

    def _batch_predict_observed(
        self,
        model: SklearnFFNNRegressor,
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

    def _select_specs(self, frames: dict[str, pd.DataFrame]) -> None:
        """Chạy ARDLOrderSelector cho từng frame và lưu các spec forecast-safe.
        Bảng điểm ARDL được giữ trong order_tables_, còn danh sách LagSpec đã
        lọc trùng được giữ trong selected_specs_.
        """
        selector = ARDLOrderSelector(self.config.ardl)
        # Reset kết quả chọn lag để mỗi lần run không dùng lẫn state cũ.
        self.order_tables_ = {}
        self.selected_specs_ = {}
        for name, frame in frames.items():
            self._log(f"Selecting ARDL lags on train only: {name}")
            # ARDL chỉ chạy trên frame train/component tương ứng.
            table, raw_specs = selector.select(frame, self.target, self.features)
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
            row: dict[str, Any] = {"split": split, "scale": transformed_scale}
            row.update(evaluate_forecast(group[actual_col], group[predicted_col]))
            rows.append(row)
            if self.config.data.log_transform or self.transform_info_.get(self.target) == "diff1":
                # Nếu pipeline đã log hoặc sai phân target, cần đổi về level để báo cáo dễ hiểu.
                actual_level, predicted_level = self._to_level(group, actual_col, predicted_col)
                forecasts.loc[group.index, "actual_level"] = actual_level
                forecasts.loc[group.index, "predicted_level"] = predicted_level
                level_row: dict[str, Any] = {"split": split, "scale": "level"}
                level_row.update(evaluate_forecast(actual_level, predicted_level))
                rows.append(level_row)
        return pd.DataFrame(rows)

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
        self._log("Starting fixed no-VMD pipeline.")
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
        # Với baseline no-VMD, toàn bộ chuỗi được xem như một component duy nhất.
        self._select_specs({component: train})
        audit = []
        for split, frame in [("train", train), ("validation", val), ("test", test)]:
            # Ghi audit split trước khi tạo lag để biết kích thước dữ liệu gốc của từng split.
            audit.extend(self._audit_rows("no_vmd", "chronological_split", split, frame))

        rows: list[dict[str, Any]] = []
        # Lưu forecast validation theo candidate_id để lấy lại candidate thắng.
        val_frames: dict[int, pd.DataFrame] = {}
        row_id = 0
        for spec in self.selected_specs_[component]:
            for hidden_units in self.config.ffnn.hidden_units_candidates:
                for alpha in self.config.ffnn.alpha_grid:
                    for seed in self.config.ffnn.seed_grid:
                        try:
                            # Fit chỉ trên train để validation phản ánh đúng out-of-sample.
                            model, n_train = self._fit_model(train, spec, int(hidden_units), float(alpha), int(seed))
                            # Validation được dự báo từ chuỗi quan sát train+validation nhưng chỉ chấm val.index.
                            val_pred = self._batch_predict_observed(
                                model, pd.concat([train, val]), val.index, spec, "validation", component
                            )
                        except ValueError:
                            # Bỏ candidate không đủ dữ liệu sau lag hoặc spec không hợp lệ.
                            continue
                        if len(val_pred) < self.config.ffnn.min_val:
                            # Không chấm candidate khi validation còn quá ít điểm dự báo.
                            continue
                        row = {
                            "candidate_id": row_id,
                            "component": component,
                            "lag_spec": spec.label,
                            "target_lags": spec.target_lags,
                            "exog_lags": spec.exog_lags,
                            "HR": int(hidden_units),
                            "hidden_layer_sizes": self.config.ffnn.architecture_for(int(hidden_units)),
                            "alpha": float(alpha),
                            "seed": int(seed),
                            "n_train": n_train,
                            "n_val": len(val_pred),
                            "forecast_safe_no_exog_lag0": True,
                            "uses_vmd": False,
                        }
                        # Gắn metric validation vào cùng dòng candidate để sort chọn model.
                        row.update({f"Val {k}": v for k, v in evaluate_forecast(val_pred["actual"], val_pred["predicted"]).items()})
                        rows.append(row)
                        val_frames[row_id] = val_pred
                        # Audit design validation ghi rõ feature lag đã dùng cho candidate này.
                        audit.extend(self._audit_rows("no_vmd", "validation_design", "validation", val_pred, spec, self._feature_names(spec)))
                        row_id += 1
        if not rows:
            raise ValueError("No no-VMD candidates could be trained.")
        # Chọn candidate tốt nhất theo RMSE, rồi MAE và MAPE làm tie-breaker.
        self.search_results_ = pd.DataFrame(rows).sort_values(["Val RMSE", "Val MAE", "Val MAPE"]).reset_index(drop=True)
        best = self.search_results_.iloc[0]
        # Khôi phục LagSpec từ dòng winner để refit và dự báo test.
        best_spec = LagSpec(tuple(best["target_lags"]), {k: tuple(v) for k, v in best["exog_lags"].items()})
        self._log("Refitting locked no-VMD winner on train+validation.")
        # Refit winner trên train+validation sau khi hyperparameter đã khóa bằng validation.
        best_model, n_refit = self._fit_model(pd.concat([train, val]), best_spec, int(best["HR"]), float(best["alpha"]), int(best["seed"]))
        # Giữ nguyên validation forecast của winner để final_forecasts gồm cả val và test.
        val_forecasts = val_frames[int(best["candidate_id"])].rename(columns={"actual": "actual_raw_transformed", "predicted": "predicted_raw_transformed"})
        # Test forecast dùng model đã refit và lịch sử train+validation+test quan sát.
        test_forecasts = self._batch_predict_observed(
            best_model, pd.concat([train, val, test]), test.index, best_spec, "test", component
        ).rename(columns={"actual": "actual_raw_transformed", "predicted": "predicted_raw_transformed"})
        final = pd.concat([val_forecasts, test_forecasts], ignore_index=True)
        # Metric có thể thêm cột level vào final khi cần đảo log/diff.
        self.final_metrics_ = self._metrics_rows(final, "actual_raw_transformed", "predicted_raw_transformed", "raw_transformed")
        self.final_forecasts_ = final.sort_values(["split", "date"]).reset_index(drop=True)
        best_model_df = pd.DataFrame([{**best.to_dict(), "n_refit_train_val": n_refit}])
        self.best_component_models_ = best_model_df
        audit.extend(self._audit_rows("no_vmd", "test_design", "test", test_forecasts, best_spec, self._feature_names(best_spec)))
        self.input_audit_ = pd.DataFrame(audit)

        diagnostics = self._diagnostics(train, {component: train}, "_no_vmd")
        # Gộp bảng chọn lag ARDL từ từng component để xuất một file duy nhất.
        ardl_orders = pd.concat([table.assign(component=name) for name, table in self.order_tables_.items()], ignore_index=True)
        # Xuất toàn bộ artifact của baseline no-VMD để đối chiếu lại sau chạy.
        self.search_results_.to_csv(self.output_dir / "ffnn_validation_search_fixed_no_vmd.csv", index=False)
        best_model_df.to_csv(self.output_dir / "best_model_fixed_no_vmd.csv", index=False)
        self.final_forecasts_.to_csv(self.output_dir / "final_forecasts_fixed_no_vmd.csv", index=False)
        self.final_metrics_.to_csv(self.output_dir / "final_metrics_fixed_no_vmd.csv", index=False)
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
        return {
            "search_results": self.search_results_,
            "ardl_orders": ardl_orders,
            "best_component_models": best_model_df,
            "final_forecasts": self.final_forecasts_,
            "final_metrics": self.final_metrics_,
            "input_audit": self.input_audit_,
            "stationarity_screen": self.stationarity_screen_,
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

    def _predict_component_from_cache(
        self,
        model: SklearnFFNNRegressor,
        component: str,
        cache: dict[pd.Timestamp, dict[str, pd.DataFrame]],
        actual_raw: pd.DataFrame,
        spec: LagSpec,
        split: str,
        candidate_id: int | None = None,
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
            if candidate_id is not None:
                row["candidate_id"] = candidate_id
            rows.append(row)
        return pd.DataFrame(rows)

    def _reconstruct(self, component_predictions: pd.DataFrame, actual_raw: pd.DataFrame) -> pd.DataFrame:
        """Cộng dự báo component theo date để tái tạo forecast của target.
        Actual được lấy từ actual_raw trên cùng thang transformed của mô hình.
        """
        # Tổng các predicted_component chính là dự báo reconstructed của target.
        out = (
            component_predictions.groupby(["split", "date"], as_index=False)
            .agg(predicted_reconstructed=("predicted_component", "sum"), component_count=("component", "nunique"))
            .sort_values(["split", "date"])
        )
        # Gắn actual target để tính metric reconstructed trên cùng timestamp.
        out["actual_raw_transformed"] = out["date"].map(actual_raw[self.target]).astype(float)
        return out

    def run(self, csv_path: str | Path) -> dict[str, pd.DataFrame]:
        """Chạy pipeline VMD đầy đủ với cache walk-forward chống leakage.
        Hàm chọn model tốt nhất cho từng component, tìm tổ hợp component tốt
        trên validation, refit trên train+validation, dự báo test và xuất kết quả.
        """
        self._log("Starting cached VMD pipeline.")
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
        # VMD ban đầu chỉ chạy trên train để chọn lag và fit candidate component.
        train_components = self.decompose(train)
        self._select_specs(train_components)
        # Cache validation theo origin để VMD không thấy điểm validation tương lai.
        val_cache = self._make_origin_cache(train, val, "validation")

        candidate_predictions: dict[str, list[pd.DataFrame]] = {}
        candidate_rows: list[dict[str, Any]] = []
        candidate_id = 0
        for component, train_component in train_components.items():
            candidate_predictions[component] = []
            for spec in self.selected_specs_[component]:
                for hidden_units in self.config.ffnn.hidden_units_candidates:
                    for alpha in self.config.ffnn.alpha_grid:
                        for seed in self.config.ffnn.seed_grid:
                            try:
                                # Fit model riêng cho từng component/spec/hyperparameter trên train component.
                                model, n_train = self._fit_model(train_component, spec, int(hidden_units), float(alpha), int(seed))
                                # Dự báo validation bằng cache VMD walk-forward của component đó.
                                pred = self._predict_component_from_cache(model, component, val_cache, val, spec, "validation", candidate_id)
                            except (ValueError, KeyError, IndexError):
                                # Bỏ candidate lỗi do thiếu lag, thiếu component hoặc history chưa đủ dài.
                                continue
                            row = {
                                "candidate_id": candidate_id,
                                "component": component,
                                "lag_spec": spec.label,
                                "target_lags": spec.target_lags,
                                "exog_lags": spec.exog_lags,
                                "HR": int(hidden_units),
                                "hidden_layer_sizes": self.config.ffnn.architecture_for(int(hidden_units)),
                                "alpha": float(alpha),
                                "seed": int(seed),
                                "n_train": n_train,
                                "n_val": len(pred),
                                "forecast_safe_no_exog_lag0": True,
                            }
                            candidate_rows.append(row)
                            candidate_predictions[component].append(pred)
                            # Audit từng candidate validation để biết spec nào sinh forecast nào.
                            audit.extend(self._audit_rows("cached_vmd", "validation_cache", "validation", pred, spec, self._feature_names(spec)))
                            candidate_id += 1
            if not candidate_predictions[component]:
                raise ValueError(f"No cached VMD validation candidates for {component}.")

        component_ranked: dict[str, pd.DataFrame] = {}
        for component, frames in candidate_predictions.items():
            scored = []
            for frame in frames:
                # Chấm candidate component bằng cách reconstruct riêng candidate đó trên validation.
                reconstructed = self._reconstruct(frame, val)
                metrics = evaluate_forecast(reconstructed["actual_raw_transformed"], reconstructed["predicted_reconstructed"])
                scored.append({"candidate_id": int(frame["candidate_id"].iloc[0]), **{f"Component Val {k}": v for k, v in metrics.items()}})
            component_ranked[component] = pd.DataFrame(scored).sort_values(["Component Val RMSE", "Component Val MAE", "Component Val MAPE"])

        # Giữ tối đa 5 candidate tốt nhất mỗi component để giảm số tổ hợp cần thử.
        top_by_component = {
            component: ranked.head(min(5, len(ranked)))["candidate_id"].astype(int).tolist()
            for component, ranked in component_ranked.items()
        }
        # Map nhanh candidate_id sang DataFrame dự báo validation của candidate đó.
        pred_by_id = {
            int(frame["candidate_id"].iloc[0]): frame
            for frames in candidate_predictions.values()
            for frame in frames
        }
        combo_rows: list[dict[str, Any]] = []
        component_order = list(top_by_component)
        for combo_id, ids in enumerate(product(*top_by_component.values())):
            # Ghép một candidate từ mỗi component để tạo dự báo reconstructed hoàn chỉnh.
            component_frames = [pred_by_id[int(candidate)] for candidate in ids]
            reconstructed = self._reconstruct(pd.concat(component_frames, ignore_index=True), val)
            metric = evaluate_forecast(reconstructed["actual_raw_transformed"], reconstructed["predicted_reconstructed"])
            row = {"combo_id": combo_id, "component_candidate_ids": tuple(int(x) for x in ids)}
            row.update({f"{component}_candidate_id": int(candidate) for component, candidate in zip(component_order, ids)})
            row.update({f"Val {k}": v for k, v in metric.items()})
            combo_rows.append(row)
        # Combo tốt nhất được chọn bằng metric reconstructed trên validation.
        combo_df = pd.DataFrame(combo_rows).sort_values(["Val RMSE", "Val MAE", "Val MAPE"]).reset_index(drop=True)

        candidates_df = pd.DataFrame(candidate_rows)
        self.search_results_ = combo_df
        # Lấy danh sách candidate_id thắng, mỗi id tương ứng một component model.
        best_ids = tuple(int(x) for x in combo_df.iloc[0]["component_candidate_ids"])
        best_candidates = candidates_df[candidates_df["candidate_id"].isin(best_ids)].copy()
        best_candidates["combo_id"] = int(combo_df.iloc[0]["combo_id"])

        self._log("Refitting cached VMD winners on train+validation.")
        # Trước khi dự báo test, refit VMD và model trên toàn bộ train+validation.
        history_before_test = pd.concat([train, val], axis=0)
        history_components = self.decompose(history_before_test)
        # Test cache cũng walk-forward: mỗi test date chỉ thấy lịch sử trước date.
        test_cache = self._make_origin_cache(history_before_test, test, "test")
        selected_component_predictions: list[pd.DataFrame] = []
        # Forecast validation của winner được giữ lại để báo cáo cùng test.
        val_selected = [pred_by_id[candidate_id] for candidate_id in best_ids]
        selected_component_predictions.extend(val_selected)
        for _, best in best_candidates.iterrows():
            component = str(best["component"])
            # Khôi phục LagSpec và hyperparameter của component winner.
            spec = LagSpec(tuple(best["target_lags"]), {k: tuple(v) for k, v in best["exog_lags"].items()})
            # Refit component winner trên component history train+validation.
            model, n_refit = self._fit_model(history_components[component], spec, int(best["HR"]), float(best["alpha"]), int(best["seed"]))
            # Dự báo test component bằng model refit và cache VMD test.
            test_pred = self._predict_component_from_cache(model, component, test_cache, test, spec, "test", int(best["candidate_id"]))
            selected_component_predictions.append(test_pred)
            best_candidates.loc[best_candidates["candidate_id"].eq(best["candidate_id"]), "n_refit_train_val"] = n_refit
            audit.extend(self._audit_rows("cached_vmd", "test_cache", "test", test_pred, spec, self._feature_names(spec)))

        # Gom forecast component validation/test rồi reconstruct thành forecast cuối.
        self.component_forecasts_ = pd.concat(selected_component_predictions, ignore_index=True)
        actual_all = pd.concat([val, test], axis=0)
        self.final_forecasts_ = self._reconstruct(self.component_forecasts_, actual_all).reset_index(drop=True)
        # Metric cuối được tính trên reconstructed transformed và level nếu có thể đảo.
        self.final_metrics_ = self._metrics_rows(
            self.final_forecasts_,
            "actual_raw_transformed",
            "predicted_reconstructed",
            "reconstructed_transformed",
        )
        self.best_component_models_ = best_candidates.sort_values("component").reset_index(drop=True)
        self.input_audit_ = pd.DataFrame(audit)
        diagnostics = self._diagnostics(train, train_components, "_cached_vmd")
        # Gộp mọi bảng ARDL order theo component để xuất audit chọn lag.
        ardl_orders = pd.concat([table.assign(component=name) for name, table in self.order_tables_.items()], ignore_index=True)

        # Ghi artifact chính của pipeline cached VMD ra output_dir.
        combo_df.to_csv(self.output_dir / "ffnn_validation_search_cached_vmd.csv", index=False)
        self.best_component_models_.to_csv(self.output_dir / "best_component_models_cached_vmd.csv", index=False)
        self.component_forecasts_.to_csv(self.output_dir / "component_predictions_cached_vmd.csv", index=False)
        self.final_forecasts_.to_csv(self.output_dir / "final_reconstructed_forecasts_cached_vmd.csv", index=False)
        self.final_metrics_.to_csv(self.output_dir / "final_reconstructed_metrics_cached_vmd.csv", index=False)
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
        return {
            "search_results": combo_df,
            "ardl_orders": ardl_orders,
            "best_component_models": self.best_component_models_,
            "component_forecasts": self.component_forecasts_,
            "final_forecasts": self.final_forecasts_,
            "final_metrics": self.final_metrics_,
            "input_audit": self.input_audit_,
            "stationarity_screen": self.stationarity_screen_,
            **diagnostics,
        }


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