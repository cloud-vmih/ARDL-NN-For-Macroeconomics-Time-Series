# Review Project `vmd_ardl_ffnn`

Tài liệu này review lại package `vmd_ardl_ffnn` theo code hiện tại: hướng đi experiment, cách chọn lag, cách build model và cách đọc output. Mục tiêu chính của project là xây một pipeline dự báo time series có kiểm soát leakage, so sánh mô hình ARDL + FFNN trực tiếp với phiên bản có phân rã VMD.

## 1. Hướng Đi Experiment

Project đang theo hướng forecasting experiment có kiểm định nghiêm ngặt theo thời gian:

- Dữ liệu được load từ CSV, parse cột thời gian, sort tăng dần và ép về frequency nếu cấu hình có `freq`.
- Chỉ giữ `target` và các biến ngoại sinh `features`.
- Split theo thứ tự thời gian thành `train / validation / test`, không shuffle.
- Mọi quyết định chọn transform, chọn lag và chọn model đều dựa trên train hoặc validation; test chỉ dùng sau cùng.
- Có hai pipeline chính:
  - `no-vmd`: dự báo trực tiếp chuỗi target bằng ARDL lag selection + FFNN.
  - `vmd`: phân rã target và features thành các component VMD, dự báo từng component bằng ARDL lag selection + FFNN, rồi cộng forecast component lại.
- Có pipeline `both` để chạy cả `no-vmd` và `vmd`, sau đó tạo bảng so sánh metrics.

Entrypoint CLI nằm ở `src/vmd_ardl_ffnn/cli.py`. Module `experiment.py` chỉ re-export `VMDARDLFFNNExperiment` từ `experiment_fixed.py`, nên logic experiment thực tế nằm trong `experiment_fixed.py`.

## 2. Tiền Xử Lý Và Stationarity

Dữ liệu được xử lý bởi `GenericDataLoader`:

- Kiểm tra đủ các cột bắt buộc: `date_col`, `target`, `features`.
- Convert `date_col` sang datetime.
- Sort và drop duplicate theo thời gian.
- Set datetime index.
- Nếu `freq` khác `None`, gọi `asfreq(freq)`.
- Ép numeric và drop NaN.
- Nếu `log_transform=True`, kiểm tra tất cả giá trị dương rồi lấy log.

Sau split, experiment chạy stationarity screen trên train-only:

- ADF và KPSS được tính cho từng chuỗi ở level.
- Nếu level đạt tiêu chí stationary: giữ `level`.
- Nếu không đạt: dùng sai phân bậc 1 `diff1`.
- Quyết định transform chỉ dựa vào train, nhưng transform được áp dụng nhất quán cho toàn bộ series.

Điểm quan trọng: `level` trong output nghĩa là level của base scale sau bước load. Nếu bật log-transform, model học trên `log(Y)` hoặc `diff(log(Y))`; khi báo metric level thì pipeline đảo diff/log để quay về đơn vị gốc.

## 3. Cách Chọn Lag

Lag được mô tả bằng `LagSpec`:

```text
target_lags: tuple[int, ...]
exog_lags: dict[str, tuple[int, ...]]
```

Ví dụ:

```text
target:[1, 2] | PCE:[4], US_Retail:[1], USD_VND:[6]
```

### 3.1 Target Lag

Target lag là các lag autoregressive của biến cần dự báo. Project hiện có ba strategy:

- `fixed`: dùng bộ lag cố định từ `fixed_target_lags`, mặc định legacy là `(1, 12)`.
- `ic_topk`: fit ARDL đơn biến theo từng target lag từ `1..max_target_lag`, chọn top theo AIC/BIC.
- `validation_screen`: sinh nhiều candidate target-lag sets, chạy FFNN nhanh trên validation, rồi chọn candidate tốt nhất theo validation metrics.

Default hiện tại là:

```text
target_lag_strategy = validation_screen
target_lag_preset = acf_pacf
```

### 3.2 Candidate Target Lag Bằng ACF/PACF

Preset `acf_pacf` thay thế cách chọn cảm tính kiểu `monthly/daily/yearly`.

Luồng hoạt động:

1. Tính ACF và PACF của target trên train-only.
2. Dùng ngưỡng xấp xỉ `1.96 / sqrt(n)` để đánh dấu lag significant.
3. Dùng PACF làm nguồn chính vì PACF phù hợp hơn để nhận diện autoregressive lag trực tiếp.
4. Dùng ACF để bổ sung lag có tương quan mạnh mà PACF có thể bỏ sót.
5. Luôn giới hạn lag trong `1..max_target_lag`.
6. Nếu `force_target_lag_1=True`, lag 1 luôn được đưa vào candidate.
7. Sinh các candidate compact, ví dụ:
   - `(1,)`
   - `(1, best_pacf_lag)`
   - `(1, top_2_pacf_lags)`
   - `(1, best_pacf_lag, strong_acf_lag)`
   - compact set từ top PACF + top ACF, bị giới hạn bởi `target_lag_acf_pacf_max_lags_per_set`.

Các cột audit được ghi vào bảng ARDL orders cho dòng `__target_lag_validation_screen__`:

- `target_lag_source`
- `acf_value`
- `pacf_value`
- `significance_threshold`
- `selected_for_target_lags`

Các preset cũ `monthly`, `daily`, `yearly` vẫn còn để tái lập experiment cũ nếu cần.

### 3.3 Exogenous Lag

Sau khi target lags đã được khóa, ARDL chọn lag cho từng biến ngoại sinh:

1. Với mỗi feature, thử từng `exog_lag` từ `1..max_exog_lag`.
2. Fit ARDL:

```text
target ~ target_lags + feature_lag
```

3. Chấm điểm bằng AIC hoặc BIC.
4. Lấy `top_n` lag tốt nhất cho từng feature.
5. Tạo các `LagSpec` đưa vào FFNN.

Lag ngoại sinh 0 bị loại vì khi forecast tương lai không biết giá trị feature cùng timestamp. Đây là một điểm chống leakage quan trọng.

### 3.4 Tạo Tổ Hợp LagSpec

Có hai cách tạo tổ hợp exogenous lags:

- `full_product`: lấy toàn bộ tích Descartes của top lags từng feature.
- `staged`: giảm số tổ hợp bằng cách ưu tiên:
  - baseline rank-1 của từng feature,
  - thay đổi từng feature một so với baseline,
  - thêm các combo có tổng ARDL score tốt nhất cho tới `max_lag_specs`.

Default là `staged` để tránh bùng nổ số candidate khi có nhiều biến ngoại sinh.

## 4. Model Được Build Như Thế Nào

Model dự báo cuối là FFNN dùng `sklearn.neural_network.MLPRegressor`, được wrap trong `SklearnFFNNRegressor`.

### 4.1 Feature Matrix

Với mỗi `LagSpec`, experiment tạo supervised dataset:

- Cột y là target tại thời điểm hiện tại.
- Các input gồm:
  - `target__lag_k` cho từng target lag.
  - `feature__lag_k` cho từng exogenous lag.
- Các dòng đầu bị thiếu do lag sẽ bị drop.

Ví dụ với `target_lags=(1, 2)` và `PCE_lag=4`, input tại thời điểm `t` gồm:

```text
Y(t-1), Y(t-2), PCE(t-4), ...
```

### 4.2 FFNN

FFNN wrapper làm các bước:

- Fit `StandardScaler` cho X.
- Fit `StandardScaler` cho y.
- Train `MLPRegressor`.
- Predict xong inverse-transform y về model scale.

Cấu hình chính:

- `hidden_layer_sizes`: một hidden layer, lấy từ `hidden_units_candidates`.
- `activation`: mặc định `relu`.
- `solver`: `adam`.
- `alpha`: regularization.
- `learning_rate_init`: mặc định `0.01`.
- `early_stopping=True`.
- `n_iter_no_change=30`.
- `random_state=seed`.

### 4.3 Search FFNN

Search có hai mode:

- `full_grid`: thử toàn bộ tổ hợp `LagSpec x hidden_units x alpha x seed`.
- `staged_halving`: chạy nhanh từng `LagSpec` bằng một cấu hình nhẹ, chọn top lag specs, rồi mới grid search hyperparameter đầy đủ.

Default là `staged_halving`:

```text
max_lag_specs * fast_config
+ top_k_lag_specs * len(hidden_units_candidates) * len(alpha_grid) * len(seed_grid)
```

Candidate được chọn theo thứ tự:

```text
Val RMSE -> Val MAE -> Val MAPE
```

## 5. Pipeline `no-vmd`

Pipeline `no-vmd` xem toàn bộ chuỗi đã transform là một component duy nhất: `raw_no_vmd`.

Flow:

1. Load data.
2. Split train/validation/test theo thời gian.
3. Chạy stationarity transform train-only.
4. Nếu `validation_screen`, chọn target lag bằng ACF/PACF candidates + validation FFNN nhanh.
5. Chọn exogenous lags bằng ARDL trên train.
6. Tạo forecast-safe `LagSpec`.
7. Train FFNN candidates trên train.
8. Predict validation theo batch bằng observed lagged inputs.
9. Chọn candidate tốt nhất trên validation.
10. Refit winner trên train + validation.
11. Predict test.
12. Lưu forecasts, metrics, lag orders, diagnostics và plots.

Vì `no-vmd` không phân rã, validation/test prediction có thể dùng batch design từ dữ liệu observed đã lag. Vẫn không có exog lag 0.

## 6. Pipeline `vmd`

Pipeline `vmd` phân rã từng cột thành các component:

```text
VMD1, VMD2, ..., VMDK, RES
```

`RES` là phần dư còn lại sau khi cộng các VMD modes.

Flow:

1. Load, split và transform giống `no-vmd`.
2. Chạy VMD trên train-only để tạo train components.
3. Với từng component, chọn target lag bằng validation screen nếu được bật.
4. Với từng component, chọn exogenous lags bằng ARDL trên train component.
5. Tạo validation VMD origin cache:
   - tại mỗi timestamp validation `t`, VMD chỉ phân rã history trước `t`;
   - actual tại `t` chưa được thêm vào history;
   - sau khi cache origin `t`, actual `t` mới được append để origin tiếp theo dùng.
6. Train FFNN candidates cho từng component.
7. Predict từng component trên validation cache.
8. Reconstruct forecast bằng cách cộng predicted components theo date.
9. Chọn top candidates từng component.
10. Thử tổ hợp component candidates và chọn tổ hợp reconstructed forecast tốt nhất trên validation.
11. Refit component winners trên train + validation.
12. Tạo test VMD origin cache theo cùng nguyên tắc.
13. Predict test components, cộng lại, tính final metrics.

Điểm quan trọng: VMD không được chạy một lần trên toàn bộ validation/test vì như vậy sẽ đưa thông tin tương lai vào decomposition. Project dùng origin cache để tránh leakage này.

## 7. Metrics Và Diagnostics

Metrics chính:

- `RMSE`
- `MAE`
- `MAPE`
- `Directional_Accuracy`

Directional Accuracy so sánh dấu thay đổi giữa actual và predicted, không phải so dấu giá trị tuyệt đối.

Diagnostics gồm:

- `series_diagnostics`: ADF/KPSS, Jarque-Bera, ARCH-LM, Ljung-Box cho từng chuỗi.
- `granger_causality`: kiểm định Granger từ từng feature tới target.
- `ardl_residual_diagnostics`: AIC/BIC/HQIC và kiểm định phần dư ARDL cho các lag specs đã chọn.

## 8. Output Chính

Các output quan trọng thường gặp:

- `ardl_selected_orders_train_only*.csv`: bảng chọn target lag, exogenous lag và lag specs.
- `ffnn_validation_search*.csv`: toàn bộ candidate FFNN được chấm trên validation.
- `best_model*.csv` hoặc `best_component_models*.csv`: candidate được khóa sau validation.
- `final_forecasts*.csv`: actual/predicted theo date và split.
- `final_metrics*.csv`: metrics theo validation/test.
- `input_audit*.csv`: audit số dòng, lag, feature columns, lag 0, NaN, transform.
- `stationarity_screen_train_only*.csv`: quyết định level/diff1 cho từng biến.
- `actual_vs_predicted*.png`: biểu đồ forecast.

Với pipeline `both`, project còn xuất:

- `vmd_vs_no_vmd_metrics_comparison.csv`
- `vmd_vs_no_vmd_forecasts_comparison.csv`

## 9. Cách Đọc Kết Quả Experiment

Khi review một lần chạy, nên đọc theo thứ tự:

1. `stationarity_screen_*` để biết model học level hay diff1.
2. `ardl_selected_orders_*` để xem target lag được chọn từ ACF/PACF + validation screen và exogenous lags được ARDL chọn.
3. `ffnn_validation_search_*` để xem candidate nào thắng, candidate nào chỉ là fast screen.
4. `best_model_*` hoặc `best_component_models_*` để biết cấu hình cuối cùng được refit.
5. `final_metrics_*` để đánh giá validation/test.
6. `input_audit_*` để kiểm tra có lag 0, NaN hoặc sai khoảng thời gian không.
7. Plot actual vs predicted để nhìn drift, bias hoặc forecast bị lệch pha.

## 10. Nhận Xét Thiết Kế

Điểm mạnh:

- Có tách train/validation/test đúng thứ tự thời gian.
- Chọn lag và stationarity bằng train-only, giảm leakage.
- Validation screen chọn target lag thực nghiệm hơn thay vì chọn cảm tính.
- VMD cache theo forecast origin là hướng đúng cho decomposition trong forecasting.
- Có baseline no-VMD để so sánh tác dụng thật sự của VMD.
- Có audit và diagnostics giúp truy vết quyết định model.

Điểm cần lưu ý:

- VMD pipeline tốn chi phí hơn nhiều vì phải cache decomposition theo từng origin.
- Nếu số features lớn, `full_product` có thể bùng nổ số tổ hợp lag; nên giữ `staged`.
- ACF/PACF candidate generation chỉ đề xuất target lags; quyết định cuối vẫn phụ thuộc validation screen.
- Metrics level cần đọc cùng transform info, nhất là khi target bị log hoặc diff1.
- Nếu validation quá ngắn, chọn model theo RMSE có thể không ổn định; nên đối chiếu thêm MAE, MAPE, Directional Accuracy và plot.

## 11. Câu Lệnh Gợi Ý

Chạy VMD với target lag từ ACF/PACF:

```bash
python -m vmd_ardl_ffnn.cli \
  --pipeline vmd \
  --data ../df_final.csv \
  --out results \
  --date-col TIME_PERIOD \
  --target Export_US \
  --features PCE US_Retail US_Sentiment USD_VND Import_CN \
  --target-lag-strategy validation-screen \
  --target-lag-preset acf-pacf \
  --max-target-lag 12 \
  --max-exog-lag 6 \
  --lag-spec-strategy staged \
  --search-strategy staged-halving
```

Chạy baseline không VMD:

```bash
python -m vmd_ardl_ffnn.cli \
  --pipeline no-vmd \
  --data ../df_final.csv \
  --out results_no_vmd \
  --date-col TIME_PERIOD \
  --target Export_US \
  --features PCE US_Retail US_Sentiment USD_VND Import_CN \
  --target-lag-strategy validation-screen \
  --target-lag-preset acf-pacf
```

Chạy so sánh cả hai:

```bash
python -m vmd_ardl_ffnn.cli \
  --pipeline both \
  --data ../df_final.csv \
  --out results_both \
  --date-col TIME_PERIOD \
  --target Export_US \
  --features PCE US_Retail US_Sentiment USD_VND Import_CN \
  --target-lag-strategy validation-screen \
  --target-lag-preset acf-pacf
```
