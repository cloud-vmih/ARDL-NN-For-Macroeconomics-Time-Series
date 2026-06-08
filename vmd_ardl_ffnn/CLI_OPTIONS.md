# CLI Options cho `vmd_ardl_ffnn`

Tài liệu này mô tả đúng các argument hiện có trong `src/vmd_ardl_ffnn/cli.py`.

Chạy từ thư mục `vmd_ardl_ffnn`:

```bash
python -m vmd_ardl_ffnn.cli \
  --data ../df_final.csv \
  --features PCE US_Retail US_Sentiment USD_VND Import_CN
```

## Dữ Liệu Đầu Vào

| Option | Default | Bắt buộc | Ý nghĩa |
| --- | --- | --- | --- |
| `--data` | không có | Có | Đường dẫn file CSV đầu vào. |
| `--out` | `results` | Không | Thư mục ghi các file kết quả CSV/PNG. |
| `--date-col` | `TIME_PERIOD` | Không | Tên cột thời gian trong CSV. |
| `--target` | `Export_US` | Không | Tên biến mục tiêu cần dự báo. |
| `--features` | không có | Có | Danh sách biến ngoại sinh, truyền cách nhau bằng khoảng trắng. |
| `--freq` | `MS` | Không | Tần suất thời gian. Nếu truyền `none`, pipeline không ép tần suất. |
| `--no-log` | `False` | Không | Nếu bật, tắt log transform. Mặc định CLI dùng `log_transform=True` vì `log_transform=not args.no_log`. |

## Pipeline

| Option | Default | Choices | Ý nghĩa |
| --- | --- | --- | --- |
| `--pipeline` | `vmd` | `vmd`, `no-vmd`, `both` | Chọn pipeline chạy. `vmd` chạy VMD + ARDL + FFNN; `no-vmd` chạy baseline không VMD; `both` chạy cả hai và xuất bảng so sánh. |
| `--vmd-modes` | `3` | integer | Số VMD modes, chưa tính residual `RES`. |
| `--no-stationarity` | `False` | flag | Nếu bật, bỏ qua ADF/KPSS và không tự chọn `level`/`diff1`. |

## ARDL Target Lag

| Option | Default | Choices | Ý nghĩa |
| --- | --- | --- | --- |
| `--max-target-lag` | `12` | integer | Lag target tối đa được phép xét. Các preset/candidate có lag lớn hơn giá trị này sẽ bị loại. |
| `--target-lag-strategy` | `validation-screen` | `validation-screen`, `ic-topk`, `fixed` | Cách chọn lag của biến target trước khi chọn exog lag và FFNN search. |
| `--fixed-target-lags` | `1 12` | one or more integers | Bộ target lag dùng khi `--target-lag-strategy fixed`. |
| `--target-lag-top-n` | `2` | integer | Số target lag tốt nhất theo AIC/BIC được giữ khi dùng `ic-topk`. |
| `--target-lag-preset` | `acf-pacf` | `acf-pacf`, `monthly`, `daily`, `yearly` | Bộ candidate target lag dùng khi `--target-lag-strategy validation-screen`. |
| `--target-lag-acf-pacf-top-n` | `3` | integer | Số lag mạnh nhất theo PACF/ACF được xét khi preset là `acf-pacf`. |
| `--target-lag-acf-pacf-max-lags-per-set` | `3` | integer | Số target lag tối đa trong một candidate set khi preset là `acf-pacf`. |
| `--no-force-target-lag-1` | `False` | flag | Nếu bật, không ép lag 1 vào target lag set. Mặc định lag 1 luôn được giữ. |

### `validation-screen`

Đây là default hiện tại. Pipeline làm theo thứ tự:

1. Tạo các candidate target lag từ `--target-lag-preset`.
2. Với mỗi candidate target lag, ARDL chọn exog lag rank-1 theo chính target lag đó.
3. Fit FFNN cấu hình nhanh (`--fast-hr`, `--fast-alpha`, `--fast-max-iter`) và chấm validation.
4. Khóa target lag có validation RMSE tốt nhất.
5. Sau đó mới chạy ARDL exog lag selection chính thức và FFNN search chính thức với target lag đã thắng.

Preset mặc định `acf-pacf` sinh candidate bằng train-only ACF/PACF của target: PACF là nguồn chính để chọn lag tự hồi quy, ACF chỉ bổ sung lag có tương quan mạnh bị PACF bỏ sót. Bảng `ardl_selected_orders...csv` ghi thêm audit gồm `target_lag_source`, `acf_value`, `pacf_value`, và `significance_threshold` cho các dòng `__target_lag_validation_screen__`.

Preset legacy:

| Preset | Candidate target lags |
| --- | --- |
| `acf-pacf` | Sinh động từ train target theo PACF-core + ACF supplement, bị giới hạn bởi `--max-target-lag` và `--target-lag-acf-pacf-max-lags-per-set`. |
| `monthly` | `(1,)`, `(1,2)`, `(1,3)`, `(1,6)`, `(1,12)`, `(1,3,12)`, `(1,6,12)` |
| `daily` | `(1,)`, `(1,2)`, `(1,3)`, `(1,5)`, `(1,7)`, `(1,14)`, `(1,7,14)`, `(1,7,30)` |
| `yearly` | `(1,)`, `(1,2)`, `(1,3)`, `(1,5)` |

## ARDL Exogenous Lag

| Option | Default | Choices | Ý nghĩa |
| --- | --- | --- | --- |
| `--max-exog-lag` | `6` | integer | Lag tối đa cho từng biến ngoại sinh khi ARDL chấm điểm. |
| `--top-n` | `3` | integer | Số exog lag tốt nhất mỗi feature lấy từ ARDL. |
| `--lag-spec-strategy` | `staged` | `staged`, `full-product` | Cách biến top exog lags thành `LagSpec` cho FFNN. |
| `--max-lag-specs` | `16` | integer | Số `LagSpec` tối đa khi dùng `--lag-spec-strategy staged`. |

Với `staged`, pipeline không lấy toàn bộ tích Descartes của top lags. Nó ưu tiên baseline rank-1, các biến thể đổi một feature, rồi các combo có tổng ARDL score tốt nhất cho đến `--max-lag-specs`.

## FFNN Search

| Option | Default | Choices | Ý nghĩa |
| --- | --- | --- | --- |
| `--hidden-layers` | `1 2 3` | one or more integers | Lưới số hidden layer khi dùng width theo số feature. |
| `--hidden-width-multipliers` | `1.0 0.5 0.25` | one or more floats | Nhân với số feature để sinh width cho từng hidden layer. |
| `--hr` | không có | one or more integers | Lưới hidden units legacy cho FFNN một hidden layer; nếu truyền sẽ override `--hidden-layers` và `--hidden-width-multipliers`. |
| `--activation` | `relu` | `relu`, `tanh` | Activation dùng cho fast screening và fallback khi cần. |
| `--activation-grid` | `relu tanh` | `relu`, `tanh` | Danh sách activation được tune trong hyperparameter search chính. |
| `--no-activation-tuning` | `False` | flag | Nếu bật, không tune activation; search chỉ dùng đúng `--activation`. Không dùng chung với `--activation-grid`. |
| `--max-iter` | `500` | integer | Số iteration tối đa cho FFNN ở search chính/refit. |
| `--search-strategy` | `staged-halving` | `staged-halving`, `full-grid` | Cách search FFNN. |
| `--top-k-lag-specs` | `4` | integer | Số `LagSpec` tốt nhất sau fast screen được đưa vào hyperparameter grid khi dùng `staged-halving`. |
| `--top-component-candidates` | `3` | integer | Với VMD, số candidate tốt nhất mỗi component được dùng để ghép reconstructed forecast. |
| `--fast-max-iter` | `150` | integer | Iteration tối đa cho FFNN fast-screen. |
| `--fast-hr` | không có | integer | Hidden units dùng cho FFNN fast-screen; nếu không truyền, fast width lấy theo số feature. |
| `--fast-hidden-layers` | `1` | integer | Số hidden layer cho fast-screen khi không dùng `--fast-hr`. |
| `--fast-hidden-width-multiplier` | `1.0` | float | Multiplier theo số feature cho fast-screen khi không dùng `--fast-hr`. |
| `--fast-alpha` | `1e-3` | float | Alpha regularization dùng cho FFNN fast-screen. |

Với `staged-halving`, chi phí chính xấp xỉ:

```text
max_lag_specs * 1 fast config
+ top_k_lag_specs * len(architectures) * len(alpha_grid) * len(seed_grid) * len(activation_grid)
```

Trong CLI hiện tại, `alpha_grid` và `seed_grid` không có argument riêng; chúng lấy default từ `FFNNConfig`.

## Ví Dụ

### VMD nhanh cho dữ liệu tháng

```bash
python -m vmd_ardl_ffnn.cli \
  --data ../df_final.csv \
  --out results_vmd_monthly \
  --target Export_US \
  --features PCE US_Retail US_Sentiment USD_VND Import_CN \
  --pipeline vmd \
  --target-lag-strategy validation-screen \
  --target-lag-preset acf-pacf \
  --lag-spec-strategy staged \
  --max-lag-specs 16 \
  --search-strategy staged-halving \
  --top-k-lag-specs 4 \
  --top-component-candidates 3
```

### No-VMD baseline

```bash
python -m vmd_ardl_ffnn.cli \
  --data ../df_final.csv \
  --out results_no_vmd \
  --target Export_US \
  --features PCE US_Retail US_Sentiment USD_VND Import_CN \
  --pipeline no-vmd
```

### Tái lập target lag cố định `(1, 12)`

```bash
python -m vmd_ardl_ffnn.cli \
  --data ../df_final.csv \
  --out results_fixed_target_lag \
  --target Export_US \
  --features PCE US_Retail US_Sentiment USD_VND Import_CN \
  --pipeline vmd \
  --target-lag-strategy fixed \
  --fixed-target-lags 1 12
```

### Chạy full grid để kiểm chứng

```bash
python -m vmd_ardl_ffnn.cli \
  --data ../df_final.csv \
  --out results_full_grid \
  --target Export_US \
  --features PCE US_Retail US_Sentiment USD_VND Import_CN \
  --pipeline vmd \
  --lag-spec-strategy full-product \
  --search-strategy full-grid
```
