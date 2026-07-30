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
| `--pipeline` | `vmd` | `vmd`, `no-vmd`, `both`, `varnn`, `model-comparison`, `benchmark`, `block-proposed-benchmark`, `sensitivity`, `vmd-ardl-select-order`, `vmd-ardl-joint-select`, `vmd-ardl-hard-ffnn` | `vmd` và `vmd-ardl-select-order` dùng proposed sequential SelectOrder-target rồi ARDL-exog screening; `benchmark` giữ study A0–A5 và thêm B7 hard-config; `block-proposed-benchmark` lấy AIC block-subset two-stage làm Proposed, chạy A0–A7 và gồm cùng comparator dưới mã B8; `vmd-ardl-hard-ffnn` grid full-product top-3 exog orders nhưng gán đúng một train-only frequency/lag FFNN config/spec; `sensitivity` chạy robustness grid; joint-select chọn target/exog đồng thời. |
| `--vmd-modes` | `3` | integer | Số VMD modes, chưa tính residual `RES`. |
| `--vmd-backend` | `self-contained` | `self-contained`, `library` | Chọn implementation VMD. `self-contained` dùng code nội bộ; `library` gọi thư viện VMD tùy chọn như `vmdpy`/`sktime.libs.vmdpy`. |
| `--no-stationarity` | `False` | flag | Nếu bật, bỏ qua ADF/KPSS và không tự chọn `level`/`diff1`. |
| `--paper-seeds` | `7 42 123 2024 2025` | one or more integers | Reporting seeds cho `benchmark`, `block-proposed-benchmark`, `vmd-ardl-hard-ffnn` và `sensitivity`; không chọn seed thắng theo test. |
| `--skip-varnn` | `False` | flag | Bỏ VARNN khi không cài TensorFlow; các benchmark/ablation còn lại vẫn chạy. |
| `--block-subset-search-strategy` | `full-grid` | `full-grid`, `two-stage` | Chỉ điều khiển comparator B6 trong `benchmark` cũ. `block-proposed-benchmark` luôn chạy A5 full-grid và A7 two-stage để có direct ablation; A7 screen bằng `alpha=0.1`, LR `0.001`, mean qua paper seeds và giữ top 4 specs/component. |

Khi dùng `--vmd-backend library`, cài dependency tùy chọn trước bằng `pip install .[vmd]` trong thư mục `vmd_ardl_ffnn`, hoặc cài trực tiếp `pip install vmdpy`.

## ARDL Target Lag

| Option | Default | Choices | Ý nghĩa |
| --- | --- | --- | --- |
| `--max-target-lag` | `12` | integer | Lag target tối đa được phép xét. Các preset/candidate có lag lớn hơn giá trị này sẽ bị loại. |
| `--target-lag-strategy` | `ardl-select-order` | `validation-screen`, `joint-validation-screen`, `ic-topk`, `fixed`, `ardl-select-order`, `ardl-select-order-joint` | Cách chọn lag của biến target trước khi chọn exog lag và FFNN search. |
| `--fixed-target-lags` | `1 12` | one or more integers | Bộ target lag dùng khi `--target-lag-strategy fixed`. |
| `--target-lag-top-n` | `2` | integer | Số target lag tốt nhất theo AIC/BIC được giữ khi dùng `ic-topk`. |
| `--target-lag-preset` | `acf-pacf` | `acf-pacf`, `monthly`, `daily`, `yearly` | Bộ candidate target lag dùng khi `--target-lag-strategy validation-screen`. |
| `--target-lag-acf-pacf-top-n` | `3` | integer | Số lag mạnh nhất theo PACF/ACF được xét khi preset là `acf-pacf`. |
| `--target-lag-acf-pacf-max-lags-per-set` | `3` | integer | Số target lag tối đa trong một candidate set khi preset là `acf-pacf`. |
| `--no-force-target-lag-1` | `False` | flag | Nếu bật, không ép lag 1 vào target lag set. Mặc định lag 1 luôn được giữ. |

### `validation-screen`

Đây là phương pháp thay thế/ablation, không phải proposed default. Pipeline làm theo thứ tự:

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
| `--ffnn-learning-rate-grid` | `0.01 0.001` | one or more positive floats | Learning rate được so sánh trên validation; winner được dùng lại khi refit train+validation. |
| `--max-iter` | `500` | integer | Số iteration tối đa cho FFNN ở search chính/refit. |
| `--search-strategy` | `staged-halving` | `staged-halving`, `full-grid` | Cách search FFNN. |
| `--top-k-lag-specs` | `4` | integer | Budget shortlist sau khi mọi `LagSpec` đã được rolling fast-screen. |
| `--top-component-candidates` | `3` | integer | Với VMD, số candidate tốt nhất mỗi component được dùng để ghép reconstructed forecast. |
| `--lag-screen-folds` | `3` | integer | Số expanding folds nằm hoàn toàn trong train để screen lag specs. |
| `--lag-screen-validation-size` | `12` | integer | Số timestamp validation trong mỗi inner fold. |
| `--lag-screen-learning-rate-grid` | dùng FFNN grid | positive floats | Learning rates dùng ở fast-screen; mặc định tái sử dụng `--ffnn-learning-rate-grid`. |
| `--lag-screen-selection-rule` | `one-standard-error` | `top-k`, `one-standard-error` | Giữ raw mean-RMSE winner và candidate đơn giản nhất trong ngưỡng một standard error. |
| `--final-selection-scale` | `level` | `level`, `transformed` | Scale dùng để chọn final reconstructed validation winner. |
| `--fast-max-iter` | `150` | integer | Iteration tối đa cho FFNN fast-screen. |
| `--fast-hr` | không có | integer | Hidden units dùng cho FFNN fast-screen; nếu không truyền, fast width lấy theo số feature. |
| `--fast-hidden-layers` | `1` | integer | Số hidden layer cho fast-screen khi không dùng `--fast-hr`. |
| `--fast-hidden-width-multiplier` | `1.0` | float | Multiplier theo số feature cho fast-screen khi không dùng `--fast-hr`. |
| `--fast-alpha` | `1e-3` | float | Alpha regularization dùng cho FFNN fast-screen. |

Với `staged-halving`, chi phí chính xấp xỉ:

```text
max_lag_specs * 1 fast config
+ top_k_lag_specs * len(architectures) * len(alpha_grid) * len(seed_grid)
  * len(activation_grid) * len(learning_rate_grid)
```

Với rolling screen mặc định, hạng đầu tiên chính xác hơn là:

```text
max_lag_specs * lag_screen_folds * len(lag_screen_learning_rate_grid)
```

Mọi lag spec đều được chấm trước khi shortlist. File stability ghi mean, standard
deviation, standard error, raw winner, one-SE winner và cờ đưa vào full grid.

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
