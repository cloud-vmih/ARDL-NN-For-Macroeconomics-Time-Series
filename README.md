# VMD-ARDL-FFNN Forecasting Package

Thư mục `Project/vmd_ardl_ffnn` là phiên bản package hóa của thực nghiệm ARDL + FFNN, có thêm pipeline VMD và các cơ chế đánh giá time series tránh leakage.

Mục tiêu chính của package là so sánh hai pipeline dự báo:

- `no-vmd`: ARDL lag selection + FFNN trực tiếp trên chuỗi mục tiêu và biến ngoại sinh.
- `vmd`: phân rã VMD, dự báo từng component bằng ARDL lag specs + FFNN, sau đó cộng component forecasts để tạo dự báo cuối.

## Điểm Khác So Với `ardl_ffnn_model.py`

File `ardl_ffnn_model.py` là flow notebook-style để thử nghiệm nhanh ARDL + FFNN. Package `vmd_ardl_ffnn` tập trung vào đánh giá dự báo nghiêm ngặt hơn:

- Dữ liệu được sort theo `TIME_PERIOD`.
- Chia `train / validation / test` theo thứ tự thời gian, không shuffle.
- ARDL lag selection chỉ dùng train.
- Exogenous lag 0 bị loại khi forecast thật sự.
- Validation dùng để chọn model.
- Test chỉ dùng sau cùng sau khi model đã khóa.
- Có input audit để kiểm tra feature columns, lags, lag 0, NaN và khoảng thời gian.
- Có biểu đồ Actual vs Predicted cho validation và test.

## Cấu Trúc Chính

```text
Project/
├── df_final.csv
├── ardl_ffnn_model.py
├── README.md
└── vmd_ardl_ffnn/
    ├── pyproject.toml
    ├── src/vmd_ardl_ffnn/
    │   ├── cli.py
    │   ├── config.py
    │   ├── data.py
    │   ├── decomposition.py
    │   ├── diagnostics.py
    │   ├── experiment.py
    │   ├── experiment_fixed.py
    │   ├── features.py
    │   ├── lag_selection.py
    │   ├── metrics.py
    │   └── models/ffnn.py
    └── tests/test_smoke.py
```

## Pipeline Hiện Tại

### Bước Chung

1. Load dữ liệu từ CSV.
2. Parse và sort `TIME_PERIOD` tăng dần.
3. Chọn các cột target và features.
4. Log-transform nếu `log_transform=True`.
5. Split chronological thành train, validation, test.
6. Chạy stationarity screen bằng ADF/KPSS trên train.
7. Áp dụng transform:
   - `level`: giữ nguyên chuỗi sau bước load/log.
   - `diff1`: sai phân bậc nhất.
8. Chạy ARDL lag selection trên train đã transform.
9. Grid search FFNN trên validation.
10. Refit model thắng trên train + validation.
11. Đánh giá test sau cùng.

### Pipeline `no-vmd`

Pipeline này không phân rã VMD.

Flow:

1. ARDL chọn lag specs trên train raw/model-scale.
2. Convert lag specs thành forecast-safe specs.
3. Với mỗi candidate `lag_spec`, `hidden_units`, `alpha`, `seed`:
   - fit FFNN đúng một lần trên train.
   - predict toàn bộ validation theo batch bằng observed lagged inputs.
4. Chọn candidate tốt nhất theo `Val RMSE`, `Val MAE`, `Val MAPE`.
5. Refit candidate thắng trên train + validation.
6. Predict test theo batch.
7. Lưu metrics ở model scale và level scale.

### Pipeline `vmd`

Pipeline này dùng VMD nhưng tránh leakage.

Flow:

1. VMD chỉ chạy trên train để tạo train components.
2. Chọn ARDL lag specs riêng cho từng component:
   - `VMD1`
   - `VMD2`
   - `VMD3` nếu cấu hình `modes=3`
   - `RES`
3. Tạo validation cache theo từng forecast origin:
   - tại timestamp `t`, VMD chỉ thấy history trước `t`.
   - không dùng actual tại `t` trước khi predict.
4. Candidate FFNN cho từng component fit đúng một lần trên component train.
5. Khi đánh giá validation, không chạy lại VMD theo candidate.
6. Predict từng component rồi cộng lại thành `predicted_reconstructed`.
7. Chọn tổ hợp component models tốt nhất theo reconstructed validation metrics.
8. Refit component models thắng trên train + validation.
9. Tạo test cache theo cùng logic validation.
10. Predict từng component test và cộng lại để đánh giá final reconstructed forecast.

## Stationarity Và Scale

Package có stationarity screen mặc định.

- ADF/KPSS chỉ chạy trên train.
- Nếu chuỗi train stationary: `Transform used = level`.
- Nếu không stationary: `Transform used = diff1`.

Lưu ý: `level` ở stationarity nghĩa là giữ nguyên chuỗi ở base scale sau bước load.

- Nếu `log_transform=True`, `level` nghĩa là dùng `log(Y)`, không phải `Y` gốc.
- Nếu `log_transform=False`, `level` nghĩa là dùng `Y` gốc.
- Nếu `diff1`, model dùng sai phân của base scale.

File metrics có thể có hai scale:

- `raw_transformed` hoặc `reconstructed_transformed`: scale model đang học/dự báo.
- `level`: scale gốc để đọc kết quả; nếu có log thì đã inverse bằng `exp`, nếu có `diff1` thì đã reconstruct từ giá trị trước đó.

## Cài Đặt

Từ thư mục `Project/vmd_ardl_ffnn`:

```bash
cd Project/vmd_ardl_ffnn
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
```

Nếu không cần chạy test:

```bash
pip install -e .
```

## Chạy Pipeline VMD

Ví dụ cho target `Export_US`:

```bash
cd Project/vmd_ardl_ffnn

python -m vmd_ardl_ffnn.cli \
  --pipeline vmd \
  --data ../df_final.csv \
  --out results \
  --date-col TIME_PERIOD \
  --target Export_US \
  --features PCE US_Retail US_Sentiment USD_VND Import_CN \
  --vmd-modes 3 \
  --max-target-lag 12 \
  --max-exog-lag 6 \
  --top-n 3 \
  --hr 4 8 12 16 \
  --max-iter 500
```

## Chạy Pipeline Không VMD

```bash
cd Project/vmd_ardl_ffnn

python -m vmd_ardl_ffnn.cli \
  --pipeline no-vmd \
  --data ../df_final.csv \
  --out results_no_vmd \
  --date-col TIME_PERIOD \
  --target Export_US \
  --features PCE US_Retail US_Sentiment USD_VND Import_CN
```

## Tùy Chọn Hữu Ích

- `--pipeline vmd`: chạy pipeline có VMD.
- `--pipeline no-vmd`: chạy pipeline không VMD.
- `--no-log`: tắt log-transform.
- `--no-stationarity`: tắt ADF/KPSS stationarity transform.
- `--vmd-modes`: số mode VMD.
- `--max-target-lag`: lag target tối đa cho cấu hình ARDL.
- `--max-exog-lag`: lag ngoại sinh tối đa.
- `--top-n`: số lag top mỗi feature được đưa vào grid.
- `--hr`: danh sách hidden units cho FFNN.
- `--max-iter`: số vòng lặp tối đa của MLPRegressor.

## Output Pipeline Không VMD

Khi chạy `--pipeline no-vmd`, các file chính gồm:

```text
results_no_vmd/
├── ffnn_validation_search_fixed_no_vmd.csv
├── best_model_fixed_no_vmd.csv
├── final_forecasts_fixed_no_vmd.csv
├── final_metrics_fixed_no_vmd.csv
├── input_audit_no_vmd.csv
├── stationarity_screen_train_only_no_vmd.csv
├── ardl_selected_orders_train_only_no_vmd.csv
├── series_diagnostics_train_only_no_vmd.csv
├── granger_causality_train_only_no_vmd.csv
├── ardl_residual_diagnostics_train_only_no_vmd.csv
├── actual_vs_predicted_val_no_vmd.png
└── actual_vs_predicted_test_no_vmd.png
```

## Output Pipeline VMD

Khi chạy `--pipeline vmd`, các file chính gồm:

```text
results/
├── ffnn_validation_search_cached_vmd.csv
├── best_component_models_cached_vmd.csv
├── component_predictions_cached_vmd.csv
├── final_reconstructed_forecasts_cached_vmd.csv
├── final_reconstructed_metrics_cached_vmd.csv
├── input_audit_cached_vmd.csv
├── stationarity_screen_train_only_cached_vmd.csv
├── ardl_selected_orders_train_only.csv
├── series_diagnostics_train_only.csv
├── granger_causality_train_only.csv
├── ardl_residual_diagnostics_train_only.csv
├── actual_vs_predicted_val_cached_vmd.png
└── actual_vs_predicted_test_cached_vmd.png
```

## Ý Nghĩa Một Số File Output

- `stationarity_screen_*`: kết quả ADF/KPSS train-only và transform được chọn cho từng biến.
- `input_audit_*`: audit số dòng, khoảng thời gian, feature columns, target lags, exogenous lags, lag 0 và NaN.
- `ffnn_validation_search_*`: bảng grid search validation.
- `best_model_*` hoặc `best_component_models_*`: model/candidate được khóa sau validation.
- `component_predictions_cached_vmd.csv`: dự báo từng component VMD.
- `final_forecasts_*`: actual/predicted theo thời gian.
- `final_metrics_*`: RMSE, MAE, MAPE, Directional Accuracy ở model scale và level scale.
- `actual_vs_predicted_*.png`: biểu đồ Actual vs Predicted theo thời gian.

## Chạy Test

```bash
cd Project/vmd_ardl_ffnn
pytest -q
```

Nếu môi trường chưa có `pytest`, cài bằng:

```bash
pip install -e ".[test]"
```

## Ghi Chú Kỹ Thuật

- Directional Accuracy được tính theo chiều thay đổi của chuỗi, không phải theo dấu dương/âm của giá trị.
- Exogenous lag 0 bị loại để tránh dùng thông tin cùng timestamp khi dự báo thật sự.
- VMD validation/test được cache theo forecast origin để không phân rã toàn bộ validation/test trước.
- Test set không được dùng để chọn lag, chọn FFNN hyperparameters hoặc chọn component model.
