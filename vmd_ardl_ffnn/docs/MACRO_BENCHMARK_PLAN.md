# Kế hoạch và workflow benchmark macro

## Mục tiêu

Triển khai `macro_indicators.csv` theo notebook Kaggle
[`forecasting-a-convenient-macro-indicator`](https://www.kaggle.com/code/awing11/forecasting-a-convenient-macro-indicator),
nhưng tách rõ hai mục đích: tái lập notebook và so sánh công bằng với pipeline
shared-mean-selector của project.

Notebook chỉ dùng chuỗi đơn biến `unemployment_rate` với 12 lag của chính target;
nó không đưa các cột macro còn lại vào MLP/LSTM/GRU. Cấu hình được tái lập là
MinMax `[0,1]`, split 80/20, seed 42, Adam `0.001`, MSE, batch 16, tối đa 150
epoch, EarlyStopping patience 15 và ReduceLROnPlateau patience 6. Bản tái lập
được gắn cảnh báo vì notebook chọn kiến trúc tốt nhất bằng test MAPE.

## Dữ liệu và giao thức công bằng

- Target: `unemployment_rate`.
- Dùng prefix tháng liên tục từ `2000-01-01` đến ngay trước target bị thiếu đầu
  tiên, hiện là `2025-09-01` (309 dòng).
- Train: `2000-01`–`2017-12`; validation: `2018-01`–`2021-10`; test:
  `2021-11`–`2025-09`.
- Candidate ngoại sinh cố định: `initial_claims`, `nonfarm_payrolls`,
  `consumer_sentiment`, `industrial_prod`, `retail_sales`, `building_permits`.
- Chỉ trên train, chấm ARDL theo target order 1–12 và distributed exogenous
  order 1–6; xếp mỗi feature bằng AIC tốt nhất và giữ đúng bốn feature.
- Shared winner được chọn bằng mean validation RMSE qua seeds
  `7, 42, 123, 2024, 2025`; test chỉ được mở sau khi khóa winner.
- Báo cáo level RMSE, MAE, MAPE, R² và Directional Accuracy.

## Hai profile thực thi

`core-only` là mặc định và chạy bảy workflow nội bộ/tham chiếu:

1. `ardl_ffnn_no_vmd`
2. `vmd_ardl_block_subset_proposed`
3. `vmd_ardl_sequential_screen`
4. `vmd_ardl_joint_select`
5. `varnn`
6. `naive_last_observation`
7. `vmd_ardl_univariate_shared_mean`

Profile này không build/train/import TensorFlow graph của MLP/LSTM/GRU notebook,
không tạo thư mục artifact notebook và tiêu đề kết quả là “internal workflow
comparison”, không gọi đó là Kaggle benchmark.

`full` chạy thêm exact reproduction và ba comparator
công bằng MLP/LSTM/GRU. Artifact exact reproduction và fair comparison nằm ở
hai thư mục riêng để không trộn test-selected result với validation-selected
result.

## Backend và audit

- FFNN project dùng `TensorFlowFFNNRegressor`; scaler train-only có thể chọn
  `standard` hoặc `minmax`. Alias `NumpyFFNNRegressor` và
  `SklearnFFNNRegressor` chỉ còn để tương thích một phiên bản và phát
  `DeprecationWarning`; không có NumPy fallback.
- FFNN/VARNN nội bộ dùng cố định activation `tanh` và learning rate `0.001`;
  không còn grid-search hai tham số này. Cấu hình exact Kaggle reproduction
  vẫn giữ nguyên để không làm sai benchmark nguồn.
- ADF/KPSS chạy trên train-only với alpha 5%. Chỉ chuỗi đồng thời có ADF
  p-value dưới 0.05 và KPSS p-value trên 0.05 mới giữ level; các chuỗi còn lại
  dùng sai phân bậc một và được audit trong `stationarity_screen`.
- VMD production dùng `sktime.libs.vmdpy`; `library` là alias cũ của `sktime`.
  Backend `self_contained` chỉ được giữ như lựa chọn compatibility ngoài macro.
- Macro mặc định yêu cầu GPU rõ ràng và fail trước khi tạo run directory nếu GPU
  không tồn tại. CPU chỉ dùng khi truyền `--tensorflow-device cpu`, phù hợp cho
  test/smoke có chủ ý.
- `manifest.json` ghi profile, pipeline bật/tắt, features được chọn, seeds,
  scaler, backend, hash dataset/notebook, TensorFlow/Keras/CUDA/cuDNN và GPU.

## Lệnh chạy

```bash
# Nhánh nhẹ mặc định: không chạy Naive/MLP/LSTM/GRU/notebook reproduction
./.venv/bin/python3 run_pipeline_advanced.py \
  --pipeline macro-benchmark \
  --data macro_indicators.csv \
  --macro-profile core-only

# Toàn bộ benchmark, gồm cả LSTM và GRU
./.venv/bin/python3 run_pipeline_advanced.py \
  --pipeline macro-benchmark \
  --data macro_indicators.csv \
  --macro-profile full

# Smoke có chủ ý trên CPU
./.venv/bin/python3 run_pipeline_advanced.py \
  --pipeline macro-benchmark \
  --macro-profile core-only \
  --tensorflow-device cpu \
  --paper-seeds 7
```

## Nhánh sequential tiết kiệm tài nguyên

`sequential-vmd-ardl-ffnn` không chạy lại feature screening. Nó dùng đúng tuple
`SEQUENTIAL_MACRO_FEATURES` trong `run_pipeline_advanced.py` và chỉ tune seed 7.
VMD origin được lưu theo content hash ở `results/_cache/vmd`, chỉ giữ tối đa hai
component đã materialize trong RAM. Candidate FFNN và từng fold staged-halving
được checkpoint trong `results/sequential-vmd-ardl-ffnn/<run-id>/checkpoints`.

```bash
./.venv/bin/python3 run_pipeline_advanced.py \
  --pipeline sequential-vmd-ardl-ffnn \
  --data macro_indicators.csv \
  --run-id macro-sequential-01

./.venv/bin/python3 run_pipeline_advanced.py \
  --pipeline sequential-vmd-ardl-ffnn \
  --data macro_indicators.csv \
  --run-id macro-sequential-01 \
  --resume
```

## Tiêu chí nghiệm thu

- Feature screening và mọi lựa chọn model không đọc validation/test sai phạm.
- VMD cache vẫn chạy origin-by-origin.
- `core-only` có đúng bảy pipeline và không sinh artifact notebook.
- `full` giữ exact reproduction tách biệt khỏi fair comparison.
- TensorFlow và sktime được gọi thật trong production; test riêng xác nhận scaler,
  device fail-fast, shape VMD và compatibility aliases.
