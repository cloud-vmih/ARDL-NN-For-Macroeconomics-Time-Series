# VMD–ARDL–FFNN Forecasting Project

## Benchmark thất nghiệp Mỹ từ BLS/FRED

Study mới dùng `UNRATE` làm target và bốn biến `AWHMAN`, `PAYEMS`, `CPIAUCSL`,
`INDPRO`. Panel được dựng từ FRED/ALFRED vintages, có file audit và mã băm;
`df_final.csv` không được dùng trong study này. Cần đặt `FRED_API_KEY` trong môi
trường trước khi dựng panel:

```bash
set -a; . ./.env; set +a  # Chỉ cần khi khóa được lưu trong .env
PYTHONPATH=vmd_ardl_ffnn/src ./.venv/bin/python3 -m vmd_ardl_ffnn.macro_fred_panel \
  --cache-root data/macro_us/raw \
  --output-root data/macro_us/processed

./.venv/bin/python3 run_pipeline_advanced.py \
  --pipeline shared-mean-benchmark \
  --dataset-profile us-unemployment \
  --data data/macro_us/processed/us_unemployment_awhman_2000_2025_09.csv \
  --tensorflow-device gpu \
  --tensorflow-run-eagerly \
  --paper-seeds 7 42 123 2024 2025 \
  --run-id us-unemployment-lstm-04
```

Benchmark này chạy bốn shared-mean workflow hiện có, VAR-NN, LSTM, GRU,
MLP-ARX, XGBoost, Ridge-ARX, SARIMAX và
naïve last observation trên cùng validation/test dates. `--tensorflow-run-eagerly` hạn chế
RAM tăng qua nhiều lượt fit FFNN và được ghi vào manifest. Panel chính có 309 tháng
01/2000–09/2025; 10/2025 không được nội suy. Ngoại sinh của dòng cuối
09/2025 dùng vintage 31/01/2026 do công bố bị chậm và được audit là không
tham gia dự báo nào. Lần tải đầu cần hơn 1.200 yêu cầu API; module tự giới hạn
tốc độ và có thể chạy lại để dùng cache. Thiết kế và các bước kiểm tra
nằm trong [kế hoạch tiếng Việt](docs/superpowers/plans/2026-10-30-us-unemployment-shared-mean-varnn-vi.md).
`AWHMAN` là số giờ làm việc bình quân mỗi tuần của lao động sản xuất và phi
quản lý trong ngành chế tạo. Chuỗi này thay `HOUST`: HOUST chưa được công bố
đúng hạn ở 09–10/2013 và 12/2018–01/2019, còn panel chính yêu cầu đúng
tháng nguồn tại mọi mốc dự báo. `ICSA` không có vintage ALFRED trước 05/2009.

Pipeline macro mới dùng TensorFlow/Keras cho FFNN và `sktime` cho VMD. Nhánh
nhẹ mặc định không chạy model notebook:

```bash
./.venv/bin/python3 run_pipeline_advanced.py \
  --pipeline macro-benchmark \
  --data macro_indicators.csv \
  --macro-profile core-only
```

Notebook nghiên cứu feature độc lập nằm tại
`notebooks/01_macro_feature_screening.ipynb`. Nó khóa validation/test, tạo các
signal trễ forecast-safe và chọn feature bằng expanding-window validation trên
train. Cài kernel notebook nếu môi trường chưa có:

```bash
./.venv/bin/python3 -m pip install -e './vmd_ardl_ffnn[notebook]'
./.venv/bin/python3 -m jupyter lab notebooks/01_macro_feature_screening.ipynb
```

Kết quả được ghi vào `results/macro-feature-screening/`; file
`selected_features.json` chứa các biến gốc khuyến nghị để đưa vào pipeline.

Nhánh production gọn chỉ chạy VMD–ARDL–FFNN, khóa một tuning seed và bốn feature:

```bash
./.venv/bin/python3 run_pipeline_advanced.py \
  --pipeline sequential-vmd-ardl-ffnn \
  --data macro_indicators.csv \
  --run-id macro-sequential-01

# Chạy tiếp sau khi máy bị dừng; config/data phải trùng manifest cũ.
./.venv/bin/python3 run_pipeline_advanced.py \
  --pipeline sequential-vmd-ardl-ffnn \
  --data macro_indicators.csv \
  --run-id macro-sequential-01 \
  --resume
```

Muốn đổi bốn biến đầu vào, chỉ sửa tuple `SEQUENTIAL_MACRO_FEATURES` gần đầu
file `run_pipeline_advanced.py`. VMD dùng cache chung tại `results/_cache/vmd`;
checkpoint FFNN nằm trong thư mục run.

Các model neural nội bộ mặc định chỉ dùng activation `tanh` và learning rate
`0.001`. Macro project workflows chạy ADF/KPSS trên train-only, giữ level khi
cả hai kiểm định đồng thuận chuỗi dừng và dùng `diff1` trong trường hợp còn lại.
Nhóm exact Kaggle reproduction giữ nguyên cấu hình notebook gốc.

Chỉ dùng `--macro-profile full` khi cần thêm exact Kaggle reproduction,
MLP, LSTM và GRU. Macro production mặc định yêu cầu GPU; CPU phải được chọn rõ
bằng `--tensorflow-device cpu` cho smoke/test. Chi tiết nằm tại
`vmd_ardl_ffnn/docs/MACRO_BENCHMARK_PLAN.md`.

Project dự báo chuỗi thời gian theo giao thức rolling one-step, với pipeline chính:

```text
VMD theo forecast origin
→ ARDL SelectOrder chọn target lag trên train
→ ARDL screen từng exogenous order với target lag đã khóa
→ rolling fast-screen mọi lag spec trên các inner train folds
→ FFNN outer-validation search trên lag shortlist
→ cộng dự báo component
→ chọn reconstructed winner bằng level RMSE
→ refit train + validation
→ đánh giá test một lần
```

File chạy chính là `run_pipeline_advanced.py`; sau khi cài package bằng
`pip install -e ./vmd_ardl_ffnn`, có thể dùng `vmd-ardl-ffnn-run` với cùng CLI.
Mọi run phải truyền `--data`.
`df_final.csv` đã bị loại vì lỗi tổng hợp dữ liệu và CLI từ chối file này.
Các ví dụ legacy dưới đây yêu cầu một CSV khác đã kiểm định, có schema
`TIME_PERIOD`, `Export_US` và các biến khai báo trong cấu hình.

## Chạy nhanh

Nên dùng đúng Python trong `.venv`, vì Python của pyenv/system có thể không nhìn
thấy TensorFlow đã cài trong project:

```bash
cd /home/cloud/cloud/DA/Project

# Pipeline proposed mặc định trên dữ liệu legacy đã kiểm định
./.venv/bin/python3 run_pipeline_advanced.py --data /path/to/validated_legacy.csv

# Gọi tên pipeline rõ ràng
./.venv/bin/python3 run_pipeline_advanced.py \
  --pipeline vmd-ardl-select-order \
  --data /path/to/validated_legacy.csv

# Full paper benchmark, ablation và kiểm định thống kê
./.venv/bin/python3 run_pipeline_advanced.py --pipeline benchmark --data /path/to/validated_legacy.csv

# Study riêng: block-subset workflow là Proposed chính, shared mean-seed selection
./.venv/bin/python3 run_pipeline_advanced.py \
  --pipeline block-proposed-benchmark \
  --data /path/to/validated_legacy.csv \
  --paper-seeds 7 42 123 2024 2025

# So sánh shared mean: bốn FFNN workflow, VAR-NN và naïve
./.venv/bin/python3 run_pipeline_advanced.py \
  --pipeline shared-mean-benchmark \
  --dataset-profile us-unemployment \
  --data data/macro_us/processed/us_unemployment_awhman_2000_2025_09.csv \
  --tensorflow-device gpu \
  --tensorflow-run-eagerly \
  --paper-seeds 7 42 123 2024 2025

# Comparator hard-config: grid lag combinations, không grid FFNN hyperparameters
./.venv/bin/python3 run_pipeline_advanced.py \
  --pipeline vmd-ardl-hard-ffnn \
  --data /path/to/validated_legacy.csv \
  --paper-seeds 7 42 123 2024 2025
```

Mỗi lần chạy bằng entrypoint này tạo một thư mục riêng:

```text
results/<pipeline>/<run_id>/
```

Trong đó `manifest.json` lưu dataset, config, dependency versions, trạng thái run
và lỗi nếu pipeline thất bại.

## Cấu hình mặc định hiện tại

Cấu hình legacy được khai báo trong `advanced_config.py`; cấu hình macro nằm
trong `macro_benchmark.py`. `run_pipeline_advanced.py` chỉ là wrapper tương thích.

| Nhóm | Cấu hình chính |
| --- | --- |
| Date | `TIME_PERIOD`, tần suất `MS` |
| Target | `Export_US` |
| Exogenous | `PCE`, `US_Retail`, `USD_VND`, `Import_CN` |
| Transform đầu vào | Log tự nhiên; mọi giá trị phải dương |
| Split | 70% train, 15% validation, 15% test; không shuffle |
| Forecast protocol | `rolling_one_step` |
| VMD | 3 modes + `RES`, `alpha=2000`, `sktime.libs.vmdpy` backend |
| ARDL | AIC, target lag tối đa 12, exogenous lag tối đa 6 |
| Lag specs | Staged search, tối đa 16 specs, giữ top 3 exogenous orders |
| Lag fast-screen | 3 expanding folds × 12 tháng; learning rate theo profile đang chạy |
| Lag shortlist | Top 4, luôn giữ raw RMSE winner và one-standard-error winner đơn giản nhất |
| FFNN activation | `tanh` |
| FFNN alpha | `1e-4`, `1e-3`, `1e-2`, `1e-1` |
| FFNN learning rate | Profile hiện tại dùng `0.001`; cấu hình grid nằm trong `build_experiment_config()` hoặc `build_macro_experiment_config()` |
| FFNN seed grid ở single pipeline | `7`, `42`, `123`; chọn bằng validation |
| Paper seeds | `7`, `42`, `123`, `2024`, `2025` |
| VARNN | 32 hidden units, `tanh`, learning rate `0.001` trong profile shared-mean hiện tại |

Nếu đổi target hoặc features, cần sửa cấu hình trong `run_pipeline_advanced.py`,
hoặc dùng CLI linh hoạt `python -m vmd_ardl_ffnn.cli` sau khi package đã được cài.

## Workflow hiện tại

### 1. Nạp và kiểm tra dữ liệu

`GenericDataLoader` thực hiện:

1. Kiểm tra file và các cột bắt buộc.
2. Parse `TIME_PERIOD`, sort tăng dần và từ chối timestamp trùng.
3. Ép tần suất tháng bằng `asfreq("MS")`.
4. Mặc định báo lỗi nếu thiếu kỳ, có NaN hoặc giá trị không phải số.
5. Kiểm tra mọi biến đều dương trước khi log transform.

Không tự động nội suy dữ liệu. Đây là chủ ý để tránh che giấu kỳ bị thiếu.

### 2. Split và stationarity

Dữ liệu được split theo thời gian thành train/validation/test trước khi lựa chọn
model. ADF và KPSS chỉ được đánh giá trên train:

- Nếu level đạt điều kiện stationary: giữ `level`.
- Nếu không: dùng sai phân bậc nhất `diff1`.

Quyết định transform từ train được áp dụng nhất quán cho toàn chuỗi. Khi báo cáo,
pipeline đảo sai phân và đảo log để tạo metric trên thang level gốc.

### 3. Proposed sequential lag selection

Với mỗi VMD component (`VMD1`, `VMD2`, `VMD3`, `RES`):

1. `statsmodels.ardl_select_order` chạy trên train để chọn target lag.
2. Target lag được khóa.
3. Với từng exogenous variable và từng order từ 1 đến `max_exog_lag`, ARDL chấm
   AIC trên cùng một `hold_back`.
4. Giữ top exogenous orders và xây candidate `LagSpec` bằng staged search.
5. Lag 0 bị cấm; FFNN chỉ nhận target/exogenous lag từ 1 trở lên.

Pipeline `vmd-ardl-joint-select` khác ở chỗ target và exogenous lags được
`ardl_select_order` chọn đồng thời. Nó là comparator/ablation A5, không phải
phương pháp proposed chính.

Benchmark còn có comparator `vmd_ardl_block_subset_mean_seed`, chạy riêng với
workflow sau và không thay thế A4 Proposed:

1. ARDL SelectOrder chọn rồi khóa target lag trên train cho từng VMD component.
2. Với mỗi exogenous variable, AIC chọn đúng một distributed-lag order `q_j`;
   block tương ứng là `(x_j,t-1, ..., x_j,t-q_j)`.
3. Sinh mọi tập con khác rỗng của các variable block. Với 4 exogenous variables
   có `2^4 - 1 = 15` lag specs cho mỗi component.
4. Chạy full FFNN grid cho mọi lag spec và mọi reporting seed.
5. Với từng component, dùng mean component validation score qua seeds để giữ
   `top_component_candidates` shared signatures.
6. Sinh Cartesian combinations giữa các component shortlists, reconstruct
   validation forecast cho từng seed và tính level RMSE.
7. Chọn một tổ hợp chung bằng **mean reconstructed level validation RMSE qua
   seeds**; level MAE, MAPE và input dimension là tie-breaker.
8. Khóa tổ hợp thắng, refit cho từng seed rồi đánh giá test trên level scale.

Ở chế độ mặc định `full-grid`, comparator cố ý bỏ qua rolling shortlist/top-k để
đánh giá exhaustive block-subset workflow. Vì thế runtime lớn hơn đáng kể.

Trong benchmark có thể bật chế độ two-stage riêng cho comparator:

```bash
./.venv/bin/python3 run_pipeline_advanced.py \
  --pipeline benchmark \
  --data /path/to/validated_legacy.csv \
  --block-subset-search-strategy two-stage
```

Two-stage vẫn sinh và fast-screen đủ 15 block specs/component. Fast screen dùng
một hidden layer nhẹ theo input dimension, `alpha=0.1`, learning rate `0.001`,
`fast_max_iter=150`, và lấy mean component validation score qua reporting seeds.
Chỉ top 4 lag specs/component đi vào full FFNN grid. Sau đó objective cuối vẫn là
mean reconstructed level validation RMSE qua seeds. Mặc định
`--block-subset-search-strategy full-grid` giữ workflow exhaustive ban đầu.

### 4. VMD không nhìn tương lai

Train được phân rã riêng. Với validation và test, pipeline tạo cache theo từng
forecast origin:

```text
dự báo tại t bằng history trước t
→ sau khi dự báo xong mới quan sát actual tại t
→ actual tại t được phép dùng khi dự báo t+1
```

Candidate FFNN dùng chung cache nên VMD không bị chạy lại theo từng candidate,
nhưng vẫn giữ đúng information set của rolling one-step forecast.

### 5. Train-only rolling lag screen và FFNN validation search

Staged-halving search có hai giai đoạn:

1. Mọi lag spec do ARDL sinh ra đều được fast-screen trên ba expanding folds
   nằm hoàn toàn trong train. Mỗi fold dài 12 tháng và thử cả learning rate
   `0.01`, `0.001`.
2. Score của lag spec là mean RMSE qua các folds. Pipeline luôn giữ candidate
   có mean RMSE thấp nhất và candidate đơn giản nhất nằm trong ngưỡng one
   standard error; các slot còn lại được điền theo mean-RMSE rank đến top 4.
3. Với lag shortlist đã khóa, outer validation search qua:
   - kiến trúc 1–3 hidden layers;
   - width theo số input features;
   - alpha grid;
   - activation grid;
   - learning-rate grid của profile đang chạy (`0.001` ở pipeline mặc định;
     `0.01` và `0.001` ở `block-proposed-benchmark`);
   - seed hiện hành.

Toàn bộ candidate đều được đánh giá ở fast-screen trước khi shortlist, nên việc
giảm số cấu hình ở full grid không phải cắt lag space trước khi quan sát. Hai file
`ffnn_lag_screen_folds_*.csv` và `ffnn_lag_screen_stability_*.csv` lưu fold score,
mean/std/SE, learning rate tốt nhất, rank và lý do candidate được giữ.

Với VMD, pipeline chọn top candidate của từng component rồi chọn tổ hợp có
reconstructed **level RMSE** tốt nhất. Level MAE, MAPE và tổng input dimension
được dùng làm tie-breaker. Metric transformed vẫn được lưu để chẩn đoán nhưng
không còn là objective chọn final paper winner.

### 6. Refit và test

Sau khi lag structure và hyperparameters đã khóa:

1. Refit winner trên train + validation.
2. Tạo test forecast theo rolling one-step.
3. Tính RMSE, MAE, MAPE và Directional Accuracy.
4. Test không được dùng để chọn lag, learning rate, kiến trúc, seed hoặc model.

Primary paper metrics là metric trên `scale="level"`. Metric transformed chỉ nên
dùng để chẩn đoán.

## Các pipeline có thể chạy

| Pipeline | Mục đích |
| --- | --- |
| `vmd-ardl-select-order` | Proposed VMD + sequential ARDL target/exog + FFNN; default |
| `vmd` | Cùng target-lag strategy với proposed, tên rút gọn |
| `vmd-ardl-joint-select` | Joint ARDL target/exogenous comparator |
| `vmd-ardl-hard-ffnn` | Full-product top-3 ARDL lag search; một train-only frequency/lag FFNN config cho mỗi candidate |
| `no-vmd` | Sequential ARDL–FFNN trực tiếp trên target, không VMD |
| `both` | So sánh VMD và no-VMD |
| `varnn` | Chạy VARNN độc lập |
| `model-comparison` | So sánh proposed VMD–ARDL–FFNN với VARNN |
| `benchmark` | Full benchmark B0–B7, proposed, ablation A0–A5, DM/Holm |
| `block-proposed-benchmark` | Study riêng lấy block-subset two-stage làm Proposed, shared mean-seed benchmarks, ablation A0–A7, DM/Holm |
| `shared-mean-benchmark` | Bốn FFNN workflow shared mean, VAR-NN, LSTM và naïve; dùng `--dataset-profile us-unemployment` cho panel BLS/FRED có audit |
| `sensitivity` | Robustness grid one-factor-at-a-time cho proposed model |

Ví dụ:

```bash
# Không VMD
./.venv/bin/python3 run_pipeline_advanced.py --pipeline no-vmd --data /path/to/validated_legacy.csv

# Joint selection comparator
./.venv/bin/python3 run_pipeline_advanced.py \
  --pipeline vmd-ardl-joint-select \
  --data /path/to/validated_legacy.csv

# Chọn output root khác
./.venv/bin/python3 run_pipeline_advanced.py \
  --pipeline vmd-ardl-select-order \
  --data /path/to/validated_legacy.csv \
  --output-root outputs
```

Lưu ý dataset khác vẫn phải có đúng các cột đang khai báo trong config của
entrypoint. Truyền `--data` không tự động đổi target/features.

## Benchmark và ablation cho paper

### Benchmark shared mean với VAR-NN, LSTM, GRU, MLP-ARX, XGBoost, Ridge-ARX, SARIMAX và naïve

```bash
./.venv/bin/python3 run_pipeline_advanced.py \
  --pipeline shared-mean-benchmark \
  --dataset-profile us-unemployment \
  --data data/macro_us/processed/us_unemployment_awhman_2000_2025_09.csv \
  --tensorflow-device gpu \
  --paper-seeds 7 42 123 2024 2025
```

Pipeline này so sánh mười hai model sau:

| Vai trò | Pipeline artifact | Cơ chế lag/search |
| --- | --- | --- |
| No-VMD | `ardl_ffnn_no_vmd` | Chọn target lag trước, ARDL screen từng exogenous variable, rolling shared-mean lag screen, không VMD |
| Proposed hiện tại | `vmd_ardl_block_subset_proposed` | Target-first, sinh AIC variable blocks, fast-screen trên train-only inner rolling folds, full-grid và chọn reconstructed winner trên outer validation |
| Proposed cũ | `vmd_ardl_sequential_screen` | VMD, target-first, ARDL screen từng exogenous variable, rolling shared-mean lag screen |
| Joint Select | `vmd_ardl_joint_select` | VMD, `ardl_select_order` chọn target và exogenous lags đồng thời |
| VAR-NN | `varnn` | Chọn một cấu hình theo mean level validation RMSE qua reporting seeds |
| LSTM | `lstm` | Dùng target và bốn ngoại sinh đến tháng `t−1`; chọn chung lookback, số hidden units và learning rate theo mean level validation RMSE qua reporting seeds |
| GRU | `gru` | Cùng causal window/grid/mean-seed selection như LSTM nhưng dùng GRU recurrent cell |
| MLP-ARX | `mlp_arx` | MLPRegressor trên causal target/exogenous windows; chọn lookback, hidden layers và alpha qua mean validation RMSE |
| XGBoost | `xgboost` | Flatten causal multivariate window; chọn chung lookback, tree depth và learning rate bằng mean validation RMSE qua seeds |
| Ridge-ARX | `ridge_arx` | Ridge regression trên target/exogenous lags đến `t−1`; chọn lookback và alpha bằng validation RMSE |
| SARIMAX | `sarimax` | Grid `(p,d,q) × (P,D,Q,12)`, ngoại sinh trễ một tháng; chọn bằng level validation RMSE rồi rolling one-step trên test |
| Naïve | `naive_last_observation` | Dự báo bằng giá trị target tháng trước; không có seed |

Với mỗi FFNN signature, validation RMSE được tính ở mọi `--paper-seeds` rồi lấy
mean để xếp hạng. Chỉ một signature chung thắng; cùng lag/architecture/LR/alpha/
activation đó được refit riêng ở từng seed. Ba pipeline có VMD chọn bằng mean
**reconstructed-level** validation RMSE; no-VMD dùng mean **level** validation
RMSE. Vì vậy seed không phải hyperparameter và pipeline không chọn “best seed”.

LSTM dùng dữ liệu ở mức gốc, chuẩn hóa bằng train khi chọn cấu hình rồi bằng
train+validation khi refit cho test. Mỗi dự báo tháng `t` chỉ nhận cửa sổ dữ liệu
đến `t−1`. Grid mặc định gồm lookback 6/12 tháng, 16/32 hidden units và learning
rate `0.001`; cấu hình thắng được khóa chung cho mọi seed. Dự báo test là rolling
one-step, nên tại tháng `t` có thể dùng quan sát thật đã có ở các tháng trước.
Kết quả chi tiết nằm trong `lstm/lstm_shared_validation_search.csv`,
`lstm/lstm_shared_mean_seed_selection.csv` và `lstm/seed_<seed>/` của run mới.

GRU dùng cùng protocol với LSTM và lưu grid tại
`gru/gru_shared_validation_search.csv`, cấu hình chung tại
`gru/gru_shared_mean_seed_selection.csv`, cùng forecast/metrics trong
`gru/seed_<seed>/`. Chạy riêng GRU đủ 5 reporting seeds bằng:

```bash
./.venv/bin/python3 run_pipeline_advanced.py \
  --pipeline gru \
  --dataset-profile us-unemployment \
  --data data/macro_us/processed/us_unemployment_awhman_2000_2025_09.csv \
  --tensorflow-device gpu \
  --tensorflow-run-eagerly \
  --paper-seeds 7 42 123 2024 2025 \
  --run-id us-unemployment-gru-5seeds
```

XGBoost dùng các cửa sổ causal tương tự nhưng không scale dữ liệu. Grid mặc định
gồm lookback 6/12, depth 2/3, learning rate `0.03/0.1`, 200 cây, subsample và
column-sample `0.8`. Chạy riêng đủ 5 seed bằng:

```bash
./.venv/bin/python3 run_pipeline_advanced.py \
  --pipeline xgboost \
  --dataset-profile us-unemployment \
  --data data/macro_us/processed/us_unemployment_awhman_2000_2025_09.csv \
  --paper-seeds 7 42 123 2024 2025 \
  --run-id us-unemployment-xgboost-5seeds
```

Ridge-ARX là comparator deterministic, chuẩn hóa lag features chỉ trên tập fit
và chọn lookback 1/3/6/12 cùng alpha Ridge `0.01–100` bằng validation RMSE:

```bash
./.venv/bin/python3 run_pipeline_advanced.py \
  --pipeline ridge-arx \
  --dataset-profile us-unemployment \
  --data data/macro_us/processed/us_unemployment_awhman_2000_2025_09.csv \
  --run-id us-unemployment-ridge-arx-01
```

Chạy riêng MLP-ARX đủ 5 seed:

```bash
./.venv/bin/python3 run_pipeline_advanced.py \
  --pipeline mlp-arx \
  --dataset-profile us-unemployment \
  --data data/macro_us/processed/us_unemployment_awhman_2000_2025_09.csv \
  --paper-seeds 7 42 123 2024 2025 \
  --run-id us-unemployment-mlp-arx-5seeds
```

SARIMAX là comparator deterministic: ngoại sinh tháng `t` được dịch trễ để chỉ
dùng thông tin đến `t−1`; sau mỗi one-step forecast, actual target mới được cập
nhật vào state nhưng tham số không được fit lại. Candidate phải hội tụ trên train
và sau refit train+validation mới đủ điều kiện. Kết quả nằm trong
`sarimax/sarimax_validation_search.csv`, `sarimax/sarimax_selection.csv`,
`sarimax/forecasts.csv` và `sarimax/metrics.csv`. SARIMAX không nhân bản forecast
qua reporting seeds; cột seed để trống như naïve.

Có thể chạy riêng SARIMAX (khoảng vài chục giây trên panel hiện tại) mà không
chạy lại toàn bộ ARDL–NN/VAR-NN/LSTM/GRU/MLP-ARX/XGBoost/Ridge-ARX:

```bash
./.venv/bin/python3 run_pipeline_advanced.py \
  --pipeline sarimax \
  --dataset-profile us-unemployment \
  --data data/macro_us/processed/us_unemployment_awhman_2000_2025_09.csv \
  --run-id us-unemployment-sarimax-01
```

Riêng block-Proposed của pipeline này không dùng outer validation để giữ top 4
block specs. Mỗi spec được fit bằng lightweight model cố định `alpha=0.1`, LR
`0.001` trên 3 expanding folds nằm hoàn toàn trong train và được xếp bằng mean
component RMSE qua folds × reporting seeds. Outer validation chỉ bắt đầu được
dùng sau khi top 4 specs/component đã khóa, cho full FFNN grid và reconstructed
combination selection. `block-proposed-benchmark` cũ vẫn giữ outer-validation
fast-screen để bảo toàn protocol/artifact lịch sử của study đó.

`benchmark_metrics_comparison.csv` là kết quả chính: mean/std/median/count của
metric test tính riêng ở từng seed. `benchmark_ensemble_metrics_comparison.csv`
là kết quả phụ sau khi average prediction của các seed. DM test cũng dùng
ensemble-mean predictions trên các test dates chung. Pipeline này không tạo
`ablation_metrics.csv`.

### Study mới: Block-subset làm Proposed chính

Chạy pipeline riêng sau để không thay đổi schema/artifact của `benchmark` cũ:

```bash
./.venv/bin/python3 run_pipeline_advanced.py \
  --pipeline block-proposed-benchmark \
  --data /path/to/validated_legacy.csv \
  --paper-seeds 7 42 123 2024 2025
```

Trong study này, `vmd_ardl_block_subset_proposed` là comparator chính. ARDL khóa
target lag, AIC tạo một distributed-lag block cho mỗi exogenous variable, rồi
mọi non-empty block subset được fast-screen. Top 4 lag specs/component đi vào
full FFNN grid. Một tổ hợp chung được chọn bằng **mean reconstructed-level
validation RMSE qua reporting seeds**, sau đó mới refit từng seed và đánh giá
test.

Các stochastic main benchmarks cũng dùng shared selection:

- no-VMD FFNN/VARNN: mean level validation RMSE qua seeds;
- VMD FFNN: mean reconstructed-level validation RMSE qua seeds;
- seed không thuộc model signature và không có “best seed”;
- test không tham gia chọn shared winner.

Main benchmark còn giữ cả hai ARDL workflow gần nhất để comparison không thiên
lệch: `vmd_ardl_sequential_screen` chọn target trước rồi screen từng exogenous
order, còn `vmd_ardl_joint_select` dùng `ardl_select_order` chọn target và
exogenous lags đồng thời. Cả hai đều khóa một FFNN winner bằng mean
reconstructed-level validation RMSE qua reporting seeds.

Benchmark còn có B8 `vmd_ardl_hard_config_ffnn`. B8 giữ nguyên ARDL target-first
workflow, lấy top 3 distributed-lag orders/feature và grid toàn bộ tích Cartesian
(tối đa 81 specs/component với 4 features). Mỗi spec chỉ nhận một FFNN config
được quyết định từ train-only IMF frequency, spectral entropy, roughness và lag
load; không chạy architecture/LR/alpha/activation grid. Sau đó vẫn giữ top 3
candidates/component và chọn combination cuối bằng mean reconstructed-level
validation RMSE qua reporting seeds.

### Hard-config VMD–ARDL–FFNN hoạt động như thế nào?

Trong source, tên chính xác của comparator “VMD–ARDL–NN hard-config” là
`vmd_ardl_hard_config_ffnn`, vì neural model hiện được triển khai là FFNN. Nó
không chọn architecture/LR/alpha/activation bằng validation grid. Quy trình là:

1. Chỉ dùng train split để stationarity-transform và VMD target thành các
   component `VMD1...VMDK` cùng `RES`.
2. ARDL SelectOrder khóa target lag; ARDL/AIC lấy top 3 distributed-lag orders
   cho từng exogenous variable. Full Cartesian product tạo tối đa
   `3^number_of_features` lag specs cho mỗi component.
3. Mỗi target component được mô tả bằng FFT train-only: dominant frequency,
   spectral centroid, bandwidth, normalized spectral entropy, roughness và
   energy ratio.
4. Với từng lag spec, policy tính input dimension `D`, maximum lag và số mẫu
   train hiệu dụng `N_eff`, rồi gán đúng **một** FFNN configuration.
5. Configuration cố định đó được fit ở mọi reporting seed. Validation chỉ xếp
   hạng lag/component candidates và reconstructed combinations; nó không tune
   hard policy.
6. Top 3 candidates/component được kết hợp; một shared winner được chọn bằng
   mean reconstructed-level validation RMSE qua seeds, sau đó refit riêng từng
   seed trên train + validation để dự báo test.

Regime được xét theo thứ tự sau:

- `irregular`: component là `RES`, hoặc entropy `>= 0.70`, hoặc đồng thời
  roughness `>= 1.50` và entropy `>= 0.65`;
- `low`: spectral centroid `< 0.125`;
- `mid`: spectral centroid `< 0.25`;
- `high_structured`: các trường hợp còn lại.

Hard policy mặc định:

| IMF regime | Architecture theo input dimension `D` | Activation | LR | Alpha |
| --- | --- | --- | ---: | ---: |
| Low frequency | `(0.5D,)` | tanh | `0.01` | `0.001` |
| Mid frequency | `(D, 0.5D)` | tanh | `0.001` | `0.01` |
| High structured | `(0.5D, 0.25D)` | tanh | `0.001` | `0.01` |
| Irregular/RES | `(0.25D,)` | relu | `0.001` | `0.1` |

Width bị chặn trong `[4,32]`. Khi `D/N_eff > 0.20`, width giảm còn 75% và
alpha tăng một mức trong `(0.0001, 0.001, 0.01, 0.1)`. Mọi width đều dùng
`ceil` trước khi chặn. Maximum lag không đổi regime trực tiếp, nhưng làm giảm
`N_eff`; số target/exogenous lag quyết định `D`. Vì vậy lag/window load vẫn tác
động vào architecture và regularization qua `D/N_eff`.

Nơi định nghĩa hard-config:

| File | Phần được định nghĩa |
| --- | --- |
| `vmd_ardl_ffnn/src/vmd_ardl_ffnn/config.py` | `HardFFNNPolicyConfig`: thresholds, width bounds, heavy-load ratio, alpha levels và số candidate giữ lại |
| `vmd_ardl_ffnn/src/vmd_ardl_ffnn/hard_ffnn_policy.py` | `describe_target_imfs()`, bốn regime, bảng architecture/LR/alpha/activation và heavy-load adjustment |
| `vmd_ardl_ffnn/src/vmd_ardl_ffnn/hard_ffnn_workflow.py` | `run_hard_ffnn_comparator()`: ARDL full-product search, fit theo seed, reconstructed selection, refit và ghi artifact |
| `vmd_ardl_ffnn/src/vmd_ardl_ffnn/hard_ffnn_experiments.py` | Adapter dùng chung để đưa hard-config comparator vào hai benchmark studies |
| `advanced_config.py` và `workflow_dispatch.py` | Chọn profile khi gọi `--pipeline vmd-ardl-hard-ffnn` và truyền reporting seeds |

`ExperimentConfig` dùng `HardFFNNPolicyConfig()` qua `default_factory`, nên các
giá trị mặc định nằm trong `config.py`, không nằm trong
`build_experiment_config()`. Muốn thay policy cho experiment riêng, truyền
`hard_ffnn=HardFFNNPolicyConfig(...)` khi tạo `ExperimentConfig`; không sửa
artifact CSV sau khi chạy.

Ví dụ:

```python
from vmd_ardl_ffnn.config import ExperimentConfig, HardFFNNPolicyConfig

config = ExperimentConfig(
    # data=..., vmd=..., ardl=..., ffnn=...
    hard_ffnn=HardFFNNPolicyConfig(
        low_frequency_max=0.10,
        mid_frequency_max=0.30,
        heavy_input_ratio=0.15,
        min_width=4,
        max_width=24,
        top_component_candidates=3,
    ),
)
```

CLI hiện chưa expose từng hard-policy threshold. Nếu muốn đổi profile mặc định
của `run_pipeline_advanced.py`, hãy import `HardFFNNPolicyConfig` và truyền
field `hard_ffnn=` ngay trong `build_experiment_config()`; logic phân regime
vẫn giữ ở `hard_ffnn_policy.py`.

Các artifact audit chính:

- `hard_ffnn_imf_descriptors.csv`: đặc tính frequency/entropy/roughness train-only;
- `hard_ffnn_policy_config.csv`: thresholds và policy version;
- `hard_ffnn_policy_assignments.csv`: lag spec → regime → FFNN config;
- `hard_ffnn_component_candidates.csv` và
  `hard_ffnn_component_shortlist.csv`: validation ranking theo component;
- `hard_ffnn_reconstructed_seed_scores.csv` và
  `hard_ffnn_shared_selection.csv`: reconstructed mean-seed selection.

Ablation mới dùng schema `block_subset_proposed_v1`:

| ID | Thay đổi chính | Selection |
| --- | --- | --- |
| A0 | No VMD, fixed lags | Shared level mean-seed |
| A1 | VMD, fixed lags | Shared reconstructed mean-seed |
| A2 | ARDL target lag, fixed exog lag | Shared reconstructed mean-seed |
| A3 | AIC blocks, luôn giữ đủ exog blocks | Shared reconstructed mean-seed |
| A4 | Full block subsets, component-wise objective | Shared component mean-seed |
| A5 | Full block subsets, full grid | Shared reconstructed mean-seed |
| A6 | Full block subsets, full grid | Reconstructed winner riêng từng seed |
| A7 | Top 4 block specs sau lightweight screen | Shared reconstructed mean-seed; Proposed |

Pipeline này cố ý chạy cả A5 exhaustive, A6 per-seed và A7 two-stage nên rất
nặng. Nên smoke bằng `--paper-seeds 7 --skip-varnn` trước. Final FFNN grid riêng
cho study này gồm learning rate `0.01` và `0.001`; A7 fast screen cố định
`alpha=0.1`, learning rate `0.001`.

Protocol đầy đủ và các comparison được phép diễn giải nằm trong
`Experimental_Plan_BlockSubset_Proposed.md`.

### Benchmark models

| ID | Pipeline | Mục đích |
| --- | --- | --- |
| B0a | Last-observation naïve | Reference tối thiểu |
| B0b | Seasonal naïve lag 12 | Reference theo mùa tháng |
| B1 | Standalone ARDL | Linear dynamic-regression baseline |
| B2 | FFNN fixed lag, no VMD | Neural baseline cơ bản |
| B3 | ARDL–FFNN, no VMD | Đo đóng góp của VMD |
| B4 | VMD + FFNN fixed lag | Đo VMD khi chưa adaptive lag selection |
| B5 | VARNN | Multivariate neural benchmark |
| B6 | VMD + AIC variable-block subsets + FFNN | Full-grid hoặc two-stage block-subset comparator; winner dùng mean reconstructed level validation RMSE qua seeds |
| B7 | VMD + ARDL top-3 full-product + hard-config FFNN | Grid lag combinations; FFNN config được gán train-only theo IMF/lag load |
| P | Proposed sequential VMD–ARDL–FFNN | Mô hình đầy đủ |

B6 chỉ là benchmark comparator, không được gán thêm một mã ablation và không làm
thay đổi A4. Forecast level của B6 được đưa vào cùng bảng metrics và mọi cặp DM/Holm.

B7 dùng cùng hard-config protocol với B8 của `block-proposed-benchmark`: top-3
exogenous orders/feature, full Cartesian lag grid, shared reconstructed-level
mean-seed selection và không grid FFNN hyperparameters. Nó là benchmark comparator,
không phải ablation mới và không thay Proposed A4 của study gốc.

### Ablation A0–A5

| ID | VMD | Target lag | Exogenous lag |
| --- | --- | --- | --- |
| A0 | Không | Fixed `(1,12)` | Fixed `(1)` |
| A1 | Có | Fixed `(1,12)` | Fixed `(1)` |
| A2 | Có | ARDL SelectOrder | Fixed `(1)` |
| A3 | Có | Fixed `(1,12)` | ARDL screening |
| A4 | Có | ARDL SelectOrder | ARDL screening; proposed |
| A5 | Có | Joint SelectOrder | Joint SelectOrder |

Benchmark mặc định chạy 5 reporting seeds:

```bash
./.venv/bin/python3 run_pipeline_advanced.py \
  --pipeline benchmark \
  --data /path/to/validated_legacy.csv \
  --paper-seeds 7 42 123 2024 2025
```

Nếu không truyền `--paper-seeds`, đúng 5 seed trên vẫn được dùng. Reporting seed
không được chọn như một hyperparameter: project xuất từng seed và tổng hợp
mean/std/median/count.

Mỗi benchmark study báo hai estimand riêng:

- `benchmark_metrics_comparison.csv`: bảng chính, tính metric riêng cho từng
  seed rồi tổng hợp mean/std/median/count. Bảng này đo expected single-model
  performance và độ ổn định giữa các random initializations.
- `benchmark_seed_robustness_metrics.csv`: explicit compatibility alias của
  cùng bảng mean/std/median/count phía trên.
- `benchmark_ensemble_metrics_comparison.csv`: bảng phụ, trung bình prediction
  qua seeds tại từng ngày trước rồi mới tính một bộ RMSE/MAE/MAPE/DA.
  DM/Holm test tiếp tục dùng chuỗi dự báo ensemble này và ghi rõ estimand.

Để giảm runtime riêng cho B6 mà không bỏ block spec trước fast screen:

```bash
./.venv/bin/python3 run_pipeline_advanced.py \
  --pipeline benchmark \
  --data /path/to/validated_legacy.csv \
  --paper-seeds 7 42 123 2024 2025 \
  --block-subset-search-strategy two-stage
```

VARNN trong benchmark giữ 32 hidden units và sigmoid, chỉ validation hai learning
rate `0.01`, `0.001`. File `varnn_validation_search.csv` lưu RMSE từng candidate.

Nếu môi trường không có TensorFlow hoặc chủ động loại B5:

```bash
./.venv/bin/python3 run_pipeline_advanced.py \
  --pipeline benchmark \
  --data /path/to/validated_legacy.csv \
  --skip-varnn
```

Không được mô tả kết quả là benchmark đầy đủ B0–B5 nếu VARNN bị skip. Kiểm tra
`skipped_models.csv` trước khi đưa bảng vào paper.

### Statistical validation

Benchmark thực hiện:

- Diebold–Mariano test trên common test dates, squared loss, horizon 1.
- Ensemble-mean prediction qua reporting seeds trước khi so sánh model.
- Mean/std metric qua seed là kết quả báo cáo chính trong
  `benchmark_metrics_comparison.csv`; ensemble-mean prediction dùng bởi DM
  được lưu riêng trong `benchmark_ensemble_metrics_comparison.csv`.
- Holm correction cho nhiều pairwise comparisons.
- Moving-block bootstrap 95% CI cho paired mean loss difference.
- Kiểm tra mọi benchmark có cùng validation/test dates và cùng actual values.

`paper_forecast_date_coverage.csv` phải có `same_dates_for_all_models=True` trước
khi sử dụng kết quả DM hoặc tuyên bố model tốt hơn.

## Sensitivity pipeline

Sensitivity chỉ chạy proposed model và thay đổi một yếu tố tại một thời điểm:

- VMD modes `K=2` và `K=4`;
- VMD alpha bằng `0.5×` và `2×` base;
- compact lag bounds;
- AIC sang BIC;
- staged sang feasible full-product search;
- split 65/20/15 và 75/10/15.

Smoke run trước bằng một seed:

```bash
./.venv/bin/python3 run_pipeline_advanced.py \
  --pipeline sensitivity \
  --data /path/to/validated_legacy.csv \
  --paper-seeds 7
```

Full robustness study:

```bash
./.venv/bin/python3 run_pipeline_advanced.py \
  --pipeline sensitivity \
  --data /path/to/validated_legacy.csv \
  --paper-seeds 7 42 123 2024 2025
```

Không chọn sensitivity setting thắng bằng test RMSE. Pipeline này dùng để báo cáo
robustness, không phải để tune cấu hình chính. Fixed-length rolling window chưa
được triển khai.

## Cấu trúc thư mục và vai trò từng file

```text
Project/
├── .python-version                 # Python version dùng bởi pyenv
├── .venv/                          # Virtual environment cục bộ; không commit
├── requirements.txt                # Dependency list tiện dụng ở repository root
├── df_final.csv                    # Dataset cũ bị loại; không dùng để train/evaluate
├── cpi_forecast_selected_variables.csv
│                                     # Dataset thay thế; cần config cột phù hợp
├── run_pipeline_advanced.py        # Entrypoint run-scoped được khuyến nghị
├── Experimental_Plan_VMD_ARDL_FFNN.md
│                                     # Protocol paper của benchmark gốc
├── Experimental_Plan_BlockSubset_Proposed.md
│                                     # Protocol paper khi A7 block-subset là Proposed
├── README.md                       # Tài liệu workflow hiện tại
├── ardl_ffnn_model.py              # Script thử nghiệm/legacy, không phải runner chính
├── compare_models/                 # Notebook/script so sánh cũ, ngoài paper runner
├── results/                        # Modern run-scoped outputs: <pipeline>/<run_id>/
├── results_export_vmd/             # Output legacy
├── results_vmd/                    # Output legacy
├── results_vmd-ardl-joint-select/  # Output legacy
├── results_vmd-ardl-select-order/  # Output legacy
└── vmd_ardl_ffnn/
    ├── pyproject.toml              # Package metadata, dependencies và pytest config
    ├── docs/
    │   ├── README.md               # Package quick-start
    │   ├── CLI_OPTIONS.md          # Tùy chọn package CLI
    │   ├── PROJECT_REVIEW.md       # Review kỹ thuật lịch sử
    │   └── plan.md                 # Plan/refactor note lịch sử
    ├── data/raw/                   # Vị trí dự phòng cho raw package data
    ├── results/                    # Output legacy trước run-scoped entrypoint
    ├── tests/
    │   ├── test_refactor.py        # Unit/regression tests
    │   └── test_smoke.py           # End-to-end và TensorFlow smoke tests
    ├── build/                      # Generated build output; không sửa trực tiếp
    └── src/
        ├── vmd_ardl_ffnn_reproduction.egg-info/
        │                              # Generated editable-install metadata
        └── vmd_ardl_ffnn/
            ├── __init__.py
            ├── cli.py
            ├── legacy_cli.py
            ├── advanced_arguments.py
            ├── advanced_config.py
            ├── advanced_runner.py
            ├── workflow_dispatch.py
            ├── dataset_profiles.py
            ├── benchmark_common.py
            ├── config.py
            ├── data.py
            ├── stationarity.py
            ├── features.py
            ├── lag_selection.py
            ├── decomposition.py
            ├── experiment.py
            ├── experiment_fixed.py
            ├── block_subset_workflow.py
            ├── shared_mean_workflow.py
            ├── hard_ffnn_workflow.py
            ├── block_proposed_experiments.py
            ├── shared_mean_benchmark_experiments.py
            ├── paper_experiments.py
            ├── hard_ffnn_experiments.py
            ├── sensitivity_experiments.py
            ├── hard_ffnn_policy.py
            ├── paper_baselines.py
            ├── paper_reporting.py
            ├── statistical_tests.py
            ├── metrics.py
            ├── evaluation.py
            ├── diagnostics.py
            ├── artifacts.py
            └── models/
                ├── __init__.py
                ├── ffnn.py
                └── varnn.py
```

Các thư mục `__pycache__/`, `.pytest_cache/`, `build/` và `*.egg-info/` là
generated state, không phải source of truth. Source cần sửa luôn nằm trong
`vmd_ardl_ffnn/src/vmd_ardl_ffnn/`.

### Các file experiment và orchestration

| File | Pipeline/chức năng |
| --- | --- |
| `experiment.py` | Public facade ổn định; chỉ re-export `VMDARDLFFNNExperiment` và `config_for_columns` từ core implementation |
| `experiment_fixed.py` | Lõi split/transform/VMD, ARDL lag selection, rolling screen, FFNN fit/refit, reconstruction và các facade benchmark. Tên `fixed` là lịch sử |
| `block_subset_workflow.py`, `shared_mean_workflow.py`, `hard_ffnn_workflow.py` | Các phương thức thực thi ba nhánh benchmark, kế thừa trong experiment core |
| `benchmark_common.py` | Hợp đồng chung cho metrics qua seed, date coverage, audit và chuẩn hóa forecast |
| `paper_experiments.py` | Orchestrator cho `--pipeline benchmark`: benchmark gốc, A0–A5, baselines/VARNN/hard-config/block comparator, primary mean/std qua seed, ensemble table phụ, efficiency, DM/Holm và paper artifacts |
| `block_proposed_experiments.py` | Orchestrator cho `--pipeline block-proposed-benchmark`: đưa A7 two-stage block-subset thành Proposed, chạy A0–A7 cùng sequential/joint/hard-config/VARNN comparators |
| `shared_mean_benchmark_experiments.py` | Orchestrator cho `--pipeline shared-mean-benchmark`: bốn FFNN workflow shared mean, VAR-NN, LSTM và naïve trên cùng ngày dự báo |
| `lstm_shared_mean.py`, `models/lstm.py` | Chọn cấu hình LSTM theo mean validation RMSE qua seed, refit và dự báo rolling one-step |
| `hard_ffnn_experiments.py` | Adapter dùng chung cho hard-config comparator trong hai benchmark; ép ARDL target-first + top-3 full-product, gọi core engine và chuẩn hóa metrics/forecasts. Policy không được định nghĩa ở file này |
| `sensitivity_experiments.py` | One-factor-at-a-time robustness study của Proposed: VMD modes/alpha, lag bounds, AIC/BIC, feasible full-product và split ratios; không dùng test để chọn profile thắng |

Luồng gọi chính:

```text
run_pipeline_advanced.py
→ advanced_arguments.py + advanced_runner.py + workflow_dispatch.py
→ VMDARDLFFNNExperiment trong experiment.py
→ lõi trong experiment_fixed.py; block/shared-mean/hard methods trong các workflow module
→ paper/block/shared-mean/hard/sensitivity orchestrator tùy --pipeline
→ artifacts trong results/<pipeline>/<run_id>/
```

### Các source module còn lại

| File | Trách nhiệm |
| --- | --- |
| `__init__.py` | Public package exports |
| `config.py` | Dataclass cấu hình và validation cho data/VMD/ARDL/FFNN, diagnostics, stationarity và `HardFFNNPolicyConfig` |
| `data.py` | Đọc CSV, calendar validation, missing-value và log checks |
| `stationarity.py` | ADF/KPSS train-only và transform `level`/`diff1` |
| `features.py` | Đại diện `LagSpec` target/exogenous lags |
| `lag_selection.py` | Target SelectOrder, joint selection, exogenous ARDL screening, variable-block subsets và lag-spec generation |
| `decomposition.py` | Self-contained VMD và optional library backend |
| `models/ffnn.py` | NumPy FFNN, scaler và Adam optimizer |
| `models/varnn.py` | VAR lag selection, TensorFlow VARNN, LR validation và refit |
| `hard_ffnn_policy.py` | Train-only IMF descriptors và deterministic mapping từ frequency/entropy/roughness + lag load sang architecture/LR/alpha/activation |
| `metrics.py` | RMSE, MAE, MAPE và Directional Accuracy |
| `evaluation.py` | Chuẩn hóa forecast/scale và bảng so sánh model |
| `diagnostics.py` | Series statistics, Granger và ARDL residual diagnostics |
| `artifacts.py` | Run ID, manifest, config hash và ghi CSV run-scoped |
| `paper_baselines.py` | Naïve và standalone ARDL baselines |
| `paper_reporting.py` | Lag frequency tables và các figure cho paper |
| `statistical_tests.py` | DM, Holm correction và block-bootstrap CI |
| `cli.py` | CLI tùy biến chi tiết; không phải entrypoint run-scoped ưu tiên |

Không import trực tiếp `experiment_fixed.py` từ code bên ngoài. Dùng:

```python
from vmd_ardl_ffnn.experiment import VMDARDLFFNNExperiment
```

## Cấu trúc output

### Single VMD run

Các file chính:

- `final_reconstructed_metrics_cached_vmd.csv`
- `final_reconstructed_forecasts_cached_vmd.csv`
- `best_component_models_cached_vmd.csv`
- `ffnn_component_validation_search_cached_vmd.csv`
- `ffnn_validation_search_cached_vmd.csv`
- `ffnn_lag_screen_folds_cached_vmd.csv`
- `ffnn_lag_screen_stability_cached_vmd.csv`
- `ardl_selected_orders_train_only.csv`
- `component_predictions_cached_vmd.csv`
- `input_audit_cached_vmd.csv`
- `runtime_breakdown.csv`
- diagnostics CSV và actual-vs-predicted PNG.

### No-VMD run

Các file chính:

- `final_metrics_fixed_no_vmd.csv`
- `final_forecasts_fixed_no_vmd.csv`
- `best_model_fixed_no_vmd.csv`
- `ffnn_validation_search_fixed_no_vmd.csv`
- `ffnn_lag_screen_folds_no_vmd.csv`
- `ffnn_lag_screen_stability_no_vmd.csv`
- `ardl_selected_orders_train_only_no_vmd.csv`
- `input_audit_no_vmd.csv`
- `runtime_breakdown.csv`.

### Benchmark run

```text
results/benchmark/<run_id>/
├── manifest.json
├── paper_experiment_settings.csv
├── paper_dataset_summary.csv
├── paper_split_summary.csv
├── paper_forecast_date_coverage.csv
├── paper_seed_metrics.csv
├── paper_metrics_summary.csv
├── paper_ensemble_metrics_summary.csv
├── paper_forecasts.csv
├── benchmark_metrics_comparison.csv          # Primary mean/std qua seeds
├── benchmark_ensemble_metrics_comparison.csv # Secondary ensemble metrics
├── benchmark_seed_robustness_metrics.csv      # Explicit primary-table alias
├── benchmark_forecasts_comparison.csv
├── ablation_metrics.csv
├── ablation_forecasts.csv
├── diebold_mariano_tests.csv
├── lag_selection_summary.csv
├── lag_selection_frequency.csv
├── lag_screen_fold_metrics.csv
├── lag_screen_stability.csv
├── efficiency_summary.csv
├── paper_*.png
├── <neural_pipeline>/seed_<seed>/
├── vmd_ardl_block_subset_mean_seed/
│   ├── ardl_variable_blocks_train_only.csv
│   ├── ffnn_block_subset_component_search.csv
│   ├── ffnn_block_subset_fast_screen_metrics.csv       # two-stage only
│   ├── ffnn_block_subset_fast_screen_summary.csv       # two-stage only
│   ├── ffnn_block_subset_component_mean_seed_ranking.csv
│   ├── ffnn_block_subset_reconstructed_seed_scores.csv
│   ├── ffnn_block_subset_mean_seed_selection.csv
│   ├── best_component_models_block_subset_mean_seed.csv
│   ├── component_predictions_block_subset_mean_seed.csv
│   ├── final_reconstructed_forecasts_block_subset_mean_seed.csv
│   └── final_reconstructed_metrics_block_subset_mean_seed.csv
├── varnn/seed_<seed>/
└── <deterministic_baseline>/
```

`shared-mean-benchmark` dùng cùng nhóm paper/benchmark artifact phía trên, có
thêm thư mục `varnn/`, `lstm/`, `gru/`, `xgboost/`, `ridge_arx/`, `sarimax/` và
`naive_last_observation/`. Nó không sinh
`ablation_metrics.csv` hoặc `ablation_forecasts.csv`.

Trong các FFNN validation tables, `learning_rate_init` là learning rate của
candidate. Trong mỗi VARNN seed directory, `varnn_validation_search.csv` lưu
các candidate thuộc grid của profile; bảng `varnn/shared_selection.csv` lưu
cấu hình thắng theo trung bình validation RMSE qua reporting seeds. LSTM lưu
grid và bảng chọn chung tại `lstm/lstm_shared_validation_search.csv` và
`lstm/lstm_shared_mean_seed_selection.csv`. SARIMAX lưu grid và trạng thái hội
tụ tại `sarimax/sarimax_validation_search.csv`; cấu hình deterministic thắng ở
`sarimax/sarimax_selection.csv`.

Các thư mục `results_vmd/`, `results_vmd-ardl-select-order/`,
`results_vmd-ardl-joint-select/`, `vmd_ardl_ffnn/results*` là artifact legacy.
Không trộn chúng với run-scoped output mới khi làm bảng cho paper.

## Cài đặt môi trường

Từ project root:

```bash
python3 -m venv .venv
./.venv/bin/python3 -m pip install --upgrade pip

# Cài core + tests + VMD library optional + TensorFlow
./.venv/bin/python3 -m pip install -e "vmd_ardl_ffnn[all]"
```

Project mặc định dùng self-contained VMD nên không bắt buộc `vmdpy`. TensorFlow
cần cho VARNN và LSTM. `requirements.txt` ở root là danh sách tối giản/legacy;
`vmd_ardl_ffnn/pyproject.toml` mới là nguồn dependency đầy đủ.

Kiểm tra TensorFlow trong đúng venv:

```bash
./.venv/bin/python3 -c \
  'import tensorflow as tf; print(tf.__version__)'
```

Thông báo không tìm thấy CUDA chỉ có nghĩa TensorFlow chạy CPU, không phải lỗi.

## Chạy test

```bash
cd /home/cloud/cloud/DA/Project/vmd_ardl_ffnn
../.venv/bin/python3 -m pytest -q
```

TensorFlow tests chỉ chạy khi interpreter hiện tại import được TensorFlow.

## Những điều cần lưu ý

1. **Dùng đúng entrypoint.** `run_pipeline_advanced.py` tạo run directory và
   manifest. Package CLI ghi trực tiếp vào `--out`, phù hợp thử nghiệm nhưng dễ
   ghi đè artifact nếu tái sử dụng cùng thư mục.
2. **Không dùng kết quả legacy làm paper result mới.** Luôn lấy một run có
   `manifest.json` với `status="success"`.
3. **Không chọn model bằng test.** Test chỉ dùng sau khi lag và hyperparameters
   đã khóa bằng train/validation.
4. **Paper seeds không phải grid để chọn winner.** Benchmark báo cáo toàn bộ seed
   và aggregate; không lấy seed có test RMSE nhỏ nhất.
5. **Learning-rate và rolling lag-screen làm tăng runtime.** Mỗi lag spec được
   fast-screen trên nhiều train-only folds với hai learning rates trước khi full
   FFNN grid chạy. VARNN cũng fit hai validation candidates cho mỗi seed.
6. **Full benchmark rất nặng.** Nó chạy deterministic baselines, 7 neural
   variants qua 5 seeds, B6, B7 hard-config và VARNN qua 5 seeds. B6 `full-grid`
   fit cả 15 block subsets/component; `two-stage` fast-screen đủ 15 nhưng chỉ
   full-grid top 4. B7 có tối đa 81 lag specs/IMF và một hard FFNN config/spec.
   Nên smoke bằng một seed trước.
   Pipeline `block-proposed-benchmark` còn nặng hơn vì chạy đồng thời A5
   shared full-grid, A6 per-seed full-grid và A7 two-stage với hai final learning
   rates.
7. **Sensitivity còn nặng hơn.** Chạy nhiều proposed configurations; không chạy
   chung để tìm setting tốt nhất theo test.
8. **VMD rolling cache là phần tốn thời gian.** Không thay bằng decomposition trên
   toàn validation/test vì cách đó tạo look-ahead leakage.
9. **Exogenous lag 0 bị cấm.** Giao thức hiện giả định không biết exogenous value
   cùng timestamp tại forecast origin.
10. **Scale `level` mới là scale so sánh chính.** MAPE trên differenced/component
    scale có thể không ổn định hoặc không có ý nghĩa kinh tế.
11. **Directional Accuracy dùng previous actual.** Nó đo đúng chiều tăng/giảm,
    không phải dấu của forecast.
12. **Log transform yêu cầu dữ liệu dương.** Pipeline sẽ dừng nếu bất kỳ model
    column nào bằng 0 hoặc âm.
13. **Thiếu kỳ/NaN mặc định gây lỗi.** Xử lý dữ liệu có chủ đích trước khi chạy;
    không im lặng drop hoặc nội suy rồi quên ghi vào methodology.
14. **`--run-id` phải duy nhất khi tạo run.** Chỉ nhánh sequential macro được mở
   lại bằng `--resume --run-id ...`; runner từ chối nếu data/config khác manifest.
15. **VARNN chạy CPU vẫn hợp lệ.** Chỉ chậm hơn GPU; không dùng `--skip-varnn` nếu
    muốn benchmark đầy đủ B0–B5.
16. **Fixed-length rolling window chưa có.** Chỉ được tuyên bố kết quả cho
    expanding-history rolling one-step hiện tại.

## Tài liệu liên quan

- `Experimental_Plan_VMD_ARDL_FFNN.md`: protocol, benchmark, ablation, statistical
  tests và checklist trước khi viết paper.
- `Experimental_Plan_BlockSubset_Proposed.md`: study riêng lấy block-subset
  two-stage làm Proposed và chuẩn hóa shared mean-seed selection.
- `vmd_ardl_ffnn/docs/CLI_OPTIONS.md`: toàn bộ option của package CLI chi tiết.
- `vmd_ardl_ffnn/docs/PROJECT_REVIEW.md`: review/refactor notes lịch sử; README
  này và code hiện tại được ưu tiên nếu có điểm không còn đồng bộ.
