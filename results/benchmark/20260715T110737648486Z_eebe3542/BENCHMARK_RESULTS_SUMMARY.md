# Benchmark Results Summary

## 1. Thông tin run

| Thuộc tính | Giá trị |
|---|---|
| Run ID | `20260715T110737648486Z_eebe3542` |
| Pipeline | `benchmark` |
| Trạng thái | `success` |
| Dataset | `df_final.csv` |
| Target | `Export_US` |
| Exogenous variables | `PCE`, `US_Retail`, `USD_VND`, `Import_CN` |
| Số quan sát | 219 tháng |
| Khoảng dữ liệu | 01/2008–03/2026 |
| Forecast protocol | `rolling_one_step` |
| Paper seeds | `7`, `42`, `123`, `2024`, `2025` |
| VARNN | Đã chạy bằng TensorFlow 2.21.0 |
| Comparator mới | `vmd_ardl_block_subset_mean_seed` |

Manifest xác nhận run hoàn tất thành công, không có model bị skip. Run này gồm
đầy đủ chín benchmark models, A0–A5 ablations, comparator AIC variable-block mới,
seed aggregation, Diebold–Mariano test và Holm correction.

## 2. Dữ liệu và cấu hình thí nghiệm

| Nhóm | Cấu hình |
|---|---|
| Split | 70% train, 15% validation, 15% test |
| Transform | Log tự nhiên; stationarity decision chỉ dùng train |
| VMD | 3 modes + residual, `alpha=2000`, self-contained backend |
| ARDL | AIC, target lag tối đa 12, exogenous order tối đa 6 |
| Proposed lag generation | Staged, tối đa 16 specs/component, top 3 exogenous orders |
| Proposed rolling screen | 3 expanding folds × 12 tháng, LR `0.01` và `0.001` |
| Proposed lag shortlist | Top 4, luôn giữ raw winner và one-SE winner |
| FFNN architecture | 1–3 hidden layers, width theo input dimension |
| FFNN alpha | `1e-4`, `1e-3`, `1e-2`, `1e-1` |
| FFNN learning rate | `0.01`, `0.001` |
| Comparator mới | 15 non-empty variable-block subsets/component, full FFNN grid |

### Outer split

| Split | Số quan sát | Bắt đầu | Kết thúc |
|---|---:|---|---|
| Train | 153 | 01/2008 | 09/2020 |
| Validation | 33 | 10/2020 | 06/2023 |
| Test | 33 | 07/2023 | 03/2026 |

Mọi benchmark dùng cùng 33 validation dates và 33 test dates. Test chỉ được dùng
sau khi lag structure và hyperparameters đã khóa.

## 3. Benchmark test results

Neural models được báo cáo bằng trung bình ± độ lệch chuẩn qua năm paper seeds.
Mọi metric trong bảng là metric sau khi reconstruct và đưa về level scale.

| Hạng RMSE | Model | Test RMSE | Test MAE | MAPE | DA |
|---:|---|---:|---:|---:|---:|
| 1 | ARDL–FFNN không VMD | **1,049,951 ± 57,929** | 806,738 ± 26,752 | **7.392%** | 65.00% |
| 2 | VMD–AIC block-subset–FFNN | **1,065,643 ± 8,021** | **779,867 ± 7,512** | 7.507% | 67.50% |
| 3 | Standalone ARDL | 1,084,833 | 808,376 | 7.535% | 68.75% |
| 4 | Proposed VMD–ARDL–FFNN | 1,124,266 ± 52,792 | 812,535 ± 63,097 | 7.862% | 63.75% |
| 5 | VMD–FFNN fixed lag | 1,249,420 ± 135,461 | 876,765 ± 102,446 | 8.601% | **70.00%** |
| 6 | FFNN fixed lag, không VMD | 1,290,430 ± 153,982 | 911,696 ± 105,259 | 8.868% | 65.63% |
| 7 | VARNN | 1,488,365 ± 168,178 | 1,127,312 ± 183,953 | 10.610% | — |
| 8 | Naive last observation | 1,611,731 | 1,092,537 | 10.436% | 0.00% |
| 9 | Naive seasonal-12 | 2,383,548 | 2,101,624 | 18.093% | 53.13% |

### Nhận xét test

- ARDL–FFNN không VMD vẫn đứng đầu theo RMSE và MAPE.
- Comparator mới đứng thứ hai theo RMSE nhưng **đứng đầu theo MAE**.
- Comparator mới có RMSE cao hơn ARDL–FFNN không VMD 1.49%, nhưng MAE thấp hơn
  3.33% và DA cao hơn 2.5 điểm phần trăm.
- Comparator mới tốt hơn Proposed 5.21% RMSE, 4.02% MAE, 4.52% MAPE và cao hơn
  3.75 điểm phần trăm DA.
- Comparator mới cũng tốt hơn Standalone ARDL 1.77% RMSE và 3.53% MAE.
- Độ lệch chuẩn test RMSE của comparator chỉ 8,021, thấp hơn rõ rệt các neural
  benchmarks khác. Đây là tính ổn định quan sát được dưới một cấu trúc chung qua
  seeds; chưa đủ để quy toàn bộ nguyên nhân cho mean-seed selection.

## 4. Benchmark validation results

| Hạng RMSE | Model | Validation RMSE | Validation MAE | MAPE | DA |
|---:|---|---:|---:|---:|---:|
| 1 | ARDL–FFNN không VMD | **980,037 ± 19,852** | 755,771 | 9.664% | 61.25% |
| 2 | Proposed VMD–ARDL–FFNN | 989,220 ± 36,770 | 743,408 | 9.604% | 67.50% |
| 3 | VMD–AIC block-subset–FFNN | 991,909 ± 4,513 | 709,384 | 9.166% | 73.13% |
| 4 | FFNN fixed lag, không VMD | 1,007,349 ± 26,923 | 715,439 | 9.194% | 72.50% |
| 5 | VMD–FFNN fixed lag | 1,023,984 ± 29,774 | **707,242** | **9.143%** | **75.63%** |
| 6 | VARNN | 1,036,693 ± 28,273 | 718,295 | 9.266% | — |
| 7 | Naive last observation | 1,099,421 | 782,292 | 10.016% | 0.00% |
| 8 | Standalone ARDL | 1,209,736 | 917,699 | 11.740% | 65.63% |
| 9 | Naive seasonal-12 | 1,944,329 | 1,722,014 | 21.090% | 53.13% |

Comparator mới đứng thứ ba theo validation RMSE, nhưng đứng thứ hai theo
validation MAE, MAPE và DA. Chênh lệch validation RMSE giữa comparator và
Proposed chỉ khoảng 2,689, trong khi comparator tổng quát hóa tốt hơn trên test.

## 5. Kết quả workflow AIC variable-block mới

Workflow mới thực hiện riêng cho `VMD1`, `VMD2`, `VMD3` và `RES`:

1. `ardl_select_order` chọn target lags trên train.
2. Khóa target lags.
3. Với từng exogenous variable, fit ARDL cho order `q=1..6` và chọn `q` có AIC
   nhỏ nhất.
4. Tạo block lag `1..q` cho mỗi biến.
5. Sinh 15 tập con block khác rỗng.
6. Cross 15 specs với full FFNN grid và năm seeds.
7. Chọn một lag-spec/hyperparameter chung cho mỗi component bằng mean component
   validation RMSE qua seeds; mean MAE và MAPE là tie-breaker.
8. Refit cấu trúc chung cho từng seed, reconstruct và tính level metrics.

### 5.1 AIC-selected order của từng variable block

| Component | PCE | US_Retail | USD_VND | Import_CN |
|---|---:|---:|---:|---:|
| VMD1 | 5 | 4 | 5 | 5 |
| VMD2 | 5 | 5 | 2 | 4 |
| VMD3 | 5 | 5 | 1 | 4 |
| RES | 1 | 4 | 1 | 1 |

Các order trong bảng là rank-1 theo AIC, không phải order cuối cùng bắt buộc phải
được dùng. Full-grid FFNN sau đó quyết định block nào được giữ hoặc bỏ.

### 5.2 Block combination và FFNN winner

| Component | Target lags | Exogenous blocks được giữ | Input dim | Hidden layers | Alpha | LR |
|---|---|---|---:|---|---:|---:|
| VMD1 | 1–8 | `Import_CN(1–5)` | 13 | `(3,3,3)` | 0.1 | 0.001 |
| VMD2 | 1–11 | `US_Retail(1–5)`, `USD_VND(1–2)` | 18 | `(4,4,4)` | 0.1 | 0.001 |
| VMD3 | 1–10 | `Import_CN(1–4)` | 14 | `(4,4)` | 0.1 | 0.001 |
| RES | 1–12 | `PCE(1)`, `Import_CN(1)` | 14 | `(14,14,14)` | 0.1 | 0.01 |

Điểm đáng chú ý:

- Không component nào giữ cả bốn variable blocks.
- `Import_CN` xuất hiện ở VMD1, VMD3 và RES.
- `PCE` chỉ xuất hiện trong RES dù AIC đã chọn một block PCE cho mọi component.
- Toàn bộ component winners dùng `alpha=0.1`.
- Ba VMD modes chọn LR `0.001`; residual chọn LR `0.01`.
- Cấu trúc trên được khóa chung cho cả năm seeds, đúng với mean-seed protocol.

### 5.3 Kết quả test theo seed

| Seed | Test RMSE | Test MAE | Test MAPE | Test DA |
|---:|---:|---:|---:|---:|
| 7 | 1,064,343 | 782,677 | 7.537% | 68.75% |
| 42 | **1,055,186** | 786,406 | 7.553% | 65.63% |
| 123 | 1,071,540 | **767,718** | **7.401%** | 65.63% |
| 2024 | 1,075,442 | 784,698 | 7.540% | 68.75% |
| 2025 | 1,061,705 | 777,835 | 7.505% | 68.75% |

Không seed nào được chọn làm reporting winner. Bảng chỉ mô tả variation sau khi
cấu trúc chung đã khóa.

### 5.4 Search-space audit

| Đại lượng | Giá trị |
|---|---:|
| Components | 4 |
| Variable-block subsets/component | 15 |
| Hyperparameter signatures/spec | 72 |
| Signatures/component trước seed replication | 1,080 |
| Signatures trên bốn components | 4,320 |
| Reporting seeds | 5 |
| FFNN fits/component | 5,400 |
| Tổng FFNN fits | **21,600** |

Mỗi spec có `9 architectures × 4 alpha × 1 activation × 2 LR = 72`
hyperparameter signatures. Comparator không dùng rolling top-k shortlist; đây là
full-grid exhaustive block-subset benchmark theo thiết kế.

## 6. So sánh comparator mới với Proposed

| Tiêu chí test | Comparator mới | Proposed | Thay đổi của comparator |
|---|---:|---:|---:|
| RMSE | 1,065,643 | 1,124,266 | **−5.21%** |
| MAE | 779,867 | 812,535 | **−4.02%** |
| MAPE | 7.507% | 7.862% | **−4.52%** |
| Directional Accuracy | 67.50% | 63.75% | **+3.75 pp** |
| RMSE standard deviation | 8,021 | 52,792 | thấp hơn 84.8% |
| Total runtime | 3,083.02 giây | 1,205.91 giây | cao hơn 2.56 lần |
| Total FFNN fits | 21,600 | 7,680 | cao hơn 2.81 lần |

Comparator tốt hơn Proposed trên mọi test metric trong run này nhưng tốn chi phí
tính toán lớn hơn. Ngoài ra objective selection không hoàn toàn giống nhau:

- comparator chọn một winner **theo từng component** bằng mean component-scale
  validation score qua seeds;
- Proposed shortlist component candidates rồi chọn tổ hợp bằng reconstructed
  level validation RMSE cho từng seed.

Vì vậy đây là benchmark giữa hai workflow hoàn chỉnh, không phải một ablation chỉ
thay đúng một yếu tố.

## 7. Ablation A0–A5

Comparator block-subset không được đưa vào A0–A5; A4 Proposed vẫn giữ nguyên.

| Hạng | ID | Cấu hình | Test RMSE | MAE | MAPE | DA |
|---:|---|---|---:|---:|---:|---:|
| 1 | A5 | VMD + joint target/exog SelectOrder | **1,104,476 ± 96,146** | 836,993 | **7.830%** | 68.13% |
| 2 | A4 | Proposed sequential selection | 1,124,266 ± 52,792 | **812,535** | 7.862% | 63.75% |
| 3 | A2 | VMD + target selection + fixed exog | 1,133,771 ± 75,847 | 815,981 | 7.840% | 68.13% |
| 4 | A1 | VMD + fixed target/exog | 1,249,420 ± 135,461 | 876,765 | 8.601% | **70.00%** |
| 5 | A0 | FFNN fixed lags, không VMD | 1,290,430 ± 153,982 | 911,696 | 8.868% | 65.63% |
| 6 | A3 | VMD + fixed target + ARDL exog | 1,334,054 ± 41,677 | 944,992 | 9.085% | 68.75% |

Các kết luận ablation không đổi:

- A1 giảm RMSE 3.18% so với A0.
- A2 giảm RMSE 9.25% so với A1.
- A4 giảm RMSE 0.84% và MAE 0.42% so với A2.
- A5 giảm RMSE 1.76% so với A4 nhưng MAE cao hơn.
- A3 làm RMSE xấu hơn A1 khoảng 6.77% khi target lags bị fixed `(1,12)`.

## 8. Diebold–Mariano tests

Có `C(9,2) = 36` pairwise DM tests trên 33 common test dates. Mười cặp có raw
`p < 0.05`; sau Holm correction còn bảy cặp có ý nghĩa. Cả bảy đều là một model
tốt hơn seasonal naive.

| Model tốt hơn seasonal naive | Holm-adjusted p-value |
|---|---:|
| ARDL–FFNN không VMD | 0.000017 |
| VMD–AIC block-subset–FFNN | 0.000224 |
| Standalone ARDL | 0.000283 |
| Proposed | 0.000577 |
| VMD–FFNN fixed | 0.003358 |
| FFNN fixed không VMD | 0.012094 |
| VARNN | 0.033224 |

### Comparator mới so với các đối thủ chính

| Đối thủ | Raw p-value | Holm p-value | Kết luận |
|---|---:|---:|---|
| ARDL–FFNN không VMD | 0.119 | 1.000 | Không khác biệt có ý nghĩa |
| Standalone ARDL | 0.788 | 1.000 | Không khác biệt có ý nghĩa |
| Proposed | 0.535 | 1.000 | Không khác biệt có ý nghĩa |
| VMD–FFNN fixed | 0.357 | 1.000 | Không khác biệt có ý nghĩa |
| VARNN | 0.080 | 1.000 | Không khác biệt có ý nghĩa |
| Naive last observation | 0.103 | 1.000 | Không khác biệt có ý nghĩa |
| Naive seasonal-12 | 0.000006 | 0.000224 | Comparator tốt hơn có ý nghĩa |

Do đó, comparator có point estimates tốt hơn Proposed nhưng chưa có bằng chứng
DM rằng hai forecast-error series khác nhau có ý nghĩa thống kê.

## 9. Proposed rolling lag-screen stability

Comparator mới không chạy rolling shortlist. Rolling one-SE chỉ áp dụng cho các
pipeline staged-halving hiện có:

| Pipeline | Spec summaries | Shortlisted | Seed-component groups | Raw và one-SE khác nhau |
|---|---:|---:|---:|---:|
| ARDL–FFNN không VMD | 80 | 20 | 5 | 3/5 |
| Proposed | 320 | 80 | 20 | 10/20 |
| A3 fixed-target/exog-screen | 320 | 80 | 20 | 9/20 |

Đối với Proposed, one-SE chọn cấu trúc khác raw mean-RMSE winner trong 10/20
seed-component groups. Fast-screen LR winners là 195 lần cho `0.001` và 125 lần
cho `0.01`, nên giữ cả hai learning rates vẫn có cơ sở thực nghiệm.

## 10. Final learning-rate selection

| Pipeline | LR `0.001` | LR `0.01` |
|---|---:|---:|
| Comparator block-subset | 15 | 5 |
| Proposed | 16 | 4 |
| A5 joint-select | 14 | 6 |
| A2 target-select/fixed-exog | 19 | 1 |
| A3 fixed-target/exog-screen | 18 | 2 |
| VMD–FFNN fixed | 17 | 3 |
| ARDL–FFNN không VMD | 4 | 1 |
| FFNN fixed không VMD | 3 | 2 |

Đối với comparator, mỗi component winner được lặp lại qua năm seeds: ba VMD
components chọn `0.001` và RES chọn `0.01`.

## 11. Runtime và search cost

| Pipeline | Số runs | Mean runtime/run | Total runtime | Tổng FFNN fits |
|---|---:|---:|---:|---:|
| Comparator block-subset | 1 shared-seed run | 3,083.02 giây | **3,083.02 giây** | **21,600** |
| Proposed | 5 | 241.18 giây | 1,205.91 giây | 7,680 |
| A3 fixed-target/exog-screen | 5 | 177.06 giây | 885.32 giây | 7,680 |
| A5 joint-select | 5 | 64.82 giây | 324.12 giây | 1,440 |
| A2 target-select/fixed-exog | 5 | 57.93 giây | 289.63 giây | 1,440 |
| ARDL–FFNN không VMD | 5 | 46.89 giây | 234.43 giây | 1,920 |
| VMD–FFNN fixed | 5 | 33.77 giây | 168.83 giây | 1,440 |
| VARNN | 5 | 8.96 giây | 44.80 giây | 5 |
| FFNN fixed không VMD | 5 | 7.95 giây | 39.76 giây | 360 |
| Standalone ARDL | 1 | 3.98 giây | 3.98 giây | 0 |

Tổng model runtime ghi trong efficiency table là 6,279.82 giây, tương đương
104.66 phút. Comparator mới chiếm khoảng 49.1% tổng thời gian. Riêng comparator:

- ARDL selection: 15.30 giây;
- VMD cache: 1.29 giây;
- FFNN full-grid search: 3,057.87 giây;
- refit và forecast: 7.45 giây;
- total internal runtime: 3,082.59 giây.

Khoảng 99.2% thời gian comparator nằm ở full FFNN grid. Đây là chi phí trực tiếp
của việc không dùng rolling shortlist/top-k.

## 12. Stationarity screening

| Biến | Level | First difference | Transform | Cảnh báo I(2) |
|---|---|---|---|---|
| Export_US | Non-stationary | Stationary | `diff1` | Không |
| PCE | Non-stationary | Stationary | `diff1` | Không |
| US_Retail | Non-stationary | Stationary | `diff1` | Không |
| Import_CN | Non-stationary | Stationary | `diff1` | Không |
| USD_VND | Mixed/unclear | Mixed/unclear | `diff1` | **Có** |

`USD_VND` vẫn cần robustness check. First difference chưa cho kết luận
stationarity đồng thuận giữa ADF và KPSS, nhưng pipeline hiện vẫn dùng `diff1`.

## 13. Forecast coverage

| Split | Models | Dates/model | Khoảng thời gian | Cùng dates |
|---|---:|---:|---|---|
| Validation | 9 | 33 | 10/2020–06/2023 | Có |
| Test | 9 | 33 | 07/2023–03/2026 | Có |

Điều kiện common-date cho benchmark và DM test được đáp ứng. Coverage file còn
có VARNN train forecasts; chúng không tham gia benchmark validation/test ranking.

## 14. Kết luận chính

| Mức độ | Kết luận |
|---|---|
| Chính | ARDL–FFNN không VMD vẫn có test RMSE và MAPE tốt nhất |
| Chính | Comparator mới đứng hạng 2 RMSE và hạng 1 MAE |
| Tích cực | Comparator cải thiện 5.21% RMSE so với Proposed và có seed variation thấp |
| Tích cực | Exhaustive block selection loại phần lớn AIC-created blocks; không component nào giữ đủ bốn biến |
| Chi phí | Comparator cần 21,600 fits và 51.38 phút, chiếm gần một nửa benchmark runtime |
| Thận trọng | Comparator và Proposed không khác biệt có ý nghĩa theo DM/Holm |
| Thận trọng | Comparator chọn per-component transformed validation score, còn Proposed chọn reconstructed level score |
| Thận trọng | Chỉ các so sánh với seasonal naive còn ý nghĩa sau Holm correction |
| Cần sửa | VARNN vẫn thiếu Directional Accuracy |
| Cần kiểm tra | `USD_VND` có dấu hiệu I(2) |

Không nên tuyên bố comparator mới hoặc Proposed vượt trội thống kê so với các
đối thủ chính. Kết luận phù hợp là comparator block-subset tạo point forecast tốt
và rất ổn định trong run này, đổi lại chi phí tính toán cao, trong khi ARDL–FFNN
không VMD vẫn dẫn đầu RMSE và MAPE.

## 15. Artifact chính

| Artifact | Số dòng | Nội dung |
|---|---:|---|
| [`benchmark_metrics_comparison.csv`](benchmark_metrics_comparison.csv) | 19 | Aggregate benchmark metrics; gồm VARNN train row |
| [`benchmark_forecasts_comparison.csv`](benchmark_forecasts_comparison.csv) | 2,928 | Benchmark forecasts theo model/seed |
| [`ablation_metrics.csv`](ablation_metrics.csv) | 12 | A0–A5 validation/test metrics |
| [`diebold_mariano_tests.csv`](diebold_mariano_tests.csv) | 36 | Pairwise DM, bootstrap CI và Holm p-values |
| [`efficiency_summary.csv`](efficiency_summary.csv) | 44 | Runtime và evaluated FFNN fits |
| [`lag_screen_fold_metrics.csv`](lag_screen_fold_metrics.csv) | 4,320 | Proposed/staged train-only fold metrics |
| [`lag_screen_stability.csv`](lag_screen_stability.csv) | 1,440 | Mean/std/SE, one-SE và shortlist audit |
| [`lag_selection_summary.csv`](lag_selection_summary.csv) | 130 | Final lags/hyperparameters theo pipeline và seed |
| [`lag_selection_frequency.csv`](lag_selection_frequency.csv) | 433 | Lag-selection frequency |
| [`paper_seed_metrics.csv`](paper_seed_metrics.csv) | 101 | Metrics chi tiết theo seed |
| [`paper_experiment_settings.csv`](paper_experiment_settings.csv) | 12 | Protocol của từng pipeline |
| [`paper_stationarity_screen.csv`](paper_stationarity_screen.csv) | 5 | Train-only stationarity decisions |
| [`paper_forecast_date_coverage.csv`](paper_forecast_date_coverage.csv) | 19 | Common-date audit |
| [`manifest.json`](manifest.json) | — | Full config, dependency versions và run status |

### Artifact riêng của comparator

| Artifact | Số dòng | Nội dung |
|---|---:|---|
| [`ffnn_block_subset_component_search.csv`](vmd_ardl_block_subset_mean_seed/ffnn_block_subset_component_search.csv) | 21,600 | Mọi fitted candidate qua năm seeds |
| [`ffnn_block_subset_mean_seed_selection.csv`](vmd_ardl_block_subset_mean_seed/ffnn_block_subset_mean_seed_selection.csv) | 4,320 | Mean-seed hyperparameter signatures và rank |
| [`best_component_models_block_subset_mean_seed.csv`](vmd_ardl_block_subset_mean_seed/best_component_models_block_subset_mean_seed.csv) | 20 | Bốn component winners × năm seeds |
| [`ardl_variable_blocks_train_only.csv`](vmd_ardl_block_subset_mean_seed/ardl_variable_blocks_train_only.csv) | 204 | Target selection, AIC orders và 15 subsets/component |
| [`final_reconstructed_forecasts_block_subset_mean_seed.csv`](vmd_ardl_block_subset_mean_seed/final_reconstructed_forecasts_block_subset_mean_seed.csv) | 330 | Reconstructed validation/test forecasts |
| [`final_reconstructed_metrics_block_subset_mean_seed.csv`](vmd_ardl_block_subset_mean_seed/final_reconstructed_metrics_block_subset_mean_seed.csv) | 20 | Transformed và level metrics theo seed/split |

Run còn có bảy paper figures về test forecast, error distribution, seed
uncertainty, accuracy–runtime, VMD decomposition và lag-selection heatmaps.
