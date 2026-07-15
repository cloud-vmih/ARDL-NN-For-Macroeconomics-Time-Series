# Benchmark Results Summary

## 1. Thông tin run

| Thuộc tính | Giá trị |
|---|---|
| Run ID | `20260715T084837306824Z_eebe3542` |
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

Run sử dụng workflow mới: train-only rolling lag-screen, one-standard-error
shortlist và final reconstructed selection trên level scale.

## 2. Cấu hình thí nghiệm

| Nhóm | Cấu hình |
|---|---|
| Split | 70% train, 15% outer validation, 15% test |
| Transform | Log tự nhiên và stationarity transform train-only |
| VMD | 3 modes + residual, `alpha=2000`, self-contained backend |
| ARDL | AIC, target lag tối đa 12, exogenous order tối đa 6 |
| Lag generation | Staged, tối đa 16 specs/component, top 3 exogenous orders |
| Rolling lag-screen | 3 expanding folds × 12 tháng |
| Fast-screen learning rates | `0.01`, `0.001` |
| Lag shortlist | `top_k_lag_specs=4`, one-standard-error rule |
| Component shortlist | `top_component_candidates=3` |
| Full FFNN learning rates | `0.01`, `0.001` |
| FFNN architecture | 1–3 hidden layers, width theo input dimension |
| FFNN alpha | `1e-4`, `1e-3`, `1e-2`, `1e-1` |
| Final selection scale | `level` |

### Outer split

| Split | Số quan sát | Bắt đầu | Kết thúc |
|---|---:|---|---|
| Train | 153 | 01/2008 | 09/2020 |
| Validation | 33 | 10/2020 | 06/2023 |
| Test | 33 | 07/2023 | 03/2026 |

### Train-only rolling lag-screen folds

Sau `diff1`, train còn 152 timestamp. Ba inner-validation blocks nằm hoàn toàn
trong outer train:

| Fold | Inner train | Số dòng train | Inner validation | Số tháng |
|---:|---|---:|---|---:|
| 1 | 02/2008–09/2017 | 116 | 10/2017–09/2018 | 12 |
| 2 | 02/2008–09/2018 | 128 | 10/2018–09/2019 | 12 |
| 3 | 02/2008–09/2019 | 140 | 10/2019–09/2020 | 12 |

Outer validation và test không được sử dụng trong bước lag-screen này.

## 3. Benchmark test results

Neural model được báo cáo bằng trung bình ± độ lệch chuẩn của năm paper seeds.

| Hạng | Model | Test RMSE | Test MAE | MAPE | DA |
|---:|---|---:|---:|---:|---:|
| 1 | ARDL–FFNN không VMD | **1,049,951 ± 57,929** | **806,738 ± 26,752** | **7.392%** | 65.00% |
| 2 | Standalone ARDL | 1,084,833 | 808,376 | 7.535% | 68.75% |
| 3 | Proposed VMD–ARDL–FFNN | 1,124,266 ± 52,792 | 812,535 ± 63,097 | 7.862% | 63.75% |
| 4 | VMD–FFNN fixed lag | 1,249,420 ± 135,461 | 876,765 ± 102,446 | 8.601% | **70.00%** |
| 5 | FFNN fixed lag, không VMD | 1,290,430 ± 153,982 | 911,696 ± 105,259 | 8.868% | 65.63% |
| 6 | VARNN | 1,488,365 ± 168,178 | 1,127,312 ± 183,953 | 10.610% | N/A |
| 7 | Naive last observation | 1,611,731 | 1,092,537 | 10.436% | 0.00% |
| 8 | Naive seasonal-12 | 2,383,548 | 2,101,624 | 18.093% | 53.13% |

### Nhận xét benchmark test

- ARDL–FFNN không VMD là model tốt nhất theo RMSE, MAE và MAPE.
- Proposed đứng thứ ba theo RMSE và kém ARDL–FFNN không VMD khoảng 7.08%.
- Proposed kém Standalone ARDL khoảng 3.64% RMSE.
- VMD–FFNN fixed lag có Directional Accuracy cao nhất, nhưng RMSE đứng thứ tư.
- VARNN vẫn chưa xuất Directional Accuracy.

## 4. Benchmark validation results

| Hạng | Model | Validation RMSE | Validation MAE | MAPE | DA |
|---:|---|---:|---:|---:|---:|
| 1 | ARDL–FFNN không VMD | **980,037 ± 19,852** | 755,771 | 9.664% | 61.25% |
| 2 | Proposed VMD–ARDL–FFNN | 989,220 ± 36,770 | 743,408 | 9.604% | 67.50% |
| 3 | FFNN fixed lag, không VMD | 1,007,349 ± 26,923 | 715,439 | 9.194% | 72.50% |
| 4 | VMD–FFNN fixed lag | 1,023,984 ± 29,774 | **707,242** | **9.143%** | **75.63%** |
| 5 | VARNN | 1,036,693 ± 28,273 | 718,295 | 9.266% | N/A |
| 6 | Naive last observation | 1,099,421 | 782,292 | 10.016% | 0.00% |
| 7 | Standalone ARDL | 1,209,736 | 917,699 | 11.740% | 65.63% |
| 8 | Naive seasonal-12 | 1,944,329 | 1,722,014 | 21.090% | 53.13% |

ARDL–FFNN không VMD đứng đầu cả validation và test RMSE. Proposed đứng thứ hai
trên validation nhưng xuống thứ ba trên test do Standalone ARDL tổng quát hóa tốt
hơn sang test period.

## 5. Ablation study

| Hạng | ID | Cấu hình | Test RMSE | MAE | MAPE | DA |
|---:|---|---|---:|---:|---:|---:|
| 1 | A5 | VMD + joint target/exog SelectOrder | **1,104,476 ± 96,146** | 836,993 | **7.830%** | 68.13% |
| 2 | A4 | Proposed sequential target/exog selection | 1,124,266 ± 52,792 | **812,535** | 7.862% | 63.75% |
| 3 | A2 | VMD + target selection + fixed exog lag 1 | 1,133,771 ± 75,847 | 815,981 | 7.840% | 68.13% |
| 4 | A1 | VMD + fixed target/exog lags | 1,249,420 ± 135,461 | 876,765 | 8.601% | **70.00%** |
| 5 | A0 | FFNN fixed lags, không VMD | 1,290,430 ± 153,982 | 911,696 | 8.868% | 65.63% |
| 6 | A3 | VMD + fixed target + ARDL exog screening | 1,334,054 ± 41,677 | 944,992 | 9.085% | 68.75% |

### Diễn giải ablation

| So sánh | Kết quả |
|---|---|
| A1 so với A0 | VMD giảm test RMSE khoảng 3.18% |
| A2 so với A1 | Target-lag selection giảm RMSE khoảng 9.25% |
| A4 so với A2 | Exogenous screening giảm RMSE khoảng 0.84% và MAE khoảng 0.42% |
| A5 so với A4 | Joint selection giảm RMSE khoảng 1.76% |
| A3 so với A1 | Chọn exog khi target bị fixed `(1,12)` làm RMSE xấu hơn khoảng 6.77% |

Workflow mới giúp Proposed vượt A2 nhẹ về RMSE và MAE. Tuy nhiên A2 vẫn tốt hơn
một ít về MAPE và cao hơn 4.38 điểm phần trăm Directional Accuracy. Chênh lệch
A2–A4 chưa có DM test trực tiếp nên không nên tuyên bố đóng góp exogenous
selection có ý nghĩa thống kê.

## 6. Proposed results theo seed

| Seed | Validation RMSE | Test RMSE | Test MAE | Test MAPE | Test DA |
|---:|---:|---:|---:|---:|---:|
| 7 | 986,590 | 1,104,529 | 812,965 | 7.886% | 56.25% |
| 42 | 1,006,089 | **1,079,650** | 757,673 | 7.281% | 68.75% |
| 123 | **935,520** | 1,200,317 | 900,033 | 8.708% | 65.63% |
| 2024 | 1,035,918 | 1,156,596 | 844,543 | 8.167% | 59.38% |
| 2025 | 981,984 | 1,080,235 | **747,463** | **7.269%** | 68.75% |

Seed 123 có validation RMSE tốt nhất nhưng test RMSE xấu nhất. Điều này tiếp tục
cho thấy không được chọn reporting seed theo validation hoặc test winner.

## 7. Diebold–Mariano tests

Có 28 pairwise tests. Sau Holm correction, chỉ sáu cặp có ý nghĩa; tất cả đều
liên quan tới seasonal naive.

| Model tốt hơn | Model kém hơn | Holm-adjusted p-value |
|---|---|---:|
| ARDL–FFNN không VMD | Naive seasonal | 0.000013 |
| Standalone ARDL | Naive seasonal | 0.000225 |
| Proposed | Naive seasonal | 0.000455 |
| VMD–FFNN fixed | Naive seasonal | 0.002623 |
| FFNN fixed không VMD | Naive seasonal | 0.009363 |
| VARNN | Naive seasonal | 0.025472 |

### Proposed so với benchmark models

| Đối thủ | Holm-adjusted p-value | Kết luận |
|---|---:|---|
| ARDL–FFNN không VMD | 1.000 | Không khác biệt có ý nghĩa |
| Standalone ARDL | 1.000 | Không khác biệt có ý nghĩa |
| FFNN fixed không VMD | 1.000 | Không khác biệt có ý nghĩa |
| VMD–FFNN fixed | 1.000 | Không khác biệt có ý nghĩa |
| VARNN | 1.000 | Không khác biệt có ý nghĩa |
| Naive last observation | 1.000 | Không khác biệt có ý nghĩa |
| Naive seasonal | 0.000455 | Proposed tốt hơn có ý nghĩa |

Proposed chưa có bằng chứng thống kê rằng nó vượt các đối thủ chính.

## 8. Rolling lag-screen stability

Rolling screening chỉ chạy với các pipeline có nhiều hơn một lag spec:

| Pipeline | Số spec-level summaries | Shortlisted | Raw/one-SE groups | Raw và one-SE khác nhau | Input giảm trung bình |
|---|---:|---:|---:|---:|---:|
| ARDL–FFNN không VMD | 80 | 20 | 5 | 3/5 | 1.2 inputs |
| Proposed | 320 | 80 | 20 | 10/20 | 1.7 inputs |
| A3 fixed-target/exog-screen | 320 | 80 | 20 | 9/20 | 0.9 inputs |

Mỗi Proposed seed có bốn components × 16 lag specs. Mỗi seed-component giữ đúng
bốn lag specs cho full FFNN grid. Trong 10/20 Proposed seed-component groups,
one-SE chọn cấu trúc đơn giản hơn raw RMSE winner.

Tổng artifact gồm:

- 4,320 fold-level rows = candidate specs × 3 folds × 2 learning rates.
- 1,440 stability rows = candidate specs × 2 learning rates.
- 720 spec-level best-learning-rate summaries.

### Fast-screen learning-rate winners

| Pipeline | LR `0.001` | LR `0.01` |
|---|---:|---:|
| ARDL–FFNN không VMD | 72 | 8 |
| Proposed | 195 | 125 |
| A3 fixed-target/exog-screen | 209 | 111 |

`0.001` chiếm ưu thế, nhưng `0.01` vẫn thắng ở tỷ lệ đáng kể với các VMD lag
specs; giữ cả hai learning rates trong fast-screen là hợp lý.

## 9. Final FFNN learning-rate selection

| Pipeline | Final LR `0.001` | Final LR `0.01` |
|---|---:|---:|
| ARDL–FFNN không VMD | 4 | 1 |
| FFNN fixed không VMD | 3 | 2 |
| Proposed | 16 | 4 |
| Joint-select A5 | 14 | 6 |
| VMD–FFNN fixed | 17 | 3 |
| A3 fixed-target/exog-screen | 18 | 2 |
| A2 target-select/fixed-exog | 19 | 1 |

Đối với Proposed, `0.001` thắng ở 16/20 component models và `0.01` thắng ở 4/20.

## 10. Proposed final lag structure

Target selection vẫn deterministic giữa seeds:

| Component | Target lags | Mean final input dimension | Khoảng input dimension |
|---|---|---:|---:|
| RES | 1–12 | 18.4 | 16–21 |
| VMD1 | 1–8 | 27.6 | 26–29 |
| VMD2 | 1–11 | 27.4 | 25–29 |
| VMD3 | 1–10 | 25.2 | 23–26 |

Exogenous lag specification và FFNN architecture thay đổi theo seed. Target lag
dải dài vẫn làm input dimension lớn, đặc biệt ở VMD1–VMD3.

## 11. Runtime và search cost

| Pipeline | Mean runtime/run | Total runtime | Mean FFNN fits/run | Lag specs/run |
|---|---:|---:|---:|---:|
| Proposed | **234.29 giây** | 1,171.43 giây | 1,536 | 64 |
| A3 fixed-target/exog-screen | 175.61 giây | 878.06 giây | 1,536 | 64 |
| Joint-select A5 | 62.94 giây | 314.69 giây | 288 | 4 |
| A2 target-select/fixed-exog | 60.59 giây | 302.95 giây | 288 | 4 |
| ARDL–FFNN không VMD | 46.23 giây | 231.13 giây | 384 | 16 |
| VMD–FFNN fixed | 34.32 giây | 171.62 giây | 288 | 4 |
| VARNN | 8.91 giây | 44.57 giây | 1 | — |
| FFNN fixed không VMD | 8.04 giây | 40.19 giây | 72 | 1 |
| Standalone ARDL | 3.90 giây | 3.90 giây | 0 | — |

Tổng thời gian ghi nhận là 3,158.57 giây, tương đương 52.64 phút. Proposed là
pipeline đắt nhất và chậm hơn A5 khoảng 3.72 lần.

## 12. So sánh với pre-rolling run

So với run `20260715T055818291775Z_77d93126` trước khi có rolling lag-screen và
level-scale final selection:

| Pipeline | RMSE cũ | RMSE mới | Thay đổi |
|---|---:|---:|---:|
| ARDL–FFNN không VMD | 1,115,131 | 1,049,951 | **−5.85%** |
| Proposed | 1,138,750 | 1,124,266 | **−1.27%** |
| VMD–FFNN fixed | 1,234,764 | 1,249,420 | +1.19% |
| Standalone ARDL | 1,084,833 | 1,084,833 | Không đổi |
| FFNN fixed không VMD | 1,290,430 | 1,290,430 | Không đổi |
| VARNN | 1,488,365 | 1,488,365 | Không đổi |

Đây là so sánh trước–sau workflow, không phải kiểm định thống kê độc lập. Không
nên dùng riêng bảng này để tuyên bố rolling selection chắc chắn cải thiện mọi run.

## 13. Stationarity screening

| Biến | Level | First difference | Transform | Cảnh báo I(2) |
|---|---|---|---|---|
| Export_US | Không stationary | Stationary | `diff1` | Không |
| PCE | Không stationary | Stationary | `diff1` | Không |
| US_Retail | Không stationary | Stationary | `diff1` | Không |
| Import_CN | Không stationary | Stationary | `diff1` | Không |
| USD_VND | Mixed/unclear | Mixed/unclear | `diff1` | **Có** |

`USD_VND` vẫn là vấn đề cần kiểm tra robustness. First difference chưa đạt kết
luận stationarity rõ ràng nhưng pipeline vẫn dùng `diff1`.

## 14. Forecast coverage

| Split | Models | Số dates/model | Khoảng thời gian | Cùng dates |
|---|---:|---:|---|---|
| Validation | 8 | 33 | 10/2020–06/2023 | Có |
| Test | 8 | 33 | 07/2023–03/2026 | Có |

Điều kiện common-date comparison cho benchmark và DM test được đáp ứng.

## 15. Kết luận chính

| Mức độ | Kết luận |
|---|---|
| Chính | ARDL–FFNN không VMD hiện là benchmark winner theo test RMSE/MAE/MAPE |
| Chính | Proposed đứng thứ ba benchmark và thứ hai trong ablation |
| Chính | A5 joint-select tốt hơn Proposed về RMSE và rẻ hơn khoảng 3.72 lần |
| Tích cực | Proposed mới tốt hơn A2 khoảng 0.84% RMSE; exog selection không còn làm RMSE xấu hơn |
| Tích cực | Rolling one-SE chọn cấu trúc đơn giản hơn trong 10/20 Proposed groups |
| Thận trọng | Proposed chỉ khác seasonal naive có ý nghĩa sau Holm correction |
| Thận trọng | Chênh lệch A2–A4 chưa có DM test trực tiếp |
| Cần sửa | VARNN thiếu Directional Accuracy |
| Cần kiểm tra | `USD_VND` có dấu hiệu I(2) |

Không nên mô tả Proposed là model tốt nhất hoặc vượt trội thống kê. Kết quả phù
hợp hơn với kết luận rằng framework mang lại hiệu năng cạnh tranh, nhưng baseline
ARDL–FFNN không VMD và joint-select comparator vẫn mạnh hơn trong run này.

## 16. Artifact chính

| Artifact | Số dòng | Nội dung |
|---|---:|---|
| [`benchmark_metrics_comparison.csv`](benchmark_metrics_comparison.csv) | 17 | Benchmark validation/test metrics |
| [`benchmark_forecasts_comparison.csv`](benchmark_forecasts_comparison.csv) | 2,598 | Benchmark forecasts theo seed |
| [`ablation_metrics.csv`](ablation_metrics.csv) | 12 | A0–A5 aggregate metrics |
| [`ablation_forecasts.csv`](ablation_forecasts.csv) | 1,980 | A0–A5 forecasts |
| [`diebold_mariano_tests.csv`](diebold_mariano_tests.csv) | 28 | Pairwise DM và Holm p-values |
| [`efficiency_summary.csv`](efficiency_summary.csv) | 43 | Runtime và evaluated FFNN fits |
| [`lag_screen_fold_metrics.csv`](lag_screen_fold_metrics.csv) | 4,320 | Train-only fold metrics cho mọi lag spec/LR |
| [`lag_screen_stability.csv`](lag_screen_stability.csv) | 1,440 | Mean/std/SE, one-SE và shortlist audit |
| [`lag_selection_summary.csv`](lag_selection_summary.csv) | 110 | Final lags và hyperparameters theo seed/component |
| [`lag_selection_frequency.csv`](lag_selection_frequency.csv) | 374 | Final lag-selection frequency |
| [`paper_seed_metrics.csv`](paper_seed_metrics.csv) | 91 | Metrics chi tiết theo seed |
| [`paper_metrics_summary.csv`](paper_metrics_summary.csv) | 23 | Aggregate paper metrics |
| [`paper_experiment_settings.csv`](paper_experiment_settings.csv) | 11 | Protocol và settings của từng pipeline |
| [`paper_stationarity_screen.csv`](paper_stationarity_screen.csv) | 5 | Train-only stationarity decisions |
| [`paper_forecast_date_coverage.csv`](paper_forecast_date_coverage.csv) | 17 | Common-date audit |
| [`manifest.json`](manifest.json) | — | Run configuration, dependency versions và status |

Thư mục còn có bảy paper figures về test forecast, error distribution, seed
uncertainty, accuracy–runtime, VMD decomposition và lag-selection heatmaps.
