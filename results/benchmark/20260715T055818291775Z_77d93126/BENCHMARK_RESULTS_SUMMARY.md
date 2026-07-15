# Benchmark Results Summary

## 1. Thông tin run

| Thuộc tính | Giá trị |
|---|---|
| Run ID | `20260715T055818291775Z_77d93126` |
| Trạng thái | `success` |
| Target | `Export_US` |
| Exogenous variables | `PCE`, `US_Retail`, `USD_VND`, `Import_CN` |
| Số quan sát | 219 tháng |
| Thời gian dữ liệu | 01/2008–03/2026 |
| Log transform | Có |
| Train / Validation / Test | 70% / 15% / 15% |
| VMD | `K=3` và residual |
| Target lag tối đa | 12 |
| Exogenous lag tối đa | 6 |
| ARDL criterion | AIC |
| FFNN learning-rate grid | `0.01`, `0.001` |
| Paper seeds | `7`, `42`, `123`, `2024`, `2025` |
| VARNN | Đã chạy thành công |

### Phạm vi dữ liệu

| Split | Số tháng | Bắt đầu | Kết thúc |
|---|---:|---|---|
| Train | 153 | 01/2008 | 09/2020 |
| Validation | 33 | 10/2020 | 06/2023 |
| Test | 33 | 07/2023 | 03/2026 |

Tất cả tám benchmark model dùng cùng 33 thời điểm test. Riêng VARNN có 150 quan sát train sau khi loại phần đầu do lag.

## 2. Benchmark trên test

Các neural model được báo cáo dưới dạng trung bình ± độ lệch chuẩn của năm paper seeds.

| Hạng | Model | RMSE | MAE | MAPE | DA |
|---:|---|---:|---:|---:|---:|
| 1 | Standalone ARDL | 1,084,833 | 808,376 | 7.535% | 68.75% |
| 2 | ARDL–FFNN không VMD | 1,115,131 ± 102,918 | 872,547 ± 114,118 | 7.986% | 64.38% |
| 3 | Proposed VMD–ARDL–FFNN | 1,138,750 ± 78,267 | 873,288 ± 116,852 | 8.374% | 63.13% |
| 4 | VMD–FFNN fixed lag | 1,234,764 ± 117,799 | 851,726 ± 86,769 | 8.377% | **70.00%** |
| 5 | FFNN fixed lag | 1,290,430 ± 153,982 | 911,696 ± 105,259 | 8.868% | 65.63% |
| 6 | VARNN | 1,488,365 ± 168,178 | 1,127,312 ± 183,953 | 10.610% | N/A |
| 7 | Naive last observation | 1,611,731 | 1,092,537 | 10.436% | 0.00% |
| 8 | Naive seasonal-12 | 2,383,548 | 2,101,624 | 18.093% | 53.13% |

### Nhận xét

- Standalone ARDL có RMSE, MAE và MAPE tốt nhất.
- Proposed đứng thứ ba theo RMSE, kém Standalone ARDL khoảng 4.97%.
- Proposed kém ARDL–FFNN không VMD khoảng 2.12% RMSE.
- VMD–FFNN fixed lag có Directional Accuracy cao nhất.
- VARNN chưa xuất Directional Accuracy.

## 3. Benchmark trên validation

| Hạng | Model | Validation RMSE |
|---:|---|---:|
| 1 | ARDL–FFNN không VMD | 957,845 ± 43,093 |
| 2 | Proposed VMD–ARDL–FFNN | 980,237 ± 26,188 |
| 3 | FFNN fixed lag | 1,007,349 ± 26,923 |
| 4 | VMD–FFNN fixed lag | 1,027,938 ± 24,424 |
| 5 | VARNN | 1,036,693 ± 28,273 |
| 6 | Naive last observation | 1,099,421 |
| 7 | Standalone ARDL | 1,209,736 |
| 8 | Naive seasonal-12 | 1,944,329 |

Standalone ARDL không tốt trên validation nhưng lại tốt nhất trên test. Hiệu năng có thể thay đổi theo giai đoạn, vì vậy không nên kết luận chỉ dựa trên một split.

## 4. Ablation study

| ID | Biến thể | Test RMSE | MAE | MAPE | DA |
|---|---|---:|---:|---:|---:|
| A0 | FFNN fixed, không VMD | 1,290,430 ± 153,982 | 911,696 | 8.868% | 65.63% |
| A1 | VMD + FFNN fixed | 1,234,764 ± 117,799 | 851,726 | 8.377% | 70.00% |
| A2 | VMD + chọn target lag + fixed exog | 1,133,771 ± 75,847 | **815,981** | 7.840% | 68.13% |
| A3 | VMD + fixed target + chọn exog lag | 1,263,883 ± 96,522 | 880,133 | 8.470% | 69.38% |
| A4 | Proposed: chọn target rồi chọn exog | 1,138,750 ± 78,267 | 873,288 | 8.374% | 63.13% |
| A5 | VMD–ARDL joint select | **1,101,043 ± 96,322** | 834,805 | **7.802%** | 68.13% |

### Đóng góp của từng thành phần

| So sánh | Kết quả |
|---|---|
| A1 so với A0 | VMD giảm RMSE khoảng 4.3% |
| A2 so với A1 | Target-lag selection giảm RMSE khoảng 8.2% |
| A4 so với A2 | Thêm exogenous-lag selection làm RMSE xấu hơn 0.44% và MAE xấu hơn khoảng 7.0% |
| A5 so với A4 | Joint selection giảm RMSE khoảng 3.3% |
| A5 so với benchmark | Tốt hơn Proposed nhưng vẫn kém Standalone ARDL |

Pipeline Proposed hiện chưa chứng minh được bước chọn exogenous lag mang lại lợi ích dự báo. A2 tốt hơn A4 ở RMSE, MAE, MAPE và Directional Accuracy.

## 5. Diebold–Mariano test

Có 28 cặp so sánh. Chỉ sáu cặp có ý nghĩa sau hiệu chỉnh Holm và tất cả đều liên quan đến `naive_seasonal`.

| Model tốt hơn | Model kém hơn | Holm-adjusted p-value |
|---|---|---:|
| ARDL–FFNN không VMD | Naive seasonal | 0.000013 |
| Proposed | Naive seasonal | 0.000163 |
| Standalone ARDL | Naive seasonal | 0.000217 |
| VMD–FFNN fixed | Naive seasonal | 0.003357 |
| FFNN fixed | Naive seasonal | 0.009363 |
| VARNN | Naive seasonal | 0.025472 |

### Các so sánh của Proposed

| Đối thủ | Kết luận sau hiệu chỉnh Holm |
|---|---|
| Standalone ARDL | Không khác biệt có ý nghĩa |
| ARDL–FFNN không VMD | Không khác biệt có ý nghĩa |
| FFNN fixed | Không khác biệt có ý nghĩa |
| VMD–FFNN fixed | Không khác biệt có ý nghĩa |
| VARNN | Không khác biệt có ý nghĩa |
| Naive last observation | Không khác biệt có ý nghĩa |
| Naive seasonal | Proposed tốt hơn có ý nghĩa |

Chưa thể kết luận Proposed vượt trội các đối thủ chính về mặt thống kê. A5 joint-select chưa được đưa vào bảng DM benchmark.

## 6. Chi phí tính toán

| Pipeline | Trung bình/run | Tổng thời gian | FFNN fits/run | Lag specifications |
|---|---:|---:|---:|---:|
| Proposed A4 | **213.26 giây** | 1,066.28 giây | 1,216 | 64 |
| A3 – chọn exog | 157.41 giây | 787.03 giây | 1,216 | 64 |
| A5 – joint select | 62.11 giây | 310.56 giây | 292 | 4 |
| A2 – target select | 56.20 giây | 281.02 giây | 292 | 4 |
| ARDL–FFNN không VMD | 42.27 giây | 211.35 giây | 304 | 16 |
| VMD–FFNN fixed | 33.05 giây | 165.27 giây | 292 | 4 |
| VARNN | 8.83 giây | 44.17 giây | — | — |
| FFNN fixed | 7.93 giây | 39.63 giây | 73 | 1 |
| Standalone ARDL | 4.19 giây | 4.19 giây | — | — |

Tổng runtime được ghi nhận là khoảng 2,909.5 giây, tương đương 48.5 phút. Proposed đắt hơn A5 khoảng 3.4 lần trong khi RMSE kém hơn.

## 7. Learning-rate selection

Các VMD model có một lần chọn learning rate cho mỗi component và mỗi seed.

| Model | Số lần chọn `0.001` | Số lần chọn `0.01` |
|---|---:|---:|
| FFNN fixed không VMD | 3 | 2 |
| ARDL–FFNN không VMD | 5 | 0 |
| VMD–FFNN fixed | 17 | 3 |
| A2 – target select | 19 | 1 |
| A3 – exog select | 16 | 4 |
| Proposed | 15 | 5 |
| Joint select | 13 | 7 |
| VARNN | 1 | 4 |

Đối với FFNN, `0.001` chiếm ưu thế rõ rệt. Riêng VARNN, `0.01` thắng ở bốn trên năm seeds.

## 8. Lag selection của Proposed

| Component | Target lags ở cả năm seeds | Số input trung bình | LR `0.001` / `0.01` |
|---|---|---:|---:|
| RES | 1–12 | 17.8 | 3 / 2 |
| VMD1 | 1–8 | 28.4 | 5 / 0 |
| VMD2 | 1–11 | 28.6 | 3 / 2 |
| VMD3 | 1–10 | 24.4 | 4 / 1 |

Target-lag selection ổn định giữa các seeds nhưng đang chọn những dải lag dài và liên tục. Điều này làm input dimension lớn và góp phần khiến Proposed phải fit 1,216 FFNN cho mỗi run.

Đối với exogenous lag:

- Lag 1 được chọn ở tất cả các cặp component–exogenous variable.
- Các lag cao hơn thay đổi theo seed.
- Heatmap đang thể hiện ARDL order tối đa theo dải liên tục `1:q`, không phải lựa chọn sparse từng lag độc lập.

## 9. Stationarity screening

| Biến | Level | First difference | Transform | Cảnh báo I(2) |
|---|---|---|---|---|
| Export_US | Không stationary | Stationary | `diff1` | Không |
| PCE | Không stationary | Stationary | `diff1` | Không |
| US_Retail | Không stationary | Stationary | `diff1` | Không |
| Import_CN | Không stationary | Stationary | `diff1` | Không |
| USD_VND | Mixed/không rõ | Mixed/không rõ | `diff1` | **Có** |

`USD_VND` cần được xem xét trước khi viết paper. First difference chưa cho bằng chứng stationarity rõ ràng nhưng pipeline vẫn dùng `diff1`. Nên kiểm tra thêm second difference, structural-break test hoặc robustness khi loại biến này.

## 10. Các vấn đề cần xử lý

| Mức độ | Vấn đề | Ảnh hưởng |
|---|---|---|
| Cao | Proposed không phải model tốt nhất | Không nên tuyên bố best-performing hoặc vượt trội |
| Cao | Exogenous-lag selection làm A4 kém A2 | Chưa chứng minh được đóng góp của bước chọn exog |
| Cao | DM chỉ có ý nghĩa so với seasonal naive | Bằng chứng thống kê về superiority còn yếu |
| Cao | `USD_VND` có dấu hiệu I(2) | Có thể ảnh hưởng giả định preprocessing |
| Trung bình | VARNN thiếu Directional Accuracy | Bảng metrics paper chưa đồng nhất |
| Trung bình | A5 joint-select chưa có trong DM benchmark | Thiếu kiểm định cho ablation tốt nhất |
| Trung bình | Target lags dài và liên tục | Tăng input dimension và chi phí FFNN |
| Trung bình | Proposed có runtime lớn nhất | Trade-off accuracy–runtime không thuận lợi |
| Thấp | Terminal chỉ in cột `split` | Lỗi hiển thị; dữ liệu aggregate vẫn có trong CSV |

## 11. Danh mục artifact chính

| Artifact | Số dòng | Nội dung |
|---|---:|---|
| [`benchmark_metrics_comparison.csv`](benchmark_metrics_comparison.csv) | 17 | Metrics validation và test của benchmark |
| [`benchmark_forecasts_comparison.csv`](benchmark_forecasts_comparison.csv) | 2,598 | Forecast của các benchmark model |
| [`ablation_metrics.csv`](ablation_metrics.csv) | 12 | Metrics của sáu cấu hình ablation |
| [`ablation_forecasts.csv`](ablation_forecasts.csv) | 1,980 | Forecast của các ablation model |
| [`diebold_mariano_tests.csv`](diebold_mariano_tests.csv) | 28 | DM test và Holm-adjusted p-values |
| [`efficiency_summary.csv`](efficiency_summary.csv) | 43 | Runtime, số fits và search-space reduction |
| [`lag_selection_frequency.csv`](lag_selection_frequency.csv) | 369 | Tần suất lựa chọn từng lag |
| [`lag_selection_summary.csv`](lag_selection_summary.csv) | 110 | Lag được chọn theo model, component và seed |
| [`paper_forecasts.csv`](paper_forecasts.csv) | 3,588 | Forecast hợp nhất dùng cho paper |
| [`paper_seed_metrics.csv`](paper_seed_metrics.csv) | 91 | Metrics chi tiết theo seed |
| [`paper_metrics_summary.csv`](paper_metrics_summary.csv) | 23 | Metrics tổng hợp dùng cho paper |
| [`paper_dataset_summary.csv`](paper_dataset_summary.csv) | 5 | Thống kê dataset |
| [`paper_split_summary.csv`](paper_split_summary.csv) | 3 | Thông tin train/validation/test |
| [`paper_stationarity_screen.csv`](paper_stationarity_screen.csv) | 5 | Kết quả stationarity screening |
| [`paper_forecast_date_coverage.csv`](paper_forecast_date_coverage.csv) | 17 | Kiểm tra forecast-date coverage |
| [`paper_experiment_settings.csv`](paper_experiment_settings.csv) | 11 | Cấu hình thí nghiệm |
| [`manifest.json`](manifest.json) | — | Cấu hình và trạng thái toàn bộ run |

Ngoài các CSV/JSON trên, thư mục còn chứa bảy biểu đồ paper về forecast test, error distribution, seed uncertainty, runtime–accuracy, VMD decomposition và lag-selection heatmap.
