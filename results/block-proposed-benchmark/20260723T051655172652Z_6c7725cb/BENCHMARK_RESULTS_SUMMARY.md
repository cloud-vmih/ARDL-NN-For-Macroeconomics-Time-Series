# Tổng hợp kết quả `block-proposed-benchmark`

## 1. Thông tin run

| Thuộc tính | Giá trị |
|---|---|
| Run ID | `20260723T051655172652Z_6c7725cb` |
| Trạng thái | `success` |
| Thời điểm ghi kết quả | 2026-07-23 09:08:15 UTC (16:08:15 UTC+7) |
| Dữ liệu | `df_final.csv` |
| Target | `Export_US` |
| Exogenous variables | `PCE`, `US_Retail`, `USD_VND`, `Import_CN` |
| Tần suất | Tháng (`MS`) |
| Số quan sát | 219, từ 2008-01 đến 2026-03 |
| Forecast protocol | `rolling_one_step` |
| VMD | 3 modes + residual, self-contained backend, `alpha=2000` |
| ARDL search bound | Target lag tối đa 12; exogenous lag order tối đa 6; AIC; causal |
| Reporting seeds | 7, 42, 123, 2024, 2025 |
| FFNN learning rates | 0.01 và 0.001 |
| TensorFlow | 2.21.0 |

Phân chia dữ liệu:

| Split | Số quan sát | Khoảng thời gian |
|---|---:|---|
| Train | 153 | 2008-01 đến 2020-09 |
| Validation | 33 | 2020-10 đến 2023-06 |
| Test | 33 | 2023-07 đến 2026-03 |

Tất cả model được so sánh trên cùng 33 ngày validation và cùng 33 ngày test.

## 2. Cách đọc hai loại kết quả

Run này báo cáo hai estimand khác nhau:

1. **Primary — distribution of per-seed metrics**: tính metric riêng cho từng seed, sau đó báo `mean/std/median/count` của 5 metric đó. Đây là estimand phản ánh expected single-model performance và độ ổn định giữa các random initializations.
2. **Secondary — ensemble mean-prediction**: lấy trung bình dự báo của 5 seed tại từng thời điểm trước, sau đó mới tính RMSE/MAE/MAPE/Directional Accuracy. Đây là kết quả của một deployment ensemble gồm 5 model.

Vì RMSE là hàm phi tuyến nên RMSE của dự báo ensemble không bắt buộc bằng trung bình RMSE của 5 seed.

Run artifact này được sinh trước khi đổi reporting convention, nên file
`benchmark_metrics_comparison.csv` bên trong run vẫn chứa ensemble metrics.
Trong báo cáo này, kết quả chính được đọc từ
`benchmark_seed_robustness_metrics.csv`. Từ các run tiếp theo,
`benchmark_metrics_comparison.csv` sẽ là bảng per-seed mean/std chính và
`benchmark_ensemble_metrics_comparison.csv` sẽ giữ ensemble metrics phụ.

## 3. Benchmark chính: mean ± std qua seed

Bảng được xếp hạng theo test RMSE mean. Model neural có 5 seed; ARDL
và naive baselines là deterministic nên `std` không áp dụng.

| Test rank | Pipeline | Val RMSE mean ± std (rank) | Test RMSE mean ± std | Test MAE mean | Test MAPE mean | Test DA mean |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `ardl_ffnn_no_vmd` | 1,011,849 ± 11,987 (3) | **1,008,860 ± 8,419** | 768,415 | **7.182%** | 62.500% |
| 2 | `vmd_ardl_block_subset_proposed` | **984,147 ± 10,191 (1)** | 1,067,400 ± 9,648 | 788,286 | 7.565% | 67.500% |
| 3 | `vmd_ardl_sequential_screen` | 1,014,099 ± 3,740 (4) | 1,073,221 ± 10,315 | 770,741 | 7.532% | 69.375% |
| 4 | `standalone_ardl` | 1,209,736 (10) | 1,084,833 | 808,376 | 7.535% | 68.750% |
| 5 | `vmd_ardl_joint_select` | 1,004,778 ± 7,141 (2) | 1,088,381 ± 12,980 | 771,660 | 7.499% | 70.625% |
| 6 | `vmd_ffnn_fixed` | 1,047,788 ± 5,125 (7) | 1,112,210 ± 13,082 | **751,059** | 7.415% | **76.250%** |
| 7 | `ffnn_fixed_no_vmd` | 1,033,583 ± 5,069 (6) | 1,147,936 ± 4,910 | 804,451 | 7.872% | 65.000% |
| 8 | `vmd_ardl_hard_config_ffnn` | 1,062,299 ± 66,257 (8) | 1,233,741 ± 126,071 | 886,361 | 8.555% | 67.500% |
| 9 | `varnn` | 1,022,504 ± 28,954 (5) | 1,406,113 ± 33,839 | 1,045,712 | 9.952% | — |
| 10 | `naive_last_observation` | 1,099,421 (9) | 1,611,731 | 1,092,537 | 10.436% | 0.000% |
| 11 | `naive_seasonal` | 1,944,329 (11) | 2,383,548 | 2,101,624 | 18.093% | 53.125% |

### So sánh trực tiếp Proposed

- Proposed có validation RMSE mean tốt nhất, thấp hơn Joint Select 2.05%, ARDL-FFNN không VMD 2.74% và Sequential Screen 2.95%.
- Trên test, Proposed đứng thứ hai nhưng RMSE mean cao hơn `ardl_ffnn_no_vmd` 5.80%.
- Proposed tốt hơn Sequential Screen cũ 0.54%, Joint Select 1.93%, VMD fixed 4.03%, hard-config 13.48%, last-observation 33.77% và seasonal naive 55.22% về test RMSE mean.
- Proposed không đứng đầu mọi metric: `vmd_ffnn_fixed` có test MAE và Directional Accuracy tốt nhất; `ardl_ffnn_no_vmd` có RMSE và MAPE tốt nhất.

## 4. Kết quả phụ: ensemble mean-prediction

Đây là metric sau khi trung bình prediction của 5 seed tại mỗi ngày. Nó phù
hợp nếu deployment thực tế cũng chạy cả 5 model rồi lấy trung bình dự báo.

| Test rank | Pipeline | Test RMSE | Test MAE | Test MAPE | Test DA |
|---:|---|---:|---:|---:|---:|
| 1 | `ardl_ffnn_no_vmd` | **1,006,994** | 768,415 | **7.182%** | 62.500% |
| 2 | `vmd_ardl_block_subset_proposed` | 1,064,699 | 780,266 | 7.494% | 65.625% |
| 3 | `vmd_ardl_sequential_screen` | 1,070,080 | 767,654 | 7.509% | 68.750% |
| 4 | `standalone_ardl` | 1,084,833 | 808,376 | 7.535% | 68.750% |
| 5 | `vmd_ardl_joint_select` | 1,086,763 | 769,909 | 7.483% | 68.750% |
| 6 | `vmd_ffnn_fixed` | 1,111,385 | **749,606** | 7.402% | **78.125%** |
| 7 | `ffnn_fixed_no_vmd` | 1,147,422 | 804,418 | 7.872% | 65.625% |
| 8 | `vmd_ardl_hard_config_ffnn` | 1,196,256 | 860,375 | 8.302% | 68.750% |
| 9 | `varnn` | 1,397,784 | 1,040,854 | 9.907% | 43.750% |
| 10 | `naive_last_observation` | 1,611,731 | 1,092,537 | 10.436% | 0.000% |
| 11 | `naive_seasonal` | 2,383,548 | 2,101,624 | 18.093% | 53.125% |

Ensemble làm RMSE của Proposed giảm từ mean single-model 1,067,400 xuống
1,064,699. Mức giảm nhỏ vì Proposed vốn đã khá ổn định giữa các seed
(`std=9,648`, khoảng 0.90% của mean).

## 5. Ablation A0–A7

Ablation table dùng **mean của metric tính riêng trên 5 seed**, không dùng RMSE của dự báo ensemble. Runtime A6 là tổng của năm lần chạy per-seed.

| ID | Mô tả ngắn | Val RMSE mean | Test RMSE mean ± std | Runtime (s) | FFNN fits |
|---|---|---:|---:|---:|---:|
| A0 | FFNN fixed lag, không VMD | 1,033,583 | 1,147,936 ± 4,910 | 35.5 | 360 |
| A1 | VMD + FFNN fixed lag | 1,047,788 | 1,112,210 ± 13,082 | 148.2 | 1,440 |
| A2 | ARDL target lag + fixed exog lag | 1,041,253 | 1,075,651 ± 7,237 | 219.5 | 1,440 |
| A3 | AIC block, giữ tất cả variable blocks | 1,037,827 | 1,076,842 ± 6,703 | 261.5 | 1,440 |
| A4 | Block subset, chọn theo component mean-seed | 991,908 | **1,065,643 ± 8,021** | 3,446.4 | 21,600 |
| A5 | Block subset, shared mean-seed full grid | 988,073 | 1,069,154 ± 10,479 | 3,251.4 | 21,600 |
| A6 | Block subset, per-seed full grid | **869,989** | 1,143,227 ± 194,388 | 3,702.1 | 21,600 |
| A7 | Proposed shared mean-seed two-stage | 984,147 | 1,067,400 ± 9,648 | **839.8** | **6,060** |

Các kết luận chính từ ablation:

- A4 có test RMSE mean tốt nhất, nhưng chỉ tốt hơn A7 khoảng 0.165%; đổi lại A4 cần 21,600 fits và lâu gấp khoảng 4.1 lần.
- A7 tốt hơn A5 khoảng 0.164% trên test, đồng thời giảm 71.94% số fits và 74.17% runtime so với shared full-grid A5.
- So với tổng runtime A6, A7 giảm khoảng 77.31%.
- A6 đạt validation RMSE rất thấp nhưng test RMSE xấu hơn A7 7.10% và test standard deviation tăng lên 194,388. Đây là dấu hiệu rõ của selection instability/validation overfitting khi mỗi seed tự chọn một cấu hình khác nhau.
- A2 và A3 gần nhau trên test; việc mở toàn bộ AIC blocks trong A3 không tạo ra cải thiện rõ so với fixed exogenous lag của A2 trong run này.

## 6. Cấu hình được Proposed chọn

Two-stage workflow hoạt động như sau:

- ARDL sinh 15 non-empty variable-block subsets cho mỗi component.
- Fast screen dùng `alpha=0.1`, `learning_rate_init=0.001`, lấy top 4 lag specs mỗi component vào full hyperparameter search.
- Sau full search, giữ top 3 component candidates; reconstruction search đánh giá `3^4 = 81` combinations.
- Combination cuối được chọn bằng mean reconstructed-level validation RMSE qua 5 seed.

Combination thắng là `combo_id=30`, tổng input dimension 54:

| Component | Target lags | Exogenous block được chọn | Hidden-layer sizes | Alpha | LR |
|---|---|---|---:|---:|---:|
| VMD1 | 1–8 | `PCE` lags 1–5 | `(3, 3, 3)` | 0.1 | 0.001 |
| VMD2 | 1–11 | `USD_VND` lags 1–2 | `(3, 3)` | 0.1 | 0.001 |
| VMD3 | 1–10 | `Import_CN` lags 1–4 | `(4, 4)` | 0.0001 | 0.001 |
| RES | 1–12 | `PCE(1)` + `Import_CN(1)` | `(14, 14, 14)` | 0.1 | 0.01 |

Mean validation RMSE của năm seed dùng để chọn combination là 984,147 (`std=10,191`). RMSE của dự báo ensemble trên validation là 982,290; hai số khác nhau đúng theo định nghĩa estimand ở mục 2.

## 7. Diebold–Mariano test

- DM test dùng squared-error loss trên 33 test observations và so sánh **ensemble mean-prediction**.
- Có 55 cặp model, 11 cặp có raw `p < 0.05`, nhưng chỉ 8 cặp còn ý nghĩa sau Holm correction.
- Cả 8 kết quả còn ý nghĩa đều là các model vượt qua `naive_seasonal`.
- Proposed vượt `naive_seasonal` có `DM=5.450`, raw `p=0.000005`, Holm-adjusted `p=0.000290`.
- Proposed và `ardl_ffnn_no_vmd` không khác biệt có ý nghĩa thống kê (`p=0.1619`, Holm-adjusted `p=1.0`).
- Proposed cũng không khác biệt có ý nghĩa với Sequential Screen (`p=0.8923`), Joint Select (`p=0.6293`), standalone ARDL (`p=0.8051`) hoặc VMD fixed (`p=0.6588`).

Vì test chỉ có 33 tháng, nên không nên diễn giải chênh lệch RMSE nhỏ giữa các model dẫn đầu như bằng chứng thống kê chắc chắn.

## 8. Những điểm đáng chú ý

1. **Proposed thắng validation nhưng không thắng test.** Kết quả này cho thấy quy tắc selection đang hợp lý hơn A6 nhưng vẫn có validation-to-test ranking shift. Model không VMD mới là model có test RMSE tốt nhất.
2. **Lợi thế rõ nhất của A7 là efficiency.** Accuracy của A4/A5/A7 rất gần nhau, trong khi two-stage A7 tiết kiệm khoảng 72% fits và 74–77% runtime so với full-grid.
3. **Không có bằng chứng DM rằng Proposed tốt hơn các model dẫn đầu.** Contribution nên nhấn mạnh cân bằng accuracy–runtime, shared mean-seed selection và độ ổn định, thay vì tuyên bố superiority tuyệt đối.
4. **VMD không bảo đảm cải thiện.** ARDL-FFNN không VMD đứng đầu test RMSE; standalone ARDL cũng đứng thứ tư và cạnh tranh với nhiều pipeline phức tạp hơn.
5. **Hard-config policy chưa đủ mạnh.** Nó nhanh hơn Proposed nhưng test RMSE mean kém hơn 13.48% và seed variance cao hơn nhiều; cần tiếp tục hiệu chỉnh policy trước khi dùng như comparator chính.
6. **VARNN có validation khá tốt nhưng generalize kém.** Theo primary estimand, nó rơi từ validation rank 5 xuống test rank 9. Seed-level Directional Accuracy chưa được lưu; ensemble DA phụ là 43.75%.
7. **Kiểm tra stationarity cần lưu ý `USD_VND`.** ADF/KPSS sau diff1 vẫn cho kết luận mixed/unclear và artifact đánh dấu “Likely I(2) = Yes”. Nên chạy sensitivity check với transform khác trước khi chốt paper.
8. **Thiếu seed-level Directional Accuracy của VARNN.** Ensemble DA vẫn tính được, nhưng `benchmark_seed_robustness_metrics.csv` có count bằng 0 cho DA của VARNN. Nếu báo uncertainty của DA theo seed trong paper thì cần bổ sung metric này.

## 9. Artifact chính

- [Primary per-seed mean/std benchmark](benchmark_seed_robustness_metrics.csv)
- [Secondary ensemble benchmark](benchmark_ensemble_metrics_comparison.csv)
- [Legacy primary-name artifact from this pre-change run](benchmark_metrics_comparison.csv)
- [Ablation metrics](ablation_metrics.csv)
- [Diebold–Mariano tests](diebold_mariano_tests.csv)
- [Efficiency summary](efficiency_summary.csv)
- [Experiment settings](paper_experiment_settings.csv)
- [Forecast date coverage](paper_forecast_date_coverage.csv)
- [Paper metrics summary](paper_metrics_summary.csv)
- [Paper ensemble metrics summary](paper_ensemble_metrics_summary.csv)
- [Run manifest](manifest.json)
