# Benchmark Results Summary

## 1. Thông tin run

| Thuộc tính | Giá trị |
|---|---|
| Run ID | `20260715T134317338124Z_33b7e495` |
| Pipeline | `benchmark` |
| Trạng thái | `success` |
| Dataset | `df_final.csv` |
| Target | `Export_US` |
| Exogenous variables | `PCE`, `US_Retail`, `USD_VND`, `Import_CN` |
| Số quan sát | 219 tháng |
| Khoảng dữ liệu | 01/2008–03/2026 |
| Forecast protocol | `rolling_one_step` |
| Paper seeds | `7`, `42`, `123`, `2024`, `2025` |
| VARNN | TensorFlow 2.21.0, đã chạy đủ 5 seeds |
| Comparator mới | `vmd_ardl_block_subset_mean_seed` |
| Comparator search | `two_stage` |

Manifest xác nhận run hoàn tất thành công, không có model bị skip. Run gồm chín
benchmark models, ablation A0–A5, Diebold–Mariano test với Holm correction, lag
stability audit và comparator variable-block mới.

## 2. Dữ liệu và cấu hình thí nghiệm

| Nhóm | Cấu hình |
|---|---|
| Split | 70% train, 15% validation, 15% test |
| Transform | Log tự nhiên; stationarity decision chỉ dùng train |
| VMD | 3 modes + residual, `alpha=2000`, self-contained backend |
| ARDL | AIC, target lag tối đa 12, exogenous order tối đa 6 |
| Proposed lag generation | Staged, tối đa 16 specs/component, top 3 exogenous orders |
| Proposed rolling screen | 3 expanding folds × 12 tháng, LR `0.01` và `0.001` |
| Proposed lag shortlist | Top 4; luôn giữ raw winner và one-SE winner |
| FFNN architecture | 1–3 hidden layers, 3 width multipliers |
| FFNN alpha | `1e-4`, `1e-3`, `1e-2`, `1e-1` |
| Final FFNN learning-rate grid | Chỉ `0.001` trong run này |
| Comparator fast screen | 15 specs/component, `alpha=0.1`, LR `0.001`, 150 iterations |
| Comparator fast-screen shortlist | Top 4 specs/component theo mean component validation RMSE qua 5 seeds |
| Comparator final objective | Mean reconstructed-level validation RMSE qua 5 seeds |

### Outer split

| Split | Số quan sát | Bắt đầu | Kết thúc |
|---|---:|---|---|
| Train | 153 | 01/2008 | 09/2020 |
| Validation | 33 | 10/2020 | 06/2023 |
| Test | 33 | 07/2023 | 03/2026 |

Mọi benchmark dùng cùng 33 validation dates và 33 test dates. Test không tham
gia chọn lag structure hoặc hyperparameters.

> **Lưu ý quan trọng:** `learning_rate_grid` của final FFNN search là
> `(0.001,)`, không phải `(0.01, 0.001)`. Vì vậy các final winners dùng LR
> `0.001` do đây là lựa chọn duy nhất; run này không chứng minh `0.001` tốt hơn
> `0.01` ở final grid. Riêng rolling lag screen của Proposed/A3/no-VMD vẫn so
> sánh cả hai learning rates.

## 3. Benchmark test results

Neural models được báo cáo bằng trung bình ± độ lệch chuẩn qua năm paper seeds.
Các metrics trong bảng đều ở reconstructed level scale.

| Hạng RMSE | Model | Test RMSE | Test MAE | MAPE | DA |
|---:|---|---:|---:|---:|---:|
| 1 | VMD–AIC block-subset–FFNN two-stage | **1,024,414 ± 51,583** | **767,399 ± 46,329** | **7.246%** | **71.25%** |
| 2 | ARDL–FFNN không VMD | 1,047,124 ± 55,554 | 811,936 ± 31,059 | 7.520% | 62.50% |
| 3 | Standalone ARDL | 1,084,833 | 808,376 | 7.535% | 68.75% |
| 4 | Proposed VMD–ARDL–FFNN | 1,115,364 ± 33,834 | 808,472 ± 42,628 | 7.791% | 63.75% |
| 5 | VMD–FFNN fixed lag | 1,250,067 ± 133,709 | 875,030 ± 104,394 | 8.473% | 68.13% |
| 6 | FFNN fixed lag, không VMD | 1,300,482 ± 142,766 | 925,681 ± 88,573 | 8.989% | 64.38% |
| 7 | VARNN | 1,488,365 ± 168,178 | 1,127,312 ± 183,953 | 10.610% | — |
| 8 | Naive last observation | 1,611,731 | 1,092,537 | 10.436% | 0.00% |
| 9 | Naive seasonal-12 | 2,383,548 | 2,101,624 | 18.093% | 53.13% |

### Nhận xét test

- Comparator two-stage đứng đầu theo cả RMSE, MAE, MAPE và DA trong các model
  có đủ metric.
- So với ARDL–FFNN không VMD, comparator giảm 2.17% RMSE, 5.49% MAE và 3.64%
  MAPE; DA cao hơn 8.75 điểm phần trăm.
- So với Proposed, comparator giảm 8.15% RMSE, 5.08% MAE và 6.99% MAPE; DA cao
  hơn 7.50 điểm phần trăm.
- So với Standalone ARDL, comparator giảm 5.57% RMSE và 5.07% MAE.
- Độ lệch chuẩn RMSE của comparator là 51,583: gần ARDL–FFNN không VMD
  (55,554), nhưng cao hơn Proposed (33,834). Vì vậy không nên gọi comparator là
  model ổn định nhất qua seeds trong run này.

Đây là xếp hạng point estimate trên test, không phải bằng chứng về ưu thế thống
kê; DM test ở phần 8 cho thấy kết luận thận trọng hơn.

## 4. Benchmark validation results

| Hạng RMSE | Model | Validation RMSE | Validation MAE | MAPE | DA |
|---:|---|---:|---:|---:|---:|
| 1 | VMD–AIC block-subset–FFNN two-stage | **983,950 ± 52,353** | 750,147 | 9.567% | 72.50% |
| 2 | ARDL–FFNN không VMD | 992,453 ± 26,170 | 760,141 | 9.743% | 65.63% |
| 3 | Proposed VMD–ARDL–FFNN | 996,417 ± 35,708 | 753,595 | 9.688% | 66.88% |
| 4 | FFNN fixed lag, không VMD | 1,018,896 ± 40,879 | 728,478 | 9.345% | 72.50% |
| 5 | VMD–FFNN fixed lag | 1,031,733 ± 36,364 | 725,007 | 9.325% | 70.00% |
| 6 | VARNN | 1,036,693 ± 28,273 | **718,295** | **9.266%** | — |
| 7 | Naive last observation | 1,099,421 | 782,292 | 10.016% | 0.00% |
| 8 | Standalone ARDL | 1,209,736 | 917,699 | 11.740% | 65.63% |
| 9 | Naive seasonal-12 | 1,944,329 | 1,722,014 | 21.090% | 53.13% |

Comparator đứng đầu đúng metric được dùng để selection là mean validation RMSE.
Nó không đứng đầu validation MAE/MAPE; VARNN có MAE và MAPE thấp nhất nhưng
Directional Accuracy chưa được xuất.

## 5. Kết quả workflow AIC variable-block two-stage

Workflow chạy riêng cho `VMD1`, `VMD2`, `VMD3` và `RES`:

1. `ardl_select_order` chọn target lags trên train.
2. Khóa target lags.
3. Fit ARDL cho từng biến ngoại sinh với `q=1..6`; chọn order AIC tốt nhất và
   tạo block lag `1..q`.
4. Sinh 15 tập con variable blocks khác rỗng cho mỗi component.
5. Fast-screen cả 15 specs bằng model một hidden layer, `alpha=0.1`, LR
   `0.001`, qua đủ năm reporting seeds.
6. Giữ top 4 specs/component theo mean component validation RMSE qua seeds.
7. Chạy final FFNN grid trên 4 specs này; xếp hạng các seed-independent
   signatures bằng mean component validation score.
8. Giữ top 3 signatures/component và đánh giá `3^4 = 81` combinations sau khi
   reconstruct validation forecast.
9. Chọn một combination chung bằng mean reconstructed-level validation RMSE qua
   năm seeds, refit theo từng seed rồi mới đánh giá test.

### 5.1 AIC-selected variable-block orders

| Component | Target lags | PCE | US_Retail | USD_VND | Import_CN |
|---|---|---:|---:|---:|---:|
| VMD1 | 1–8 | 5 | 4 | 5 | 5 |
| VMD2 | 1–11 | 5 | 5 | 2 | 4 |
| VMD3 | 1–10 | 5 | 5 | 1 | 4 |
| RES | 1–12 | 1 | 4 | 1 | 1 |

Mỗi số `q` tạo block lags `1..q`. AIC chỉ xác định nội dung từng variable block;
FFNN stages sau đó quyết định block nào được giữ hay bỏ.

### 5.2 Top 4 specs sau fast screen

| Component | #1 | #2 | #3 | #4 |
|---|---|---|---|---|
| VMD1 | USD_VND + Import_CN | PCE + Import_CN | PCE + USD_VND + Import_CN | PCE |
| VMD2 | PCE + US_Retail + Import_CN | Import_CN | PCE + US_Retail | USD_VND |
| VMD3 | Import_CN | USD_VND + Import_CN | PCE + US_Retail + USD_VND | USD_VND |
| RES | PCE + USD_VND | PCE + Import_CN | US_Retail | PCE |

Fast screen chỉ dùng để giảm search space. Nó không trực tiếp chọn final block
combination vì final objective được tính sau reconstruction.

### 5.3 Final shared combination

| Component | Fast-screen rank của lag spec | Component-signature rank | Exogenous blocks được giữ | Input dim | Hidden layers | Alpha | LR |
|---|---:|---:|---|---:|---|---:|---:|
| VMD1 | 4 | 2 | `PCE(1–5)` | 13 | `(3,3,3)` | 0.1 | 0.001 |
| VMD2 | 4 | 1 | `USD_VND(1–2)` | 13 | `(3,3)` | 0.1 | 0.001 |
| VMD3 | 1 | 2 | `Import_CN(1–4)` | 14 | `(4,4)` | 0.0001 | 0.001 |
| RES | 4 | 2 | `PCE(1)` | 13 | `(13,)` | 0.0001 | 0.001 |

Final combination là `combo_id=31`, tổng input dimension bằng 53. Ba trong bốn
lag specs cuối cùng chỉ đứng hạng 4 ở fast screen. Điều này không phải mâu thuẫn:
fast screen dùng lightweight component-scale model, còn final winner dùng full
grid và joint reconstructed-level objective.

Selected combination có:

| Metric lựa chọn | Mean qua seeds | Std qua seeds |
|---|---:|---:|
| Validation level RMSE | **983,950** | 52,353 |
| Validation level MAE | 750,147 | 54,711 |
| Validation level MAPE | 9.567% | 0.732% |
| Validation DA | 72.50% | 2.61 pp |

Combination hạng 2 có validation RMSE 984,102, chỉ kém hạng 1 khoảng 151
(0.015%). Vì khoảng cách rất nhỏ so với variation giữa seeds, exact combination
winner cần được xem là nhạy với seed/config; báo cáo nên giữ cả ranking artifact
thay vì chỉ nêu một winner tuyệt đối.

### 5.4 Kết quả comparator theo seed

| Seed | Validation RMSE | Test RMSE | Test MAE | Test MAPE | Test DA |
|---:|---:|---:|---:|---:|---:|
| 7 | **915,256** | **961,095** | **725,557** | **6.839%** | 68.75% |
| 42 | 952,716 | 994,478 | 747,559 | 7.066% | **78.13%** |
| 123 | 982,987 | 1,015,930 | 740,185 | 7.014% | 75.00% |
| 2024 | 1,043,840 | 1,059,626 | 782,274 | 7.257% | 75.00% |
| 2025 | 1,024,953 | 1,090,941 | 841,421 | 8.055% | 59.38% |

Không seed nào được chọn làm reporting winner. Một lag/hyperparameter combination
chung đã được khóa trước, sau đó refit và báo cáo variation trên cả năm seeds.

### 5.5 Search-space audit

| Giai đoạn | Công thức | Số lượng |
|---|---|---:|
| Variable-block specs ban đầu | 4 components × 15 | 60 specs |
| Fast-screen fits | 60 specs × 5 seeds | 300 |
| Specs vào full grid | 4 components × top 4 | 16 specs |
| Final hyperparameter signatures/spec | 9 architectures × 4 alpha × 1 LR | 36 |
| Full-grid fits | 16 specs × 36 × 5 seeds | 2,880 |
| Evaluated fits ghi trong efficiency table | 300 + 2,880 | **3,180** |
| Mean-seed component signatures | 4 × 4 × 36 | 576 |
| Reconstructed combinations | top `3^4` | 81 |
| Reconstructed seed scores | 81 × 5 | 405 |
| Final component refits | 4 × 5 | 20, không tính trong `evaluated_ffnn_fits` |

Nếu chạy full grid hiện tại trên cả 15 specs/component thì cần 10,800 candidate
fits. Two-stage giảm số evaluated candidate fits xuống 3,180, tương đương giảm
70.56%. So sánh này dùng cùng grid một learning rate của run hiện tại.

## 6. Comparator mới so với Proposed

| Tiêu chí | Comparator two-stage | Proposed | Thay đổi của comparator |
|---|---:|---:|---:|
| Validation RMSE | 983,950 | 996,417 | **−1.25%** |
| Test RMSE | 1,024,414 | 1,115,364 | **−8.15%** |
| Test MAE | 767,399 | 808,472 | **−5.08%** |
| Test MAPE | 7.246% | 7.791% | **−6.99%** |
| Test DA | 71.25% | 63.75% | **+7.50 pp** |
| RMSE std qua seeds | 51,583 | 33,834 | comparator cao hơn 52.5% |
| Total runtime cho 5 seeds | 437.00 giây | 721.44 giây | comparator thấp hơn 39.4% |
| Evaluated candidate fits | 3,180 | 4,800 | comparator thấp hơn 33.8% |

Hai workflow hiện dùng cùng final metric scale nhưng khác seed aggregation:

- Comparator chọn **một shared combination** bằng mean reconstructed-level
  validation RMSE qua năm seeds.
- Proposed chọn reconstructed-level validation winner **riêng cho từng seed**
  sau rolling shortlist.

Vì vậy đây là so sánh hai workflow hoàn chỉnh, không phải ablation chỉ thay một
yếu tố. Comparator có point estimates tốt hơn trong run này nhưng variation qua
seeds cao hơn Proposed.

## 7. Ablation A0–A5

Comparator block-subset không thay thế A4; Proposed vẫn là A4.

| Hạng | ID | Cấu hình | Test RMSE | MAE | MAPE | DA |
|---:|---|---|---:|---:|---:|---:|
| 1 | A5 | VMD + joint target/exog SelectOrder | **1,082,187 ± 44,304** | 809,259 | **7.549%** | **70.00%** |
| 2 | A4 | Proposed sequential selection | 1,115,364 ± 33,834 | **808,472** | 7.791% | 63.75% |
| 3 | A2 | VMD + target selection + fixed exog | 1,139,928 ± 71,770 | 816,924 | 7.832% | 67.50% |
| 4 | A1 | VMD + fixed target/exog | 1,250,067 ± 133,709 | 875,030 | 8.473% | 68.13% |
| 5 | A0 | FFNN fixed lags, không VMD | 1,300,482 ± 142,766 | 925,681 | 8.989% | 64.38% |
| 6 | A3 | VMD + fixed target + ARDL exog | 1,336,628 ± 42,529 | 938,735 | 9.034% | **70.00%** |

Nhận xét:

- A1 giảm 3.88% RMSE so với A0.
- A2 giảm 8.81% RMSE so với A1.
- A4 giảm 2.15% RMSE và 1.03% MAE so với A2.
- A5 giảm thêm 2.97% RMSE so với A4, nhưng MAE cao hơn nhẹ 0.10%.
- A3 có validation RMSE rất tốt (959,186) nhưng test RMSE tệ nhất (1,336,628),
  là dấu hiệu lựa chọn cấu trúc không tổng quát hóa tốt khi target lag bị fixed.

## 8. Diebold–Mariano tests

Có `C(9,2) = 36` pairwise tests trên 33 common test dates. Mười một cặp có raw
`p < 0.05`; sau Holm correction còn bảy cặp có ý nghĩa. Cả bảy đều là một model
tốt hơn seasonal naive.

| Model tốt hơn seasonal naive | Holm-adjusted p-value |
|---|---:|
| Comparator block-subset two-stage | 0.000037 |
| ARDL–FFNN không VMD | 0.000060 |
| Standalone ARDL | 0.000283 |
| Proposed | 0.000565 |
| VMD–FFNN fixed | 0.005023 |
| FFNN fixed không VMD | 0.013619 |
| VARNN | 0.033224 |

### Comparator so với các đối thủ chính

| Đối thủ | Raw p-value | Holm p-value | Kết luận |
|---|---:|---:|---|
| ARDL–FFNN không VMD | 0.928 | 1.000 | Không khác biệt có ý nghĩa |
| Standalone ARDL | 0.387 | 1.000 | Không khác biệt có ý nghĩa |
| Proposed | 0.397 | 1.000 | Không khác biệt có ý nghĩa |
| VMD–FFNN fixed | 0.266 | 1.000 | Không khác biệt có ý nghĩa |
| FFNN fixed không VMD | 0.195 | 1.000 | Không khác biệt có ý nghĩa |
| VARNN | 0.075 | 1.000 | Không khác biệt có ý nghĩa |
| Naive last observation | 0.097 | 1.000 | Không khác biệt có ý nghĩa |
| Naive seasonal-12 | 0.000001 | 0.000037 | Comparator tốt hơn có ý nghĩa |

Do đó, comparator dẫn đầu point metrics nhưng chưa có bằng chứng DM rằng forecast
errors khác Proposed, ARDL–FFNN không VMD hoặc Standalone ARDL một cách có ý
nghĩa thống kê.

## 9. Rolling lag-screen và learning rate

Rolling one-standard-error screen chỉ áp dụng cho các staged pipelines, không áp
dụng cho comparator block-subset.

| Pipeline | Spec summaries | Shortlisted | Seed-component groups | Raw và one-SE khác nhau |
|---|---:|---:|---:|---:|
| ARDL–FFNN không VMD | 80 | 20 | 5 | 3/5 |
| Proposed | 320 | 80 | 20 | 10/20 |
| A3 fixed-target/exog-screen | 320 | 80 | 20 | 9/20 |

Trong Proposed rolling screen, LR `0.001` thắng 195/320 spec comparisons và LR
`0.01` thắng 125/320. Điều này cho thấy cả hai learning rates vẫn có ích ở fast
screen. Tuy nhiên final grid của run chỉ chứa `0.001`, nên mọi final selected
model đều dùng `0.001` theo cấu hình chứ không qua một cuộc so sánh hai LR.

## 10. Runtime và search cost

| Pipeline | Số runs | Mean runtime/run | Total runtime | Tổng evaluated fits |
|---|---:|---:|---:|---:|
| Proposed | 5 | 144.29 giây | **721.44 giây** | 4,800 |
| A3 fixed-target/exog-screen | 5 | 101.38 giây | 506.89 giây | 4,800 |
| Comparator block-subset two-stage | 1 shared-seed run | 437.00 giây | 437.00 giây | 3,180 |
| A2 target-select/fixed-exog | 5 | 42.15 giây | 210.77 giây | 720 |
| A5 joint-select | 5 | 41.77 giây | 208.83 giây | 720 |
| ARDL–FFNN không VMD | 5 | 26.81 giây | 134.05 giây | 1,200 |
| VMD–FFNN fixed | 5 | 20.20 giây | 101.00 giây | 720 |
| VARNN | 5 | 9.07 giây | 45.37 giây | 5 |
| FFNN fixed không VMD | 5 | 4.79 giây | 23.97 giây | 180 |
| Standalone ARDL | 1 | 4.11 giây | 4.11 giây | 0 |

Tổng model runtime ghi trong efficiency table là 2,393.45 giây, tương đương
39.89 phút. Comparator không còn là pipeline tốn tổng thời gian nhất; Proposed
tốn khoảng 12.02 phút, comparator khoảng 7.28 phút.

Runtime breakdown nội bộ của comparator:

| Giai đoạn | Thời gian |
|---|---:|
| ARDL selection | 15.84 giây |
| VMD cache | 1.19 giây |
| Fast screen + full FFNN search | 415.56 giây |
| Refit + forecast | 3.60 giây |
| Internal total | 436.88 giây |

Search vẫn chiếm khoảng 95.1% comparator runtime, nhưng two-stage đã tránh full
grid trên 11/15 lag specs ở mỗi component.

## 11. Stationarity screening

| Biến | Level | First difference | Transform | Cảnh báo I(2) |
|---|---|---|---|---|
| Export_US | Non-stationary | Stationary | `diff1` | Không |
| PCE | Non-stationary | Stationary | `diff1` | Không |
| US_Retail | Non-stationary | Stationary | `diff1` | Không |
| Import_CN | Non-stationary | Stationary | `diff1` | Không |
| USD_VND | Mixed/unclear | Mixed/unclear | `diff1` | **Có** |

`USD_VND` vẫn cần robustness check. ADF và KPSS chưa đồng thuận sau first
difference, nhưng pipeline hiện vẫn dùng `diff1`.

## 12. Forecast coverage và checkpoint

| Split | Models | Dates/model | Khoảng thời gian | Cùng dates |
|---|---:|---:|---|---|
| Validation | 9 | 33 | 10/2020–06/2023 | Có |
| Test | 9 | 33 | 07/2023–03/2026 | Có |

Điều kiện common-date cho benchmark và DM test được đáp ứng.

Run này dùng bản `NumpyFFNNRegressor` đã restore weights tại training loss tốt
nhất, nên không còn lỗi trả về weights của iteration cuối. Cần phân biệt rằng
checkpoint nội bộ hiện vẫn dựa trên **training loss**, không phải validation loss
theo từng epoch. Lag spec/hyperparameter candidates được đánh giá bằng outer
validation sau khi fit.

## 13. Kết luận chính

| Mức độ | Kết luận |
|---|---|
| Chính | Comparator two-stage đứng hạng 1 test RMSE, MAE, MAPE và DA trong run này |
| Chính | Comparator giảm 8.15% test RMSE so với Proposed |
| Hiệu quả | Two-stage dùng 3,180 evaluated fits thay vì 10,800 fits của current-grid exhaustive search |
| Objective | Comparator đã chọn đúng bằng mean reconstructed-level validation RMSE qua reporting seeds |
| Tích cực | Joint reconstructed selection chọn lag specs khác component fast-screen winner ở 3/4 components |
| Thận trọng | Comparator không khác Proposed hoặc các đối thủ chính có ý nghĩa theo DM/Holm |
| Thận trọng | Top-2 reconstructed combinations chỉ cách nhau 151 RMSE; exact winner nhạy với seed/config |
| Thận trọng | Comparator có RMSE variation qua seeds cao hơn Proposed |
| Cấu hình | Final LR grid chỉ có `0.001`; không thể rút kết luận so sánh LR cuối cùng |
| Cần sửa | VARNN vẫn thiếu Directional Accuracy |
| Cần kiểm tra | `USD_VND` có dấu hiệu có thể là I(2) |

Kết luận phù hợp cho paper là: workflow block-subset two-stage đạt point forecast
tốt nhất và giảm mạnh search cost trong run này, nhưng chưa chứng minh được ưu
thế thống kê so với Proposed/ARDL alternatives. Không nên dùng từ “significantly
outperforms” ngoài so sánh với seasonal naive.

## 14. Artifact chính

| Artifact | Số dòng dữ liệu | Nội dung |
|---|---:|---|
| [`benchmark_metrics_comparison.csv`](benchmark_metrics_comparison.csv) | 19 | Aggregate benchmark metrics; gồm VARNN train row |
| [`benchmark_forecasts_comparison.csv`](benchmark_forecasts_comparison.csv) | 2,928 | Forecasts theo model/seed |
| [`ablation_metrics.csv`](ablation_metrics.csv) | 12 | A0–A5 validation/test metrics |
| [`diebold_mariano_tests.csv`](diebold_mariano_tests.csv) | 36 | Pairwise DM, bootstrap CI và Holm p-values |
| [`efficiency_summary.csv`](efficiency_summary.csv) | 44 | Runtime và evaluated FFNN fits |
| [`lag_screen_fold_metrics.csv`](lag_screen_fold_metrics.csv) | 4,320 | Staged train-only rolling fold metrics |
| [`lag_screen_stability.csv`](lag_screen_stability.csv) | 1,440 | Mean/std/SE, one-SE và shortlist audit |
| [`lag_selection_summary.csv`](lag_selection_summary.csv) | 130 | Final lags/hyperparameters theo pipeline và seed |
| [`lag_selection_frequency.csv`](lag_selection_frequency.csv) | 426 | Lag-selection frequency |
| [`paper_seed_metrics.csv`](paper_seed_metrics.csv) | 101 | Metrics chi tiết theo seed |
| [`paper_experiment_settings.csv`](paper_experiment_settings.csv) | 12 | Protocol của từng pipeline |
| [`paper_stationarity_screen.csv`](paper_stationarity_screen.csv) | 5 | Train-only stationarity decisions |
| [`paper_forecast_date_coverage.csv`](paper_forecast_date_coverage.csv) | 19 | Common-date audit |
| [`manifest.json`](manifest.json) | — | Full config, dependency versions và run status |

### Artifact riêng của comparator two-stage

| Artifact | Số dòng dữ liệu | Nội dung |
|---|---:|---|
| [`ffnn_block_subset_fast_screen_metrics.csv`](vmd_ardl_block_subset_mean_seed/ffnn_block_subset_fast_screen_metrics.csv) | 300 | 15 specs × 4 components × 5 seeds |
| [`ffnn_block_subset_fast_screen_summary.csv`](vmd_ardl_block_subset_mean_seed/ffnn_block_subset_fast_screen_summary.csv) | 60 | Mean-seed fast ranks và top-4 flags |
| [`ffnn_block_subset_component_search.csv`](vmd_ardl_block_subset_mean_seed/ffnn_block_subset_component_search.csv) | 2,880 | Full-grid candidates trên top-4 specs |
| [`ffnn_block_subset_component_mean_seed_ranking.csv`](vmd_ardl_block_subset_mean_seed/ffnn_block_subset_component_mean_seed_ranking.csv) | 576 | Mean-seed component signatures và top-3 flags |
| [`ffnn_block_subset_reconstructed_seed_scores.csv`](vmd_ardl_block_subset_mean_seed/ffnn_block_subset_reconstructed_seed_scores.csv) | 405 | 81 combinations × 5 reconstructed seed scores |
| [`ffnn_block_subset_mean_seed_selection.csv`](vmd_ardl_block_subset_mean_seed/ffnn_block_subset_mean_seed_selection.csv) | 81 | Shared reconstructed-level combination ranking |
| [`best_component_models_block_subset_mean_seed.csv`](vmd_ardl_block_subset_mean_seed/best_component_models_block_subset_mean_seed.csv) | 20 | 4 selected components × 5 seeds |
| [`ardl_variable_blocks_train_only.csv`](vmd_ardl_block_subset_mean_seed/ardl_variable_blocks_train_only.csv) | 204 | Target lags, AIC variable orders và 15 subsets/component |
| [`final_reconstructed_forecasts_block_subset_mean_seed.csv`](vmd_ardl_block_subset_mean_seed/final_reconstructed_forecasts_block_subset_mean_seed.csv) | 330 | Validation/test forecasts qua 5 seeds |
| [`final_reconstructed_metrics_block_subset_mean_seed.csv`](vmd_ardl_block_subset_mean_seed/final_reconstructed_metrics_block_subset_mean_seed.csv) | 20 | Transformed và level metrics theo seed/split |

Run còn có bảy paper figures về test forecast, error distribution, seed
uncertainty, accuracy–runtime, VMD decomposition và lag-selection heatmaps.
