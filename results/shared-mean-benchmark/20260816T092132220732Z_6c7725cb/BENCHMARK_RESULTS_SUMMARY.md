# Tổng hợp kết quả `shared-mean-benchmark`

## 1. Thông tin run

| Thuộc tính | Giá trị |
|---|---|
| Run ID | `20260816T092132220732Z_6c7725cb` |
| Trạng thái | `success` |
| Thời điểm hoàn tất | 2026-08-16 10:01:48 UTC (17:01:48 UTC+7) |
| Dữ liệu | `df_final.csv` |
| Target | `Export_US` |
| Exogenous variables | `PCE`, `US_Retail`, `USD_VND`, `Import_CN` |
| Tần suất | Tháng (`MS`) |
| Số quan sát | 219, từ 2008-01 đến 2026-03 |
| Forecast protocol | `rolling_one_step` |
| VMD | 3 modes + residual, self-contained backend, `alpha=2000` |
| ARDL bound | Target lag tối đa 12; exogenous order tối đa 6; AIC; causal |
| Reporting seeds | 7, 42, 123, 2024, 2025 |
| FFNN learning rates | 0.01 và 0.001 |
| Shared selector | Mean validation RMSE qua đủ 5 reporting seeds |
| Tổng runtime của bốn model | 2,413.9 giây, xấp xỉ 40.23 phút |
| Tổng evaluated search fits | 17,700; không tính final refit |

Phân chia dữ liệu theo thời gian, không shuffle:

| Split | Số quan sát | Khoảng thời gian |
|---|---:|---|
| Train | 153 | 2008-01 đến 2020-09 |
| Validation | 33 | 2020-10 đến 2023-06 |
| Test | 33 | 2023-07 đến 2026-03 |

Cả bốn model có cùng validation/test dates và cùng actual values. Tất cả năm
biến được stationarity screen trên train và dùng sai phân bậc một sau log
transform.

## 2. Bốn workflow được benchmark

| Pipeline | Vai trò | Lag và model-selection protocol |
|---|---|---|
| `ardl_ffnn_no_vmd` | No-VMD | Target-first ARDL, exogenous rolling screen và shared mean-seed FFNN selection trên level validation RMSE |
| `vmd_ardl_block_subset_proposed` | Proposed hiện tại | AIC variable blocks, train-only inner rolling fast-screen, full FFNN grid và reconstructed shared winner |
| `vmd_ardl_sequential_screen` | Proposed cũ | Target-first ARDL, exogenous rolling screen và reconstructed shared mean-seed winner |
| `vmd_ardl_joint_select` | Joint comparator | `ardl_select_order` chọn target/exogenous lags đồng thời, sau đó reconstructed shared mean-seed FFNN selection |

Mỗi candidate signature gồm lag specification, architecture, alpha, activation
và learning rate. Candidate được fit ở đủ năm seeds và được xếp bằng mean
validation RMSE. Seed không thuộc model signature và pipeline không chọn một
“best seed”. Sau khi khóa một shared winner, cùng signature đó được refit độc
lập ở từng seed trên train + validation rồi mới đánh giá test.

## 3. Kết quả chính: mean ± std qua seed

Đây là primary estimand trong `benchmark_metrics_comparison.csv`: metric được
tính riêng cho từng seed, sau đó mới báo mean, standard deviation, median và
count. Bảng dưới xếp theo test RMSE mean.

| Test rank | Pipeline | Val RMSE mean ± std (val rank) | Test RMSE mean ± std | Test RMSE median | Test MAE mean | Test MAPE mean | Test DA mean | Seed count |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `ardl_ffnn_no_vmd` | 1,011,849 ± 11,987 (3) | **1,008,860 ± 8,419** | 1,011,828 | **768,415** | **7.182%** | 62.500% | 5 |
| 2 | `vmd_ardl_sequential_screen` | 1,014,099 ± 3,740 (4) | 1,073,221 ± 10,315 | 1,073,784 | 770,741 | 7.532% | 69.375% | 5 |
| 3 | `vmd_ardl_block_subset_proposed` | 1,011,151 ± 11,178 (2) | 1,080,730 ± 9,333 | 1,084,574 | 781,405 | 7.566% | 66.875% | 5 |
| 4 | `vmd_ardl_joint_select` | **1,004,778 ± 7,141 (1)** | 1,088,381 ± 12,980 | 1,085,394 | 771,660 | 7.499% | **70.625%** | 5 |

### So sánh trực tiếp block-Proposed

- Block-Proposed có test RMSE mean cao hơn no-VMD 7.12% và cao hơn Sequential
  Screen cũ 0.70%.
- Block-Proposed tốt hơn Joint Select 0.70% về test RMSE mean.
- Block-Proposed đứng thứ hai trên validation nhưng rơi xuống thứ ba trên test.
- No-VMD thắng đồng thời test RMSE, MAE và MAPE; Joint Select thắng Directional
  Accuracy.
- Độ lệch chuẩn test RMSE của block-Proposed là 9,333, tương đương khoảng 0.86%
  mean. Nó ổn định thứ hai sau no-VMD, nhưng stability không bù được khoảng cách
  accuracy.
- Thứ hạng validation gần như đảo ngược trên test: Joint thắng validation nhưng
  đứng cuối test, còn no-VMD từ hạng 3 validation lên hạng 1 test. Đây là
  validation-to-test ranking shift đáng chú ý.

## 4. Kết quả phụ: ensemble mean-prediction

Bảng phụ trung bình prediction của năm seed tại từng timestamp trước khi tính
metric. Nó mô tả deployment ensemble năm model, không phải expected performance
của một single-seed model.

| Test rank | Pipeline | Test RMSE | Test MAE | Test MAPE | Test DA | Ensemble members |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `ardl_ffnn_no_vmd` | **1,006,994** | 768,415 | **7.182%** | 62.500% | 5 |
| 2 | `vmd_ardl_sequential_screen` | 1,070,080 | **767,654** | 7.509% | **68.750%** | 5 |
| 3 | `vmd_ardl_block_subset_proposed` | 1,079,522 | 780,144 | 7.555% | 65.625% | 5 |
| 4 | `vmd_ardl_joint_select` | 1,086,763 | 769,909 | 7.483% | **68.750%** | 5 |

Ensemble không thay đổi thứ hạng RMSE. Block-Proposed giảm từ mean single-model
RMSE 1,080,730 xuống ensemble RMSE 1,079,522, tức chỉ khoảng 0.11%; lợi ích
ensemble nhỏ vì các seed vốn đã khá ổn định.

## 5. Audit inner validation của block-Proposed

Block-Proposed mới không dùng outer validation để giữ top 4 block specs. Mỗi
component có 15 non-empty subsets từ bốn AIC variable blocks. Lightweight screen
dùng `alpha=0.1`, learning rate `0.001`, ba expanding folds và đủ năm seeds.

| Fold | Inner-train period | Inner-validation period | Validation size |
|---:|---|---|---:|
| 1 | 2008-02 đến 2017-09 | 2017-10 đến 2018-09 | 12 |
| 2 | 2008-02 đến 2018-09 | 2018-10 đến 2019-09 | 12 |
| 3 | 2008-02 đến 2019-09 | 2019-10 đến 2020-09 | 12 |

Audit kết quả:

- `4 components × 15 specs × 3 folds × 5 seeds = 900` fast-screen fits;
- đủ ba folds và năm seeds cho mọi spec;
- không có fast-screen fit nào ghi lỗi;
- giữ đúng top 4 specs cho từng `VMD1`, `VMD2`, `VMD3`, `RES`;
- full grid sau shortlist có 5,760 fits;
- tổng evaluated search fits của Proposed là 6,660;
- outer validation chỉ được dùng sau khi shortlist đã khóa, cho full FFNN grid
  và reconstructed combination selection.

## 6. Cấu hình block-Proposed được chọn

Reconstruction search giữ top 3 signatures/component và đánh giá tối đa
`3^4 = 81` combinations. Combination thắng là `combo_id=54`, tổng input
dimension 70, với mean reconstructed-level validation RMSE 1,011,151 và
standard deviation 11,178 qua năm seeds.

| Component | Target lags | Exogenous blocks | Hidden layers | Alpha | LR |
|---|---|---|---:|---:|---:|
| VMD1 | 1–8 | `PCE(1–5)` + `USD_VND(1–5)` + `Import_CN(1–5)` | `(23, 23)` | 0.001 | 0.001 |
| VMD2 | 1–11 | `US_Retail(1–5)` + `USD_VND(1–2)` | `(4, 4, 4)` | 0.1 | 0.001 |
| VMD3 | 1–10 | `USD_VND(1)` | `(3, 3, 3)` | 0.1 | 0.001 |
| RES | 1–12 | `PCE(1)` + `US_Retail(1–4)` + `Import_CN(1)` | `(18,)` | 0.1 | 0.01 |

Mọi seed dùng cùng lag blocks và cùng hyperparameters phía trên; chỉ weights do
random initialization khác nhau.

## 7. Runtime và search efficiency

`evaluated_ffnn_fits` đếm fits trong screening/search, không tính final refit.

| Pipeline | Runtime (s) | Runtime (phút) | Evaluated search fits | Components |
|---|---:|---:|---:|---:|
| `ardl_ffnn_no_vmd` | 202.2 | 3.37 | 1,920 | 1 |
| `vmd_ardl_joint_select` | 236.4 | 3.94 | **1,440** | 4 |
| `vmd_ardl_block_subset_proposed` | 913.1 | 15.22 | 6,660 | 4 |
| `vmd_ardl_sequential_screen` | 1,062.2 | 17.70 | 7,680 | 4 |

So với Proposed cũ, block-Proposed:

- giảm 1,020 search fits, tương đương 13.28%;
- giảm runtime khoảng 149.1 giây, tương đương 14.04%;
- nhưng test RMSE mean cao hơn 0.70%.

Joint Select rẻ nhất theo search fits và nhanh gần bằng no-VMD, nhưng test RMSE
chỉ kém block-Proposed 0.70%. Chênh lệch nhỏ này cũng không có ý nghĩa thống kê
theo DM test.

## 8. Diebold–Mariano test

DM test dùng squared-error loss trên 33 test observations và ensemble-mean
prediction. Không cặp nào có raw `p < 0.05`; sau Holm correction, cả sáu cặp đều
có adjusted p-value bằng 1.0.

| Cặp so sánh | Raw p-value | Holm-adjusted p-value |
|---|---:|---:|
| No-VMD vs Block-Proposed | 0.2521 | 1.0000 |
| No-VMD vs Joint Select | 0.2654 | 1.0000 |
| No-VMD vs Sequential | 0.3602 | 1.0000 |
| Block-Proposed vs Joint Select | 0.5941 | 1.0000 |
| Block-Proposed vs Sequential | 0.6473 | 1.0000 |
| Joint Select vs Sequential | 0.4526 | 1.0000 |

Vì vậy không thể tuyên bố model dẫn đầu vượt trội thống kê so với ba model còn
lại từ run này. Test chỉ có 33 tháng nên confidence intervals của paired loss
difference đều đi qua 0.

## 9. Ảnh hưởng của việc chuyển block fast-screen sang inner folds

Để tham khảo, run `block-proposed-benchmark/20260723T051655172652Z_6c7725cb`
dùng cùng dataset, config hash và reporting seeds nhưng block fast-screen trên
outer validation. So sánh này là cross-run diagnostic, không phải một DM test.

| Chỉ tiêu block-Proposed | Outer-validation screen cũ | Inner-fold screen hiện tại | Thay đổi |
|---|---:|---:|---:|
| Validation RMSE mean | 984,147 | 1,011,151 | +2.74% |
| Test RMSE mean | 1,067,400 | 1,080,730 | +1.25% |
| Search fits | 6,060 | 6,660 | +9.90% |
| Runtime | 839.8 s | 913.1 s | +8.73% |
| Winning combo/input dimension | `30` / 54 | `54` / 70 | Thay đổi cấu trúc |

Inner-fold protocol cho validation separation sạch hơn nhưng không cải thiện
RMSE trong lần chạy này. Nó chọn một lag/configuration combination khác, tăng
chi phí đúng 600 fast-screen fits và làm test RMSE tăng khoảng 13,330. Không nên
dựa vào một cross-run comparison duy nhất để kết luận inner folds luôn kém hơn;
đây chủ yếu là bằng chứng rằng block shortlist nhạy với validation protocol.

## 10. Những điểm đáng chú ý

1. **No-VMD là model tốt nhất trên test.** Nó thắng RMSE, MAE và MAPE, đồng thời
   có RMSE standard deviation thấp nhất. Trong run này không có bằng chứng rằng
   VMD cải thiện forecasting accuracy.
2. **Block-Proposed không vượt Proposed cũ.** Inner-fold block model đứng thứ ba
   và kém Sequential 0.70% test RMSE, dù nhanh hơn khoảng 14%.
3. **Joint Select có validation-to-test shift lớn nhất.** Nó thắng validation
   nhưng đứng cuối test; outer-validation winner không bảo đảm test winner.
4. **Không có statistical winner.** Mọi DM comparison đều không có ý nghĩa,
   nên các chênh lệch nhỏ giữa ba VMD workflows không nên được diễn giải thành
   superiority claim.
5. **Shared selection tạo model ổn định.** RMSE standard deviation của cả bốn
   model chỉ khoảng 0.83%–1.19% mean; không model nào dựa vào một seed thắng riêng.
6. **Inner fast-screen chạy đúng protocol.** Đủ 900 rows, ba folds, năm seeds và
   không lỗi; kết quả xấu hơn không đến từ thiếu seed/fold coverage.
7. **`USD_VND` cần sensitivity check.** ADF/KPSS sau diff1 vẫn mixed/unclear và
   artifact đánh dấu “Likely I(2) = Yes”. Nên kiểm tra transform khác trước khi
   chốt specification cho paper.
8. **Contribution phù hợp nhất hiện tại là methodology/efficiency, không phải
   absolute accuracy.** No-VMD vẫn là accuracy benchmark mạnh nhất, còn DM test
   không xác nhận khác biệt giữa bốn workflow.

## 11. Artifact chính

- [Primary per-seed mean/std benchmark](benchmark_metrics_comparison.csv)
- [Primary compatibility alias](benchmark_seed_robustness_metrics.csv)
- [Secondary ensemble benchmark](benchmark_ensemble_metrics_comparison.csv)
- [Per-seed raw metrics](paper_seed_metrics.csv)
- [Diebold–Mariano tests](diebold_mariano_tests.csv)
- [Efficiency summary](efficiency_summary.csv)
- [Experiment settings](paper_experiment_settings.csv)
- [Forecast date coverage](paper_forecast_date_coverage.csv)
- [Lag selection summary](lag_selection_summary.csv)
- [Aggregated rolling-fold metrics](lag_screen_fold_metrics.csv)
- [Block inner fast-screen details](vmd_ardl_block_subset_proposed/ffnn_block_subset_fast_screen_metrics.csv)
- [Block inner fast-screen summary](vmd_ardl_block_subset_proposed/ffnn_block_subset_fast_screen_summary.csv)
- [Block reconstructed selection](vmd_ardl_block_subset_proposed/ffnn_block_subset_mean_seed_selection.csv)
- [Run manifest](manifest.json)
