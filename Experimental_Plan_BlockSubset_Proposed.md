# Experimental Plan — Block-Subset Proposed Comparator

## 1. Mục tiêu

Plan này dành riêng cho pipeline `block-proposed-benchmark`. Nó không thay thế
pipeline `benchmark` cũ và không đổi ý nghĩa lịch sử của
`Experimental_Plan_VMD_ARDL_FFNN.md`.

Trong study này, phương pháp chính (Proposed) là:

```text
VMD theo forecast origin
→ ARDL SelectOrder chọn target lag trên train
→ khóa target lag
→ AIC chọn một distributed-lag order cho từng exogenous variable
→ tạo mọi block subset khác rỗng
→ fast-screen đủ block specs bằng lightweight FFNN
→ giữ top 4 specs/component
→ full FFNN grid trên shortlist và mọi reporting seed
→ ghép candidate giữa các VMD components
→ chọn một shared winner bằng mean reconstructed-level validation RMSE qua seeds
→ khóa winner, refit riêng từng seed trên train + validation
→ đánh giá test đúng một lần
```

Tên pipeline/model chính:

- pipeline: `vmd_ardl_block_subset_proposed`;
- model: `VMD_ARDL_BlockSubset_TwoStage_Proposed`;
- ablation ID: `A7`;
- artifact schema: `block_subset_proposed_v1`.

## 2. Câu hỏi nghiên cứu

1. Proposed có dự báo tốt hơn các statistical, neural, no-VMD và
   closest-workflow benchmarks hay không?
2. VMD, target-lag selection, AIC variable blocks, block-subset search và
   reconstructed-level objective đóng góp gì?
3. Chọn một structure/hyperparameter chung bằng mean validation qua seeds khác
   gì so với chọn riêng từng seed?
4. Two-stage search tiết kiệm bao nhiêu chi phí so với full-grid và có giữ được
   accuracy hay không?

## 3. Giao thức dữ liệu và đánh giá

- Split theo thời gian: 70% train, 15% validation, 15% test.
- Không shuffle.
- Stationarity decision chỉ dùng train.
- VMD validation/test chạy rolling one-step theo forecast origin.
- Exogenous lag 0 không được dùng.
- Primary selection metric: reconstructed/level validation RMSE.
- Tie-breakers: level validation MAE, MAPE, rồi input dimension/complexity.
- Test tuyệt đối không tham gia chọn lag, block subset, architecture, learning
  rate, seed hay model.
- Primary report dùng `scale="level"`; transformed/component metrics chỉ dùng
  để audit hoặc ablation có chủ đích.

Reporting seeds cố định:

```text
7, 42, 123, 2024, 2025
```

Mọi stochastic main benchmark phải dùng cùng seed set này.

## 4. Shared mean-seed selection

“Shared” nghĩa là seed không nằm trong model signature. Mỗi candidate được fit
ở toàn bộ reporting seeds, sau đó được xếp hạng bằng mean validation score. Chỉ
một signature/combination chung được khóa trước khi test.

Quy tắc theo model family:

| Model family | Shared selection objective |
| --- | --- |
| No-VMD FFNN | Mean level validation RMSE qua seeds |
| VMD FFNN | Mean reconstructed-level validation RMSE qua seeds |
| VARNN | Mean level validation RMSE qua seeds |
| A4 counterfactual | Mean component validation score qua seeds, sau đó reconstruct |

Sau khi chọn shared winner, model vẫn được refit riêng ở mỗi seed để báo cáo
mean/std/median/count trên test. Không chọn “best seed”.

## 5. Main benchmark family

| ID | Pipeline | Vai trò |
| --- | --- | --- |
| B0a | `naive_last_observation` | Minimum reference |
| B0b | `naive_seasonal` | Seasonal lag-12 reference |
| B1 | `standalone_ardl` | Linear dynamic-regression baseline |
| B2 | `ffnn_fixed_no_vmd` | Fixed-lag neural baseline, shared selection |
| B3 | `vmd_ffnn_fixed` | Fixed-lag VMD neural baseline, shared selection |
| B4 | `ardl_ffnn_no_vmd` | Adaptive lag FFNN without VMD, shared selection |
| B5 | `varnn` | Multivariate neural baseline, shared selection |
| B6 | `vmd_ardl_sequential_screen` | Closest previous sequential workflow, shared selection |
| B7 | `vmd_ardl_joint_select` | `ardl_select_order` chọn target/exogenous lags đồng thời, shared selection |
| B8 | `vmd_ardl_hard_config_ffnn` | Full-product top-3 exog orders; train-only frequency/lag hard FFNN config |
| P | `vmd_ardl_block_subset_proposed` | A7 two-stage Proposed |

VARNN chỉ được xem là đã có trong benchmark khi `paper_experiment_settings.csv`
ghi `enabled=True`. Nếu dùng `--skip-varnn` hoặc TensorFlow không sẵn sàng, phải
báo cáo rõ benchmark thiếu B5.

## 6. Ablation A0–A7

| ID | VMD | Target lag | Exogenous structure | Selection objective | Seed aggregation | Search |
| --- | --- | --- | --- | --- | --- | --- |
| A0 | No | Fixed | Fixed | Level validation | Shared mean-seed | Full grid |
| A1 | Yes | Fixed | Fixed | Reconstructed level | Shared mean-seed | Full grid |
| A2 | Yes | ARDL SelectOrder | Fixed exog lag | Reconstructed level | Shared mean-seed | Full grid |
| A3 | Yes | ARDL SelectOrder | AIC rank-1 block của đủ mọi exog | Reconstructed level | Shared mean-seed | Full grid |
| A4 | Yes | ARDL SelectOrder | Mọi non-empty AIC block subset | Component validation | Shared mean-seed | Full grid |
| A5 | Yes | ARDL SelectOrder | Mọi non-empty AIC block subset | Reconstructed level | Shared mean-seed | Full grid |
| A6 | Yes | ARDL SelectOrder | Mọi non-empty AIC block subset | Reconstructed level | Chọn riêng từng seed | Full grid |
| A7 | Yes | ARDL SelectOrder | Top 4 từ mọi AIC block subset | Reconstructed level | Shared mean-seed | Two-stage; Proposed |

### A0 → A1: đóng góp của VMD

Giữ fixed target/exogenous lags và shared FFNN selection. Khác biệt chính là có
hay không có VMD.

### A1 → A2: đóng góp của target-lag selection

Giữ fixed exogenous lags; thay fixed target lag bằng ARDL SelectOrder trên train.

### A2 → A3: đóng góp của AIC variable-block construction

Mỗi exogenous variable có đúng một order do ARDL AIC chọn với target lag đã
khóa. A3 luôn giữ đủ các variable blocks, nên chưa có block-subset selection.

### A3 → A5: đóng góp của block-subset search

A3 chỉ có all-block spec. A5 xét mọi tập con khác rỗng của các variable blocks,
nhưng dùng cùng reconstructed-level shared selection và full-grid budget.

### A4 → A5: đóng góp của reconstructed-level objective

A4 chọn rank-1 signature độc lập trên component scale rồi reconstruct. A5 ghép
component candidates và chọn trực tiếp bằng reconstructed-level validation
objective. Hai model đều dùng shared mean-seed selection.

### A6 → A5: shared versus per-seed selection

A6 cho phép structure/hyperparameters thắng khác nhau theo seed. A5 khóa một
winner chung bằng mean validation qua seeds. So sánh này đo ảnh hưởng của
selection protocol, không phải đóng góp của một forecasting module mới.

### A5 → A7: exhaustive versus two-stage search

A5 full-grid toàn bộ block specs. A7 fast-screen đủ specs bằng:

- `alpha=0.1`;
- learning rate `0.001`;
- lightweight width theo input dimension;
- mean score qua reporting seeds;
- giữ `top_component_candidates=4` lag specs/component.

Sau shortlist, A7 dùng cùng final FFNN grid và cùng reconstructed-level
mean-seed objective với A5.

## 7. Hyperparameter budget

Pipeline riêng này dùng final FFNN learning-rate grid:

```text
0.01, 0.001
```

Các learning rate phải được đánh giá trên validation ở mọi reporting seed trước
khi khóa shared winner. Fast screen của A7 vẫn cố định ở `0.001`; không dùng
final learning-rate grid trong fast screen.

Để comparison có ý nghĩa:

- A3–A7 dùng cùng ARDL bounds và AIC settings;
- A4–A6 dùng cùng full block-spec pool;
- A4 và A5 chỉ khác final objective;
- A5 và A6 chỉ khác seed aggregation;
- A5 và A7 chỉ khác full-grid versus two-stage candidate budget.

## 8. Statistical tests

Main benchmark dùng:

- primary RMSE/MAE/MAPE/DA là mean/std/median/count của metric tính riêng
  từng reporting seed;
- Diebold–Mariano test trên common test dates;
- squared-error loss, horizon 1;
- secondary ensemble-mean prediction qua reporting seeds cho stochastic models;
- DM/Holm dùng chuỗi ensemble-mean prediction và phải ghi rõ estimand này;
- Holm correction cho pairwise comparisons;
- moving-block bootstrap 95% CI cho paired mean loss difference.

Trước khi diễn giải DM/Holm, kiểm tra
`paper_forecast_date_coverage.csv` có `same_dates_for_all_models=True`.

Ưu tiên báo cáo các cặp:

- Proposed versus sequential-screen B6;
- Proposed versus joint SelectOrder B7;
- Proposed versus hard-config FFNN B8, kèm runtime và fit count;
- Proposed versus ARDL–FFNN no VMD B4;
- Proposed versus standalone ARDL B1;
- Proposed versus VARNN B5;
- Proposed versus naïve baselines.

Không suy diễn “significantly better” chỉ từ test RMSE point estimate.

## 9. Efficiency analysis

Báo cáo tối thiểu:

- total runtime;
- ARDL selection time;
- VMD cache time;
- FFNN search time;
- refit/forecast time;
- evaluated FFNN fits;
- số lag specs trước/sau shortlist;
- search-space reduction percentage.

A7 chỉ được gọi là efficient approximation của A5 nếu artifact chứng minh giảm
runtime/fit count. Nếu accuracy giảm, phải báo cáo rõ trade-off thay vì chỉ chọn
kết quả thuận lợi.

## 10. Artifacts bắt buộc

Main outputs:

- `benchmark_metrics_comparison.csv` (primary per-seed mean/std/median/count);
- `benchmark_ensemble_metrics_comparison.csv` (secondary ensemble metrics);
- `benchmark_seed_robustness_metrics.csv` (explicit primary-table alias);
- `benchmark_forecasts_comparison.csv`;
- `paper_seed_metrics.csv`;
- `paper_metrics_summary.csv`;
- `ablation_metrics.csv`;
- `ablation_forecasts.csv`;
- `diebold_mariano_tests.csv`;
- `paper_experiment_settings.csv`;
- `paper_forecast_date_coverage.csv`;
- `efficiency_summary.csv`;
- `lag_selection_summary.csv`;
- `lag_selection_frequency.csv`;
- `skipped_models.csv` khi có model bị bỏ.

Selection audit của từng shared variant nằm trong subdirectory tương ứng. Với
Proposed/A5 cần giữ component search, component mean-seed ranking,
reconstructed seed scores và shared selection table.

## 11. Lệnh chạy

Smoke trước bằng một seed:

```bash
./.venv/bin/python3 run_pipeline_advanced.py \
  --pipeline block-proposed-benchmark \
  --paper-seeds 7 \
  --skip-varnn
```

Full paper run:

```bash
./.venv/bin/python3 run_pipeline_advanced.py \
  --pipeline block-proposed-benchmark \
  --paper-seeds 7 42 123 2024 2025
```

Pipeline này cố ý chạy cả A5 full-grid, A6 per-seed full-grid và A7 two-stage,
nên nặng hơn `benchmark` cũ. Không dùng artifact từ smoke run làm final paper
result.

## 12. Checklist trước khi viết paper

- `manifest.json` có `status="success"`.
- Schema là `block_subset_proposed_v1`.
- A0–A7 đều xuất hiện đúng một định nghĩa trong settings/ablation table.
- Mọi stochastic main benchmark dùng đủ cùng reporting seeds.
- Shared models có đúng một selected signature/combination qua seeds.
- A6 được phép có winner khác nhau theo seed.
- A7 fast screen đủ input specs và chỉ chuyển tối đa top 4/component vào grid.
- Learning-rate grid của pipeline này là `0.01`, `0.001`.
- Test không xuất hiện trong selection artifacts.
- Forecast dates/actual values khớp giữa benchmark models.
- Nếu VARNN bị skip, paper ghi rõ benchmark không đầy đủ.
- Conclusions bám vào DM/Holm, confidence interval và observed efficiency; không
  chọn narrative sau khi nhìn model có test RMSE thấp nhất.

## 13. Đóng góp có thể tuyên bố nếu kết quả hỗ trợ

1. Một structured lag-space construction dựa trên ARDL AIC variable blocks.
2. Một reconstructed-level joint selection objective phù hợp trực tiếp với
   target metric sau VMD reconstruction.
3. Một shared mean-seed protocol giúp model definition ổn định và tái lập hơn.
4. Một two-stage candidate screening strategy giảm chi phí so với exhaustive
   block-subset full grid.
5. Một evaluation framework đồng nhất gồm strong benchmarks, A0–A7 ablations,
   seed aggregation, DM/Holm và search-efficiency audit.

Mỗi claim chỉ được dùng khi comparison tương ứng trong Sections 6–9 thực sự hỗ
trợ; nếu không, mô tả đó là methodological design thay vì empirical superiority.
