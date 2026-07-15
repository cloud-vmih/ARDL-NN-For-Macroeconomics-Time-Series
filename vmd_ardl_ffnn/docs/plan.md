# Plan Giảm Mạnh Thời Gian Training ARDL + FFNN

  ## Summary

  Thay exhaustive grid hiện tại bằng staged search có ngân sách cố định: ARDL vẫn chọn top lag, nhưng không sinh toàn bộ tích Descartes top_n^features; FFNN cũng
  không grid full trên mọi lag spec. Mục tiêu là giảm số lần fit từ hàng nghìn/hơn 10k xuống khoảng vài chục đến vài trăm, vẫn chọn model bằng validation RMSE/MAE/
  MAPE và giữ train-only lag selection để tránh leakage.

  ## Key Changes

  - Thêm chế độ search mới, ví dụ search_strategy="staged_halving" trong config/CLI.
  - ARDL selector vẫn chấm lag từng biến, nhưng sinh candidate lag specs theo ngân sách:
      - 1 spec tốt nhất toàn cục: chọn lag rank 1 của từng feature.
      - Thêm các spec “one-feature perturbation”: chỉ đổi một feature sang rank 2 hoặc rank 3, giữ các feature khác ở rank 1.
      - Thêm tối đa max_lag_specs spec có combo_score_sum tốt nhất nếu cần.
      - Default đề xuất: max_lag_specs=16 cho no-VMD, max_lag_specs=12 mỗi component cho VMD.
  - FFNN search chuyển thành 3 stage:
      - Stage 1: search lag spec nhanh với một cấu hình FFNN mặc định, ví dụ HR=8, alpha=1e-3, seed=7, max_iter_fast=150.
      - Stage 2: giữ top k_lag_specs, ví dụ top_k_lag_specs=4, rồi search hyperparameter đầy đủ hơn trên các lag spec này.
      - Stage 3: refit winner bằng max_iter=500 trên train+validation như hiện tại.
  - Với VMD:
      - Chấm candidate theo từng component như hiện tại, nhưng áp dụng budget trên từng component.
      - Giữ top_component_candidates=3 thay vì 5 để giảm reconstruct combo từ 5^components xuống 3^components.
      - Nếu có 4 components như VMD1,VMD2,VMD3,RES, combo giảm từ 625 xuống 81.
  - Giữ nguyên nguyên tắc chống leakage:
      - ARDL chỉ chọn lag trên train.
      - Validation dùng walk-forward VMD cache.
      - Test chỉ chạy sau khi đã khóa winner bằng validation.

  ## Public Config / CLI

  - Thêm vào ARDLSelectionConfig:
      - lag_spec_strategy: str = "staged" với lựa chọn "full_product" và "staged".
      - max_lag_specs: int = 16.
  - Thêm vào FFNNConfig:
      - search_strategy: str = "staged_halving".
      - fast_hidden_units: int = 8.
      - fast_alpha: float = 1e-3.
      - fast_max_iter: int = 150.
      - top_k_lag_specs: int = 4.
      - top_component_candidates: int = 3.
  - Thêm CLI flags tương ứng:
      - --search-strategy staged-halving|full-grid
      - --max-lag-specs
      - --top-k-lag-specs
      - --top-component-candidates
      - --fast-max-iter

  ## Implementation Plan

  - Cập nhật ARDLOrderSelector.select() để hỗ trợ staged lag spec generation:
      - Vẫn xuất bảng ARDL đầy đủ theo từng lag.
      - Thêm cột/row ghi rõ lag_spec_strategy, combo_score_sum, selected_for_stage.
      - Khi full_product, giữ behavior cũ để so sánh.
  - Tách search FFNN thành helper chung:
      - _search_lag_specs_fast(...): fit nhanh mỗi lag spec với cấu hình mặc định, trả bảng validation score.
      - _search_hyperparams_on_shortlist(...): grid hidden/alpha/seed chỉ trên top lag specs.
      - _select_best_candidate(...): sort theo Val RMSE, Val MAE, Val MAPE.
  - Áp dụng helper cho cả run_without_vmd() và run():
      - No-VMD chọn winner trực tiếp từ staged search.
      - VMD chọn shortlist từng component, sau đó reconstruct combo trên top component candidates.
  - Artifact CSV vẫn giữ tên hiện tại, nhưng thêm cột:
      - search_stage
      - lag_spec_rank
      - lag_spec_strategy
      - fast_screen_rmse
      - selected_for_hyperparam_search
  - README cập nhật công thức chi phí:
      - Cũ: top_n^features * hidden * alpha * seed * components.
      - Mới: (max_lag_specs * 1 fast config) + (top_k_lag_specs * hidden * alpha * seed) mỗi component.

  ## Test Plan

  - Unit/smoke test:
      - full_product tạo số LagSpec giống hiện tại.
      - staged không vượt quá max_lag_specs, không tạo lag 0, không duplicate LagSpec.label.
      - Stage 2 chỉ train trên lag specs đã được Stage 1 shortlist.
  - Smoke experiment:
      - Chạy no-VMD với dataset nhỏ trong tests/test_smoke.py.
      - Kiểm tra output có best_component_models, final_metrics, ffnn_validation_search.
  - Regression check:
      - Chạy một lần --search-strategy full-grid để đảm bảo behavior cũ vẫn hoạt động.
      - Chạy --search-strategy staged-halving và so sánh số dòng/candidate giảm mạnh.
  - Acceptance criteria:
      - Số lần fit FFNN giảm ít nhất 70%.
      - Không dùng validation/test trong chọn lag ARDL.
      - Final metrics vẫn được tính trên validation/test như pipeline hiện tại.
      - CSV audit đủ thông tin để giải thích vì sao lag spec/hyperparameter được chọn.

  ## Assumptions

  - Ưu tiên hiện tại là giảm rất mạnh thời gian training, chấp nhận không duyệt toàn bộ tổ hợp.
  - Objective chọn model vẫn là validation RMSE, tie-break bằng MAE, rồi MAPE.
  - Giữ seed_grid=(7,) mặc định để giảm variance và thời gian; chỉ mở nhiều seed khi cần báo cáo robustness.
  - Không thay đổi mô hình lõi SklearnFFNNRegressor; tối ưu tập trung vào search strategy và budget.