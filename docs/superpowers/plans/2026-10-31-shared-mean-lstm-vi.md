# Thêm LSTM vào shared-mean benchmark

**Mục tiêu:** Bổ sung LSTM đa biến vào `shared-mean-benchmark` trên đúng validation/test dates, reporting seeds và estimand hiện có.

**Thiết kế:** Mỗi dự báo tháng `t` chỉ dùng target và ngoại sinh đến `t−1`. Scaler chỉ fit trên train khi chọn cấu hình, và trên train+validation khi refit cho test. Chọn một `(lookback, hidden_units, learning_rate)` chung bằng mean validation level RMSE qua mọi reporting seed; test chỉ đánh giá sau khi khóa cấu hình. Giữ `macro-benchmark` core-only như hiện tại bằng `include_lstm=False` ở lời gọi nội bộ.

**Ràng buộc:** Không sửa dữ liệu BLS/FRED hoặc run directory hiện có. Không chạy full benchmark nhiều giờ để kiểm tra tính đúng của mã.

## Bước thực hiện

- [x] Viết test cho cửa sổ đầu vào không chứa giá trị tháng dự báo, scaler train-only và ranking mean qua seed.
- [x] Viết test cho model set, seed/date coverage và artifact của LSTM trong benchmark orchestration.
- [x] Tạo `models/lstm.py` cho cấu hình, chuẩn bị cửa sổ và fit/predict TensorFlow.
- [x] Tạo `lstm_shared_mean.py` cho search, khóa cấu hình, refit và xuất metrics/forecasts.
- [x] Tích hợp vào `shared_mean_benchmark_experiments.py`, metadata manifest và validator của panel thất nghiệp.
- [x] Giữ `macro-benchmark` core-only đúng model set cũ; cập nhật README và tài liệu benchmark.
- [x] Chạy test tập trung, một fit LSTM CPU nhỏ và toàn bộ test suite; kiểm tra `git diff --check`.
