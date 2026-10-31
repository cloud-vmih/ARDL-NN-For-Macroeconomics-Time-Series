# CLI chính của project

`run_pipeline_advanced.py`, `python -m vmd_ardl_ffnn.cli` và lệnh
`vmd-ardl-ffnn-run` cùng gọi một runner. Hai cách gọi qua package yêu cầu cài
`pip install -e ./vmd_ardl_ffnn` hoặc đặt `PYTHONPATH=vmd_ardl_ffnn/src`.
Để xem toàn bộ option, chạy:

```bash
./.venv/bin/python3 run_pipeline_advanced.py --help
```

## Benchmark thất nghiệp Mỹ đang dùng

```bash
./.venv/bin/python3 run_pipeline_advanced.py \
  --pipeline shared-mean-benchmark \
  --dataset-profile us-unemployment \
  --data data/macro_us/processed/us_unemployment_awhman_2000_2025_09.csv \
  --tensorflow-device gpu \
  --tensorflow-run-eagerly \
  --paper-seeds 7 42 123 2024 2025 \
  --run-id us-unemployment-lstm-04
```

| Option | Ý nghĩa |
| --- | --- |
| `--pipeline` | Workflow hoặc benchmark cần chạy. |
| `--data` | CSV đầu vào, **bắt buộc**. `df_final.csv` bị từ chối do dữ liệu sai. |
| `--dataset-profile us-unemployment` | Bật kiểm tra panel BLS/FRED, audit và ngày dự báo; chỉ dùng với `shared-mean-benchmark`. |
| `--tensorflow-device gpu/cpu/auto` | Chọn thiết bị TensorFlow; mặc định là `gpu`. |
| `--tensorflow-run-eagerly` | Fit FFNN bằng Keras eager execution để hạn chế RAM tăng qua nhiều lượt train. |
| `--paper-seeds` | Các seed báo cáo riêng và tổng hợp. |
| `--output-root`, `--run-id` | Vị trí và tên thư mục run; mặc định là `results/` trong thư mục đang đứng. |
| `--resume` | Chạy tiếp `sequential-vmd-ardl-ffnn` với cùng data/config và `--run-id`. |
| `--macro-profile core-only/full` | Chọn nhóm model trong `macro-benchmark`. |
| `--block-subset-search-strategy full-grid/two-stage` | Chọn chiến lược cho comparator trong benchmark legacy. |
| `--skip-varnn` | Chỉ dùng với `benchmark` hoặc `block-proposed-benchmark`; shared-mean luôn chạy VAR-NN. |

`shared-mean-benchmark` cũng chạy LSTM mặc định. Grid LSTM hiện được khai báo
trong `LSTMConfig` (`models/lstm.py`); CLI chưa có option riêng để đổi grid.
Nếu chạy lại sau khi cập nhật model set, dùng `--run-id` mới để giữ nguyên
artifact của lần chạy trước.

Các pipeline legacy dùng cấu hình `TIME_PERIOD`/`Export_US` và cần CSV khác đã
kiểm định. Giao diện CLI cũ với `--features`, `--target`, `--date-col` vẫn được
lưu tại [`LEGACY_CLI_OPTIONS.md`](LEGACY_CLI_OPTIONS.md) và có thể gọi bằng
`python -m vmd_ardl_ffnn.legacy_cli`.
