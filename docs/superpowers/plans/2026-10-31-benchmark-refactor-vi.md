# Kế hoạch refactor benchmark (31/10/2026)

**Mục tiêu:** Giữ nguyên kết quả và artifact của benchmark đang dùng, bỏ mặc định `df_final.csv`, giảm trách nhiệm của entrypoint và làm rõ phần dùng chung giữa các workflow.

**Thiết kế đã thống nhất:** Đề xuất refactor trong hội thoại ngày 31/10/2026; thực hiện từng phần, giữ nguyên giao thức chọn mô hình, lịch dự báo, seed, manifest và tên bảng kết quả.

**Công nghệ:** Python 3.11, pytest, pandas, TensorFlow, statsmodels.

## Bước 1: Chặn dataset mặc định sai

- [x] Viết test cho `parse_args([])`: thiếu `--data` phải báo lỗi trước khi chạy model.
- [x] Chạy test để xác nhận lỗi hiện tại.
- [x] Yêu cầu `--data` rõ ràng; giữ nguyên profile macro và FRED khi có đường dẫn hợp lệ.
- [x] Chạy test liên quan; sửa README để bỏ ví dụ không có dataset và mô tả `df_final.csv` là mặc định.

## Bước 2: Tách CLI và điều phối

- [x] Viết test kiểm tra API entrypoint, dispatch pipeline và hành vi manifest không đổi.
- [x] Tách parser, cấu hình profile và điều phối run thành các module trong package; giữ `run_pipeline_advanced.py` là wrapper tương thích với các import hiện có.
- [x] Kiểm tra `--help`, lỗi tham số, metadata run và các test hiện có.

## Bước 3: Làm rõ phần benchmark dùng chung

- [x] Viết test đặc trưng cho tổng hợp seed và kiểm tra date coverage.
- [x] Đưa các hàm benchmark dùng chung vào module công khai; ngừng nhập helper `_private` xuyên module.
- [x] Tách các workflow block-subset, shared-mean và hard-FFNN khỏi `experiment_fixed.py` mà không thay đổi thân hàm.
- [x] So sánh schema và giá trị seed summary/date coverage với artifact lịch sử; chạy toàn bộ test.

## Ràng buộc

- Không chạy lại benchmark đầy đủ nhiều giờ chỉ để kiểm tra refactor; dùng test đặc trưng và smoke có chi phí nhỏ.
- Không sửa thư mục `results/shared-mean-benchmark/us-unemployment-01/` hoặc dữ liệu `data/macro_us/`.
- Không xóa nhánh legacy trong đợt này; chỉ không cho chọn dataset sai một cách ngầm định.
