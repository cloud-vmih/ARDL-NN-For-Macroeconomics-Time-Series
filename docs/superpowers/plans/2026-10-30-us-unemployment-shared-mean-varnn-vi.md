# Kế hoạch benchmark thất nghiệp Mỹ với BLS/FRED và VAR-NN

> Bản tiếng Việt để đọc và rà soát. [Thiết kế chi tiết](../specs/2026-10-30-us-unemployment-shared-mean-varnn-design.md) và [kế hoạch triển khai tiếng Anh](2026-10-30-us-unemployment-shared-mean-varnn.md) là tài liệu kỹ thuật đi kèm. Tài liệu này là kế hoạch; chưa tải dữ liệu hoặc chạy benchmark mới.

## Mục tiêu

Sử dụng lại pipeline `shared-mean-benchmark` để so sánh bốn workflow hiện có với **VAR-NN**, trên một bộ dữ liệu thất nghiệp Mỹ được dựng lại từ BLS/FRED. Thêm dự báo naïve (lấy giá trị tháng trước) làm mốc tham chiếu. Không dùng `df_final.csv` hoặc kết quả sinh từ file đó để chọn mô hình hay kết luận cho nghiên cứu này.

## Chốt bài toán và dữ liệu

- **Biến cần dự báo:** `UNRATE`, tỷ lệ thất nghiệp Mỹ đã điều chỉnh mùa vụ, đơn vị phần trăm.
- **Thời điểm dự báo:** cuối tháng `t` theo giờ New York, dự báo giá trị `UNRATE` của chính tháng `t` trước khi BLS công bố.
- **Thông tin mô hình được dùng:** các giá trị thuộc tháng `t-1` trở về trước, với điều kiện chúng đã được công bố tại thời điểm dự báo. Không dùng dữ liệu tháng `t` hay giá trị được sửa đổi về sau.
- **Khoảng dữ liệu chính:** 01/2000–09/2025, tổng cộng 309 tháng liên tục. Giữ chia tập 216 tháng train, 46 tháng validation, 47 tháng test. Không tự điền 10/2025 vì BLS không thu thập ước tính từ khảo sát hộ gia đình cho tháng đó.
- **Chỉ tiêu chính:** RMSE trên giá trị mức của tỷ lệ thất nghiệp, tính trên cùng các ngày dự báo. Báo cáo thêm MAE, MAPE và Directional Accuracy.

| Cột trong bộ dữ liệu | Chuỗi FRED | Cách lấy |
| --- | --- | --- |
| `unemployment_rate` | `UNRATE` | Giá trị được công bố lần đầu của từng tháng; dùng làm nhãn và giá trị trễ. |
| `manufacturing_hours` | `AWHMAN` | Số giờ làm việc bình quân mỗi tuần của lao động sản xuất và phi quản lý ngành chế tạo, đã điều chỉnh mùa vụ. |
| `nonfarm_payrolls` | `PAYEMS` | Giá trị mức theo tháng. |
| `cpi_all_items` | `CPIAUCSL` | Giá trị mức theo tháng. |
| `industrial_prod` | `INDPRO` | Giá trị mức theo tháng. |

Dùng dữ liệu theo **vintage** từ API FRED/ALFRED. Với biến đầu vào của tháng `m`, cố định phiên bản đã thấy tại cuối tháng `m+1`, là thời điểm sớm nhất được phép dùng trong thiết kế trễ một tháng. Mọi chuỗi phải có đúng quan sát của tháng `m` tại vintage đó; thiếu thì báo lỗi, không thay bằng tháng cũ, không nội suy hay lấy vintage tương lai. Lưu mã chuỗi, đơn vị, mốc vintage, thời gian tải và mã băm của dữ liệu nguồn lẫn panel. Khóa `FRED_API_KEY` chỉ đọc từ biến môi trường, không ghi vào log hay artifact.

**Quyết định cập nhật:** dùng AWHMAN thay HOUST trong benchmark chính. HOUST chưa công bố đúng hạn ở 09–10/2013 và 12/2018–01/2019; không gán giá trị tháng khác vào cột cùng tháng. AWHMAN gần thị trường lao động hơn vì giờ làm ngành chế tạo thường được điều chỉnh trước quyết định tuyển hoặc sa thải. ALFRED chỉ có lịch sử vintage ICSA từ tháng 05/2009, nên ICSA cũng không thể dùng cho phần train 2000–2009.

**Ngoại lệ đã kiểm toán cho dòng cuối 09/2025:** INDPRO tháng này chưa có tại 30/11/2025 do đợt đóng cửa chính phủ. Lấy bản vintage 31/01/2026 và đánh dấu `used_in_forecasts=false` trong audit. Mọi lag ngoại sinh đều ít nhất một tháng, còn 10/2025 không có nhãn `UNRATE`, nên các ô ngoại sinh 09/2025 không đi vào dự báo nào của study. Dữ liệu tháng trước đó vẫn phải đạt kiểm tra công bố đúng hạn. Bộ tải giới hạn dưới 120 yêu cầu/phút và thử lại khi gặp lỗi 429/5xx.

Đầu vào bất biến của benchmark sẽ là `data/macro_us/processed/us_unemployment_awhman_2000_2025_09.csv`, gồm cột ngày và năm cột số ở bảng trên. File audit JSON nằm cạnh panel.

## Các workflow được so sánh

| Nhóm | Workflow | Ghi chú |
| --- | --- | --- |
| Hiện có | ARDL–FFNN không VMD | Mốc mạng nơ-ron không phân rã. |
| Hiện có | VMD block-subset two-stage Proposed | Phương án Proposed cần đánh giá lại trên dữ liệu mới. |
| Hiện có | VMD sequential screen | Phương án chọn biến tuần tự. |
| Hiện có | VMD joint select | Phương án chọn kết hợp. |
| Bổ sung | VAR-NN | Tái sử dụng helper shared-mean hiện có; chọn một cấu hình bằng validation rồi fit lại theo từng seed. |
| Tham chiếu | Naïve last observation | Dự báo tháng `t` bằng `UNRATE[t-1]`; không cần seed hoặc tuning. |

Năm workflow có huấn luyện ngẫu nhiên dùng cùng các seed `7, 42, 123, 2024, 2025`. Mỗi workflow chọn cấu hình theo **RMSE validation trung bình qua các seed**; sau khi khóa cấu hình mới tính test. VAR-NN cần được bổ sung Directional Accuracy theo từng seed. Nếu thiếu TensorFlow, thiếu một seed, hoặc thiếu ngày dự báo của một mô hình, run phải báo lỗi thay vì ghi kết quả so sánh như đã hoàn thành.

Đầu ra gồm metrics theo seed, metrics tổng hợp, forecast theo ngày, cấu hình đã chọn, độ phủ ngày dự báo và kiểm định Diebold–Mariano. Naïve xuất hiện một lần trong bảng, không nhân thành năm seed. Cả sáu workflow phải được chấm trên đúng cùng ngày validation/test.

## Trình tự triển khai

### 1. Tải và lưu nguồn có thể kiểm toán

- Viết module lấy đúng năm chuỗi từ FRED/ALFRED, đọc `FRED_API_KEY` ở điểm vào CLI.
- Cache phản hồi theo mã chuỗi, ngày vintage, kiểu dữ liệu API và phiên bản schema; lưu mã băm SHA-256.
- Giới hạn tốc độ yêu cầu theo quy định FRED và retry khi gặp 429/5xx; chạy lại sau gián đoạn sẽ dùng cache đã có.
- Viết test offline bằng phản hồi API giả để kiểm tra tham số, giá trị thiếu, cache hit và việc khóa API không lọt vào file/log.

**Hoàn thành khi:** cùng một yêu cầu đọc được từ cache mà không gọi mạng và audit truy ra được dữ liệu nguồn đã dùng.

### 2. Dựng panel theo thời điểm công bố

- Ghép năm chuỗi tháng `UNRATE`, `AWHMAN`, `PAYEMS`, `CPIAUCSL`, `INDPRO` theo tháng tham chiếu.
- Lấy `UNRATE` bản công bố đầu tiên làm nhãn; cố định vintage của các biến đầu vào; kiểm tra chúng đã sẵn có ở thời điểm dự báo.
- Bắt buộc mọi giá trị ngoại sinh đúng tháng tham chiếu và đã công bố tại ngày chốt; nếu thiếu thì dừng với lỗi chỉ rõ chuỗi và tháng.
- Riêng ngoại sinh 09/2025 dùng vintage 31/01/2026 và ghi rõ chúng không được dùng làm đầu vào dự báo; thêm fixture cho lịch công bố trễ.
- Từ chối tháng trùng, tháng thiếu, giá trị không hữu hạn, lịch tháng sai và trường hợp 10/2025 bị điền giả.
- Xuất panel CSV, audit JSON và kiểm tra chính xác 309 dòng, các mốc chia 216/46/47.

**Hoàn thành khi:** fixture offline chứng minh ánh xạ AWHMAN, đúng giới hạn vintage và báo lỗi khi thiếu tháng nguồn.

Lệnh dựng panel từ thư mục gốc project (sau khi đặt `FRED_API_KEY` trong môi trường):

```bash
set -a; . ./.env; set +a  # Chỉ cần khi khóa được lưu trong .env
PYTHONPATH=vmd_ardl_ffnn/src ./.venv/bin/python3 -m vmd_ardl_ffnn.macro_fred_panel \
  --cache-root data/macro_us/raw \
  --output-root data/macro_us/processed
```

### 3. Tích hợp VAR-NN vào shared-mean

- Thêm VAR-NN vào orchestrator, truyền cấu hình từ CLI qua lớp experiment đến runner.
- Dùng helper VAR-NN sẵn có để chọn cấu hình dựa trên validation và fit lại theo từng seed.
- Ghi kết quả VAR-NN vào các bảng metrics, forecast, cấu hình, chi phí fit; lưu riêng bảng tìm kiếm/chọn cấu hình dưới thư mục `varnn/`.
- Bổ sung Directional Accuracy trên forecast mức đã chuẩn hóa định dạng.

**Hoàn thành khi:** test xác nhận VAR-NN có đủ năm seed và một cấu hình được chọn chung; thiếu seed hoặc ngày phải làm run thất bại.

### 4. Thêm mốc naïve và kiểm tra so sánh công bằng

- Tái sử dụng baseline `naive_last_observation` của project.
- Đưa baseline vào bảng kết quả và kiểm định DM, nhưng không đưa nó vào vòng lặp seed của mô hình học máy.
- So sánh tập ngày validation/test giữa cả sáu workflow và báo cụ thể ngày thiếu nếu có.

**Hoàn thành khi:** dự báo naïve cho tháng `t` đúng bằng `UNRATE[t-1]` và sáu workflow có cùng ngày được chấm.

### 5. Thêm profile chạy riêng cho bộ thất nghiệp

- Thêm `--dataset-profile us-unemployment` cho `shared-mean-benchmark`, bắt buộc truyền `--data` trỏ đến panel đã audit.
- Profile này dùng `date` làm cột thời gian, `unemployment_rate` làm target, đúng bốn biến đầu vào và **không log-transform**.
- Manifest của run phải ghi mã băm panel/audit, tên profile, quy tắc thời điểm dự báo, danh sách seed và sáu workflow.
- Cập nhật metadata của `macro-benchmark` nếu nó kế thừa shared runner để số lượng/tên workflow trong manifest khớp kết quả thực tế.
- Bổ sung lệnh tái lập và cách đọc kết quả vào tài liệu project.

**Hoàn thành khi:** test từ chối thiếu panel, thiếu cột, sai pipeline/profile hoặc audit không khớp.

### 6. Kiểm tra và chạy benchmark

1. Chạy test nguồn, panel, CLI, orchestrator và toàn bộ test suite.
2. Khi có `FRED_API_KEY`, tải dữ liệu thật; kiểm tra audit, 309 tháng liên tục, không có ô thiếu ở cột mô hình và đúng ranh giới ba tập.
3. Chạy smoke test một seed vào thư mục kết quả riêng; kiểm tra VAR-NN, naïve, độ phủ ngày và manifest.
4. Chạy benchmark chính với năm seed và run ID mới; đọc bảng metrics, forecast, cấu hình, độ phủ và DM tests.
5. Báo cáo validation và test riêng, so Proposed với từng workflow và naïve. Không dùng test để đổi phương án; nếu muốn kiểm tra thêm nhiều giai đoạn dự báo, đăng ký các mốc rolling-origin trước khi xem điểm test của chúng.

## Lưu ý khi diễn giải

Validation hiện chứa cú sốc COVID-19, còn test tương đối ổn định hơn; cần nêu sự khác biệt này khi giải thích điểm số. Kết quả cũ từ `df_final.csv` không chứng minh Proposed tốt hơn trên bộ dữ liệu mới. Toàn bộ file và kết quả chưa commit đang có trong workspace cần được giữ nguyên khi triển khai.

## Nguồn chính thức

- FRED: [UNRATE](https://fred.stlouisfed.org/series/UNRATE), [AWHMAN](https://fred.stlouisfed.org/series/AWHMAN), [PAYEMS](https://fred.stlouisfed.org/series/PAYEMS), [CPIAUCSL](https://fred.stlouisfed.org/series/CPIAUCSL), [INDPRO](https://fred.stlouisfed.org/series/INDPRO).
- BLS: [giờ làm ngành chế tạo là chỉ báo sớm](https://www.bls.gov/opub/mlr/2020/article/pdf/employment-expansion-continued-in-2019-but-growth-slowed-in-several-industries.pdf).
- ALFRED: [lịch sử vintage ICSA bắt đầu từ 05/2009](https://alfred.stlouisfed.org/series?seid=ICSA).
- FRED API: [quy tắc thời gian thực](https://fred.stlouisfed.org/docs/api/fred/realtime_period.html), [series observations](https://fred.stlouisfed.org/docs/api/fred/series_observations.html).
- BLS: [gián đoạn khảo sát CPS tháng 10/2025](https://www.bls.gov/cps/methods/2025-federal-government-shutdown-impact-cps.htm).
- BLS: [công bố số liệu 09/2025 bị chậm](https://www.bls.gov/news.release/archives/empsit_11202025.htm).
- FRED: [giới hạn 120 yêu cầu/phút và lỗi 429](https://fred.stlouisfed.org/docs/api/fred/errors.html).
