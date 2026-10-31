# Data dictionary — benchmark thất nghiệp Mỹ

**Panel:** `data/macro_us/processed/us_unemployment_awhman_2000_2025_09.csv`

**Phạm vi:** 309 tháng, 01/2000–09/2025; train 216, validation 46, test 47. Mỗi dòng ứng với một tháng tham chiếu. Các chuỗi số đều có tần suất tháng và đã điều chỉnh mùa vụ.

| Cột | Nguồn, đơn vị | Vai trò và lý do chọn |
| --- | --- | --- |
| `date` | Ngày đầu tháng (`YYYY-MM-01`) | Khóa thời gian để chia train/validation/test và tạo các lag; không đưa trực tiếp vào FFNN như biến số. |
| `unemployment_rate` | [UNRATE](https://fred.stlouisfed.org/series/UNRATE), % lực lượng lao động | **Biến mục tiêu**: tỷ lệ thất nghiệp cần dự báo. Giá trị quá khứ của chính chuỗi này cũng là đầu vào tự hồi quy. Panel dùng bản công bố đầu tiên làm nhãn. |
| `manufacturing_hours` | [AWHMAN](https://fred.stlouisfed.org/series/AWHMAN), giờ/tuần | **Biến đầu vào**: giờ làm của lao động sản xuất và phi quản lý ngành chế tạo. Doanh nghiệp có thể điều chỉnh giờ làm trước khi thay đổi nhân sự, nên đây là ứng viên tín hiệu sớm của thị trường lao động. [BLS](https://www.bls.gov/opub/mlr/2016/article/pdf/current-employment-statistics-survey-100-years-of-employment-hours-and-earnings.pdf). |
| `nonfarm_payrolls` | [PAYEMS](https://fred.stlouisfed.org/series/PAYEMS), nghìn người | **Biến đầu vào**: việc làm hưởng lương ngoài nông nghiệp; phản ánh quy mô việc làm và nhu cầu lao động. |
| `cpi_all_items` | [CPIAUCSL](https://fred.stlouisfed.org/series/CPIAUCSL), chỉ số 1982–84 = 100 | **Biến đầu vào**: mức giá tiêu dùng chung; đại diện cho bối cảnh lạm phát có thể đi cùng biến động kinh tế và việc làm. Đây là chỉ số mức, không phải tỷ lệ lạm phát đã tính sẵn. |
| `industrial_prod` | [INDPRO](https://fred.stlouisfed.org/series/INDPRO), chỉ số 2017 = 100 | **Biến đầu vào**: sản lượng thực của công nghiệp; phản ánh chu kỳ hoạt động sản xuất và nhu cầu lao động liên quan. |

**Quy tắc đưa vào mô hình:** tại cuối tháng `t`, mô hình dự báo `unemployment_rate[t]` chỉ từ các quan sát thuộc `t−1` trở về trước đã công bố ở thời điểm đó. Các biến ngoại sinh được lấy theo vintage FRED/ALFRED tại cuối tháng kế tiếp tháng tham chiếu; không dùng số sửa đổi về sau, không thay tháng nguồn hay nội suy. Riêng ngoại sinh của dòng cuối 09/2025 dùng vintage 31/01/2026 và được audit là không tham gia dự báo. Lag và tập biến cụ thể do từng workflow chọn trên train/validation, không chọn bằng test. Audit nguồn và mã băm nằm cạnh file CSV.

**Lưu ý:** lý do chọn biến là cơ sở kinh tế để đưa vào danh sách ứng viên; khả năng cải thiện dự báo phải được kiểm tra bằng benchmark, không suy ra chỉ từ mô tả biến.
