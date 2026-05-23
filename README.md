# Multiscaled Neural Autoregressive Distributed Lag

**Multiscaled Neural Autoregressive Distributed Lag: Mô hình phân rã mode thực nghiệm mới cho bài toán dự báo chuỗi thời gian phi tuyến.**

File `ardl_ffnn_model.py`, đóng vai trò là module chính cho quy trình lựa chọn độ trễ dựa trên ARDL và dự báo bằng mạng nơ-ron truyền thẳng (Feed-Forward Neural Network - FFNN). 

## Nội dung Repository

- `ardl_ffnn_model.py` - Module Python chính cho quy trình ARDL + FFNN hiện tại.
- `DA_Prj1_final_ARDL_FFNN_merged.ipynb` - Notebook được sử dụng trong quá trình thực nghiệm và phân tích.
- `df_final.csv` - Bộ dữ liệu hiện đang được sử dụng.
- `Nhom19_DA.docx` - Bản nháp tài liệu/báo cáo của dự án.

## Quy trình mô hình hiện tại

Module hiện tại hỗ trợ các bước sau:

1. Đọc dữ liệu chuỗi thời gian từ `pandas DataFrame` hoặc file CSV.
2. Áp dụng biến đổi log cho các biến kinh tế có giá trị dương.
3. Kiểm tra sơ bộ tính dừng của chuỗi bằng kiểm định ADF và KPSS.
4. Biến đổi các biến không dừng bằng sai phân bậc nhất.
5. Lựa chọn các độ trễ tiềm năng của biến đầu vào bằng mô hình ARDL.
6. Tìm kiếm các tổ hợp ánh xạ độ trễ bằng mạng nơ-ron truyền thẳng.
7. So sánh hai đặc tả: mỗi biến có độ trễ riêng và các biến sử dụng độ trễ chung.
8. Thực hiện các kiểm định chẩn đoán phần dư của mô hình ARDL.

## Cài đặt

Tạo môi trường ảo và cài đặt các thư viện cần thiết:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Bắt đầu nhanh

Chạy quy trình mẫu với bộ dữ liệu `df_final.csv`:

```bash
python ardl_ffnn_model.py
```

Hoặc import mô hình vào một script hoặc notebook khác:

```python
from ardl_ffnn_model import ARDLFFNNModel

model = ARDLFFNNModel(
    data_path="df_final.csv",
    target="Export_US",
    features=["PCE", "US_Retail", "US_Sentiment", "USD_VND", "Import_CN"],
    name="US_Export",
    lag_candidates=range(0, 7),
)

model.run_all()
print(model.best_lag_map)
print(model.final_summary)
```

## Định hướng phát triển

- Tích hợp phương pháp phân rã mode thực nghiệm vào quy trình dự báo.
- Bổ sung bước xây dựng và đánh giá đặc trưng đa thang đo.
- Tái cấu trúc file `ardl_ffnn_model.py` sau khi đặc tả mô hình được hoàn thiện.
- Bổ sung các bài kiểm thử cho bước lựa chọn độ trễ, logic biến đổi dữ liệu và các chỉ số đánh giá dự báo.
- Tách riêng notebook, mã nguồn, dữ liệu và kết quả đầu ra theo cấu trúc package rõ ràng hơn.

## Ghi chú

Đây là repository của một dự án học thuật đang trong quá trình phát triển. File `ardl_ffnn_model.py` hiện chỉ là module chính tạm thời và sẽ tiếp tục được điều chỉnh khi phương pháp nghiên cứu được hoàn thiện hơn.

Trong workspace này, đường dẫn `.git` thông thường đã được môi trường sử dụng sẵn, vì vậy cơ sở dữ liệu Git được lưu trong thư mục `.git-store`. Khi sử dụng các lệnh Git, hãy chạy theo cú pháp sau:

```bash
GIT_DIR=.git-store GIT_WORK_TREE=. git status
```