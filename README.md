# Image Classification Demo - Triton Inference Server

![Demo Screenshot](image.png)

## Giới thiệu

Dự án này cung cấp một **Demo Phân loại hình ảnh** sử dụng **Triton Inference Server**. Ứng dụng bao gồm giao diện đơn giản, nơi người dùng có thể tải lên hình ảnh và nhận được dự đoán phân loại kèm theo độ chính xác.

---

## Tính năng

- **Tải hình ảnh**: Cho phép người dùng kéo/thả hoặc chọn hình ảnh từ máy tính.
- **Dự đoán thời gian thực**: Hiển thị kết quả ngay lập tức sau khi tải hình ảnh.
- **Giao diện tương tác**: Giao diện trực quan, thân thiện với người dùng.
- **Tích hợp Triton**: Ứng dụng sử dụng Triton Inference Server để xử lý dự đoán nhanh chóng.

---

## Yêu cầu

### Môi trường
- **Python 3.8+**
- **Triton Inference Server** (để xử lý mô hình)

### Thư viện Python
Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
