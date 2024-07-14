# Helmet Detection

## Giới thiệu

## PHƯƠNG HƯỚNG PHÁT TRIỂN


### Model

Train lại từ đầu model YOLOv5 train trên tập dữ liệu tự thu thập.

Fine-tune mô hình, điều chỉnh các hyperparameters, thay đổi kiến trúc mạng.

Có thể ứng dụng transfer learning kết hợp với machine learning và các model deep learning khác.

### Input

Camera theo thời gian thực, ảnh, video hình ảnh đường phố.

### Output

Vẽ bounding box và nhãn kết quả lên hình ảnh gốc.

Đảm bảo thời gian gần thời gian thực.

(optional) Trả về thông báo khi người không đội mũ được phát hiện

## Dataset

Thu thập và annotate dataset thông qua cắt ảnh từ camera đường phố và chụp ảnh những người tham gia giao thông trực tiếp trên đường, camera đường phố ảnh trên mạng.
- Dự kiến >1000 ảnh và >5000 instance cho mỗi class
- Hình ảnh đa dang, từ các thời điểm khác nhau trong ngày, thời tiết khác nhau, ánh sáng khác nhau, các góc khác nhau
- Hình background chiếm khoảng 1% tổng số hình ảnh, hình ảnh không có đối tượng nào được thêm vào tập dữ liệu để giảm kết quả FP

## Thuật toán xử lý hình ảnh và video để ứng dụng thời gian thực

- Xử lý hình ảnh ban đêm: gamma correction và histogram equalization
- Tăng fps: frame skipping


