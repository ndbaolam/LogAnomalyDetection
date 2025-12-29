# Phát hiện chuỗi log bất thường theo request / instance (Sequence Anomaly Detection)
1. Định nghĩa bài toán (Problem Definition)
1.1 Bối cảnh

Trong hệ thống Cloud / Distributed System (OpenStack),
mỗi yêu cầu (request) hoặc mỗi máy ảo (instance) tạo ra một chuỗi log sự kiện theo thời gian.

Các chuỗi này thường ổn định và có quy luật, phản ánh luồng xử lý bình thường của hệ thống.

👉 Khi xảy ra sự cố:

Thứ tự sự kiện bị thay đổi

Một số sự kiện bị thiếu hoặc lặp bất thường

Xuất hiện các chuỗi chưa từng thấy

1.2 Mục tiêu bài toán

Học hành vi chuỗi log bình thường, sau đó:

Phát hiện chuỗi bất thường

Phát hiện sự kiện bất thường trong chuỗi

Xác định vị trí gây bất thường

# Run Project
```sh
export $(cat .env | xargs)
```