# spoof_image_face
link dataset: [CelebA-spoof](https://pan.baidu.com/s/12qe13-jFJ9pE-_E3iSZtkw#list/path=%2F)

link dataset cameraip preprocessed: [Pre-Dataset](https://drive.google.com/drive/u/0/folders/17nBKBWs-A5mcee04rkBjFexA_wEjtLjU)

link paper : [Spoof image via LCD](http://www.gipsa-lab.fr/~kai.wang/papers/report_recap4n6.pdf)
## Introduction

Thực hiện phương pháp phân tích thống kê trên miền vi phân hình ảnh để phát hiện một bức ảnh là live hay spoof,với loại giả mạo trên màn hình điện thoại hay laptop.

## Quick Start

1. Download repo về directory, giải nén và cd tới "spoof_image_face".
2. Download dataset.zip tại "Pre-Dataset" giải nén và lưu vào spoof_image_face.
3. Chạy trình phân loại SVM để đưa ra dự đoán từ một hình ảnh đầu vào.

``
python .\evaluate.py -i path_to_image
``

Lệnh trên sẽ tải model SVM đã được training cùng với trình trích xuất đặc trưng thống kê sử dụng hệ số tương quan trên một hình ảnh đầu vào với đối số "path_to_image". Nếu là hình ảnh spoof thì output = 1, ngược lại là hình ảnh live thì output = 0.

## Training

Thực hiện lệnh sau

```
python training.py
```
Trong file training_and_testing.py thực hiện những công việc sau

1. Load 2 tập dữ liệu từ dir
2. Thực hiện trích xuất các đặc trưng thống kê sử dụng hệ số tương quan, mỗi hình ảnh sẽ đại diên cho vector 28 chiều
3. Concat tất cả vector thành một dataset thống nhất, tiền xử lý và chuẩn hóa dữ liệu để phù hợp với input của SVM
4. Thực hiện split dataset thành traning và testing
5. Training với SVM sử dụng kernel = "rbf"
6. Testing trên tập dữ liệu mới
7. Lưu dataset vector extracted và model dùng cho eval

## Some issues to know

1. Môi trường thử nghiêm code là:
  - Python 3.7.6
  - Tensorflow 2.0
  - Open CV 4.2.0
  - Sklearn 0.22.1
  - MTCNN
 2. Khi evaluate luôn phải nhớ giải nén 2 thư mục hình ảnh và model
 3. Chiến lược traning vẫn đang cải thiện để đạt được độ chính xác cao hơn.
