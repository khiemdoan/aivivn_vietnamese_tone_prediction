# Thêm dấu tiếng Việt

Dữ liệu của cuộc thi được cho tại [vietnamese_tone_prediction](https://drive.google.com/file/d/1m_5CDQQSavev5zWb8JUq97_zUTnOcVvS/view).

Cấu trúc của bộ dữ liệu như sau:

```base
vietnamese_tone_prediction
    .
    ├── sample_submission.csv
    ├── test.txt
    ├── test_word_per_line.txt
    └── train.txt
```

## Dữ liệu huấn luyện
Dữ liệu huấn luyện (file train.txt) gồm 2,126,280 dòng. Mỗi dòng là một câu hoặc một đoạn văn. Dữ liệu này được crawl từ nhiều nguồn và chưa được xử lý, có thể lẫn các từ không phải tiếng Việt nhưng tỉ lệ nhỏ.

## Dữ liệu kiểm tra
Dữ liệu kiểm tra (file test.txt) bao gồm 8240 dòng, mỗi dòng là một câu hoặc một đoạn văn đã được bỏ dấu. Ba ký tự đầu tiên trước dấu phẩy (,) của mỗi dòng là mã dòng. Tương tự như các cuộc thi trước, dữ liệu kiểm tra công khai và bí mật đã được trộn lẫn vào nhau một cách ngẫu nhiên. Dữ liệu công khai để hiển thị điểm trong quá trình cuộc thi diễn ra. Dữ liệu bí mật để tính kết quả cuối cùng.

## Cách mã hóa tiếng Việt có dấu
Để đồng nhất việc đánh giá kết quả, các từ tiếng Việt được mã hóa dưới dạng VNI chuẩn hóa. Cụ thể:

```
â = a6, Â = A6
ă = a8, Ă = A8
đ = d9, Đ = D9
ê = e6, Ê = E6
ô = o6, Ô = O6
ơ = o7, Ơ = O7
ư = u7, Ư = U7
sắc = 1
huyền = 2
hỏi = 3
ngã = 4
nặng = 5
```

Phần dấu được đẩy về cuối của mỗi chữ. Một vài ví dụ:

```
Con = Con
Đường = D9u7o7ng2
Trí = Tri1
Tuệ = Tue65
Nhân = Nha6n
Tạo = Tao5
```

Trong file nộp bài, nhãn của mỗi chữ là dãy chữ số viết theo thứ tự chuẩn hóa VNI.

| **không dấu**  |  **Dự đoán**  | **VNI chuẩn hóa** | **Nộp bài**  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Con            | Con           | Con               | 0            |
| Duong          | Đường         | D9u7o7ng2         | 9772         |
| Tri            | Trí           | Tri1              | 1            |
| Tue            | Tuệ           | Tue65             | 65           |
| nhan           | nhân          | nha6n             | 6            |
| tao            | tạo           | tao5              | 5            |

## Nộp bài
File nộp bài bắt buộc là một file .csv có 453,447 dòng, bao gồm một dòng tiêu đề 453,446 dòng tương ứng với kết quả dự đoán của từng từ ở dạng chuẩn hóa VNI (Xem thêm github repo).

Mã số của mỗi từ có dạng xyz012 trong đó xyz là mã số dòng, 012 là số thứ tự của từ trong dòng, bắt đầu từ 000. Mỗi dòng đã được đảm bảo không quá 1000 từ. Ví dụ: Nếu một dòng trong file test.txt là

`aBc, Con Duong Tri Tue nhan tao`

và mô hình của bạn dự đoán kết quả là Con Đường Trí Tuệ nhân tạo thì file nộp bài tương ứng có dạng:

```
id,label
aBc000,0
aBc001,9772
aBc003,1
aBc004,65
aBc005,6
aBc006,5
...
```

Trong đó dòng đầu tiên là tiêu đề của file, bắt buộc phải là id,label (không có dấu cách).

Để thuận tiện cho việc đối chiếu kết quả, file `test_word_per_line.txt` bao gồm các chữ đã được bỏ dấu tương ứng.

Các hàm phụ trợ
Các hàm phụ trợ cho cuộc thi được cho tại [Github repo này](https://github.com/aivivn/vietnamese_tone_prediction_utils)

Hàm `remove_ton_file(in_path, out_path) ` giúp bỏ dấu của một file tiếng Việt.

Hàm `decompose_predicted_test_file` giúp tạo file nộp bài sau khi đã điền dấu cho file `test.txt`.