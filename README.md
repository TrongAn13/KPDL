# DỰ ÁN: DỰ ĐOÁN SUY TIM SỬ DỤNG THUẬT TOÁN ID3

## 1\. TỔNG QUAN DỰ ÁN

Dự án này xây dựng một ứng dụng Web hỗ trợ chẩn đoán nguy cơ mắc bệnh tim (Heart Disease) dựa trên các chỉ số lâm sàng. Điểm nổi bật về mặt kỹ thuật của dự án là việc tự cài đặt (implement from scratch) thuật toán Cây quyết định ID3 (Iterative Dichotomiser 3) mà không phụ thuộc vào các thư viện học máy có sẵn như Scikit-learn cho phần lõi thuật toán.

Mục tiêu của dự án:

  * Xây dựng mô hình phân lớp có khả năng xử lý cả dữ liệu số và dữ liệu phân loại.
  * Cung cấp khả năng giải thích (Explainable AI), cho phép người dùng hiểu được logic suy luận của máy tính.
  * Tích hợp mô hình vào giao diện tương tác trực quan sử dụng Framework Streamlit.

## 2\. CẤU TRÚC HỆ THỐNG VÀ TỆP TIN

Dự án bao gồm các thành phần mã nguồn và dữ liệu sau:

### 2.1. Mã nguồn (Source Code)

  * **`id3_core.py`**: Chứa mã nguồn lõi của thuật toán. Định nghĩa lớp đối tượng `TreeNode` và `ID3DecisionTree`. Tệp này thực hiện các tính toán toán học bao gồm Entropy, Information Gain (Độ lợi thông tin) và logic tìm ngưỡng tối ưu cho biến số thực.
  * **`id3_from_scratch.py`**: Đảm nhiệm chức năng tiền xử lý và làm sạch dữ liệu. Bao gồm các logic xử lý giá trị khuyết thiếu (missing values) và loại bỏ dữ liệu nhiễu.
  * **`id3_train_optimized.py`**: Script thực thi quy trình huấn luyện. Tệp này đọc dữ liệu sạch, khởi tạo thuật toán từ `id3_core.py`, huấn luyện mô hình và lưu trữ mô hình đã học ra file định dạng `.pkl`.
  * **`app.py`**: Ứng dụng giao diện người dùng (Frontend & Backend). Tệp này tải mô hình đã huấn luyện, hiển thị biểu mẫu nhập liệu và trả về kết quả dự đoán cùng đường dẫn suy luận.

### 2.2. Dữ liệu và Mô hình (Data & Model)

  * **`heart.csv`**: Tập dữ liệu thô ban đầu (Raw Data).
  * **`cleaned_heart.csv`**: Tập dữ liệu sau khi đã qua bước làm sạch và chuẩn hóa.
  * **`best_id3_model.pkl`**: File nhị phân chứa mô hình cây quyết định sau khi huấn luyện thành công. File này được ứng dụng web sử dụng để dự đoán thời gian thực.

## 3\. PHƯƠNG PHÁP LUẬN VÀ KỸ THUẬT

### 3.1. Thuật toán ID3 Cải tiến

Khác với thuật toán ID3 nguyên bản chỉ xử lý được dữ liệu phân loại (Categorical), phiên bản cài đặt trong dự án này đã được cải tiến để xử lý dữ liệu số (Numeric) như Tuổi, Huyết áp, Cholesterol.

  * **Cơ chế:** Thuật toán tự động sắp xếp các giá trị số và thử nghiệm các ngưỡng cắt (threshold) tại trung điểm của các giá trị liền kề để tìm ra điểm phân chia có Độ lợi thông tin lớn nhất.
  * **Tiêu chuẩn phân chia:** Sử dụng Entropy để đo độ hỗn loạn và Information Gain để chọn thuộc tính phân loại.

### 3.2. Khả năng giải thích (Explainability)

Hệ thống được thiết kế để minh bạch hóa quá trình ra quyết định. Khi đưa ra kết quả (Bệnh/Không bệnh), hệ thống truy xuất ngược đường đi từ nút lá (Leaf node) lên nút gốc (Root node) để cung cấp chuỗi logic.

  * *Ví dụ minh họa:* Kết luận "Có bệnh" được đưa ra dựa trên cơ sở: Độ dốc ST = Up VÀ Tuổi \> 55 VÀ Cholesterol \<= 200.

## 4\. HƯỚNG DẪN CÀI ĐẶT VÀ SỬ DỤNG

### 4.1. Yêu cầu môi trường

Dự án yêu cầu cài đặt ngôn ngữ lập trình Python và các thư viện phụ thuộc sau:

  * pandas
  * numpy
  * scikit-learn (chỉ dùng để chia tập train/test và tính toán độ đo)
  * streamlit

Lệnh cài đặt:

```bash
pip install pandas numpy scikit-learn streamlit
```

### 4.2. Quy trình vận hành

**Bước 1: Huấn luyện mô hình**
Trước khi khởi chạy ứng dụng, cần thực hiện huấn luyện để sinh ra file mô hình. Chạy lệnh:

```bash
python id3_train_optimized.py
```

*Kết quả:* Hệ thống sẽ in ra cấu trúc cây quyết định dạng văn bản và độ chính xác trên tập kiểm thử, đồng thời tạo ra file `best_id3_model.pkl`.

**Bước 2: Khởi chạy ứng dụng Web**
Sử dụng lệnh sau để mở giao diện người dùng:

```bash
streamlit run app.py
```

Ứng dụng sẽ tự động mở trên trình duyệt mặc định tại địa chỉ Localhost (thường là cổng 8501).

## 5\. ĐÁNH GIÁ HIỆU NĂNG

Mô hình được huấn luyện và đánh giá trên bộ dữ liệu Heart Failure Prediction. Dữ liệu được chia theo tỷ lệ 80% cho huấn luyện và 20% cho kiểm thử.

  * Độ chính xác (Accuracy): [80.98%]
  * Khả năng tổng quát hóa: Mô hình sử dụng tham số `min_samples_split` để hạn chế hiện tượng quá khớp (overfitting).

## 6\. THÔNG TIN TÁC GIẢ

  * Tên dự án: Dự đoán Suy tim ID3
  * Người thực hiện: [Hoàng Trọng An]
  * Mục đích: Nghiên cứu khoa học và ứng dụng thuật toán cây quyết định.
