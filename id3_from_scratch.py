import pandas as pd
import numpy as np

def clean_data():
    input_path = "heart.csv"
    output_path = "cleaned_heart.csv"

    # 1. Đọc dữ liệu
    try:
        df = pd.read_csv(input_path)
        print(f"Đã đọc dữ liệu gốc: {df.shape}")
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file heart.csv")
        return

    # 2. Xóa dữ liệu trùng lặp
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        df = df.drop_duplicates()
        print(f" + Đã xóa {dup_count} dòng trùng lặp.")

    # 3. Xử lý Huyết áp (RestingBP) = 0 -> Vô lý về y khoa -> Xóa
    initial_len = len(df)
    df = df[df['RestingBP'] > 0]
    print(f" + Đã xóa {initial_len - len(df)} dòng có Huyết áp = 0.")

    # 4. Xử lý Cholesterol = 0 -> Đây là lỗi missing value
    # Thay thế bằng trung vị (median) của những người có chỉ số > 0
    median_chol = df[df['Cholesterol'] > 0]['Cholesterol'].median()
    count_zero_chol = len(df[df['Cholesterol'] == 0])
    
    if count_zero_chol > 0:
        df['Cholesterol'] = df['Cholesterol'].replace(0, median_chol)
        print(f" + Đã thay thế {count_zero_chol} giá trị Cholesterol=0 bằng Median ({median_chol}).")

    # 5. Xóa các dòng thiếu dữ liệu (NaN) nếu còn sót
    if df.isnull().sum().sum() > 0:
        df = df.dropna()
        print(" + Đã xóa các dòng chứa NaN.")

    # 6. Lưu file sạch
    df.to_csv(output_path, index=False)
    print(f"\nDữ liệu sạch đã lưu tại: {output_path}")
    print(f"Kích thước cuối cùng: {df.shape}")

if __name__ == "__main__":
    clean_data()