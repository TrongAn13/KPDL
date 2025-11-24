import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from id3_core import ID3DecisionTree  

def train_and_evaluate():
    # 1. Đọc dữ liệu ĐÃ LÀM SẠCH
    try:
        df = pd.read_csv("cleaned_heart.csv")
        print("Đã load dữ liệu 'cleaned_heart.csv'.")
    except FileNotFoundError:
        print("LỖI: Hãy chạy file clean_data_optimized.py trước!")
        return

    # 2. Tách X, y
    X = df.drop("HeartDisease", axis=1)
    y = df["HeartDisease"]

    # 3. Định nghĩa loại dữ liệu cho ID3
    # (ID3 Core cần biết cột nào là số để tìm ngưỡng cắt)
    feature_types = {
        "Age": "numeric",
        "Sex": "categorical",
        "ChestPainType": "categorical",
        "RestingBP": "numeric",
        "Cholesterol": "numeric",
        "FastingBS": "categorical", # FastingBS chỉ có 0 và 1 -> coi là categorical cũng được
        "RestingECG": "categorical",
        "MaxHR": "numeric",
        "ExerciseAngina": "categorical",
        "Oldpeak": "numeric",
        "ST_Slope": "categorical",
    }

    # 4. Chia tập Train/Test (80-20)
    # Stratify để đảm bảo tỷ lệ bệnh nhân trong 2 tập đều nhau
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nKích thước Train: {X_train.shape}")
    print(f"Kích thước Test:  {X_test.shape}")

    # 5. Huấn luyện mô hình
    print("\nĐang huấn luyện ID3 (có tìm ngưỡng tối ưu cho số thực)...")
    # min_samples_split=10: Ngăn cây mọc quá chi tiết (tránh học vẹt)
    tree = ID3DecisionTree(feature_types=feature_types, min_samples_split=10)
    tree.fit(X_train, y_train)
    print("-> Huấn luyện xong!")

    # 6. Đánh giá
    print("\n--- KẾT QUẢ ĐÁNH GIÁ ---")
    y_pred = tree.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Độ chính xác (Accuracy): {acc:.4f} ({acc*100:.2f}%)")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # 7. Lưu mô hình (để dùng cho App sau này)
    with open("best_id3_model.pkl", "wb") as f:
        pickle.dump(tree, f)
    print("\nĐã lưu mô hình vào 'best_id3_model.pkl'")

    # # 8. IN CẤU TRÚC CÂY
    # print("\n--- CẤU TRÚC CÂY QUYẾT ĐỊNH ---")
    # tree.print_tree()
    with open("tree_structure.txt", "w", encoding="utf-8") as f:
        # Dùng contextlib để chuyển hướng lệnh print vào file f
        from contextlib import redirect_stdout
        with redirect_stdout(f):
            print("CẤU TRÚC CÂY QUYẾT ĐỊNH ID3")
            print("="*40)
            tree.print_tree()
            print("="*40)
    print("Đã lưu file cấu trúc cây thành công.")

    # 9. DỰ ĐOÁN THỬ 1 BỆNH NHÂN VÀ XEM LÝ DO
    sample_patient = X_test.iloc[0] # Lấy người đầu tiên trong tập test
    pred, reason = tree.predict_one_with_reason(sample_patient)

    print("\n--- GIẢI THÍCH DỰ ĐOÁN ---")
    print(f"Dự đoán: {'BỆNH TIM' if pred==1 else 'BÌNH THƯỜNG'}")
    print("Lý do suy luận:")
    print(" -> ".join(reason))

if __name__ == "__main__":
    train_and_evaluate()