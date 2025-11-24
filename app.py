import streamlit as st
import pandas as pd
import pickle
from id3_core import ID3DecisionTree # Bắt buộc import để hiểu được file model


# 1. LOAD MÔ HÌNH ĐÃ CÓ SẴN (KHÔNG TRAIN LẠI)
@st.cache_resource
def load_model():
    try:
        # Đọc file model đã train từ trước 
        with open("best_id3_model.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        return None


# 2. GIAO DIỆN NGƯỜI DÙNG

st.set_page_config(page_title="Dự đoán bệnh tim", layout="wide")
st.title("Ứng dụng Chẩn đoán Bệnh tim (ID3)")
st.caption("Chạy trên mô hình đã được huấn luyện trước (Pre-trained Model)")

# Load model
model = load_model()

if model is None:
    st.error("LỖI: Không tìm thấy file `best_id3_model.pkl`!")
    st.warning("Hãy chạy file `train_id3_optimized.py` trước để tạo file mô hình.")
    st.stop()

# TỶ LỆ : [2, 1] 
left_col, right_col = st.columns([2, 1]) 

with left_col:
    st.markdown("###  Nhập thông tin bệnh nhân")
    st.markdown("---")
    with st.form("input_form"):

        c1, c2, c3 = st.columns(3)
        
        with c1:
            age = st.number_input("Tuổi", 1, 120, 55)
            sex = st.selectbox("Giới tính", ["M", "F"])
            chest_pain = st.selectbox("Đau ngực (ChestPain)", ["ASY", "NAP", "ATA", "TA"])
            resting_bp = st.number_input("Huyết áp nghỉ", 50, 250, 130)
            
        with c2:
            cholesterol = st.number_input("Cholesterol", 50, 600, 220)
            fasting_bs = st.selectbox("Đường huyết > 120?", [0, 1], format_func=lambda x: "Có" if x==1 else "Không")
            resting_ecg = st.selectbox("Điện tâm đồ", ["Normal", "LVH", "ST"])
            max_hr = st.number_input("Nhịp tim Max", 60, 220, 140)
            
        with c3:
            exercise_angina = st.selectbox("Đau khi tập?", ["Y", "N"])
            oldpeak = st.number_input("Oldpeak", -5.0, 10.0, 0.0, step=0.1)
            st_slope = st.selectbox("Độ dốc ST", ["Flat", "Up", "Down"])

        st.markdown("") # Tạo khoảng trống
        submitted = st.form_submit_button(" Dự đoán ngay", type="primary", use_container_width=True)


# 3. DỰ ĐOÁN & HIỂN THỊ

with right_col:
    st.markdown("###  Kết quả")
    st.markdown("---")
    
    if submitted:
        # Gom dữ liệu input
        input_data = pd.Series({
            "Age": age, "Sex": sex, "ChestPainType": chest_pain,
            "RestingBP": resting_bp, "Cholesterol": cholesterol,
            "FastingBS": fasting_bs, "RestingECG": resting_ecg,
            "MaxHR": max_hr, "ExerciseAngina": exercise_angina,
            "Oldpeak": oldpeak, "ST_Slope": st_slope
        })

        # Dự đoán
        pred, reason_path = model.predict_one_with_reason(input_data)

        if pred == 1:
            st.error("NGUY CƠ CAO")
            st.write("Có dấu hiệu bệnh tim.")
        else:
            st.success("AN TOÀN")
            st.write("Nguy cơ thấp.")

        # In đường dẫn suy luận
        with st.expander(" Xem lý do suy luận", expanded=True):
            path_str = ""
            for i, step in enumerate(reason_path):
                prefix = "└─ " if i == len(reason_path) - 1 else "├─ "
                path_str += f"**{prefix}{step}**\n\n"
            st.markdown(path_str)

    else:
        st.info(" Vui lòng nhập thông tin bên trái.")