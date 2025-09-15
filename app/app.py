import streamlit as st
import pandas as pd
import joblib

# ===============================
# 1. LOAD PIPELINE
# ===============================
MODEL_PATH = "stroke_pipeline.pkl"

@st.cache_resource
def load_pipeline(path):
    return joblib.load(path)

pipeline = load_pipeline(MODEL_PATH)

# ===============================
# 2. STREAMLIT UI
# ===============================
st.set_page_config(page_title="Prediksi Penyakit Stroke", page_icon="🧠", layout="centered")

st.title("🧠 Prediksi Penyakit Stroke")
st.markdown("""
Aplikasi ini menggunakan **Logistic Regression** untuk memprediksi kemungkinan seseorang terkena stroke.
Silakan masukkan data pasien di bawah ini.
""")

# ===============================
# 3. INPUT USER
# ===============================
gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
age = st.slider("Umur", 0, 100, 25)
hypertension = st.selectbox("Hipertensi", [0, 1], help="0 = Tidak, 1 = Ya")
heart_disease = st.selectbox("Penyakit Jantung", [0, 1], help="0 = Tidak, 1 = Ya")
ever_married = st.selectbox("Pernah Menikah", ["Yes", "No"])
work_type = st.selectbox("Jenis Pekerjaan", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
Residence_type = st.selectbox("Tipe Tempat Tinggal", ["Urban", "Rural"])
avg_glucose_level = st.number_input("Rata-rata Glukosa", min_value=0.0, value=100.0)
bmi = st.number_input("BMI", min_value=0.0, value=25.0)
smoking_status = st.selectbox("Status Merokok", ["formerly smoked", "never smoked", "smokes", "Unknown"])

# Buat dataframe dari input user
input_data = pd.DataFrame([{
    "gender": gender,
    "age": age,
    "hypertension": hypertension,
    "heart_disease": heart_disease,
    "ever_married": ever_married,
    "work_type": work_type,
    "Residence_type": Residence_type,
    "avg_glucose_level": avg_glucose_level,
    "bmi": bmi,
    "smoking_status": smoking_status
}])

# ===============================
# 4. PREDIKSI
# ===============================
if st.button("🔍 Prediksi Stroke"):
    try:
        prediction = pipeline.predict(input_data)[0]
        prob = pipeline.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.error(f"⚠️ Pasien **berisiko Stroke** dengan probabilitas {prob:.2%}")
        else:
            st.success(f"✅ Pasien **tidak berisiko Stroke** dengan probabilitas {1-prob:.2%}")

        st.write("### Detail Probabilitas")
        st.progress(float(prob))

    except Exception as e:
        st.exception(f"Terjadi error saat prediksi: {e}")

# ===============================
# 5. FOOTER
# ===============================
st.markdown("---")
st.caption("Dibuat dengan ❤️ menggunakan Streamlit & Logistic Regression")
