# app/app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import json
from pathlib import Path

# --- config ---
MODEL_PATH = Path(__file__).parent / 'model' / 'stroke_pipeline.joblib'
METADATA_PATH = Path(__file__).parent / 'model' / 'artifacts_metadata.json'

st.set_page_config(page_title='Stroke Prediction', layout='centered')

st.title('Prediksi Risiko Penyakit Stroke (Logistic Regression)')
st.write('Catatan: ini hanya model pembelajaran — bukan diagnosis medis.')

# Load model
@st.cache_resource
def load_pipeline(path):
    return joblib.load(path)

pipeline = load_pipeline(MODEL_PATH)

# Load metadata
with open(METADATA_PATH,'r') as f:
    metadata = json.load(f)

# Build input form
with st.form('input_form'):
    st.header('Masukkan data pasien')

    # Numeric inputs
    numeric_inputs = {}
    for n in metadata['numeric_features']:
        # inference default values from metadata not provided here -> use placeholder
        if n == 'age':
            val = st.number_input('Age', min_value=0.0, max_value=120.0, value=50.0)
        elif n == 'avg_glucose_level':
            val = st.number_input('Average Glucose Level', value=100.0)
        elif n == 'bmi':
            val = st.number_input('BMI', value=25.0)
        else:
            val = st.number_input(n, value=0.0)
        numeric_inputs[n] = val

    # Categorical inputs: show options from metadata
    categorical_inputs = {}
    for c, vals in metadata['categorical_features'].items():
        # Show selectbox with str options
        choice = st.selectbox(c, options=[str(x) for x in vals])
        categorical_inputs[c] = choice

    submitted = st.form_submit_button('Prediksi')

if submitted:
    # build DataFrame in same column order as training X
    input_df = pd.DataFrame([ {**numeric_inputs, **categorical_inputs} ])

    # Model pipeline handles preprocessing; just pass the raw input
    prob = pipeline.predict_proba(input_df)[0][1]
    label = int(pipeline.predict(input_df)[0])

    st.subheader('Hasil Prediksi')
    st.write('Probabilitas terindikasi stroke: **{:.3f}**'.format(prob))
    st.write('Prediksi label: **{}**'.format('Terindikasi Stroke' if label==1 else 'Tidak Terindikasi Stroke'))

    # Simple explanation using coefficients (if logistic) — attempt to show top positive contributors
    try:
        classifier = pipeline.named_steps['classifier']
        preproc = pipeline.named_steps['preprocessor']
        # get feature names after preprocessing
        ohe = preproc.named_transformers_['cat'].named_steps['onehot']
        cat_cols = preproc.transformers_[1][2]
        ohe_names = list(ohe.get_feature_names_out(cat_cols))
        feat_names = preproc.transformers_[0][2] + ohe_names
        coefs = classifier.coef_[0]
        feat_imp = pd.DataFrame({'feature':feat_names,'coef':coefs})
        feat_imp = feat_imp.reindex(feat_imp.coef.abs().sort_values(ascending=False).index)
        st.write('Top fitur berpengaruh (koefisien):')
        st.write(feat_imp.head(8))
    except Exception as e:
        # ignore if structure different
        st.write('Ringkasan fitur tidak tersedia (error: {})'.format(e))

# Footer
st.markdown('---')
st.caption('Model dibuat & disimpan di Google Colab; aplikasi memuat pipeline joblib yang berisi preprocessing + model.')