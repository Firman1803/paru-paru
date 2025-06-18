import streamlit as st
import pandas as pd
import joblib

st.title("🩺 Prediksi Penyakit Paru-Paru")

model = joblib.load("model.pkl")

def predict(input_df):
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=model.feature_names_in_, fill_value=0)
    return model.predict(input_encoded)[0]

# Input form
usia = st.slider("Usia", 1, 100, 30)
jk = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
merokok = st.selectbox("Merokok", ["Ya", "Tidak"])
bekerja = st.selectbox("Bekerja", ["Ya", "Tidak"])
rumahtangga = st.selectbox("Rumah Tangga", ["Ya", "Tidak"])
begadang = st.selectbox("Begadang", ["Ya", "Tidak"])
olahraga = st.selectbox("Olahraga", ["Ya", "Tidak"])
asuransi = st.selectbox("Asuransi", ["Ya", "Tidak"])
penyakit = st.selectbox("Penyakit Bawaan", ["Ya", "Tidak"])

data = {
    "Usia": [usia],
    "Jenis Kelamin": [jk],
    "Merokok": [merokok],
    "Bekerja": [bekerja],
    "Rumah Tangga": [rumahtangga],
    "Begadang": [begadang],
    "Olahraga": [olahraga],
    "Asuransi": [asuransi],
    "Penyakit Bawaan": [penyakit],
}

input_df = pd.DataFrame(data)

if st.button("Prediksi"):
    hasil = predict(input_df)
    st.success(f"Hasil Prediksi: **{hasil}**")

import os

if st.button("Prediksi dan Simpan"):
    hasil = predict(input_df)
    input_df["Prediksi"] = hasil
    if not os.path.exists("data/history.csv"):
        input_df.to_csv("data/history.csv", index=False)
    else:
        input_df.to_csv("data/history.csv", mode="a", header=False, index=False)
    st.success(f"Hasil Prediksi: **{hasil}** (juga disimpan ke riwayat)")
