
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("📊 Dashboard - Exploratory Data Analysis")
df = pd.read_csv("data/data.csv")

st.subheader("Tampilan Data")
st.dataframe(df)

st.subheader("Distribusi Target (Hasil)")
st.bar_chart(df["Hasil"].value_counts())

st.subheader("Usia vs Hasil")
fig, ax = plt.subplots()
sns.histplot(data=df, x="Usia", hue="Hasil", multiple="stack", ax=ax)
st.pyplot(fig)
