import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

st.title("📈 Model Performance")

df = pd.read_csv("data/data.csv")
X = df.drop("Hasil", axis=1)
y = df["Hasil"]

# Konversi kategorik
X = pd.get_dummies(X)

# Split dan training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Simpan model
joblib.dump(model, "model.pkl")

st.code(classification_report(y_test, y_pred), language='text')
