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

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt

# Confusion Matrix
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
fig, ax = plt.subplots()
disp.plot(ax=ax)
st.pyplot(fig)

# ROC Curve
st.subheader("ROC Curve")
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label=y.unique()[1])
roc_auc = auc(fpr, tpr)
fig, ax = plt.subplots()
ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend()
st.pyplot(fig)

