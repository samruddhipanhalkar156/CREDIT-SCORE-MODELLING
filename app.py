# app.py
import streamlit as st
import pandas as pd
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# -------------------------
# Paths
# -------------------------
MODELS_DIR = Path("Models")
RESULTS_DIR = Path("Results")

LR_MODEL_PATH = MODELS_DIR / "LR" / "LogisticRegression.pkl"
DT_MODEL_PATH = MODELS_DIR / "DT" / "DecisionTree.pkl"
RF_MODEL_PATH = MODELS_DIR / "RF" / "RandomForest.pkl"

SCALER_PATH = MODELS_DIR / "scaler.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"

# -------------------------
# Load scaler and preprocessor
# -------------------------
with open(PREPROCESSOR_PATH, "rb") as f:
    preprocessor = pickle.load(f)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

# Feature names (numerical only)
feature_names = [
    "Age", "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts",
    "Num_Credit_Card", "Interest_Rate", "Num_of_Loan", "Delay_from_due_date",
    "Num_of_Delayed_Payment", "Changed_Credit_Limit", "Num_Credit_Inquiries",
    "Outstanding_Debt", "Credit_Utilization_Ratio", "Credit_History_Age",
    "Total_EMI_per_month", "Amount_invested_monthly", "Monthly_Balance"
]

# -------------------------
# Load models
# -------------------------
with open(LR_MODEL_PATH, "rb") as f:
    lr_model = pickle.load(f)

with open(DT_MODEL_PATH, "rb") as f:
    dt_model = pickle.load(f)

with open(RF_MODEL_PATH, "rb") as f:
    rf_model = pickle.load(f)

MODELS = {
    "Logistic Regression": lr_model,
    "Decision Tree": dt_model,
    "Random Forest": rf_model
}

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Credit Score Prediction", layout="wide")
st.title("📊 Credit Score Prediction System")

# Model selection
model_choice = st.selectbox("Choose a trained model:", list(MODELS.keys()))
model = MODELS[model_choice]

# -------------------------
# Single Input Prediction
# -------------------------
st.subheader("Single Input Prediction")
inputs = {}
cols = st.columns(4)
for i, feature in enumerate(feature_names):
    with cols[i % 4]:
        val = st.number_input(feature, value=0.0, step=1.0, format="%.2f")
        inputs[feature] = val

if st.button("Predict Single Input"):
    single_df = pd.DataFrame([inputs])
    single_preprocessed = preprocessor.transform(single_df)
    single_scaled = scaler.transform(single_preprocessed)
    pred = model.predict(single_scaled)[0]
    st.success(f"Predicted Credit Score: **{pred}**")

# -------------------------
# Batch Prediction
# -------------------------
st.subheader("Batch Prediction via CSV")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    user_df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Preview")
    st.dataframe(user_df.head())

    transformed = preprocessor.transform(user_df)
    scaled = scaler.transform(transformed)
    preds = model.predict(scaled)
    user_df["Predicted_Credit_Score"] = preds

    st.write("### Predictions")
    st.dataframe(user_df.head())

    csv = user_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Predictions", data=csv, file_name="predictions.csv", mime="text/csv")

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center;'>Created by: Samruddhi R. Panhalkar | Maratha Mandal Engineering College</p>",
    unsafe_allow_html=True
)
