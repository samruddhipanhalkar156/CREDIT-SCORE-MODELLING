# app.py
import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

# -------------------------
# Paths
# -------------------------
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "Models"
RESULTS_DIR = BASE_DIR / "Results"

# -------------------------
# Load preprocessor & scaler
# -------------------------
with open(MODELS_DIR / "preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

with open(MODELS_DIR / "scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# -------------------------
# Load label classes and feature names from metrics file
# -------------------------
metrics_file = RESULTS_DIR / "model_metrics.xlsx"
if metrics_file.exists():
    metrics_df = pd.read_excel(metrics_file)
else:
    metrics_df = pd.DataFrame()

# You can also store label_classes & feature_names manually if needed
label_classes = ["Poor", "Standard", "Good"]  # Adjust as per your training data
feature_names = [
    "Age","Annual_Income","Monthly_Inhand_Salary","Num_Bank_Accounts","Num_Credit_Card",
    "Interest_Rate","Num_of_Loan","Delay_from_due_date","Num_of_Delayed_Payment",
    "Changed_Credit_Limit","Num_Credit_Inquiries","Credit_Mix","Outstanding_Debt",
    "Credit_Utilization_Ratio","Credit_History_Age","Payment_of_Min_Amount",
    "Total_EMI_per_month","Amount_invested_monthly","Payment_Behaviour","Monthly_Balance",
    "Occupation"
]

# -------------------------
# Model paths
# -------------------------
MODEL_PATHS = {
    "Logistic Regression": MODELS_DIR / "LR" / "LogisticRegression.pkl",
    "Decision Tree": MODELS_DIR / "DT" / "DecisionTree.pkl",
    "Random Forest": MODELS_DIR / "RF" / "RandomForest.pkl",
}

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Credit Score Prediction", layout="wide")
st.title("📊 Credit Score Prediction System")

# -------------------------
# Model Selection
# -------------------------
model_choice = st.selectbox("Choose a trained model:", list(MODEL_PATHS.keys()))
with open(MODEL_PATHS[model_choice], "rb") as f:
    model = pickle.load(f)

# Show metrics if available
if not metrics_df.empty:
    st.write(f"### Metrics for {model_choice}")
    st.dataframe(metrics_df[metrics_df["Model"] == model_choice.replace(" ", "")])

# -------------------------
# Single Input Prediction
# -------------------------
st.subheader("Single Input Prediction")
inputs = {}
cols = st.columns(4)
for i, feature in enumerate(feature_names):
    with cols[i % 4]:
        val = st.text_input(feature, "0", key=f"feat_{i}")
        try:
            val = float(val)
        except:
            val = 0.0
        inputs[feature] = val

if st.button("Predict Single Input"):
    single_df = pd.DataFrame([inputs])
    # Transform categorical columns
    single_transformed = preprocessor.transform(single_df)
    single_scaled = scaler.transform(single_transformed)
    pred = model.predict(single_scaled)[0]
    st.success(f"Predicted Credit Score: **{label_classes[pred]}**")

# -------------------------
# Batch Prediction via CSV
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
    user_df["Predicted_Credit_Score"] = [label_classes[p] for p in preds]

    st.write("### Predictions")
    st.dataframe(user_df.head())

    csv = user_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Predictions", data=csv, file_name="predictions.csv", mime="text/csv")

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center;">
        <p><strong>Created by: Samruddhi R. Panhalkar</strong></p>
        <p>📧 samruddhipanhalkar156@gmail.com </p>
        <p>🏫 Maratha Mandal Engineering College</p>
    </div>
    """,
    unsafe_allow_html=True,
)
