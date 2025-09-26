# app.py
import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

# -------------------------
# Paths
# -------------------------
MODELS_DIR = Path("Models")
RESULTS_DIR = Path("Model Results")

# Load latest results
results_file = RESULTS_DIR / "model_results.xlsx"
metrics_df = pd.read_excel(results_file) if results_file.exists() else pd.DataFrame()

# Load scaler
scaler_path = MODELS_DIR / "scaler.pkl"
if not scaler_path.exists():
    st.error(" scaler.pkl not found! Please run main.py first.")
    st.stop()
else:
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

# Load feature names
feature_names_path = MODELS_DIR / "feature_names.pkl"
if not feature_names_path.exists():
    st.error(" feature_names.pkl not found! Please run main.py first.")
    st.stop()
else:
    with open(feature_names_path, "rb") as f:
        feature_names_model = pickle.load(f)

# Occupation columns
occupation_columns = [col for col in feature_names_model if col.startswith("Occupation_")]

# Load models
MODEL_PATHS = {
    "LogisticRegression": MODELS_DIR / "LogisticRegression.pkl",
    "DecisionTree": MODELS_DIR / "DecisionTree.pkl",
    "RandomForest": MODELS_DIR / "RandomForest.pkl",
}

# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="💳 Credit Score Prediction", layout="wide")
st.title("💳 Credit Score Prediction System")

# Model selection
model_choice = st.selectbox("Choose a trained model:", list(MODEL_PATHS.keys()))

# Show metrics if available
if not metrics_df.empty:
    st.write("### Model Metrics")
    metrics_model = metrics_df[metrics_df["Model"] == model_choice]
    st.dataframe(metrics_model)

# Load model
model_path = MODEL_PATHS[model_choice]
if not model_path.exists():
    st.error(f" {model_choice} model not found! Please run main.py first.")
    st.stop()
else:
    with open(model_path, "rb") as f:
        model = pickle.load(f)

# -------------------------
# Single Customer Prediction
# -------------------------
st.subheader("🔹 Single Customer Prediction")

# Features except Occupation
input_features = [col for col in feature_names_model if not col.startswith("Occupation_")]

inputs = {}
cols = st.columns(3)
for i, feature in enumerate(input_features):
    with cols[i % 3]:
        val = st.text_input(feature, "0", key=f"feat_{i}")
        try:
            val = float(val)
        except:
            val = 0.0
        inputs[feature] = val

# Occupation selection
selected_occupation = st.selectbox("Occupation", [col.replace("Occupation_", "") for col in occupation_columns])

if st.button("Predict Credit Score"):
    single_df = pd.DataFrame([inputs])
    # Add one-hot occupation columns
    for occ_col in occupation_columns:
        single_df[occ_col] = 1 if occ_col == f"Occupation_{selected_occupation}" else 0
    # Ensure columns match training order
    single_df = single_df[feature_names_model]
    # Scale
    single_scaled = scaler.transform(single_df)
    # Predict
    pred = model.predict(single_scaled)[0]
    st.success(f" Predicted Credit Score: **{pred}**")

# -------------------------
# Batch Prediction via CSV
# -------------------------
st.subheader("📂 Batch Prediction via CSV")
uploaded_file = st.file_uploader("Upload CSV with features", type=["csv"])

if uploaded_file is not None:
    user_df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Preview")
    st.dataframe(user_df.head())

    # Add missing one-hot columns for Occupation
    for occ_col in occupation_columns:
        if occ_col not in user_df.columns:
            user_df[occ_col] = 0

    # Reorder columns to match training
    user_df = user_df[feature_names_model]

    # Scale
    scaled = scaler.transform(user_df)
    preds = model.predict(scaled)

    user_df["Credit_Score_Prediction"] = preds
    st.write("### Predictions")
    st.dataframe(user_df.head())

    csv = user_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Predictions", data=csv, file_name="predictions.csv", mime="text/csv")

# -------------------------
# Footer
# -------------------------
st.markdown("---", unsafe_allow_html=True)
st.markdown(
    """
    <div style="text-align: center;">
        <p><strong>Created by: Samruddhi R. Panhalkar</strong></p>
        <p>📧 samruddhipanhalkar156@gmail.com </p>
    </div>
    """,
    unsafe_allow_html=True,
)

