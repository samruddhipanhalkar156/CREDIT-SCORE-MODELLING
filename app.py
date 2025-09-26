# app.py
import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

# -------------------------
# Paths
# -------------------------
MODELS_DIR = Path("Model")
RESULTS_DIR = Path("Results")

# Load metrics
metrics_path = RESULTS_DIR / "model_metrics.xlsx"
metrics_df = pd.read_excel(metrics_path) if metrics_path.exists() else pd.DataFrame()

# Load manifest
manifest_path = RESULTS_DIR / "manifest.pkl"
if not manifest_path.exists():
    st.error("❌ manifest.pkl not found! Please run main.py first.")
    st.stop()
else:
    with open(manifest_path, "rb") as f:
        manifest = pickle.load(f)

# Load preprocessor & scaler
with open(manifest["preprocessor"], "rb") as f:
    preprocessor = pickle.load(f)

with open(manifest["scaler"], "rb") as f:
    scaler = pickle.load(f)

label_classes = manifest["label_classes"]
feature_names = manifest["feature_names"]

MODEL_PATHS = {
    "Logistic Regression": manifest["lr_model"],
    "Decision Tree": manifest["dt_model"],
    "Random Forest": manifest["rf_model"],
}

# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="Credit Score Prediction", layout="wide")

# Title
col1, col2 = st.columns([6, 1])
with col1:
    st.title("📊 Credit Score Prediction System")

# -------------------------
# Model selection
# -------------------------
model_choice = st.selectbox("Choose a trained model:", list(MODEL_PATHS.keys()))

# Show metrics
if not metrics_df.empty:
    st.write("### Model Metrics")
    st.dataframe(metrics_df[metrics_df["Model"] == model_choice.replace(" ", "")])

# Load model
with open(MODEL_PATHS[model_choice], "rb") as f:
    model = pickle.load(f)

# -------------------------
# Single Input Prediction
# -------------------------
st.subheader("Single Input Prediction")

raw_columns = preprocessor.feature_names_in_  # original dataset columns
inputs = {}
cols = st.columns(4)

for i, col in enumerate(raw_columns):
    with cols[i % 4]:
        if pd.api.types.is_numeric_dtype(pd.Series(dtype="float64")):  # default numeric
            default_val = "0"
        else:
            default_val = ""

        if col.lower() in ["occupation", "credit_mix", "payment_behaviour"]:
            val = st.text_input(col, default_val, key=f"feat_{i}")
        else:
            val = st.text_input(col, default_val, key=f"feat_{i}")
            try:
                val = float(val)
            except:
                val = 0.0
        inputs[col] = val

if st.button("Predict Single Input"):
    single_df = pd.DataFrame([inputs])
    try:
        transformed = preprocessor.transform(single_df)
        scaled = scaler.transform(transformed)
        pred = model.predict(scaled)[0]
        st.success(f"Predicted Credit Score: **{label_classes[pred]}**")
    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")

# -------------------------
# Batch Prediction
# -------------------------
st.subheader("Batch Prediction via CSV")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    user_df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Preview")
    st.dataframe(user_df.head())

    # Validate columns
    missing_cols = set(preprocessor.feature_names_in_) - set(user_df.columns)
    if missing_cols:
        st.error(f"⚠️ Missing required columns: {missing_cols}")
    else:
        try:
            transformed = preprocessor.transform(user_df)
            scaled = scaler.transform(transformed)
            preds = model.predict(scaled)
            user_df["Predicted_Credit_Score"] = [label_classes[p] for p in preds]

            st.write("### Predictions")
            st.dataframe(user_df.head())

            csv = user_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Predictions",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"⚠️ Error during batch prediction: {e}")

# -------------------------
# Footer
# -------------------------
st.markdown("---", unsafe_allow_html=True)
st.markdown(
    """
    <div style="text-align: center;">
        <p><strong>Created by: Samruddhi R. Panhalkar</strong></p>
        <p><strong>Roll No: USN- 2MM22RI014</strong></p>
        <p>📧 samruddhipanhalkar156@gmail.com </p>
        <p>🏫 Maratha Mandal Engineering College</p>
        <p>
            <a href="https://instagram.com/yourusername" target="_blank">
                <img src="https://cdn-icons-png.flaticon.com/512/2111/2111463.png" width="30">
            </a>
            <a href="https://facebook.com/yourusername" target="_blank">
                <img src="https://cdn-icons-png.flaticon.com/512/733/733547.png" width="30">
            </a>
            <a href="https://twitter.com/yourusername" target="_blank">
                <img src="https://cdn-icons-png.flaticon.com/512/733/733579.png" width="30">
            </a>
            <a href="https://www.linkedin.com/in/samruddhi-panhalkar" target="_blank">
                <img src="https://cdn-icons-png.flaticon.com/512/3536/3536505.png" width="30">
            </a>
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

