# main.py
"""
End-to-end pipeline for Credit Score Modeling
"""

import os
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
import matplotlib.pyplot as plt

print("Starting Credit Score Modeling Pipeline...")

# -------------------------
# Paths
# -------------------------
ROOT = Path.cwd()
MODELS_DIR = ROOT / "Models"
RESULTS_DIR = ROOT / "Results"
RAW_DATA_PATH = ROOT / "Raw Data" / "Cleaned_data.csv"

for p in [MODELS_DIR, RESULTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

(MODELS_DIR / "LR").mkdir(exist_ok=True)
(MODELS_DIR / "DT").mkdir(exist_ok=True)
(MODELS_DIR / "RF").mkdir(exist_ok=True)

# -------------------------
# Load data
# -------------------------
if RAW_DATA_PATH.exists():
    df = pd.read_csv(RAW_DATA_PATH)
else:
    raise FileNotFoundError(f"{RAW_DATA_PATH} not found!")

print(f"Original data shape: {df.shape}")

# -------------------------
# Target & features
# -------------------------
TARGET = "Credit_Score"
X = df.drop(columns=[TARGET])
y = df[TARGET].astype(str)

# Encode target
le = LabelEncoder()
y_enc = le.fit_transform(y)
label_classes = list(le.classes_)
print("Label classes:", label_classes)

# Identify categorical and numeric columns
cat_cols = ["Occupation"]
num_cols = [c for c in X.columns if c not in cat_cols]

# -------------------------
# Undersample dataset for LR & DT
# -------------------------
min_class_size = df[TARGET].value_counts().min()
df_bal = df.groupby(TARGET, group_keys=False).apply(
    lambda x: x.sample(n=min_class_size, random_state=42)
).reset_index(drop=True)

X_bal = df_bal.drop(columns=[TARGET])
y_bal = df_bal[TARGET].astype(str)
y_bal_enc = le.transform(y_bal)

print(f"Balanced data shape for LR & DT: {X_bal.shape}")

# -------------------------
# Preprocessing
# -------------------------
one_hot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
preprocessor = ColumnTransformer([
    ("ohe", one_hot, cat_cols),
    ("passthrough", "passthrough", num_cols)
])

X_pre = preprocessor.fit_transform(X_bal)
ohe_cols = list(preprocessor.named_transformers_['ohe'].get_feature_names_out(cat_cols))
feature_names = ohe_cols + num_cols
X_pre_df = pd.DataFrame(X_pre, columns=feature_names)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_pre_df)
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)

# Save preprocessor and scaler
with open(MODELS_DIR / "preprocessor.pkl", "wb") as f:
    pickle.dump(preprocessor, f)
with open(MODELS_DIR / "scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Save scaled data
with open(RESULTS_DIR / "scaled_data.pkl", "wb") as f:
    pickle.dump({
        "X_scaled": X_scaled_df,
        "y": y_bal_enc,
        "feature_names": feature_names,
        "label_classes": label_classes
    }, f)

# -------------------------
# Train/Test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled_df, y_bal_enc, test_size=0.2, random_state=42, stratify=y_bal_enc
)

# -------------------------
# Evaluation utility
# -------------------------
results_rows = []

def evaluate_and_record(name, model, Xtr, ytr, Xte, yte):
    ytr_pred = model.predict(Xtr)
    yte_pred = model.predict(Xte)

    results_rows.append({
        "Model": name, "Dataset": "Train",
        "Accuracy": accuracy_score(ytr, ytr_pred),
        "Precision_macro": precision_score(ytr, ytr_pred, average="macro"),
        "Recall_macro": recall_score(ytr, ytr_pred, average="macro"),
        "F1_macro": f1_score(ytr, ytr_pred, average="macro"),
        "LogLoss": log_loss(ytr, model.predict_proba(Xtr), labels=np.arange(len(label_classes)))
    })

    results_rows.append({
        "Model": name, "Dataset": "Test",
        "Accuracy": accuracy_score(yte, yte_pred),
        "Precision_macro": precision_score(yte, yte_pred, average="macro"),
        "Recall_macro": recall_score(yte, yte_pred, average="macro"),
        "F1_macro": f1_score(yte, yte_pred, average="macro"),
        "LogLoss": log_loss(yte, model.predict_proba(Xte), labels=np.arange(len(label_classes)))
    })

# -------------------------
# Models and hyperparameters
# -------------------------
models = {
    "LogisticRegression": (
        LogisticRegression(max_iter=1000, multi_class="multinomial"),
        {"C": [0.01, 0.1, 1, 10]}
    ),
    "DecisionTree": (
        DecisionTreeClassifier(random_state=42),
        {"max_depth": [None, 3, 5, 7]}
    ),
    "RandomForest": (
        RandomForestClassifier(random_state=42),
        {"n_estimators": [50, 100], "max_depth": [None, 5, 10]}
    )
}

# -------------------------
# Train Logistic Regression
# -------------------------
cv_lr = GridSearchCV(models["LogisticRegression"][0], models["LogisticRegression"][1], cv=3, scoring="accuracy")
cv_lr.fit(X_train, y_train)
best_lr = cv_lr.best_estimator_
with open(MODELS_DIR / "LR" / "LogisticRegression.pkl", "wb") as f:
    pickle.dump(best_lr, f)
evaluate_and_record("LogisticRegression", best_lr, X_train, y_train, X_test, y_test)

# -------------------------
# Train Decision Tree
# -------------------------
cv_dt = GridSearchCV(models["DecisionTree"][0], models["DecisionTree"][1], cv=3, scoring="accuracy")
cv_dt.fit(X_train, y_train)
best_dt = cv_dt.best_estimator_
with open(MODELS_DIR / "DT" / "DecisionTree.pkl", "wb") as f:
    pickle.dump(best_dt, f)

# Save decision tree plot
fig = plt.figure(figsize=(12,8))
plot_tree(best_dt, filled=True, feature_names=feature_names, class_names=label_classes)
plt.savefig(MODELS_DIR / "DT" / "decision_tree_plot.png")
plt.close(fig)

evaluate_and_record("DecisionTree", best_dt, X_train, y_train, X_test, y_test)

# -------------------------
# Random Forest: 8K samples per class
# -------------------------
df_rf = df.groupby(TARGET, group_keys=False).apply(
    lambda x: x.sample(n=min(3000, len(x)), random_state=42)
).reset_index(drop=True)

X_rf = df_rf.drop(columns=[TARGET])
y_rf = df_rf[TARGET].astype(str)
y_rf_enc = le.transform(y_rf)

X_rf_pre = preprocessor.transform(X_rf)
X_rf_scaled = scaler.transform(X_rf_pre)

X_rf_train, X_rf_test, y_rf_train, y_rf_test = train_test_split(
    X_rf_scaled, y_rf_enc, test_size=0.2, random_state=42, stratify=y_rf_enc
)

cv_rf = GridSearchCV(models["RandomForest"][0], models["RandomForest"][1], cv=3, scoring="accuracy")
cv_rf.fit(X_rf_train, y_rf_train)
best_rf = cv_rf.best_estimator_
with open(MODELS_DIR / "RF" / "RandomForest.pkl", "wb") as f:
    pickle.dump(best_rf, f)

evaluate_and_record("RandomForest", best_rf, X_rf_train, y_rf_train, X_rf_test, y_rf_test)

# -------------------------
# Save metrics
# -------------------------
metrics_df = pd.DataFrame(results_rows)
metrics_excel_path = RESULTS_DIR / "model_metrics.xlsx"
metrics_df.to_excel(metrics_excel_path, index=False)

# -------------------------
# Save manifest
# -------------------------
manifest = {
    "scaled_data": str(RESULTS_DIR / "scaled_data.pkl"),
    "preprocessor": str(MODELS_DIR / "preprocessor.pkl"),
    "scaler": str(MODELS_DIR / "scaler.pkl"),
    "lr_model": str(MODELS_DIR / "LR" / "LogisticRegression.pkl"),
    "dt_model": str(MODELS_DIR / "DT" / "DecisionTree.pkl"),
    "dt_tree_plot": str(MODELS_DIR / "DT" / "decision_tree_plot.png"),
    "rf_model": str(MODELS_DIR / "RF" / "RandomForest.pkl"),
    "metrics_excel": str(metrics_excel_path),
    "label_classes": label_classes,
    "feature_names": feature_names
}

with open(RESULTS_DIR / "manifest.pkl", "wb") as f:
    pickle.dump(manifest, f)

print("✅ Done. Models & results saved.")
