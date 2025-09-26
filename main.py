import os
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.under_sampling import RandomUnderSampler

# =====================
# Folder Setup
# =====================
RAW_DIR = "Raw Data"
MODELS_DIR = "Models"
RESULTS_DIR = "Model Results"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# =====================
# Load Cleaned Dataset
# =====================
df = pd.read_csv(os.path.join(RAW_DIR, "Cleaned_data.csv"))

# =====================
# One-Hot Encoding for Occupation
# =====================
df = pd.get_dummies(df, columns=['Occupation'], drop_first=True)

# =====================
# Features and Target
# =====================
X = df.drop("Credit_Score", axis=1)
y = df["Credit_Score"]

# =====================
# Save feature names for app
# =====================
with open(os.path.join(MODELS_DIR, "feature_names.pkl"), "wb") as f:
    pickle.dump(X.columns.tolist(), f)

# =====================
# Undersampling for Logistic Regression & Decision Tree
# =====================
rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(X, y)

# =====================
# Train-Test Split
# =====================
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
)

# =====================
# Feature Scaling
# =====================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
with open(os.path.join(MODELS_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

# =====================
# Models and Hyperparameter Grids
# =====================
models_params = {
    "LogisticRegression": (
        LogisticRegression(max_iter=1000, multi_class="ovr"),
        {
            "C": [0.01, 0.1, 1, 10],
            "solver": ["newton-cg", "lbfgs", "saga"],
            "multi_class": ["ovr", "multinomial"]
        }
    ),
    "DecisionTree": (
        DecisionTreeClassifier(random_state=42),
        {
            "max_depth": [3, 5, 10, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        }
    )
}

# =====================
# Training Function with GridSearchCV
# =====================
def train_and_evaluate(models_params, X_train, X_test, y_train, y_test):
    results = []

    for name, (model, params) in models_params.items():
        print(f"Training {name} with GridSearchCV...")
        grid = GridSearchCV(
            model,
            param_grid=params,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring="accuracy",
            n_jobs=-1
        )
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_

        # Save model
        with open(os.path.join(MODELS_DIR, f"{name}.pkl"), "wb") as f:
            pickle.dump(best_model, f)

        # Predictions
        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)

        if hasattr(best_model, "predict_proba"):
            y_train_prob = best_model.predict_proba(X_train)
            y_test_prob = best_model.predict_proba(X_test)
            auc_train = roc_auc_score(y_train, y_train_prob, multi_class="ovr")
            auc_test = roc_auc_score(y_test, y_test_prob, multi_class="ovr")
        else:
            auc_train = np.nan
            auc_test = np.nan

        for split, y_true, y_pred, auc in [
            ("Train", y_train, y_train_pred, auc_train),
            ("Test", y_test, y_test_pred, auc_test)
        ]:
            results.append({
                "Model": name,
                "Split": split,
                "Accuracy": accuracy_score(y_true, y_pred),
                "Precision_weighted": precision_score(y_true, y_pred, average="weighted"),
                "Recall_weighted": recall_score(y_true, y_pred, average="weighted"),
                "F1_weighted": f1_score(y_true, y_pred, average="weighted"),
                "AUC_ROC": auc
            })

    return pd.DataFrame(results)

# =====================
# Train Logistic Regression & Decision Tree
# =====================
results_df = train_and_evaluate(models_params, X_train_scaled, X_test_scaled, y_train, y_test)

# =====================
# Random Forest with 3000 Samples per Class
# =====================
sampled_df = df.groupby("Credit_Score").apply(lambda x: x.sample(n=6000, replace=True, random_state=42)).reset_index(drop=True)
X_rf = sampled_df.drop("Credit_Score", axis=1)
y_rf = sampled_df["Credit_Score"]

X_rf_train, X_rf_test, y_rf_train, y_rf_test = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42, stratify=y_rf)
scaler_rf = StandardScaler()
X_rf_train_scaled = scaler_rf.fit_transform(X_rf_train)
X_rf_test_scaled = scaler_rf.transform(X_rf_test)

rf_model = RandomForestClassifier(random_state=42)
rf_params = {
    "n_estimators": [50,100],
    "max_depth": [5,7],
    "min_samples_split": [2, 3],
    "min_samples_leaf": [7,11],
    "max_features": ["sqrt", "log2"]
}

print("Training RandomForest with GridSearchCV...")
rf_grid = GridSearchCV(
    rf_model,
    param_grid=rf_params,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring="accuracy",
    n_jobs=-1
)
rf_grid.fit(X_rf_train_scaled, y_rf_train)
best_rf = rf_grid.best_estimator_

# Save Random Forest model and scaler
with open(os.path.join(MODELS_DIR, "RandomForest.pkl"), "wb") as f:
    pickle.dump(best_rf, f)
with open(os.path.join(MODELS_DIR, "scaler_rf.pkl"), "wb") as f:
    pickle.dump(scaler_rf, f)

# RF Predictions
y_rf_train_pred = best_rf.predict(X_rf_train_scaled)
y_rf_test_pred = best_rf.predict(X_rf_test_scaled)

y_rf_train_prob = best_rf.predict_proba(X_rf_train_scaled)
y_rf_test_prob = best_rf.predict_proba(X_rf_test_scaled)

auc_train_rf = roc_auc_score(y_rf_train, y_rf_train_prob, multi_class="ovr")
auc_test_rf = roc_auc_score(y_rf_test, y_rf_test_prob, multi_class="ovr")

# Append RF results
results_df = pd.concat([
    results_df,
    pd.DataFrame([
        {
            "Model": "RandomForest",
            "Split": "Train",
            "Accuracy": accuracy_score(y_rf_train, y_rf_train_pred),
            "Precision_weighted": precision_score(y_rf_train, y_rf_train_pred, average="weighted"),
            "Recall_weighted": recall_score(y_rf_train, y_rf_train_pred, average="weighted"),
            "F1_weighted": f1_score(y_rf_train, y_rf_train_pred, average="weighted"),
            "AUC_ROC": auc_train_rf
        },
        {
            "Model": "RandomForest",
            "Split": "Test",
            "Accuracy": accuracy_score(y_rf_test, y_rf_test_pred),
            "Precision_weighted": precision_score(y_rf_test, y_rf_test_pred, average="weighted"),
            "Recall_weighted": recall_score(y_rf_test, y_rf_test_pred, average="weighted"),
            "F1_weighted": f1_score(y_rf_test, y_rf_test_pred, average="weighted"),
            "AUC_ROC": auc_test_rf
        }
    ])
], ignore_index=True)

# =====================
# Save Results
# =====================
results_file = os.path.join(RESULTS_DIR, "model_results.xlsx")
results_df.to_excel(results_file, index=False)

print("✅ Pipeline Completed! All models, scalers, feature names, and results saved.")
