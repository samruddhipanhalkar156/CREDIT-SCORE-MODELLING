# main.py
"""
Simplified Credit Score Modeling Pipeline
"""

import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
import matplotlib.pyplot as plt
import joblib

# -------------------------
# Paths
# -------------------------
ROOT = Path.cwd()
MODELS_DIR = ROOT / 'Models'
RESULTS_DIR = ROOT / 'Results'
RAW_CLEANED_PATH = ROOT / 'Raw Data' / 'Cleaned_Data.csv'

for p in [MODELS_DIR, RESULTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

(MODELS_DIR / 'LR').mkdir(exist_ok=True)
(MODELS_DIR / 'DT').mkdir(exist_ok=True)
(MODELS_DIR / 'RF').mkdir(exist_ok=True)

# -------------------------
# Load data
# -------------------------
df = pd.read_csv(RAW_CLEANED_PATH)
print(f"Data shape before sampling: {df.shape}")

TARGET = 'Credit_Score'
if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' not found")

# -------------------------
# Sampling
# -------------------------
# 5000 samples per class for Logistic Regression & Decision Tree
df_balanced = df.groupby(TARGET, group_keys=False).apply(lambda x: x.sample(5000, random_state=42))
print(f"Balanced data shape (5000/class): {df_balanced.shape}")

# 3500 samples per class for Random Forest
df_rf = df.groupby(TARGET, group_keys=False).apply(lambda x: x.sample(3500, random_state=42))
print(f"RF data shape (3500/class): {df_rf.shape}")

# -------------------------
# Preprocessing
# -------------------------
def preprocess(data):
    X = data.drop(columns=[TARGET])
    y = data[TARGET].astype(str)

    # Encode target
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    label_classes = list(le.classes_)

    # Column split
    cat_cols = [c for c in X.columns if X[c].dtype == 'object']
    num_cols = [c for c in X.columns if c not in cat_cols]

    # OneHotEncode + passthrough numerics
    one_hot = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    preprocessor = ColumnTransformer([
        ('ohe', one_hot, cat_cols),
        ('passthrough', 'passthrough', num_cols)
    ])

    X_pre = preprocessor.fit_transform(X)
    ohe_cols = list(preprocessor.named_transformers_['ohe'].get_feature_names_out(cat_cols))
    feature_names = ohe_cols + num_cols

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_pre)

    return X_scaled, y_enc, feature_names, label_classes, preprocessor, scaler

# -------------------------
# Train/Test split
# -------------------------
X_bal, y_bal, feature_names, label_classes, preproc, scaler = preprocess(df_balanced)
X_train, X_test, y_train, y_test = train_test_split(
    X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal
)

X_rf, y_rf, _, _, preproc_rf, scaler_rf = preprocess(df_rf)
X_rf_train, X_rf_test, y_rf_train, y_rf_test = train_test_split(
    X_rf, y_rf, test_size=0.2, random_state=42, stratify=y_rf
)

# Save preprocessor & scaler
with open(MODELS_DIR / 'preprocessor.pkl', "wb") as f:
    pickle.dump(preproc, f)

with open(MODELS_DIR / 'scaler.pkl', "wb") as f:
    pickle.dump(scaler, f)

# -------------------------
# Utility
# -------------------------
results_rows = []

def evaluate_and_record(name, model, Xtr, ytr, Xte, yte):
    ytr_pred = model.predict(Xtr)
    yte_pred = model.predict(Xte)

    results_rows.append({
        'Model': name, 'Dataset': 'Train',
        'Accuracy': accuracy_score(ytr, ytr_pred),
        'Precision_macro': precision_score(ytr, ytr_pred, average='macro'),
        'Recall_macro': recall_score(ytr, ytr_pred, average='macro'),
        'F1_macro': f1_score(ytr, ytr_pred, average='macro'),
        'LogLoss': log_loss(ytr, model.predict_proba(Xtr), labels=np.arange(len(label_classes)))
    })

    results_rows.append({
        'Model': name, 'Dataset': 'Test',
        'Accuracy': accuracy_score(yte, yte_pred),
        'Precision_macro': precision_score(yte, yte_pred, average='macro'),
        'Recall_macro': recall_score(yte, yte_pred, average='macro'),
        'F1_macro': f1_score(yte, yte_pred, average='macro'),
        'LogLoss': log_loss(yte, model.predict_proba(Xte), labels=np.arange(len(label_classes)))
    })

# -------------------------
# Models
# -------------------------
# Logistic Regression
cv_lr = GridSearchCV(
    LogisticRegression(max_iter=1000, multi_class='multinomial'),
    {'C': [0.01, 0.1, 1, 10]}, cv=5, scoring='accuracy'
)
cv_lr.fit(X_train, y_train)
best_lr = cv_lr.best_estimator_
# Save models with pickle
with open(MODELS_DIR / 'LR' / 'lr_model.pkl', "wb") as f:
    pickle.dump(best_lr, f)

evaluate_and_record('LogisticRegression', best_lr, X_train, y_train, X_test, y_test)

# Decision Tree
cv_dt = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    {
            "max_depth": [None, 3, 5, 7, 10],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
}, cv=3, scoring='accuracy'
)
cv_dt.fit(X_train, y_train)
best_dt = cv_dt.best_estimator_



with open(MODELS_DIR / 'DT' / 'dt_model.pkl', "wb") as f:
    pickle.dump(best_dt, f)

fig = plt.figure(figsize=(12,8))
plot_tree(best_dt, filled=True, feature_names=feature_names, class_names=label_classes)
plt.savefig(MODELS_DIR / 'DT' / 'decision_tree_plot.png')
plt.close(fig)

evaluate_and_record('DecisionTree', best_dt, X_train, y_train, X_test, y_test)

# Random Forest (3500 samples per class)
cv_rf = GridSearchCV(
    RandomForestClassifier(random_state=42),
    {"n_estimators": [50,100],
            "max_depth": [None, 5],
            "min_samples_split": [2],
            "min_samples_leaf": [3,5],
            "max_features": ["sqrt", "log2"]}, cv=3, scoring='accuracy'
)
cv_rf.fit(X_rf_train, y_rf_train)
best_rf = cv_rf.best_estimator_



with open(MODELS_DIR / 'RF' / 'rf_model.pkl', "wb") as f:
    pickle.dump(best_rf, f)





evaluate_and_record('RandomForest', best_rf, X_rf_train, y_rf_train, X_rf_test, y_rf_test)

# -------------------------
# Save metrics
# -------------------------
metrics_df = pd.DataFrame(results_rows)
metrics_excel_path = RESULTS_DIR / 'model_metrics.xlsx'
metrics_df.to_excel(metrics_excel_path, index=False)

# Save manifest
manifest = {
    'preprocessor': str(MODELS_DIR / 'preprocessor.pkl'),
    'scaler': str(MODELS_DIR / 'scaler.pkl'),
    'lr_model': str(MODELS_DIR / 'LR' / 'lr_model.pkl'),
    'dt_model': str(MODELS_DIR / 'DT' / 'dt_model.pkl'),
    'dt_tree_plot': str(MODELS_DIR / 'DT' / 'decision_tree_plot.png'),
    'rf_model': str(MODELS_DIR / 'RF' / 'rf_model.pkl'),
    'metrics_excel': str(metrics_excel_path),
    'label_classes': label_classes,
    'feature_names': feature_names
}
with open(RESULTS_DIR / 'manifest.pkl', 'wb') as f:
    pickle.dump(manifest, f)

print(" Done. Models & results saved.")
