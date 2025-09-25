# main.py
"""
End-to-end pipeline for Credit Score Modeling
"""

import os
import io
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
import matplotlib.pyplot as plt
import joblib
from datetime import datetime as dt

print("Starting Credit Score Modeling Pipeline...")
print(dt.now())

# -------------------------
# Paths
# -------------------------
ROOT = Path.cwd()
MODELS_DIR = ROOT / 'Models'
RESULTS_DIR = ROOT / 'Results'
RAW_CLEANED_PATH = ROOT / 'Raw Data' / 'Cleaned_Data' / 'Cleaned_data2025-09-18_15_06.csv'

for p in [MODELS_DIR, RESULTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

(MODELS_DIR / 'LR').mkdir(exist_ok=True)
(MODELS_DIR / 'DT').mkdir(exist_ok=True)
(MODELS_DIR / 'RF').mkdir(exist_ok=True)

# -------------------------
# Sample fallback data
# -------------------------
SAMPLE_CSV = '''Age\tAnnual_Income\tMonthly_Inhand_Salary\tNum_Bank_Accounts\tNum_Credit_Card\tInterest_Rate\tNum_of_Loan\tDelay_from_due_date\tNum_of_Delayed_Payment\tChanged_Credit_Limit\tNum_Credit_Inquiries\tCredit_Mix\tOutstanding_Debt\tCredit_Utilization_Ratio\tCredit_History_Age\tPayment_of_Min_Amount\tTotal_EMI_per_month\tAmount_invested_monthly\tPayment_Behaviour\tMonthly_Balance\tCredit_Score\tOccupation
23\t19114.12\t1824.843333\t3\t4\t3\t4\t3\t7\t11.27\t4\t2.8\t809.98\t26.82261962\t265\t0\t49.57494921\t80.41529544\t4\t312.4940887\tGood\tScientist
23\t19114.12\t4267.133667\t3\t4\t3\t4\t0\t6.4\t11.27\t4\t3\t809.98\t31.94496006\t247.6\t0\t49.57494921\t118.2802216\t3\t284.6291625\tGood\tScientist
'''

# -------------------------
# Load data
# -------------------------
if RAW_CLEANED_PATH.exists():
    print(f"Loading data from: {RAW_CLEANED_PATH}")
    df = pd.read_csv(RAW_CLEANED_PATH)
else:
    print("Cleaned_data.csv not found. Using embedded sample data.")
    df = pd.read_csv(io.StringIO(SAMPLE_CSV), sep='\t')

print(f"Data shape: {df.shape}")

# -------------------------
# Preprocessing
# -------------------------
TARGET = 'Credit_Score'
if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' not found")

X = df.drop(columns=[TARGET])
y = df[TARGET].astype(str)

# Encode target
le = LabelEncoder()
y_enc = le.fit_transform(y)
label_classes = list(le.classes_)
print("Label classes:", label_classes)

cat_cols = ['Occupation']
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

X_pre_df = pd.DataFrame(X_pre, columns=feature_names)

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_pre_df)
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)

# Balance data to 5000 per class (for LR & DT)
df_balanced = (
    df.groupby(TARGET, group_keys=False)
      .apply(lambda x: x.sample(n=min(5000, len(x)), random_state=42))
      .reset_index(drop=True)
)

X_bal = df_balanced.drop(columns=[TARGET])
y_bal = df_balanced[TARGET].astype(str)
y_bal_enc = le.transform(y_bal)

X_bal_pre = preprocessor.transform(X_bal)
X_bal_pre_df = pd.DataFrame(X_bal_pre, columns=feature_names)
X_bal_scaled = scaler.transform(X_bal_pre_df)

# Save scaled data
scaled_pickle_path = RESULTS_DIR / 'scaled_data.pkl'
with open(scaled_pickle_path, 'wb') as f:
    pickle.dump({
        'X_scaled': X_bal_scaled,
        'y': y_bal_enc,
        'feature_names': feature_names,
        'label_classes': label_classes
    }, f)

joblib.dump(preprocessor, MODELS_DIR / 'preprocessor.pkl')
joblib.dump(scaler, MODELS_DIR / 'scaler.pkl')

# -------------------------
# Train/Test split (for LR & DT)
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_bal_scaled, y_bal_enc, test_size=0.2, random_state=42, stratify=y_bal_enc
)

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
# Logistic Regression
# -------------------------
cv_lr = GridSearchCV(
    LogisticRegression(max_iter=1000, multi_class='multinomial'),
    {'C': [0.01, 0.1, 1, 10]}, cv=3, scoring='accuracy'
)
cv_lr.fit(X_train, y_train)
best_lr = cv_lr.best_estimator_
joblib.dump(best_lr, MODELS_DIR / 'LR' / 'lr_model.pkl')
evaluate_and_record('LogisticRegression', best_lr, X_train, y_train, X_test, y_test)

# -------------------------
# Decision Tree
# -------------------------
cv_dt = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    {'max_depth': [None, 3, 5, 7]}, cv=3, scoring='accuracy'
)
cv_dt.fit(X_train, y_train)
best_dt = cv_dt.best_estimator_
joblib.dump(best_dt, MODELS_DIR / 'DT' / 'dt_model.pkl')

fig = plt.figure(figsize=(12,8))
plot_tree(best_dt, filled=True, feature_names=feature_names, class_names=label_classes)
plt.savefig(MODELS_DIR / 'DT' / 'decision_tree_plot.png')
plt.close(fig)

evaluate_and_record('DecisionTree', best_dt, X_train, y_train, X_test, y_test)

# -------------------------
# Random Forest (use 3000 per class)
# -------------------------
df_rf = (
    df.groupby(TARGET, group_keys=False)
      .apply(lambda x: x.sample(n=min(3000, len(x)), random_state=42))
      .reset_index(drop=True)
)

X_rf = df_rf.drop(columns=[TARGET])
y_rf = df_rf[TARGET].astype(str)
y_rf_enc = le.transform(y_rf)

X_rf_pre = preprocessor.transform(X_rf)
X_rf_pre_df = pd.DataFrame(X_rf_pre, columns=feature_names)
X_rf_scaled = scaler.transform(X_rf_pre_df)

X_rf_train, X_rf_test, y_rf_train, y_rf_test = train_test_split(
    X_rf_scaled, y_rf_enc, test_size=0.2, random_state=42, stratify=y_rf_enc
)

cv_rf = GridSearchCV(
    RandomForestClassifier(random_state=42),
    {'n_estimators': [50, 100]}, cv=3, scoring='accuracy'
)
cv_rf.fit(X_rf_train, y_rf_train)
best_rf = cv_rf.best_estimator_
joblib.dump(best_rf, MODELS_DIR / 'RF' / 'rf_model.pkl')

evaluate_and_record('RandomForest', best_rf, X_rf_train, y_rf_train, X_rf_test, y_rf_test)

# -------------------------
# Save metrics
# -------------------------
metrics_df = pd.DataFrame(results_rows)
metrics_excel_path = RESULTS_DIR / 'model_metrics.xlsx'
metrics_df.to_excel(metrics_excel_path, index=False)

# Save manifest
manifest = {
    'scaled_data': str(scaled_pickle_path),
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

print("✅ Done. Models & results saved.")
