# 📊 Credit Score Prediction System  

## 🔹 Project Overview  
This project is a *machine learning-based credit score prediction system* that predicts whether a person’s credit score is *Good, Standard, or Poor* based on financial and demographic data.  

The workflow includes:  
1. *Data Cleaning & Preprocessing*  
2. *Exploratory Data Analysis (EDA)*  
3. *Model Building & Hyperparameter Tuning* (Logistic Regression, Decision Tree, Random Forest)  
4. *Model Evaluation & Comparison*  
5. *Interactive Web App using Streamlit*  

The app allows users to:  
✅ Select a trained model and view its performance metrics  
✅ Enter values manually for prediction  
✅ Upload a CSV file and download predictions  
✅ View results interactively in a dashboard  

---

## 🔹 Folder Structure  


'''

Credit Score Modeling
│
├── Models
│   ├── DT
│   ├── RF
│   ├── LR
│   └── knn\_imputer.pkl
│
├── Raw Data
│   ├── Customer Credit Scoring\_Train.csv
│   └── Cleaned Data
│       └── Cleaned\_data.csv
│
├── Results
│   ├── model\_metrics.xlsx
│   ├── manifest.pkl
│   └── EDA Results
│       └── Reports
│           ├── unique\_values\_info.csv
│           ├── missing\_data\_info.csv
│           ├── Ydata\_profile\_report.html
│
├── main.py        ← Script for training models
├── app.py         ← Streamlit web app
├── requirements.txt
└── README.md

'''


---

## 🔹 Data Cleaning & Preprocessing  
- Removed unnecessary columns: ID, Customer_ID, Month, Name, SSN, Type_of_Loan.  
- Converted categorical/object fields to numeric:  
  - *Age, Income, Loans, Debt, etc.* cleaned from “_” and cast to numeric.  
  - Credit_History_Age converted into *months*.  
  - Payment_of_Min_Amount → 1/0.  
  - Credit_Mix → mapped to {Good=3, Standard=2, Bad=1}.  
  - Payment_Behaviour mapped to numeric values.  
- Handled missing values using *KNNImputer*.  
- Outlier treatment via *clipping at quantiles (1%–99%)*.  
- Feature scaling using *StandardScaler*.  

---

## 🔹 Models Implemented  
1. *Logistic Regression*  
   - Hyperparameter tuning using GridSearchCV  
2. *Decision Tree*  
   - Max depth & splitting criteria tuned  
   - Decision Tree plot saved  
3. *Random Forest*  
   - Tuned with number of trees & depth  

All models were saved using *Joblib/Pickle* for reuse.  

---

## 🔹 Evaluation Metrics  
For each model, we calculated on *train and test data*:  
- Accuracy  
- Precision  
- Recall  
- F1-score  
- Log Loss  

📊 Results stored in: Results/model_metrics.xlsx.  

---

## 🔹 Streamlit Web App (app.py)  
Features:  
- Model selection dropdown (LR, DT, RF).  
- Display model metrics from Excel.  
- Single input prediction via manual entry.  
- Batch prediction via CSV upload → Download predictions.  
- Footer with *author info + social links*.  
- Logo display at the top right corner.  

---

## 🔹 How to Run  

### 1️⃣ Install Dependencies  
bash
cd "Credit Score Modelling"
pip install -r requirements.txt
`

### 2️⃣ Run Training Script (optional, only if retraining needed)

bash
python main.py


### 3️⃣ Run Streamlit App

bash
streamlit run app.py


Then open the link in your browser → [http://localhost:8501](http://localhost:8501)

---

## 🔹 Requirements

See requirements.txt for full list.
Key packages:

* pandas, numpy
* scikit-learn
* matplotlib, seaborn
* joblib, pickle
* streamlit
* openpyxl

---

## 🔹 Future Enhancements

* Deploy app online via *Streamlit Cloud / Heroku*.
* Add more advanced models (XGBoost, LightGBM).
* Feature importance visualization.
* Secure API endpoints for mobile integration.

---

## 🔹 Author

👩‍💻 *Sunil Pasupula*
📧 Email: [abcdefgh@gmail.com](mailto:abcdefgh@gmail.com)
📱 Phone: +91-1234567890
🏫 Designation: Junior Data Scientst

🌐 Social Links:
🔗 [Instagram](https://instagram.com/yourusername) | [Facebook](https://facebook.com/yourusername) | [Twitter](https://twitter.com/yourusername) | [LinkedIn](https://www.linkedin.com/in/samruddhi-panhalkar)
