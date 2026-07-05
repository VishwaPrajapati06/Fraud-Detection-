# 💳 Credit Card Fraud Detection

A machine learning project that detects fraudulent credit card transactions, trained on the highly imbalanced ULB Credit Card Fraud dataset and deployed as an interactive Streamlit web app.

---

## 🚀 Features

- Upload a CSV of transactions and get instant fraud predictions
- Model trained specifically to handle severe class imbalance (fraud is ~0.17% of all transactions)
- Compares three modeling approaches (Logistic Regression, XGBoost, XGBoost + SMOTE) and selects the best based on precision, recall, and PR-AUC — not accuracy
- Sample CSV provided so users can try the app without their own data
- Displays fraud probability per transaction, summary metrics, and lets users download results

---

## 📊 Dataset

- **Source:** [ULB Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (via Kaggle/`kagglehub`)
- **Size:** 284,807 transactions, 492 labeled as fraud (~0.17%)
- **Features:** `V1`–`V28` are anonymized, PCA-transformed features (their real-world meaning is not disclosed to protect sensitive financial data), plus `Time` and `Amount`
- **Target:** `Class` (0 = Normal, 1 = Fraud)

### Data Cleaning
- Checked for missing values (none found)
- Removed 1,081 duplicate rows (284,807 → 283,726 rows)
- Stratified 80/20 train-test split to preserve the original fraud ratio in both sets
- Scaled `Time` and `Amount` using `StandardScaler` (the `V1`–`V28` features are already PCA-scaled)

---

## 🧠 Model Building & Evaluation

Three approaches were trained and compared to handle the class imbalance:

| Model | Precision (Fraud) | Recall (Fraud) | F1-score (Fraud) | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Logistic Regression (class-weighted) | 0.06 | 0.87 | 0.11 | 0.966 | 0.705 |
| XGBoost (`scale_pos_weight`) | 0.96 | 0.77 | 0.85 | 0.979 | 0.823 |
| XGBoost + SMOTE | 0.73 | 0.80 | 0.76 | 0.966 | 0.814 |

Because the dataset is so imbalanced, **accuracy is not a meaningful metric** — all three models exceed 98% accuracy simply by predicting "Normal" most of the time. Models were instead evaluated using confusion matrices, precision, recall, F1-score, ROC-AUC, and PR-AUC (Precision-Recall AUC), which is more informative than ROC-AUC for rare-event detection.

**Final model selected: XGBoost with `scale_pos_weight`** — it achieved the best balance of precision, F1-score, and PR-AUC while maintaining strong recall, making it the most reliable choice for production use.

---

## 📂 Project Structure

- `notebook.ipynb` → Full data exploration, preprocessing, model training, and evaluation
- `app.py` → Streamlit web app for uploading transactions and getting predictions
- `creditcard.csv` → Dataset used for training
- `fraud_detection_model.pkl` → Saved, trained XGBoost model
- `scaler.pkl` → Saved `StandardScaler` fitted on `Time` and `Amount`
- `feature_columns.pkl` → Saved column order expected by the model
- `feature_defaults.pkl` → Median feature values (used for generating sample/test inputs)
- `selected_features.pkl` → Top features most correlated with fraud
- `sample_csv/` → Sample transaction CSVs for trying out the app
- `requirements.txt` → Project dependencies

---

## 🛠️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/VishwaPrajapati06/Fraud-Detection-.git
cd Fraud-Detection-
pip install -r requirements.txt
```

Run the web app locally:

```bash
streamlit run app.py
```

---

## ▶️ Usage

1. Launch the app and upload a CSV file containing transaction data (must include the required feature columns).
2. The app scales `Time` and `Amount`, feeds the data to the trained XGBoost model, and flags each transaction as **Fraud** or **Normal**.
3. View summary metrics (total transactions, flagged frauds, fraud rate) and a results table with fraud probabilities.
4. Download the results as a CSV.
5. No file handy? Use the sample CSV provided in the app to try it out immediately.

---

## ⚙️ Technologies Used

- Python 3.x
- Pandas, NumPy — data handling
- Matplotlib, Seaborn — data visualization (EDA, confusion matrices, ROC/PR curves)
- Scikit-learn — Logistic Regression, StandardScaler, train-test split, evaluation metrics
- XGBoost — final production model
- imbalanced-learn (SMOTE) — used during model experimentation to compare oversampling vs. algorithm-level imbalance handling
- Streamlit — web app deployment
- Joblib — saving/loading the trained model and preprocessing artifacts

---

## 📌 Notes

- The `V1`–`V28` features are anonymized/PCA-transformed by the original dataset publishers; their real-world meaning is unknown, but the model learns which patterns in them correlate with fraud.
- The deployed app uses the plain XGBoost model (not the SMOTE version), based on its stronger precision, F1-score, and PR-AUC on the test set.
