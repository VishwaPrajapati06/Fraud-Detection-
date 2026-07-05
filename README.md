# Fraud Detection

A machine learning web application that predicts whether a credit card transaction is fraudulent, based on the popular [Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud). The trained model is served through a simple web app for real-time predictions.

## 🔍 Overview

This project trains a classification model on anonymized credit card transaction data (features `V1`–`V28`, `Time`, and `Amount`) to flag potentially fraudulent transactions. The final model, along with its preprocessing artifacts, is bundled and served via `app.py`.

## 📂 Project Structure

```
Fraud-Detection/
├── app.py                     # Web app that serves predictions
├── notebook.ipynb             # EDA, feature engineering, and model training
├── fraud_detection_model.pkl  # Trained ML model
├── scaler.pkl                 # Fitted scaler for numeric features
├── feature_columns.pkl        # Ordered list of feature columns expected by the model
├── feature_defaults.pkl       # Default values for optional/missing features
├── selected_features.pkl      # Final feature subset used by the model
├── creditcard.csv             # Dataset (tracked via Git LFS)
├── requirements.txt           # Python dependencies
└── sample_csv/                # Sample input data for testing
```

> **Note:** `creditcard.csv` is a large file (~147 MB) and is tracked using [Git LFS](https://git-lfs.com). Make sure Git LFS is installed before cloning.

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git lfs install
   git clone https://github.com/VishwaPrajapati06/Fraud-Detection-.git
   cd Fraud-Detection-
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

Run the app locally:

```bash
python app.py
```

Then open your browser to the local address shown in the terminal (e.g. `http://127.0.0.1:5000`) and submit transaction details to get a fraud prediction.

## 🧠 Model Details

- **Dataset:** European cardholders' transactions dataset (features anonymized via PCA, plus `Time` and `Amount`)
- **Preprocessing:** Feature scaling via `scaler.pkl`, with a fixed feature set defined in `selected_features.pkl` and `feature_columns.pkl`
- **Artifacts:** All preprocessing objects and the trained model are serialized with `pickle` for consistent inference at prediction time
- **Training:** See `notebook.ipynb` for exploratory data analysis, feature selection, model comparison, and evaluation metrics

## 📊 Dataset

The dataset used (`creditcard.csv`) contains transactions made by European cardholders in September 2013. It is highly imbalanced — fraudulent transactions account for a small fraction of all transactions.

## 🛠️ Tech Stack

- Python
- scikit-learn
- Pandas / NumPy
- Streamlit (web app)
- Jupyter Notebook (model development)

## 📌 Future Improvements

- Add model explainability (e.g. SHAP values)
- Experiment with ensemble/boosting models for improved recall on fraud cases


## 🙋 Author

**Vishwa Prajapati**
GitHub: [@VishwaPrajapati06](https://github.com/VishwaPrajapati06)
