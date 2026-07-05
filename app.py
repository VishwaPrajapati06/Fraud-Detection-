import streamlit as st
import pandas as pd
import joblib

# ── Page setup ──────────────────────────────────────────────
st.set_page_config(page_title="Credit Card Fraud Detector", page_icon="💳", layout="wide")

st.title("💳 Credit Card Fraud Detection")
st.markdown(
    """
    Upload a CSV of transactions and this app will flag which ones are likely **fraudulent**,
    using a trained XGBoost model.

    **Note on the data:** Features `V1`–`V28` are anonymized (PCA-transformed) versions of the
    original transaction details, released this way to protect sensitive financial information.
    We don't know their real-world meaning, but the model has learned which patterns in them
    are linked to fraud.
    """
)

# ── Load model and scaler (cached so it only loads once) ───
@st.cache_resource
def load_artifacts():
    model = joblib.load("fraud_detection_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, scaler, feature_columns

model, scaler, feature_columns = load_artifacts()

# ── CSV upload ───────────────────────────────────────────────
st.subheader("1. Upload your transaction CSV")
st.caption("The CSV must contain these columns: " + ", ".join(feature_columns))

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

# ── Sample data option ───────────────────────────────────────
with st.expander("Don't have a file? Download a sample CSV to try"):
    st.write("Use this sample (a few real transactions from the test set) to try the app.")
    try:
        with open("sample_transactions.csv", "rb") as f:
            st.download_button(
                "Download sample_transactions.csv",
                f,
                file_name="sample_transactions.csv",
                mime="text/csv",
            )
    except FileNotFoundError:
        st.info("Sample file not found in this deployment.")

# ── Run prediction ───────────────────────────────────────────
if uploaded_file is not None:
    try:
        input_df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read the CSV file: {e}")
        st.stop()

    # Drop Class column if present (in case user uploads labeled data)
    true_labels = None
    if "Class" in input_df.columns:
        true_labels = input_df["Class"]
        input_df = input_df.drop(columns=["Class"])

    # Check required columns
    missing_cols = [c for c in feature_columns if c not in input_df.columns]
    if missing_cols:
        st.error(f"Your CSV is missing required columns: {missing_cols}")
        st.stop()

    # Reorder columns to match training order
    input_df = input_df[feature_columns].copy()

    # Scale Time and Amount using the saved scaler
    input_df[["Time", "Amount"]] = scaler.transform(input_df[["Time", "Amount"]])

    # Predict
    predictions = model.predict(input_df)
    probabilities = model.predict_proba(input_df)[:, 1]

    # Build results table (show original, unscaled values for readability)
    display_df = input_df.copy()
    display_df["Fraud Probability"] = probabilities.round(4)
    display_df["Prediction"] = ["Fraud" if p == 1 else "Normal" for p in predictions]
    if true_labels is not None:
        display_df["Actual Label"] = ["Fraud" if v == 1 else "Normal" for v in true_labels]

    st.subheader("2. Results")
    n_fraud = int((predictions == 1).sum())
    n_total = len(predictions)
    col1, col2, col3 = st.columns(3)
    col1.metric("Total transactions", n_total)
    col2.metric("Flagged as fraud", n_fraud)
    col3.metric("Fraud rate", f"{n_fraud / n_total * 100:.2f}%")

    st.dataframe(
        display_df.style.apply(
            lambda row: ["background-color: #ffe6e6" if row["Prediction"] == "Fraud" else "" for _ in row],
            axis=1,
        ),
        use_container_width=True,
    )

    # Download results
    csv_out = display_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download results as CSV", csv_out, file_name="fraud_predictions.csv", mime="text/csv")

else:
    st.info("Upload a CSV file above to get started.")

st.markdown("---")
st.caption(
    "Model: XGBoost trained on the ULB Credit Card Fraud dataset (highly imbalanced, ~0.17% fraud). "
    "Evaluated using precision, recall, and PR-AUC due to class imbalance."
)