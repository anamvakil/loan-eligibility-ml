"""
app.py
------
Streamlit app for the Loan Eligibility Prediction Model.
Tabs: Overview | Train & Evaluate | Predict
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Make src importable when run from project root
sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import load_data, get_target_distribution
from src.preprocessor import (
    impute_missing_values,
    encode_features,
    split_features_target,
    scale_features,
)
from src.model import (
    train_logistic_regression,
    train_decision_tree,
    train_random_forest,
    evaluate_model,
    cross_validate_model,
    get_feature_importances,
)
from src.utils import setup_logging, plot_confusion_matrix, plot_cv_scores, plot_feature_importances

setup_logging()
logger = logging.getLogger(__name__)

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Eligibility Predictor",
    page_icon=None,
    layout="wide",
)

st.title("Loan Eligibility Prediction")
st.caption("CST2216 — Modularizing and Deploying ML Code | Algonquin College")

DATA_PATH = os.path.join("data", "credit.csv")

# ── Tabs ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Data Overview", "Train & Evaluate", "Predict"])


# ─────────────────────────────────────────────────────────────────────────
# TAB 1 — Data Overview
# ─────────────────────────────────────────────────────────────────────────
with tab1:
    st.header("Dataset Overview")

    uploaded = st.file_uploader(
        "Upload credit.csv (or use the default in /data/credit.csv)",
        type="csv",
        key="upload_tab1",
    )

    @st.cache_data
    def load_and_preview(path_or_bytes):
        if isinstance(path_or_bytes, str):
            return pd.read_csv(path_or_bytes)
        return pd.read_csv(path_or_bytes)

    try:
        source = uploaded if uploaded else DATA_PATH
        raw_df = load_and_preview(source)

        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", raw_df.shape[0])
        col2.metric("Columns", raw_df.shape[1])
        col3.metric("Missing Values", int(raw_df.isnull().sum().sum()))

        st.subheader("Sample Records")
        st.dataframe(raw_df.head(10), use_container_width=True)

        st.subheader("Target Variable Distribution")
        dist = raw_df["Loan_Approved"].value_counts()
        fig, ax = plt.subplots(figsize=(4, 3))
        dist.plot.bar(ax=ax, color=["#e74c3c", "#2ecc71"], edgecolor="white")
        ax.set_xlabel("Loan Approved")
        ax.set_ylabel("Count")
        ax.set_title("Loan Approval Counts")
        ax.set_xticklabels(["Approved (Y)", "Denied (N)"], rotation=0)
        plt.tight_layout()
        st.pyplot(fig)

        st.subheader("Missing Values by Column")
        missing = raw_df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if missing.empty:
            st.success("No missing values found.")
        else:
            st.bar_chart(missing)

    except FileNotFoundError:
        st.warning(
            "⚠️ Default dataset not found at `data/credit.csv`. "
            "Please upload the file above."
        )
    except Exception as e:
        st.error(f"Error loading data: {e}")
        logger.exception("Tab1 data load error")


# ─────────────────────────────────────────────────────────────────────────
# TAB 2 — Train & Evaluate
# ─────────────────────────────────────────────────────────────────────────
with tab2:
    st.header("Train & Evaluate Models")

    uploaded2 = st.file_uploader(
        "Upload credit.csv (or use default)",
        type="csv",
        key="upload_tab2",
    )

    with st.expander("⚙️ Hyperparameter Settings (Random Forest)", expanded=False):
        n_estimators = st.slider("n_estimators", 50, 300, 100, step=50)
        max_depth = st.selectbox("max_depth", [None, 3, 4, 5, 8, 10])
        test_size = st.slider("Test set size", 0.10, 0.30, 0.20, step=0.05)

    if st.button("Run Training Pipeline"):
        try:
            source2 = uploaded2 if uploaded2 else DATA_PATH
            df_raw = pd.read_csv(source2) if not isinstance(source2, str) else pd.read_csv(source2)

            with st.spinner("Processing data…"):
                df_clean = impute_missing_values(df_raw)
                df_enc = encode_features(df_clean)
                X, y = split_features_target(df_enc)
                feature_names = X.columns.tolist()

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, stratify=y, random_state=42
                )
                X_tr_sc, X_te_sc, scaler = scale_features(X_train, X_test)

            with st.spinner("Training models…"):
                lr = train_logistic_regression(X_tr_sc, y_train)
                dt = train_decision_tree(X_tr_sc, y_train)
                rf = train_random_forest(
                    X_tr_sc, y_train,
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                )

            st.success("Training complete!")

            # ── Accuracy summary table
            st.subheader("Model Accuracy Comparison")
            rows = []
            models_map = {
                "Logistic Regression": lr,
                "Decision Tree": dt,
                "Random Forest": rf,
            }
            eval_store = {}
            for mname, model in models_map.items():
                ev = evaluate_model(model, X_te_sc, y_test, model_name=mname)
                cv = cross_validate_model(model, X_tr_sc, y_train, model_name=mname)
                eval_store[mname] = {"ev": ev, "cv": cv, "model": model}
                rows.append({
                    "Model": mname,
                    "Test Accuracy": f"{ev['accuracy']:.4f}",
                    "CV Mean": f"{cv['mean_accuracy']:.4f}",
                    "CV Std": f"{cv['std_deviation']:.4f}",
                })
            st.table(pd.DataFrame(rows))

            # ── Confusion matrices
            st.subheader("Confusion Matrices")
            cols = st.columns(3)
            for idx, (mname, data) in enumerate(eval_store.items()):
                with cols[idx]:
                    fig = plot_confusion_matrix(data["ev"]["confusion_matrix"], mname)
                    st.pyplot(fig)

            # ── CV bar charts
            st.subheader("Cross-Validation Scores")
            cols2 = st.columns(3)
            for idx, (mname, data) in enumerate(eval_store.items()):
                with cols2[idx]:
                    fig = plot_cv_scores(data["cv"]["scores"], mname)
                    st.pyplot(fig)

            # ── Feature importances
            st.subheader("Random Forest — Feature Importances")
            fi_df = get_feature_importances(rf, feature_names)
            fig_fi = plot_feature_importances(fi_df, top_n=15)
            st.pyplot(fig_fi)

            # Store models in session for Predict tab
            st.session_state["trained_models"] = {
                "scaler": scaler,
                "feature_names": feature_names,
                "lr": lr, "dt": dt, "rf": rf,
            }

        except FileNotFoundError:
            st.warning("Dataset not found. Please upload credit.csv.")
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            logger.exception("Tab2 pipeline error")


# ─────────────────────────────────────────────────────────────────────────
# TAB 3 — Predict
# ─────────────────────────────────────────────────────────────────────────
with tab3:
    st.header("Predict Loan Eligibility for a New Applicant")

    if "trained_models" not in st.session_state:
        st.info("Please train the models first in the **Train & Evaluate** tab.")
    else:
        tm = st.session_state["trained_models"]
        scaler = tm["scaler"]
        feature_names = tm["feature_names"]

        st.subheader("Applicant Details")

        col_a, col_b = st.columns(2)
        with col_a:
            gender = st.selectbox("Gender", ["Male", "Female"])
            married = st.selectbox("Married", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            self_employed = st.selectbox("Self Employed", ["Yes", "No"])
            property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

        with col_b:
            applicant_income = st.number_input("Applicant Income ($/month)", 0, 100000, 5000, step=500)
            coapplicant_income = st.number_input("Co-Applicant Income ($/month)", 0, 100000, 0, step=500)
            loan_amount = st.number_input("Loan Amount ($000s)", 10, 700, 150, step=10)
            loan_term = st.selectbox("Loan Amount Term (months)", [360, 120, 180, 240, 300, 480, 60, 84])
            credit_history = st.selectbox("Credit History", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

        model_choice = st.selectbox(
            "Select Model",
            ["Logistic Regression", "Decision Tree", "Random Forest"],
        )

        if st.button("Predict"):
            try:
                # Build a one-row DataFrame matching raw input format
                input_dict = {
                    "ApplicantIncome": applicant_income,
                    "CoapplicantIncome": coapplicant_income,
                    "LoanAmount": loan_amount,
                    "Loan_Amount_Term": str(int(loan_term)),
                    "Credit_History": str(int(credit_history)),
                    "Gender": gender,
                    "Married": married,
                    "Dependents": dependents,
                    "Education": education,
                    "Self_Employed": self_employed,
                    "Property_Area": property_area,
                }
                input_df = pd.DataFrame([input_dict])

                # One-hot encode — align to training feature set
                ohe_cols = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]
                input_enc = pd.get_dummies(input_df, columns=ohe_cols, dtype=int)

                # Add any missing columns (from training) with 0
                for col in feature_names:
                    if col not in input_enc.columns:
                        input_enc[col] = 0

                input_enc = input_enc[feature_names]
                input_scaled = scaler.transform(input_enc)

                model_map = {"Logistic Regression": tm["lr"], "Decision Tree": tm["dt"], "Random Forest": tm["rf"]}
                model = model_map[model_choice]
                prediction = model.predict(input_scaled)[0]
                proba = model.predict_proba(input_scaled)[0]

                st.markdown("---")
                if prediction == 1:
                    st.success(f"✅ **Loan Approved** — Confidence: {proba[1]*100:.1f}%")
                else:
                    st.error(f"❌ **Loan Denied** — Confidence: {proba[0]*100:.1f}%")

                st.caption(f"Model used: {model_choice}")

            except Exception as e:
                st.error(f"Prediction error: {e}")
                logger.exception("Tab3 prediction error")
