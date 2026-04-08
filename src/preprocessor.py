"""
preprocessor.py
---------------
Handles all data cleaning, feature engineering and scaling steps
for the loan eligibility dataset.
"""

import logging
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Gender"].fillna("Male", inplace=True)
    df["Married"].fillna(df["Married"].mode()[0], inplace=True)
    df["Dependents"].fillna(df["Dependents"].mode()[0], inplace=True)
    df["Self_Employed"].fillna(df["Self_Employed"].mode()[0], inplace=True)
    df["Loan_Amount_Term"].fillna(360.0, inplace=True)
    df["Credit_History"].fillna(1.0, inplace=True)
    df["LoanAmount"].fillna(df["LoanAmount"].median(), inplace=True)

    # DO NOT cast to object — keep as numeric so sklearn doesn't get NaN
    # df["Credit_History"] = df["Credit_History"].astype("object")  ← removed
    # df["Loan_Amount_Term"] = df["Loan_Amount_Term"].astype("object")  ← removed

    # Catch-all: fill anything still missing
    for col in df.select_dtypes(include="number").columns:
        df[col].fillna(df[col].median(), inplace=True)
    for col in df.select_dtypes(include="object").columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply one-hot encoding to categorical columns and map the
    target variable Loan_Approved from Y/N to 1/0.

    Args:
        df: DataFrame after imputation, still containing Loan_ID.

    Returns:
        Fully encoded DataFrame ready for modelling.
    """
    df = df.copy()

    # Drop identifier column — not a feature
    if "Loan_ID" in df.columns:
        df.drop("Loan_ID", axis=1, inplace=True)
        logger.info("Dropped 'Loan_ID' column.")

    # One-hot encode nominal categorical features
    ohe_cols = [
        "Gender", "Married", "Dependents", "Education",
        "Self_Employed", "Property_Area",
        "Credit_History", "Loan_Amount_Term",
    ]
    ohe_cols = [c for c in ohe_cols if c in df.columns]
    df = pd.get_dummies(df, columns=ohe_cols, dtype=int)
    logger.info("One-hot encoding applied. New shape: %s", df.shape)

    # Encode target
    df["Loan_Approved"] = df["Loan_Approved"].replace({"Y": 1, "N": 0})
    logger.info("Target variable encoded: Y → 1, N → 0.")

    return df


def split_features_target(df: pd.DataFrame):
    """
    Separate feature matrix X and target vector y.

    Args:
        df: Fully encoded DataFrame.

    Returns:
        Tuple (X, y) as DataFrames/Series.
    """
    X = df.drop("Loan_Approved", axis=1)
    y = df["Loan_Approved"]
    logger.info("Features shape: %s | Target shape: %s", X.shape, y.shape)
    return X, y


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple:
    """
    Apply MinMax scaling. Fit on training data only to prevent
    data leakage; transform both train and test sets.

    Args:
        X_train: Training feature matrix.
        X_test:  Test feature matrix.

    Returns:
        Tuple (X_train_scaled, X_test_scaled, scaler).
    """
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    logger.info("MinMax scaling applied to train and test sets.")
    return X_train_scaled, X_test_scaled, scaler
