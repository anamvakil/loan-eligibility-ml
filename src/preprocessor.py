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
    """
    Fill missing values using domain-appropriate strategies:
    - Categorical columns: mode imputation
    - LoanAmount (numerical with outliers): median imputation
    - Gender: hardcoded to 'Male' (most frequent)

    Args:
        df: Raw DataFrame.

    Returns:
        DataFrame with no missing values.
    """
    df = df.copy()

    # Impute BEFORE casting — NaN-safe operations first
    df["Gender"].fillna("Male", inplace=True)
    df["Married"].fillna(df["Married"].mode()[0], inplace=True)
    df["Dependents"].fillna(df["Dependents"].mode()[0], inplace=True)
    df["Self_Employed"].fillna(df["Self_Employed"].mode()[0], inplace=True)
    df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mode()[0], inplace=True)
    df["Credit_History"].fillna(df["Credit_History"].mode()[0], inplace=True)

    median_val = df["LoanAmount"].median()
    df["LoanAmount"].fillna(median_val, inplace=True)
    logger.debug("Imputed 'LoanAmount' with median: %.2f", median_val)

    # Cast AFTER NaNs are gone — safe to convert to object now
    df["Credit_History"] = df["Credit_History"].astype("object")
    df["Loan_Amount_Term"] = df["Loan_Amount_Term"].astype("object")

    remaining_nulls = df.isnull().sum().sum()
    if remaining_nulls > 0:
        logger.warning("Still %d null values after imputation.", remaining_nulls)
    else:
        logger.info("All missing values successfully imputed.")

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
