"""
data_loader.py
--------------
Handles loading and initial validation of the loan eligibility dataset.
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the raw credit dataset from a CSV file.

    Args:
        filepath: Path to the CSV file (e.g. 'data/credit.csv').

    Returns:
        Raw DataFrame with all original columns.

    Raises:
        FileNotFoundError: If the file does not exist at the given path.
        ValueError: If required columns are missing from the dataset.
    """
    required_columns = [
        "Loan_ID", "Gender", "Married", "Dependents", "Education",
        "Self_Employed", "ApplicantIncome", "CoapplicantIncome",
        "LoanAmount", "Loan_Amount_Term", "Credit_History",
        "Property_Area", "Loan_Approved",
    ]

    try:
        df = pd.read_csv(filepath)
        logger.info("Dataset loaded successfully. Shape: %s", df.shape)
    except FileNotFoundError:
        logger.error("File not found: %s", filepath)
        raise

    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        logger.error("Missing required columns: %s", missing_cols)
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    logger.info("All required columns present.")
    return df


def get_target_distribution(df: pd.DataFrame) -> pd.Series:
    """
    Return the value counts for the Loan_Approved target column.

    Args:
        df: Raw DataFrame containing the Loan_Approved column.

    Returns:
        Series with counts for each class (Y / N).
    """
    dist = df["Loan_Approved"].value_counts()
    logger.info("Target distribution:\n%s", dist)
    return dist
