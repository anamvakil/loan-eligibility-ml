import logging
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Cast to object BEFORE imputing
    df['Credit_History'] = df['Credit_History'].astype('object')
    df['Loan_Amount_Term'] = df['Loan_Amount_Term'].astype('object')

    # Use assignment NOT inplace=True (inplace silently fails on pandas 2.x)
    df['Gender'] = df['Gender'].fillna('Male')
    df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
    df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])
    df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])
    df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0])
    df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])
    df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())

    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'Loan_ID' in df.columns:
        df = df.drop('Loan_ID', axis=1)

    ohe_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
    df = pd.get_dummies(df, columns=ohe_cols, dtype=int)
    df['Loan_Approved'] = df['Loan_Approved'].replace({'Y': 1, 'N': 0})

    # Convert remaining object columns to numeric
    for col in df.select_dtypes(include='object').columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fill any remaining NaNs
    df = df.fillna(df.median(numeric_only=True))

    return df


def split_features_target(df: pd.DataFrame):
    X = df.drop('Loan_Approved', axis=1)
    y = df['Loan_Approved']
    return X, y


def scale_features(X_train, X_test):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler
