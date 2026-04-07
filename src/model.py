"""
model.py
--------
Trains Logistic Regression, Decision Tree and Random Forest classifiers,
evaluates them and runs 5-fold cross-validation.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score, KFold

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_logistic_regression(X_train, y_train) -> LogisticRegression:
    """
    Fit a Logistic Regression classifier.

    Args:
        X_train: Scaled training features.
        y_train: Training labels.

    Returns:
        Fitted LogisticRegression instance.
    """
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    logger.info("Logistic Regression trained.")
    return model


def train_decision_tree(X_train, y_train) -> DecisionTreeClassifier:
    """
    Fit a Decision Tree classifier.

    Args:
        X_train: Scaled training features.
        y_train: Training labels.

    Returns:
        Fitted DecisionTreeClassifier instance.
    """
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    logger.info("Decision Tree trained.")
    return model


def train_random_forest(
    X_train,
    y_train,
    n_estimators: int = 100,
    max_depth: int = None,
    max_features: str = "sqrt",
) -> RandomForestClassifier:
    """
    Fit a Random Forest classifier with configurable hyperparameters.

    Args:
        X_train:      Scaled training features.
        y_train:      Training labels.
        n_estimators: Number of trees (default 100).
        max_depth:    Maximum tree depth (default None = unlimited).
        max_features: Features to consider per split (default 'sqrt').

    Returns:
        Fitted RandomForestClassifier instance.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        random_state=42,
    )
    model.fit(X_train, y_train)
    logger.info(
        "Random Forest trained. n_estimators=%d, max_depth=%s",
        n_estimators,
        max_depth,
    )
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, X_test, y_test, model_name: str = "Model") -> dict:
    """
    Generate predictions and compute accuracy, confusion matrix
    and a full classification report.

    Args:
        model:      Fitted sklearn classifier.
        X_test:     Scaled test features.
        y_test:     True test labels.
        model_name: Label used in log messages.

    Returns:
        Dictionary with keys: accuracy, confusion_matrix, report, y_pred.
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    logger.info("%s | Accuracy: %.4f", model_name, acc)
    logger.info("%s | Confusion Matrix:\n%s", model_name, cm)

    return {
        "accuracy": acc,
        "confusion_matrix": cm,
        "report": report,
        "y_pred": y_pred,
    }


def evaluate_with_threshold(
    model,
    X_test,
    y_test,
    threshold: float = 0.7,
) -> dict:
    """
    Evaluate Logistic Regression using a custom probability threshold.

    Args:
        model:     Fitted LogisticRegression model.
        X_test:    Scaled test features.
        y_test:    True test labels.
        threshold: Probability cutoff for positive class (default 0.70).

    Returns:
        Dictionary with accuracy and y_pred at the custom threshold.
    """
    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= threshold).astype(int)
    acc = accuracy_score(y_test, y_pred)
    logger.info("Custom threshold (%.2f) accuracy: %.4f", threshold, acc)
    return {"accuracy": acc, "y_pred": y_pred, "threshold": threshold}


# ---------------------------------------------------------------------------
# Cross-Validation
# ---------------------------------------------------------------------------

def cross_validate_model(
    model,
    X_train,
    y_train,
    n_splits: int = 5,
    model_name: str = "Model",
) -> dict:
    """
    Run KFold cross-validation on the training set.

    Args:
        model:      Fitted or unfitted sklearn classifier.
        X_train:    Scaled training features.
        y_train:    Training labels.
        n_splits:   Number of CV folds (default 5).
        model_name: Label used in log messages.

    Returns:
        Dictionary with scores array, mean accuracy and std deviation.
    """
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=kfold)

    logger.info(
        "%s | CV Mean: %.4f | CV Std: %.4f",
        model_name,
        scores.mean(),
        scores.std(),
    )
    return {
        "scores": scores,
        "mean_accuracy": scores.mean(),
        "std_deviation": scores.std(),
    }


# ---------------------------------------------------------------------------
# Feature Importance
# ---------------------------------------------------------------------------

def get_feature_importances(
    model: RandomForestClassifier,
    feature_names: list,
) -> pd.DataFrame:
    """
    Extract feature importances from a trained Random Forest model.

    Args:
        model:         Fitted RandomForestClassifier.
        feature_names: List of column names matching the training features.

    Returns:
        DataFrame with Feature and Importance columns, sorted descending.
    """
    importances = model.feature_importances_
    df = pd.DataFrame(
        {"Feature": feature_names, "Importance": importances}
    ).sort_values("Importance", ascending=False).reset_index(drop=True)

    logger.info("Feature importances extracted. Top feature: %s", df.iloc[0]["Feature"])
    return df
