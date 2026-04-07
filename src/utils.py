"""
utils.py
--------
Shared utilities: logging configuration, chart helpers and
model persistence (save / load).
"""

import logging
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_dir: str = "logs", level: int = logging.INFO) -> None:
    """
    Configure root logger to write to both console and a rotating log file.

    Args:
        log_dir: Directory where loan_eligibility.log will be written.
        level:   Logging level (default INFO).
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "loan_eligibility.log")

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, mode="a", encoding="utf-8"),
        ],
    )
    logging.getLogger(__name__).info("Logging initialised. Log file: %s", log_path)


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    model_name: str = "Model",
) -> plt.Figure:
    """
    Render a labelled heatmap for a confusion matrix.

    Args:
        cm:         2×2 confusion matrix array.
        model_name: Title prefix for the chart.

    Returns:
        Matplotlib Figure object (caller is responsible for displaying/saving).
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Denied", "Approved"],
        yticklabels=["Denied", "Approved"],
        ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"{model_name} — Confusion Matrix")
    plt.tight_layout()
    return fig


def plot_feature_importances(
    importance_df: pd.DataFrame,
    top_n: int = 15,
) -> plt.Figure:
    """
    Plot a horizontal bar chart of the top N feature importances.

    Args:
        importance_df: DataFrame with 'Feature' and 'Importance' columns.
        top_n:         Number of top features to display (default 15).

    Returns:
        Matplotlib Figure object.
    """
    data = importance_df.head(top_n)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(data["Feature"][::-1], data["Importance"][::-1], color="steelblue")
    ax.set_xlabel("Importance Score")
    ax.set_title(f"Top {top_n} Feature Importances (Random Forest)")
    plt.tight_layout()
    return fig


def plot_cv_scores(scores: np.ndarray, model_name: str = "Model") -> plt.Figure:
    """
    Bar chart showing cross-validation accuracy per fold with a mean line.

    Args:
        scores:     Array of CV accuracy scores.
        model_name: Label used in the chart title.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    folds = [f"Fold {i+1}" for i in range(len(scores))]
    ax.bar(folds, scores, color="teal", alpha=0.8, label="Fold Accuracy")
    ax.axhline(scores.mean(), color="red", linestyle="--", label=f"Mean: {scores.mean():.3f}")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{model_name} — 5-Fold Cross-Validation")
    ax.legend()
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_model(model, filepath: str) -> None:
    """
    Serialise a trained sklearn model to disk using joblib.

    Args:
        model:    Fitted sklearn estimator.
        filepath: Destination path (e.g. 'models/rf_model.pkl').
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    logging.getLogger(__name__).info("Model saved to %s", filepath)


def load_model(filepath: str):
    """
    Load a serialised sklearn model from disk.

    Args:
        filepath: Path to the .pkl file.

    Returns:
        Deserialised sklearn estimator.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    model = joblib.load(filepath)
    logging.getLogger(__name__).info("Model loaded from %s", filepath)
    return model
