"""
pipeline.py
-----------
Orchestrates the full training pipeline:
  load → impute → encode → split → scale → train → evaluate → save

Run directly:  py src/pipeline.py
"""

import logging
import os
from sklearn.model_selection import train_test_split

from src.data_loader import load_data
from src.preprocessor import impute_missing_values, encode_features, split_features_target, scale_features
from src.model import (
    train_logistic_regression,
    train_decision_tree,
    train_random_forest,
    evaluate_model,
    cross_validate_model,
    get_feature_importances,
)
from src.utils import setup_logging, save_model

logger = logging.getLogger(__name__)

DATA_PATH = os.path.join("data", "credit.csv")
MODEL_DIR = "models"


def run_pipeline(data_path: str = DATA_PATH) -> dict:
    """
    Execute the complete loan eligibility modelling pipeline.

    Args:
        data_path: Path to raw credit.csv file.

    Returns:
        Dictionary containing trained models and evaluation results
        keyed by model name.
    """
    setup_logging()
    logger.info("=== Loan Eligibility Pipeline START ===")

    # 1. Load
    df = load_data(data_path)

    # 2. Impute
    df = impute_missing_values(df)

    # 3. Encode
    df = encode_features(df)

    # 4. Split features / target
    X, y = split_features_target(df)
    feature_names = X.columns.tolist()

    # 5. Train / test split (stratified to preserve class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    logger.info(
        "Train size: %d | Test size: %d", len(X_train), len(X_test)
    )

    # 6. Scale
    X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test)

    # 7. Train models
    lr = train_logistic_regression(X_train_sc, y_train)
    dt = train_decision_tree(X_train_sc, y_train)
    rf = train_random_forest(X_train_sc, y_train)

    # 8. Evaluate
    results = {}
    for name, model in [
        ("Logistic Regression", lr),
        ("Decision Tree", dt),
        ("Random Forest", rf),
    ]:
        eval_result = evaluate_model(model, X_test_sc, y_test, model_name=name)
        cv_result = cross_validate_model(model, X_train_sc, y_train, model_name=name)
        results[name] = {**eval_result, "cv": cv_result, "model": model}

    # 9. Feature importances for Random Forest
    results["Random Forest"]["feature_importances"] = get_feature_importances(
        rf, feature_names
    )

    # 10. Save models and scaler
    os.makedirs(MODEL_DIR, exist_ok=True)
    save_model(lr, os.path.join(MODEL_DIR, "logistic_regression.pkl"))
    save_model(dt, os.path.join(MODEL_DIR, "decision_tree.pkl"))
    save_model(rf, os.path.join(MODEL_DIR, "random_forest.pkl"))
    save_model(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    save_model(feature_names, os.path.join(MODEL_DIR, "feature_names.pkl"))

    logger.info("=== Pipeline COMPLETE ===")
    return results, scaler, feature_names


if __name__ == "__main__":
    results, _, _ = run_pipeline()
    for name, res in results.items():
        print(f"\n{name}")
        print(f"  Accuracy : {res['accuracy']:.4f}")
        print(f"  CV Mean  : {res['cv']['mean_accuracy']:.4f} ± {res['cv']['std_deviation']:.4f}")
