"""
evaluate.py
-----------
Evaluation script for the 20-feature TarAlign model.

This script:
- Loads the trained pipeline saved in models/
- Loads test data using data_loader.py
- Runs predictions and prints evaluation metrics
- Ensures evaluation uses the same 20-feature schema
"""

import os
import joblib
import logging
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)
from data_loader import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = r"D:\my_stuff\Projects\new_projects\TarAlign V2\models\logisticregression_model.joblib"


def evaluate_model():
    """
    Loads the trained model and evaluates it on the test set.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

    logger.info(f"📥 Loading trained model → {MODEL_PATH}")
    pipeline = joblib.load(MODEL_PATH)

    logger.info("📥 Loading dataset (20-feature version)...")
    _, X_test, _, y_test = load_dataset()

    logger.info("🔍 Running predictions on test set...")
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # -------------------------
    # PERFORMANCE METRICS
    # -------------------------
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)

    logger.info("\n🎯 MODEL PERFORMANCE (20-FEATURE VERSION):")
    logger.info(f"Accuracy:  {acc:.4f}")
    logger.info(f"Precision: {prec:.4f}")
    logger.info(f"Recall:    {rec:.4f}")
    logger.info(f"F1 Score:  {f1:.4f}")
    logger.info(f"ROC-AUC:   {auc:.4f}")

    logger.info("\n📊 Classification Report:")
    logger.info("\n" + classification_report(y_test, y_pred))

    # -------------------------
    # PROBABILITY DISTRIBUTION
    # -------------------------
    logger.info("\n🔎 Prediction Probability Stats:")
    logger.info(f"Min prob:   {np.min(y_prob):.4f}")
    logger.info(f"Max prob:   {np.max(y_prob):.4f}")
    logger.info(f"Mean prob:  {np.mean(y_prob):.4f}")
    logger.info(f"Median prob:{np.median(y_prob):.4f}")

    logger.info("\n✨ Evaluation complete.")


if __name__ == "__main__":
    evaluate_model()
