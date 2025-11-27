"""
train_models.py
----------------
A light wrapper around pipeline.py to:
- Train the 20-feature TarAlign model,
- Print performance metrics,
- Save model + feature list.

This replaces the older version that required engineered 63 features.
"""

import logging
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Local imports
from pipeline import train_and_save
from data_loader import load_dataset
from data_loader import get_feature_list

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_and_evaluate():
    """
    Trains the model using pipeline.py and prints evaluation metrics.
    """
    logger.info("🚀 Starting TarAlign model training (20-feature version)...")

    # First, train and save the model using the pipeline
    train_and_save()

    # Load dataset split
    X_train, X_test, y_train, y_test = load_dataset()

    # Load freshly saved model
    model_path = r"D:\my_stuff\Projects\new_projects\TarAlign V2\models\logisticregression_model.joblib"
    logger.info(f"📥 Loading saved model from: {model_path}")
    pipeline = joblib.load(model_path)

    # Predict on test
    logger.info("🔍 Running predictions for evaluation...")
    y_pred = pipeline.predict(X_test)

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    logger.info("\n🎯 MODEL PERFORMANCE (20-FEATURE VERSION):")
    logger.info(f"Accuracy:  {acc:.4f}")
    logger.info(f"Precision: {prec:.4f}")
    logger.info(f"Recall:    {rec:.4f}")
    logger.info(f"F1-Score:  {f1:.4f}")

    # Print detailed report
    logger.info("\n📊 Classification Report:")
    logger.info("\n" + classification_report(y_test, y_pred))


if __name__ == "__main__":
    train_and_evaluate()
