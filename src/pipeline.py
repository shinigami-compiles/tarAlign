"""
pipeline.py
-----------
Core training pipeline for the new TarAlign 20-feature model.

This version:
- Loads the new 20-feature dataset
- Uses FeatureEngineer20 (simple pass-through)
- Scales the 20 features
- Trains Logistic Regression
- Saves model + feature list for the Flask web app
"""

import os
import joblib
import json
import logging
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Local imports
from data_loader import load_dataset, get_feature_list
from feature_engineering import FeatureEngineer20

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths for saving the model & feature list
MODELS_DIR = r"D:\my_stuff\Projects\new_projects\TarAlign V2\models"
MODEL_PATH = os.path.join(MODELS_DIR, "logisticregression_model.joblib")
FEATURES_JSON_PATH = os.path.join(MODELS_DIR, "features.json")

os.makedirs(MODELS_DIR, exist_ok=True)


def build_pipeline(feature_list):
    """
    Creates a clean ML pipeline:
      FeatureEngineer20 -> StandardScaler -> LogisticRegression
    """
    pipeline = Pipeline([
        ("engineer", FeatureEngineer20(feature_list)),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=500,
            solver="lbfgs",
            class_weight="balanced"  # improves calibration & precision
        ))
    ])
    return pipeline


def train_and_save():
    logger.info("📥 Loading dataset...")
    X_train, X_test, y_train, y_test = load_dataset()

    feature_list = get_feature_list()
    logger.info(f"📌 Using {len(feature_list)} features.")

    logger.info("🔧 Building training pipeline...")
    pipeline = build_pipeline(feature_list)

    logger.info("🚀 Training model...")
    pipeline.fit(X_train, y_train)

    logger.info("📊 Evaluating...")
    score = pipeline.score(X_test, y_test)
    logger.info(f"✔️ Test Accuracy: {score:.4f}")

    # --------------------------
    # SAVE MODEL + FEATURE LIST
    # --------------------------

    logger.info(f"💾 Saving model → {MODEL_PATH}")
    joblib.dump(pipeline, MODEL_PATH)

    logger.info(f"💾 Saving feature list → {FEATURES_JSON_PATH}")
    with open(FEATURES_JSON_PATH, "w") as f:
        json.dump(feature_list, f, indent=2)

    logger.info("🎉 Training complete! Model + features saved.")


if __name__ == "__main__":
    train_and_save()
