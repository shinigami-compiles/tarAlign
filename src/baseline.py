"""
baseline.py
-----------
A simple baseline comparison model for the 20-feature TarAlign dataset.

This script:
- Loads the dataset using data_loader.py
- Trains a baseline classifier (DecisionTree or Dummy)
- Prints evaluation metrics
"""

import logging
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from data_loader import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_baseline(use_tree=True):
    """
    Trains a simple baseline model to compare against Logistic Regression.

    Args:
        use_tree (bool): If True → Decision Tree baseline
                         If False → Dummy Classifier baseline
    """
    logger.info("📥 Loading dataset for baseline...")
    X_train, X_test, y_train, y_test = load_dataset()

    if use_tree:
        logger.info("🌳 Training Decision Tree baseline...")
        model = DecisionTreeClassifier(
            max_depth=4,
            random_state=42
        )
    else:
        logger.info("🎯 Training Dummy (most frequent class) baseline...")
        model = DummyClassifier(strategy="most_frequent")

    # Train
    model.fit(X_train, y_train)

    # Predict
    logger.info("🔍 Running predictions...")
    y_pred = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    logger.info("\n🎯 BASELINE MODEL PERFORMANCE:")
    logger.info(f"Accuracy:  {acc:.4f}")
    logger.info(f"Precision: {prec:.4f}")
    logger.info(f"Recall:    {rec:.4f}")
    logger.info(f"F1 Score:  {f1:.4f}")

    logger.info("\n📊 Classification Report:")
    logger.info("\n" + classification_report(y_test, y_pred))

    logger.info("✨ Baseline evaluation complete.")


if __name__ == "__main__":
    # Set use_tree=True for Decision Tree baseline
    run_baseline(use_tree=True)
