"""
explainability.py
-----------------
Explainability utilities for the 20-feature TarAlign model.

This script:
- Loads the trained pipeline (with Logistic Regression)
- Loads the 20-feature dataset
- Extracts feature importances from Logistic Regression coefficients
- Shows which features push the model towards "Aligned" (1)
  and "Misaligned" (0).

NOTE:
- This does NOT use SHAP (to avoid extra dependencies),
  but uses Logistic Regression coefficients as a simple
  linear explainability method.
"""

import os
import logging
import numpy as np
import joblib
import pandas as pd

from data_loader import load_dataset, get_feature_list

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Must match your other scripts
MODELS_DIR = r"D:\my_stuff\Projects\new_projects\TarAlign V2\models"
MODEL_PATH = os.path.join(MODELS_DIR, "logisticregression_model.joblib")


def load_pipeline():
    """Load the trained pipeline from disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

    logger.info(f"📥 Loading pipeline from: {MODEL_PATH}")
    pipeline = joblib.load(MODEL_PATH)
    return pipeline


def get_logistic_coefficients(pipeline):
    """
    Extract Logistic Regression coefficients and feature names
    from the trained pipeline.
    """
    if "model" not in pipeline.named_steps:
        raise ValueError("Pipeline has no 'model' step. Cannot extract coefficients.")

    model = pipeline.named_steps["model"]

    if not hasattr(model, "coef_"):
        raise ValueError("Model does not have 'coef_' attribute; not a linear model.")

    coefs = model.coef_[0]  # binary classification
    feature_names = get_feature_list()

    if len(coefs) != len(feature_names):
        raise ValueError(
            f"Coefficient length {len(coefs)} != number of features {len(feature_names)}."
        )

    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefs
    })

    # Sort by coefficient strength
    coef_df["abs_coeff"] = coef_df["coefficient"].abs()
    coef_df = coef_df.sort_values("abs_coeff", ascending=False)

    return coef_df


def print_global_importance(coef_df, top_n=10):
    """
    Print top positive and negative features influencing alignment.
    """
    logger.info("\n🌟 GLOBAL FEATURE IMPORTANCE (by Logistic Regression coefficients)")

    # Top positive (push towards aligned = 1)
    pos_df = coef_df.sort_values("coefficient", ascending=False).head(top_n)
    logger.info("\n🔼 Top features pushing TOWARDS 'Aligned' (class 1):")
    for _, row in pos_df.iterrows():
        logger.info(f"  {row['feature']:<30}  coef={row['coefficient']:+.4f}")

    # Top negative (push towards misaligned = 0)
    neg_df = coef_df.sort_values("coefficient", ascending=True).head(top_n)
    logger.info("\n🔽 Top features pushing TOWARDS 'Misaligned' (class 0):")
    for _, row in neg_df.iterrows():
        logger.info(f"  {row['feature']:<30}  coef={row['coefficient']:+.4f}")


def explain_single_example(pipeline, X_sample: pd.Series, coef_df: pd.DataFrame):
    """
    Provide a simple explanation for a single example
    based on feature * value * coefficient contribution.

    Args:
        pipeline: trained pipeline (with scaler + model)
        X_sample: a single row from the dataset (before scaling)
        coef_df: DataFrame with feature & coefficient
    """
    # Get feature names and coefficients
    feature_names = coef_df["feature"].tolist()
    coefs = coef_df.set_index("feature")["coefficient"]

    # Ensure order
    x_vec = X_sample[feature_names].astype(float)

    # Contribution in the linear space: value * coef
    contributions = x_vec * coefs

    contrib_df = pd.DataFrame({
        "feature": feature_names,
        "value": x_vec.values,
        "coefficient": coefs.values,
        "contribution": contributions.values
    })

    contrib_df["abs_contribution"] = contrib_df["contribution"].abs()
    contrib_df = contrib_df.sort_values("abs_contribution", ascending=False)

    # Predict probability for this sample
    input_df = pd.DataFrame([X_sample[feature_names].values], columns=feature_names)
    prob = float(pipeline.predict_proba(input_df)[0, 1])
    pred_class = int(pipeline.predict(input_df)[0])

    logger.info("\n🧪 SINGLE EXAMPLE EXPLANATION:")
    logger.info(f"  Predicted class: {pred_class} ({'Aligned' if pred_class == 1 else 'Misaligned'})")
    logger.info(f"  Predicted probability (class 1 - Aligned): {prob:.4f}")

    logger.info("\n  Top contributing features for THIS example:")
    for _, row in contrib_df.head(10).iterrows():
        logger.info(
            f"   {row['feature']:<30} "
            f"value={row['value']:.3f}  "
            f"coef={row['coefficient']:+.4f}  "
            f"contrib={row['contribution']:+.4f}"
        )


def main():
    # Load model & data
    pipeline = load_pipeline()
    X_train, X_test, y_train, y_test = load_dataset()

    # Get global coefficients
    coef_df = get_logistic_coefficients(pipeline)
    print_global_importance(coef_df, top_n=10)

    # Explain a single example (e.g., the first test sample)
    if len(X_test) > 0:
        sample_idx = 0
        X_sample = X_test.iloc[sample_idx]
        logger.info(f"\n🧾 Explaining test sample index: {sample_idx}")
        explain_single_example(pipeline, X_sample, coef_df)
    else:
        logger.info("No test samples available to explain.")


if __name__ == "__main__":
    main()
