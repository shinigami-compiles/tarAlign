"""
data_loader.py
--------------
Loads the NEW 20-feature TarAlign dataset and prepares it
for model training & evaluation.

This module replaces the old version that depended on
63 engineered features. Now everything is consistent with
the web application's 20-core input features.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Path to your new dataset (5400 rows, 20 features)
DATASET_PATH = r"D:\my_stuff\Projects\new_projects\TarAlign V2\alignment_dataset_5400_20features.csv"

# The exact 20 features used for model training & inference
FEATURE_COLUMNS = [
    "minutes_goal",
    "consistency_index",
    "sleep_hours",
    "avg_minutes_last_week",
    "avg_consistency_last_week",
    "avg_sleep_last_week",
    "avg_minutes_last_month",
    "momentum_last_month",
    "friction_last_month",
    "avg_sleep_last_month",
    "baseline_goal_minutes",
    "baseline_consistency",
    "baseline_sleep_hours",
    "task_switch_avg",
    "exercise_avg",
    "day_of_week",
    "is_weekend",
    "goal_weight_career",
    "goal_weight_fitness",
    "goal_weight_learning",
]

TARGET_COLUMN = "aligned_label"


def load_dataset(test_size=0.2, random_state=42):
    """
    Loads the 20-feature dataset and returns train/test splits.

    Returns:
        X_train, X_test, y_train, y_test
    """
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH}. "
            f"Ensure the 20-feature dataset has been generated first."
        )

    print(f"📥 Loading dataset: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)

    # Validate columns
    missing_cols = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset missing required columns: {missing_cols}")

    # Feature matrix and target vector
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].astype(int)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        shuffle=True,
        stratify=y,
        random_state=random_state
    )

    print("✅ Dataset loaded successfully.")
    print(f"   Total rows      : {len(df)}")
    print(f"   Train rows      : {len(X_train)}")
    print(f"   Test rows       : {len(X_test)}")
    print(f"   Class balance   : {df[TARGET_COLUMN].value_counts(normalize=True)}")

    return X_train, X_test, y_train, y_test


def get_feature_list():
    """Returns the list of 20 features."""
    return FEATURE_COLUMNS


if __name__ == "__main__":
    # Test loading
    load_dataset()
