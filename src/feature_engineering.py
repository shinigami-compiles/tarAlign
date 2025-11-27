"""
feature_engineering.py
----------------------

Pass-through feature engineering for the 20-feature TarAlign model.

- The dataset already contains all 20 core features needed for training.
- This transformer simply selects those 20 columns.
- Implemented as a proper sklearn Transformer so it works inside Pipeline.
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer20(BaseEstimator, TransformerMixin):
    """
    Minimal pass-through feature engineering class.

    It keeps the structure from the old codebase but returns
    the DataFrame restricted to the 20 features specified by
    `feature_list`.
    """

    def __init__(self, feature_list):
        """
        Args:
            feature_list (list[str]): List of the 20 feature names.
        """
        self.feature_list = feature_list

    def fit(self, X: pd.DataFrame, y=None):
        """
        No fitting needed, but kept for pipeline compatibility.

        sklearn will call this as fit(X, y), so we must accept y.
        """
        # You could add validation here if you like:
        # missing = set(self.feature_list) - set(X.columns)
        # if missing:
        #     raise ValueError(f"Missing features in input: {missing}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a copy of X containing ONLY the 20 features.

        Any extra columns (like user_id, date, etc.) are dropped here
        to maintain a clean ML feature matrix.
        """
        return X[self.feature_list].copy()
