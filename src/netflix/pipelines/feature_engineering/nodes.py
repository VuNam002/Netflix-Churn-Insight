"""Nodes of the 'feature_engineering' pipeline: build model features and split.

Encoding and scaling are NOT done here. They live inside the scikit-learn pipeline
of the 'training' pipeline, so they are fitted on training folds only and cannot
leak information from the test set.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def create_features(netflix_clean: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    """Turn the clean table into the table the model learns from.

    - ``log_columns`` are replaced by ``log_<name>`` (``log1p``) because they are
      strongly right-skewed (see the 'statistics' step).
    - ``is_inactive``: no login for more than ``inactive_days_threshold`` days.
      Churn jumps around that point.
    - ``has_many_profiles``: at least ``many_profiles_threshold`` profiles. Churn
      drops at that point.
    - ``drop_columns`` (ids and columns unrelated to churn) are removed.

    The target column is kept when present, so the same node serves scoring data.
    """
    df = netflix_clean.copy()

    for col in parameters["log_columns"]:
        df[f"log_{col}"] = np.log1p(df[col])
    df["is_inactive"] = (
        df["last_login_days"] > parameters["inactive_days_threshold"]
    ).astype("int64")
    df["has_many_profiles"] = (
        df["number_of_profiles"] >= parameters["many_profiles_threshold"]
    ).astype("int64")

    to_drop = [*parameters["log_columns"], *parameters["drop_columns"]]
    return df.drop(columns=to_drop)


def split_data(
    model_features: pd.DataFrame, parameters: dict, model_options: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into train and test sets, stratified on the target.

    Returns:
        ``X_train, X_test, y_train, y_test``. The targets are one-column
        DataFrames so they can be stored as parquet.
    """
    target = parameters["target_column"]
    X = model_features.drop(columns=target)
    y = model_features[[target]]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=model_options["test_size"],
        random_state=model_options["random_state"],
        stratify=y[target],
    )
    logger.info("split_data: %d train rows, %d test rows", len(X_train), len(X_test))
    return X_train, X_test, y_train, y_test
