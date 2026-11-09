"""Nodes of the 'data_processing' pipeline: validate and clean the raw churn table."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def clean_data(netflix_raw: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    """Return a validated table with one row per customer.

    Drops duplicated customers and rows with missing values or impossible values
    (non-positive age or fee, negative watch hours), then fixes column dtypes.
    The target column is optional so that the same node cleans new customers
    at scoring time.

    Args:
        netflix_raw: Raw customer table.
        parameters: ``data_processing`` parameters (id and target column names).
    """
    id_col = parameters["id_column"]
    target = parameters["target_column"]

    df = netflix_raw.copy()
    n_start = len(df)

    df = df.drop_duplicates(subset=id_col)
    df = df.dropna()
    df = df[(df["age"] > 0) & (df["monthly_fee"] > 0) & (df["watch_hours"] >= 0)]

    if target in df.columns:
        df[target] = df[target].astype("int64")
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype("string").str.strip()

    logger.info("clean_data: kept %d of %d rows", len(df), n_start)
    return df.reset_index(drop=True)
