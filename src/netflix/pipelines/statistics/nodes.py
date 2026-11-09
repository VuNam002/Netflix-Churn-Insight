"""Nodes of the 'statistics' pipeline: describe the raw churn table.

Each node answers one question about the data and returns a small table or dict
that is saved under ``data/08_reporting``.
"""

import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def data_quality(netflix_raw: pd.DataFrame, parameters: dict) -> dict:
    """Is the table complete and free of duplicates?"""
    id_col = parameters["id_column"]
    report = {
        "rows": len(netflix_raw),
        "columns": netflix_raw.shape[1],
        "missing_cells": int(netflix_raw.isna().sum().sum()),
        "duplicated_rows": int(netflix_raw.duplicated().sum()),
        "duplicated_ids": int(netflix_raw[id_col].duplicated().sum()),
        "columns_with_missing": netflix_raw.columns[netflix_raw.isna().any()].tolist(),
    }
    logger.info("data_quality: %s", report)
    return report


def target_distribution(netflix_raw: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    """How many customers churned? Shows whether the classes are balanced."""
    counts = netflix_raw[parameters["target_column"]].value_counts().sort_index()
    return pd.DataFrame(
        {"class": counts.index, "count": counts.values, "share": counts / counts.sum()}
    ).reset_index(drop=True)


def describe_numeric(netflix_raw: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    """Centre, spread, shape and outliers of every numeric feature.

    ``outliers_iqr`` counts values beyond 1.5 * IQR from the quartiles.
    """
    target = parameters["target_column"]
    numeric = netflix_raw.select_dtypes(include="number").drop(columns=target)

    summary = numeric.describe().T
    summary["median"] = numeric.median()
    summary["skew"] = numeric.skew()
    q1, q3 = summary["25%"], summary["75%"]
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    summary["outliers_iqr"] = ((numeric < low) | (numeric > high)).sum()
    return summary.round(3).rename_axis("feature").reset_index()


def describe_categorical(netflix_raw: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    """Number of levels and the most common level of every categorical feature."""
    id_col = parameters["id_column"]
    categorical = netflix_raw.select_dtypes(exclude="number").drop(columns=id_col)

    rows = []
    for col in categorical:
        shares = categorical[col].value_counts(normalize=True)
        rows.append(
            {
                "feature": col,
                "n_levels": len(shares),
                "top_level": shares.index[0],
                "top_share": round(shares.iloc[0], 3),
                "smallest_level": shares.index[-1],
                "smallest_share": round(shares.iloc[-1], 3),
            }
        )
    return pd.DataFrame(rows)


def churn_rate_by_feature(netflix_raw: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    """Churn rate inside each level of a categorical feature.

    Numeric features are cut into ``numeric_bins`` equal-size groups first.
    """
    target = parameters["target_column"]
    id_col = parameters["id_column"]
    features = netflix_raw.drop(columns=[target, id_col])

    frames = []
    for col in features:
        if pd.api.types.is_numeric_dtype(features[col]) and features[col].nunique() > 8:  # noqa: PLR2004
            group = pd.qcut(
                features[col], parameters["numeric_bins"], duplicates="drop"
            )
        else:
            group = features[col]
        rate = netflix_raw.groupby(group, observed=True)[target].agg(["count", "mean"])
        rate.index = rate.index.astype(str)
        frames.append(
            rate.rename(columns={"mean": "churn_rate"})
            .round(3)
            .rename_axis("level")
            .reset_index()
            .assign(feature=col)[["feature", "level", "count", "churn_rate"]]
        )
    return pd.concat(frames, ignore_index=True)


def target_association(netflix_raw: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    """Which features are related to churn, and how strongly?

    Numeric features use Spearman correlation with the target. Categorical features
    use the chi-square test and Cramer's V. ``p_value`` below ``alpha`` means the
    relation is unlikely to be chance; ``strength`` is the effect size (0 to 1).
    """
    target = parameters["target_column"]
    id_col = parameters["id_column"]
    alpha = parameters["alpha"]
    y = netflix_raw[target]

    rows = []
    for col in netflix_raw.drop(columns=[target, id_col]):
        x = netflix_raw[col]
        if pd.api.types.is_numeric_dtype(x):
            rho, p_value = stats.spearmanr(x, y)
            rows.append(("numeric", "spearman", col, rho, abs(rho), p_value))
        else:
            table = pd.crosstab(x, y)
            chi2, p_value, _, _ = stats.chi2_contingency(table)
            cramers_v = np.sqrt(
                chi2 / (table.to_numpy().sum() * (min(table.shape) - 1))
            )
            rows.append(("categorical", "chi2", col, np.nan, cramers_v, p_value))

    result = pd.DataFrame(
        rows,
        columns=["type", "test", "feature", "correlation", "strength", "p_value"],
    )
    result["significant"] = result["p_value"] < alpha
    return (
        result.sort_values("strength", ascending=False).round(4).reset_index(drop=True)
    )
