import numpy as np
import pandas as pd

from netflix.pipelines.statistics.nodes import (
    churn_rate_by_feature,
    data_quality,
    describe_categorical,
    describe_numeric,
    target_association,
    target_distribution,
)

PARAMS = {
    "id_column": "customer_id",
    "target_column": "churned",
    "numeric_bins": 4,
    "alpha": 0.05,
}


def make_raw(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    churned = rng.integers(0, 2, n)
    return pd.DataFrame(
        {
            "customer_id": [f"c{i}" for i in range(n)],
            "signal": churned * 10 + rng.normal(0, 1, n),  # strongly tied to churn
            "noise": rng.normal(0, 1, n),
            "plan": np.where(churned == 1, "Basic", "Premium"),  # tied to churn
            "colour": rng.choice(["red", "blue"], n),  # unrelated to churn
            "churned": churned,
        }
    )


def test_data_quality_counts_duplicates_and_missing():
    raw = make_raw(10)
    raw = pd.concat([raw, raw.iloc[[0]]])
    raw.loc[1, "noise"] = None

    report = data_quality(raw, PARAMS)

    assert report["rows"] == 11
    assert report["duplicated_ids"] == 1
    assert report["missing_cells"] == 1
    assert report["columns_with_missing"] == ["noise"]


def test_target_distribution_shares_sum_to_one():
    result = target_distribution(make_raw(), PARAMS)

    assert result["share"].sum() == 1.0
    assert result["count"].sum() == 400


def test_describe_numeric_excludes_target_and_counts_outliers():
    raw = make_raw()
    raw.loc[0, "noise"] = 1000  # extreme outlier

    summary = describe_numeric(raw, PARAMS).set_index("feature")

    assert "churned" not in summary.index
    assert summary.loc["noise", "outliers_iqr"] >= 1


def test_describe_categorical_excludes_id():
    summary = describe_categorical(make_raw(), PARAMS)

    assert set(summary["feature"]) == {"plan", "colour"}


def test_churn_rate_by_feature_covers_numeric_and_categorical():
    rates = churn_rate_by_feature(make_raw(), PARAMS)

    assert set(rates["feature"]) == {"signal", "noise", "plan", "colour"}
    basic = rates[(rates["feature"] == "plan") & (rates["level"] == "Basic")]
    assert basic["churn_rate"].iloc[0] == 1.0


def test_target_association_separates_signal_from_noise():
    result = target_association(make_raw(), PARAMS).set_index("feature")

    assert result.loc["signal", "significant"]
    assert result.loc["plan", "significant"]
    assert not result.loc["colour", "significant"]
    assert not result.loc["noise", "significant"]
    assert result["strength"].is_monotonic_decreasing
