"""Pipeline 'statistics': describe the raw data and its relation to churn."""

from kedro.pipeline import Node, Pipeline

from .nodes import (
    churn_rate_by_feature,
    data_quality,
    describe_categorical,
    describe_numeric,
    target_association,
    target_distribution,
)


def create_pipeline(**kwargs) -> Pipeline:
    params = "params:statistics"
    return Pipeline(
        [
            Node(data_quality, ["netflix_raw", params], "stats_data_quality"),
            Node(target_distribution, ["netflix_raw", params], "stats_target"),
            Node(describe_numeric, ["netflix_raw", params], "stats_numeric"),
            Node(describe_categorical, ["netflix_raw", params], "stats_categorical"),
            Node(churn_rate_by_feature, ["netflix_raw", params], "stats_churn_rate"),
            Node(target_association, ["netflix_raw", params], "stats_association"),
        ]
    )
