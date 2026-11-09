"""Pipeline 'inference': new_customers -> predictions (batch scoring).

Reuses ``clean_data`` and ``create_features`` from the training pipelines, so new
customers are prepared exactly like the training data.
"""

from kedro.pipeline import Node, Pipeline

from netflix.pipelines.data_processing.nodes import clean_data
from netflix.pipelines.feature_engineering.nodes import create_features

from .nodes import predict_churn


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                clean_data,
                inputs=["new_customers", "params:data_processing"],
                outputs="new_customers_clean",
                name="clean_new_customers",
            ),
            Node(
                create_features,
                inputs=["new_customers_clean", "params:feature_engineering"],
                outputs="new_features",
                name="create_new_features",
            ),
            Node(
                predict_churn,
                inputs=[
                    "classifier",
                    "new_customers_clean",
                    "new_features",
                    "params:inference",
                    "params:decision",
                ],
                outputs="predictions",
                name="predict_churn",
            ),
        ]
    )
