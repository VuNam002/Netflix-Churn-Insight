"""Pipeline 'feature_engineering': netflix_clean -> model_features -> train/test."""

from kedro.pipeline import Node, Pipeline

from .nodes import create_features, split_data


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                create_features,
                inputs=["netflix_clean", "params:feature_engineering"],
                outputs="model_features",
                name="create_features",
            ),
            Node(
                split_data,
                inputs=[
                    "model_features",
                    "params:feature_engineering",
                    "params:model_options",
                ],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data",
            ),
        ]
    )
