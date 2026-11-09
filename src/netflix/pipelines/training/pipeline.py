"""Pipeline 'training': compare candidate models, then train the best one."""

from kedro.pipeline import Node, Pipeline

from .nodes import compare_models, train_best_model


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                compare_models,
                inputs=[
                    "X_train",
                    "y_train",
                    "params:model_options",
                    "params:training",
                ],
                outputs="model_comparison",
                name="compare_models",
            ),
            Node(
                train_best_model,
                inputs=[
                    "X_train",
                    "y_train",
                    "model_comparison",
                    "params:model_options",
                    "params:training",
                ],
                outputs="classifier",
                name="train_best_model",
            ),
        ]
    )
