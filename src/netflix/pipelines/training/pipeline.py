"""Pipeline 'training': compare candidates, tune the winner, train all, pick the best."""

from kedro.pipeline import Node, Pipeline

from .nodes import compare_models, select_best_model, train_all_models, tune_model


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
                tune_model,
                inputs=[
                    "X_train",
                    "y_train",
                    "model_comparison",
                    "params:model_options",
                    "params:training",
                ],
                outputs=["best_hyperparameters", "tuning_results"],
                name="tune_model",
            ),
            Node(
                train_all_models,
                inputs=[
                    "X_train",
                    "y_train",
                    "best_hyperparameters",
                    "params:model_options",
                    "params:training",
                ],
                outputs=["trained_models", "trained_models_info"],
                name="train_all_models",
            ),
            Node(
                select_best_model,
                inputs=["trained_models", "best_hyperparameters"],
                outputs="classifier",
                name="select_best_model",
            ),
        ]
    )
