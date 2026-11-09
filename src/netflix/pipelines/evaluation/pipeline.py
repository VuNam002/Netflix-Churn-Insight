"""Pipeline 'evaluation': metrics, segment analysis, importance and plots."""

from kedro.pipeline import Node, Pipeline

from .nodes import (
    compute_feature_importance,
    evaluate_all_models,
    evaluate_by_segment,
    evaluate_model,
    plot_evaluation,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                evaluate_model,
                inputs=["classifier", "X_test", "y_test", "params:decision"],
                outputs="metrics",
                name="evaluate_model",
            ),
            # Reference table only: the model was already chosen by cross-validation.
            Node(
                evaluate_all_models,
                inputs=[
                    "trained_models",
                    "X_test",
                    "y_test",
                    "model_comparison",
                    "best_hyperparameters",
                    "params:decision",
                ],
                outputs="model_comparison_test",
                name="evaluate_all_models",
            ),
            Node(
                evaluate_by_segment,
                inputs=[
                    "classifier",
                    "X_test",
                    "y_test",
                    "params:evaluation",
                    "params:decision",
                ],
                outputs="metrics_by_segment",
                name="evaluate_by_segment",
            ),
            Node(
                compute_feature_importance,
                inputs=["classifier", "X_test", "y_test", "params:evaluation"],
                outputs="feature_importance",
                name="compute_feature_importance",
            ),
            Node(
                plot_evaluation,
                inputs=[
                    "classifier",
                    "X_test",
                    "y_test",
                    "feature_importance",
                    "params:decision",
                ],
                outputs="evaluation_figures",
                name="plot_evaluation",
            ),
        ]
    )
