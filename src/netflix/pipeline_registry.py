"""Project pipelines.

Pipelines are registered explicitly, not with ``find_pipelines``: Kedro narrows
``find_pipelines`` to the name given to ``--pipeline``, so it cannot build the
combined pipelines below. When you add a pipeline, add it here too.
"""

from __future__ import annotations

from kedro.pipeline import Pipeline

from netflix.pipelines import (
    data_processing,
    evaluation,
    feature_engineering,
    inference,
    statistics,
    training,
    visualization,
)


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    ``kedro run`` trains and evaluates the model. Exploration and scoring are
    separate, so they run only when asked for by name:

        kedro run                       # steps 3, 5, 6: clean, features, train, evaluate
        kedro run --pipeline eda        # steps 2, 4: statistics and charts
        kedro run --pipeline inference  # step 7: score data/01_raw/new_customers.csv

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    clean = data_processing.create_pipeline()
    features = feature_engineering.create_pipeline()
    train = training.create_pipeline()
    evaluate = evaluation.create_pipeline()
    stats = statistics.create_pipeline()
    charts = visualization.create_pipeline()

    return {
        "data_processing": clean,
        "feature_engineering": features,
        "statistics": stats,
        "visualization": charts,
        "training": train,
        "evaluation": evaluate,
        "inference": inference.create_pipeline(),
        "eda": clean + stats + charts,
        "__default__": clean + features + train + evaluate,
    }
