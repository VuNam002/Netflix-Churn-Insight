"""Nodes of the 'inference' pipeline: score new customers with the trained model.

The cleaning and feature nodes are the same ones used for training (see the
pipeline), so new data goes through exactly the same transformations.
"""

import logging

import pandas as pd
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def predict_churn(
    classifier: Pipeline,
    new_customers_clean: pd.DataFrame,
    new_features: pd.DataFrame,
    parameters: dict,
    decision: dict,
) -> pd.DataFrame:
    """Churn probability and yes/no prediction for every new customer.

    Args:
        classifier: Fitted scikit-learn pipeline.
        new_customers_clean: Cleaned customers; supplies the id column.
        new_features: Features built from the same customers (same row order).
        parameters: ``inference`` parameters (id column name).
        decision: Decision threshold.
    """
    expected = list(classifier.feature_names_in_)
    missing = [col for col in expected if col not in new_features.columns]
    if missing:
        raise ValueError(f"New data is missing the columns the model needs: {missing}")

    probability = classifier.predict_proba(new_features[expected])[:, 1]
    predictions = pd.DataFrame(
        {
            parameters["id_column"]: new_customers_clean[
                parameters["id_column"]
            ].to_numpy(),
            "churn_probability": probability.round(4),
            "churn_prediction": (probability >= decision["threshold"]).astype(int),
        }
    )
    logger.info(
        "predict_churn: %d customers, %d predicted to churn",
        len(predictions),
        predictions["churn_prediction"].sum(),
    )
    return predictions
