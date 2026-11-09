import pytest

from netflix.pipelines.data_processing.nodes import clean_data
from netflix.pipelines.feature_engineering.nodes import create_features
from netflix.pipelines.inference.nodes import predict_churn

PARAMS = {"id_column": "customer_id"}


@pytest.fixture
def new_customers(raw_customers, dp_params, fe_params):
    """Ten customers as they arrive at scoring time: no churn label."""
    clean = clean_data(raw_customers.drop(columns="churned").head(10), dp_params)
    return clean, create_features(clean, fe_params)


def test_predict_churn_returns_one_row_per_customer(
    classifier, new_customers, decision
):
    clean, features = new_customers

    predictions = predict_churn(classifier, clean, features, PARAMS, decision)

    assert list(predictions["customer_id"]) == list(clean["customer_id"])
    assert predictions["churn_probability"].between(0, 1).all()
    expected = (predictions["churn_probability"] >= decision["threshold"]).astype(int)
    assert (predictions["churn_prediction"] == expected).all()


def test_predict_churn_ignores_column_order(classifier, new_customers, decision):
    clean, features = new_customers

    shuffled = features[features.columns[::-1]]
    a = predict_churn(classifier, clean, features, PARAMS, decision)
    b = predict_churn(classifier, clean, shuffled, PARAMS, decision)

    assert a.equals(b)


def test_predict_churn_names_missing_columns(classifier, new_customers, decision):
    clean, features = new_customers

    with pytest.raises(ValueError, match="is_inactive"):
        predict_churn(
            classifier, clean, features.drop(columns="is_inactive"), PARAMS, decision
        )
