import numpy as np
import pytest

from netflix.pipelines.data_processing.nodes import clean_data
from netflix.pipelines.feature_engineering.nodes import create_features, split_data


def test_create_features_replaces_and_drops_columns(model_features, fe_params):
    for col in fe_params["drop_columns"] + fe_params["log_columns"]:
        assert col not in model_features.columns
    for col in fe_params["log_columns"]:
        assert f"log_{col}" in model_features.columns
    assert "churned" in model_features.columns  # target is kept


def test_create_features_flags_follow_thresholds(raw_customers, dp_params, fe_params):
    clean = clean_data(raw_customers, dp_params)

    features = create_features(clean, fe_params)

    inactive = clean["last_login_days"] > fe_params["inactive_days_threshold"]
    many = clean["number_of_profiles"] >= fe_params["many_profiles_threshold"]
    assert (features["is_inactive"] == inactive.astype(int)).all()
    assert (features["has_many_profiles"] == many.astype(int)).all()


def test_log_features_are_log1p_of_the_original(raw_customers, dp_params, fe_params):
    clean = clean_data(raw_customers, dp_params)

    features = create_features(clean, fe_params)

    np.testing.assert_allclose(
        features["log_watch_hours"], np.log1p(clean["watch_hours"])
    )


def test_create_features_works_without_target(raw_customers, dp_params, fe_params):
    clean = clean_data(raw_customers.drop(columns="churned"), dp_params)

    features = create_features(clean, fe_params)

    assert "churned" not in features.columns


def test_split_data_is_stratified_and_has_no_target_in_x(
    model_features, fe_params, model_options
):
    X_train, X_test, y_train, y_test = split_data(
        model_features, fe_params, model_options
    )

    assert len(X_train) + len(X_test) == len(model_features)
    assert len(X_test) == pytest.approx(
        len(model_features) * model_options["test_size"], abs=1
    )
    assert y_train["churned"].mean() == pytest.approx(
        y_test["churned"].mean(), abs=0.03
    )
    assert "churned" not in X_train.columns
    assert list(X_train.index) == list(y_train.index)
