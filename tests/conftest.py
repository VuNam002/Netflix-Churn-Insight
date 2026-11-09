"""Shared fixtures: a small synthetic table shaped like the raw churn CSV."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from netflix.pipelines.data_processing.nodes import clean_data
from netflix.pipelines.feature_engineering.nodes import create_features, split_data
from netflix.pipelines.training.nodes import _make_pipeline

CONF = Path(__file__).resolve().parents[1] / "conf" / "base"
FEE = {"Basic": 8.99, "Standard": 13.99, "Premium": 17.99}


def read_params(filename: str, key: str) -> dict:
    """Read one parameter group from conf/base, so tests use the real config."""
    with (CONF / filename).open(encoding="utf-8") as f:
        return yaml.safe_load(f)[key]


@pytest.fixture
def raw_customers() -> pd.DataFrame:
    """300 customers; churn depends on login recency and watch time."""
    rng = np.random.default_rng(0)
    n = 300
    last_login = rng.integers(0, 61, n)
    watch = rng.gamma(2.0, 6.0, n).round(2)
    logit = (last_login - 30) / 10 - (watch - 12) / 6
    plan = rng.choice(list(FEE), n)
    return pd.DataFrame(
        {
            "customer_id": [f"c{i}" for i in range(n)],
            "age": rng.integers(18, 71, n),
            "gender": rng.choice(["Female", "Male", "Other"], n),
            "subscription_type": plan,
            "watch_hours": watch,
            "last_login_days": last_login,
            "region": rng.choice(["Asia", "Europe", "Africa"], n),
            "device": rng.choice(["TV", "Mobile", "Laptop"], n),
            "monthly_fee": [FEE[p] for p in plan],
            "churned": (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int),
            "payment_method": rng.choice(["PayPal", "Gift Card"], n),
            "number_of_profiles": rng.integers(1, 6, n),
            "avg_watch_time_per_day": (watch / (last_login + 1)).round(2),
            "favorite_genre": rng.choice(["Drama", "Comedy"], n),
        }
    )


@pytest.fixture
def dp_params() -> dict:
    return read_params("parameters_data_processing.yml", "data_processing")


@pytest.fixture
def fe_params() -> dict:
    return read_params("parameters_feature_engineering.yml", "feature_engineering")


@pytest.fixture
def model_options() -> dict:
    return read_params("parameters.yml", "model_options")


@pytest.fixture
def decision() -> dict:
    return read_params("parameters.yml", "decision")


@pytest.fixture
def training_params() -> dict:
    """Real config, but small models so the tests stay fast."""
    params = read_params("parameters_training.yml", "training")
    params["cv_folds"] = 3
    params["candidates"]["random_forest"]["n_estimators"] = 20
    params["candidates"]["gradient_boosting"]["n_estimators"] = 20
    return params


@pytest.fixture
def model_features(raw_customers, dp_params, fe_params) -> pd.DataFrame:
    return create_features(clean_data(raw_customers, dp_params), fe_params)


@pytest.fixture
def split(model_features, fe_params, model_options):
    """``X_train, X_test, y_train, y_test`` built by the real nodes."""
    return split_data(model_features, fe_params, model_options)


@pytest.fixture
def classifier(split, model_options, training_params):
    """A fitted logistic regression pipeline."""
    X_train, _, y_train, _ = split
    model = _make_pipeline(
        X_train,
        "logistic_regression",
        training_params["candidates"]["logistic_regression"],
        model_options["random_state"],
    )
    return model.fit(X_train, y_train.iloc[:, 0])
