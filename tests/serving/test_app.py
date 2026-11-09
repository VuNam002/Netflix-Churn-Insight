import pickle

import pytest
from fastapi.testclient import TestClient

from netflix.serving.app import MODEL_PATH_ENV, app


@pytest.fixture
def client(classifier, tmp_path, monkeypatch):
    """The API running with the small test model instead of data/06_models."""
    model_file = tmp_path / "classifier.pkl"
    model_file.write_bytes(pickle.dumps(classifier))
    monkeypatch.setenv(MODEL_PATH_ENV, str(model_file))
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def customer(raw_customers) -> dict:
    row = raw_customers.drop(columns=["churned", "avg_watch_time_per_day"]).iloc[0]
    return row.to_dict() | {"age": int(row["age"])}


def test_health_reports_the_model_in_use(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_path"].endswith("classifier.pkl")


def test_predict_returns_probability_and_decision(client, customer):
    response = client.post("/predict", json=customer)

    body = response.json()
    assert response.status_code == 200
    assert body["customer_id"] == customer["customer_id"]
    assert 0 <= body["churn_probability"] <= 1
    assert body["churn_prediction"] == (body["churn_probability"] >= body["threshold"])


def test_inactive_customer_is_riskier_than_active_one(client, customer):
    active = customer | {"last_login_days": 1, "watch_hours": 40.0}
    inactive = customer | {"last_login_days": 59, "watch_hours": 0.5}

    p_active = client.post("/predict", json=active).json()["churn_probability"]
    p_inactive = client.post("/predict", json=inactive).json()["churn_probability"]

    assert p_inactive > p_active


def test_batch_predict_keeps_order(client, customer):
    batch = [customer | {"customer_id": f"id{i}"} for i in range(3)]

    response = client.post("/predict/batch", json=batch)

    assert response.status_code == 200
    assert [p["customer_id"] for p in response.json()] == ["id0", "id1", "id2"]


def test_invalid_customer_is_rejected(client, customer):
    response = client.post("/predict", json=customer | {"subscription_type": "Gold"})

    assert response.status_code == 422


def test_empty_batch_is_rejected(client):
    assert client.post("/predict/batch", json=[]).status_code == 422


def test_missing_model_fails_at_startup(tmp_path, monkeypatch):
    monkeypatch.setenv(MODEL_PATH_ENV, str(tmp_path / "does_not_exist.pkl"))

    with pytest.raises(FileNotFoundError), TestClient(app):
        pass
