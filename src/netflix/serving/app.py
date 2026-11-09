"""FastAPI service that returns churn predictions for customers sent as JSON.

Run locally from the project root:

    uvicorn netflix.serving.app:app --reload

The model is the newest ``data/06_models/classifier.pkl/<timestamp>/classifier.pkl``
written by ``kedro run``; set ``CHURN_MODEL_PATH`` to use a specific file.
Customers go through the same ``create_features`` step as in training.
"""

import os
import pickle
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from netflix.pipelines.feature_engineering.nodes import create_features

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONF_DIR = PROJECT_ROOT / "conf" / "base"
MODEL_PATH_ENV = "CHURN_MODEL_PATH"


class Customer(BaseModel):
    """One customer, with the same fields as the raw CSV (without ``churned``)."""

    customer_id: str
    age: int = Field(ge=18)
    gender: str
    subscription_type: Literal["Basic", "Standard", "Premium"]
    watch_hours: float = Field(ge=0)
    last_login_days: int = Field(ge=0)
    region: str
    device: str
    monthly_fee: float = Field(gt=0)
    payment_method: str
    number_of_profiles: int = Field(ge=1)
    favorite_genre: str
    # In the training data this equals watch_hours / (last_login_days + 1).
    avg_watch_time_per_day: float | None = Field(default=None, ge=0)


class Prediction(BaseModel):
    customer_id: str
    churn_probability: float
    churn_prediction: bool
    threshold: float


def _read_params(filename: str) -> dict:
    with (CONF_DIR / filename).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_model_path() -> Path:
    """The model named by ``CHURN_MODEL_PATH``, else the newest trained version."""
    if os.environ.get(MODEL_PATH_ENV):
        return Path(os.environ[MODEL_PATH_ENV])
    versions = sorted(
        (PROJECT_ROOT / "data" / "06_models" / "classifier.pkl").glob(
            "*/classifier.pkl"
        )
    )
    if not versions:
        raise RuntimeError(
            "No trained model found. Run `kedro run` first, or set "
            f"{MODEL_PATH_ENV} to a classifier.pkl file."
        )
    return versions[-1]


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = find_model_path()
    with path.open("rb") as f:
        app.state.model = pickle.load(f)  # noqa: S301  # our own trained artifact
    app.state.model_path = str(path)
    app.state.features_params = _read_params("parameters_feature_engineering.yml")[
        "feature_engineering"
    ]
    app.state.threshold = _read_params("parameters.yml")["decision"]["threshold"]
    yield


app = FastAPI(title="Netflix churn", version="0.1", lifespan=lifespan)


def _score(customers: list[Customer]) -> list[Prediction]:
    model = app.state.model
    raw = pd.DataFrame([c.model_dump() for c in customers])
    derived = raw["watch_hours"] / (raw["last_login_days"] + 1)
    raw["avg_watch_time_per_day"] = (
        raw["avg_watch_time_per_day"].astype(float).fillna(derived)
    )

    features = create_features(raw, app.state.features_params)
    try:
        probability = model.predict_proba(features[list(model.feature_names_in_)])[:, 1]
    except KeyError as error:
        raise HTTPException(status_code=500, detail=f"Model/feature mismatch: {error}")

    threshold = app.state.threshold
    return [
        Prediction(
            customer_id=customer.customer_id,
            churn_probability=round(float(p), 4),
            churn_prediction=bool(p >= threshold),
            threshold=threshold,
        )
        for customer, p in zip(customers, probability, strict=True)
    ]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_path": app.state.model_path}


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer) -> Prediction:
    return _score([customer])[0]


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(customers: list[Customer]) -> list[Prediction]:
    if not customers:
        raise HTTPException(status_code=422, detail="Send at least one customer.")
    return _score(customers)
