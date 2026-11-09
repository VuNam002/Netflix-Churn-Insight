"""Nodes of the 'training' pipeline: compare candidate models and train the best.

Candidates are compared by cross-validation on the TRAIN set only. The test set is
kept for the 'evaluation' pipeline, so the final score is not tuned on it.
"""

import logging

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

logger = logging.getLogger(__name__)

ESTIMATORS = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
    "svm": SVC,
    "lightgbm": LGBMClassifier,
}


def _make_pipeline(
    X: pd.DataFrame, name: str, hyperparameters: dict, random_state: int
) -> Pipeline:
    """One-hot encode text columns, scale numbers, then fit the named estimator."""
    if name not in ESTIMATORS:
        raise ValueError(f"Unknown model '{name}'. Choose from {sorted(ESTIMATORS)}.")

    categorical = X.select_dtypes(exclude="number").columns.tolist()
    numeric = X.select_dtypes(include="number").columns.tolist()
    preprocess = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", StandardScaler(), numeric),
        ]
    )
    estimator = ESTIMATORS[name](**hyperparameters, random_state=random_state)
    return Pipeline([("preprocess", preprocess), ("model", estimator)])


def compare_models(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    model_options: dict,
    training: dict,
) -> pd.DataFrame:
    """Cross-validate every candidate and rank them by mean validation score.

    ``overfit_gap`` is the train score minus the validation score; a large gap
    means the model memorises the training data.
    """
    y = y_train.iloc[:, 0]
    random_state = model_options["random_state"]
    folds = StratifiedKFold(
        n_splits=training["cv_folds"], shuffle=True, random_state=random_state
    )

    rows = []
    for name, hyperparameters in training["candidates"].items():
        estimator = _make_pipeline(X_train, name, hyperparameters, random_state)
        scores = cross_validate(
            estimator,
            X_train,
            y,
            cv=folds,
            scoring=training["scoring"],
            return_train_score=True,
        )
        rows.append(
            {
                "model": name,
                "cv_score_mean": scores["test_score"].mean(),
                "cv_score_std": scores["test_score"].std(),
                "train_score_mean": scores["train_score"].mean(),
                "overfit_gap": scores["train_score"].mean()
                - scores["test_score"].mean(),
                "fit_seconds": scores["fit_time"].mean(),
            }
        )
        logger.info("compare_models: %s -> %.4f", name, rows[-1]["cv_score_mean"])

    comparison = pd.DataFrame(rows).sort_values("cv_score_mean", ascending=False)
    return comparison.round(4).reset_index(drop=True)


def train_best_model(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    model_comparison: pd.DataFrame,
    model_options: dict,
    training: dict,
) -> Pipeline:
    """Refit the best candidate (highest mean validation score) on all train data."""
    best = model_comparison.loc[model_comparison["cv_score_mean"].idxmax(), "model"]
    logger.info("train_best_model: selected %s", best)

    classifier = _make_pipeline(
        X_train,
        best,
        training["candidates"][best],
        model_options["random_state"],
    )
    classifier.fit(X_train, y_train.iloc[:, 0])
    return classifier
