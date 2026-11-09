"""Nodes of the 'training' pipeline: compare, tune, train every model, pick the best.

Candidates are compared, and the winner's hyperparameters are grid-searched, by
cross-validation on the TRAIN set only. The test set is kept for the 'evaluation'
pipeline, so the final score is not tuned on it.
"""

import logging
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

logger = logging.getLogger(__name__)

MODEL_FILE = "model"  # <model>/model.pkl
INFO_FILE = "info"  # <model>/info.json

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


def _best_candidate(model_comparison: pd.DataFrame) -> str:
    """Name of the candidate with the highest mean validation score."""
    return model_comparison.loc[model_comparison["cv_score_mean"].idxmax(), "model"]


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


def tune_model(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    model_comparison: pd.DataFrame,
    model_options: dict,
    training: dict,
) -> tuple[dict, pd.DataFrame]:
    """Grid-search the hyperparameters of the model that won ``compare_models``.

    Every combination in ``training.tuning.grids[<model>]`` is cross-validated on
    the TRAIN set with the same folds and score as ``compare_models``; the test set
    is not touched. Only the winning model is tuned, to keep the run short.
    Tuning is skipped (the candidate's own values are kept) when it is disabled or
    the winner has no grid.

    Returns:
        ``{"model": name, "hyperparameters": {...}}`` for ``train_all_models``, and a
        table with one row per combination, best first.
    """
    best = _best_candidate(model_comparison)
    defaults = training["candidates"][best]
    tuning = training.get("tuning", {})
    grid = tuning.get("grids", {}).get(best)

    if not tuning.get("enabled", False) or not grid:
        reason = (
            "tuning disabled"
            if not tuning.get("enabled", False)
            else f"no grid for {best}"
        )
        logger.info("tune_model: skipped (%s)", reason)
        skipped = pd.DataFrame([{"status": f"skipped: {reason}"}])
        return {"model": best, "hyperparameters": defaults}, skipped

    random_state = model_options["random_state"]
    folds = StratifiedKFold(
        n_splits=training["cv_folds"], shuffle=True, random_state=random_state
    )
    search = GridSearchCV(
        _make_pipeline(X_train, best, defaults, random_state),
        param_grid={f"model__{name}": values for name, values in grid.items()},
        scoring=training["scoring"],
        cv=folds,
        refit=False,
        return_train_score=True,
    )
    search.fit(X_train, y_train.iloc[:, 0])

    cv = search.cv_results_
    combinations = [
        {name.removeprefix("model__"): value for name, value in params.items()}
        for params in cv["params"]
    ]
    table = pd.DataFrame(combinations).assign(
        cv_score_mean=cv["mean_test_score"],
        cv_score_std=cv["std_test_score"],
        train_score_mean=cv["mean_train_score"],
        overfit_gap=cv["mean_train_score"] - cv["mean_test_score"],
        fit_seconds=cv["mean_fit_time"],
    )
    table = table.sort_values("cv_score_mean", ascending=False).round(4)

    best_combination = combinations[int(np.argmax(cv["mean_test_score"]))]
    default_score = model_comparison.loc[
        model_comparison["model"] == best, "cv_score_mean"
    ].item()
    logger.info(
        "tune_model: %s, %d combinations, cv score %.4f -> %.4f with %s",
        best,
        len(combinations),
        default_score,
        table["cv_score_mean"].iloc[0],
        best_combination,
    )
    chosen = {"model": best, "hyperparameters": {**defaults, **best_combination}}
    return chosen, table.reset_index(drop=True)


def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    best_hyperparameters: dict,
    model_options: dict,
    training: dict,
) -> tuple[dict[str, Pipeline], dict[str, dict]]:
    """Fit every candidate on all train data, one folder per model.

    The model chosen by ``compare_models`` uses its tuned hyperparameters; the
    others use the values in ``candidates``. Keys are ``<model>/model`` and
    ``<model>/info``, so saving them with a ``PartitionedDataset`` gives

        data/06_models/all_models/<model>/model.pkl
        data/06_models/all_models/<model>/info.json

    Returns:
        The fitted pipelines, and for each a small JSON description (which
        hyperparameters, whether it is the selected model, when and on what data).
    """
    selected = best_hyperparameters["model"]
    y = y_train.iloc[:, 0]
    trained_at = datetime.now(UTC).isoformat(timespec="seconds")

    models, info = {}, {}
    for name, candidate in training["candidates"].items():
        is_selected = name == selected
        hyperparameters = (
            best_hyperparameters["hyperparameters"] if is_selected else candidate
        )
        models[f"{name}/{MODEL_FILE}"] = _make_pipeline(
            X_train, name, hyperparameters, model_options["random_state"]
        ).fit(X_train, y)
        info[f"{name}/{INFO_FILE}"] = {
            "model": name,
            "selected": is_selected,
            "hyperparameters_tuned": is_selected,
            "hyperparameters": hyperparameters,
            "train_rows": len(X_train),
            "features": X_train.columns.tolist(),
            "trained_at": trained_at,
        }
        logger.info("train_all_models: fitted %s (selected=%s)", name, is_selected)
    return models, info


def get_trained_model(trained_models: dict, name: str) -> Pipeline:
    """One fitted model from ``trained_models``.

    Loaded from a ``PartitionedDataset`` the values are functions that read the
    file when called; in memory they are the models themselves.
    """
    model = trained_models[f"{name}/{MODEL_FILE}"]
    return model() if callable(model) else model


def select_best_model(trained_models: dict, best_hyperparameters: dict) -> Pipeline:
    """The model chosen by ``compare_models``, ready to be saved as ``classifier``."""
    name = best_hyperparameters["model"]
    logger.info("select_best_model: %s", name)
    return get_trained_model(trained_models, name)
