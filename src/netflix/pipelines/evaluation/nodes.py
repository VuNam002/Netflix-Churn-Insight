"""Nodes of the 'evaluation' pipeline: judge the trained model on the test set.

The test set was never used to choose or fit the model, so these numbers estimate
how the model behaves on new customers.
"""

import logging

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from netflix.pipelines.training.nodes import ESTIMATORS, _make_pipeline

logger = logging.getLogger(__name__)

CLASS_NAMES = ["Stayed", "Churned"]


def _predict(
    classifier: Pipeline, X: pd.DataFrame, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    proba = classifier.predict_proba(X)[:, 1]
    return proba, (proba >= threshold).astype(int)


def _model_name(classifier: Pipeline) -> str:
    """Name of the fitted model, as written in ``parameters_training.yml``."""
    estimator = classifier.named_steps["model"]
    for name, estimator_class in ESTIMATORS.items():
        if isinstance(estimator, estimator_class):
            return name
    return type(estimator).__name__


def _scores(y_true: np.ndarray, proba: np.ndarray, pred: np.ndarray) -> dict:
    """The metrics shared by the headline report and the all-models table."""
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "average_precision": float(average_precision_score(y_true, proba)),
    }


def evaluate_model(
    classifier: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    decision: dict,
) -> dict[str, float | int | str]:
    """Headline metrics of the selected model at the decision threshold.

    ``baseline_accuracy`` is the accuracy of always predicting the majority class:
    the model is only useful if it clearly beats it.
    """
    y_true = y_test.iloc[:, 0].to_numpy()
    proba, pred = _predict(classifier, X_test, decision["threshold"])
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    metrics = {
        "model": _model_name(classifier),
        "threshold": float(decision["threshold"]),
        "n_test": int(len(y_true)),
        "baseline_accuracy": float(max(y_true.mean(), 1 - y_true.mean())),
        **_scores(y_true, proba, pred),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }
    logger.info("evaluate_model: %s", metrics)
    return metrics


def evaluate_all_models(  # noqa: PLR0913
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    model_comparison: pd.DataFrame,
    model_options: dict,
    training: dict,
    decision: dict,
) -> pd.DataFrame:
    """Score every candidate on the test set, for the report only.

    The model was ALREADY chosen by cross-validation on the train set
    (``model_comparison``); ``selected`` marks it. This table must not be used to
    pick a different model: choosing by test score would use the test set to
    decide, and the reported score would then be too optimistic. Rows keep the
    cross-validation order (``cv_rank``), not the test order, on purpose.
    """
    y_train_flat = y_train.iloc[:, 0]
    y_true = y_test.iloc[:, 0].to_numpy()
    selected = model_comparison.loc[model_comparison["cv_score_mean"].idxmax(), "model"]

    rows = []
    for rank, name in enumerate(model_comparison["model"], start=1):
        classifier = _make_pipeline(
            X_train,
            name,
            training["candidates"][name],
            model_options["random_state"],
        )
        classifier.fit(X_train, y_train_flat)
        proba, pred = _predict(classifier, X_test, decision["threshold"])
        rows.append(
            {
                "model": name,
                "cv_rank": rank,
                "selected": name == selected,
                **_scores(y_true, proba, pred),
            }
        )
    return pd.DataFrame(rows).round(4)


def evaluate_by_segment(
    classifier: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    parameters: dict,
    decision: dict,
) -> pd.DataFrame:
    """Metrics inside each level of ``segment_column``.

    A good overall score can hide a segment where the model is much worse.
    """
    segment = parameters["segment_column"]
    y_true = y_test.iloc[:, 0].to_numpy()
    proba, pred = _predict(classifier, X_test, decision["threshold"])

    rows = []
    for level in sorted(X_test[segment].unique()):
        mask = (X_test[segment] == level).to_numpy()
        y_seg = y_true[mask]
        has_both_classes = len(np.unique(y_seg)) == 2  # noqa: PLR2004
        rows.append(
            {
                "segment": level,
                "n": int(mask.sum()),
                "churn_rate": y_seg.mean(),
                "accuracy": accuracy_score(y_seg, pred[mask]),
                "recall": recall_score(y_seg, pred[mask], zero_division=0),
                "roc_auc": roc_auc_score(y_seg, proba[mask])
                if has_both_classes
                else np.nan,
            }
        )
    return pd.DataFrame(rows).round(4)


def compute_feature_importance(
    classifier: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    parameters: dict,
) -> pd.DataFrame:
    """Permutation importance: how much ROC-AUC drops when a column is shuffled.

    Works on the original columns, for any model type.
    """
    result = permutation_importance(
        classifier,
        X_test,
        y_test.iloc[:, 0],
        scoring="roc_auc",
        n_repeats=parameters["permutation_repeats"],
        random_state=parameters["random_state"],
    )
    importance = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    return importance.sort_values("importance_mean", ascending=False).round(4)


def plot_evaluation(
    classifier: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    feature_importance: pd.DataFrame,
    decision: dict,
) -> dict[str, Figure]:
    """Confusion matrix, ROC curve and feature importance, keyed by file name."""
    y_true = y_test.iloc[:, 0].to_numpy()
    proba, pred = _predict(classifier, X_test, decision["threshold"])

    matrix = confusion_matrix(y_true, pred, labels=[0, 1])
    fig_cm = Figure(figsize=(4.5, 4), layout="constrained")
    ax = fig_cm.subplots()
    ax.imshow(matrix, cmap="Blues")
    for (row, col), value in np.ndenumerate(matrix):
        ax.text(col, row, str(value), ha="center", va="center", fontsize=14)
    ax.set_xticks([0, 1], CLASS_NAMES)
    ax.set_yticks([0, 1], CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix (threshold {decision['threshold']})")

    fpr, tpr, _ = roc_curve(y_true, proba)
    fig_roc = Figure(figsize=(4.5, 4), layout="constrained")
    ax = fig_roc.subplots()
    ax.plot(fpr, tpr, label=f"model (AUC {roc_auc_score(y_true, proba):.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="random guess")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right")

    top = feature_importance.sort_values("importance_mean")
    fig_imp = Figure(figsize=(6, 0.45 * len(top) + 1.2), layout="constrained")
    ax = fig_imp.subplots()
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"])
    ax.set_xlabel("Drop in ROC-AUC when the column is shuffled")
    ax.set_title("Permutation importance")

    return {
        "confusion_matrix.png": fig_cm,
        "roc_curve.png": fig_roc,
        "feature_importance.png": fig_imp,
    }
