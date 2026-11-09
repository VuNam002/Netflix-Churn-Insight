import pytest
from matplotlib.figure import Figure

from netflix.pipelines.evaluation.nodes import (
    compute_feature_importance,
    evaluate_all_models,
    evaluate_by_segment,
    evaluate_model,
    plot_evaluation,
)
from netflix.pipelines.training.nodes import compare_models

PARAMS = {
    "segment_column": "subscription_type",
    "permutation_repeats": 3,
    "random_state": 0,
}


@pytest.fixture
def model_comparison(split, model_options, training_params):
    X_train, _, y_train, _ = split
    return compare_models(X_train, y_train, model_options, training_params)


def test_evaluate_model_reports_which_model_it_scored(split, classifier, decision):
    _, X_test, _, y_test = split

    metrics = evaluate_model(classifier, X_test, y_test, decision)

    assert metrics["model"] == "logistic_regression"


def test_evaluate_all_models_covers_every_candidate(
    split, model_comparison, model_options, training_params, decision
):
    X_train, X_test, y_train, y_test = split

    table = evaluate_all_models(
        X_train,
        y_train,
        X_test,
        y_test,
        model_comparison,
        model_options,
        training_params,
        decision,
    )

    assert set(table["model"]) == set(training_params["candidates"])
    assert {"accuracy", "precision", "recall", "f1", "roc_auc"} <= set(table.columns)
    assert table[["accuracy", "recall", "roc_auc"]].stack().between(0, 1).all()


def test_selected_model_is_the_cross_validation_winner_not_the_test_winner(
    split, model_comparison, model_options, training_params, decision
):
    X_train, X_test, y_train, y_test = split

    table = evaluate_all_models(
        X_train,
        y_train,
        X_test,
        y_test,
        model_comparison,
        model_options,
        training_params,
        decision,
    )

    assert table["selected"].sum() == 1
    assert (
        table.loc[table["selected"], "model"].item() == model_comparison.loc[0, "model"]
    )
    # Rows follow the cross-validation order, so the table does not invite
    # re-ranking by test score.
    assert list(table["model"]) == list(model_comparison["model"])
    assert list(table["cv_rank"]) == list(range(1, len(table) + 1))


def test_evaluate_model_metrics_are_consistent(split, classifier, decision):
    _, X_test, _, y_test = split

    metrics = evaluate_model(classifier, X_test, y_test, decision)

    cells = (
        metrics["true_negatives"]
        + metrics["false_positives"]
        + metrics["false_negatives"]
        + metrics["true_positives"]
    )
    assert cells == metrics["n_test"] == len(X_test)
    assert 0.5 <= metrics["baseline_accuracy"] <= 1
    assert metrics["roc_auc"] > 0.7  # the synthetic signal is strong
    assert metrics["accuracy"] > metrics["baseline_accuracy"]


def test_evaluate_by_segment_covers_every_level(split, classifier, decision):
    _, X_test, _, y_test = split

    result = evaluate_by_segment(classifier, X_test, y_test, PARAMS, decision)

    assert set(result["segment"]) == set(X_test["subscription_type"])
    assert result["n"].sum() == len(X_test)


def test_feature_importance_puts_the_real_signal_first(split, classifier):
    _, X_test, _, y_test = split

    importance = compute_feature_importance(classifier, X_test, y_test, PARAMS)

    assert set(importance["feature"]) == set(X_test.columns)
    assert importance["importance_mean"].is_monotonic_decreasing
    assert importance.iloc[0]["feature"] in {
        "last_login_days",
        "log_watch_hours",
        "log_avg_watch_time_per_day",
        "is_inactive",
    }


def test_plot_evaluation_returns_named_figures(split, classifier, decision):
    _, X_test, _, y_test = split
    importance = compute_feature_importance(classifier, X_test, y_test, PARAMS)

    figures = plot_evaluation(classifier, X_test, y_test, importance, decision)

    assert set(figures) == {
        "confusion_matrix.png",
        "roc_curve.png",
        "feature_importance.png",
    }
    assert all(isinstance(fig, Figure) for fig in figures.values())


def test_threshold_zero_predicts_everyone_will_churn(split, classifier):
    _, X_test, _, y_test = split

    metrics = evaluate_model(classifier, X_test, y_test, {"threshold": 0.0})

    assert metrics["true_negatives"] + metrics["false_negatives"] == 0
    assert metrics["recall"] == 1.0


def test_threshold_above_one_predicts_nobody_will_churn(split, classifier):
    _, X_test, _, y_test = split

    metrics = evaluate_model(classifier, X_test, y_test, {"threshold": 1.01})

    assert metrics["true_positives"] + metrics["false_positives"] == 0
    assert metrics["recall"] == 0.0
