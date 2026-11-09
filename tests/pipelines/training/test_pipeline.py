import pytest

from netflix.pipelines.training.nodes import (
    ESTIMATORS,
    _make_pipeline,
    compare_models,
    train_best_model,
)


def test_compare_models_ranks_every_candidate(split, model_options, training_params):
    X_train, _, y_train, _ = split

    comparison = compare_models(X_train, y_train, model_options, training_params)

    assert set(comparison["model"]) == set(training_params["candidates"])
    assert comparison["cv_score_mean"].is_monotonic_decreasing
    assert comparison["cv_score_mean"].between(0, 1).all()
    assert {"overfit_gap", "cv_score_std", "fit_seconds"} <= set(comparison.columns)


def test_train_best_model_uses_the_top_ranked_candidate(
    split, model_options, training_params
):
    X_train, X_test, y_train, _ = split
    comparison = compare_models(X_train, y_train, model_options, training_params)

    classifier = train_best_model(
        X_train, y_train, comparison, model_options, training_params
    )

    best = comparison.loc[0, "model"]
    assert isinstance(classifier.named_steps["model"], ESTIMATORS[best])
    assert len(classifier.predict(X_test)) == len(X_test)


def test_unknown_model_name_raises(split, model_options):
    X_train, *_ = split

    with pytest.raises(ValueError, match="Unknown model"):
        _make_pipeline(X_train, "magic_forest", {}, model_options["random_state"])


def test_classifier_handles_unseen_category(split, classifier):
    X_train, *_ = split
    X_new = X_train.head(3).assign(subscription_type="Platinum")

    assert len(classifier.predict(X_new)) == 3
