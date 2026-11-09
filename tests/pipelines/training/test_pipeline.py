import json
import math

import pytest

from netflix.pipelines.training.nodes import (
    ESTIMATORS,
    _make_pipeline,
    compare_models,
    get_trained_model,
    select_best_model,
    tune_model,
)


def test_compare_models_ranks_every_candidate(split, model_options, training_params):
    X_train, _, y_train, _ = split

    comparison = compare_models(X_train, y_train, model_options, training_params)

    assert set(comparison["model"]) == set(training_params["candidates"])
    assert comparison["cv_score_mean"].is_monotonic_decreasing
    assert comparison["cv_score_mean"].between(0, 1).all()
    assert {"overfit_gap", "cv_score_std", "fit_seconds"} <= set(comparison.columns)


def test_tune_model_tries_every_combination_of_the_winners_grid(
    tuned, model_comparison, training_params
):
    best, results = tuned

    winner = model_comparison.loc[0, "model"]
    grid = training_params["tuning"]["grids"][winner]
    assert best["model"] == winner
    assert len(results) == math.prod(len(values) for values in grid.values())
    assert results["cv_score_mean"].is_monotonic_decreasing
    assert set(grid) <= set(results.columns)


def test_tune_model_picks_the_best_scoring_combination(tuned, training_params):
    best, results = tuned

    grid = training_params["tuning"]["grids"][best["model"]]
    top = results.iloc[0]
    for name in grid:
        assert best["hyperparameters"][name] == top[name]


def test_tune_model_keeps_untuned_values_from_candidates(tuned, training_params):
    best, _ = tuned

    defaults = training_params["candidates"][best["model"]]
    grid = training_params["tuning"]["grids"][best["model"]]
    for name, value in defaults.items():
        if name not in grid:
            assert best["hyperparameters"][name] == value


def test_tune_model_is_skipped_when_disabled(
    split, model_comparison, model_options, training_params
):
    X_train, _, y_train, _ = split
    training_params["tuning"]["enabled"] = False

    best, results = tune_model(
        X_train, y_train, model_comparison, model_options, training_params
    )

    winner = model_comparison.loc[0, "model"]
    assert best == {
        "model": winner,
        "hyperparameters": training_params["candidates"][winner],
    }
    assert "skipped" in results.loc[0, "status"]


def test_tune_model_is_skipped_when_the_winner_has_no_grid(
    split, model_comparison, model_options, training_params
):
    X_train, _, y_train, _ = split
    winner = model_comparison.loc[0, "model"]
    del training_params["tuning"]["grids"][winner]

    best, results = tune_model(
        X_train, y_train, model_comparison, model_options, training_params
    )

    assert best["hyperparameters"] == training_params["candidates"][winner]
    assert "no grid" in results.loc[0, "status"]


def test_train_all_models_gives_every_candidate_its_own_folder(
    trained, training_params
):
    models, info = trained

    assert set(models) == {f"{name}/model" for name in training_params["candidates"]}
    assert set(info) == {f"{name}/info" for name in training_params["candidates"]}


def test_train_all_models_fits_each_model_with_the_right_hyperparameters(
    trained, tuned, training_params
):
    models, _ = trained
    best, _ = tuned

    for name, candidate in training_params["candidates"].items():
        expected = best["hyperparameters"] if name == best["model"] else candidate
        estimator = models[f"{name}/model"].named_steps["model"]
        assert isinstance(estimator, ESTIMATORS[name])
        for param, value in expected.items():
            assert estimator.get_params()[param] == value


def test_trained_models_can_predict(trained, split):
    models, _ = trained
    _, X_test, _, _ = split

    for model in models.values():
        proba = model.predict_proba(X_test)
        assert proba.shape == (len(X_test), 2)


def test_info_marks_only_the_selected_model_and_is_json_serialisable(
    trained, tuned, split
):
    _, info = trained
    X_train, *_ = split

    assert [i["model"] for i in info.values() if i["selected"]] == [tuned[0]["model"]]
    for item in info.values():
        json.dumps(item)  # raises if a value is not JSON-serialisable
        assert item["selected"] == item["hyperparameters_tuned"]
        assert item["features"] == X_train.columns.tolist()
        assert item["train_rows"] == len(X_train)


def test_select_best_model_returns_the_selected_model(trained, tuned):
    models, _ = trained

    classifier = select_best_model(models, tuned[0])

    assert classifier is models[f"{tuned[0]['model']}/model"]


def test_select_best_model_calls_loaders_like_a_partitioned_dataset(trained, tuned):
    models, _ = trained
    loaders = {
        key: (lambda m=model: m) for key, model in models.items()
    }  # lazy, as on disk

    classifier = select_best_model(loaders, tuned[0])

    assert classifier is get_trained_model(models, tuned[0]["model"])


def test_unknown_model_name_raises(split, model_options):
    X_train, *_ = split

    with pytest.raises(ValueError, match="Unknown model"):
        _make_pipeline(X_train, "magic_forest", {}, model_options["random_state"])


def test_classifier_handles_unseen_category(split, classifier):
    X_train, *_ = split
    X_new = X_train.head(3).assign(subscription_type="Platinum")

    assert len(classifier.predict(X_new)) == 3
