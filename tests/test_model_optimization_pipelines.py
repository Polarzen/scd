import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import src.nested_cv as nested_cv
from src.full_model import build_model, canonical_model_name, get_param_distributions


FEATURES = [f"x{i}" for i in range(10)]


def _training_data(n: int = 12) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(101)
    x = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
    y = np.asarray([0, 1] * (n // 2), dtype=int)
    return x, y


def test_elasticnet_pipeline_and_full_grid_are_fixed():
    model = build_model("elasticnet", FEATURES, seed=17, n_jobs=1)
    pre = model.named_steps["pre"]
    clf = model.named_steps["clf"]

    assert isinstance(pre, Pipeline)
    assert isinstance(pre.named_steps["imputer"], SimpleImputer)
    assert pre.named_steps["imputer"].get_params()["strategy"] == "median"
    assert pre.named_steps["imputer"].get_params()["keep_empty_features"] is True
    assert pre.named_steps["imputer"].get_params()["add_indicator"] is True
    assert isinstance(pre.named_steps["scale"], StandardScaler)
    assert isinstance(clf, LogisticRegression)
    assert clf.get_params()["solver"] == "saga"
    assert clf.get_params()["penalty"] == "elasticnet"
    assert clf.get_params()["max_iter"] >= 10000
    assert clf.get_params()["random_state"] == 17

    assert get_param_distributions("elasticnet") == {
        "clf__C": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3],
        "clf__l1_ratio": [0, 0.1, 0.25, 0.5, 0.75, 1],
        "clf__class_weight": [None, "balanced"],
    }


def test_selected_elasticnet_and_regularized_extratrees_contracts_are_fixed():
    selected = build_model("elasticnet_selected", FEATURES)
    select = selected.named_steps["pre"].named_steps["select"]
    assert isinstance(select, SelectKBest)
    assert select.score_func is f_classif
    assert get_param_distributions("elasticnet_selected")["pre__select__k"] == [8, 12, 20, 30]

    regularized = build_model("extratrees_regularized", FEATURES, n_jobs=1)
    assert isinstance(regularized.named_steps["pre"], SimpleImputer)
    clf = regularized.named_steps["clf"]
    assert isinstance(clf, ExtraTreesClassifier)
    assert get_param_distributions("extratrees_regularized") == {
        "clf__n_estimators": [300, 500, 800],
        "clf__max_depth": [3, 5, 8, 12, None],
        "clf__min_samples_leaf": [2, 4, 6, 10, 15],
        "clf__min_samples_split": [4, 8, 12, 20],
        "clf__max_features": ["sqrt", 0.2, 0.3, 0.5],
        "clf__class_weight": ["balanced", "balanced_subsample"],
        "clf__criterion": ["gini", "log_loss"],
        "clf__bootstrap": [False, True],
    }

    assert canonical_model_name("elasticnet-selected") == "elasticnet_selected"
    assert canonical_model_name("extra_trees_regularized") == "extratrees_regularized"


def test_search_strategy_and_scoring_match_model_family(monkeypatch):
    x, y = _training_data()
    seen: dict[str, dict[str, object]] = {}
    real_grid = nested_cv.GridSearchCV
    real_randomized = nested_cv.RandomizedSearchCV

    def grid(*args, **kwargs):
        seen["grid"] = kwargs
        return real_grid(*args, **kwargs)

    def randomized(*args, **kwargs):
        seen["randomized"] = kwargs
        return real_randomized(*args, **kwargs)

    monkeypatch.setattr(nested_cv, "GridSearchCV", grid)
    monkeypatch.setattr(nested_cv, "RandomizedSearchCV", randomized)

    nested_cv._fit_search(
        "elasticnet",
        x,
        y,
        FEATURES,
        inner_folds=2,
        n_iter=1,
        seed=3,
        n_jobs=1,
        tune_n_jobs=None,
        param_distributions={"clf__C": [0.1], "clf__l1_ratio": [0.5], "clf__class_weight": [None]},
        estimator_params=None,
    )
    assert "grid" in seen
    assert seen["grid"]["scoring"] == "average_precision"
    assert seen["grid"]["error_score"] == "raise"

    nested_cv._fit_search(
        "elasticnet_selected",
        x,
        y,
        FEATURES,
        inner_folds=2,
        n_iter=1,
        seed=3,
        n_jobs=1,
        tune_n_jobs=None,
        param_distributions={
            "clf__C": [0.1],
            "clf__l1_ratio": [0.5],
            "clf__class_weight": [None],
            "pre__select__k": [8],
        },
        estimator_params=None,
    )
    assert "randomized" in seen
    assert seen["randomized"]["scoring"] == "average_precision"
    assert seen["randomized"]["n_iter"] == 1
    assert seen["randomized"]["error_score"] == "raise"

    nested_cv._fit_search(
        "extratrees_regularized",
        x,
        y,
        FEATURES,
        inner_folds=2,
        n_iter=1,
        seed=3,
        n_jobs=1,
        tune_n_jobs=None,
        param_distributions={
            "clf__n_estimators": [5],
            "clf__max_depth": [3],
            "clf__min_samples_leaf": [2],
            "clf__min_samples_split": [4],
            "clf__max_features": ["sqrt"],
            "clf__class_weight": ["balanced"],
            "clf__criterion": ["gini"],
            "clf__bootstrap": [False],
        },
        estimator_params=None,
    )
    assert seen["randomized"]["scoring"] == "average_precision"
    assert seen["randomized"]["n_iter"] == 1
