import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectKBest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import src.full_model as full_model
from src.nested_cv import run_nested_cv


FEATURES = [f"x{i}" for i in range(10)]


def _frame(n: int = 24) -> pd.DataFrame:
    rng = np.random.default_rng(303)
    frame = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
    frame.insert(0, "patient_id", [f"P{i:03d}" for i in range(n)])
    frame["label"] = np.asarray([0, 1] * (n // 2), dtype=int)
    frame.loc[0, FEATURES[0]] = np.nan
    frame.loc[1, FEATURES[1]] = np.nan
    return frame


def test_nested_cv_fits_preprocessing_only_on_training_subsets(monkeypatch):
    frame = _frame()
    fit_lengths: dict[str, list[int]] = {"imputer": [], "scaler": [], "selector": []}

    targets = ((SimpleImputer, "imputer"), (StandardScaler, "scaler"), (SelectKBest, "selector"))
    for estimator_cls, name in targets:
        original_fit = estimator_cls.fit

        def record_fit(self, x, y=None, *args, _original=original_fit, _name=name, **kwargs):
            fit_lengths[_name].append(len(x))
            return _original(self, x, y, *args, **kwargs)

        monkeypatch.setattr(estimator_cls, "fit", record_fit)

    result = run_nested_cv(
        frame,
        model="elasticnet_selected",
        feature_cols=FEATURES,
        outer_folds=3,
        inner_folds=2,
        n_iter=1,
        bootstrap_resamples=0,
        seed=7,
        param_distributions={
            "clf__C": [0.1],
            "clf__l1_ratio": [0.5],
            "clf__class_weight": [None],
            "pre__select__k": [8],
        },
    )

    assert len(result.oof) == len(frame)
    for name, lengths in fit_lengths.items():
        assert lengths, f"expected {name} to be fitted"
        assert max(lengths) < len(frame), f"{name} received the full patient table"
    pipeline = full_model.build_model("elasticnet_selected", FEATURES)
    assert not any(isinstance(step, CalibratedClassifierCV) for step in pipeline.named_steps.values())
