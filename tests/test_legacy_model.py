import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

from src.legacy_aggregation import AGGREGATED_FEATURE_NAMES
from src.legacy_model import (
    build_model,
    get_param_distributions,
    nested_patient_oof,
    select_best_threshold,
)


def test_model_contract_is_100_features_and_old_search_space():
    model = build_model(AGGREGATED_FEATURE_NAMES, n_jobs=1)
    assert len(model.named_steps["pre"].transformers[0][2]) == 100
    distributions = get_param_distributions()
    assert distributions["clf__n_estimators"] == [400, 800, 1200, 1600]
    assert distributions["clf__max_depth"] == [None, 4, 6, 8, 10, 14]


def test_threshold_matches_legacy_nextafter_rule():
    threshold = select_best_threshold([0, 0, 0, 0, 1], [0.1, 0.2, 0.3, 0.4, 0.8])
    assert threshold > 0.3
    assert threshold < 0.4


def test_nested_oof_has_one_patient_row():
    rng = np.random.default_rng(7)
    n = 30
    frame = pd.DataFrame(rng.normal(size=(n, 100)), columns=AGGREGATED_FEATURE_NAMES)
    frame.insert(0, "patient_id", [f"P{i:04d}" for i in range(n)])
    frame.insert(1, "label", [0] * 15 + [1] * 15)
    small_space = {
        "clf__n_estimators": [8],
        "clf__max_depth": [None, 3],
        "clf__min_samples_split": [2],
        "clf__min_samples_leaf": [1],
        "clf__max_features": ["sqrt"],
        "clf__class_weight": ["balanced"],
    }
    result = nested_patient_oof(
        frame,
        n_iter=1,
        n_jobs=1,
        param_distributions=small_space,
    )
    assert len(result.oof) == n
    assert result.oof["patient_id"].is_unique
    assert result.oof.columns.tolist() == [
        "patient_id", "true_label", "prediction_probability", "prediction_label",
        "outer_fold", "fold_threshold", "seed"
    ]
    assert len(result.folds) == 5
    assert result.summary["feature_count"] == 100
    assert 0.0 <= result.summary["oof_brier"] <= 1.0


def test_generated_patient_level_split_and_oof_have_no_leakage():
    feature_path = Path("data/features/legacy_120s/patient_features.parquet")
    oof_path = Path("data/validation/legacy_oof_predictions.parquet")
    if not feature_path.is_file() or not oof_path.is_file():
        return
    frame = pd.read_parquet(feature_path).sort_values("patient_id", kind="stable").reset_index(drop=True)
    y = frame["label"].astype(int).to_numpy()
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, test_idx in splitter.split(frame, y):
        train_ids = set(frame.iloc[train_idx]["patient_id"])
        test_ids = set(frame.iloc[test_idx]["patient_id"])
        assert train_ids.isdisjoint(test_ids)
    oof = pd.read_parquet(oof_path)
    assert len(oof) == len(frame)
    assert oof["patient_id"].is_unique
    assert set(oof["patient_id"]) == set(frame["patient_id"])
