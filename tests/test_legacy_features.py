import numpy as np
import pandas as pd
from pathlib import Path
import yaml

from src.legacy_features import FEATURE_NAMES, preprocess_ecg, window_features


def test_legacy_feature_order_and_count():
    rng = np.random.default_rng(42)
    signal = rng.normal(size=4_000)
    values = window_features(preprocess_ecg(signal, 100.0), 100.0)
    assert list(values) == list(FEATURE_NAMES)
    assert len(values) == 20


def test_preprocessing_is_float64_and_deterministic():
    signal = np.arange(2_000, dtype=np.int16)
    first = preprocess_ecg(signal, 100.0)
    second = preprocess_ecg(signal, 100.0)
    assert first.dtype == np.float64
    np.testing.assert_array_equal(first, second)


def test_generated_feature_table_has_exactly_20_float64_features():
    path = Path("data/features/legacy_120s/features.parquet")
    if not path.is_file():
        return
    table = pd.read_parquet(path)
    cols = [f"feature_{i:02d}" for i in range(1, 21)]
    assert all(column in table for column in cols)
    assert all(table[column].dtype == np.dtype("float64") for column in cols)
    assert table[["patient_id", "window_id"]].duplicated().sum() == 0


def test_frozen_feature_schema_has_20_allowed_categories():
    schema = yaml.safe_load(Path("config/legacy_features.yaml").read_text(encoding="utf-8"))
    assert schema["feature_count"] == 20
    assert len(schema["features"]) == 20
    assert [item["feature_id"] for item in schema["features"]] == [f"feature_{i:02d}" for i in range(1, 21)]
    allowed = {"TIME", "FREQUENCY", "NONLINEAR", "MORPHOLOGY", "SIGNAL_QUALITY", "HR_DYNAMICS"}
    assert {item["category"] for item in schema["features"]} <= allowed
