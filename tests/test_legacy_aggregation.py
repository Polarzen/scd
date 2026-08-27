import numpy as np
import pandas as pd
from pathlib import Path

from src.legacy_aggregation import (
    AGGREGATED_FEATURE_NAMES,
    aggregate_dynamic_features,
    aggregate_patient_features,
)
from src.legacy_features import FEATURE_NAMES


def _windows():
    rows = []
    for idx, status in enumerate(["SUCCESS", "SUCCESS", "OUTSIDE_RECORD"]):
        row = {name: np.nan for name in FEATURE_NAMES}
        row["sig_mean"] = float(idx + 1)
        row.update(
            {
                "patient_id": "P0001",
                "window_idx": idx,
                "window_status": status,
                "raw_rr_count": idx + 3,
                "valid_rr_count": idx + 3,
                "removed_rr_count": 0,
                "label": 1,
                "followup_days": 10.0,
                "cause_of_death": 3.0,
                "fs": 200.0,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def test_old_aggregation_uses_sample_sd_and_percentiles():
    result = aggregate_dynamic_features(_windows().iloc[:2])
    assert result["sig_mean_mean"] == 1.5
    assert result["sig_mean_std"] == np.sqrt(0.5)
    assert result["sig_mean_p10"] == 1.1
    assert result["sig_mean_p50"] == 1.5
    assert result["sig_mean_p90"] == 1.9
    assert result["n_windows_used"] == 2.0


def test_patient_aggregation_excludes_failed_rows_but_keeps_counts():
    result = aggregate_patient_features(_windows(), min_successful_windows=2)
    assert result.shape[0] == 1
    assert len(AGGREGATED_FEATURE_NAMES) == 100
    row = result.iloc[0]
    assert row["sig_mean_mean"] == 1.5
    assert row["n_windows_theoretical"] == 3
    assert row["n_windows_successful"] == 2
    assert row["n_windows_used"] == 2
    assert row["window_success_rate"] == 2 / 3
    assert row["raw_rr_count_total"] == 7
    assert row["removed_rr_count_total"] == 0


def test_generated_patient_features_are_88_by_100_model_features():
    path = Path("data/features/legacy_120s/patient_features.parquet")
    if not path.is_file():
        return
    table = pd.read_parquet(path)
    assert len(table) == 88
    assert table["patient_id"].is_unique
    assert len(AGGREGATED_FEATURE_NAMES) == 100
    assert all(name in table.columns for name in AGGREGATED_FEATURE_NAMES)
