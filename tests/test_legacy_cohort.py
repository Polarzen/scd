import pandas as pd

from scripts.build_legacy_cohort import compare_with_old, derive_legacy_cohort


def test_legacy_patient_selection_is_source_derived_and_deterministic():
    subjects = pd.read_parquet("data/cohort/subjects.parquet")
    first = derive_legacy_cohort(subjects)
    second = derive_legacy_cohort(subjects.sample(frac=1, random_state=99))
    pd.testing.assert_frame_equal(first, second)
    assert first["patient_id"].is_unique
    assert first["cause_of_death_raw"].eq("3").all()
    assert first["holter_record_id"].notna().all()
    assert first["label"].eq(first["followup_days"].le(365).astype("Int64")).all()


def test_derived_cohort_matches_old_dataset_without_using_it_for_selection():
    subjects = pd.read_parquet("data/cohort/subjects.parquet")
    old = pd.read_csv("scd_dataset.csv", dtype={"patient_id": "string"})
    comparison = compare_with_old(derive_legacy_cohort(subjects), old)
    assert comparison["agreement"].all()
    assert set(comparison["reason"]) == {"MATCH"}
