#!/usr/bin/env python3
"""Derive the legacy SCD timing cohort from the frozen Phase 2 table."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[1]


def derive_legacy_cohort(subjects: pd.DataFrame) -> pd.DataFrame:
    selected = subjects.loc[
        subjects["cause_of_death_raw"].eq("3")
        & subjects["event_source_valid"].fillna(False)
        & subjects["has_holter"].fillna(False),
        ["patient_id", "followup_days", "cause_of_death_raw", "holter_record_id"],
    ].copy()
    selected["label"] = selected["followup_days"].le(365).astype("Int64")
    selected["group"] = selected["label"].map({1: "LE_365D_SCD", 0: "GT_365D_SCD"}).astype("string")
    return selected.sort_values("patient_id", kind="stable").reset_index(drop=True)


def compare_with_old(derived: pd.DataFrame, old: pd.DataFrame) -> pd.DataFrame:
    old_state = old[["patient_id", "scd_high_risk"]].copy()
    old_state["group_old"] = pd.to_numeric(old_state["scd_high_risk"], errors="coerce").map({1: "LE_365D_SCD", 0: "GT_365D_SCD"})
    new_state = derived[["patient_id", "group"]].rename(columns={"group": "group_new"})
    merged = new_state.merge(old_state[["patient_id", "group_old"]], on="patient_id", how="outer", indicator=True)
    merged["derived_from_phase2"] = merged["_merge"].ne("right_only")
    merged["present_in_old_scd_dataset"] = merged["_merge"].ne("left_only")
    merged["agreement"] = merged["_merge"].eq("both") & merged["group_new"].eq(merged["group_old"])
    merged["reason"] = "MATCH"
    merged.loc[merged["_merge"].eq("left_only"), "reason"] = "NEW_ONLY"
    merged.loc[merged["_merge"].eq("right_only"), "reason"] = "OLD_ONLY"
    merged.loc[merged["_merge"].eq("both") & ~merged["agreement"], "reason"] = "GROUP_MISMATCH"
    return merged[["patient_id", "derived_from_phase2", "present_in_old_scd_dataset", "agreement", "group_new", "group_old", "reason"]].sort_values("patient_id").reset_index(drop=True)


def main() -> int:
    subjects = pd.read_parquet(REPO / "data/cohort/subjects.parquet")
    old = pd.read_csv(REPO / "scd_dataset.csv", dtype={"patient_id": "string"})
    derived = derive_legacy_cohort(subjects)
    comparison = compare_with_old(derived, old)
    output = REPO / "reports/LEGACY_COHORT_COMPARISON.csv"
    comparison.to_csv(output, index=False, encoding="utf-8")
    derived_ids, old_ids = set(derived["patient_id"]), set(old["patient_id"])
    summary = {
        "derived_legacy_patient_count": len(derived),
        "old_dataset_patient_count": len(old),
        "intersection": len(derived_ids & old_ids),
        "new_only": sorted(derived_ids - old_ids),
        "old_only": sorted(old_ids - derived_ids),
        "derived_le_365d": int(derived["label"].sum()),
        "derived_gt_365d": int((derived["label"] == 0).sum()),
        "group_mismatches": int((comparison["reason"] == "GROUP_MISMATCH").sum()),
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if not summary["new_only"] and not summary["old_only"] and summary["group_mismatches"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
