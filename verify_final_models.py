from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from run_final_models import FIXED_COMBO, EXPECTED_FIXED_MEDIAN, EXPECTED_SELECTOR_MEDIAN, fingerprint_snapshot


SEEDS = set(range(100))
PATIENTS = 878
POSITIVES = 37
NEGATIVES = 841
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260927
METRIC_TOLERANCE = 1e-10


def _need_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def _normalize_oof(
    frame: pd.DataFrame,
    *,
    model: str,
    label: str,
    expected_labels: dict[str, int],
) -> pd.DataFrame:
    _need_columns(frame, {"patient_id", "true_label", "probability", "seed"}, label)
    out = frame.copy()
    out["patient_id"] = out["patient_id"].astype("string")
    if out["patient_id"].isna().any() or out["patient_id"].str.strip().eq("").any():
        raise ValueError(f"{label} contains missing or blank patient IDs")
    out["seed"] = pd.to_numeric(out["seed"], errors="raise").astype(int)
    labels = pd.to_numeric(out["true_label"], errors="raise")
    if not labels.isin([0, 1]).all():
        raise ValueError(f"{label} labels are not binary")
    out["true_label"] = labels.astype(int)
    out["probability"] = pd.to_numeric(out["probability"], errors="raise").astype(float)
    if not np.isfinite(out["probability"].to_numpy()).all():
        raise ValueError(f"{label} has non-finite probabilities")
    if ((out["probability"] < 0.0) | (out["probability"] > 1.0)).any():
        raise ValueError(f"{label} has probabilities outside [0, 1]")
    if "model" in out and set(out["model"].astype(str)) != {model}:
        raise ValueError(f"{label} model column does not contain exactly {model}")
    if out.duplicated(["seed", "patient_id"]).any():
        raise ValueError(f"{label} has duplicate (seed, patient_id) rows")
    if set(out["seed"]) != SEEDS:
        raise ValueError(f"{label} seed coverage is not exactly 0..99")
    if len(out) != len(SEEDS) * PATIENTS:
        raise ValueError(f"{label} row count is {len(out)}, expected {len(SEEDS) * PATIENTS}")

    reference: dict[str, int] | None = None
    for seed, rows in out.groupby("seed", sort=True):
        if len(rows) != PATIENTS or rows["patient_id"].nunique() != PATIENTS:
            raise ValueError(f"{label} seed {seed} does not have exactly {PATIENTS} unique patients")
        if int(rows["true_label"].sum()) != POSITIVES or int((rows["true_label"] == 0).sum()) != NEGATIVES:
            raise ValueError(f"{label} seed {seed} has unexpected label counts")
        labels_by_patient = dict(zip(rows["patient_id"].astype(str), rows["true_label"].astype(int)))
        if reference is None:
            reference = labels_by_patient
        elif labels_by_patient != reference:
            raise ValueError(f"{label} patient set or labels differ at seed {seed}")
        if set(labels_by_patient) != set(expected_labels):
            missing = sorted(set(expected_labels) - set(labels_by_patient))[:5]
            extra = sorted(set(labels_by_patient) - set(expected_labels))[:5]
            raise ValueError(f"{label} seed {seed} patient IDs differ from input anchor; missing={missing}, extra={extra}")
        mismatched = [patient for patient, expected in expected_labels.items() if labels_by_patient.get(patient) != expected]
        if mismatched:
            raise ValueError(f"{label} seed {seed} labels differ from input anchor for patients {mismatched[:5]}")
    return out


def _anchor_cohort(project_root: Path) -> tuple[dict[str, int], dict[str, Any]]:
    """Build only the input frames and label maps; no validation/model fitting is called."""
    project_text = str(project_root)
    inserted = project_text not in sys.path
    if inserted:
        sys.path.insert(0, project_text)
    try:
        from analysis.validate_nested_selector import build_selector_frame
        from analysis.validate_p5_combos import build_combo

        selector_frame, _recipes, _contract = build_selector_frame()
        fixed_frame, _features, fixed_meta = build_combo(FIXED_COMBO)
    finally:
        if inserted:
            try:
                sys.path.remove(project_text)
            except ValueError:
                pass

    def label_map(frame: pd.DataFrame, name: str) -> dict[str, int]:
        _need_columns(frame, {"patient_id", "label"}, f"{name} input frame")
        ids = frame["patient_id"].astype("string")
        labels = pd.to_numeric(frame["label"], errors="raise")
        if ids.isna().any() or ids.duplicated().any() or not labels.isin([0, 1]).all():
            raise ValueError(f"{name} input anchor has duplicate/missing IDs or non-binary labels")
        return dict(zip(ids.astype(str), labels.astype(int)))

    selector_labels = label_map(selector_frame, "selector")
    fixed_labels = label_map(fixed_frame, "fixed P5")
    if selector_labels != fixed_labels:
        missing = sorted(set(selector_labels) - set(fixed_labels))[:5]
        extra = sorted(set(fixed_labels) - set(selector_labels))[:5]
        different = sorted(patient for patient in set(selector_labels) & set(fixed_labels) if selector_labels[patient] != fixed_labels[patient])[:5]
        raise ValueError(
            "input builders disagree on patient/label anchor; "
            f"selector_n={len(selector_labels)}, fixed_n={len(fixed_labels)}, missing={missing}, extra={extra}, label_mismatch={different}"
        )
    positives = int(sum(selector_labels.values()))
    negatives = len(selector_labels) - positives
    if (len(selector_labels), positives, negatives) != (PATIENTS, POSITIVES, NEGATIVES):
        raise ValueError(
            "input-builder cohort drift: "
            f"patients/positive/negative={(len(selector_labels), positives, negatives)}, "
            f"expected={(PATIENTS, POSITIVES, NEGATIVES)}"
        )
    anchor = {
        "patients": int(len(selector_labels)),
        "positive": positives,
        "negative": negatives,
        "selector_builder": "analysis.validate_nested_selector.build_selector_frame",
        "fixed_builder": "analysis.validate_p5_combos.build_combo(pro_bnp_nsvt)",
        "fixed_combo_raw_feature_count": int(fixed_meta.get("raw_feature_count", -1)),
    }
    return selector_labels, anchor


def _compare_artifacts(
    batch_frame: pd.DataFrame,
    combined_frame: pd.DataFrame,
    *,
    key_columns: tuple[str, ...],
    label: str,
    errors: list[str],
    tolerance: float = 1e-12,
) -> None:
    """Compare a concatenated batch artifact with its aggregate/merged copy."""
    if list(batch_frame.columns) != list(combined_frame.columns):
        errors.append(f"{label} columns/order differ between raw batches and combined artifact")
        return
    _need_columns(batch_frame, set(key_columns), f"{label} raw batch artifact")
    _need_columns(combined_frame, set(key_columns), f"{label} combined artifact")
    left = batch_frame.copy()
    right = combined_frame.copy()
    for column in key_columns:
        if column in {"seed", "outer_fold"}:
            left[column] = pd.to_numeric(left[column], errors="raise").astype(int)
            right[column] = pd.to_numeric(right[column], errors="raise").astype(int)
        else:
            left[column] = left[column].astype("string")
            right[column] = right[column].astype("string")
    if left.duplicated(list(key_columns)).any() or right.duplicated(list(key_columns)).any():
        errors.append(f"{label} contains duplicate comparison keys")
        return
    left = left.sort_values(list(key_columns), kind="stable").reset_index(drop=True)
    right = right.sort_values(list(key_columns), kind="stable").reset_index(drop=True)
    if len(left) != len(right):
        errors.append(f"{label} row count differs: batches={len(left)}, combined={len(right)}")
        return
    for column in key_columns:
        if not left[column].astype("string").fillna("<NULL>").equals(right[column].astype("string").fillna("<NULL>")):
            errors.append(f"{label} key/order differs in {column}")
            return
    for column in batch_frame.columns:
        if column in key_columns:
            continue
        left_values = left[column]
        right_values = right[column]
        numeric_left = pd.to_numeric(left_values, errors="coerce")
        numeric_right = pd.to_numeric(right_values, errors="coerce")
        is_numeric = (
            pd.api.types.is_numeric_dtype(left_values.dtype)
            or pd.api.types.is_numeric_dtype(right_values.dtype)
        )
        if is_numeric:
            a = numeric_left.to_numpy(dtype=float, na_value=np.nan)
            b = numeric_right.to_numpy(dtype=float, na_value=np.nan)
            if not np.allclose(a, b, rtol=0.0, atol=tolerance, equal_nan=True):
                errors.append(f"{label} differs in numeric column {column}")
                return
        else:
            a = left_values.astype("string").fillna("<NULL>")
            b = right_values.astype("string").fillna("<NULL>")
            if not a.equals(b):
                errors.append(f"{label} differs in column {column}")
                return


def _compare_patient_mean_csv(
    reported: pd.DataFrame,
    independent: pd.DataFrame,
    *,
    expected_labels: dict[str, int],
    label: str,
    errors: list[str],
) -> None:
    _need_columns(reported, {"patient_id", "true_label", "probability"}, label)
    actual = reported.loc[:, ["patient_id", "true_label", "probability"]].copy()
    actual["patient_id"] = actual["patient_id"].astype("string")
    actual["true_label"] = pd.to_numeric(actual["true_label"], errors="raise").astype(int)
    actual["probability"] = pd.to_numeric(actual["probability"], errors="raise").astype(float)
    expected = independent.loc[:, ["patient_id", "true_label", "probability"]].copy()
    expected["patient_id"] = expected["patient_id"].astype("string")
    if actual["patient_id"].isna().any() or actual["patient_id"].duplicated().any():
        errors.append(f"{label} has missing or duplicate patient IDs")
        return
    if len(actual) != len(expected) or set(actual["patient_id"].astype(str)) != set(expected["patient_id"].astype(str)):
        errors.append(f"{label} patient keys differ from independently computed patient means")
        return
    actual_labels = dict(zip(actual["patient_id"].astype(str), actual["true_label"].astype(int)))
    if actual_labels != expected_labels:
        errors.append(f"{label} patient labels differ from the input-builder anchor")
        return
    left = actual.sort_values("patient_id", kind="stable").reset_index(drop=True)
    right = expected.sort_values("patient_id", kind="stable").reset_index(drop=True)
    if not np.allclose(left["probability"].to_numpy(), right["probability"].to_numpy(), rtol=0.0, atol=1e-12):
        errors.append(f"{label} probabilities differ from independently averaged seed OOF predictions")


def _metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    return {
        "AUC": float(roc_auc_score(y, probability)),
        "AP": float(average_precision_score(y, probability)),
        "Brier": float(brier_score_loss(y, probability)),
    }


def _distribution_median(values: pd.Series) -> float:
    array = pd.to_numeric(values, errors="raise").to_numpy(dtype=float)
    if array.size != 100 or not np.isfinite(array).all():
        raise ValueError("expected 100 finite per-seed metric values")
    return float(np.median(array))


def _compare(actual: float, expected: float, *, label: str, errors: list[str], tolerance: float = METRIC_TOLERANCE) -> None:
    if not np.isfinite(actual) or not np.isfinite(expected) or abs(actual - expected) > tolerance:
        errors.append(f"{label}: independent={actual:.12g}, provided/expected={expected:.12g}")


def _bootstrap_patient_auc(y: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values: list[float] = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        index = rng.integers(0, len(y), size=len(y))
        sampled_y = y[index]
        if np.unique(sampled_y).size < 2:
            continue
        values.append(float(roc_auc_score(sampled_y, probability[index])))
    array = np.asarray(values, dtype=float)
    return {
        "lower_95": float(np.percentile(array, 2.5)),
        "upper_95": float(np.percentile(array, 97.5)),
        "median": float(np.median(array)),
        "valid_resamples": int(array.size),
        "requested_resamples": BOOTSTRAP_RESAMPLES,
        "seed": BOOTSTRAP_SEED,
        "interpretation": "conditional interval for the already-computed patient-mean OOF predictions; not a confidence interval for the 100-seed median",
    }


def _validate_folds(
    folds: pd.DataFrame,
    *,
    label: str,
    oof: pd.DataFrame,
    errors: list[str],
) -> dict[str, Any]:
    _need_columns(folds, {"seed", "outer_fold", "train_n", "test_n", "train_positive", "test_positive"}, label)
    work = folds.copy()
    for column in ("seed", "outer_fold", "train_n", "test_n", "train_positive", "test_positive"):
        work[column] = pd.to_numeric(work[column], errors="raise").astype(int)
    if len(work) != 500 or work.duplicated(["seed", "outer_fold"]).any():
        errors.append(f"{label} must have exactly 500 unique seed/fold summaries")
    if set(work["seed"]) != SEEDS:
        errors.append(f"{label} seed coverage is not exactly 0..99")
    fold_sizes: list[int] = []
    for seed, rows in work.groupby("seed", sort=True):
        if set(rows["outer_fold"]) != {1, 2, 3, 4, 5}:
            errors.append(f"{label} seed {seed} does not contain outer folds 1..5")
            continue
        if not (rows["train_n"] + rows["test_n"]).eq(PATIENTS).all():
            errors.append(f"{label} seed {seed} has train/test counts that do not sum to {PATIENTS}")
        if not (rows["train_positive"] + rows["test_positive"]).eq(POSITIVES).all():
            errors.append(f"{label} seed {seed} has train/test positives that do not sum to {POSITIVES}")
        if int(rows["test_n"].sum()) != PATIENTS or int(rows["test_positive"].sum()) != POSITIVES:
            errors.append(f"{label} seed {seed} outer tests do not cover the cohort and positives once")
        if "outer_fold" in oof.columns:
            seed_oof = oof.loc[oof["seed"].eq(seed)]
            if seed_oof["outer_fold"].isna().any():
                errors.append(f"{label} OOF seed {seed} has missing outer-fold IDs")
            else:
                oof_counts = seed_oof.groupby("outer_fold").agg(
                    test_n=("patient_id", "size"),
                    test_positive=("true_label", "sum"),
                )
                fold_index = rows.set_index("outer_fold")
                if set(oof_counts.index.astype(int)) != set(rows["outer_fold"]):
                    errors.append(f"{label} OOF fold IDs disagree with fold summaries for seed {seed}")
                else:
                    for fold_id, values in oof_counts.iterrows():
                        if int(values["test_n"]) != int(fold_index.loc[int(fold_id), "test_n"]):
                            errors.append(f"{label} seed {seed} fold {fold_id} OOF count disagrees with summary")
                        if int(values["test_positive"]) != int(fold_index.loc[int(fold_id), "test_positive"]):
                            errors.append(f"{label} seed {seed} fold {fold_id} positive count disagrees with summary")
        fold_sizes.extend(rows["test_n"].astype(int).tolist())
    return {
        "fold_rows": int(len(work)),
        "folds_per_seed": 5,
        "test_patient_count_min": min(fold_sizes) if fold_sizes else None,
        "test_patient_count_max": max(fold_sizes) if fold_sizes else None,
        "cohort_count_conservation_checked": True,
    }


def _independent_run(
    oof_raw: pd.DataFrame,
    per_seed: pd.DataFrame,
    folds: pd.DataFrame,
    *,
    model: str,
    label: str,
    expected_labels: dict[str, int],
    errors: list[str],
) -> tuple[dict[str, Any], pd.DataFrame]:
    oof = _normalize_oof(oof_raw, model=model, label=f"{label} OOF", expected_labels=expected_labels)
    _need_columns(per_seed, {"seed", "AUC", "AP", "Brier"}, f"{label} per_seed")
    per = per_seed.copy()
    per["seed"] = pd.to_numeric(per["seed"], errors="raise").astype(int)
    if len(per) != 100 or per["seed"].duplicated().any() or set(per["seed"]) != SEEDS:
        errors.append(f"{label} per_seed must contain exactly one row for each seed 0..99")
    if "model" in per and set(per["model"].astype(str)) != {model}:
        errors.append(f"{label} per_seed model column does not contain exactly {model}")

    calculated: list[dict[str, float | int]] = []
    for seed, rows in oof.groupby("seed", sort=True):
        metrics = _metrics(rows["true_label"].to_numpy(dtype=int), rows["probability"].to_numpy(dtype=float))
        calculated.append({"seed": int(seed), **metrics})
    independent_per_seed = pd.DataFrame(calculated).sort_values("seed").reset_index(drop=True)
    per_index = per.set_index("seed")
    for row in independent_per_seed.to_dict("records"):
        seed = int(row["seed"])
        if seed not in per_index.index:
            continue
        for metric in ("AUC", "AP", "Brier"):
            _compare(float(row[metric]), float(per_index.loc[seed, metric]), label=f"{label} seed {seed} {metric}", errors=errors)

    patient_labels = oof.groupby("patient_id")["true_label"].nunique()
    if (patient_labels != 1).any():
        errors.append(f"{label} has patient labels that vary across seeds")
    patient = (
        oof.groupby(["patient_id", "true_label"], as_index=False, sort=True)["probability"]
        .mean()
        .sort_values("patient_id", kind="stable")
        .reset_index(drop=True)
    )
    if len(patient) != PATIENTS or patient["patient_id"].duplicated().any():
        errors.append(f"{label} patient-mean OOF does not contain {PATIENTS} unique patients")
    if int(patient["true_label"].sum()) != POSITIVES:
        errors.append(f"{label} patient-mean OOF does not contain {POSITIVES} positives")
    patient_y = patient["true_label"].to_numpy(dtype=int)
    patient_p = patient["probability"].to_numpy(dtype=float)
    patient_metrics = _metrics(patient_y, patient_p)
    prevalence = float(patient_y.mean())
    dummy_brier = float(prevalence * (1.0 - prevalence))
    auc_median = _distribution_median(independent_per_seed["AUC"])
    fold_report = _validate_folds(folds, label=f"{label} folds", oof=oof, errors=errors)
    return (
        {
            "model": model,
            "seed_count": int(independent_per_seed["seed"].nunique()),
            "patient_count": int(len(patient)),
            "positive_count": int(patient_y.sum()),
            "negative_count": int(len(patient_y) - patient_y.sum()),
            "cv_auc_median_independent": auc_median,
            "cv_auc_median_round_6dp": f"{auc_median:.6f}",
            "patient_mean_oof": patient_metrics,
            "prevalence": prevalence,
            "patient_ap_over_prevalence": float(patient_metrics["AP"] / prevalence) if prevalence else None,
            "dummy_brier_q_times_one_minus_q": dummy_brier,
            "patient_brier_vs_dummy": float(patient_metrics["Brier"] - dummy_brier),
            "patient_auc_bootstrap": _bootstrap_patient_auc(patient_y, patient_p),
            "fold_summary": fold_report,
            "per_seed_auc_min": float(independent_per_seed["AUC"].min()),
            "per_seed_auc_max": float(independent_per_seed["AUC"].max()),
        },
        patient,
    )


def _selector_inputs(results_root: Path) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
    pd.DataFrame,
]:
    batch_root = results_root / "selector" / "batches"
    batch_dirs = sorted(path for path in batch_root.glob("batch_*") if path.is_dir())
    if len(batch_dirs) != 10:
        raise ValueError(f"selector must have exactly ten batch directories; found {len(batch_dirs)}")
    per_frames: list[pd.DataFrame] = []
    oof_frames: list[pd.DataFrame] = []
    fold_frames: list[pd.DataFrame] = []
    contract_paths: list[Path] = []
    for batch_index, folder in enumerate(batch_dirs):
        seed_start = batch_index * 10
        expected_seeds = set(range(seed_start, seed_start + 10))
        expected_name = f"batch_{seed_start:02d}_{seed_start + 9:02d}"
        if folder.name != expected_name:
            raise ValueError(f"selector batch directory {folder.name!r} should be {expected_name!r}")
        required = ("per_seed.csv", "oof.csv", "folds.csv", "selector_contract.json")
        missing = [name for name in required if not (folder / name).is_file()]
        if missing:
            raise ValueError(f"selector batch {folder.name} is missing outputs: {missing}")
        batch_per = pd.read_csv(folder / "per_seed.csv")
        batch_oof = pd.read_csv(folder / "oof.csv")
        batch_folds = pd.read_csv(folder / "folds.csv")
        if set(batch_per["seed"].astype(int)) != expected_seeds or len(batch_per) != 10:
            raise ValueError(f"selector batch {folder.name} per_seed coverage differs from {sorted(expected_seeds)}")
        if set(batch_oof["seed"].astype(int)) != expected_seeds or len(batch_oof) != 10 * PATIENTS:
            raise ValueError(f"selector batch {folder.name} OOF coverage differs from its ten expected seeds")
        if set(batch_folds["seed"].astype(int)) != expected_seeds or len(batch_folds) != 50:
            raise ValueError(f"selector batch {folder.name} fold coverage differs from its ten expected seeds")
        per_frames.append(batch_per)
        oof_frames.append(batch_oof)
        fold_frames.append(batch_folds)
        contract_paths.append(folder / "selector_contract.json")
    if len(contract_paths) != 10:
        raise ValueError("selector must have exactly ten complete batch artifact sets")
    batch_per = pd.concat(per_frames, ignore_index=True)
    batch_oof = pd.concat(oof_frames, ignore_index=True)
    batch_folds = pd.concat(fold_frames, ignore_index=True)
    aggregate = results_root / "selector" / "aggregate"
    per = pd.read_csv(aggregate / "nested_selector_per_seed.csv")
    folds = pd.read_csv(aggregate / "nested_selector_folds.csv")
    patient_csv = pd.read_csv(aggregate / "nested_selector_patient_mean_oof.csv")
    summary = json.loads((aggregate / "nested_selector_summary.json").read_text(encoding="utf-8"))
    contract_values = [json.loads(path.read_text(encoding="utf-8")) for path in contract_paths]
    canonical = json.dumps(contract_values[0], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if any(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) != canonical for value in contract_values[1:]):
        raise ValueError("selector batch contracts differ")
    summary_contract = json.dumps(summary.get("contract"), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if summary_contract != canonical:
        raise ValueError("selector aggregate contract differs from raw batch contracts")
    if len(batch_per) != 100 or set(batch_per["seed"].astype(int)) != SEEDS:
        raise ValueError("selector batch per_seed files do not cover exactly seeds 0..99")
    if len(batch_oof) != 100 * PATIENTS or len(batch_folds) != 500:
        raise ValueError("selector batch OOF/fold row counts are incomplete")
    return per, batch_per, batch_oof, folds, batch_folds, summary, patient_csv


def _fixed_inputs(results_root: Path) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
    pd.DataFrame,
]:
    merged_root = results_root / "fixed_p5" / "merged_only"
    combo_dirs = [path for path in merged_root.iterdir() if path.is_dir()] if merged_root.is_dir() else []
    if len(combo_dirs) != 1 or combo_dirs[0].name != FIXED_COMBO:
        raise ValueError("fixed P5 summarize input must be merged-only with only pro_bnp_nsvt")
    merged = combo_dirs[0]
    batch_dirs = sorted((results_root / "fixed_p5" / "batches").glob("batch_*"))
    if len(batch_dirs) != 10:
        raise ValueError("fixed P5 must contain exactly ten raw batch directories")
    metadata = [json.loads((folder / "combo.json").read_text(encoding="utf-8")) for folder in batch_dirs]
    batch_per_frames: list[pd.DataFrame] = []
    batch_oof_frames: list[pd.DataFrame] = []
    batch_fold_frames: list[pd.DataFrame] = []
    for index, folder in enumerate(batch_dirs):
        seed_start = index * 10
        expected_name = f"batch_{seed_start:02d}_{seed_start + 9:02d}"
        if folder.name != expected_name:
            raise ValueError(f"fixed P5 batch directory {folder.name!r} should be {expected_name!r}")
        for name in ("per_seed.csv", "oof.parquet", "folds.csv"):
            if not (folder / name).is_file():
                raise ValueError(f"fixed P5 batch output is missing: {folder / name}")
        batch_per = pd.read_csv(folder / "per_seed.csv")
        batch_oof = pd.read_parquet(folder / "oof.parquet")
        batch_folds = pd.read_csv(folder / "folds.csv")
        _need_columns(batch_per, {"seed"}, f"fixed P5 batch {folder.name} per_seed")
        start = index * 10
        expected = set(range(start, start + 10))
        if len(batch_per) != 10 or set(batch_per["seed"].astype(int)) != expected:
            raise ValueError(f"fixed P5 {folder.name} does not contain exactly seeds {start}..{start + 9}")
        batch_per_frames.append(batch_per)
        batch_oof_frames.append(batch_oof)
        batch_fold_frames.append(batch_folds)
    metadata.append(json.loads((merged / "combo.json").read_text(encoding="utf-8")))
    canonical = json.dumps(metadata[0], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if any(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) != canonical for value in metadata[1:]):
        raise ValueError("fixed P5 batch and merged combo.json metadata are not identical")
    per = pd.read_csv(merged / "per_seed.csv")
    oof = pd.read_parquet(merged / "oof.parquet")
    folds = pd.read_csv(merged / "folds.csv")
    summary_dir = results_root / "fixed_p5" / "summary"
    ranking = pd.read_csv(summary_dir / "combo_ranking.csv")
    if len(ranking) != 1 or str(ranking.iloc[0]["combo"]) != FIXED_COMBO:
        raise ValueError("fixed P5 supplied summarizer must report only pro_bnp_nsvt")
    batch_per = pd.concat(batch_per_frames, ignore_index=True).sort_values("seed", kind="stable").reset_index(drop=True)
    batch_oof = pd.concat(batch_oof_frames, ignore_index=True).sort_values(["seed", "patient_id"], kind="stable").reset_index(drop=True)
    batch_folds = pd.concat(batch_fold_frames, ignore_index=True).sort_values(["seed", "outer_fold"], kind="stable").reset_index(drop=True)
    return per, oof, folds, batch_per, batch_oof, batch_folds, metadata[-1], ranking


def verify(project_root: Path, results_root: Path) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    manifest_path = results_root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETE":
        errors.append(f"run manifest status is {manifest.get('status')!r}, expected COMPLETE")
    before = manifest.get("fingerprints_before")
    after = manifest.get("fingerprints_after")
    if before is None or after is None or before != after:
        errors.append("run manifest does not contain matching before/after source and input fingerprints")
    current = fingerprint_snapshot(project_root)
    if before != current:
        errors.append("current source/input fingerprints do not match the run manifest snapshot")
    requested = set(manifest.get("tasks_requested", []))
    if requested != {"selector", "fixed"}:
        errors.append(f"final verification requires both selector and fixed tasks; manifest requested {sorted(requested)}")

    expected_labels, anchor = _anchor_cohort(project_root)
    selector_per, selector_batch_per, selector_oof_raw, selector_folds, selector_batch_folds, selector_summary, selector_patient_csv = _selector_inputs(results_root)
    fixed_per, fixed_oof_raw, fixed_folds, fixed_batch_per, fixed_batch_oof, fixed_batch_folds, combo_meta, fixed_summary = _fixed_inputs(results_root)
    _compare_artifacts(
        selector_batch_per,
        selector_per,
        key_columns=("seed",),
        label="selector batch per_seed vs aggregate per_seed",
        errors=errors,
    )
    _compare_artifacts(
        selector_batch_folds,
        selector_folds,
        key_columns=("seed", "outer_fold"),
        label="selector batch folds vs aggregate folds",
        errors=errors,
    )
    _compare_artifacts(
        fixed_batch_per,
        fixed_per,
        key_columns=("seed",),
        label="fixed P5 batch per_seed vs merged per_seed",
        errors=errors,
    )
    _compare_artifacts(
        fixed_batch_oof,
        fixed_oof_raw,
        key_columns=("seed", "patient_id"),
        label="fixed P5 batch OOF vs merged OOF",
        errors=errors,
    )
    _compare_artifacts(
        fixed_batch_folds,
        fixed_folds,
        key_columns=("seed", "outer_fold"),
        label="fixed P5 batch folds vs merged folds",
        errors=errors,
    )
    selector, selector_patient = _independent_run(
        selector_oof_raw,
        selector_per,
        selector_folds,
        model="NESTED_RECIPE_SELECTOR",
        label="strict nested selector",
        expected_labels=expected_labels,
        errors=errors,
    )
    fixed, fixed_patient = _independent_run(
        fixed_oof_raw,
        fixed_per,
        fixed_folds,
        model="P5_PRO_BNP_NSVT",
        label="fixed pro_bnp_nsvt",
        expected_labels=expected_labels,
        errors=errors,
    )

    _compare_patient_mean_csv(
        selector_patient_csv,
        selector_patient,
        expected_labels=expected_labels,
        label="selector aggregate patient-mean OOF CSV",
        errors=errors,
    )

    if selector_patient[["patient_id", "true_label"]].to_records(index=False).tolist() != fixed_patient[["patient_id", "true_label"]].to_records(index=False).tolist():
        errors.append("selector and fixed P5 patient sets/labels are not identical")
    selector_expected = f"{EXPECTED_SELECTOR_MEDIAN:.6f}"
    fixed_expected = f"{EXPECTED_FIXED_MEDIAN:.6f}"
    if selector["cv_auc_median_round_6dp"] != selector_expected:
        errors.append(f"selector independent AUC median rounds to {selector['cv_auc_median_round_6dp']}, expected {selector_expected}")
    if fixed["cv_auc_median_round_6dp"] != fixed_expected:
        errors.append(f"fixed P5 independent AUC median rounds to {fixed['cv_auc_median_round_6dp']}, expected {fixed_expected}")

    selector_reported_median = float(selector_summary["per_seed"]["AUC"]["median"])
    _compare(selector["cv_auc_median_independent"], selector_reported_median, label="selector provided aggregate median", errors=errors)
    selector_metrics = selector_summary["patient_mean_oof"]
    for metric in ("AUC", "AP", "Brier"):
        _compare(selector["patient_mean_oof"][metric], float(selector_metrics[metric]), label=f"selector provided patient-mean {metric}", errors=errors)
    provided_bootstrap = selector_summary.get("bootstrap_auc", {})
    for key, output_key in (("lower", "lower_95"), ("upper", "upper_95"), ("median", "median")):
        if key in provided_bootstrap:
            _compare(
                selector["patient_auc_bootstrap"][output_key],
                float(provided_bootstrap[key]),
                label=f"selector provided bootstrap {key}",
                errors=errors,
                tolerance=1e-12,
            )

    fixed_row = fixed_summary.iloc[0]
    _compare(fixed["cv_auc_median_independent"], float(fixed_row["cv_auc_median"]), label="fixed P5 supplied summary median", errors=errors)
    for metric, column in (("AUC", "patient_auc"), ("AP", "patient_ap"), ("Brier", "patient_brier")):
        _compare(fixed["patient_mean_oof"][metric], float(fixed_row[column]), label=f"fixed P5 supplied patient-mean {metric}", errors=errors)
    selector_contract = selector_summary.get("contract", {})
    selector_counts = (
        selector_contract.get("patient_count"),
        selector_contract.get("positive_count"),
        selector_contract.get("negative_count"),
    )
    if selector_counts != (PATIENTS, POSITIVES, NEGATIVES):
        errors.append(f"selector contract cohort counts differ from input anchor: {selector_counts}")
    fixed_counts = (
        combo_meta.get("patient_count"),
        combo_meta.get("positive_count"),
        combo_meta.get("negative_count"),
    )
    if fixed_counts != (PATIENTS, POSITIVES, NEGATIVES):
        errors.append(f"fixed P5 combo cohort counts differ from input anchor: {fixed_counts}")
    recipe_library = selector_contract.get("recipes", {})
    if not recipe_library:
        errors.append("selector contract has no predeclared recipe library")
    else:
        recipe_sizes = {str(name): int(value.get("raw_feature_count", -1)) for name, value in recipe_library.items()}
        for artifact_name, frame in (("batch", selector_batch_folds), ("aggregate", selector_folds)):
            _need_columns(frame, {"selected_recipe", "selected_raw_feature_count"}, f"selector {artifact_name} folds")
            unknown = sorted(set(frame["selected_recipe"].astype(str)) - set(recipe_sizes))
            if unknown:
                errors.append(f"selector {artifact_name} folds contain recipes outside the fixed contract: {unknown[:5]}")
            observed_sizes = pd.to_numeric(frame["selected_raw_feature_count"], errors="raise").astype(int)
            expected_sizes = frame["selected_recipe"].astype(str).map(recipe_sizes)
            if expected_sizes.isna().any() or not np.array_equal(observed_sizes.to_numpy(), expected_sizes.to_numpy(dtype=int)):
                errors.append(f"selector {artifact_name} selected recipe/input counts disagree with the fixed contract")
    if set(selector_summary.get("selection_frequency", {})) != set(selector_folds["selected_recipe"].astype(str)):
        errors.append("selector selection frequency summary has a different recipe set from fold outputs")
    expected_frequency = selector_folds["selected_recipe"].astype(str).value_counts().to_dict()
    actual_frequency = {str(key): int(value) for key, value in selector_summary.get("selection_frequency", {}).items()}
    if actual_frequency != {str(key): int(value) for key, value in expected_frequency.items()}:
        errors.append("selector selection frequencies disagree with the 500 outer-fold rows")
    frequency_report = {str(key): int(value) for key, value in expected_frequency.items()}
    if int(combo_meta.get("raw_feature_count", -1)) != 24 or len(combo_meta.get("features", [])) != 24:
        errors.append("fixed P5 metadata does not document exactly 24 raw inputs")

    result = {
        "schema_version": 1,
        "status": "PASS" if not errors else "FAIL",
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "project_root": str(project_root),
        "results_root": str(results_root),
        "fingerprints_match": before == current and before == after,
        "cohort": {"patients": PATIENTS, "positive": POSITIVES, "negative": NEGATIVES},
        "input_builder_anchor": anchor,
        "targets": {
            "selector_cv_auc_median_expected_6dp": selector_expected,
            "fixed_p5_cv_auc_median_expected_6dp": fixed_expected,
            "selector_cv_median_ge_0_70": selector["cv_auc_median_independent"] >= 0.70,
            "selector_patient_mean_auc_ge_0_70": selector["patient_mean_oof"]["AUC"] >= 0.70,
            "fixed_p5_cv_median_ge_0_70": fixed["cv_auc_median_independent"] >= 0.70,
            "fixed_p5_patient_mean_auc_ge_0_70": fixed["patient_mean_oof"]["AUC"] >= 0.70,
        },
        "reproducibility": {
            "selector_median_matches_expected_6dp": selector["cv_auc_median_round_6dp"] == selector_expected,
            "fixed_p5_median_matches_expected_6dp": fixed["cv_auc_median_round_6dp"] == fixed_expected,
            "selector_supplied_aggregate_matches_independent_oof": not any("selector provided" in error for error in errors),
            "fixed_supplied_summary_matches_independent_oof": not any("fixed P5 supplied" in error for error in errors),
            "selector_batch_artifacts_match_aggregate": not any("selector batch" in error for error in errors),
            "fixed_batch_artifacts_match_merged": not any("fixed P5 batch" in error for error in errors),
            "selector_patient_mean_csv_matches_independent_oof": not any("patient-mean OOF CSV" in error for error in errors),
        },
        "selector": selector,
        "fixed_p5": {**fixed, "combo": combo_meta.get("combo"), "features_24": combo_meta.get("features", [])},
        "selector_recipe_frequency": frequency_report,
        "scope": {
            "evaluation": "repeated internal patient-level cross-validation on the same 878-patient cohort",
            "external_validation_evidence": False,
            "interpretation": "these outputs provide no external-cohort validation claim",
        },
        "validation_errors": errors,
    }
    return result, errors


def _markdown(report: dict[str, Any]) -> str:
    selector = report["selector"]
    fixed = report["fixed_p5"]
    targets = report["targets"]
    cohort = report["cohort"]
    lines = [
        "# Final100复现核验",
        "",
        f"- 核验状态：**{report['status']}**",
        f"- 同源与输入文件指纹匹配：**{report['fingerprints_match']}**",
        f"- 病例数：**{cohort['patients']}**（阳性 {cohort['positive']}，阴性 {cohort['negative']}）",
        "- 病例ID与标签锚点由原始 `build_selector_frame()` 与 `build_combo('pro_bnp_nsvt')` 输入构建器共同生成。",
        "- 两个模型的每个种子均独立从逐患者 OOF 概率重算；验证集和标签覆盖逐种子核对。",
        "",
        "## 100种子结果",
        "",
        "| 方案 | CV AUC中位数（独立重算） | 患者均值OOF AUC | AP | Brier | CV中位数≥0.70 | 患者AUC≥0.70 |",
        "|---|---:|---:|---:|---:|:---:|:---:|",
        f"| 严格嵌套配方选择 | {selector['cv_auc_median_independent']:.6f} | {selector['patient_mean_oof']['AUC']:.6f} | {selector['patient_mean_oof']['AP']:.6f} | {selector['patient_mean_oof']['Brier']:.6f} | {targets['selector_cv_median_ge_0_70']} | {targets['selector_patient_mean_auc_ge_0_70']} |",
        f"| 固定 pro_bnp_nsvt（24项） | {fixed['cv_auc_median_independent']:.6f} | {fixed['patient_mean_oof']['AUC']:.6f} | {fixed['patient_mean_oof']['AP']:.6f} | {fixed['patient_mean_oof']['Brier']:.6f} | {targets['fixed_p5_cv_median_ge_0_70']} | {targets['fixed_p5_patient_mean_auc_ge_0_70']} |",
        "",
        f"- 目标核对：选择器 {selector['cv_auc_median_round_6dp']} / {report['targets']['selector_cv_auc_median_expected_6dp']}；固定模型 {fixed['cv_auc_median_round_6dp']} / {report['targets']['fixed_p5_cv_auc_median_expected_6dp']}。",
        f"- 可复现性核对：选择器目标匹配 {report['reproducibility']['selector_median_matches_expected_6dp']}；固定模型目标匹配 {report['reproducibility']['fixed_p5_median_matches_expected_6dp']}；独立OOF与既有汇总一致性见核验JSON。",
        f"- 阳性率基线 AP：{selector['prevalence']:.6f}；患者均值 AP / 阳性率：选择器 {selector['patient_ap_over_prevalence']:.3f}，固定模型 {fixed['patient_ap_over_prevalence']:.3f}。",
        f"- Dummy Brier基线 q(1−q)：{selector['dummy_brier_q_times_one_minus_q']:.6f}；患者均值 Brier 与基线之差：选择器 {selector['patient_brier_vs_dummy']:+.6f}，固定模型 {fixed['patient_brier_vs_dummy']:+.6f}。",
        "",
        "## 患者均值OOF AUC区间",
        "",
        f"- 选择器：{selector['patient_auc_bootstrap']['lower_95']:.6f}–{selector['patient_auc_bootstrap']['upper_95']:.6f}。",
        f"- 固定模型：{fixed['patient_auc_bootstrap']['lower_95']:.6f}–{fixed['patient_auc_bootstrap']['upper_95']:.6f}。",
        "- 这是对已固定患者均值OOF预测进行患者重抽样的条件区间，不是100种子AUC中位数的置信区间。",
        "",
        "## 选择器配方频次",
        "",
    ]
    lines.extend(f"- `{name}`：{count}/500个外层折。" for name, count in sorted(report["selector_recipe_frequency"].items(), key=lambda item: (-item[1], item[0])))
    lines.extend(["", "## 固定模型24项输入", ""])
    lines.extend(f"- `{feature}`" for feature in fixed["features_24"])
    lines.extend(
        [
            "",
            "## 解读",
            "",
            "严格嵌套配方选择作为主要方法；固定 pro_bnp_nsvt 结果用于固定方案的可复现性对照。两种结果均来自同一878人队列的内部交叉验证，不构成外部验证。",
            "",
            "逐折训练/测试样本数和阳性数已按100个种子核对；每种方案均应有500条外层折摘要。",
        ]
    )
    if report["validation_errors"]:
        lines.extend(["", "## 未通过项", ""])
        lines.extend(f"- {error}" for error in report["validation_errors"])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Independently verify final100 OOF predictions and supplied summaries.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, help="defaults to RESULTS_ROOT/verification")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve(strict=True)
    results_root = args.results_root.resolve(strict=True)
    output_dir = (args.output_dir or (results_root / "verification")).resolve()
    if output_dir == project_root or project_root in output_dir.parents:
        raise SystemExit("verification output must be outside project-root")
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        report, errors = verify(project_root, results_root)
    except Exception as exc:
        report = {
            "schema_version": 1,
            "status": "FAIL",
            "project_root": str(project_root),
            "results_root": str(results_root),
            "fatal_error": f"{type(exc).__name__}: {exc}",
            "validation_errors": [f"{type(exc).__name__}: {exc}"],
        }
        errors = report["validation_errors"]
    (output_dir / "verification.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "verification.md").write_text(_markdown(report) if "selector" in report else "# Final100复现核验\n\n- 核验状态：**FAIL**\n- 错误：" + report.get("fatal_error", "unknown") + "\n", encoding="utf-8")
    print(f"verification {report['status']}: {output_dir}")
    if errors:
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
