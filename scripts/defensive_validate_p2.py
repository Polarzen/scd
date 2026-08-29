#!/usr/bin/env python3
"""Post-hoc defensive validation for the frozen P2 candidate.

This script never refits or retunes P2.  It consumes the already-frozen
100-seed nested-CV artifacts and verifies their patient-level integrity.  It
also removes the dependence created by pooling repeated OOF rows by averaging
P2 probabilities per unique patient across seeds before calculating the final
subgroup diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


METRIC_TOL = 1e-12
BOOTSTRAP_SEED = 20260829
BOOTSTRAP_RESAMPLES = 2000


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    probability = np.asarray(probability, dtype=float)
    if len(np.unique(y)) < 2:
        auc = float("nan")
        ap = float("nan")
    else:
        auc = float(roc_auc_score(y, probability))
        ap = float(average_precision_score(y, probability))
    brier = float(brier_score_loss(y, probability))
    prevalence = float(np.mean(y))
    reference_brier = float(prevalence * (1.0 - prevalence))
    brier_skill = float(1.0 - brier / reference_brier) if reference_brier > 0 else float("nan")
    return {
        "AUC": auc,
        "AP": ap,
        "Brier": brier,
        "BrierSkill": brier_skill,
        "prevalence": prevalence,
    }


def _calibration(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    probability = np.clip(np.asarray(probability, dtype=float), 1e-8, 1.0 - 1e-8)
    if len(np.unique(y)) < 2:
        return {"calibration_intercept": float("nan"), "calibration_slope": float("nan")}
    logit = np.log(probability / (1.0 - probability)).reshape(-1, 1)
    model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=10000)
    model.fit(logit, y)
    return {
        "calibration_intercept": float(model.intercept_[0]),
        "calibration_slope": float(model.coef_[0, 0]),
    }


def _bootstrap_unique_patients(frame: pd.DataFrame, n_resamples: int = BOOTSTRAP_RESAMPLES) -> dict:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    y = frame["y_true"].to_numpy(dtype=int)
    probability = frame["probability_mean"].to_numpy(dtype=float)
    n = len(frame)
    values = {"AUC": [], "AP": [], "Brier": []}
    for _ in range(n_resamples):
        indices = rng.integers(0, n, size=n)
        ys = y[indices]
        ps = probability[indices]
        values["Brier"].append(float(brier_score_loss(ys, ps)))
        if len(np.unique(ys)) >= 2:
            values["AUC"].append(float(roc_auc_score(ys, ps)))
            values["AP"].append(float(average_precision_score(ys, ps)))
    result: dict[str, dict[str, float | int | None]] = {}
    for metric, samples in values.items():
        arr = np.asarray(samples, dtype=float)
        result[metric] = {
            "lower": float(np.quantile(arr, 0.025)) if len(arr) else None,
            "upper": float(np.quantile(arr, 0.975)) if len(arr) else None,
            "n_valid": int(len(arr)),
        }
    return result


def _content_digest(paths: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda p: str(p.relative_to(root))):
        relative = str(path.relative_to(root)).replace("\\", "/")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _subgroup_metrics(seed_frame: pd.DataFrame, seed: int) -> list[dict]:
    rows: list[dict] = []
    groups = {
        "overall": np.ones(len(seed_frame), dtype=bool),
        "AF": seed_frame["af_flag"].astype(bool).to_numpy(),
        "sinus": ~seed_frame["af_flag"].astype(bool).to_numpy(),
    }
    for name, mask in groups.items():
        part = seed_frame.loc[mask]
        metric = _metrics(part["y_true"].to_numpy(), part["probability"].to_numpy())
        rows.append(
            {
                "seed": seed,
                "subgroup": name,
                "patient_count": int(len(part)),
                "positive_count": int(part["y_true"].sum()),
                **metric,
            }
        )
    return rows


def validate(artifact_root: Path, freeze_path: Path, output_dir: Path) -> dict:
    freeze = _load_json(freeze_path)
    expected_seeds = set(range(int(freeze["formal_seeds"]["start"]), int(freeze["formal_seeds"]["stop_inclusive"]) + 1))
    summary_paths = sorted(artifact_root.rglob("optimization_P2_365d_seed*_summary.json"))
    if len(summary_paths) != len(expected_seeds):
        raise AssertionError(f"expected {len(expected_seeds)} P2 summaries, found {len(summary_paths)}")

    all_oof: list[pd.DataFrame] = []
    subgroup_rows: list[dict] = []
    observed_seeds: set[int] = set()
    reference_patient_ids: tuple[str, ...] | None = None
    reference_labels: pd.Series | None = None
    reference_af: pd.Series | None = None
    reference_manifest_hash: str | None = None
    checked_paths: list[Path] = []

    for summary_path in summary_paths:
        summary = _load_json(summary_path)
        seed = int(summary["seed"])
        if seed in observed_seeds:
            raise AssertionError(f"duplicate P2 seed {seed}")
        observed_seeds.add(seed)

        assert summary["candidate"] == freeze["candidate"] == "P2"
        assert summary["endpoint_horizon_days"] == freeze["endpoint_horizon_days"] == 365
        assert summary["profile"] == freeze["profile"] == "rhythm_safe"
        assert summary["model"] == freeze["model"] == "elasticnet"
        assert summary["patient_count"] == freeze["patient_count"] == 878
        assert summary["positive_count"] == freeze["positive_count"] == 37
        assert summary["negative_count"] == freeze["negative_count"] == 841
        assert summary["outer_folds"] == freeze["outer_folds"] == 5
        assert summary["inner_folds"] == freeze["inner_folds"] == 3
        assert summary["feature_count"] == freeze["feature_count"] == 24
        assert summary["feature_columns"] == freeze["feature_columns"]
        assert summary["af_included_count"] == freeze["af_included_count"] == 164
        assert summary["af_positive_included_count"] == freeze["af_positive_included_count"] == 10

        stem = summary_path.name.replace("_summary.json", "")
        oof_path = summary_path.with_name(stem + "_oof.csv")
        manifest_path = summary_path.with_name("population_manifest.csv")
        if not oof_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(f"missing OOF or population manifest beside {summary_path}")

        oof = pd.read_csv(oof_path)
        required = {
            "candidate", "patient_id", "y_true", "probability", "outer_fold", "model",
            "profile", "seed", "endpoint_horizon_days", "af_flag", "pvc_count_24h",
        }
        missing = required - set(oof.columns)
        if missing:
            raise AssertionError(f"seed {seed} OOF missing columns: {sorted(missing)}")
        if len(oof) != freeze["patient_count"]:
            raise AssertionError(f"seed {seed}: expected 878 OOF rows, got {len(oof)}")
        if oof["patient_id"].isna().any() or oof["patient_id"].duplicated().any():
            raise AssertionError(f"seed {seed}: patient_id must be unique and non-null")
        if set(oof["outer_fold"].astype(int)) != {1, 2, 3, 4, 5}:
            raise AssertionError(f"seed {seed}: invalid outer fold set")
        if not oof["seed"].astype(int).eq(seed).all():
            raise AssertionError(f"seed {seed}: OOF seed column mismatch")
        if not oof["candidate"].eq("P2").all() or not oof["profile"].eq("rhythm_safe").all() or not oof["model"].eq("elasticnet").all():
            raise AssertionError(f"seed {seed}: candidate/profile/model drift")
        if not oof["endpoint_horizon_days"].astype(int).eq(365).all():
            raise AssertionError(f"seed {seed}: endpoint drift")
        if set(oof["y_true"].astype(int).unique()) != {0, 1}:
            raise AssertionError(f"seed {seed}: y_true must contain both classes")
        probability = pd.to_numeric(oof["probability"], errors="coerce")
        if probability.isna().any() or ((probability < 0) | (probability > 1)).any():
            raise AssertionError(f"seed {seed}: invalid probabilities")

        oof = oof.sort_values("patient_id").reset_index(drop=True)
        ids = tuple(oof["patient_id"].astype(str))
        labels = oof.set_index("patient_id")["y_true"].astype(int).sort_index()
        af = oof.set_index("patient_id")["af_flag"].astype(bool).sort_index()
        if reference_patient_ids is None:
            reference_patient_ids = ids
            reference_labels = labels
            reference_af = af
        else:
            if ids != reference_patient_ids:
                raise AssertionError(f"seed {seed}: patient population differs from seed reference")
            if not labels.equals(reference_labels):
                raise AssertionError(f"seed {seed}: labels differ across seeds")
            if not af.equals(reference_af):
                raise AssertionError(f"seed {seed}: AF status differs across seeds")

        raw_metric = _metrics(oof["y_true"].to_numpy(), probability.loc[oof.index].to_numpy())
        for key in ("AUC", "AP", "Brier", "BrierSkill"):
            if not np.isclose(raw_metric[key], float(summary[key]), atol=METRIC_TOL, rtol=0):
                raise AssertionError(f"seed {seed}: recomputed {key} != summary")

        manifest = pd.read_csv(manifest_path).sort_values("patient_id").reset_index(drop=True)
        if int(manifest["included"].astype(bool).sum()) != 878:
            raise AssertionError(f"seed {seed}: manifest included count is not 878")
        included_ids = set(manifest.loc[manifest["included"].astype(bool), "patient_id"].astype(str))
        if included_ids != set(ids):
            raise AssertionError(f"seed {seed}: OOF population != included manifest population")
        evaluable_positive = manifest["label"].eq(1)
        if int(evaluable_positive.sum()) != freeze["full_365d_positive_count"]:
            raise AssertionError(f"seed {seed}: full 365d positive count drift")
        if int((manifest["included"].astype(bool) & evaluable_positive).sum()) != 37:
            raise AssertionError(f"seed {seed}: included positive count drift")
        if int((manifest["included"].astype(bool) & manifest["af_flag"].astype(bool)).sum()) != 164:
            raise AssertionError(f"seed {seed}: included AF count drift")
        if int((manifest["included"].astype(bool) & manifest["af_flag"].astype(bool) & evaluable_positive).sum()) != 10:
            raise AssertionError(f"seed {seed}: included AF-positive count drift")
        excluded_positive = manifest.loc[~manifest["included"].astype(bool) & evaluable_positive]
        if len(excluded_positive) != 1 or excluded_positive["exclusion_reason"].astype(str).tolist() != ["NO_HOLTER"]:
            raise AssertionError(f"seed {seed}: expected exactly one NO_HOLTER positive exclusion")

        manifest_check = manifest[["patient_id", "endpoint_state", "label", "af_flag", "included", "exclusion_reason"]].copy()
        manifest_hash = hashlib.sha256(manifest_check.to_csv(index=False).encode("utf-8")).hexdigest()
        if reference_manifest_hash is None:
            reference_manifest_hash = manifest_hash
        elif manifest_hash != reference_manifest_hash:
            raise AssertionError(f"seed {seed}: population manifest drift across seeds")

        subgroup_rows.extend(_subgroup_metrics(oof, seed))
        all_oof.append(oof[["patient_id", "y_true", "probability", "af_flag", "pvc_count_24h", "seed"]].copy())
        checked_paths.extend([summary_path, oof_path, manifest_path])

    if observed_seeds != expected_seeds:
        raise AssertionError(f"seed set mismatch: missing={sorted(expected_seeds-observed_seeds)}, extra={sorted(observed_seeds-expected_seeds)}")

    repeated = pd.concat(all_oof, ignore_index=True)
    counts = repeated.groupby("patient_id").size()
    if not counts.eq(len(expected_seeds)).all():
        raise AssertionError("each patient must have exactly one OOF prediction per seed")
    for column in ("y_true", "af_flag", "pvc_count_24h"):
        if (repeated.groupby("patient_id")[column].nunique(dropna=False) != 1).any():
            raise AssertionError(f"patient-level {column} changes across seeds")

    unique_patient = (
        repeated.groupby("patient_id", as_index=False)
        .agg(
            y_true=("y_true", "first"),
            af_flag=("af_flag", "first"),
            pvc_count_24h=("pvc_count_24h", "first"),
            probability_mean=("probability", "mean"),
            probability_sd=("probability", "std"),
            seed_count=("seed", "nunique"),
        )
        .sort_values("patient_id")
        .reset_index(drop=True)
    )
    if len(unique_patient) != 878 or not unique_patient["seed_count"].eq(100).all():
        raise AssertionError("unique-patient repeated-OOF aggregation is incomplete")

    unique_rows: list[dict] = []
    unique_json: dict[str, dict] = {}
    subgroup_masks = {
        "overall": np.ones(len(unique_patient), dtype=bool),
        "AF": unique_patient["af_flag"].astype(bool).to_numpy(),
        "sinus": ~unique_patient["af_flag"].astype(bool).to_numpy(),
    }
    for subgroup, mask in subgroup_masks.items():
        part = unique_patient.loc[mask].copy()
        metrics = _metrics(part["y_true"].to_numpy(), part["probability_mean"].to_numpy())
        calibration = _calibration(part["y_true"].to_numpy(), part["probability_mean"].to_numpy())
        bootstrap = _bootstrap_unique_patients(part)
        row = {
            "subgroup": subgroup,
            "patient_count": int(len(part)),
            "positive_count": int(part["y_true"].sum()),
            **metrics,
            **calibration,
        }
        unique_rows.append(row)
        unique_json[subgroup] = {**row, "patient_bootstrap_95ci": bootstrap}

    seed_subgroups = pd.DataFrame(subgroup_rows)
    distributions = []
    for subgroup, group in seed_subgroups.groupby("subgroup", sort=False):
        for metric in ("AUC", "AP", "Brier", "BrierSkill"):
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(dtype=float)
            distributions.append(
                {
                    "subgroup": subgroup,
                    "metric": metric,
                    "n_seeds": int(len(values)),
                    "mean": float(np.mean(values)),
                    "sd": float(np.std(values, ddof=1)),
                    "median": float(np.median(values)),
                    "p2_5": float(np.quantile(values, 0.025)),
                    "p25": float(np.quantile(values, 0.25)),
                    "p75": float(np.quantile(values, 0.75)),
                    "p97_5": float(np.quantile(values, 0.975)),
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(unique_rows).to_csv(output_dir / "unique_patient_metrics.csv", index=False)
    unique_patient.to_csv(output_dir / "unique_patient_predictions.csv", index=False)
    seed_subgroups.to_csv(output_dir / "subgroup_metrics_by_seed.csv", index=False)
    pd.DataFrame(distributions).to_csv(output_dir / "subgroup_seed_distributions.csv", index=False)

    result = {
        "freeze_id": freeze["freeze_id"],
        "status": "PASS",
        "candidate": "P2",
        "endpoint_horizon_days": 365,
        "source_run_id": freeze["formal_seed_source_run_id"],
        "seed_count": len(observed_seeds),
        "seed_set_exact_0_99": observed_seeds == set(range(100)),
        "oof_rows_total": int(len(repeated)),
        "unique_patient_count": int(len(unique_patient)),
        "predictions_per_patient": 100,
        "patient_population_identical_across_seeds": True,
        "labels_identical_across_seeds": True,
        "af_status_identical_across_seeds": True,
        "population_manifest_identical_across_seeds": True,
        "unique_patient_metrics": unique_json,
        "source_content_sha256": _content_digest(checked_paths, artifact_root),
        "bootstrap_note": "95% percentile intervals resample the 878 unique patients after averaging each patient's 100 OOF probabilities; they are distinct from the 100-seed split-sensitivity percentiles.",
        "af_note": "AF subgroup estimates remain exploratory because only 10 unique AF-positive patients are available.",
        "freeze_policy_enforced": freeze["freeze_policy"],
    }
    (output_dir / "defensive_validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    af_metric = unique_json["AF"]
    overall_metric = unique_json["overall"]
    sinus_metric = unique_json["sinus"]
    markdown = f"""# P2 frozen candidate defensive validation\n\n"
    markdown += f"- Freeze: `{freeze['freeze_id']}`\n"
    markdown += f"- Source formal seed run: `{freeze['formal_seed_source_run_id']}`\n"
    markdown += f"- Exact seed set: 0-99 ({len(observed_seeds)} seeds)\n"
    markdown += f"- OOF integrity: {len(repeated):,} rows = {len(unique_patient)} unique patients × 100 seeds; one OOF prediction per patient per seed.\n"
    markdown += f"- Frozen population: {len(unique_patient)} patients, {int(unique_patient['y_true'].sum())} positives; AF {int(unique_patient['af_flag'].sum())} patients / {int((unique_patient['af_flag'].astype(bool) & unique_patient['y_true'].eq(1)).sum())} positives.\n\n"
    markdown += "## Unique-patient aggregation across 100 seeds\n\n"
    markdown += "The 100 OOF probabilities for each patient were averaged first; metrics below therefore use each patient exactly once.\n\n"
    markdown += "| subgroup | N | positives | AUC | AP | Brier | Brier Skill | calibration slope | calibration intercept |\n"
    markdown += "|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    for name, metric in (("overall", overall_metric), ("AF", af_metric), ("sinus", sinus_metric)):
        markdown += f"| {name} | {metric['patient_count']} | {metric['positive_count']} | {metric['AUC']:.4f} | {metric['AP']:.4f} | {metric['Brier']:.4f} | {metric['BrierSkill']:.4f} | {metric['calibration_slope']:.4f} | {metric['calibration_intercept']:.4f} |\n"
    markdown += "\nThe AF subgroup remains exploratory because it contains only 10 unique positive patients. Patient-bootstrap intervals are stored in `defensive_validation.json`; the 100-seed split distributions are stored separately and must not be called population confidence intervals.\n"
    markdown += "\nNo model, feature, endpoint, threshold strategy, or patient inclusion rule was retuned during this validation.\n"
    (output_dir / "P2_FREEZE_VALIDATION.md").write_text(markdown, encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--freeze-config", type=Path, default=Path("config/p2_frozen_v1.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.artifact_root, args.freeze_config, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
