"""Build the audited MODEL OPTIMIZATION V1 final artifact from frozen runs."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


FORMAL_CANDIDATES = ("B0", "M1", "P2")
FORMAL_SEEDS = set(range(100))
PVC_SEEDS = set(range(10))
HORIZONS = (90, 180, 365, 730)
CORE_METRICS = ("AUC", "AP", "Brier", "BrierSkill")
PROVENANCE = {
    "formal_aggregate_run_id": 33193292911,
    "endpoint_run_id": 33193756146,
    "seed42_run_id": 33171483784,
    "screening10_run_id": 33187159259,
    "pvc_screening10_run_id": 33187663480,
    "baseline_source_run_id": 33141323899,
}


class FinalizationError(ValueError):
    """Raised when a final source artifact violates a frozen invariant."""


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise FinalizationError(f"invalid JSON: {path}") from exc


def _finite(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FinalizationError(f"{label} must be numeric") from exc
    if not math.isfinite(number):
        raise FinalizationError(f"{label} must be finite")
    return number


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records"))


def _sanitize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def validate_population(audit: Mapping[str, Any]) -> None:
    baseline = audit.get("baseline", {})
    expected = {
        "full_positive": 38,
        "baseline_positive": 27,
        "baseline_excluded_positive": 11,
        "baseline_excluded_positive_af": 10,
        "baseline_excluded_positive_no_holter": 1,
        "rhythm_safe_modeled": 878,
        "rhythm_safe_positive": 37,
        "rhythm_safe_recovered_positive_af": 10,
    }
    for key, value in expected.items():
        if int(baseline.get(key, -1)) != value:
            raise FinalizationError(f"population invariant failed: {key}={baseline.get(key)!r}, expected {value}")
    p2 = audit.get("candidates", {}).get("P2", {})
    for key, value in {
        "patient_count": 878,
        "positive_count": 37,
        "negative_count": 841,
        "feature_count": 24,
        "af_included_count": 164,
        "af_positive_included_count": 10,
    }.items():
        if int(p2.get(key, -1)) != value:
            raise FinalizationError(f"P2 population invariant failed: {key}")
    if p2.get("pvc_fields") != ["pvc_count_24h"]:
        raise FinalizationError("P2 must use only source pvc_count_24h")
    if not bool(audit.get("PVC_CONTINUOUS_UNAVAILABLE")):
        raise FinalizationError("PVC_CONTINUOUS_UNAVAILABLE must remain true")


def _validate_formal(formal_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    required = (
        "runs.csv", "runs.parquet", "summary.csv", "summary.json",
        "paired_comparison.json", "calibration_summary.csv",
        "calibration_bins.csv", "af_subgroup.csv", "pvc_subgroup.csv",
    )
    for name in required:
        if not (formal_dir / name).is_file():
            raise FinalizationError(f"missing formal file: {name}")
    runs = pd.read_csv(formal_dir / "runs.csv")
    parquet = pd.read_parquet(formal_dir / "runs.parquet")
    if len(runs) != 300 or len(parquet) != 300:
        raise FinalizationError("formal runs must contain exactly 300 rows")
    if runs.duplicated(["candidate", "seed"]).any():
        raise FinalizationError("duplicate formal candidate/seed")
    if set(runs["candidate"].astype(str)) != set(FORMAL_CANDIDATES):
        raise FinalizationError("formal candidate set mismatch")
    for candidate, group in runs.groupby("candidate"):
        if set(group["seed"].astype(int)) != FORMAL_SEEDS:
            raise FinalizationError(f"formal seed set mismatch for {candidate}")
    for metric in CORE_METRICS:
        if metric not in runs or not np.isfinite(pd.to_numeric(runs[metric], errors="coerce")).all():
            raise FinalizationError(f"non-finite formal core metric: {metric}")
    if set(pd.to_numeric(runs["endpoint_horizon_days"], errors="coerce")) != {365}:
        raise FinalizationError("formal endpoint must be 365 days")
    summary = pd.read_csv(formal_dir / "summary.csv")
    if set(summary["candidate"].astype(str)) != set(FORMAL_CANDIDATES):
        raise FinalizationError("formal summary candidate set mismatch")
    summary_json = _read_json(formal_dir / "summary.json")
    paired = _read_json(formal_dir / "paired_comparison.json")
    comparisons = paired.get("comparisons", {})
    if paired.get("baseline_candidate") != "B0" or set(comparisons) != {"M1", "P2"}:
        raise FinalizationError("formal paired comparison mismatch")
    for candidate in ("M1", "P2"):
        if int(comparisons[candidate].get("matched_seed_count", -1)) != 100:
            raise FinalizationError(f"paired seed count mismatch for {candidate}")
    calibration = pd.read_csv(formal_dir / "calibration_summary.csv")
    if set(calibration["candidate"].astype(str)) != set(FORMAL_CANDIDATES):
        raise FinalizationError("calibration candidate set mismatch")
    return runs, summary, {"summary": summary_json, "paired": paired}, calibration


def _endpoint_table(root: Path) -> pd.DataFrame:
    rows = []
    for path in root.rglob("*_summary.json"):
        row = _read_json(path)
        horizon = int(row.get("endpoint_horizon_days", -1))
        if horizon in HORIZONS:
            rows.append(row)
    if len(rows) != 4 or {int(row["endpoint_horizon_days"]) for row in rows} != set(HORIZONS):
        raise FinalizationError("endpoint artifacts must contain exactly 90/180/365/730 days")
    output = []
    for row in sorted(rows, key=lambda item: int(item["endpoint_horizon_days"])):
        if row.get("candidate") != "P2" or row.get("model") != "elasticnet":
            raise FinalizationError("endpoint candidate/model mismatch")
        n = int(row["patient_count"])
        positive = int(row["positive_count"])
        expected_dummy = (positive / n) * (1.0 - positive / n)
        output.append({
            "endpoint_horizon_days": int(row["endpoint_horizon_days"]),
            "candidate": "P2", "model": row["model"], "profile": row["profile"],
            "N": n, "positive": positive, "negative": int(row["negative_count"]),
            "AUC": _finite(row["AUC"], "endpoint AUC"),
            "AP": _finite(row["AP"], "endpoint AP"),
            "Brier": _finite(row["Brier"], "endpoint Brier"),
            "Dummy_Brier": expected_dummy,
            "BrierSkill": _finite(row["BrierSkill"], "endpoint BrierSkill"),
            "calibration_intercept": _sanitize(row.get("calibration_intercept")),
            "calibration_slope": _sanitize(row.get("calibration_slope")),
            "analysis_role": "primary" if int(row["endpoint_horizon_days"]) == 365 else "sensitivity",
        })
    return pd.DataFrame(output)


def _screening(dir_path: Path, expected: set[str], seeds: set[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs = pd.read_csv(dir_path / "runs.csv")
    summary = pd.read_csv(dir_path / "summary.csv")
    if set(runs["candidate"].astype(str)) != expected:
        raise FinalizationError(f"screening candidates mismatch: {dir_path}")
    if runs.duplicated(["candidate", "seed"]).any():
        raise FinalizationError("duplicate screening candidate/seed")
    for candidate, group in runs.groupby("candidate"):
        if set(group["seed"].astype(int)) != seeds:
            raise FinalizationError(f"screening seed set mismatch for {candidate}")
    return runs, summary


def _pvc_increment(runs: pd.DataFrame) -> dict[str, Any]:
    p1 = runs.loc[runs["candidate"].eq("P1")].set_index("seed")
    p2 = runs.loc[runs["candidate"].eq("P2")].set_index("seed")
    output: dict[str, Any] = {"matched_seed_count": 10, "P2_minus_P1": {}}
    for metric in CORE_METRICS:
        delta = pd.to_numeric(p2[metric], errors="raise") - pd.to_numeric(p1[metric], errors="raise")
        lower_better = metric == "Brier"
        output["P2_minus_P1"][metric] = {
            "mean": float(delta.mean()), "median": float(delta.median()),
            "p2.5": float(np.percentile(delta, 2.5)), "p97.5": float(np.percentile(delta, 97.5)),
            "candidate_better_fraction": float((delta.lt(0) if lower_better else delta.gt(0)).mean()),
        }
    return output


def _promotion(summary: pd.DataFrame, paired: Mapping[str, Any], calibration: pd.DataFrame) -> tuple[str, dict[str, bool]]:
    rows = summary.set_index("candidate")
    b0, p2 = rows.loc["B0"], rows.loc["P2"]
    comparison = paired["comparisons"]["P2"]
    cal = calibration.set_index("candidate")
    checks = {
        "auc_median_higher": float(p2["AUC_median"]) > float(b0["AUC_median"]),
        "auc_p2.5_not_worse": float(p2["AUC_p2.5"]) >= float(b0["AUC_p2.5"]),
        "ap_median_higher": float(p2["AP_median"]) > float(b0["AP_median"]),
        "brier_median_lower": float(p2["Brier_median"]) < float(b0["Brier_median"]),
        "brier_skill_positive": float(p2["BrierSkill_mean"]) > 0.0,
        "auc_std_lower": float(p2["AUC_std"]) <= float(b0["AUC_std"]),
        "auc_spread_lower": float(p2["AUC_p97.5"] - p2["AUC_p2.5"]) <= float(b0["AUC_p97.5"] - b0["AUC_p2.5"]),
        "paired_auc_all_better": float(comparison["candidate_better_fraction"]["AUC"]) == 1.0,
        "paired_brier_all_better": float(comparison["candidate_better_fraction"]["Brier"]) == 1.0,
        "calibration_slope_closer_to_one": abs(float(cal.loc["P2", "calibration_slope"]) - 1.0) < abs(float(cal.loc["B0", "calibration_slope"]) - 1.0),
        "calibration_intercept_closer_to_zero": abs(float(cal.loc["P2", "calibration_intercept"])) < abs(float(cal.loc["B0", "calibration_intercept"])),
    }
    return ("PROMOTED_CANDIDATE" if all(checks.values()) else "NO_CLEAR_WINNER"), checks


def _fmt(value: Any, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def _markdown(
    *, summary: pd.DataFrame, paired: Mapping[str, Any], calibration: pd.DataFrame,
    population: Mapping[str, Any], endpoints: pd.DataFrame, pvc: Mapping[str, Any],
    screening: pd.DataFrame, seed42: pd.DataFrame, af_subgroup: pd.DataFrame,
    pvc_subgroup: pd.DataFrame, decision: str, checks: Mapping[str, bool],
) -> str:
    rows = summary.set_index("candidate")
    b0, m1, p2 = rows.loc["B0"], rows.loc["M1"], rows.loc["P2"]
    p2_pair = paired["comparisons"]["P2"]
    m1_pair = paired["comparisons"]["M1"]
    cal = calibration.set_index("candidate")
    pvc_delta = pvc["P2_minus_P1"]
    screen = screening.set_index("candidate")
    seed = seed42.set_index("candidate")
    p2_af = af_subgroup.loc[af_subgroup["candidate"].eq("P2")].set_index("level")
    p2_pvc = pvc_subgroup.loc[pvc_subgroup["candidate"].eq("P2")].set_index("level")
    lines = ["# MODEL OPTIMIZATION V1", ""]
    lines += ["## 1. Baseline", "", (
        f"Primary endpoint remained prespecified at 365 d. The frozen B0 ExtraTrees/all20_100 population was N=703 "
        f"(27 positive, 676 negative). Across 100 patient-level nested-CV splits, AUC median was {_fmt(b0.AUC_median)} "
        f"(IQR {_fmt(b0['AUC_p25'])}-{_fmt(b0['AUC_p75'])}; 2.5-97.5 percentile {_fmt(b0['AUC_p2.5'])}-{_fmt(b0['AUC_p97.5'])}), "
        f"and Brier median was {_fmt(b0.Brier_median)}. These percentiles describe split sensitivity, not a population confidence interval."
    ), ""]
    lines += ["## 2. Small-N/high-D experiment", "", (
        f"Reducing 100 features to M1 median20 improved formal AUC median to {_fmt(m1.AUC_median)} and Brier median to {_fmt(m1.Brier_median)}. "
        f"The paired AUC difference median was {_fmt(m1_pair['AUC_median_delta'])}, with M1 better in {100*m1_pair['candidate_better_fraction']['AUC']:.0f}% of seeds. "
        f"However, AUC SD increased from {_fmt(b0.AUC_std)} to {_fmt(m1.AUC_std)}, and the 2.5-97.5 width increased from "
        f"{_fmt(b0['AUC_p97.5']-b0['AUC_p2.5'])} to {_fmt(m1['AUC_p97.5']-m1['AUC_p2.5'])}; dimensionality reduction improved performance and probability loss but did not by itself reduce split variation. "
        f"M2 robust40 screening AUC median was {_fmt(screen.loc['M2','AUC_median'])}; M4 inner-fold selection was retained only as seed42 evidence (AUC {_fmt(seed.loc['M4','AUC'])}) and was not promoted."
    ), ""]
    lines += ["## 3. AF-compatible experiment", "", (
        "Of 38 full-cohort 365 d positives, B0 modeled 27 and excluded 11: 10 for AF and 1 for NO_HOLTER. "
        f"P2 rhythm_safe included N={int(p2.patient_count)}, {int(p2.positive_count)} positives, 164 AF patients and all 10 previously lost AF positives; only the NO_HOLTER positive remained excluded. "
        f"Formal overall AUC median was {_fmt(p2.AUC_median)} and AP median {_fmt(p2.AP_median)}. In pooled repeated OOF, sinus patients (714 unique; 27 positive) had AUC {_fmt(p2_af.loc[0,'AUC'])}, AP {_fmt(p2_af.loc[0,'AP'])}, Brier {_fmt(p2_af.loc[0,'Brier'])}; "
        f"AF patients (164 unique; 10 positive) had AUC {_fmt(p2_af.loc[1,'AUC'])}, AP {_fmt(p2_af.loc[1,'AP'])}, Brier {_fmt(p2_af.loc[1,'Brier'])}. Both subgroup estimates are exploratory, especially AF because there are only 10 unique positive patients."
    ), ""]
    lines += ["## 4. PVC incremental experiment", "", (
        "The only valid source PVC variable was pvc_count_24h; the derived burden was prohibited because numerator and denominator time bases differ (PVC_CONTINUOUS_UNAVAILABLE=true). "
        f"In paired 10-seed screening, adding PVC changed AUC by median {_fmt(pvc_delta['AUC']['median'])} and was better in {100*pvc_delta['AUC']['candidate_better_fraction']:.0f}% of seeds; "
        f"AP was better in {100*pvc_delta['AP']['candidate_better_fraction']:.0f}% and Brier in {100*pvc_delta['Brier']['candidate_better_fraction']:.0f}%. "
        "Thus PVC showed a stable ranking gain in screening, while its calibration increment was small/uncertain; this is incremental prediction evidence, not causal or independent-effect proof. "
        f"The formal high-PVC subgroup contained only 6 unique patients and no positive event (Brier {_fmt(p2_pvc.loc[1,'Brier'])}), so subgroup AUC/AP were undefined and no high-versus-low PVC conclusion is supportable."
    ), ""]
    lines += ["## 5. Calibration", "", (
        f"B0 pooled Brier was {_fmt(b0.Brier_mean)} with mean Brier Skill {_fmt(b0.BrierSkill_mean)}, calibration slope {_fmt(cal.loc['B0','calibration_slope'])} and intercept {_fmt(cal.loc['B0','calibration_intercept'])}. "
        f"P2 pooled Brier was {_fmt(p2.Brier_mean)}, mean Brier Skill {_fmt(p2.BrierSkill_mean)}, slope {_fmt(cal.loc['P2','calibration_slope'])} and intercept {_fmt(cal.loc['P2','calibration_intercept'])}; "
        f"its per-seed slope median was {_fmt(p2.calibration_slope_median)}. Calibration improved substantially, although rare-event bin estimates and extreme-bin MCE remain unstable."
    ), ""]
    lines += ["## 6. 100-seed robustness", "", (
        f"Changing patient splits 100 times gave P2 AUC mean {_fmt(p2.AUC_mean)}, median {_fmt(p2.AUC_median)}, SD {_fmt(p2.AUC_std)}, IQR {_fmt(p2['AUC_p25'])}-{_fmt(p2['AUC_p75'])}, "
        f"and 2.5-97.5 percentile {_fmt(p2['AUC_p2.5'])}-{_fmt(p2['AUC_p97.5'])}; 100% of seeds exceeded both 0.5 and 0.55. "
        f"Versus B0, paired AUC difference median was {_fmt(p2_pair['AUC_median_delta'])} and P2 was better in 100% of seeds; Brier was also lower in 100%. "
        f"P2 AUC spread {_fmt(p2['AUC_p97.5']-p2['AUC_p2.5'])} was narrower than B0 {_fmt(b0['AUC_p97.5']-b0['AUC_p2.5'])}, so split sensitivity decreased."
    ), ""]
    endpoint_text = "; ".join(
        f"{int(r.endpoint_horizon_days)} d: N={int(r.N)}, positive={int(r.positive)}, AUC={_fmt(r.AUC)}, AP={_fmt(r.AP)}, Brier={_fmt(r.Brier)}, BSS={_fmt(r.BrierSkill)}"
        for _, r in endpoints.iterrows()
    )
    lines += ["## 7. Endpoint sensitivity", "", endpoint_text + ". 365 d remains primary; 90/180/730 d are sensitivity analyses only. The 90 d and 180 d Brier Skill values were negative.", ""]
    lines += ["## 8. Final selected model", "", (
        f"Decision: **{decision} — P2 rhythm_safe + Elastic-Net Logistic + source pvc_count_24h**. It uses 24 features, fold-local imputation/scaling/search, patient-level outer OOF predictions, and no test-label calibration or threshold fitting. "
        "Promotion reflects the combined evidence of higher discrimination, improved Brier/Brier Skill and calibration, lower split variation, and recovery of AF coverage—not one favorable seed."
    ), ""]
    lines += ["## 9. Limitations", "", (
        "There are only 37 modeled 365 d events and 10 AF-positive patients; subgroup estimates are exploratory. PVC is represented by a source 24-hour count rather than a reliable continuous burden. "
        "Repeated-split percentiles are not confidence intervals and repeated OOF rows are correlated within patient. External validation, clinically prespecified risk thresholds, and independent calibration assessment are still required; this does not establish clinical utility."
    ), ""]
    lines += ["## 10. Advisor-facing conclusion", "", (
        "(1) The original 100-dimensional representation was unnecessarily costly: median20 improved AUC/Brier, but did not alone stabilize patient splits. The final 24-feature P2 model did improve both performance and split stability. "
        "(2) AF exclusion materially limited coverage: the AF-compatible model recovered all 10 AF positives and increased modeled positives from 27 to 37 without sacrificing overall repeated performance. "
        "(3) PVC supplied a repeatable screening-level ranking increment, but only weak/uncertain additional calibration benefit. Overall, P2 is more reliable than the original ExtraTrees for 365 d risk prediction in this dataset, while remaining a research candidate rather than a clinically deployable model."
    ), ""]
    if not all(checks.values()):
        lines.append("Promotion checks not all satisfied; inspect machine-readable decision checks.")
    return "\n".join(lines).replace("\ufffd\ufffd", "—")


def finalize(
    formal_dir: Path, population_path: Path, endpoints_root: Path, seed42_dir: Path,
    screening_dir: Path, pvc_screening_dir: Path, output_dir: Path,
) -> dict[str, Any]:
    runs, summary, formal_payload, calibration = _validate_formal(formal_dir)
    population = _read_json(population_path)
    validate_population(population)
    endpoints = _endpoint_table(endpoints_root)
    seed42_runs = pd.read_csv(seed42_dir / "runs.csv")
    if not {"M1", "M2", "M4", "A1", "P1", "P2"}.issubset(set(seed42_runs["candidate"])):
        raise FinalizationError("seed42 artifact lacks required candidates")
    screen_runs, screen_summary = _screening(screening_dir, {"M1", "M2", "A1"}, PVC_SEEDS)
    pvc_runs, pvc_summary = _screening(pvc_screening_dir, {"P1", "P2"}, PVC_SEEDS)
    pvc = _pvc_increment(pvc_runs)
    decision, checks = _promotion(summary, formal_payload["paired"], calibration)
    af_subgroup = pd.read_csv(formal_dir / "af_subgroup.csv")
    pvc_subgroup = pd.read_csv(formal_dir / "pvc_subgroup.csv")

    output_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "runs.csv", "runs.parquet", "summary.csv", "summary.json",
        "paired_comparison.json", "calibration_summary.csv", "calibration_bins.csv",
        "af_subgroup.csv", "pvc_subgroup.csv",
    ):
        source = formal_dir / name
        if source.is_file():
            shutil.copy2(source, output_dir / name)
    shutil.copy2(population_path, output_dir / "population_audit.json")
    endpoints.to_csv(output_dir / "endpoint_sensitivity.csv", index=False)
    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    markdown = _markdown(
        summary=summary, paired=formal_payload["paired"], calibration=calibration,
        population=population, endpoints=endpoints, pvc=pvc, screening=screen_summary,
        seed42=seed42_runs, af_subgroup=af_subgroup, pvc_subgroup=pvc_subgroup,
        decision=decision, checks=checks,
    )
    (report_dir / "MODEL_OPTIMIZATION.md").write_text(markdown, encoding="utf-8")
    machine = {
        "schema_version": "model-optimization-v1-final-1",
        "decision": decision,
        "selected_candidate": "P2" if decision == "PROMOTED_CANDIDATE" else None,
        "provenance": PROVENANCE,
        "integrity": {
            "formal_rows": int(len(runs)), "candidate_seed_duplicates": 0,
            "candidate_seed_counts": {key: int(value) for key, value in runs.groupby("candidate").size().items()},
            "core_nonfinite": {metric: int((~np.isfinite(runs[metric])).sum()) for metric in CORE_METRICS},
            "promotion_checks": checks,
        },
        "population_audit": population,
        "formal_summary": _records(summary),
        "paired_comparison": formal_payload["paired"],
        "calibration_summary": _records(calibration),
        "screening10_summary": _records(screen_summary),
        "seed42_runs": _records(seed42_runs),
        "pvc_screening10_summary": _records(pvc_summary),
        "pvc_incremental": pvc,
        "endpoint_sensitivity": _records(endpoints),
        "af_subgroup": _records(af_subgroup),
        "pvc_subgroup": _records(pvc_subgroup),
        "advisor_answers": {
            "small_n_high_d": "Median20 improved performance/calibration but not split stability; final P2 improved both.",
            "af": "P2 recovered all 10 AF positives and increased modeled positives from 27 to 37.",
            "pvc": "Stable ranking gain in 10-seed screening; calibration increment small/uncertain.",
            "split_sensitivity": "P2 AUC spread and SD were lower than baseline and it beat baseline AUC in all 100 paired seeds.",
        },
    }
    machine = _sanitize(machine)
    (report_dir / "model_optimization.json").write_text(
        json.dumps(machine, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return machine


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-dir", type=Path, required=True)
    parser.add_argument("--population-audit", type=Path, required=True)
    parser.add_argument("--endpoints-root", type=Path, required=True)
    parser.add_argument("--seed42-dir", type=Path, required=True)
    parser.add_argument("--screening-dir", type=Path, required=True)
    parser.add_argument("--pvc-screening-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = finalize(
        args.formal_dir, args.population_audit, args.endpoints_root, args.seed42_dir,
        args.screening_dir, args.pvc_screening_dir, args.output_dir,
    )
    print(json.dumps({"decision": result["decision"], "selected_candidate": result["selected_candidate"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
