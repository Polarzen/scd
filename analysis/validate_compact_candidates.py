from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from analysis.validate_p3 import (
    ENDPOINT_DAYS,
    INNER_FOLDS,
    OUTER_FOLDS,
    _distribution,
    build_frame,
    metric_row,
    validate_one_seed,
)

USER_MODEL = "USER_LOW20"
ASSISTANT_MODEL = "ASSISTANT_COMPACT23"
MODELS = (USER_MODEL, ASSISTANT_MODEL)

REMOVED_REDUNDANT = (
    "sig_std_median",
    "sig_std_p90_p10",
    "beats_median",
    "beats_p90_p10",
)


def build_candidate_frame() -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, Any]]:
    frame, p2_cols, _p3_cols, base_counts = build_frame()

    compact_ecg = [column for column in p2_cols if column not in {*REMOVED_REDUNDANT, "af_flag", "pvc_count_24h"}]
    if len(compact_ecg) != 18:
        raise RuntimeError(f"expected 18 compact ECG inputs, got {len(compact_ecg)}: {compact_ecg}")

    pvc = pd.to_numeric(frame["pvc_count_24h"], errors="coerce").astype("float64")
    if (pvc.dropna() < 0).any():
        raise RuntimeError("PVC count contains negative values")
    frame["log1p_pvc_count_24h"] = np.log1p(pvc)

    nyha = pd.to_numeric(frame["nyha"], errors="coerce").astype("float64")
    observed_nyha = set(nyha.dropna().unique().tolist())
    if not observed_nyha <= {2.0, 3.0}:
        raise RuntimeError(f"unexpected NYHA classes in manuscript cohort: {sorted(observed_nyha)}")
    frame["nyha_III"] = np.where(nyha.isna(), np.nan, (nyha == 3.0).astype(float))

    user_cols = compact_ecg + ["af_flag", "pvc_count_24h"]
    assistant_cols = compact_ecg + [
        "af_flag",
        "log1p_pvc_count_24h",
        "age",
        "lvef",
        "nyha_III",
    ]

    if len(user_cols) != 20:
        raise RuntimeError(f"user low-dimensional model must have 20 raw inputs, got {len(user_cols)}")
    if len(assistant_cols) != 23:
        raise RuntimeError(f"assistant compact model must have 23 raw inputs, got {len(assistant_cols)}")

    for column in set(user_cols + assistant_cols):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")

    meta = {
        **base_counts,
        "candidate_models": {
            USER_MODEL: {
                "raw_feature_count": len(user_cols),
                "definition": "P2 with deterministic redundancy removal only",
                "removed_from_P2": list(REMOVED_REDUNDANT),
                "features": user_cols,
            },
            ASSISTANT_MODEL: {
                "raw_feature_count": len(assistant_cols),
                "definition": "compact ECG + AF + log1p(PVC) + Age + LVEF + NYHA III indicator",
                "removed_from_P2": list(REMOVED_REDUNDANT),
                "features": assistant_cols,
            },
        },
    }
    return frame, {USER_MODEL: user_cols, ASSISTANT_MODEL: assistant_cols}, meta


def command_batch(args: argparse.Namespace) -> int:
    frame, model_features, meta = build_candidate_frame()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    per_seed_rows: list[dict[str, Any]] = []
    oof_frames: list[pd.DataFrame] = []
    fold_frames: list[pd.DataFrame] = []

    for seed in range(int(args.seed_start), int(args.seed_stop) + 1):
        for model_name in MODELS:
            summary, oof, folds = validate_one_seed(
                frame,
                model_features[model_name],
                model_name=model_name,
                seed=seed,
            )
            per_seed_rows.append(summary)
            oof_frames.append(oof)
            fold_frames.append(folds)

    pd.DataFrame(per_seed_rows).to_csv(output / "per_seed.csv", index=False)
    pd.concat(oof_frames, ignore_index=True).to_parquet(
        output / "oof.parquet", index=False, compression="zstd"
    )
    pd.concat(fold_frames, ignore_index=True).to_csv(output / "folds.csv", index=False)
    (output / "candidate_contract.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(pd.DataFrame(per_seed_rows)[["model", "seed", "AUC", "AP", "Brier", "BSS"]].to_string(index=False))
    return 0


def bootstrap_pair(patient: pd.DataFrame, *, n_resamples: int, seed: int) -> dict[str, Any]:
    y = patient["true_label"].to_numpy(dtype=int)
    pa = patient[USER_MODEL].to_numpy(dtype=float)
    pb = patient[ASSISTANT_MODEL].to_numpy(dtype=float)
    rng = np.random.default_rng(int(seed))
    auc_a: list[float] = []
    auc_b: list[float] = []
    delta: list[float] = []
    n = len(y)
    for _ in range(int(n_resamples)):
        idx = rng.integers(0, n, size=n)
        yy = y[idx]
        if np.unique(yy).size < 2:
            continue
        aa = float(roc_auc_score(yy, pa[idx]))
        ab = float(roc_auc_score(yy, pb[idx]))
        auc_a.append(aa)
        auc_b.append(ab)
        delta.append(ab - aa)

    def ci(values: list[float]) -> dict[str, float | int]:
        arr = np.asarray(values, dtype=float)
        return {
            "lower": float(np.percentile(arr, 2.5)),
            "upper": float(np.percentile(arr, 97.5)),
            "n_valid": int(arr.size),
        }

    return {
        USER_MODEL: ci(auc_a),
        ASSISTANT_MODEL: ci(auc_b),
        "delta_ASSISTANT_minus_USER": ci(delta),
        "n_resamples": int(n_resamples),
        "seed": int(seed),
    }


def command_aggregate(args: argparse.Namespace) -> int:
    root = Path(args.input_root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    per_seed_paths = sorted(root.rglob("per_seed.csv"))
    oof_paths = sorted(root.rglob("oof.parquet"))
    contract_paths = sorted(root.rglob("candidate_contract.json"))
    if len(per_seed_paths) != 10 or len(oof_paths) != 10:
        raise RuntimeError(
            f"expected 10 batches, found {len(per_seed_paths)} metric files and {len(oof_paths)} OOF files"
        )

    per_seed = pd.concat([pd.read_csv(path) for path in per_seed_paths], ignore_index=True)
    oof = pd.concat([pd.read_parquet(path) for path in oof_paths], ignore_index=True)
    contract = json.loads(contract_paths[0].read_text(encoding="utf-8"))

    expected_seeds = set(range(100))
    for model in MODELS:
        seeds = set(per_seed.loc[per_seed["model"].eq(model), "seed"].astype(int))
        if seeds != expected_seeds:
            raise RuntimeError(f"{model} seed coverage is not exactly 0..99")
    if len(per_seed) != 200:
        raise RuntimeError(f"expected 200 model-seed rows, got {len(per_seed)}")

    summary: dict[str, Any] = {
        "cohort": {
            key: contract[key]
            for key in ("patient_count", "positive_count", "negative_count")
        },
        "models": contract["candidate_models"],
        "cv": {
            "endpoint_days": ENDPOINT_DAYS,
            "outer_folds": OUTER_FOLDS,
            "inner_folds": INNER_FOLDS,
            "seeds": "0..99",
            "selection_metric": "average_precision",
            "model": "elastic-net logistic regression",
            "preprocessing": "training-fold median imputation + missing indicators + standardization",
        },
        "per_seed": {},
    }

    for model in MODELS:
        subset = per_seed.loc[per_seed["model"].eq(model)].sort_values("seed")
        summary["per_seed"][model] = {
            metric: _distribution(subset[metric])
            for metric in ("AUC", "AP", "Brier", "BSS")
        }

    user_seed = per_seed.loc[per_seed["model"].eq(USER_MODEL)].set_index("seed").sort_index()
    assistant_seed = per_seed.loc[per_seed["model"].eq(ASSISTANT_MODEL)].set_index("seed").sort_index()
    delta_auc = assistant_seed["AUC"] - user_seed["AUC"]
    delta_ap = assistant_seed["AP"] - user_seed["AP"]
    delta_brier = assistant_seed["Brier"] - user_seed["Brier"]
    summary["paired_seed_difference"] = {
        "AUC_ASSISTANT_minus_USER": _distribution(delta_auc),
        "AP_ASSISTANT_minus_USER": _distribution(delta_ap),
        "Brier_ASSISTANT_minus_USER": _distribution(delta_brier),
        "assistant_auc_higher": int((delta_auc > 0).sum()),
        "assistant_auc_equal": int((delta_auc == 0).sum()),
        "assistant_auc_lower": int((delta_auc < 0).sum()),
    }

    patient = (
        oof.groupby(["patient_id", "true_label", "model"], as_index=False)["probability"]
        .mean()
        .pivot(index=["patient_id", "true_label"], columns="model", values="probability")
        .reset_index()
    )
    for model in MODELS:
        if model not in patient.columns:
            raise RuntimeError(f"missing patient-level predictions for {model}")

    y = patient["true_label"].to_numpy(dtype=int)
    patient_metrics = {
        model: metric_row(y, patient[model].to_numpy(dtype=float))
        for model in MODELS
    }
    patient_metrics["delta_AUC_ASSISTANT_minus_USER"] = float(
        patient_metrics[ASSISTANT_MODEL]["AUC"] - patient_metrics[USER_MODEL]["AUC"]
    )
    summary["patient_mean_oof"] = patient_metrics
    summary["bootstrap_patient_mean_oof"] = bootstrap_pair(
        patient,
        n_resamples=int(args.bootstrap_resamples),
        seed=int(args.bootstrap_seed),
    )
    summary["target_AUC_0_70"] = {
        model: {
            "per_seed_median_reaches": bool(summary["per_seed"][model]["AUC"]["median"] >= 0.70),
            "patient_mean_oof_reaches": bool(patient_metrics[model]["AUC"] >= 0.70),
        }
        for model in MODELS
    }

    per_seed.to_csv(output / "compact_candidates_per_seed.csv", index=False)
    patient.to_csv(output / "compact_candidates_patient_mean_oof.csv", index=False)
    (output / "compact_candidates_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    ua = summary["per_seed"][USER_MODEL]["AUC"]
    aa = summary["per_seed"][ASSISTANT_MODEL]["AUC"]
    pm = summary["patient_mean_oof"]
    boot = summary["bootstrap_patient_mean_oof"]
    paired = summary["paired_seed_difference"]["AUC_ASSISTANT_minus_USER"]

    lines = [
        "# Two compact-model validations",
        "",
        f"Cohort: {summary['cohort']['patient_count']} patients; "
        f"{summary['cohort']['positive_count']} SCD / {summary['cohort']['negative_count']} non-SCD.",
        "",
        "## Definitions",
        "",
        f"- {USER_MODEL}: 20 raw inputs; original P2 minus standardized-signal-SD pair and beat-count pair; AF + raw PVC retained.",
        f"- {ASSISTANT_MODEL}: 23 raw inputs; same compact ECG set + AF + log1p(PVC) + Age + LVEF + NYHA III indicator.",
        "",
        "## 100 repeated nested-CV",
        "",
        "| Model | Inputs | AUC median | AUC 2.5-97.5% split sensitivity | AP median | Brier median | BSS median |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| {USER_MODEL} | 20 | {ua['median']:.6f} | {ua['p2_5']:.6f}-{ua['p97_5']:.6f} | "
        f"{summary['per_seed'][USER_MODEL]['AP']['median']:.6f} | "
        f"{summary['per_seed'][USER_MODEL]['Brier']['median']:.6f} | "
        f"{summary['per_seed'][USER_MODEL]['BSS']['median']:.6f} |",
        f"| {ASSISTANT_MODEL} | 23 | {aa['median']:.6f} | {aa['p2_5']:.6f}-{aa['p97_5']:.6f} | "
        f"{summary['per_seed'][ASSISTANT_MODEL]['AP']['median']:.6f} | "
        f"{summary['per_seed'][ASSISTANT_MODEL]['Brier']['median']:.6f} | "
        f"{summary['per_seed'][ASSISTANT_MODEL]['BSS']['median']:.6f} |",
        "",
        "## Paired seed comparison",
        "",
        f"- Median AUC difference (assistant - user): {paired['median']:+.6f}",
        f"- AUC higher/equal/lower for assistant candidate: "
        f"{summary['paired_seed_difference']['assistant_auc_higher']}/"
        f"{summary['paired_seed_difference']['assistant_auc_equal']}/"
        f"{summary['paired_seed_difference']['assistant_auc_lower']}",
        "",
        "## Patient-level mean OOF",
        "",
        f"- {USER_MODEL} AUC: {pm[USER_MODEL]['AUC']:.6f} "
        f"(fixed-prediction bootstrap 95% {boot[USER_MODEL]['lower']:.6f}-{boot[USER_MODEL]['upper']:.6f})",
        f"- {ASSISTANT_MODEL} AUC: {pm[ASSISTANT_MODEL]['AUC']:.6f} "
        f"(fixed-prediction bootstrap 95% {boot[ASSISTANT_MODEL]['lower']:.6f}-{boot[ASSISTANT_MODEL]['upper']:.6f})",
        f"- Patient-level AUC difference (assistant - user): {pm['delta_AUC_ASSISTANT_minus_USER']:+.6f} "
        f"(paired fixed-prediction bootstrap 95% "
        f"{boot['delta_ASSISTANT_minus_USER']['lower']:+.6f} to "
        f"{boot['delta_ASSISTANT_minus_USER']['upper']:+.6f})",
        "",
        "## AUC 0.70 checks",
        "",
        f"- {USER_MODEL}: repeated-CV median >=0.70 = {summary['target_AUC_0_70'][USER_MODEL]['per_seed_median_reaches']}; "
        f"patient-mean OOF >=0.70 = {summary['target_AUC_0_70'][USER_MODEL]['patient_mean_oof_reaches']}",
        f"- {ASSISTANT_MODEL}: repeated-CV median >=0.70 = {summary['target_AUC_0_70'][ASSISTANT_MODEL]['per_seed_median_reaches']}; "
        f"patient-mean OOF >=0.70 = {summary['target_AUC_0_70'][ASSISTANT_MODEL]['patient_mean_oof_reaches']}",
    ]
    (output / "compact_candidates_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate two compact SCD model candidates")
    sub = parser.add_subparsers(dest="command", required=True)

    batch = sub.add_parser("batch")
    batch.add_argument("--seed-start", type=int, required=True)
    batch.add_argument("--seed-stop", type=int, required=True)
    batch.add_argument("--output-dir", type=Path, required=True)

    aggregate = sub.add_parser("aggregate")
    aggregate.add_argument("--input-root", type=Path, required=True)
    aggregate.add_argument("--output-dir", type=Path, required=True)
    aggregate.add_argument("--bootstrap-resamples", type=int, default=2000)
    aggregate.add_argument("--bootstrap-seed", type=int, default=20260927)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "batch":
        return command_batch(args)
    if args.command == "aggregate":
        return command_aggregate(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
