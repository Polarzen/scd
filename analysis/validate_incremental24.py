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

from analysis.validate_compact_candidates import (
    ASSISTANT_MODEL,
    build_candidate_frame,
)
from analysis.validate_p3 import (
    ENDPOINT_DAYS,
    INNER_FOLDS,
    OUTER_FOLDS,
    _distribution,
    metric_row,
    validate_one_seed,
)

BASE = "COMPACT23"
QRS = "COMPACT24_QRS"
NSVT = "COMPACT24_NSVT"
BNP = "COMPACT24_LOG_PRO_BNP"
BNP_NSVT_NO_AGE = "COMPACT24_BNP_NSVT_NO_AGE"
BNP_NSVT_NO_AGE_NYHA = "COMPACT23_BNP_NSVT_NO_AGE_NYHA"
MODELS = (BASE, QRS, NSVT, BNP, BNP_NSVT_NO_AGE, BNP_NSVT_NO_AGE_NYHA)

SUBJECTS_PATH = REPO_ROOT / "data" / "cohort" / "subjects.parquet"


def _num(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip().str.strip("'").str.strip('"')
    text = text.replace({"": pd.NA, "NA": pd.NA, "N/A": pd.NA, "<NA>": pd.NA, "None": pd.NA})
    text = text.str.replace(",", ".", regex=False)
    return pd.to_numeric(text, errors="coerce").astype("float64")


def _binary(series: pd.Series) -> pd.Series:
    values = _num(series)
    return values.where(values.isin([0.0, 1.0]), np.nan)


def build_incremental_frame() -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, Any]]:
    frame, candidate_features, meta = build_candidate_frame()
    base_cols = list(candidate_features[ASSISTANT_MODEL])

    subjects = pd.read_parquet(SUBJECTS_PATH)
    subjects["patient_id"] = subjects["patient_id"].astype("string")
    required = {
        "patient_id",
        "QRS duration (ms)",
        "Non-sustained ventricular tachycardia (CH>10)",
        "Pro-BNP (ng/L)",
    }
    missing = sorted(required - set(subjects.columns))
    if missing:
        raise RuntimeError(f"subjects table is missing incremental variables: {missing}")

    extra = subjects.loc[:, list(required)].copy()
    qrs = _num(extra["QRS duration (ms)"])
    extra["qrs_ms"] = qrs.where(qrs.between(40.0, 250.0), np.nan)
    extra["nsvt"] = _binary(extra["Non-sustained ventricular tachycardia (CH>10)"])
    pro_bnp = _num(extra["Pro-BNP (ng/L)"]).where(lambda x: x >= 0)
    extra["log_pro_bnp"] = np.log1p(pro_bnp)
    extra = extra.drop(
        columns=[
            "QRS duration (ms)",
            "Non-sustained ventricular tachycardia (CH>10)",
            "Pro-BNP (ng/L)",
        ]
    )

    frame = frame.merge(extra, on="patient_id", how="left", validate="one_to_one")
    no_age = [column for column in base_cols if column != "age"]
    no_age_nyha = [column for column in no_age if column != "nyha_III"]
    if len(no_age) != len(base_cols) - 1 or len(no_age_nyha) != len(base_cols) - 2:
        raise RuntimeError("expected age and nyha_III in COMPACT23 baseline")

    model_features = {
        BASE: base_cols,
        QRS: base_cols + ["qrs_ms"],
        NSVT: base_cols + ["nsvt"],
        BNP: base_cols + ["log_pro_bnp"],
        BNP_NSVT_NO_AGE: no_age + ["log_pro_bnp", "nsvt"],
        BNP_NSVT_NO_AGE_NYHA: no_age_nyha + ["log_pro_bnp", "nsvt"],
    }
    info = {
        "cohort": {
            "patient_count": int(len(frame)),
            "positive_count": int(frame["label"].sum()),
            "negative_count": int(len(frame) - frame["label"].sum()),
        },
        "models": {
            name: {
                "raw_feature_count": len(cols),
                "features": cols,
                "missing_counts": {col: int(frame[col].isna().sum()) for col in cols},
            }
            for name, cols in model_features.items()
        },
    }
    return frame, model_features, info


def command_batch(args: argparse.Namespace) -> int:
    frame, model_features, info = build_incremental_frame()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    oofs: list[pd.DataFrame] = []
    folds: list[pd.DataFrame] = []
    for seed in range(args.seed_start, args.seed_stop + 1):
        for name in MODELS:
            summary, oof, fold = validate_one_seed(
                frame,
                model_features[name],
                model_name=name,
                seed=seed,
            )
            rows.append(summary)
            oofs.append(oof)
            folds.append(fold)
    pd.DataFrame(rows).to_csv(out / "per_seed.csv", index=False)
    pd.concat(oofs, ignore_index=True).to_parquet(out / "oof.parquet", index=False, compression="zstd")
    pd.concat(folds, ignore_index=True).to_csv(out / "folds.csv", index=False)
    (out / "contract.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(pd.DataFrame(rows)[["model", "seed", "AUC", "AP", "Brier", "BSS"]].to_string(index=False))
    return 0


def _bootstrap(patient: pd.DataFrame, *, n_resamples: int, seed: int) -> dict[str, Any]:
    y = patient["true_label"].to_numpy(int)
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {m: [] for m in MODELS}
    deltas: dict[str, list[float]] = {m: [] for m in MODELS if m != BASE}
    n = len(y)
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        yy = y[idx]
        if np.unique(yy).size < 2:
            continue
        aucs = {}
        for m in MODELS:
            aucs[m] = float(roc_auc_score(yy, patient[m].to_numpy(float)[idx]))
            values[m].append(aucs[m])
        for m in MODELS:
            if m != BASE:
                deltas[m].append(aucs[m] - aucs[BASE])

    def ci(arr: list[float]) -> dict[str, float | int]:
        x = np.asarray(arr, dtype=float)
        return {
            "lower": float(np.percentile(x, 2.5)),
            "upper": float(np.percentile(x, 97.5)),
            "n_valid": int(x.size),
        }

    return {
        "auc": {m: ci(v) for m, v in values.items()},
        "delta_vs_base": {m: ci(v) for m, v in deltas.items()},
        "n_resamples": n_resamples,
        "seed": seed,
    }


def command_aggregate(args: argparse.Namespace) -> int:
    root = Path(args.input_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    per_paths = sorted(root.rglob("per_seed.csv"))
    oof_paths = sorted(root.rglob("oof.parquet"))
    contract_paths = sorted(root.rglob("contract.json"))
    if len(per_paths) != 10 or len(oof_paths) != 10:
        raise RuntimeError(f"expected 10 batches, found {len(per_paths)} metrics and {len(oof_paths)} OOF")

    per = pd.concat([pd.read_csv(p) for p in per_paths], ignore_index=True)
    oof = pd.concat([pd.read_parquet(p) for p in oof_paths], ignore_index=True)
    expected = set(range(100))
    for m in MODELS:
        got = set(per.loc[per["model"].eq(m), "seed"].astype(int))
        if got != expected:
            raise RuntimeError(f"{m} seed coverage mismatch")

    patient = (
        oof.groupby(["patient_id", "true_label", "model"], as_index=False)["probability"]
        .mean()
        .pivot(index=["patient_id", "true_label"], columns="model", values="probability")
        .reset_index()
    )
    y = patient["true_label"].to_numpy(int)

    summary: dict[str, Any] = {
        "contract": json.loads(contract_paths[0].read_text(encoding="utf-8")),
        "cv": {
            "endpoint_days": ENDPOINT_DAYS,
            "outer_folds": OUTER_FOLDS,
            "inner_folds": INNER_FOLDS,
            "seeds": "0..99",
            "selection_metric": "average_precision",
        },
        "per_seed": {},
        "patient_mean_oof": {},
    }
    for m in MODELS:
        subset = per.loc[per["model"].eq(m)].sort_values("seed")
        summary["per_seed"][m] = {metric: _distribution(subset[metric]) for metric in ("AUC", "AP", "Brier", "BSS")}
        summary["patient_mean_oof"][m] = metric_row(y, patient[m].to_numpy(float))

    summary["bootstrap"] = _bootstrap(patient, n_resamples=args.bootstrap_resamples, seed=args.bootstrap_seed)
    summary["target_0_70"] = {
        m: {
            "median_auc": summary["per_seed"][m]["AUC"]["median"],
            "patient_auc": summary["patient_mean_oof"][m]["AUC"],
            "median_reaches": bool(summary["per_seed"][m]["AUC"]["median"] >= 0.70),
            "patient_reaches": bool(summary["patient_mean_oof"][m]["AUC"] >= 0.70),
        }
        for m in MODELS
    }

    per.to_csv(out / "incremental24_per_seed.csv", index=False)
    patient.to_csv(out / "incremental24_patient_mean_oof.csv", index=False)
    (out / "incremental24_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Compact23 incremental 24-input validation",
        "",
        "| Model | Inputs | AUC median | AP median | Brier median | Patient-mean OOF AUC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for m in MODELS:
        lines.append(
            f"| {m} | {summary['contract']['models'][m]['raw_feature_count']} | "
            f"{summary['per_seed'][m]['AUC']['median']:.6f} | "
            f"{summary['per_seed'][m]['AP']['median']:.6f} | "
            f"{summary['per_seed'][m]['Brier']['median']:.6f} | "
            f"{summary['patient_mean_oof'][m]['AUC']:.6f} |"
        )
    lines += [
        "",
        f"QRS missing: {summary['contract']['models'][QRS]['missing_counts'].get('qrs_ms')}",
        f"NSVT missing: {summary['contract']['models'][NSVT]['missing_counts'].get('nsvt')}",
        f"log(Pro-BNP+1) missing: {summary['contract']['models'][BNP]['missing_counts'].get('log_pro_bnp')}",
    ]
    (out / "incremental24_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("batch")
    b.add_argument("--seed-start", type=int, required=True)
    b.add_argument("--seed-stop", type=int, required=True)
    b.add_argument("--output-dir", type=Path, required=True)
    a = sub.add_parser("aggregate")
    a.add_argument("--input-root", type=Path, required=True)
    a.add_argument("--output-dir", type=Path, required=True)
    a.add_argument("--bootstrap-resamples", type=int, default=2000)
    a.add_argument("--bootstrap-seed", type=int, default=20260927)
    return p


def main() -> int:
    args = parser().parse_args()
    if args.command == "batch":
        return command_batch(args)
    return command_aggregate(args)


if __name__ == "__main__":
    raise SystemExit(main())
