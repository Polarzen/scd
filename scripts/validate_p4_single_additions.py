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

from scripts.validate_compact_candidates import ASSISTANT_MODEL, build_candidate_frame
from scripts.validate_p3 import SUBJECTS_PATH, _distribution, metric_row, validate_one_seed

CANDIDATES = {
    "qrs_duration": "QRS duration (ms)",
    "nsvt": "Non-sustained ventricular tachycardia (CH>10)",
    "qtc": "QT corrected ",
    "male": "Gender (male=1)",
    "prior_mi": "Prior Myocardial Infarction (yes=1)",
    "syncope": "Syncope",
    "ventricular_tachycardia": "Ventricular Tachycardia",
    "log_pro_bnp": "Pro-BNP (ng/L)",
    "log_creatinine": "Creatinine (?mol/L)",
    "sodium": "Sodium (mEq/L)",
}


def _num(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip().str.strip("'").str.strip('"')
    text = text.replace({"": pd.NA, "NA": pd.NA, "N/A": pd.NA, "<NA>": pd.NA, "None": pd.NA})
    text = text.str.replace(",", ".", regex=False)
    return pd.to_numeric(text, errors="coerce").astype("float64")


def _binary(series: pd.Series, valid: set[float], positive: set[float]) -> pd.Series:
    x = _num(series)
    return pd.Series(
        np.where(x.isin(valid), x.isin(positive).astype(float), np.nan),
        index=series.index,
        dtype="float64",
    )


def candidate_value(name: str, subjects: pd.DataFrame) -> pd.Series:
    source = CANDIDATES[name]
    raw = subjects[source]
    if name == "qrs_duration":
        x = _num(raw)
        return x.where(x.between(40, 250))
    if name == "qtc":
        x = _num(raw)
        return x.where(x.between(250, 700))
    if name in {"male", "prior_mi", "nsvt"}:
        return _binary(raw, {0.0, 1.0}, {1.0})
    if name == "syncope":
        return _binary(raw, {0.0, 1.0, 2.0}, {1.0, 2.0})
    if name == "ventricular_tachycardia":
        return _binary(raw, {0.0, 1.0, 2.0, 3.0}, {1.0, 2.0, 3.0})
    if name == "log_pro_bnp":
        x = _num(raw).where(lambda s: s >= 0)
        return np.log1p(x)
    if name == "log_creatinine":
        x = _num(raw).where(lambda s: s > 0)
        return np.log(x)
    if name == "sodium":
        x = _num(raw)
        return x.where(x.between(110, 170))
    raise KeyError(name)


def build_candidate(name: str) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    if name not in CANDIDATES:
        raise ValueError(f"unknown candidate {name!r}; choose from {sorted(CANDIDATES)}")

    frame, model_features, meta = build_candidate_frame()
    subjects = pd.read_parquet(SUBJECTS_PATH)
    subjects = subjects.loc[:, ["patient_id", CANDIDATES[name]]].copy()
    subjects["patient_id"] = subjects["patient_id"].astype("string")
    subjects[name] = candidate_value(name, subjects)

    frame = frame.merge(subjects[["patient_id", name]], on="patient_id", how="left", validate="one_to_one")
    base = list(model_features[ASSISTANT_MODEL])
    cols = base + [name]
    if len(cols) != 24:
        raise RuntimeError(f"candidate model must have 24 raw inputs, got {len(cols)}")

    detail = {
        "candidate": name,
        "source_field": CANDIDATES[name],
        "raw_feature_count": len(cols),
        "missing_count": int(frame[name].isna().sum()),
        "nonmissing_count": int(frame[name].notna().sum()),
        "features": cols,
        "base_model": ASSISTANT_MODEL,
        "base_raw_feature_count": len(base),
        "patient_count": int(len(frame)),
        "positive_count": int(frame["label"].sum()),
        "negative_count": int((frame["label"] == 0).sum()),
    }
    return frame, cols, detail


def command_run(args: argparse.Namespace) -> int:
    frame, cols, detail = build_candidate(args.candidate)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    oofs: list[pd.DataFrame] = []
    folds: list[pd.DataFrame] = []
    model_name = f"P4_{args.candidate.upper()}"

    for seed in range(int(args.seed_start), int(args.seed_stop) + 1):
        summary, oof, fold = validate_one_seed(
            frame,
            cols,
            model_name=model_name,
            seed=seed,
        )
        summaries.append(summary)
        oofs.append(oof)
        folds.append(fold)

    pd.DataFrame(summaries).to_csv(output / "per_seed.csv", index=False)
    pd.concat(oofs, ignore_index=True).to_parquet(output / "oof.parquet", index=False, compression="zstd")
    pd.concat(folds, ignore_index=True).to_csv(output / "folds.csv", index=False)
    (output / "candidate.json").write_text(
        json.dumps(detail, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


def command_summarize(args: argparse.Namespace) -> int:
    root = Path(args.input_root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    candidate_files = sorted(root.rglob("candidate.json"))
    rows: list[dict[str, Any]] = []
    all_oof: list[pd.DataFrame] = []

    for meta_path in candidate_files:
        folder = meta_path.parent
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        per_seed = pd.read_csv(folder / "per_seed.csv")
        oof = pd.read_parquet(folder / "oof.parquet")
        candidate = meta["candidate"]

        if set(per_seed["seed"].astype(int)) != set(range(100)):
            raise RuntimeError(f"{candidate} does not contain seeds 0..99")

        patient = (
            oof.groupby(["patient_id", "true_label"], as_index=False)["probability"]
            .mean()
        )
        metrics = metric_row(
            patient["true_label"].to_numpy(dtype=int),
            patient["probability"].to_numpy(dtype=float),
        )
        auc_dist = _distribution(per_seed["AUC"])
        ap_dist = _distribution(per_seed["AP"])
        brier_dist = _distribution(per_seed["Brier"])
        rows.append(
            {
                "candidate": candidate,
                "source_field": meta["source_field"],
                "missing_count": meta["missing_count"],
                "raw_feature_count": meta["raw_feature_count"],
                "cv_auc_median": auc_dist["median"],
                "cv_auc_p2_5": auc_dist["p2_5"],
                "cv_auc_p97_5": auc_dist["p97_5"],
                "cv_ap_median": ap_dist["median"],
                "cv_brier_median": brier_dist["median"],
                "patient_auc": metrics["AUC"],
                "patient_ap": metrics["AP"],
                "patient_brier": metrics["Brier"],
                "patient_bss": metrics["BSS"],
            }
        )
        patient["candidate"] = candidate
        all_oof.append(patient)

    table = pd.DataFrame(rows).sort_values(
        ["patient_auc", "cv_auc_median", "patient_ap"],
        ascending=[False, False, False],
        kind="stable",
    ).reset_index(drop=True)
    table["patient_auc_ge_0_70"] = table["patient_auc"] >= 0.70
    table["cv_auc_median_ge_0_70"] = table["cv_auc_median"] >= 0.70

    table.to_csv(output / "single_addition_ranking.csv", index=False)
    pd.concat(all_oof, ignore_index=True).to_parquet(output / "single_addition_patient_oof.parquet", index=False, compression="zstd")

    best = table.iloc[0].to_dict()
    (output / "best_candidate.json").write_text(
        json.dumps(best, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Compact23 + one-variable screening",
        "",
        "All models use 24 raw inputs and the same 878-patient / 37-event cohort.",
        "",
        "| Candidate | Missing | CV AUC median | Patient-mean OOF AUC | AP | Brier |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in table.to_dict("records"):
        lines.append(
            f"| {row['candidate']} | {row['missing_count']} | {row['cv_auc_median']:.6f} | "
            f"{row['patient_auc']:.6f} | {row['patient_ap']:.6f} | {row['patient_brier']:.6f} |"
        )
    lines.extend(
        [
            "",
            f"Best candidate: **{best['candidate']}**",
            f"Patient-mean OOF AUC: **{best['patient_auc']:.6f}**",
            f"100-seed nested-CV median AUC: **{best['cv_auc_median']:.6f}**",
            f"Patient AUC >= 0.70: **{bool(best['patient_auc_ge_0_70'])}**",
            f"CV median AUC >= 0.70: **{bool(best['cv_auc_median_ge_0_70'])}**",
        ]
    )
    (output / "single_addition_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run")
    run.add_argument("--candidate", required=True, choices=sorted(CANDIDATES))
    run.add_argument("--seed-start", type=int, default=0)
    run.add_argument("--seed-stop", type=int, default=99)
    run.add_argument("--output-dir", type=Path, required=True)

    summarize = sub.add_parser("summarize")
    summarize.add_argument("--input-root", type=Path, required=True)
    summarize.add_argument("--output-dir", type=Path, required=True)
    return p


def main() -> int:
    args = parser().parse_args()
    if args.command == "run":
        return command_run(args)
    return command_summarize(args)


if __name__ == "__main__":
    raise SystemExit(main())
