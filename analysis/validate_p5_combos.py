from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from analysis.validate_compact_candidates import ASSISTANT_MODEL, build_candidate_frame
from analysis.validate_p3 import SUBJECTS_PATH, _distribution, metric_row, validate_one_seed
from analysis.validate_p4_single_additions import CANDIDATES, candidate_value

# Keep the final raw input count at 24. Compact23 + two new variables would
# be 25, so drop sig_mean_median, a baseline-centered signal-location summary.
DROP_FROM_COMPACT = "sig_mean_median"

COMBOS: dict[str, tuple[str, str]] = {
    "pro_bnp_nsvt": ("log_pro_bnp", "nsvt"),
    "pro_bnp_qrs": ("log_pro_bnp", "qrs_duration"),
    "pro_bnp_qtc": ("log_pro_bnp", "qtc"),
    "pro_bnp_vt": ("log_pro_bnp", "ventricular_tachycardia"),
}


def build_combo(name: str) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    if name not in COMBOS:
        raise ValueError(f"unknown combo {name}")
    frame, model_features, meta = build_candidate_frame()
    base = [c for c in model_features[ASSISTANT_MODEL] if c != DROP_FROM_COMPACT]
    if len(base) != 22:
        raise RuntimeError(f"expected 22 features after one deterministic removal, got {len(base)}")

    subjects = pd.read_parquet(SUBJECTS_PATH)
    subjects["patient_id"] = subjects["patient_id"].astype("string")
    additions: list[str] = []
    for candidate in COMBOS[name]:
        subjects[candidate] = candidate_value(candidate, subjects)
        additions.append(candidate)

    frame = frame.merge(
        subjects[["patient_id", *additions]],
        on="patient_id",
        how="left",
        validate="one_to_one",
    )
    cols = base + additions
    if len(cols) != 24:
        raise RuntimeError(f"combo {name} must have 24 raw inputs, got {len(cols)}")

    detail = {
        "combo": name,
        "raw_feature_count": len(cols),
        "base_model": ASSISTANT_MODEL,
        "dropped_from_compact23": DROP_FROM_COMPACT,
        "added_candidates": additions,
        "source_fields": {candidate: CANDIDATES[candidate] for candidate in additions},
        "missing_counts": {candidate: int(frame[candidate].isna().sum()) for candidate in additions},
        "features": cols,
        "patient_count": int(len(frame)),
        "positive_count": int(frame["label"].sum()),
        "negative_count": int((frame["label"] == 0).sum()),
    }
    return frame, cols, detail


def command_run(args: argparse.Namespace) -> int:
    frame, cols, detail = build_combo(args.combo)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_name = f"P5_{args.combo.upper()}"

    rows: list[dict[str, Any]] = []
    oofs: list[pd.DataFrame] = []
    folds: list[pd.DataFrame] = []
    for seed in range(int(args.seed_start), int(args.seed_stop) + 1):
        summary, oof, fold = validate_one_seed(frame, cols, model_name=model_name, seed=seed)
        rows.append(summary)
        oofs.append(oof)
        folds.append(fold)

    pd.DataFrame(rows).to_csv(out / "per_seed.csv", index=False)
    pd.concat(oofs, ignore_index=True).to_parquet(out / "oof.parquet", index=False, compression="zstd")
    pd.concat(folds, ignore_index=True).to_csv(out / "folds.csv", index=False)
    (out / "combo.json").write_text(
        json.dumps(detail, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


def command_summarize(args: argparse.Namespace) -> int:
    root = Path(args.input_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for meta_path in sorted(root.rglob("combo.json")):
        folder = meta_path.parent
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        per_seed = pd.read_csv(folder / "per_seed.csv")
        oof = pd.read_parquet(folder / "oof.parquet")
        if set(per_seed["seed"].astype(int)) != set(range(100)):
            raise RuntimeError(f"{meta['combo']} missing seeds")
        patient = oof.groupby(["patient_id", "true_label"], as_index=False)["probability"].mean()
        metrics = metric_row(
            patient["true_label"].to_numpy(dtype=int),
            patient["probability"].to_numpy(dtype=float),
        )
        rows.append(
            {
                "combo": meta["combo"],
                "raw_feature_count": meta["raw_feature_count"],
                "cv_auc_median": _distribution(per_seed["AUC"])["median"],
                "cv_auc_p2_5": _distribution(per_seed["AUC"])["p2_5"],
                "cv_auc_p97_5": _distribution(per_seed["AUC"])["p97_5"],
                "cv_ap_median": _distribution(per_seed["AP"])["median"],
                "cv_brier_median": _distribution(per_seed["Brier"])["median"],
                "patient_auc": metrics["AUC"],
                "patient_ap": metrics["AP"],
                "patient_brier": metrics["Brier"],
                "patient_bss": metrics["BSS"],
            }
        )

    table = pd.DataFrame(rows).sort_values(
        ["patient_auc", "cv_auc_median", "patient_ap"],
        ascending=[False, False, False],
        kind="stable",
    ).reset_index(drop=True)
    table["patient_auc_ge_0_70"] = table["patient_auc"] >= 0.70
    table["cv_auc_median_ge_0_70"] = table["cv_auc_median"] >= 0.70
    table.to_csv(out / "combo_ranking.csv", index=False)
    best = table.iloc[0].to_dict()
    (out / "best_combo.json").write_text(
        json.dumps(best, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# 24-input combination screening",
        "",
        "Compact23 with sig_mean_median removed, then two clinically distinct variables added.",
        "",
        "| Combo | CV AUC median | Patient-mean OOF AUC | AP | Brier |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in table.to_dict("records"):
        lines.append(
            f"| {row['combo']} | {row['cv_auc_median']:.6f} | {row['patient_auc']:.6f} | "
            f"{row['patient_ap']:.6f} | {row['patient_brier']:.6f} |"
        )
    lines.extend([
        "",
        f"Best: **{best['combo']}**",
        f"CV median AUC: **{best['cv_auc_median']:.6f}**",
        f"Patient-mean OOF AUC: **{best['patient_auc']:.6f}**",
        f"CV median >=0.70: **{bool(best['cv_auc_median_ge_0_70'])}**",
        f"Patient AUC >=0.70: **{bool(best['patient_auc_ge_0_70'])}**",
    ])
    (out / "combo_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--combo", required=True, choices=sorted(COMBOS))
    run.add_argument("--seed-start", type=int, default=0)
    run.add_argument("--seed-stop", type=int, default=99)
    run.add_argument("--output-dir", type=Path, required=True)
    sm = sub.add_parser("summarize")
    sm.add_argument("--input-root", type=Path, required=True)
    sm.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    if args.command == "run":
        return command_run(args)
    return command_summarize(args)


if __name__ == "__main__":
    raise SystemExit(main())
