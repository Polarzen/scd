from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_compact_candidates import ASSISTANT_MODEL, build_candidate_frame
from scripts.validate_p3 import validate_one_seed

SUBJECTS = ROOT / "data" / "cohort" / "subjects.parquet"

CANDIDATES = {
    "QRS": ("QRS duration (ms)", "qrs"),
    "QTC": ("QT corrected ", "qtc"),
    "PRIOR_MI": ("Prior Myocardial Infarction (yes=1)", "binary"),
    "SYNCOPE": ("Syncope", "syncope"),
    "VT_ANY": ("Ventricular Tachycardia", "vt"),
    "NSVT_GT10": ("Non-sustained ventricular tachycardia (CH>10)", "binary"),
    "SEX_MALE": ("Gender (male=1)", "binary"),
    "LOG_PROBNP": ("Pro-BNP (ng/L)", "log"),
    "CREATININE": ("Creatinine (?mol/L)", "positive"),
    "SODIUM": ("Sodium (mEq/L)", "sodium"),
}

def num(s):
    s = s.astype("string").str.strip().str.strip("'").str.strip('"')
    s = s.replace({"": pd.NA, "NA": pd.NA, "N/A": pd.NA, "<NA>": pd.NA, "None": pd.NA})
    return pd.to_numeric(s.str.replace(",", ".", regex=False), errors="coerce").astype(float)

def transform(s, kind):
    x = num(s)
    if kind == "qrs": return x.where(x.between(40, 300))
    if kind == "qtc": return x.where(x.between(250, 700))
    if kind == "binary": return x.where(x.isin([0, 1]))
    if kind == "syncope":
        x = x.where(x.isin([0, 1, 2]))
        return pd.Series(np.where(x.isna(), np.nan, (x > 0).astype(float)), index=x.index)
    if kind == "vt":
        x = x.where(x.isin([0, 1, 2, 3]))
        return pd.Series(np.where(x.isna(), np.nan, (x > 0).astype(float)), index=x.index)
    if kind == "log": return np.log1p(x.where(x >= 0))
    if kind == "positive": return x.where(x > 0)
    if kind == "sodium": return x.where(x.between(100, 180))
    raise ValueError(kind)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--seed-stop", type=int, default=19)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    candidate = args.candidate.upper()
    frame, features, meta = build_candidate_frame()
    base_cols = list(features[ASSISTANT_MODEL])

    if candidate == "BASE":
        model = "COMPACT23_BASE"
        cols = base_cols
        audit = {"candidate": "BASE", "missing": 0, "missing_fraction": 0.0}
    else:
        if candidate not in CANDIDATES:
            raise ValueError(candidate)
        source, kind = CANDIDATES[candidate]
        subjects = pd.read_parquet(SUBJECTS)
        if source not in subjects.columns:
            raise ValueError(f"missing source column: {source}")
        extra = pd.DataFrame({
            "patient_id": subjects["patient_id"].astype("string"),
            "candidate_value": transform(subjects[source], kind),
        })
        frame = frame.merge(extra, on="patient_id", how="left", validate="one_to_one")
        cols = base_cols + ["candidate_value"]
        model = f"COMPACT24_{candidate}"
        audit = {
            "candidate": candidate,
            "source": source,
            "kind": kind,
            "missing": int(frame["candidate_value"].isna().sum()),
            "missing_fraction": float(frame["candidate_value"].isna().mean()),
            "unique_observed": int(frame["candidate_value"].dropna().nunique()),
        }

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    rows, oofs = [], []
    for seed in range(args.seed_start, args.seed_stop + 1):
        summary, oof, _ = validate_one_seed(frame, cols, model_name=model, seed=seed)
        rows.append(summary)
        oofs.append(oof)

    pd.DataFrame(rows).to_csv(out / "per_seed.csv", index=False)
    pd.concat(oofs, ignore_index=True).to_parquet(out / "oof.parquet", index=False, compression="zstd")
    (out / "audit.json").write_text(json.dumps({
        **audit,
        "model": model,
        "feature_count": len(cols),
        "seed_start": args.seed_start,
        "seed_stop": args.seed_stop,
    }, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(rows)[["model","seed","AUC","AP","Brier","BSS"]].to_string(index=False))

if __name__ == "__main__":
    main()
