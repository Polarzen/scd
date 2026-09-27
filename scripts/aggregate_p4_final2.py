from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from scripts.validate_p3 import _distribution, metric_row

MODELS = ("COMPACT23_BASE", "COMPACT24_LOG_PROBNP")

def bootstrap(patient, n, seed):
    y = patient["true_label"].to_numpy(int)
    rng = np.random.default_rng(seed)
    vals = {m: [] for m in MODELS}
    delta = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        yy = y[idx]
        if np.unique(yy).size < 2:
            continue
        aucs = {}
        for m in MODELS:
            aucs[m] = roc_auc_score(yy, patient[m].to_numpy(float)[idx])
            vals[m].append(float(aucs[m]))
        delta.append(float(aucs[MODELS[1]] - aucs[MODELS[0]]))
    def ci(x):
        a=np.asarray(x,float)
        return {"lower":float(np.percentile(a,2.5)),"upper":float(np.percentile(a,97.5)),"n":int(len(a))}
    return {MODELS[0]:ci(vals[MODELS[0]]),MODELS[1]:ci(vals[MODELS[1]]),"delta":ci(delta)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input-root",type=Path,required=True)
    ap.add_argument("--output-dir",type=Path,required=True)
    args=ap.parse_args()
    metric_paths=sorted(args.input_root.rglob("per_seed.csv"))
    oof_paths=sorted(args.input_root.rglob("oof.parquet"))
    if len(metric_paths)!=20 or len(oof_paths)!=20:
        raise RuntimeError(f"expected 20 batch artifacts; got {len(metric_paths)} metrics/{len(oof_paths)} oof")
    metrics=pd.concat([pd.read_csv(p) for p in metric_paths],ignore_index=True)
    oof=pd.concat([pd.read_parquet(p) for p in oof_paths],ignore_index=True)
    summary={"models":{}}
    for m in MODELS:
        d=metrics.loc[metrics.model.eq(m)].sort_values("seed")
        if set(d.seed.astype(int))!=set(range(100)):
            raise RuntimeError(f"{m} missing seeds")
        summary["models"][m]={k:_distribution(d[k]) for k in ("AUC","AP","Brier","BSS")}
    wide=(oof.groupby(["patient_id","true_label","model"],as_index=False).probability.mean()
          .pivot(index=["patient_id","true_label"],columns="model",values="probability").reset_index())
    y=wide.true_label.to_numpy(int)
    summary["patient_mean_oof"]={m:metric_row(y,wide[m].to_numpy(float)) for m in MODELS}
    summary["patient_mean_oof"]["delta_AUC"]=float(summary["patient_mean_oof"][MODELS[1]]["AUC"]-summary["patient_mean_oof"][MODELS[0]]["AUC"])
    summary["bootstrap"]=bootstrap(wide,2000,20260927)
    base=metrics.loc[metrics.model.eq(MODELS[0])].set_index("seed").sort_index()
    cand=metrics.loc[metrics.model.eq(MODELS[1])].set_index("seed").sort_index()
    delta=cand.AUC-base.AUC
    summary["paired_seed_delta_AUC"]=_distribution(delta)
    summary["candidate_higher_equal_lower"]=[int((delta>0).sum()),int((delta==0).sum()),int((delta<0).sum())]
    summary["target_0_70"]={
        "median":bool(summary["models"][MODELS[1]]["AUC"]["median"]>=0.70),
        "patient_mean":bool(summary["patient_mean_oof"][MODELS[1]]["AUC"]>=0.70),
    }
    args.output_dir.mkdir(parents=True,exist_ok=True)
    metrics.to_csv(args.output_dir/"per_seed.csv",index=False)
    wide.to_csv(args.output_dir/"patient_mean_oof.csv",index=False)
    (args.output_dir/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    b=summary["models"][MODELS[0]]["AUC"]; c=summary["models"][MODELS[1]]["AUC"]
    pm=summary["patient_mean_oof"]; boot=summary["bootstrap"]
    lines=[
        "# P4 Pro-BNP full validation","",
        "| Model | AUC median | 2.5-97.5% | AP median | Brier median | BSS median |",
        "|---|---:|---:|---:|---:|---:|",
        f"| {MODELS[0]} | {b['median']:.6f} | {b['p2_5']:.6f}-{b['p97_5']:.6f} | {summary['models'][MODELS[0]]['AP']['median']:.6f} | {summary['models'][MODELS[0]]['Brier']['median']:.6f} | {summary['models'][MODELS[0]]['BSS']['median']:.6f} |",
        f"| {MODELS[1]} | {c['median']:.6f} | {c['p2_5']:.6f}-{c['p97_5']:.6f} | {summary['models'][MODELS[1]]['AP']['median']:.6f} | {summary['models'][MODELS[1]]['Brier']['median']:.6f} | {summary['models'][MODELS[1]]['BSS']['median']:.6f} |",
        "",
        f"- Patient-mean OOF AUC base: {pm[MODELS[0]]['AUC']:.6f}",
        f"- Patient-mean OOF AUC +log(Pro-BNP): {pm[MODELS[1]]['AUC']:.6f}",
        f"- Delta AUC: {pm['delta_AUC']:+.6f}",
        f"- Candidate bootstrap 95% AUC: {boot[MODELS[1]]['lower']:.6f}-{boot[MODELS[1]]['upper']:.6f}",
        f"- Paired delta bootstrap 95%: {boot['delta']['lower']:+.6f} to {boot['delta']['upper']:+.6f}",
        f"- Candidate higher/equal/lower across 100 seeds: {summary['candidate_higher_equal_lower']}",
        f"- AUC>=0.70 median: {summary['target_0_70']['median']}",
        f"- AUC>=0.70 patient mean: {summary['target_0_70']['patient_mean']}",
    ]
    (args.output_dir/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("\n".join(lines))

if __name__=="__main__":
    main()
