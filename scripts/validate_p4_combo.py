from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from scripts.validate_compact_candidates import ASSISTANT_MODEL, build_candidate_frame
from scripts.validate_p4 import CANDIDATES, SUBJECTS, transform
from scripts.validate_p3 import validate_one_seed

SECONDS=("PRIOR_MI","SYNCOPE","NSVT_GT10","SEX_MALE","VT_ANY","QRS","QTC")

def build(second):
    frame, features, _ = build_candidate_frame()
    cols=list(features[ASSISTANT_MODEL])
    subjects=pd.read_parquet(SUBJECTS)
    for cand in ("LOG_PROBNP",second):
        source,kind=CANDIDATES[cand]
        extra=pd.DataFrame({
            "patient_id":subjects["patient_id"].astype("string"),
            f"cand_{cand.lower()}":transform(subjects[source],kind),
        })
        frame=frame.merge(extra,on="patient_id",how="left",validate="one_to_one")
        cols.append(f"cand_{cand.lower()}")
    return frame,cols

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--second",required=True)
    ap.add_argument("--seed-start",type=int,default=0)
    ap.add_argument("--seed-stop",type=int,default=19)
    ap.add_argument("--output-dir",type=Path,required=True)
    a=ap.parse_args()
    second=a.second.upper()
    if second not in SECONDS: raise ValueError(second)
    frame,cols=build(second)
    model=f"COMPACT25_PROBNP_{second}"
    a.output_dir.mkdir(parents=True,exist_ok=True)
    rows=[]; oofs=[]
    for seed in range(a.seed_start,a.seed_stop+1):
        s,o,_=validate_one_seed(frame,cols,model_name=model,seed=seed)
        rows.append(s); oofs.append(o)
    pd.DataFrame(rows).to_csv(a.output_dir/"per_seed.csv",index=False)
    pd.concat(oofs,ignore_index=True).to_csv(a.output_dir/"oof.csv",index=False)
    audit={c:int(frame[f"cand_{c.lower()}"].isna().sum()) for c in ("LOG_PROBNP",second)}
    (a.output_dir/"audit.json").write_text(json.dumps({"model":model,"features":len(cols),"missing":audit},indent=2)+"\n")
    print(pd.DataFrame(rows)[["model","seed","AUC","AP","Brier","BSS"]].to_string(index=False))

if __name__=="__main__":
    main()
