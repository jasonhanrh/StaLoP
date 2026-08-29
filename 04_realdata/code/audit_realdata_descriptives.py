from __future__ import annotations
import argparse,json
from pathlib import Path
from realdata_core import RealDataConfig,data_audit,load_binned_edges

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--input",type=Path,required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args()
    e=load_binned_edges(a.input); rows,s=data_audit(e,RealDataConfig().target_bins); a.out.mkdir(parents=True,exist_ok=True); rows.to_csv(a.out/"target_counts.csv",index=False); json.dump(s,open(a.out/"data_summary.json","w"),indent=2)
