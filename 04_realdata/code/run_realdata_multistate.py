"""Run/aggregate Georgia, Florida and Texas with the same frozen protocol."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from realdata_core import RealDataConfig
from run_realdata_migration import run


def main():
    p=argparse.ArgumentParser(); p.add_argument("--ga",type=Path,required=True); p.add_argument("--fl",type=Path,required=True); p.add_argument("--tx",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True); a=p.parse_args()
    a.output_dir.mkdir(parents=True,exist_ok=True); cfg=RealDataConfig()
    specs=[("Georgia",a.ga),("Florida",a.fl),("Texas",a.tx)]
    frames=[]
    for name,path in specs:
        od=a.output_dir/name.lower(); run(path,od,name,cfg); frames.append(pd.read_csv(od/"realdata_overall_summary.csv"))
    pd.concat(frames,ignore_index=True).to_csv(a.output_dir/"realdata_multistate_summary.csv",index=False)

if __name__=="__main__": main()
