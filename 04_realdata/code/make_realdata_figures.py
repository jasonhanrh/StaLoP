"""Create the manuscript two-panel Georgia real-data figure from final outputs."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(); p.add_argument("--results",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
    b=pd.read_csv(a.results/"realdata_bin_summary.csv"); m=pd.read_csv(a.results/"realdata_stalop_period_mass.csv")
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    x=np.arange(len(b)); labels=b.tbin.astype(str).tolist()
    for col,label in [("global_mspe","Global Linear"),("time_mspe","Time-Local"),("knn_mspe","KNN-Covariate"),("stalop_mspe","StaLoP")]: axes[0].plot(x,b[col],marker="o",label=label)
    axes[0].set_xticks(x,labels,rotation=45,ha="right"); axes[0].set_ylabel("MSPE"); axes[0].set_xlabel("Target bin"); axes[0].legend(frameon=False); axes[0].set_title("(a) Rolling prediction")
    piv=m.pivot(index="period",columns="tbin",values="combined_mass").fillna(0.0)
    arr=piv.to_numpy()
    im=axes[1].imshow(arr,aspect="auto",origin="lower")
    # Annotate the six largest historical-period masses in each target column.
    for j in range(arr.shape[1]):
        order=np.argsort(-arr[:,j],kind="mergesort")[:min(6,arr.shape[0])]
        for rank,i in enumerate(order,1):
            if arr[i,j]>0:
                axes[1].text(j,i,str(rank),ha="center",va="center",fontsize=7)
    axes[1].set_xticks(np.arange(len(piv.columns)),[str(x) for x in piv.columns],rotation=45,ha="right")
    # Thin y labels for readability.
    step=max(1,len(piv.index)//10); yy=np.arange(0,len(piv.index),step); axes[1].set_yticks(yy,[str(piv.index[i]) for i in yy]); axes[1].set_xlabel("Target bin"); axes[1].set_ylabel("Historical bin"); axes[1].set_title("(b) StaLoP historical-panel weight mass"); fig.colorbar(im,ax=axes[1],label="Combined relevance mass")
    fig.tight_layout(); a.output.parent.mkdir(parents=True,exist_ok=True); fig.savefig(a.output,dpi=220,bbox_inches="tight");
    if a.output.suffix.lower()!=".pdf": fig.savefig(a.output.with_suffix(".pdf"),bbox_inches="tight")
    plt.close(fig)

if __name__=="__main__": main()
