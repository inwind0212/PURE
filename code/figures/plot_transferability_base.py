#!/usr/bin/env python3
"""Shared plotting utilities for Figure 3."""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
import pandas as pd
POINT_COLOR="#71818C"; PCTS=(0,1,2,5,10,20,40,60,80,100)
SCOPE_STYLE={"city":("City","#D55E00","o"),"country":("Country","#009E73","^"),"country_loco":("Country LOCO","#0072B2","s")}
def panel_label(ax,label): ax.text(-.12,1.03,label,transform=ax.transAxes,ha="left",va="bottom",fontsize=10,fontweight="bold",clip_on=False)
def draw_illustration(ax,path:Path):
 artwork=plt.imread(path); rgb=artwork[...,:3]; rows,cols=np.where(np.any(rgb<.985,axis=2)); pad=18; ax.imshow(artwork[max(0,rows.min()-pad):min(artwork.shape[0],rows.max()+pad+1),max(0,cols.min()-pad):min(artwork.shape[1],cols.max()+pad+1)]); ax.axis("off")
def load_quantity(path):
 data=pd.read_csv(path); required={"scope","poi_retained_pct","r2","delta_r2"}
 if required-set(data.columns): raise RuntimeError(f"Missing columns: {sorted(required-set(data.columns))}")
 expected={(scope,pct) for scope in SCOPE_STYLE for pct in PCTS}; observed=set(zip(data.scope,data.poi_retained_pct.astype(int)))
 if observed!=expected or len(data)!=len(expected): raise RuntimeError("Expected a complete 3 x 10 grid")
 zero=data[data.poi_retained_pct.eq(0)].r2.to_numpy(float)
 if not np.allclose(zero,zero[0],atol=1e-12): raise RuntimeError("Zero-POI baselines differ")
 return data,float(zero[0])
def draw_line(ax,data,baseline,label):
 for scope,(name,color,marker) in SCOPE_STYLE.items():
  part=data[data.scope.eq(scope)].sort_values("poi_retained_pct"); ax.plot(part.poi_retained_pct.to_numpy(int)[1:],part.r2.to_numpy(float)[1:],color=color,marker=marker,markersize=3.8,linewidth=1.45,label=name)
 ax.axhline(baseline,color="#4D4D4D",linewidth=1,linestyle=(0,(4,2.5)),label="AlphaEarth"); ax.set_xlim(-2,102); values=data.loc[data.poi_retained_pct.gt(0),"r2"].to_numpy(float); ax.set_ylim(np.floor((min(baseline,values.min())-.004)*1000)/1000,np.ceil((max(baseline,values.max())+.004)*1000)/1000); ax.set_xticks([0,10,20,40,60,80,100]); ax.set_xticks([1,2,5],minor=True); ax.set_xlabel("POIs retained (%)"); ax.set_ylabel(r"Mean downstream $R^2$"); ax.legend(frameon=False,loc="lower right",bbox_to_anchor=(1,.12),fontsize=6.8,handlelength=2.2,labelspacing=.35); ax.spines[["top","right"]].set_visible(False); ax.tick_params(direction="out",length=2.7,width=.7); ax.tick_params(axis="x",which="minor",direction="out",length=1.6,width=.6)
 if label is not None: panel_label(ax,label)
