#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, Normalize
from matplotlib.cm import ScalarMappable
from pathlib import Path

def args():
    p = argparse.ArgumentParser()
    p.add_argument("--enriched", required=True)
    p.add_argument("--country-summary", required=True)
    p.add_argument("--out-prefix", required=True)
    p.add_argument("--split-year", type=int, default=2010)
    p.add_argument("--min-n", type=int, default=5)
    p.add_argument("--top-countries", type=int, default=10)
    p.add_argument("--map-unknown-to-russia", action="store_true")
    return p.parse_args()

def arch(x):
    x = str(x)
    if "OXA-23" in x: return "OXA-23-like"
    if "OXA-24/40" in x or "OXA-72" in x: return "OXA-72-like (OXA-24/40)"
    if x == "MBL" or "NDM" in x or "VIM" in x or "IMP" in x: return "MBL (NDM/VIM/IMP)"
    if "Intrinsic" in x or "OXA-51" in x: return "No recognised carbapenemase"
    return "Other / mixed"

def mapname(x):
    return {
        "USA": "United States of America",
        "UK": "United Kingdom",
        "Czech Republic": "Czechia",
        "The Netherlands": "Netherlands",
        "Russia": "Russia",
    }.get(x, x)

a = args()
out = Path(a.out_prefix)
out.parent.mkdir(parents=True, exist_ok=True)

en = pd.read_csv(a.enriched, sep="\t", low_memory=False)
cty = pd.read_csv(a.country_summary, sep="\t")

if "Country_clean" not in en.columns:
    en["Country_clean"] = en["Country"]
    if "Country2" in en.columns:
        en["Country_clean"] = en["Country_clean"].fillna(en["Country2"])

if a.map_unknown_to_russia:
    en["Country_clean"] = en["Country_clean"].fillna("Russia").replace({"Unknown":"Russia"})
    cty["Country_clean"] = cty["Country_clean"].fillna("Russia").replace({"Unknown":"Russia"})

en["map_name"] = en["Country_clean"].map(mapname)
cty["map_name"] = cty["Country_clean"].map(mapname)

en["any_gt8"] = en["imipenem_consensus_gt8"].astype(bool) | en["meropenem_consensus_gt8"].astype(bool)
en["architecture"] = en["amrfinder_carbapenemase_group"].map(arch)

burden = cty.rename(columns={"any_consensus_prop":"burden"}).copy()

time = en.dropna(subset=["Year","Country_clean"]).copy()
time["Year"] = pd.to_numeric(time["Year"], errors="coerce")
time = time.dropna(subset=["Year"])
time["period"] = np.where(time["Year"] < a.split_year, "before", "since")
ts = time.groupby(["Country_clean","period"]).agg(n=("sample","count"), prop=("any_gt8","mean")).reset_index()
wide = ts.pivot(index="Country_clean", columns="period", values=["n","prop"])
wide.columns = [f"{x}_{y}" for x,y in wide.columns]
wide = wide.reset_index()
for col in ["n_before","n_since","prop_before","prop_since"]:
    if col not in wide: wide[col] = np.nan
wide["rise"] = wide["prop_since"] - wide["prop_before"]
wide = wide[(wide["n_before"] >= a.min_n) & (wide["n_since"] >= a.min_n)]
wide["map_name"] = wide["Country_clean"].map(mapname)

top = burden.sort_values("burden", ascending=False).head(a.top_countries)["Country_clean"].tolist()
mc = en[en["Country_clean"].isin(top)].groupby(["Country_clean","architecture"]).size().reset_index(name="count")
mc["total"] = mc.groupby("Country_clean")["count"].transform("sum")
mc["prop"] = mc["count"] / mc["total"]

world = gpd.read_file("https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip")
world = world.rename(columns={"ADMIN":"name"}) if "ADMIN" in world.columns and "name" not in world.columns else world
wb = world.merge(burden, left_on="name", right_on="map_name", how="left")
wr = world.merge(wide, left_on="name", right_on="map_name", how="left")

burden_cmap = LinearSegmentedColormap.from_list("burden", ["#FFFFFF","#F8D8CE","#CC7A69","#B52A1F","#7F1D13"])
rise_cmap = LinearSegmentedColormap.from_list("rise", ["#2F5D62","#F2F2F2","#F8D8CE","#CC7A69","#B52A1F"])

cols = {
    "OXA-23-like":"#7F1D13",
    "OXA-72-like (OXA-24/40)":"#CC7A69",
    "MBL (NDM/VIM/IMP)":"#4B2E52",
    "Other / mixed":"#8C7E91",
    "No recognised carbapenemase":"#D9D9D9",
}
order = list(cols)

fig = plt.figure(figsize=(15.5,9.0))
gs = fig.add_gridspec(2,2,height_ratios=[1.2,1.0],hspace=0.25,wspace=0.12)
axA, axB, axC = fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1]), fig.add_subplot(gs[1,:])

for ax in [axA, axB]:
    ax.axis("off"); ax.set_xlim(-180,180); ax.set_ylim(-60,85)

wb.plot(ax=axA, color="#F0F0F0", edgecolor="#C8C8C8", linewidth=0.25)
wb[wb["burden"].notna()].plot(ax=axA, column="burden", cmap=burden_cmap, vmin=0, vmax=1, edgecolor="#888", linewidth=0.3)
axA.set_title("A  Predicted high-MIC burden (any carbapenem)\n   Proportion of isolates", loc="left", fontsize=15, fontweight="bold")

wr.plot(ax=axB, color="#F0F0F0", edgecolor="#C8C8C8", linewidth=0.25)
wr[wr["rise"].notna()].plot(ax=axB, column="rise", cmap=rise_cmap, norm=TwoSlopeNorm(vmin=-0.5,vcenter=0,vmax=0.5), edgecolor="#888", linewidth=0.3)
axB.set_title(f"B  Increase in predicted high-MIC burden since {a.split_year}\n   Change in proportion", loc="left", fontsize=15, fontweight="bold")

for ax, cmap, label, norm in [
    (axA, burden_cmap, "Proportion of isolates", Normalize(0,1)),
    (axB, rise_cmap, "Change in proportion", TwoSlopeNorm(vmin=-0.5,vcenter=0,vmax=0.5))
]:
    cax = ax.inset_axes([0.28,-0.13,0.52,0.04])
    cb = fig.colorbar(ScalarMappable(norm=norm,cmap=cmap), cax=cax, orientation="horizontal")
    cb.set_label(label, fontsize=10); cb.outline.set_visible(False)

axC.axis("off")
ordered = burden[burden["Country_clean"].isin(top)].sort_values("burden", ascending=False)
countries = ordered["Country_clean"].tolist()
axC.set_xlim(0,1); axC.set_ylim(-1,len(countries)+1); axC.invert_yaxis()
axC.text(0,-0.85,"C  Dominant carbapenem resistance architecture by country",fontsize=15,fontweight="bold")

x_country, x_b0, x_b1, x_val, x_n, x_arch0, x_arch1 = 0.00,0.10,0.20,0.225,0.30,0.38,0.99
axC.text(0.15,-0.25,"Burden\n(any carbapenem)",ha="center",fontsize=10,fontweight="bold")
axC.text(x_n,-0.25,"n (isolates)",ha="center",fontsize=10,fontweight="bold")
axC.text(0.68,-0.55,"Proportion of isolates with architecture (%)",ha="center",fontsize=11)

for i,c in enumerate(countries):
    row = ordered[ordered["Country_clean"]==c].iloc[0]
    y=i
    axC.text(x_country,y,c,va="center",fontsize=11)
    b=float(row["burden"])
    axC.add_patch(plt.Rectangle((x_b0,y-0.35),(x_b1-x_b0)*b,0.7,color=burden_cmap(b),ec="white"))
    axC.text(x_val,y,f"{b:.2f}",ha="center",va="center",fontsize=9)
    axC.text(x_n,y,str(int(row["n"])),ha="center",va="center",fontsize=9)
    x=x_arch0
    sub=mc[mc["Country_clean"]==c]
    for o in order:
        p=float(sub[sub["architecture"]==o]["prop"].sum())
        w=(x_arch1-x_arch0)*p
        if w>0:
            axC.add_patch(plt.Rectangle((x,y-0.35),w,0.7,color=cols[o],ec="white",lw=0.6))
            if p>=0.055:
                axC.text(x+w/2,y,f"{p*100:.0f}%",ha="center",va="center",fontsize=8,color="white" if o!="No recognised carbapenemase" else "black",fontweight="bold")
            x+=w

lx=0.12
for i,o in enumerate(order):
    axC.add_patch(plt.Rectangle((lx+i*0.17,len(countries)+0.25),0.025,0.25,color=cols[o],clip_on=False))
    axC.text(lx+i*0.17+0.035,len(countries)+0.38,o,fontsize=9,va="center")

fig.savefig(f"{out}.svg", bbox_inches="tight", transparent=True)
fig.savefig(f"{out}.pdf", bbox_inches="tight", transparent=True)
fig.savefig(f"{out}.png", dpi=300, bbox_inches="tight", transparent=True)

mc.to_csv(f"{out}.architecture_counts.tsv",sep="\t",index=False)
wide.to_csv(f"{out}.rise_since_{a.split_year}.tsv",sep="\t",index=False)

print("[OK] wrote")
print(f"{out}.svg")
print(f"{out}.pdf")
print(f"{out}.png")
