#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import re

outdir = Path("outputs/error_characterisation/icassigner_validation")
outdir.mkdir(parents=True, exist_ok=True)

meta = pd.read_csv("data/phenotypes_validation/data.csv", sep=None, engine="python")

# Normalise ID used by validation MIC table
if "id" not in meta.columns:
    raise SystemExit("No 'id' column found in data/phenotypes_validation/data.csv")

meta["sample_id"] = meta["id"].astype(str)

# ICassigner expects an existing IC column. Seed it conservatively from Pasteur ST first,
# then Oxford ST if Pasteur is unavailable.
PASTEUR_ST_TO_IC = {
    "ST1": "IC1",
    "ST2": "IC2",
    "ST3": "IC3",
    "ST15": "IC4",
    "ST79": "IC5",
    "ST78": "IC6",
    "ST25": "IC7",
    "ST10": "IC8",
    "ST85": "IC9",
}

OXFORD_ST_TO_IC = {
    "ST231": "IC1",
    "ST208": "IC2",
    "ST451": "IC2",
    "ST348": "IC3",
    "ST15": "IC4",
    "ST79": "IC5",
    "ST78": "IC6",
    "ST25": "IC7",
    "ST85": "IC9",
}

def norm_st(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return None
    # handle messy values such as "208;1806"
    first = re.split(r"[;, ]+", s)[0]
    if first.upper().startswith("ST"):
        return first.upper()
    if first.isdigit():
        return "ST" + first
    return first.upper()

pasteur_col = "ST (MLST (Pasteur))"
oxford_col = "ST (MLST (Oxford))"

ic = []
ic_source = []

for _, row in meta.iterrows():
    call = "UA"
    source = "unassigned"

    if pasteur_col in meta.columns:
        stp = norm_st(row.get(pasteur_col))
        if stp in PASTEUR_ST_TO_IC:
            call = PASTEUR_ST_TO_IC[stp]
            source = "Pasteur_ST_anchor"

    if call == "UA" and oxford_col in meta.columns:
        sto = norm_st(row.get(oxford_col))
        if sto in OXFORD_ST_TO_IC:
            call = OXFORD_ST_TO_IC[sto]
            source = "Oxford_ST_anchor"

    ic.append(call)
    ic_source.append(source)

meta["IC_seed"] = ic
meta["IC_seed_source"] = ic_source

out = outdir / "validation_icassigner_input.csv"
meta.to_csv(out, index=False)

print("[OK] wrote", out)
print("Rows:", len(meta))
print("\nSeed IC counts:")
print(meta["IC_seed"].value_counts(dropna=False).to_string())
print("\nSeed source counts:")
print(meta["IC_seed_source"].value_counts(dropna=False).to_string())