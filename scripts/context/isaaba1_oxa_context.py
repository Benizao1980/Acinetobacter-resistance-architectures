#!/usr/bin/env python3
"""Assess ISAba1 context around OXA-23-family and OXA-51-family loci.

Inputs
------
1. Directory of FASTA assemblies.
2. Directory of isolate-level AMRFinderPlus TSV files.

Approach
--------
ISAba1 is detected with a sequence-internal marker defined by the published
ISAba1 PCR primer pair:
    F: CATTGGCATTAAACTGAGGAGAAA
    R: TTGGAAATGGGGAAAACGAA
which yields a 451-bp product. Exact primer-pair matches in the expected
orientation and spacing are used as evidence that an intact internal ISAba1
marker is present on a contig.

For each AMRFinderPlus OXA-23-family or OXA-51-family hit, the script records:
- locus contig, coordinates and strand;
- distance from the promoter-facing side of the beta-lactamase to the contig edge;
- genome-wide ISAba1-marker detection;
- whether an intact ISAba1 marker is present on the same contig;
- nearest promoter-side same-contig ISAba1 marker, if present;
- a conservative context classification.

Important limitation
--------------------
Absence of an intact marker from the same contig is NOT evidence that ISAba1 is
biologically absent upstream. Repetitive insertion sequences commonly break
short-read assemblies. Such cases are classified as unresolved, particularly
when the beta-lactamase promoter-facing side lies near a contig edge.

This script does not infer promoter activation from genome-wide ISAba1 presence
alone and does not attempt assembly-graph reconstruction.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

from Bio import SeqIO
from Bio.Seq import Seq

ISA_F = "CATTGGCATTAAACTGAGGAGAAA"
ISA_R = "TTGGAAATGGGGAAAACGAA"
ISA_F_RC = str(Seq(ISA_F).reverse_complement())
ISA_R_RC = str(Seq(ISA_R).reverse_complement())
EXPECTED_PRODUCT = 451
EDGE_THRESHOLD = 500  # bp; descriptive flag, not a biological adjacency cutoff
UPSTREAM_SEARCH = 2000  # bp; only used to report nearest same-contig marker


def find_all(seq: str, query: str) -> List[int]:
    """Return zero-based start positions of exact query matches."""
    out = []
    start = 0
    while True:
        i = seq.find(query, start)
        if i < 0:
            break
        out.append(i)
        start = i + 1
    return out


def detect_isa_markers(fasta_path: Path) -> Tuple[Dict[str, int], List[dict]]:
    """Return contig lengths and intact 451-bp ISAba1 internal-marker calls."""
    lengths: Dict[str, int] = {}
    hits: List[dict] = []

    for rec in SeqIO.parse(str(fasta_path), "fasta"):
        seq = str(rec.seq).upper()
        cid = rec.id
        lengths[cid] = len(seq)

        # Reference orientation: F primer followed by reverse-complement of R.
        for a in find_all(seq, ISA_F):
            b = a + EXPECTED_PRODUCT - len(ISA_R)
            if b >= 0 and b + len(ISA_R_RC) <= len(seq) and seq[b:b+len(ISA_R_RC)] == ISA_R_RC:
                hits.append({
                    "contig": cid,
                    "start": a + 1,
                    "stop": b + len(ISA_R_RC),
                    "strand": "+",
                    "marker_length": b + len(ISA_R_RC) - a,
                })

        # Reverse orientation: R primer followed by reverse-complement of F.
        for a in find_all(seq, ISA_R):
            b = a + EXPECTED_PRODUCT - len(ISA_F)
            if b >= 0 and b + len(ISA_F_RC) <= len(seq) and seq[b:b+len(ISA_F_RC)] == ISA_F_RC:
                hits.append({
                    "contig": cid,
                    "start": a + 1,
                    "stop": b + len(ISA_F_RC),
                    "strand": "-",
                    "marker_length": b + len(ISA_F_RC) - a,
                })

    return lengths, hits


def family_of(row: dict) -> str | None:
    name = row.get("Element name", "")
    symbol = row.get("Element symbol", "")
    if "OXA-23 family" in name or symbol == "blaOXA-23":
        return "OXA-23-family"
    if "OXA-51 family" in name:
        return "OXA-51-family"
    return None


def promoter_edge_distance(start: int, stop: int, strand: str, contig_len: int) -> int:
    """Distance from promoter-facing side of CDS to nearest relevant contig end."""
    if strand == "+":
        return start - 1
    if strand == "-":
        return contig_len - stop
    return min(start - 1, contig_len - stop)


def promoter_side_gap(start: int, stop: int, strand: str, hit: dict) -> int | None:
    """Gap from CDS promoter-facing side to ISA marker if marker lies upstream."""
    if strand == "+" and hit["stop"] < start:
        return start - hit["stop"] - 1
    if strand == "-" and hit["start"] > stop:
        return hit["start"] - stop - 1
    return None


def classify(same_contig_hits: List[dict], promoter_candidates: List[Tuple[int, dict]], edge_dist: int, isa_any: bool) -> str:
    if promoter_candidates:
        gap, _ = min(promoter_candidates, key=lambda x: x[0])
        if gap <= UPSTREAM_SEARCH:
            return "same-contig upstream ISAba1 marker"
    if same_contig_hits:
        return "ISAba1 marker on same contig, not promoter-side within search window"
    if isa_any and edge_dist <= EDGE_THRESHOLD:
        return "unresolved: ISAba1 elsewhere; promoter side near contig edge"
    if isa_any:
        return "unresolved: ISAba1 elsewhere in assembly"
    if edge_dist <= EDGE_THRESHOLD:
        return "unresolved: no intact ISAba1 marker; promoter side near contig edge"
    return "no intact ISAba1 marker detected; adjacency not demonstrated"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta-dir", required=True, type=Path)
    ap.add_argument("--amr-dir", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    fasta_paths = {p.stem: p for p in args.fasta_dir.glob("*.fasta")}
    amr_paths = {p.name.replace(".amrfinder.tsv", ""): p for p in args.amr_dir.glob("*.amrfinder.tsv")}
    isolates = sorted(set(fasta_paths) & set(amr_paths))

    results = []
    for iso in isolates:
        lengths, isa_hits = detect_isa_markers(fasta_paths[iso])
        isa_any = bool(isa_hits)

        with amr_paths[iso].open() as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                fam = family_of(row)
                if fam is None:
                    continue

                cid = row["Contig id"]
                if cid not in lengths:
                    continue
                start = int(row["Start"])
                stop = int(row["Stop"])
                strand = row["Strand"]
                clen = lengths[cid]
                edge_dist = promoter_edge_distance(start, stop, strand, clen)

                same = [h for h in isa_hits if h["contig"] == cid]
                candidates = []
                for h in same:
                    gap = promoter_side_gap(start, stop, strand, h)
                    if gap is not None:
                        candidates.append((gap, h))

                nearest_gap = ""
                nearest_start = ""
                nearest_stop = ""
                nearest_strand = ""
                if candidates:
                    gap, h = min(candidates, key=lambda x: x[0])
                    nearest_gap = gap
                    nearest_start = h["start"]
                    nearest_stop = h["stop"]
                    nearest_strand = h["strand"]

                results.append({
                    "isolate": iso,
                    "oxa_family": fam,
                    "element_symbol": row["Element symbol"],
                    "element_name": row["Element name"],
                    "amrfinder_coverage_pct": row.get("% Coverage of reference", ""),
                    "amrfinder_identity_pct": row.get("% Identity to reference", ""),
                    "contig": cid,
                    "contig_length_bp": clen,
                    "oxa_start": start,
                    "oxa_stop": stop,
                    "oxa_strand": strand,
                    "promoter_side_contig_edge_distance_bp": edge_dist,
                    "promoter_side_near_edge_le500bp": edge_dist <= EDGE_THRESHOLD,
                    "isaaba1_intact_internal_marker_detected_in_genome": isa_any,
                    "isaaba1_marker_count_genome": len(isa_hits),
                    "isaaba1_marker_on_same_contig": bool(same),
                    "nearest_promoter_side_isaaba1_gap_bp": nearest_gap,
                    "nearest_isaaba1_marker_start": nearest_start,
                    "nearest_isaaba1_marker_stop": nearest_stop,
                    "nearest_isaaba1_marker_strand": nearest_strand,
                    "context_classification": classify(same, candidates, edge_dist, isa_any),
                })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(results[0].keys()) if results else []
    with args.output.open("w", newline="") as out:
        w = csv.DictWriter(out, fieldnames=fields)
        w.writeheader()
        w.writerows(results)

    print(f"Matched isolates: {len(isolates)}")
    print(f"Locus rows written: {len(results)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
