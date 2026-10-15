#!/usr/bin/env python3
"""Filter SAbDab2 antibody–protein antigen complexes.

- Exclude pre-cutoff structures, engineered constructs, VNARs, and membrane proteins.
- Require a heavy chain and one protein antigen; light chains are optional.
- Require X-ray or EM structures with 1.0 <= resolution < 3.5 Å.
- Require X-ray R-factor < 0.3 and available, eligible mmCIF records.
- Deduplicate entity-equivalent complexes by resolved residue count.
- Require 350 < full-sequence residue count < 2200.
- Write the filtered registry and stage counts.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

from ..paths import LOGS_ROOT, DATA_ROOT, INTERMEDIATE_ROOT, SABDAB2_SUMMARY_CSV
from ..utils import (
    StageMetrics,
    atomic_write_dataframe,
    atomic_write_json,
    chain_ids_from_row,
    complex_id_from_row,
    complex_residue_count,
    make_abag_mask,
    make_quality_masks,
    make_scfv_mask,
    normalize_sabdab2_summary,
    parse_sabdab_cif,
    pdb_to_cif_path,
    residue_to_one_letter,
    resolve_chain,
)

TRAINING_IDS_JSON = INTERMEDIATE_ROOT / "training_data_ids.json"

# ── Configuration ────────────────────────────────────────────────────────────
SUMMARY_CSV = SABDAB2_SUMMARY_CSV

FILTERED_TSV = INTERMEDIATE_ROOT / "02_quality_filtered.tsv"
METRICS_OUTPUT = INTERMEDIATE_ROOT / "02_quality_metrics.json"

# Whole-complex residue bounds (defaults).  Every residue of every chain
# counts towards these bounds.
LOWER_RESIDUES = int(os.environ.get("ABAG_MIN_RESIDUES", "350"))
UPPER_RESIDUES = int(os.environ.get("ABAG_MAX_RESIDUES", "2200"))


# ── Training data index ──────────────────────────────────────────────────────

def load_training_ids() -> tuple[set, str]:
    """Load pre-cutoff PDB IDs and the cutoff date; reject missing or empty metadata."""
    if not TRAINING_IDS_JSON.is_file():
        raise FileNotFoundError(
            f"Training de-leak artifact not found: {TRAINING_IDS_JSON}\n"
            f"Run s1_training_data_check first (it flags the complexes "
            f"released before the training cutoff)."
        )
    with open(TRAINING_IDS_JSON) as f:
        data = json.load(f)
    cutoff = data.get("cutoff")
    ids = data.get("ids")
    if not cutoff:
        raise ValueError(
            f"{TRAINING_IDS_JSON}: missing 'cutoff' — re-run s1_training_data_check"
        )
    if not isinstance(ids, list) or len(ids) == 0:
        raise ValueError(
            f"{TRAINING_IDS_JSON}: missing/empty 'ids' (pre-cutoff PDB IDs) — "
            f"re-run s1_training_data_check"
        )
    union_ids = {str(i).lower() for i in ids}
    return union_ids, str(cutoff)


# ── Filter stages ────────────────────────────────────────────────────────────

def apply_training_deleak(
    df: pd.DataFrame,
    union_ids: set,
    cutoff: str,
    metrics: StageMetrics,
) -> pd.DataFrame:
    """Drop rows whose PDB base ID is in the s1 pre-cutoff training set."""
    df = df.copy()
    df["_pdb_lower"] = df["pdb"].astype(str).str.strip().str.lower()

    mask_in_union = df["_pdb_lower"].isin(union_ids)
    kept = df[~mask_in_union].copy()
    metrics.add(
        "Training data de-leak (pre-cutoff)",
        len(df), len(kept),
        reasons={f"released before {cutoff}": int(mask_in_union.sum())},
    )
    if not (len(kept) > 0):
        raise RuntimeError(f"All {len(df)} entries were released before the training cutoff "
        f"{cutoff} — the de-leak removed the entire raw dataset. "
        f"Re-check the --cutoff passed to s1.")
    return kept


def apply_abag_mask(df: pd.DataFrame, metrics: StageMetrics) -> pd.DataFrame:
    """Keep complexes with a heavy chain and a single protein antigen."""
    n = len(df)
    reasons = {
        "no H chain": int(df["Hchain"].isna().sum()),
        "no antigen chain": int(df["antigen_chain"].isna().sum()),
        "multimeric antigen": int(
            df["antigen_chain"].fillna("").str.contains("|", regex=False).sum()
        ),
        "non-protein antigen": int(
            (~df["antigen_type"].str.contains("protein", case=False, na=False)).sum()
        ),
    }
    mask = make_abag_mask(df)
    kept = df[mask]
    metrics.add("H chain + single protein antigen", n, len(kept), reasons=reasons)
    return kept


def apply_construct_removal(df: pd.DataFrame, metrics: StageMetrics) -> pd.DataFrame:
    """Exclude engineered constructs using SAbDab2 annotations, chain IDs, and compound names."""
    n = len(df)
    mask_construct = make_scfv_mask(df)
    reasons = {
        "SAbDab2 chainsharing_construct (SCFV/DIABODY/CODV/DVD/SCDB/OTHER)": int(
            df["chainsharing_construct"].isin(
                ["SCFV", "DIABODY", "CODV", "DVD", "SCDB", "OTHER"]
            ).sum()
        ),
        "same-letter H/L heuristic": int(
            (
                df["Lchain"].notna()
                & (df["Hchain"].str.upper() == df["Lchain"].str.upper())
            ).sum()
        ),
        "compound-name scan": int(
            df["compound"].astype(str)
            .str.contains(r"scfv", case=False, regex=True, na=False).sum()
        ),
    }
    kept = df[~mask_construct]
    metrics.add("Engineered-construct removal", n, len(kept), reasons=reasons)
    return kept


def apply_vnar_removal(df: pd.DataFrame, metrics: StageMetrics) -> pd.DataFrame:
    """Exclude shark VNAR/IgNAR single-domain antibodies."""
    n = len(df)
    mask_vnar = df["type"].eq("VNAR")
    kept = df[~mask_vnar]
    metrics.add(
        "VNAR exclusion (shark IgNAR)",
        n, len(kept),
        reasons={"type == VNAR": int(mask_vnar.sum())},
    )
    return kept


def apply_strict_cif_eligibility(
    df: pd.DataFrame, metrics: StageMetrics
) -> pd.DataFrame:
    """Exclude complexes with unsuitable mmCIF records.

    - Reject multi-character chain IDs, malformed records, and shared entities.
    - Reject missing full sequences or unmapped residues.
    - Substitute unresolved chains with resolved copies of the same entity.
    - Reject unresolved chains without a resolved copy.
    """
    n = len(df)
    reasons = {
        "multi-char chain ID": 0,
        "CIF parse failure (duplicate positions / malformed)": 0,
        "requested chains share an entity (copies or split segments)": 0,
        "un-modeled chain with no resolved copy": 0,
        "unmapped residue (X) in entity sequence": 0,
        "entity has no _entity_poly_seq sequence": 0,
    }
    drop_idx: list[int] = []
    cache: dict[str, object] = {}
    parse_msgs: dict[str, str] = {}
    for idx, row in df.iterrows():
        ids = chain_ids_from_row(row)
        if any(len(c) != 1 for c in ids):
            reasons["multi-char chain ID"] += 1
            drop_idx.append(idx)
            continue
        pdb_id = str(row["pdb"]).strip().lower()
        data = cache.get(pdb_id)
        if data is None:
            try:
                data = parse_sabdab_cif(pdb_to_cif_path(pdb_id))
            except ValueError as e:
                data = None
                parse_msgs[pdb_id] = str(e)
            cache[pdb_id] = data
        if data is None:
            reasons["CIF parse failure (duplicate positions / malformed)"] += 1
            drop_idx.append(idx)
            continue
        ents: list[str] = []
        ok = True
        for c in ids:
            info = resolve_chain(data, c)
            if info is None:
                reasons["un-modeled chain with no resolved copy"] += 1
                ok = False
                break
            ents.append(info["entity_id"])
        if not ok:
            drop_idx.append(idx)
            continue
        if len(set(ents)) != len(ents):
            reasons["requested chains share an entity (copies or split segments)"] += 1
            drop_idx.append(idx)
            continue
        for ent in ents:
            mon_ids = data["poly_seq"].get(ent)
            if not mon_ids:
                reasons["entity has no _entity_poly_seq sequence"] += 1
                drop_idx.append(idx)
                break
            if any(residue_to_one_letter(m) == "X" for m in mon_ids):
                reasons["unmapped residue (X) in entity sequence"] += 1
                drop_idx.append(idx)
                break

    kept = df.drop(index=drop_idx)
    metrics.add("Strict CIF eligibility", n, len(kept), reasons=reasons)
    for pdb_id, msg in list(parse_msgs.items())[:5]:
        print(f"  [strict CIF] {pdb_id}: {str(msg)[:120]}")
    return kept


def apply_entity_copy_dedup(
    df: pd.DataFrame, metrics: StageMetrics
) -> pd.DataFrame:
    """Keep one complex per PDB and heavy/light/antigen entity combination.

    - Prefer the copy with the most resolved residues.
    - Break ties by registry row order.
    """
    n = len(df)
    counts = df.groupby("pdb").size()
    multi = counts[counts > 1]
    if len(multi) == 0:
        metrics.add("Entity-copy dedup (identical copies)", n, n)
        return df.copy()

    cache: dict[str, dict] = {}
    drop_idx: list[int] = []
    dropped = 0
    for pdb_id in multi.index:
        data = cache.get(pdb_id)
        if data is None:
            data = parse_sabdab_cif(pdb_to_cif_path(pdb_id))
            cache[pdb_id] = data
        rows = df[df["pdb"] == pdb_id]
        best: dict[tuple, tuple[int, int]] = {}
        for idx, row in rows.iterrows():
            ids = tuple(chain_ids_from_row(row))
            infos: list[dict] = []
            for c in ids:
                info = resolve_chain(data, c)
                if info is None:
                    raise ValueError(
                        f"{pdb_to_cif_path(pdb_id)}: chain {c!r} (rows of "
                        f"{pdb_id}) has no coordinates and no resolved "
                        f"same-entity copy — cannot dedupe copies by entity"
                    )
                infos.append(info)
            ents = tuple(i["entity_id"] for i in infos)
            ressum = sum(len(i["resolved"]) for i in infos)
            prev = best.get(ents)
            if prev is None or ressum > prev[1]:
                best[ents] = (idx, ressum)
        keep = set(v[0] for v in best.values())
        drop_idx.extend(i for i in rows.index if i not in keep)
        dropped += len(rows) - len(best)

    kept = df.drop(index=drop_idx)
    metrics.add(
        "Entity-copy dedup (one row per entity triple)",
        n, len(kept),
        reasons={"identical copies sharing one entity set": dropped},
    )
    return kept


def apply_method_quality_mask(df: pd.DataFrame, metrics: StageMetrics) -> pd.DataFrame:
    """Keep eligible X-ray/EM structures and assign their experimental modality."""
    n = len(df)
    import os

    methods = tuple(x.lower() for x in os.environ.get("ABAG_METHODS", "xray,em").split(",") if x)
    resolution_min = float(os.environ.get("ABAG_RESOLUTION_MIN", "1.0"))
    resolution_max = float(os.environ.get("ABAG_RESOLUTION_MAX", "3.5"))
    r_factor_max = float(os.environ.get("ABAG_XRAY_R_FACTOR_MAX", "0.3"))
    exclude_membrane = os.environ.get("ABAG_EXCLUDE_MEMBRANE", "1") == "1"
    mask_xray, mask_em = make_quality_masks(
        df, resolution_min, resolution_max, r_factor_max, exclude_membrane, methods
    )

    res = pd.to_numeric(df["resolution"], errors="coerce")
    r_factor = pd.to_numeric(df["r_factor"], errors="coerce")
    allowed = ([("XRAY" if x == "xray" else "ELECTRON_MICROSCOPY") for x in methods])
    bad_method = ~df["method"].isin(allowed)
    res_out = res.isna() | (res < resolution_min) | (res >= resolution_max)
    rfail = (
        r_factor.isna() | (r_factor >= r_factor_max)
        if "r_factor" in df.columns
        else pd.Series(False, index=df.index)
    )
    membrane = (
        df["short_header"].str.contains("MEMBRANE", case=False, na=False)
        if exclude_membrane else pd.Series(False, index=df.index)
    )

    reasons = {
        f"method not {methods}": int(bad_method.sum()),
        f"resolution out of [{resolution_min}, {resolution_max}) or missing": int(res_out.sum()),
        f"r-factor >= {r_factor_max} or missing (X-ray)": int(
            (rfail & (df["method"] == "XRAY")).sum()
        ),
        "membrane protein": int(membrane.sum()) if exclude_membrane else 0,
    }

    df = df.copy()
    df["modality"] = None
    df.loc[mask_xray, "modality"] = "X-ray"
    df.loc[mask_em, "modality"] = "EM"
    kept = df[df["modality"].notna()]
    metrics.add("Method/resolution/R-factor/membrane", n, len(kept), reasons=reasons)
    return kept


def apply_source_availability(
    df: pd.DataFrame, metrics: StageMetrics
) -> pd.DataFrame:
    """Exclude complexes without a source SAbDab2 mmCIF file."""
    n = len(df)
    missing_ids: set[str] = set()
    for pdb_id in df["pdb"].astype(str).str.strip().str.lower().unique():
        if not pdb_to_cif_path(pdb_id).is_file():
            missing_ids.add(pdb_id)

    reasons = {
        "missing SAbDab2 mmCIF": len(missing_ids),
    }
    kept = df[~df["pdb"].astype(str).str.strip().str.lower().isin(missing_ids)]
    metrics.add("SAbDab2 mmCIF exists", n, len(kept), reasons=reasons)
    return kept


def compute_complex_residue_lengths(
    df: pd.DataFrame,
    cache: dict | None = None,
) -> pd.Series:
    """Return full-sequence residue counts indexed by registry row, reusing cached counts."""
    counts: dict = {} if cache is None else cache
    lengths: dict[int, float] = {}
    for idx, row in df.iterrows():
        pdb_id = str(row["pdb"]).strip().lower()
        ids = tuple(chain_ids_from_row(row))
        key = (pdb_id, ids)
        if key not in counts:
            counts[key] = complex_residue_count(pdb_id, ids)
        lengths[idx] = counts[key]
    return pd.Series(lengths, dtype="float64")


def filter_by_complex_residue_count(
    df: pd.DataFrame,
    lower: int = LOWER_RESIDUES,
    upper: int = UPPER_RESIDUES,
    cache: dict | None = None,
) -> pd.DataFrame:
    """Keep complexes with lower < full-sequence residue count < upper; add residue_length."""
    df = df.copy()
    df["residue_length"] = compute_complex_residue_lengths(df, cache)
    mask = (df["residue_length"] > lower) & (df["residue_length"] < upper)
    return df[mask]


def apply_complex_size_filter(
    df: pd.DataFrame,
    metrics: StageMetrics,
    lower: int,
    upper: int,
    cache: dict,
) -> pd.DataFrame:
    """Apply complex residue-count bounds and record filter statistics."""
    n = len(df)
    filtered = filter_by_complex_residue_count(df, lower=lower, upper=upper, cache=cache)
    lengths = compute_complex_residue_lengths(filtered, cache)
    all_lengths = compute_complex_residue_lengths(df, cache)
    dropped_idx = ~df.index.isin(filtered.index)
    dropped_len = all_lengths[dropped_idx]
    reasons = {
        f"<= {lower} residues (whole complex)": int((dropped_len <= lower).sum()),
        f">= {upper} residues (whole complex)": int((dropped_len >= upper).sum()),
    }
    metrics.add(
        f"Whole-complex residue bounds ({lower} < n < {upper})",
        n, len(filtered), reasons=reasons,
    )
    return filtered


# ── Main ─────────────────────────────────────────────────────────────────────

def run(force: bool = False, lower: int = LOWER_RESIDUES, upper: int = UPPER_RESIDUES) -> dict:
    """Write the quality-filtered registry and return filter metadata."""
    print("\n" + "=" * 60)
    print("STEP 2: SAbDab2 Quality Filter (unified EM/X-ray)")
    print("=" * 60)

    if SUMMARY_CSV is None or not SUMMARY_CSV.exists():
        raise FileNotFoundError(f"SAbDab2 summary not found: {SUMMARY_CSV or '(not configured)'}")
    if FILTERED_TSV.exists() and not force:
        raise FileExistsError(f"Output already exists: {FILTERED_TSV}; use --force or --resume")

    union_ids, cutoff = load_training_ids()

    metrics = StageMetrics("STEP 2: quality filter")
    df_raw = pd.read_csv(SUMMARY_CSV, low_memory=False)
    if not (len(df_raw) > 0):
        raise RuntimeError(f"SAbDab2 summary is empty: {SUMMARY_CSV}")
    df_raw = normalize_sabdab2_summary(df_raw)
    metrics.add(
        "Raw SAbDab2 summary",
        len(df_raw),
        len(df_raw),
    )

    df = apply_training_deleak(df_raw, union_ids, cutoff, metrics)
    df = apply_abag_mask(df, metrics)
    if os.environ.get("ABAG_EXCLUDE_ENGINEERED", "1") == "1":
        df = apply_construct_removal(df, metrics)
    else:
        metrics.add("Engineered-construct removal", len(df), len(df), reasons={"filter disabled": 0})
    if os.environ.get("ABAG_EXCLUDE_VNAR", "1") == "1":
        df = apply_vnar_removal(df, metrics)
    else:
        metrics.add("VNAR exclusion (shark IgNAR)", len(df), len(df), reasons={"filter disabled": 0})
    df = apply_method_quality_mask(df, metrics)
    df = apply_source_availability(df, metrics)
    df = apply_strict_cif_eligibility(df, metrics)
    df = apply_entity_copy_dedup(df, metrics)
    df = apply_complex_size_filter(df, metrics, lower, upper, {})

    # ── Registry output ──────────────────────────────────────────────────────
    df = df.drop(columns=["_pdb_lower"], errors="ignore")
    df["complex_id"] = df.apply(complex_id_from_row, axis=1)

    # Case-insensitive filesystem safety: chain IDs may differ only by case
    # (SAbDab2 lists both 'H/L/C' and 'h/l/c' copies of the same antibody in
    # one PDB, e.g. pdb_00005tru).  On macOS/Windows those complex_ids would
    # collide as filenames.  Keep the first row per lowercased complex_id and
    # drop the case-variant with a counted reason — the variants are identical
    # antibody copies that FoldSeek clustering would merge anyway.
    key = df["complex_id"].str.lower()
    n_case_variants = int(key.duplicated().sum())
    if n_case_variants:
        df = df[~key.duplicated()].copy()
        metrics.add(
            "Case-collision dedup (identical chain copies)",
            metrics.stages[-1]["after"], len(df),
            reasons={"chain IDs differ only by case (kept first)": n_case_variants},
        )

    dup = df["complex_id"].duplicated().sum()
    if not (dup == 0):
        raise RuntimeError(f"registry: {dup} duplicated complex_id values after filters — "
        f"chain-suffix collision, fix the filter logic.")
    out_path = FILTERED_TSV
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = df.drop(columns=["_pdb_lower"], errors="ignore")
    atomic_write_dataframe(df, out_path, sep="\t", index=False)

    # Stage metrics artifact — the plots funnel is rendered from THIS, so it
    # always shows the real cascade (raw -> de-leak -> ... -> registry) and
    # cannot drift from the actual s2 filter order.
    stage_path = INTERMEDIATE_ROOT / "stage_metrics.json"
    stage_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        stage_path,
        {"cutoff": cutoff, "final": int(len(df)), "stages": metrics.to_dict()},
    )
    atomic_write_json(METRICS_OUTPUT, {
        "stages": metrics.to_dict(),
        "final": int(len(df)),
    })

    n_total = len(df)
    n_em = int((df["modality"] == "EM").sum())
    n_xray = int((df["modality"] == "X-ray").sum())
    if not (n_em + n_xray == n_total):
        raise RuntimeError(f"Modality split does not account for all rows ({n_em}+{n_xray} != {n_total})")

    metrics.report()
    print(f"\n  Final (post-s2) registry: {n_total:,} complexes "
          f"(EM={n_em:,}, X-ray={n_xray:,})")
    print(f"  Registry: {out_path}")

    LOG_DIR = LOGS_ROOT
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"quality_filter_{timestamp}.log"
    with open(log_path, "w") as lf:
        lf.write("=== Step 2: quality filter ===\n")
        lf.write(f"Started at: {timestamp}\n")
        for s in metrics.to_dict():
            lf.write(f"{s['stage']}: {s['before']} -> {s['after']} "
                     f"(-{s['dropped']}) {s['reasons']}\n")
        lf.write(f"Final: {n_total} (EM={n_em}, X-ray={n_xray})\n")
        lf.write(f"Registry: {out_path}\n")
    print(f"  Log: {log_path}")

    return {
        "n_raw": len(df_raw),
        "n_after_deleak": int(metrics.stages[1]["after"]),
        "n_em": n_em,
        "n_xray": n_xray,
        "total": n_total,
        "summary_path": str(out_path),
        "metrics": metrics.to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 2: SAbDab2 quality filter.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs.")
    parser.add_argument("--lower", type=int, default=LOWER_RESIDUES,
                        help="Lower whole-complex residue bound (default 350).")
    parser.add_argument("--upper", type=int, default=UPPER_RESIDUES,
                        help="Upper whole-complex residue bound (default 2200).")
    args = parser.parse_args()
    run(force=args.force, lower=args.lower, upper=args.upper)


if __name__ == "__main__":
    main()
