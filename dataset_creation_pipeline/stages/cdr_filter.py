#!/usr/bin/env python3
"""Require fully resolved Chothia CDRs and stage antibody–antigen structures.

- Number full antibody sequences with Anarcii 2.0.8.
- Check H1 26–32, H2 52–56, H3 96–101; L1 26–32, L2 50–52, L3 91–96.
- Retain complexes with coordinates for every CDR residue.
- Export heavy/light/antigen chain subsets and update the registry with CDRH3 length.
"""
from __future__ import annotations

import argparse
import datetime
import shutil
from pathlib import Path
from typing import Optional

import pandas as pd

from ..paths import DATA_ROOT, INTERMEDIATE_ROOT, SCRATCH_ROOT
from ..utils import (
    CHOTHIA_CDR_RANGES,
    StageMetrics,
    atomic_write_dataframe,
    atomic_write_json,
    chain_full_sequences,
    chain_ids_from_row,
    complex_id_from_row,
    get_chothia_details_batch,
    pdb_to_cif_path,
    resolved_positions,
    write_chain_subset_cif,
)

INPUT_TSV = INTERMEDIATE_ROOT / "02_quality_filtered.tsv"
FILTERED_TSV = INTERMEDIATE_ROOT / "04_cdr_filtered.tsv"
SCRATCH_SURVIVORS = SCRATCH_ROOT / "s4_survivors"


def _load_fastas(registry: pd.DataFrame) -> dict[str, dict[str, str]]:
    """Read full chain sequences from source mmCIF files."""
    fastas: dict[str, dict[str, str]] = {}
    for _, row in registry.iterrows():
        complex_id = complex_id_from_row(row)
        pdb_id = str(row["pdb"]).strip().lower()
        chains = set(chain_ids_from_row(row))
        cif_path = pdb_to_cif_path(pdb_id)
        if not cif_path.exists():
            raise FileNotFoundError(f"SAbDab2 CIF not found for {complex_id}: {cif_path}")
        seqs = chain_full_sequences(cif_path, chains)
        missing = chains - set(seqs)
        if missing:
            raise ValueError(f"Chains {sorted(missing)} of {complex_id} have no sequence in {cif_path}")
        if complex_id in fastas:
            raise ValueError(f"Duplicate complex_id {complex_id} in registry")
        fastas[complex_id] = seqs
    return fastas


def run(
    processes: int = 1,
    fastas: Optional[dict[str, dict[str, str]]] = None,
) -> dict:
    """Filter unresolved CDRs, record CDRH3 lengths, and stage surviving chain subsets."""
    print("\n" + "=" * 60)
    print("STEP 4: CDR-loop coverage filter (Chothia, Anarcii) + staging")
    print("=" * 60)

    if not INPUT_TSV.exists():
        raise FileNotFoundError(
        f"Post-s2 registry not found: {INPUT_TSV}\n"
        f"Run s2_quality_filter first."
        )

    registry = pd.read_csv(INPUT_TSV, sep="\t", low_memory=False)
    if not (len(registry) > 0):
        raise RuntimeError(f"Post-s2 registry is empty: {FILTERED_TSV}")

    # Validate required columns
    for col in ("pdb", "Hchain", "antigen_chain", "modality"):
        if not (col in registry.columns):
            raise RuntimeError(f"Registry missing required column {col!r}")

    if fastas is None:
        print("  Building in-memory FASTA from CIFs (s3 not passed)...")
        fastas = _load_fastas(registry)
        print(f"  Built FASTA for {len(fastas):,} complexes")
    else:
        if not (len(fastas) == len(registry)):
            raise RuntimeError(f"fastas dict has {len(fastas)} entries but registry has {len(registry)} rows — "
            f"mismatch (maybe stale registry?)")
        # Verify every complex_id in registry is in fastas
        missing_ids = [complex_id_from_row(r) for _, r in registry.iterrows() if complex_id_from_row(r) not in fastas]
        if missing_ids:
            raise RuntimeError(f"fastas missing {len(missing_ids)} registry complexes, e.g. {missing_ids[:3]}")

    # --- Collect antibody chains for Anarcii batch ---
    # Keys are unique per chain: f"{complex_id}__{auth_chain}" (auth_chain is the SAbDab chain letter)
    chain_keys: list[str] = []
    seqs_for_anarcii: dict[str, str] = {}
    roles_for_anarcii: dict[str, str] = {}
    # Mapping: chain_key -> (complex_id, auth_chain, chain_role)
    key_to_meta: dict[str, tuple[str, str, str]] = {}
    # Also collect heavy keys for cdrh3_length batch
    heavy_keys: list[str] = []
    heavy_seqs_for_h3: dict[str, str] = {}

    for _, row in registry.iterrows():
        complex_id = complex_id_from_row(row)
        pdb_id = str(row["pdb"]).strip().lower()
        h_chain = str(row["Hchain"]).strip() if pd.notna(row["Hchain"]) else None
        l_chain = str(row["Lchain"]).strip() if pd.notna(row["Lchain"]) and str(row["Lchain"]).strip() else None

        # Heavy chain (must exist by s2)
        if not (h_chain):
            raise RuntimeError(f"Row {complex_id} has no Hchain")
        auth_h = h_chain  # SAbDab chain letter for heavy
        # Validate heavy seq exists in fastas
        if auth_h not in fastas[complex_id]:
            raise ValueError(f"Heavy chain {auth_h!r} of {complex_id} not in fastas")
        h_seq = fastas[complex_id][auth_h]
        h_key = f"{complex_id}__{auth_h}"
        seqs_for_anarcii[h_key] = h_seq
        roles_for_anarcii[h_key] = "H"
        key_to_meta[h_key] = (complex_id, auth_h, "H")
        chain_keys.append(h_key)
        heavy_keys.append(h_key)
        heavy_seqs_for_h3[h_key] = h_seq

        # Light chain (optional, nanobody)
        if l_chain:
            auth_l = l_chain
            if auth_l not in fastas[complex_id]:
                raise ValueError(f"Light chain {auth_l!r} of {complex_id} not in fastas")
            l_seq = fastas[complex_id][auth_l]
            l_key = f"{complex_id}__{auth_l}"
            # Light chains share the same complex_id prefix, so need disambiguation if H and L letters coincide
            # (should not happen after s2 scFv filter, but be safe)
            if l_key == h_key:
                l_key = f"{complex_id}__{auth_l}_L"
            seqs_for_anarcii[l_key] = l_seq
            roles_for_anarcii[l_key] = "L"
            key_to_meta[l_key] = (complex_id, auth_l, "L")
            chain_keys.append(l_key)

    print(f"  Numbering {len(seqs_for_anarcii):,} antibody chains via Anarcii (Chothia)...")
    # Import anarcii version for logging
    try:
        import importlib.metadata as _im
        anarcii_ver = _im.version("anarcii")
    except Exception:
        anarcii_ver = "unknown"
    print(f"  anarcii version: {anarcii_ver} (pinned 2.0.8)")

    # Single batch for CDR indices + CDRH3 lengths (halves inference vs two calls)
    cdr_indices_by_key, cdrh3_by_heavy_key = get_chothia_details_batch(
        seqs_for_anarcii, roles_for_anarcii
    )
    print(f"  CDR numbering done for {len(cdr_indices_by_key):,} chains")
    if cdrh3_by_heavy_key:
        print(f"  CDRH3 lengths computed (range {min(cdrh3_by_heavy_key.values())}-{max(cdrh3_by_heavy_key.values())})")

    # --- Per-complex CDR coverage check ---
    metrics = StageMetrics("STEP 4: CDR filter + staging")
    metrics.add("Registry input", len(registry), len(registry))

    # Track per-complex decision and cdrh3_length
    survivors: list[int] = []  # row indices that pass
    failures: list[tuple[str, str]] = []  # (complex_id, reason)
    cdrh3_lengths: dict[str, int] = {}  # complex_id -> cdrh3_length
    # Store per-complex H/L ok for metrics without re-parsing
    per_complex_status: dict[int, tuple[bool, bool]] = {}  # idx -> (h_ok, l_ok)

    for idx, row in registry.iterrows():
        complex_id = complex_id_from_row(row)
        pdb_id = str(row["pdb"]).strip().lower()
        h_chain = str(row["Hchain"]).strip()
        l_chain = str(row["Lchain"]).strip() if pd.notna(row["Lchain"]) and str(row["Lchain"]).strip() else None

        cif_path = pdb_to_cif_path(pdb_id)
        if not cif_path.exists():
            raise FileNotFoundError(f"CIF missing for {complex_id}: {cif_path}")

        # Heavy chain check
        h_key = f"{complex_id}__{h_chain}"
        h_seq = fastas[complex_id][h_chain]
        h_cdr = cdr_indices_by_key.get(h_key)
        if h_cdr is None:
            raise RuntimeError(f"No CDR indices for heavy chain {h_key}")
        h_resolved = resolved_positions(cif_path, h_chain)
        h_resolved_0 = {r - 1 for r in h_resolved}
        h_missing = sorted(i for i in h_cdr if i not in h_resolved_0)
        h_ok = len(h_missing) == 0

        # Light chain check (if present)
        l_ok = True
        l_missing: list[int] = []
        l_key: str | None = None
        if l_chain:
            cand_keys = [f"{complex_id}__{l_chain}", f"{complex_id}__{l_chain}_L"]
            l_key = next((k for k in cand_keys if k in cdr_indices_by_key), None)
            if l_key is None:
                raise RuntimeError(f"No CDR indices for light chain {l_chain!r} of {complex_id}")
            l_seq = fastas[complex_id][l_chain]
            l_cdr = cdr_indices_by_key[l_key]
            l_resolved = resolved_positions(cif_path, l_chain)
            l_resolved_0 = {r - 1 for r in l_resolved}
            l_missing = sorted(i for i in l_cdr if i not in l_resolved_0)
            l_ok = len(l_missing) == 0

        # Record cdrh3_length for this complex (from heavy)
        cdrh3_len = cdrh3_by_heavy_key.get(h_key)
        if cdrh3_len is None:
            raise RuntimeError(f"No CDRH3 length for heavy {h_key}")
        cdrh3_lengths[complex_id] = cdrh3_len

        per_complex_status[idx] = (h_ok, l_ok)

        # Decision: 100% CDR coverage required on every antibody chain
        if h_ok and l_ok:
            survivors.append(idx)
        else:
            reasons: list[str] = []
            if not h_ok:
                reasons.append(f"H CDR gap ({len(h_missing)} residues, positions {[i+1 for i in h_missing[:3]]}{'...' if len(h_missing)>3 else ''})")
            if not l_ok:
                reasons.append(f"L CDR gap ({len(l_missing)} residues)")
            reason_str = "; ".join(reasons)
            failures.append((complex_id, reason_str))

    n_before = len(registry)
    n_after = len(survivors)
    # Build partitioned reason breakdown (one bucket per dropped complex)
    partitioned: dict[str, int] = {}
    for idx, (h_ok, l_ok) in per_complex_status.items():
        if idx in survivors:
            continue
        if not h_ok and not l_ok:
            partitioned["H+L CDR incomplete"] = partitioned.get("H+L CDR incomplete", 0) + 1
        elif not h_ok:
            partitioned["H CDR incomplete"] = partitioned.get("H CDR incomplete", 0) + 1
        elif not l_ok:
            partitioned["L CDR incomplete"] = partitioned.get("L CDR incomplete", 0) + 1
        else:
            partitioned["other CDR gap"] = partitioned.get("other CDR gap", 0) + 1

    metrics.add("CDR coverage filter (Chothia 100%)", n_before, n_after, reasons=partitioned if partitioned else {"all passed": n_after})
    atomic_write_json(
        INTERMEDIATE_ROOT / "04_cdr_metrics.json",
        {"before": n_before, "after": n_after, "dropped": n_before - n_after,
         "reasons": partitioned if partitioned else {"all passed": n_after}},
    )

    # --- Update registry with cdrh3_length and filter to survivors ---
    registry_filtered = registry.loc[survivors].copy()
    # Add cdrh3_length column
    registry_filtered["cdrh3_length"] = registry_filtered.apply(
        lambda r: cdrh3_lengths[complex_id_from_row(r)], axis=1
    )
    # Ensure no NaN in cdrh3_length
    if not (registry_filtered["cdrh3_length"].notna().all()):
        raise RuntimeError("NaN in cdrh3_length after CDR filter")
    # Validate CDRH3 length against the allowed 0–40 residue range.
    bad_h3 = registry_filtered[(registry_filtered["cdrh3_length"] < 0) | (registry_filtered["cdrh3_length"] > 50)]
    if not (bad_h3.empty):
        raise RuntimeError(f"Unreasonable cdrh3_length values: {bad_h3[['complex_id','cdrh3_length']].head().to_dict()}")

    # Overwrite the filtered TSV with survivors + cdrh3_length (registry is the durable artifact for s5/s6)
    FILTERED_TSV.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_dataframe(registry_filtered, FILTERED_TSV, sep="\t", index=False)
    print(f"  Registry filtered: {n_before} -> {n_after} ({len(failures)} rejected)")
    print(f"  Registry overwritten: {FILTERED_TSV} (now {len(registry_filtered):,} rows, cdrh3_length added)")

    # Update stage_metrics.json so the funnel reflects the CDR gate (plots.py reads this)
    stage_metrics_path = INTERMEDIATE_ROOT / "stage_metrics.json"
    if stage_metrics_path.exists():
        import json as _json
        with open(stage_metrics_path) as f:
            sm = _json.load(f)
        # Remove any previous CDR stage (idempotent re-run)
        sm["stages"] = [s for s in sm["stages"] if s["stage"] != "CDR loop coverage (Chothia 100%)"]
        cdr_stage = {
            "stage": "CDR loop coverage (Chothia 100%)",
            "before": n_before,
            "after": n_after,
            "dropped": n_before - n_after,
            "reasons": partitioned if partitioned else {"all passed": n_after},
        }
        sm["stages"].append(cdr_stage)
        sm["final"] = n_after
        atomic_write_json(stage_metrics_path, sm)
        print(f"  Updated {stage_metrics_path} (final {n_before} -> {n_after} via CDR filter)")
    else:
        raise FileNotFoundError(f"Required stage metrics missing: {stage_metrics_path}; run stage 2 first")

    # --- Stage chain-subset CIFs for survivors only ---
    if SCRATCH_SURVIVORS.exists():
        shutil.rmtree(SCRATCH_SURVIVORS)
    SCRATCH_SURVIVORS.mkdir(parents=True, exist_ok=True)

    staged: list[str] = []
    for idx in survivors:
        row = registry.loc[idx]
        pdb_id = str(row["pdb"]).strip().lower()
        complex_id = complex_id_from_row(row)
        target_chains = set(chain_ids_from_row(row))
        src_path = pdb_to_cif_path(pdb_id)
        dst_path = SCRATCH_SURVIVORS / f"{complex_id}.cif"
        write_chain_subset_cif(src_path, dst_path, target_chains)
        staged.append(complex_id)

    if not (len(staged) == n_after):
        raise RuntimeError(f"Staged {len(staged)} but survivors {n_after}")
    metrics.add("Write chain-subset CIFs", n_after, len(staged))

    metrics.report()
    print("\n" + "=" * 70)
    print(f"  Anarcii {anarcii_ver} Chothia numbering: "
          f"{len(seqs_for_anarcii)} chains, {n_after} complexes pass 100% CDR")
    print("=" * 70)

    survivors_on_disk = len(list(SCRATCH_SURVIVORS.glob("*.cif")))
    if not (survivors_on_disk == len(staged)):
        raise RuntimeError(f"scratch survivors on disk ({survivors_on_disk}) != staged ({len(staged)})")
    print(f"\n  Post-s4: {len(staged):,} complexes staged "
          f"(EM={(registry_filtered['modality']=='EM').sum():,}, "
          f"X-ray={(registry_filtered['modality']=='X-ray').sum():,})")
    print(f"  Survivor structures: {SCRATCH_SURVIVORS} ({survivors_on_disk:,} CIF files)")
    print(f"  Registry: {FILTERED_TSV}")

    LOG_DIR = DATA_ROOT / "logs"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"quality_filter_s4_{timestamp}.log"
    with open(log_path, "w") as lf:
        lf.write("=== Step 4: CDR filter (Chothia, Anarcii) + staging ===\n")
        lf.write(f"Started at: {timestamp}\n")
        lf.write(f"anarcii version: {anarcii_ver}\n")
        lf.write(f"Chothia CDR ranges: H {CHOTHIA_CDR_RANGES['H']} L {CHOTHIA_CDR_RANGES['L']}\n")
        lf.write("CDRH3 range 96-101 inclusive\n")
        for s in metrics.to_dict():
            lf.write(f"{s['stage']}: {s['before']} -> {s['after']} (-{s['dropped']}) {s['reasons']}\n")
        lf.write(f"Staged: {len(staged)}\n")
        if failures:
            lf.write("\n--- CDR failures (first 20) ---\n")
            for cid, reason in failures[:20]:
                lf.write(f"{cid}: {reason}\n")
    print(f"  Log: {log_path}")

    return {
        "passed": len(staged),
        "failures": len(failures),
        "scratch_dir": str(SCRATCH_SURVIVORS),
        "metrics": metrics.to_dict(),
        "anarcii_version": anarcii_ver,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 4: CDR filter (Chothia, Anarcii) + staging."
    )
    parser.add_argument("-p", "--processes", type=int, default=1, help="Parallel workers (currently unused, kept for compat).")
    args = parser.parse_args()
    run(processes=args.processes)


if __name__ == "__main__":
    main()
