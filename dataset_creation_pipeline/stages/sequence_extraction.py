#!/usr/bin/env python3
"""Extract full chain sequences from SAbDab2 mmCIF records.

- Read _entity_poly_seq, including unresolved residues.
- Return a complex-to-chain sequence mapping for CDR checks and FASTA export.
- Reject missing entity sequences.
"""
from __future__ import annotations

import argparse
import datetime
from pathlib import Path

import pandas as pd

from ..paths import DATA_ROOT, INTERMEDIATE_ROOT
from ..utils import (
    StageMetrics,
    chain_full_sequences,
    chain_ids_from_row,
    complex_id_from_row,
    pdb_to_cif_path,
)

FILTERED_TSV = INTERMEDIATE_ROOT / "02_quality_filtered.tsv"


def build_fasta_from_seqres(
    registry: pd.DataFrame,
) -> tuple[dict[str, dict[str, str]], StageMetrics]:
    """Return {complex_id: {chain: sequence}} for every registry row."""
    metrics = StageMetrics("STEP 3: in-memory FASTA")
    n_complexes = len(registry)
    fastas: dict[str, dict[str, str]] = {}

    for _, row in registry.iterrows():
        complex_id = complex_id_from_row(row)
        if complex_id in fastas:
            raise ValueError(
                f"Duplicate complex_id {complex_id} in post-s2 registry — "
                f"chain-suffix collision; fix the registry before proceeding."
            )
        target_chains = set(chain_ids_from_row(row))
        cif_path = pdb_to_cif_path(str(row["pdb"]).strip().lower())
        if not cif_path.exists():
            raise FileNotFoundError(
                f"SAbDab2 CIF not found for FASTA: {cif_path} ({complex_id})"
            )
        sequences = chain_full_sequences(cif_path, target_chains)
        found_chains = set(sequences)
        missing = target_chains - found_chains
        if missing:
            raise ValueError(
                f"Chains {sorted(missing)} of {complex_id} have no sequence "
                f"in {cif_path}"
            )
        fastas[complex_id] = sequences

    metrics.add("Build in-memory FASTA", n_complexes, len(fastas))
    return fastas, metrics


def run(processes: int | None = None) -> dict:
    """Return full chain sequences and extraction metadata."""
    print("\n" + "=" * 60)
    print("STEP 3: SAbDab2 CIF -> FASTA (in memory, entity_poly_seq)")
    print("=" * 60)

    if not (FILTERED_TSV.exists()):
        raise RuntimeError(f"Post-s2 registry not found: {FILTERED_TSV}\n"
        f"Run s2_quality_filter first.")

    registry = pd.read_csv(FILTERED_TSV, sep="\t", low_memory=False)
    if not (len(registry) > 0):
        raise RuntimeError(f"Post-s2 registry is empty: {FILTERED_TSV}")

    fastas, metrics = build_fasta_from_seqres(registry)
    metrics.report()
    print(f"\n  In-memory FASTA for {len(fastas):,} complexes "
          f"({sum(len(v) for v in fastas.values()):,} chains).")

    LOG_DIR = DATA_ROOT / "logs"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"pdb_to_fasta_{timestamp}.log"
    with open(log_path, "w") as lf:
        lf.write("=== Step 3: SAbDab2 CIF to FASTA (in memory) ===\n")
        lf.write(f"Started at: {timestamp}\n")
        lf.write(f"Complexes: {len(fastas)}\n")
        lf.write(f"Chains: {sum(len(v) for v in fastas.values())}\n")
        lf.write("No coverage computation performed (see 20 Aug 2026 redesign).\n")
    print(f"  Log: {log_path}")

    return {
        "fastas": fastas,
        "n_complexes": len(fastas),
        "n_chains": sum(len(v) for v in fastas.values()),
        "metrics": metrics.to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 3: Build in-memory per-chain FASTA from SAbDab2 mmCIF."
    )
    parser.add_argument("-p", "--processes", type=int, default=None, help="Unused (kept for CLI compatibility).")
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
