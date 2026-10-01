#!/usr/bin/env python3
"""Export biological features for the final benchmark representatives.

- Read the final filtered registry.
- Write complex ID, PDB ID, modality, nanobody status, residue count, and CDRH3 length.
- Save dataset_features.csv.
"""
from __future__ import annotations

import argparse
import datetime
from pathlib import Path

import pandas as pd

from ..paths import DATA_ROOT, BENCHMARK_DIR
from ..utils import atomic_write_dataframe

FILTERED_TSV = BENCHMARK_DIR / "registry.tsv"
OUTPUT_CSV = BENCHMARK_DIR / "dataset_features.csv"

COLUMNS = [
    "complex_id", "pdb_id", "modality", "is_nanobody",
    "residue_length", "cdrh3_length",
]


def run() -> dict:
    """Write the final biological feature table and return export metadata."""
    print("\n" + "=" * 60)
    print("STEP 7: Assemble Dataset Features (final registry)")
    print("=" * 60)

    if not (FILTERED_TSV.exists()):
        raise RuntimeError(f"Final registry not found: {FILTERED_TSV}\n"
        f"Run s2/s4 first and s6 last — the registry must hold the "
        f"representative complexes.")

    registry = pd.read_csv(FILTERED_TSV, sep="\t", low_memory=False)
    if not (len(registry) > 0):
        raise RuntimeError(f"Final registry is empty: {FILTERED_TSV}")

    required = {"complex_id", "modality", "residue_length", "cdrh3_length"}
    missing_cols = required - set(registry.columns)
    if missing_cols:
        raise RuntimeError(f"Registry missing required columns: {sorted(missing_cols)}")

    base_cols = [c for c in COLUMNS if c in registry.columns]
    df = registry[base_cols].copy()
    df["pdb_id"] = df["complex_id"].str.split("_", n=1).str[0].str.lower()
    # is_nanobody: two chain letters after the pdb (H + antigen, no L).
    # complex_id is '<pdb>_<HLA>' concatenation (strict CIF eligibility
    # guarantees single-letter chain IDs), so a nanobody's chain part has
    # length 2.
    df["is_nanobody"] = df["complex_id"].str.split("_", n=1).str[1].str.len() == 2

    nan_rows = df[["modality", "residue_length", "cdrh3_length"]].isna().sum().to_dict()
    if any(nan_rows.values()):
        raise RuntimeError(f"NaN in assembled features — refusing to write silently: {nan_rows}")
    # cdrh3_length sanity: Chothia H3 is typically 3-35, allow 0-50
    if not (df["cdrh3_length"].between(0, 50).all()):
        raise RuntimeError(f"Unreasonable cdrh3_length in assembly: {df['cdrh3_length'].describe().to_dict()}")

    df = df[COLUMNS]
    atomic_write_dataframe(df, OUTPUT_CSV, index=False)

    n_em = int((df["modality"] == "EM").sum())
    n_xray = int((df["modality"] == "X-ray").sum())
    print(f"\n  Wrote {len(df):,} rows -> {OUTPUT_CSV}")
    print(f"  EM={n_em:,}  X-ray={n_xray:,}  nanobodies="
          f"{int(df['is_nanobody'].sum()):,}")

    LOG_DIR = DATA_ROOT / "logs"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"assemble_features_{timestamp}.log"
    with open(log_path, "w") as lf:
        lf.write("=== Step 7: Assemble features ===\n")
        lf.write(f"Started at: {timestamp}\n")
        lf.write(f"Total: {len(df)}\n")
        lf.write(f"Columns: {list(df.columns)}\n")
        lf.write("No in_training_* flags by design (novelty guaranteed by s2+s5).\n")
    print(f"  Log: {log_path}")

    return {
        "total": len(df),
        "n_em": n_em,
        "n_xray": n_xray,
        "output_path": str(OUTPUT_CSV),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 7: Assemble dataset features (post-s6)."
    )
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
