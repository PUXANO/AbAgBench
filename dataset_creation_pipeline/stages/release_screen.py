#!/usr/bin/env python3
"""Identify SAbDab2 structures released before the training cutoff.

- Read PDB release dates; default cutoff: 2023-06-01.
- Export pre-cutoff PDB IDs and release-date summary statistics.
- Reject missing or invalid release dates.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path

import pandas as pd

from ..paths import INTERMEDIATE_ROOT, SABDAB2_SUMMARY_CSV
from ..utils import StageMetrics, atomic_write_json, normalize_sabdab2_summary

RAW_SUMMARY = SABDAB2_SUMMARY_CSV
IDS_OUTPUT = INTERMEDIATE_ROOT / "training_data_ids.json"
MATCHES_OUTPUT = INTERMEDIATE_ROOT / "training_data_matches.txt"

# Default training cutoff: Boltz-2 was trained on PDB structures released
# before 2023-06-01 (Passaro et al. 2025).  With Boltz-2 as the benchmark
# baseline and Protenix-v2 / OpenDDE opendde_v1 both at 2021-09-30, this is
# the latest (binding) cutoff — override with --cutoff if the model set
# changes.
CUTOFF_DEFAULT = "2023-06-01"


def parse_cutoff(value: str) -> datetime.date:
    """Validate a --cutoff value (ISO YYYY-MM-DD) and return it as a date."""
    try:
        return datetime.date.fromisoformat(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"invalid cutoff {value!r}: expected an ISO date YYYY-MM-DD"
        ) from e


def load_release_dates(summary: pd.DataFrame) -> pd.Series:
    """Parse PDB release dates in YYYY/MM/DD format; reject missing or invalid values."""
    if "date" not in summary.columns:
        raise ValueError(
            f"SAbDab2 summary has no 'date' column (PDB release date); got "
            f"{sorted(summary.columns)[:12]}..."
        )
    raw = summary["date"].astype(str).str.strip()
    empty = raw.eq("") | raw.eq("nan") | raw.isna()
    if empty.any():
        raise ValueError(
            f"{len(empty):d} rows have an empty 'date' (PDB release date) — "
            f"cannot run a cutoff de-leak on them"
        )
    parsed = pd.to_datetime(raw, format="%Y/%m/%d", errors="coerce")
    bad = parsed.isna()
    if bad.any():
        bad_ids = summary.loc[bad, "pdb"].astype(str).str.lower().head(10).tolist()
        raise ValueError(
            f"{int(bad.sum()):d} rows have an unparseable 'date': {bad_ids}... "
            f"expected format YYYY/MM/DD"
        )
    return parsed.dt.date


def run(cutoff: str = CUTOFF_DEFAULT) -> dict:
    """Export pre-cutoff PDB IDs and return release-date metadata."""
    cutoff_date = parse_cutoff(cutoff)

    print("=" * 60)
    print("STEP 1: Training Data Check (cutoff-based de-leak)")
    print(f"        cutoff: {cutoff_date.isoformat()} "
          f"(default Boltz-2 training cutoff: {CUTOFF_DEFAULT})")
    print("=" * 60)

    if RAW_SUMMARY is None or not RAW_SUMMARY.exists():
        configured_path = RAW_SUMMARY or "(not configured)"
        raise FileNotFoundError(
            f"SAbDab2 summary not found: {configured_path}; pass --sabdab-dir."
        )

    summary = pd.read_csv(RAW_SUMMARY, low_memory=False)
    if not (len(summary) > 0):
        raise RuntimeError(f"Raw SAbDab2 summary is empty: {RAW_SUMMARY}")
    summary = normalize_sabdab2_summary(summary)

    release_dates = load_release_dates(summary)
    summary = summary.copy()
    summary["_release_date"] = release_dates

    n_raw = len(summary)
    raw_pdb_ids = set(summary["pdb"])

    pre_mask = summary["_release_date"] < cutoff_date
    n_pre_rows = int(pre_mask.sum())
    pre_pdb_ids = set(summary.loc[pre_mask, "pdb"])
    n_post_rows = n_raw - n_pre_rows
    post_pdb_ids = raw_pdb_ids - pre_pdb_ids

    metrics = StageMetrics("STEP 1: training data check (cutoff de-leak)")
    metrics.add("Raw SAbDab summary (rows)", n_raw, n_raw)
    reasons = {
        f"released before {cutoff_date.isoformat()} (in training)": n_pre_rows
    }
    metrics.add(
        "After cutoff de-leak (rows)", n_raw, n_post_rows, reasons=reasons,
    )
    metrics.report()

    print(f"\n  Raw SAbDab summary: {n_raw:,} rows, {len(raw_pdb_ids):,} unique PDB IDs")
    print(f"  Pre-cutoff ({cutoff_date.isoformat()}): {n_pre_rows:,} rows "
          f"({len(pre_pdb_ids):,} unique PDB IDs)  -> flagged as in training")
    print(f"  Post-cutoff (novel by ID): {n_post_rows:,} rows "
          f"({len(post_pdb_ids):,} unique PDB IDs)")
    if not (n_post_rows > 0):
        raise RuntimeError(f"No SAbDab complexes released on/after {cutoff_date.isoformat()} — "
        f"the cutoff de-leak removed the entire dataset. Re-check --cutoff.")

    # Write the shared artifact consumed by s2 and plots.
    ids_json = {
        "cutoff": cutoff_date.isoformat(),
        "cutoff_default": cutoff == CUTOFF_DEFAULT,
        "raw_summary_total": n_raw,
        "raw_unique_pdb": len(raw_pdb_ids),
        "pre_cutoff_rows": n_pre_rows,
        "pre_cutoff_unique_pdb": len(pre_pdb_ids),
        "post_cutoff_rows": n_post_rows,
        "post_cutoff_unique_pdb": len(post_pdb_ids),
        "union_in_training": len(pre_pdb_ids),
        "ids": sorted(pre_pdb_ids),
    }
    atomic_write_json(IDS_OUTPUT, ids_json)
    print(f"\n  Training de-leak artifact: {IDS_OUTPUT}")

    # Detailed pre-cutoff ID report (the future s5 FoldSeek reference set).
    with open(MATCHES_OUTPUT, "w") as f:
        f.write(f"Training de-leak (cutoff {cutoff_date.isoformat()})\n\n")
        f.write(f"Raw summary: {n_raw:,} rows, {len(raw_pdb_ids):,} unique PDB IDs\n")
        f.write(f"Pre-cutoff (in training): {n_pre_rows:,} rows, "
                f"{len(pre_pdb_ids):,} unique PDB IDs\n")
        f.write(f"Post-cutoff (novel by ID): {n_post_rows:,} rows, "
                f"{len(post_pdb_ids):,} unique PDB IDs\n")
        f.write(f"\nPre-cutoff PDB IDs ({len(pre_pdb_ids):,}):\n")
        for pdb_id in sorted(pre_pdb_ids):
            f.write(f"  {pdb_id}\n")
    print(f"  Pre-cutoff ID list (future s5 reference set): {MATCHES_OUTPUT}")

    LOG_DIR = IDS_OUTPUT.parents[1] / "logs"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"training_data_check_{timestamp}.log"
    with open(log_path, "w") as lf:
        lf.write("=== Training Data Check (cutoff de-leak) ===\n")
        lf.write(f"Started at: {timestamp}\n")
        lf.write(f"Cutoff: {cutoff_date.isoformat()} "
                 f"(default={cutoff == CUTOFF_DEFAULT})\n")
        lf.write(f"Raw rows: {n_raw}\n")
        lf.write(f"Pre-cutoff rows: {n_pre_rows} "
                 f"({len(pre_pdb_ids)} unique IDs)\n")
        lf.write(f"Post-cutoff rows: {n_post_rows} "
                 f"({len(post_pdb_ids)} unique IDs)\n")
        lf.write(f"=== Ended at: {datetime.datetime.now().strftime('%Y%m%d_%H%M%S')} ===\n")
    print(f"  Log: {log_path}")

    return {
        "cutoff": cutoff_date.isoformat(),
        "n_raw": n_raw,
        "raw_unique_pdb": len(raw_pdb_ids),
        "pre_cutoff_rows": n_pre_rows,
        "pre_cutoff_unique_pdb": len(pre_pdb_ids),
        "post_cutoff_rows": n_post_rows,
        "post_cutoff_unique_pdb": len(post_pdb_ids),
        "ids_output": str(IDS_OUTPUT),
        "metrics": metrics.to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 1: cutoff-based training de-leak. Flags every "
                    "SAbDab2 complex whose PDB release date is before the "
                    "training cutoff (default: Boltz-2 cutoff 2023-06-01)."
    )
    parser.add_argument(
        "--cutoff", default=CUTOFF_DEFAULT, type=parse_cutoff,
        help="Training cutoff date (YYYY-MM-DD). Complexes released before "
             "this date are treated as 'in training' and dropped. "
             f"Default: {CUTOFF_DEFAULT} (Boltz-2).",
    )
    args = parser.parse_args()
    run(cutoff=args.cutoff.isoformat())


if __name__ == "__main__":
    main()
