#!/usr/bin/env python3
"""Screen structural training overlap with FoldSeek multimer search.

- Search candidate complexes against whole pre-cutoff SAbDab2 structures.
- Require hits covering every query chain.
- Exclude hits with I-LDDT >= 0.7 and query antibody–antigen interface coverage >= 0.5.
- Define interface contacts by Cα distances within 8 Å.
- Discard queries without a measurable full-complex hit.
- Export retained structures, registry rows, and overlap diagnostics.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..foldseek_utils import (
    FOLDSEEK_FORMAT,
    ILDDT_THRESHOLD,
    ILDDT_THRESHOLDS,
    COVERAGE_THRESHOLD,
    REFERENCE_MIN,
    best_full_complex_hits,
    compute_query_interfaces,
    compute_query_interfaces_pool,
    compute_coverage_pool,
    decide_novelty,
    derive_best,
    filter_antigen_present,
    find_foldseek_bin,
    load_chain_alignments,
    parse_complex_chains,
    parse_report,
    run_foldseek,
)
from ..paths import LOGS_ROOT, DATA_ROOT, INTERMEDIATE_ROOT, SCRATCH_ROOT, SABDAB2_STRUCTURE_DIR
from ..utils import (
    StageMetrics,
    atomic_write_dataframe,
    atomic_write_json,
    complex_id_from_row,
    materialize_cif,
    pdb_to_cif_path,
)

TRAINING_IDS_JSON = INTERMEDIATE_ROOT / "training_data_ids.json"
INPUT_TSV = INTERMEDIATE_ROOT / "04_cdr_filtered.tsv"
FILTERED_TSV = INTERMEDIATE_ROOT / "05_deleaked.tsv"
STAGE_METRICS_PATH = INTERMEDIATE_ROOT / "stage_metrics.json"
SCRATCH_SURVIVORS = SCRATCH_ROOT / "s4_survivors"
SCRATCH_TRAINING = SCRATCH_ROOT / "s5_training_db"
SCRATCH_FOLDSEEK = SCRATCH_ROOT / "s5_foldseek"
SCRATCH_DELEAKED = SCRATCH_ROOT / "s5_deleaked"
REPORT_PATH = SCRATCH_FOLDSEEK / "results.tsv_report"
TSV_PATH = SCRATCH_FOLDSEEK / "results.tsv"
INTERFACE_CACHE = SCRATCH_FOLDSEEK / "interface_cache.pkl"

STAGE_NAME = "FoldSeek structural de-leak (I-LDDT>=0.7 & cov>=0.5)"
REASON_NO_EVIDENCE = "no full-complex hit (NaN, conservative)"


# ── Training DB build ────────────────────────────────────────────────────────

def build_training_db() -> dict:
    """Symlink each whole pre-cutoff SAbDab2 mmCIF into the training reference directory."""
    if not (TRAINING_IDS_JSON.is_file()):
        raise RuntimeError(f"Step 1 artifact missing: {TRAINING_IDS_JSON} — run s1 first.")
    with open(TRAINING_IDS_JSON) as f:
        ids_data = json.load(f)
    cutoff = ids_data["cutoff"]
    pre_cutoff_ids = set(ids_data["ids"])
    if not (cutoff and pre_cutoff_ids):
        raise RuntimeError(f"Step 1 artifact has no cutoff/ids: {TRAINING_IDS_JSON}")
    if not (len(pre_cutoff_ids) >= REFERENCE_MIN):
        raise RuntimeError(f"Pre-cutoff PDB set too small: {len(pre_cutoff_ids)} < "
        f"{REFERENCE_MIN} — the structural de-leak cannot be trusted")
    if SABDAB2_STRUCTURE_DIR is None or not SABDAB2_STRUCTURE_DIR.is_dir():
        raise FileNotFoundError(f"SAbDab2 structures missing: {SABDAB2_STRUCTURE_DIR or '(not configured)'}")

    if SCRATCH_TRAINING.exists():
        shutil.rmtree(SCRATCH_TRAINING)
    SCRATCH_TRAINING.mkdir(parents=True)

    missing: list[str] = []
    for pdb_id in sorted(pre_cutoff_ids):
        src = pdb_to_cif_path(pdb_id)
        if not src.is_file():
            missing.append(str(src))
            continue
        materialize_cif(src, SCRATCH_TRAINING / f"{pdb_id}.cif")
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} pre-cutoff SAbDab2 CIFs missing from the "
            f"download: {missing[:5]}{' ...' if len(missing) > 5 else ''} — "
            f"check the configured SAbDab2 structure directory "
            f"{SABDAB2_STRUCTURE_DIR}"
        )

    n_staged = len(list(SCRATCH_TRAINING.glob("*.cif")))
    if not (n_staged == len(pre_cutoff_ids)):
        raise RuntimeError(f"Training DB staged ({n_staged}) != pre-cutoff PDBs "
        f"({len(pre_cutoff_ids)})")
    print(f"\n  Training DB staged: {n_staged:,} whole SAbDab2 CIFs "
          f"(pre-cutoff, cutoff {cutoff}) -> {SCRATCH_TRAINING}")

    metrics = StageMetrics("STEP 5: training DB build (pre-cutoff, whole CIFs)")
    metrics.add("Pre-cutoff PDBs (s1 ids)", n_staged, n_staged)
    metrics.report()
    return {"cutoff": cutoff, "n_training": n_staged}


# ── FoldSeek DB build + search ──────────────────────────────────────────────

def run_foldseek_search(
    threads: int, gpu: int, foldseek_bin: str | None, dry_run: bool
) -> None:
    """Build queryDB/targetDB(+pad) and run easy-multimersearch."""
    for d in (SCRATCH_SURVIVORS, SCRATCH_TRAINING):
        if not (d.is_dir()):
            raise RuntimeError(f"Missing FoldSeek input dir: {d}")
        n = len(list(d.glob("*.cif")))
        if not (n > 0):
            raise RuntimeError(f"Empty FoldSeek input dir: {d}")

    if SCRATCH_FOLDSEEK.exists():
        shutil.rmtree(SCRATCH_FOLDSEEK)
    SCRATCH_FOLDSEEK.mkdir(parents=True)

    query_db = SCRATCH_FOLDSEEK / "queryDB"
    target_db = SCRATCH_FOLDSEEK / "targetDB"
    target_db_pad = SCRATCH_FOLDSEEK / "targetDB_pad"
    results_tsv = SCRATCH_FOLDSEEK / "results.tsv"
    search_tmp = SCRATCH_FOLDSEEK / "tmp_search"

    if dry_run:
        # Dry-run must work without a foldseek binary (e.g. the local Mac):
        # print the exact commands that a real run would execute.
        for cmd in [
            ["foldseek", "createdb", str(SCRATCH_SURVIVORS), str(query_db),
             "--threads", str(threads)],
            ["foldseek", "createdb", str(SCRATCH_TRAINING), str(target_db),
             "--threads", str(threads)],
            ["foldseek", "makepaddedseqdb", str(target_db), str(target_db_pad),
             "--threads", str(threads)],
            [
                "foldseek", "easy-multimersearch",
                str(query_db), str(target_db_pad),
                str(results_tsv), str(search_tmp),
                "--alignment-type", "2",
                "--format-output", FOLDSEEK_FORMAT,
                "--threads", str(threads),
                "--prefilter-mode", "1",
            ] + (["--gpu", "1"] if gpu else []),
        ]:
            run_foldseek(cmd, dry_run=True)
        return

    bin_path = find_foldseek_bin(foldseek_bin)
    print(f"  foldseek binary: {bin_path}")

    run_foldseek(
        [bin_path, "createdb", str(SCRATCH_SURVIVORS), str(query_db),
         "--threads", str(threads)],
        dry_run,
    )
    run_foldseek(
        [bin_path, "createdb", str(SCRATCH_TRAINING), str(target_db),
         "--threads", str(threads)],
        dry_run,
    )
    run_foldseek(
        [bin_path, "makepaddedseqdb", str(target_db), str(target_db_pad),
         "--threads", str(threads)],
        dry_run,
    )
    cmd = [
        bin_path, "easy-multimersearch",
        str(query_db), str(target_db_pad),
        str(results_tsv), str(search_tmp),
        "--alignment-type", "2",
        "--format-output", FOLDSEEK_FORMAT,
        "--threads", str(threads),
        "--prefilter-mode", "1",
    ]
    if gpu:
        cmd += ["--gpu", "1"]
    run_foldseek(cmd, dry_run)

    if not results_tsv.is_file():
        raise RuntimeError(f"FoldSeek wrote no results: {results_tsv}")
    if not REPORT_PATH.is_file():
        raise RuntimeError(f"FoldSeek wrote no report: {REPORT_PATH}")


# ── Parsing & decision (5-stage port of precompute_edges.py) ────────────────

def parse_and_decide(processes: int) -> tuple[dict, dict, dict]:
    """Return coverage-qualified hits, diagnostics, and best full-complex hits."""
    unique_queries, candidates, diagnostics = parse_report(REPORT_PATH)
    if not candidates:
        raise RuntimeError(
            f"Zero candidate hits (I-LDDT >= {ILDDT_THRESHOLD}) against the "
            f"training DB — impossible with {REFERENCE_MIN}+ pre-cutoff "
            f"training structures; the de-leak cannot be trusted. "
            f"Check the report {REPORT_PATH} and the foldseek invocation."
        )

    interfaces, chain_meta = compute_query_interfaces_pool(
        sorted(unique_queries), SCRATCH_SURVIVORS, INTERFACE_CACHE, processes
    )

    print("\n  B. Antigen-present filter...")
    coverage_hits, needed_pairs, n_no_antigen = filter_antigen_present(
        candidates, chain_meta
    )
    diagnostics["n_no_antigen"] = n_no_antigen

    print("\n  C. Loading chain alignments (needed pairs only)...")
    t0 = time.time()
    chain_alignments = load_chain_alignments(TSV_PATH, needed_pairs)
    print(f"     Chain-level alignments kept: {len(chain_alignments):,} "
          f"({time.time() - t0:.0f}s)")

    results = compute_coverage_pool(
        coverage_hits, interfaces, chain_alignments, processes
    )
    diagnostics["n_cov_hits"] = len(results)

    best, cov_values = derive_best(results)
    diagnostics["cov_values"] = cov_values
    diagnostics["best_values"] = [v[0] for v in best.values()]

    print("\n  E. Best full-complex hit over ALL I-LDDT values "
          "(incl. < 0.7)...")
    best_full = best_full_complex_hits(REPORT_PATH, chain_meta)
    diagnostics["n_queries_with_full_complex_hit"] = len(best_full)
    diagnostics["n_nan_queries"] = len(unique_queries) - len(best_full)
    print(f"     Queries with >= 1 full-complex hit: {len(best_full):,} "
          f"({len(unique_queries) - len(best_full):,} NaN — no full-complex "
          f"hit at any I-LDDT)")
    return best, diagnostics, best_full


# ── Main ────────────────────────────────────────────────────────────────────

def run(
    processes: int = 16,
    threads: int = 16,
    gpu: int = 0,
    foldseek_bin: str | None = None,
    dry_run: bool = False,
) -> dict:
    """Screen structural training overlap and export surviving complexes."""
    print("\n" + "=" * 60)
    print("STEP 5: FoldSeek structural de-leak")
    print("=" * 60)

    if not INPUT_TSV.exists():
        raise FileNotFoundError(
        f"Post-s4 registry not found: {INPUT_TSV}\n"
        f"Run s1..s4 first."
        )
    registry = pd.read_csv(INPUT_TSV, sep="\t", low_memory=False)
    if not (len(registry) > 0):
        raise RuntimeError(f"Post-s4 registry is empty: {FILTERED_TSV}")
    if not ("complex_id" in registry.columns):
        raise RuntimeError("Registry missing complex_id")
    if not ("cdrh3_length" in registry.columns):
        raise RuntimeError("Registry missing cdrh3_length")
    n_queries = len(registry)
    survivors_on_disk = len(list(SCRATCH_SURVIVORS.glob("*.cif")))
    if not (survivors_on_disk == n_queries):
        raise RuntimeError(f"s4 survivor CIFs ({survivors_on_disk}) != registry rows ({n_queries})")
    # complex_id column consistency (never trust a silent divergence)
    for _, row in registry.iterrows():
        cid = complex_id_from_row(row)
        if cid != str(row["complex_id"]):
            raise ValueError(
                f"complex_id column {row['complex_id']!r} != derived "
                f"{cid!r} for PDB {row['pdb']}"
            )

    train_meta = build_training_db()

    run_foldseek_search(threads, gpu, foldseek_bin, dry_run)
    if dry_run:
        print("\n  [DRY-RUN] validating interfaces on 5 sample queries...")
        for q in sorted(registry["complex_id"])[:5]:
            h, l, a = parse_complex_chains(q)
            parts = compute_query_interfaces(
                SCRATCH_SURVIVORS / f"{q}.cif", h, l, a
            )
            sizes = {
                c: len(iface.get(c, set()))
                for iface, chains in parts for c in chains
            }
            print(f"     {q}: chains H={h} L={l} A={a}, "
                  f"interface residues {sizes}")
        print("\n  [DRY-RUN] stopped before the FoldSeek search — run without "
              "--dry-run on the HPC.")
        return {"dry_run": True, "n_queries": n_queries, **train_meta}

    best, diagnostics, best_full = parse_and_decide(processes)

    print("\n  Deciding novelty (I-LDDT first, then coverage; "
          "NaN -> conservative discard)...")
    novel_ids: list[str] = []
    leaked: dict[str, str] = {}
    best_ilddts: list[float] = []
    n_novel_below_07 = 0
    n_novel_cov_fail = 0
    for _, row in registry.iterrows():
        cid = str(row["complex_id"])
        top = best_full.get(cid)   # best full-complex hit, ANY I-LDDT
        b = best.get(cid)          # best full-complex hit with cov >= 0.5
        outcome = decide_novelty(top, b)
        if outcome == "nan":
            leaked[cid] = REASON_NO_EVIDENCE
        elif outcome == "leaked":
            if not (b is not None and top is not None):
                raise RuntimeError(f"{cid}: leaked without full-complex evidence")
            leaked[cid] = (
                f"best I-LDDT {b[0]:.3f} >= {ILDDT_THRESHOLD} with coverage "
                f"{b[2]:.2f} >= {COVERAGE_THRESHOLD} (best target {b[1]})"
            )
        else:
            if not (top is not None):
                raise RuntimeError(f"{cid}: novel without full-complex evidence")
            novel_ids.append(cid)
            best_ilddts.append(top[0])
            if top[0] < ILDDT_THRESHOLD:
                n_novel_below_07 += 1
            else:
                n_novel_cov_fail += 1
    if not (len(novel_ids) + len(leaked) == n_queries):
        raise RuntimeError(f"Decision covered {len(novel_ids) + len(leaked)}/{n_queries} queries")
    n_na = sum(1 for r in leaked.values() if r == REASON_NO_EVIDENCE)
    n_ge = len(leaked) - n_na

    metrics = StageMetrics("STEP 5: FoldSeek structural de-leak")
    metrics.add("Registry input (post-s4)", n_queries, n_queries)
    metrics.add(
        STAGE_NAME,
        n_queries, len(novel_ids),
        reasons={
            f"best I-LDDT >= {ILDDT_THRESHOLD} & cov >= {COVERAGE_THRESHOLD}":
                n_ge,
            REASON_NO_EVIDENCE: n_na,
        },
    )
    atomic_write_json(
        INTERMEDIATE_ROOT / "05_deleak_metrics.json",
        {"before": n_queries, "after": len(novel_ids), "dropped": n_queries - len(novel_ids),
         "reasons": {f"best I-LDDT >= {ILDDT_THRESHOLD} & cov >= {COVERAGE_THRESHOLD}": n_ge,
                     REASON_NO_EVIDENCE: n_na}},
    )
    metrics.report()

    # Overwrite registry with novel rows only (all columns preserved).
    registry_novel = registry.loc[registry["complex_id"].isin(novel_ids)].copy()
    if not (len(registry_novel) == len(novel_ids)):
        raise RuntimeError(f"Novel registry rows ({len(registry_novel)}) != novel ids "
        f"({len(novel_ids)})")
    FILTERED_TSV.parent.mkdir(parents=True, exist_ok=True)
    registry_novel.to_csv(FILTERED_TSV, sep="\t", index=False)
    print(f"  Registry written: {FILTERED_TSV} "
          f"({len(registry_novel):,} novel rows)")

    # Stage novel CIFs for s6.
    if SCRATCH_DELEAKED.exists():
        shutil.rmtree(SCRATCH_DELEAKED)
    SCRATCH_DELEAKED.mkdir(parents=True)
    for cid in novel_ids:
        src = SCRATCH_SURVIVORS / f"{cid}.cif"
        if not (src.is_file()):
            raise RuntimeError(f"Survivor CIF missing: {src}")
        shutil.copy2(src, SCRATCH_DELEAKED / f"{cid}.cif")
    deleaked_on_disk = len(list(SCRATCH_DELEAKED.glob("*.cif")))
    if not (deleaked_on_disk == len(novel_ids)):
        raise RuntimeError(f"s5_deleaked CIFs ({deleaked_on_disk}) != novel ({len(novel_ids)})")
    print(f"  Novel CIFs staged: {SCRATCH_DELEAKED} ({deleaked_on_disk:,})")

    # Append the de-leak stage to stage_metrics.json (idempotent re-run).
    if STAGE_METRICS_PATH.exists():
        with open(STAGE_METRICS_PATH) as f:
            sm = json.load(f)
        sm["stages"] = [s for s in sm["stages"] if s["stage"] != STAGE_NAME]
        sm["stages"].append({
            "stage": STAGE_NAME,
            "before": n_queries,
            "after": len(novel_ids),
            "dropped": n_queries - len(novel_ids),
            "reasons": {
                f"best I-LDDT >= {ILDDT_THRESHOLD} & cov >= {COVERAGE_THRESHOLD}":
                    n_ge,
                REASON_NO_EVIDENCE: n_na,
            },
        })
        sm["final"] = len(novel_ids)
        atomic_write_json(STAGE_METRICS_PATH, sm)
        print(f"  stage_metrics.json updated (final {n_queries} -> "
              f"{len(novel_ids)})")
    else:
        raise FileNotFoundError(f"Required stage metrics missing: {STAGE_METRICS_PATH}; run stage 2 first")

    _write_leakage_summary(
        n_queries, len(novel_ids), n_na, n_novel_below_07, n_novel_cov_fail,
        best_ilddts, diagnostics, registry, registry_novel, train_meta,
    )
    _write_log(n_queries, novel_ids, leaked, diagnostics, train_meta, metrics)

    return {
        "n_queries": n_queries,
        "n_novel": len(novel_ids),
        "n_leaked": len(leaked),
        "n_nan_dropped": n_na,
        "n_training": train_meta["n_training"],
        "scratch_dir": str(SCRATCH_DELEAKED),
        "metrics": metrics.to_dict(),
    }


def _write_leakage_summary(
    n_queries, n_novel, n_na, n_novel_below_07, n_novel_cov_fail,
    best_ilddts, diagnostics, registry, registry_novel, train_meta,
) -> None:
    out = SCRATCH_FOLDSEEK / "leakage_summary.txt"
    lines: list[str] = []
    lines.append("=== FoldSeek structural de-leak summary ===")
    lines.append(f"Queries (post-s4):          {n_queries:,}")
    lines.append(f"Training DB (pre-cutoff):   {train_meta['n_training']:,} "
                 f"whole SAbDab2 structures (as-is)")
    lines.append(f"Novel (kept):                {n_novel:,}")
    lines.append(f"  - best full-complex I-LDDT < 0.7:      {n_novel_below_07:,}")
    lines.append(f"  - >= 0.7 hits, all cov < 0.5:          {n_novel_cov_fail:,}")
    lines.append(f"Leaked (dropped):            {n_queries - n_novel:,}")
    lines.append(f"  - best I-LDDT >= 0.7 & cov >= 0.5:     "
                 f"{n_queries - n_novel - n_na:,}")
    lines.append(f"  - no full-complex hit (NaN):           {n_na:,}")

    leaked_mask = ~registry["complex_id"].isin(set(registry_novel["complex_id"]))
    mod_leaked = registry.loc[leaked_mask, "modality"].value_counts().to_dict()
    mod_novel = registry_novel["modality"].value_counts().to_dict()
    lines.append(f"Leaked by modality: {mod_leaked}")
    lines.append(f"Novel by modality:  {mod_novel}")

    if best_ilddts:
        arr = np.array(best_ilddts)
        lines.append(f"Novel best I-LDDT: n={len(arr)} min={arr.min():.3f} "
                     f"median={np.median(arr):.3f} max={arr.max():.3f}")
    if diagnostics["cov_values"]:
        covs = np.array(diagnostics["cov_values"])
        bins = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        counts, _ = np.histogram(covs, bins=bins)
        lines.append("Coverage histogram (non-self, I-LDDT >= 0.7):")
        for i in range(len(bins) - 1):
            lines.append(f"  [{bins[i]:.1f}, {bins[i+1]:.1f}): {counts[i]}")
    lines.append("I-LDDT pass counts (whole report):")
    for thr in ILDDT_THRESHOLDS:
        lines.append(f"  I-LDDT >= {thr}: {diagnostics['ilddt_pass'].get(thr, 0)}")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"\n  Leakage summary: {out}")


def _write_log(
    n_queries, novel_ids, leaked, diagnostics, train_meta, metrics,
) -> None:
    log_dir = LOGS_ROOT
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"s5_foldseek_deleak_{ts}.log"
    with open(log_path, "w") as f:
        f.write("=== Step 5: FoldSeek structural de-leak ===\n")
        f.write(f"Started at: {ts}\n")
        f.write(f"Queries: {n_queries}\n")
        f.write(f"Training DB staged: {train_meta['n_training']}\n")
        f.write(f"Novel: {len(novel_ids)}  Leaked: {len(leaked)}\n")
        for s in metrics.to_dict():
            f.write(f"{s['stage']}: {s['before']} -> {s['after']} "
                    f"(-{s['dropped']}) {s['reasons']}\n")
        f.write("\n--- Leaked complexes (reason) ---\n")
        for cid in sorted(leaked):
            f.write(f"{cid}: {leaked[cid]}\n")
    print(f"  Log: {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 5: FoldSeek structural de-leak vs the pre-cutoff "
                    "training DB."
    )
    parser.add_argument("-p", "--processes", type=int, default=16,
                        help="Parallel workers for interface/coverage stages.")
    parser.add_argument("--threads", type=int, default=16,
                        help="foldseek --threads.")
    parser.add_argument("--gpu", type=int, default=0, choices=[0, 1],
                        help="Pass --gpu 1 to foldseek (HPC only).")
    parser.add_argument("--foldseek-bin", default=None,
                        help="Path to the foldseek binary (else FOLDSEEK_BIN "
                             "env, else PATH).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Stage training DB + work dirs, validate "
                             "interfaces on 5 samples, skip the FoldSeek "
                             "search.")
    args = parser.parse_args()
    run(
        processes=args.processes,
        threads=args.threads,
        gpu=args.gpu,
        foldseek_bin=args.foldseek_bin,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
