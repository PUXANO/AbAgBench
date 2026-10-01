#!/usr/bin/env python3
"""Cluster benchmark candidates by antibody–antigen interface similarity.

- Run all-versus-all FoldSeek multimer searches.
- Retain edges with I-LDDT >= 0.7 and query interface coverage >= 0.5.
- Use complete-linkage clustering at distance 0.3; preserve singleton complexes.
- Select representatives by resolution, R-factor, then complex ID.
- Export structures, FASTA sequences, clusters, and similarity diagnostics.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch

from ..foldseek_utils import (
    COVERAGE_THRESHOLD,
    FOLDSEEK_FORMAT,
    ILDDT_THRESHOLD,
    compute_coverage_pool,
    compute_query_interfaces_pool,
    filter_antigen_present,
    find_foldseek_bin,
    load_chain_alignments,
    parse_report,
    run_foldseek,
)
from ..paths import DATA_ROOT, INTERMEDIATE_ROOT, SCRATCH_ROOT, BENCHMARK_DIR as FINAL_BENCHMARK_DIR
from ..utils import (
    StageMetrics,
    atomic_write_json,
    chain_full_sequences,
    chain_ids_from_row,
    pdb_to_cif_path,
)

INPUT_TSV = INTERMEDIATE_ROOT / "05_deleaked.tsv"
STAGE_METRICS_PATH = INTERMEDIATE_ROOT / "stage_metrics.json"
SCRATCH_DELEAKED = SCRATCH_ROOT / "s5_deleaked"
SCRATCH_FOLDSEEK = SCRATCH_ROOT / "s6_foldseek"
REPORT_PATH = SCRATCH_FOLDSEEK / "results_all_vs_all.tsv_report"
TSV_PATH = SCRATCH_FOLDSEEK / "results_all_vs_all.tsv"
INTERFACE_CACHE = SCRATCH_FOLDSEEK / "s6_interface_cache.pkl"
BENCHMARK_DIR = FINAL_BENCHMARK_DIR
FILTERED_TSV = BENCHMARK_DIR / "registry.tsv"
STRUCTURES_DIR = BENCHMARK_DIR / "structures"
FASTA_PATH = BENCHMARK_DIR / "benchmark.fasta"
EDGES_PATH = BENCHMARK_DIR / "edges.tsv"
CLUSTERS_PATH = BENCHMARK_DIR / "clusters.tsv"
REPS_PATH = BENCHMARK_DIR / "representatives.tsv"
LINKAGE_PATH = BENCHMARK_DIR / "linkage.npy"
NODES_PATH = BENCHMARK_DIR / "nodes.json"
LEAKAGE_SUMMARY_PATH = BENCHMARK_DIR / "leakage_summary.txt"

STAGE_NAME = "FoldSeek clustering (complete linkage, cut 0.3)"


# ── Clustering (port of 11_analysis/clustering_utils.py) ────────────────────

def cluster_complexes(
    edges_df: pd.DataFrame,
    ilddt_threshold: float = ILDDT_THRESHOLD,
    linkage_method: str = "complete",
) -> tuple[dict, np.ndarray, dict, np.ndarray, list[str]]:
    """Return complete-linkage clusters, distances, node indices, linkage, and node order.

    - Distance is 1 − I-LDDT.
    - Use the smaller directional distance for each pair.
    """
    print("  Building distance matrix for clustering...")
    nodes = sorted(set(edges_df["complex1"]) | set(edges_df["complex2"]))
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)

    dist_matrix = np.ones((n, n))
    np.fill_diagonal(dist_matrix, 0.0)
    for _, row in edges_df.iterrows():
        i = node_to_idx[row["complex1"]]
        j = node_to_idx[row["complex2"]]
        d = 1.0 - row["ilddt"]
        dist_matrix[i, j] = min(dist_matrix[i, j], d)
        dist_matrix[j, i] = min(dist_matrix[j, i], d)

    condensed = []
    for i in range(n):
        for j in range(i + 1, n):
            condensed.append(dist_matrix[i, j])
    print(f"  {n} nodes, {len(condensed)} pairwise distances — "
          f"complete linkage, cut at {1.0 - ilddt_threshold}")
    Z = sch.linkage(np.asarray(condensed, dtype=float), method=linkage_method)
    labels = sch.fcluster(Z, t=1.0 - ilddt_threshold, criterion="distance")

    clusters: dict = {}
    for node, label in zip(nodes, labels):
        clusters.setdefault(int(label), []).append(node)
    return clusters, dist_matrix, node_to_idx, Z, nodes


def compute_leakage(clusters: dict, edges_df: pd.DataFrame) -> dict:
    """Count cluster pairs connected by an I-LDDT >= 0.7 edge."""
    membership = {m: cid for cid, members in clusters.items() for m in members}
    n_clusters = len(clusters)
    total_pairs = n_clusters * (n_clusters - 1) // 2

    pair_max: dict = {}
    n_leaked_edges = 0
    for c1, c2, ilddt in zip(
        edges_df["complex1"], edges_df["complex2"], edges_df["ilddt"]
    ):
        if c1 == c2 or c1 not in membership or c2 not in membership:
            continue
        a, b = membership[c1], membership[c2]
        if a == b:
            continue
        n_leaked_edges += 1
        key = (min(a, b), max(a, b))
        pair_max[key] = max(pair_max.get(key, -1.0), ilddt)

    n_leaked_pairs = len(pair_max)
    return {
        "n_clusters": n_clusters,
        "total_pairs": total_pairs,
        "leaked_pairs": n_leaked_pairs,
        "clean_pairs": total_pairs - n_leaked_pairs,
        "pct_leaked": (100 * n_leaked_pairs / total_pairs) if total_pairs else 0.0,
        "total_leaked_edges": n_leaked_edges,
    }


def select_representatives_by_quality(
    clusters: dict, reg_df: pd.DataFrame
) -> dict:
    """Select one representative per cluster by resolution, R-factor (missing last), then ID."""
    if "resolution" not in reg_df.columns or "r_factor" not in reg_df.columns:
        raise ValueError(
            "Registry lacks resolution/r_factor — cannot select "
            "representatives by experimental quality"
        )
    reg_idx = reg_df.set_index("complex_id")

    def _sort_key(m):
        if m not in reg_idx.index:
            raise ValueError(
                f"Cluster member {m} missing from the registry — cannot "
                f"select representatives"
            )
        row = reg_idx.loc[m]
        res = float(row["resolution"]) if pd.notna(row["resolution"]) else float("inf")
        rfac = float(row["r_factor"]) if pd.notna(row["r_factor"]) else float("inf")
        return (res, rfac, m)

    return {cid: min(members, key=_sort_key) for cid, members in clusters.items()}


# ── FoldSeek all-vs-all ─────────────────────────────────────────────────────

def run_all_vs_all_search(
    threads: int, gpu: int, foldseek_bin: str | None, dry_run: bool
) -> None:
    """Build combinedDB(+pad) from s5_deleaked and search all-vs-all."""
    if not (SCRATCH_DELEAKED.is_dir()):
        raise RuntimeError(f"Missing s5 input dir: {SCRATCH_DELEAKED} — run s5 first.")
    n_in = len(list(SCRATCH_DELEAKED.glob("*.cif")))
    if not (n_in > 0):
        raise RuntimeError(f"Empty s5_deleaked dir: {SCRATCH_DELEAKED}")

    if SCRATCH_FOLDSEEK.exists():
        shutil.rmtree(SCRATCH_FOLDSEEK)
    SCRATCH_FOLDSEEK.mkdir(parents=True)

    combined_db = SCRATCH_FOLDSEEK / "combinedDB"
    combined_db_pad = SCRATCH_FOLDSEEK / "combinedDB_pad"
    results_tsv = SCRATCH_FOLDSEEK / "results_all_vs_all.tsv"
    search_tmp = SCRATCH_FOLDSEEK / "tmp_search"

    if dry_run:
        # Dry-run must work without a foldseek binary: print the commands.
        for cmd in [
            ["foldseek", "createdb", str(SCRATCH_DELEAKED), str(combined_db),
             "--threads", str(threads)],
            ["foldseek", "makepaddedseqdb", str(combined_db), str(combined_db_pad),
             "--threads", str(threads)],
            [
                "foldseek", "easy-multimersearch",
                str(combined_db), str(combined_db_pad),
                str(results_tsv), str(search_tmp),
                "--alignment-type", "2",
                "--format-output", FOLDSEEK_FORMAT,
                "--threads", str(threads),
                "--prefilter-mode", "1",
                "--chain-tm-threshold", "0",
            ] + (["--gpu", "1"] if gpu else []),
        ]:
            run_foldseek(cmd, dry_run=True)
        return

    bin_path = find_foldseek_bin(foldseek_bin)
    print(f"  foldseek binary: {bin_path}")

    run_foldseek(
        [bin_path, "createdb", str(SCRATCH_DELEAKED), str(combined_db),
         "--threads", str(threads)],
        dry_run,
    )
    run_foldseek(
        [bin_path, "makepaddedseqdb", str(combined_db), str(combined_db_pad),
         "--threads", str(threads)],
        dry_run,
    )
    cmd = [
        bin_path, "easy-multimersearch",
        str(combined_db), str(combined_db_pad),
        str(results_tsv), str(search_tmp),
        "--alignment-type", "2",
        "--format-output", FOLDSEEK_FORMAT,
        "--threads", str(threads),
        "--prefilter-mode", "1",
        "--chain-tm-threshold", "0",
    ]
    if gpu:
        cmd += ["--gpu", "1"]
    run_foldseek(cmd, dry_run)

    if not results_tsv.is_file():
        raise RuntimeError(f"FoldSeek wrote no results: {results_tsv}")
    if not REPORT_PATH.is_file():
        raise RuntimeError(f"FoldSeek wrote no report: {REPORT_PATH}")


# ── Edge precompute (same stages as s5) ─────────────────────────────────────

def compute_edges(processes: int, registry_ids: set[str]) -> tuple[pd.DataFrame, dict]:
    """Build coverage-qualified similarity edges, including self-edges for every complex."""
    unique_queries, candidates, diagnostics = parse_report(REPORT_PATH)

    interfaces, chain_meta = compute_query_interfaces_pool(
        sorted(unique_queries), SCRATCH_DELEAKED, INTERFACE_CACHE, processes
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

    # Edges: I-LDDT >= 0.7 (candidate) AND coverage >= 0.5; symmetric pairs
    # keep the strongest I-LDDT; then force self-edges for every complex.
    edge_map: dict[tuple[str, str], float] = {}
    for q, t, ilddt, cov in results:
        if cov >= COVERAGE_THRESHOLD:
            key = (q, t)
            edge_map[key] = max(edge_map.get(key, -1.0), ilddt)
    n_from_search = len(edge_map)
    for cid in registry_ids:
        edge_map.setdefault((cid, cid), 1.0)
    n_forced = len(edge_map) - n_from_search
    print(f"     Edges from search (cov >= {COVERAGE_THRESHOLD}): "
          f"{n_from_search:,}")
    print(f"     Forced self-edges (every complex is a node): {n_forced:,}")
    diagnostics["n_edges_search"] = n_from_search
    diagnostics["n_forced_self_edges"] = n_forced

    edges_df = pd.DataFrame(
        [
            {"complex1": q, "complex2": t, "ilddt": v}
            for (q, t), v in sorted(edge_map.items())
        ],
        columns=["complex1", "complex2", "ilddt"],
    )
    return edges_df, diagnostics


# ── Materialization ─────────────────────────────────────────────────────────

def write_benchmark_fasta(registry, reps: dict, fastas: dict | None) -> int:
    """Write one FASTA record per representative chain from supplied or mmCIF-derived sequences."""
    records: list[str] = []
    for cluster_id in sorted(reps):
        cid = reps[cluster_id]
        if fastas is not None:
            seqs = fastas.get(cid)
            if seqs is None:
                raise ValueError(f"No in-memory FASTA for representative {cid}")
        else:
            row = registry.loc[registry["complex_id"] == cid]
            if not (len(row) == 1):
                raise RuntimeError(f"Representative {cid} not unique in registry "
                f"({len(row)} rows)")
            pdb_id = str(row.iloc[0]["pdb"]).strip().lower()
            seqs = chain_full_sequences(
                pdb_to_cif_path(pdb_id), set(chain_ids_from_row(row.iloc[0]))
            )
        for chain in sorted(seqs):
            records.append(f">{cid}_{chain}|protein")
            records.append(seqs[chain])
    FASTA_PATH.write_text("\n".join(records) + "\n")
    n_records = sum(1 for l in records if l.startswith(">"))
    return n_records


# ── Main ────────────────────────────────────────────────────────────────────

def run(
    processes: int = max(1, os.cpu_count() or 1),
    threads: int = max(1, os.cpu_count() or 1),
    gpu: int = 0,
    foldseek_bin: str | None = None,
    dry_run: bool = False,
    fastas: dict | None = None,
) -> dict:
    """Cluster candidates and export benchmark representatives."""
    global BENCHMARK_DIR, STRUCTURES_DIR, FASTA_PATH, EDGES_PATH, CLUSTERS_PATH
    global REPS_PATH, LINKAGE_PATH, NODES_PATH, LEAKAGE_SUMMARY_PATH, FILTERED_TSV
    print("\n" + "=" * 60)
    print("STEP 6: FoldSeek all-vs-all clustering & representatives")
    print("=" * 60)

    if not INPUT_TSV.exists():
        raise FileNotFoundError(f"Post-s5 registry not found: {INPUT_TSV}; run stages 1–5 first.")
    registry = pd.read_csv(INPUT_TSV, sep="\t", low_memory=False)
    if not (len(registry) > 0):
        raise RuntimeError(f"Post-s5 registry is empty: {FILTERED_TSV}")
    for col in ("complex_id", "resolution", "r_factor", "modality",
                "cdrh3_length", "residue_length"):
        if not (col in registry.columns):
            raise RuntimeError(f"Registry missing {col!r}")
    registry_ids = set(registry["complex_id"])
    if not (len(registry_ids) == len(registry)):
        raise RuntimeError(f"Duplicate complex_id in registry ({len(registry)} rows, "
        f"{len(registry_ids)} unique)")
    deleaked_on_disk = len(list(SCRATCH_DELEAKED.glob("*.cif")))
    if not (deleaked_on_disk == len(registry)):
        raise RuntimeError(f"s5_deleaked CIFs ({deleaked_on_disk}) != registry rows "
        f"({len(registry)})")

    run_all_vs_all_search(threads, gpu, foldseek_bin, dry_run)
    if dry_run:
        print("\n  [DRY-RUN] stopped before the FoldSeek search — run "
              "without --dry-run on the HPC.")
        return {"dry_run": True, "n_complexes": len(registry)}

    edges_df, diagnostics = compute_edges(processes, registry_ids)
    if not (len(edges_df) > 0):
        raise RuntimeError("No edges at all — every complex would be a singleton")
    print("\n  Clustering (complete linkage, cut 0.3)...")
    clusters, dist_matrix, _node_to_idx, Z, nodes = cluster_complexes(
        edges_df, ilddt_threshold=ILDDT_THRESHOLD, linkage_method="complete"
    )
    cluster_members = set().union(*clusters.values()) if clusters else set()
    missing_nodes = registry_ids - cluster_members
    if missing_nodes:
        raise ValueError(
            f"{len(missing_nodes)} registry complexes have no cluster, e.g. "
            f"{sorted(missing_nodes)[:3]} — they would be silently dropped"
        )
    print(f"  {len(clusters):,} clusters for {len(nodes):,} complexes")

    leakage = compute_leakage(clusters, edges_df)
    print(f"  Inter-cluster leakage: {leakage['leaked_pairs']:,}/{leakage['total_pairs']:,} "
          f"pairs ({leakage['pct_leaked']:.3f}%)")

    print("\n  Selecting representatives (quality only)...")
    reps = select_representatives_by_quality(clusters, registry)
    if not (len(reps) == len(clusters)):
        raise RuntimeError(f"Representatives ({len(reps)}) != clusters ({len(clusters)})")
    for cluster_id, rep in reps.items():
        if not (rep in clusters[cluster_id]):
            raise RuntimeError(f"Representative {rep} not in cluster {cluster_id}")
    print(f"  {len(reps):,} representatives (one per cluster)")

    # Prepare a complete replacement beside the durable benchmark. The old
    # directory remains available until all new outputs have been validated.
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    staged_benchmark = Path(tempfile.mkdtemp(prefix=".benchmark-staging-", dir=DATA_ROOT))
    BENCHMARK_DIR = staged_benchmark
    STRUCTURES_DIR = BENCHMARK_DIR / "structures"
    FASTA_PATH = BENCHMARK_DIR / "benchmark.fasta"
    EDGES_PATH = BENCHMARK_DIR / "edges.tsv"
    CLUSTERS_PATH = BENCHMARK_DIR / "clusters.tsv"
    REPS_PATH = BENCHMARK_DIR / "representatives.tsv"
    LINKAGE_PATH = BENCHMARK_DIR / "linkage.npy"
    NODES_PATH = BENCHMARK_DIR / "nodes.json"
    LEAKAGE_SUMMARY_PATH = BENCHMARK_DIR / "leakage_summary.txt"
    FILTERED_TSV = BENCHMARK_DIR / "registry.tsv"
    STRUCTURES_DIR.mkdir(parents=True)
    edges_df.to_csv(EDGES_PATH, sep="\t", index=False)
    print(f"  Edges staged: {EDGES_PATH} ({len(edges_df):,} edges)")
    for cluster_id in sorted(reps):
        cid = reps[cluster_id]
        src = SCRATCH_DELEAKED / f"{cid}.cif"
        if not (src.is_file()):
            raise RuntimeError(f"Representative CIF missing: {src}")
        shutil.copy2(src, STRUCTURES_DIR / f"{cid}.cif")
    n_structures = len(list(STRUCTURES_DIR.glob("*.cif")))
    if not (n_structures == len(reps)):
        raise RuntimeError(f"Benchmark structures ({n_structures}) != representatives ({len(reps)})")

    n_fasta_records = write_benchmark_fasta(registry, reps, fastas)
    print(f"  Structures: {STRUCTURES_DIR} ({n_structures:,} CIFs)")
    print(f"  FASTA: {FASTA_PATH} ({n_fasta_records:,} chain records)")

    cluster_rows = []
    for cluster_id in sorted(clusters):
        for member in sorted(clusters[cluster_id]):
            cluster_rows.append({"cluster_id": cluster_id, "complex_id": member})
    pd.DataFrame(cluster_rows).to_csv(CLUSTERS_PATH, sep="\t", index=False)

    reg_idx = registry.set_index("complex_id")
    rep_rows = []
    for cluster_id in sorted(reps):
        cid = reps[cluster_id]
        row = reg_idx.loc[cid]
        rep_rows.append({
            "cluster_id": cluster_id,
            "complex_id": cid,
            "cluster_size": len(clusters[cluster_id]),
            "resolution": row["resolution"],
            "r_factor": row["r_factor"],
        })
    reps_df = pd.DataFrame(rep_rows)
    reps_df.to_csv(REPS_PATH, sep="\t", index=False)
    print("  clusters.tsv / representatives.tsv written")

    np.save(LINKAGE_PATH, Z)
    NODES_PATH.write_text(json.dumps(nodes, indent=2) + "\n")

    _write_leakage_summary(leakage, diagnostics, clusters)
    _write_log(registry, reps, clusters, leakage, diagnostics)

    # ── Trim registry to representatives + stage metrics ──────────────────
    registry_reps = registry.loc[registry["complex_id"].isin(set(reps.values()))].copy()
    if not (len(registry_reps) == len(reps)):
        raise RuntimeError(f"Registry reps ({len(registry_reps)}) != selected ({len(reps)})")
    registry_reps.to_csv(FILTERED_TSV, sep="\t", index=False)
    print(f"  Registry staged: {FILTERED_TSV} ({len(registry_reps):,} reps)")

    if STAGE_METRICS_PATH.exists():
        with open(STAGE_METRICS_PATH) as f:
            sm = json.load(f)
        sm["stages"] = [s for s in sm["stages"] if s["stage"] != STAGE_NAME]
        sm["stages"].append({
            "stage": STAGE_NAME,
            "before": len(registry),
            "after": len(registry_reps),
            "dropped": len(registry) - len(registry_reps),
            "reasons": {
                "redundant complexes merged into clusters":
                    len(registry) - len(registry_reps),
            },
        })
        sm["final"] = len(registry_reps)
        atomic_write_json(STAGE_METRICS_PATH, sm)
        print(f"  stage_metrics.json updated (final {len(registry)} -> "
              f"{len(registry_reps)})")
    else:
        raise FileNotFoundError(f"Required stage metrics missing: {STAGE_METRICS_PATH}; run stage 2 first")

    metrics = StageMetrics("STEP 6: FoldSeek clustering")
    metrics.add(
        STAGE_NAME,
        len(registry), len(registry_reps),
        reasons={"redundant complexes merged into clusters":
                 len(registry) - len(registry_reps)},
    )
    atomic_write_json(
        INTERMEDIATE_ROOT / "06_clustering_metrics.json",
        {"before": len(registry), "after": len(registry_reps),
         "dropped": len(registry) - len(registry_reps),
         "reasons": {"redundant complexes merged into clusters": len(registry) - len(registry_reps)}},
    )

    previous = DATA_ROOT / f".benchmark-previous-{time.time_ns()}"
    had_previous = FINAL_BENCHMARK_DIR.exists()
    if had_previous:
        os.replace(FINAL_BENCHMARK_DIR, previous)
    try:
        os.replace(staged_benchmark, FINAL_BENCHMARK_DIR)
    except Exception:
        if had_previous and previous.exists():
            os.replace(previous, FINAL_BENCHMARK_DIR)
        raise
    if previous.exists():
        shutil.rmtree(previous)
    BENCHMARK_DIR = FINAL_BENCHMARK_DIR
    FILTERED_TSV = FINAL_BENCHMARK_DIR / "registry.tsv"
    print(f"  Benchmark committed atomically: {FINAL_BENCHMARK_DIR}")

    return {
        "n_complexes": len(registry),
        "n_clusters": len(clusters),
        "n_representatives": len(reps),
        "n_edges": len(edges_df),
        "leakage_pct": leakage["pct_leaked"],
        "benchmark_dir": str(BENCHMARK_DIR),
        "metrics": metrics.to_dict(),
    }


def _write_leakage_summary(leakage: dict, diagnostics: dict, clusters: dict) -> None:
    sizes = [len(m) for m in clusters.values()]
    sizes_arr = np.array(sizes)
    lines: list[str] = []
    lines.append("=== FoldSeek clustering summary ===")
    lines.append(f"Clusters: {leakage['n_clusters']:,}")
    lines.append(f"Total cluster pairs: {leakage['total_pairs']:,}")
    lines.append(f"Leaked pairs (cross-cluster I-LDDT >= 0.7): "
                 f"{leakage['leaked_pairs']:,} "
                 f"({leakage['pct_leaked']:.3f}%)")
    lines.append(f"Clean pairs: {leakage['clean_pairs']:,}")
    lines.append(f"Total leaked edges: {leakage['total_leaked_edges']:,}")
    lines.append(f"Cluster sizes: min={sizes_arr.min()}, "
                 f"median={np.median(sizes_arr):.0f}, max={sizes_arr.max()}")
    lines.append(f"Edges from search (cov >= 0.5): {diagnostics['n_edges_search']:,}")
    lines.append(f"Forced self-edges: {diagnostics['n_forced_self_edges']:,}")
    lines.append(f"Report hits: {diagnostics['n_total_hits']:,}")
    lines.append(f"Candidates (I-LDDT >= 0.7): {diagnostics['n_candidates']:,}")
    if diagnostics.get("cov_values"):
        covs = np.array(diagnostics["cov_values"])
        bins = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        counts, _ = np.histogram(covs, bins=bins)
        lines.append("Coverage histogram (non-self, I-LDDT >= 0.7):")
        for i in range(len(bins) - 1):
            lines.append(f"  [{bins[i]:.1f}, {bins[i+1]:.1f}): {counts[i]}")
    LEAKAGE_SUMMARY_PATH.write_text("\n".join(lines) + "\n")
    print(f"  Leakage summary: {LEAKAGE_SUMMARY_PATH}")


def _write_log(registry, reps, clusters, leakage, diagnostics) -> None:
    log_dir = DATA_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"s6_foldseek_cluster_{ts}.log"
    with open(log_path, "w") as f:
        f.write("=== Step 6: FoldSeek clustering & representatives ===\n")
        f.write(f"Started at: {ts}\n")
        f.write(f"Complexes in: {len(registry)}\n")
        f.write(f"Clusters: {len(clusters)}\n")
        f.write(f"Representatives: {len(reps)}\n")
        f.write(f"Leakage: {leakage}\n")
        f.write(f"Diagnostics: {diagnostics}\n")
        f.write("\n--- Representatives ---\n")
        for cluster_id in sorted(reps):
            f.write(f"{cluster_id}\t{reps[cluster_id]}\t"
                    f"{len(clusters[cluster_id])}\n")
    print(f"  Log: {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 6: FoldSeek all-vs-all clustering and representative "
                    "selection."
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
                        help="Validate inputs, skip the FoldSeek search.")
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
