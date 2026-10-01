"""Provide FoldSeek interface comparison utilities.

- Parse multimer reports and per-chain CIGAR alignments.
- Define interfaces by Cα contacts within 8 Å.
- Measure directional query antibody–antigen interface coverage.
- Screen overlap using I-LDDT >= 0.7 and coverage >= 0.5.
- Resolve the FoldSeek executable and run searches.
"""
from __future__ import annotations

import os
import pickle
import re
import shutil
import subprocess
import time
from functools import partial
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.NeighborSearch import NeighborSearch

# ── Frozen recipe constants ──────────────────────────────────────────────────

FOLDSEEK_FORMAT = (
    "query,target,complexqtmscore,complexttmscore,interfacelddt,"
    "qchains,tchains,qstart,qend,qlen,cigar"
)
ILDDT_THRESHOLD = 0.7
COVERAGE_THRESHOLD = 0.5
INTERFACE_DIST = 8.0
ILDDT_THRESHOLDS = [0.7, 0.6, 0.5, 0.4, 0.3]
REFERENCE_MIN = 1000


# ── CIGAR parsing ────────────────────────────────────────────────────────────

def parse_cigar_to_query_ordinals(qstart, cigar) -> set[int]:
    """Return one-based query residue ordinals aligned by a FoldSeek CIGAR.

    - M: record residues and advance the query position.
    - I: advance the query position without recording residues.
    - D: leave the query position unchanged.
    - Reject unsupported operations.
    """
    aligned: set[int] = set()
    curr = int(qstart)
    if curr < 1:
        raise ValueError(f"qstart must be >= 1, got {qstart!r}")
    for length_str, op in re.findall(r"(\d+)([A-Z])", str(cigar)):
        length = int(length_str)
        if op == "M":
            aligned.update(range(curr, curr + length))
            curr += length
        elif op == "I":
            curr += length
        elif op == "D":
            pass
        else:
            raise ValueError(
                f"Unsupported CIGAR op {op!r} in {cigar!r} "
                f"(only M/I/D expected)"
            )
    return aligned


# ── Interface residues from a chain-subset CIF ──────────────────────────────

def get_ordinal_interface_residues_cif(
    cif_path: Path | str,
    allowed_chains: list[str],
    dist: float = INTERFACE_DIST,
) -> dict[str, set[int]]:
    """Return per-chain one-based residue ordinals at interchain Cα contacts within 8 Å."""
    parser = MMCIFParser(QUIET=True)
    try:
        structure = parser.get_structure("query", str(cif_path))
    except Exception as e:
        raise ValueError(f"Failed to parse {cif_path}: {e}") from e
    if len(structure) == 0:
        raise ValueError(f"{cif_path}: no models in structure")
    model = structure[0]

    present = {c.get_id() for c in model}
    missing = set(allowed_chains) - present
    if missing:
        raise ValueError(
            f"{cif_path}: requested chains missing after parse: "
            f"{sorted(missing)} (present: {sorted(present)})"
        )

    chain_ca: dict[str, list[tuple[object, int]]] = {}
    for chain in model:
        cid = chain.get_id()
        if cid not in allowed_chains:
            continue
        std_residues = [
            res for res in chain.get_residues() if res.get_id()[0] == " "
        ]
        atoms: list[tuple[object, int]] = []
        for ordinal, res in enumerate(std_residues, start=1):
            if "CA" in res:
                atoms.append((res["CA"], ordinal))
        if not atoms:
            raise ValueError(f"{cif_path}: chain {cid!r} has no CA atoms")
        chain_ca[cid] = atoms

    ordinal_of: dict[int, tuple[str, int]] = {}
    for cid, atoms in chain_ca.items():
        for atom, ordinal in atoms:
            ordinal_of[id(atom)] = (cid, ordinal)

    interface: dict[str, set[int]] = {c: set() for c in allowed_chains}
    for i in range(len(allowed_chains)):
        for j in range(i + 1, len(allowed_chains)):
            c1, c2 = allowed_chains[i], allowed_chains[j]
            a1 = [a for a, _ in chain_ca[c1]]
            a2 = [a for a, _ in chain_ca[c2]]
            ns = NeighborSearch(a1 + a2)
            for atom1, atom2 in ns.search_all(dist):
                info1 = ordinal_of.get(id(atom1))
                info2 = ordinal_of.get(id(atom2))
                if info1 is None or info2 is None:
                    raise ValueError(
                        f"{cif_path}: NeighborSearch returned an atom outside "
                        f"the searched sets — internal error"
                    )
                chain1, ord1 = info1
                chain2, ord2 = info2
                if chain1 == chain2:
                    continue
                interface[chain1].add(ord1)
                interface[chain2].add(ord2)
    return interface


def parse_complex_chains(complex_id: str) -> tuple[str, str | None, str]:
    """Split ``<pdb>_<HLA>`` into (H, L or None, A) chain letters."""
    chain_str = complex_id.rsplit("_", 1)[-1]
    if len(chain_str) == 3:
        return chain_str[0], chain_str[1], chain_str[2]
    if len(chain_str) == 2:
        return chain_str[0], None, chain_str[1]
    raise ValueError(f"Cannot parse chains from complex id {complex_id!r}")


def compute_query_interfaces(
    cif_path: Path | str,
    h: str,
    l: str | None,
    a: str,
    dist: float = INTERFACE_DIST,
) -> list[tuple[dict[str, set[int]], list[str]]]:
    """Return antibody–antigen interface residues and chain roles, excluding heavy–light contacts."""
    if l is None:
        iface = get_ordinal_interface_residues_cif(cif_path, [h, a], dist)
        return [(iface, [h, a])]
    ha = get_ordinal_interface_residues_cif(cif_path, [h, a], dist)
    la = get_ordinal_interface_residues_cif(cif_path, [l, a], dist)
    merged = {
        h: ha.get(h, set()),
        l: la.get(l, set()),
        a: ha.get(a, set()) | la.get(a, set()),
    }
    return [(merged, [h, l, a])]


def is_full_complex(qchains, h: str, l: str | None, a: str) -> bool:
    """Check whether the query hit includes every heavy, optional light, and antigen chain."""
    qset = frozenset(qchains)
    has_H = h in qset
    has_A = a in qset
    if l is None:
        return has_H and has_A
    return has_H and (l in qset) and has_A


# ── Shared parse pipeline (s5 de-leak + s6 clustering) ──────────────────────
# Stage 1: report -> candidates; Stage A: interfaces; Stage B: antigen-present
# filter; Stage C: chain alignments; Stage D: coverage.  Both steps run the
# exact same code so the edge definition can never drift between them.

def parse_report(
    report_path: Path | str,
) -> tuple[set[str], list[tuple], dict]:
    """Return query IDs, I-LDDT-qualified candidates, and report diagnostics; reject malformed or empty reports."""
    report_path = Path(report_path)
    unique_queries: set[str] = set()
    candidates: list[tuple] = []
    n_total = 0
    ilddt_pass = {thr: 0 for thr in ILDDT_THRESHOLDS}
    self_hits: list[float] = []
    with open(report_path) as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split("\t")
            q, t, qchains, tchains, ilddt = parse_report_row(parts)
            n_total += 1
            unique_queries.add(q)
            if not _isnan(ilddt):
                for thr in ILDDT_THRESHOLDS:
                    if ilddt >= thr:
                        ilddt_pass[thr] += 1
                if q == t:
                    self_hits.append(ilddt)
            if not _isnan(ilddt) and ilddt >= ILDDT_THRESHOLD:
                candidates.append((q, t, qchains, tchains, ilddt))
    if n_total == 0:
        raise RuntimeError(f"Report is empty: {report_path}")
    print(f"     Total report hits: {n_total:,}")
    print(f"     Unique queries:    {len(unique_queries):,}")
    print(f"     Candidates >= {ILDDT_THRESHOLD}: {len(candidates):,}")
    if self_hits:
        import numpy as np
        arr = np.array(self_hits)
        print(f"     Self-hits: {len(arr)} "
              f"(min {arr.min():.3f}, median {np.median(arr):.3f}, "
              f"max {arr.max():.3f})")
    log_ilddt_thresholds(ilddt_pass)
    diagnostics = {
        "n_total_hits": n_total,
        "n_candidates": len(candidates),
        "ilddt_pass": ilddt_pass,
        "self_hits": self_hits,
    }
    return unique_queries, candidates, diagnostics


def best_full_complex_hits(
    report_path: Path | str,
    chain_meta: dict[str, tuple[str, str | None, str]],
) -> dict[str, tuple[float, str]]:
    """Return each query’s highest finite I-LDDT full-complex hit, including hits below the screening threshold."""
    report_path = Path(report_path)
    best: dict[str, tuple[float, str]] = {}
    with open(report_path) as f:
        for line in f:
            if not line.strip():
                continue
            q, t, qchains, _tchains, ilddt = parse_report_row(
                line.strip().split("\t")
            )
            if _isnan(ilddt):
                continue
            meta = chain_meta.get(q)
            if meta is None:
                raise ValueError(
                    f"Report query {q} has no chain meta — cannot classify "
                    f"its hits as full-complex or partial"
                )
            h, l, a = meta
            if not is_full_complex(qchains, h, l, a):
                continue
            prev = best.get(q)
            if prev is None or ilddt > prev[0]:
                best[q] = (ilddt, t)
    return best


def _isnan(v) -> bool:
    return v != v  # NaN check that needs no numpy import


def _compute_interfaces_for_query(q: str, cif_dir: Path):
    """Compute antibody–antigen interface residues and chain roles for one query."""
    h, l, a = parse_complex_chains(q)
    cif_path = cif_dir / f"{q}.cif"
    if not cif_path.is_file():
        raise FileNotFoundError(f"Query CIF missing for interface stage: {cif_path}")
    parts = compute_query_interfaces(cif_path, h, l, a)
    return q, parts, (h, l, a)


def compute_query_interfaces_pool(
    queries: list[str],
    cif_dir: Path | str,
    cache_path: Path | str,
    processes: int,
) -> tuple[dict, dict]:
    """Return cached or parallel-computed interfaces and chain roles; reject incomplete caches."""
    cif_dir = Path(cif_dir)
    cache_path = Path(cache_path)
    if cache_path.is_file():
        print(f"  A. Loading interfaces from cache: {cache_path}")
        with open(cache_path, "rb") as fh:
            cached = pickle.load(fh)
        interfaces = cached["interfaces"]
        chain_meta = cached["meta"]
        missing_meta = set(queries) - set(chain_meta)
        if missing_meta:
            raise ValueError(
                f"Interface cache misses {len(missing_meta)} queries, e.g. "
                f"{sorted(missing_meta)[:3]} — cache is stale, delete it "
                f"and rerun"
            )
        missing_iface = set(queries) - set(interfaces)
        if missing_iface:
            raise ValueError(
                f"Interface cache has no interfaces for {len(missing_iface)} "
                f"queries, e.g. {sorted(missing_iface)[:3]} — cache is stale"
            )
        print(f"     {len(chain_meta)} metas, "
              f"{len(interfaces)} with interfaces")
        return interfaces, chain_meta

    print(f"  A. Computing interfaces for {len(queries)} queries "
          f"({processes} workers)...")
    t0 = time.time()
    worker = partial(_compute_interfaces_for_query, cif_dir=cif_dir)
    interfaces: dict = {}
    chain_meta: dict = {}
    if processes > 1:
        with Pool(processes=processes, maxtasksperchild=200) as pool:
            for i, (q, parts, meta) in enumerate(
                pool.imap_unordered(worker, queries, chunksize=1), 1
            ):
                chain_meta[q] = meta
                interfaces[q] = parts
                if i % 250 == 0 or i == len(queries):
                    print(f"     {i}/{len(queries)} "
                          f"({i / (time.time() - t0):.1f} q/s)")
    else:
        for i, q in enumerate(queries, 1):
            _, parts, meta = worker(q)
            chain_meta[q] = meta
            interfaces[q] = parts
            if i % 250 == 0 or i == len(queries):
                print(f"     {i}/{len(queries)}")

    empty_queries = [
        q for q, parts in interfaces.items() if _is_empty_interface(parts)
    ]
    if empty_queries:
        print(f"  WARNING: {len(empty_queries)} queries have an empty antigen "
              f"interface (chains > {INTERFACE_DIST} A apart at CA): "
              f"{empty_queries[:5]}{'...' if len(empty_queries) > 5 else ''}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as fh:
        pickle.dump(
            {"interfaces": interfaces, "meta": chain_meta}, fh,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    print(f"     Interfaces: {len(interfaces)}, metas: {len(chain_meta)}; "
          f"cache: {cache_path} ({time.time() - t0:.0f}s)")
    return interfaces, chain_meta


def _is_empty_interface(parts) -> bool:
    for interfaces, chains in parts:
        for c in chains:
            if interfaces.get(c):
                return False
    return True


def filter_antigen_present(
    candidates: list[tuple], chain_meta: dict
) -> tuple[list[tuple], set[tuple[str, str]], int]:
    """Return full-complex candidates, required alignment pairs, and excluded-hit count."""
    coverage_hits: list[tuple] = []
    needed_pairs: set[tuple[str, str]] = set()
    n_no_antigen = 0
    for hit in candidates:
        q, t, qchains, _, _ = hit
        meta = chain_meta.get(q)
        if meta is None:
            raise ValueError(
                f"Candidate hit {q} has no chain meta — complex_id parsing "
                f"failed; cannot evaluate its antigen coverage"
            )
        h, l, a = meta
        if not is_full_complex(qchains, h, l, a):
            n_no_antigen += 1
            continue
        coverage_hits.append(hit)
        needed_pairs.add((q, t))
    print(f"     Hits with full complex: {len(coverage_hits)}")
    print(f"     Hits without antigen:   {n_no_antigen}")
    print(f"     Distinct pairs needing alignments: {len(needed_pairs)}")
    return coverage_hits, needed_pairs, n_no_antigen


def compute_coverage_pool(
    coverage_hits: list[tuple],
    interfaces: dict,
    chain_alignments: dict,
    processes: int,
) -> list[tuple[str, str, float, float]]:
    """Compute directional query interface coverage for each candidate in parallel."""
    print(f"\n  D. Computing CIGAR coverage for {len(coverage_hits)} hits "
          f"({processes} workers)...")
    t0 = time.time()
    cov_worker = partial(
        compute_coverage_for_hit,
        interfaces_by_query=interfaces,
        chain_alignments=chain_alignments,
    )
    if processes > 1 and coverage_hits:
        chunksize = max(1, len(coverage_hits) // (processes * 8))
        with Pool(processes=processes) as pool:
            results = list(
                pool.imap(cov_worker, coverage_hits, chunksize=chunksize)
            )
    else:
        results = [cov_worker(h) for h in coverage_hits]
    print(f"     Coverage computed for {len(results)} hits "
          f"({time.time() - t0:.0f}s)")
    return results


def derive_best(
    results: list[tuple[str, str, float, float]]
) -> tuple[dict[str, tuple[float, str, float]], list[float]]:
    """Return each query’s highest-I-LDDT hit meeting the interface coverage threshold and coverage values."""
    best: dict[str, tuple[float, str, float]] = {}
    cov_values: list[float] = []
    for q, t, ilddt, cov in results:
        if q != t:
            cov_values.append(cov)
        if cov >= COVERAGE_THRESHOLD:
            prev = best.get(q)
            if prev is None or ilddt > prev[0]:
                best[q] = (ilddt, t, cov)
    print(f"     Queries with a best hit (cov >= {COVERAGE_THRESHOLD}): "
          f"{len(best)}")
    return best, cov_values


def decide_novelty(
    top_full: tuple[float, str] | None,
    best_cov: tuple[float, str, float] | None,
    ilddt_threshold: float = ILDDT_THRESHOLD,
) -> str:
    """Classify structural overlap from full-complex hits.

    - nan: no measurable full-complex hit.
    - novel: no hit passes both I-LDDT and interface coverage thresholds.
    - leaked: at least one hit passes both thresholds.
    """
    if top_full is None:
        return "nan"
    if top_full[0] < ilddt_threshold:
        return "novel"
    if best_cov is None:
        return "novel"
    if not (best_cov[0] >= ilddt_threshold):
        raise RuntimeError(f"best coverage-passing hit I-LDDT {best_cov[0]} < {ilddt_threshold} "
        f"— violates the coverage-pool invariant")
    return "leaked"


def parse_report_row(parts: list[str]) -> tuple[str, str, tuple[str, ...], tuple[str, ...], float]:
    """Parse a multimer report row; map FoldSeek's missing I-LDDT tokens to NaN."""
    if len(parts) < 13:
        raise ValueError(
            f"Malformed report row: {len(parts)} columns (< 13): {parts[:4]}..."
        )
    query, target = parts[0], parts[1]
    qchains = tuple(parts[2].split(",")) if parts[2] else ()
    tchains = tuple(parts[3].split(",")) if parts[3] else ()
    if parts[12].strip().lower() in {"-na", "na"}:
        ilddt = float("nan")
    else:
        try:
            ilddt = float(parts[12])
        except ValueError as e:
            raise ValueError(
                f"Unparseable interfacelddt {parts[12]!r} in report row "
                f"{query}:{target}"
            ) from e
    return query, target, qchains, tchains, ilddt


# ── Chain-alignment TSV loading (Stage C) ───────────────────────────────────

def load_chain_alignments(
    tsv_path: Path | str,
    needed_pairs: set[tuple[str, str]],
) -> dict[tuple[str, str, str, str], tuple[str, str]]:
    """Load query starts and CIGARs for required complex/chain pairs; reject missing alignment pairs."""
    df = pd.read_csv(
        tsv_path,
        sep="\t",
        header=None,
        usecols=[0, 1, 7, 10],
        names=["q_full", "t_full", "qstart", "cigar"],
        dtype={"q_full": str, "t_full": str, "qstart": str, "cigar": str},
        index_col=False,
        low_memory=False,
    )
    if len(df) == 0:
        raise ValueError(f"FoldSeek TSV has no rows: {tsv_path}")

    q_split = df["q_full"].str.rsplit("_", n=1, expand=True)
    t_split = df["t_full"].str.rsplit("_", n=1, expand=True)
    df["q_comp"], df["qc"] = q_split[0], q_split[1]
    df["t_comp"], df["tc"] = t_split[0], t_split[1]

    needed = {f"{q}\t{t}" for q, t in needed_pairs}
    df = df[(df["q_comp"] + "\t" + df["t_comp"]).isin(needed)]
    found = set(df["q_comp"] + "\t" + df["t_comp"])
    missing = needed - found
    if missing:
        raise ValueError(
            f"{len(missing)} needed (query, target) pairs have no chain rows "
            f"in {tsv_path}, e.g. {sorted(missing)[:3]} — report/TSV mismatch"
        )

    alignments: dict[tuple[str, str, str, str], tuple[str, str]] = {}
    for q_comp, t_comp, qc, tc, qs, cig in zip(
        df["q_comp"], df["t_comp"], df["qc"], df["tc"], df["qstart"], df["cigar"]
    ):
        if pd.isna(qs) or pd.isna(cig) or str(qs).strip() == "" or str(cig).strip() == "":
            raise ValueError(
                f"Chain alignment {q_comp}:{qc} -> {t_comp}:{tc} has empty "
                f"qstart/cigar ({qs!r}/{cig!r})"
            )
        key = (q_comp, t_comp, qc, tc)
        if key in alignments:
            raise ValueError(f"Duplicate chain alignment row for {key}")
        alignments[key] = (str(qs).strip(), str(cig).strip())
    return alignments


# ── Coverage (Stage D) ──────────────────────────────────────────────────────

def compute_coverage_for_hit(
    hit: tuple[str, str, tuple[str, ...], tuple[str, ...], float],
    interfaces_by_query: dict[str, list[tuple[dict[str, set[int]], list[str]]]],
    chain_alignments: dict[tuple[str, str, str, str], tuple[str, str]],
) -> tuple[str, str, float, float]:
    """Return the aligned fraction of query antibody–antigen interface residues.

    - Use per-chain CIGAR mappings across the interface.
    - Return 1.0 for self-hits and 0.0 for empty query interfaces.
    """
    q_comp, t_comp, qchains, tchains, ilddt = hit
    if q_comp == t_comp:
        return q_comp, t_comp, ilddt, 1.0

    iface_parts = interfaces_by_query.get(q_comp)
    if not iface_parts:
        raise ValueError(f"No computed interfaces for query {q_comp!r}")

    total_int_res = 0
    total_aln_res = 0
    empty_parts = 0
    for interfaces, chains_to_check in iface_parts:
        if not interfaces:
            empty_parts += 1
            continue
        for c in chains_to_check:
            total_int_res += len(interfaces.get(c, set()))
        for qc, tc in zip(qchains, tchains):
            if qc not in chains_to_check or qc not in interfaces:
                continue
            aln = chain_alignments.get((q_comp, t_comp, qc, tc))
            if aln is None:
                raise ValueError(
                    f"Missing chain alignment for {q_comp}:{qc} -> "
                    f"{t_comp}:{tc} (needed for coverage)"
                )
            qstart, cigar = aln
            aligned = parse_cigar_to_query_ordinals(qstart, cigar)
            total_aln_res += len(interfaces[qc].intersection(aligned))

    if total_int_res == 0:
        return q_comp, t_comp, ilddt, 0.0
    return q_comp, t_comp, ilddt, total_aln_res / total_int_res


# ── Binary discovery & execution ────────────────────────────────────────────

def find_foldseek_bin(cli_value: str | None = None) -> str:
    """Resolve an executable from the CLI path, FOLDSEEK_BIN, then PATH; reject invalid explicit paths."""
    if cli_value:
        p = Path(cli_value).expanduser()
        if not p.is_file():
            raise FileNotFoundError(
                f"--foldseek-bin path is not a file: {p}"
            )
        if not os.access(p, os.X_OK):
            raise PermissionError(f"--foldseek-bin path is not executable: {p}")
        return str(p)

    env = os.environ.get("FOLDSEEK_BIN")
    if env:
        p = Path(env).expanduser()
        if not p.is_file():
            raise FileNotFoundError(
                f"FOLDSEEK_BIN is not a file: {p} — unset it or fix the path"
            )
        if not os.access(p, os.X_OK):
            raise PermissionError(f"FOLDSEEK_BIN is not executable: {p}")
        return str(p)

    found = shutil.which("foldseek")
    if found is None:
        raise FileNotFoundError(
            "foldseek binary not found. Set FOLDSEEK_BIN or pass --foldseek-bin "
            "Use --dry-run to validate everything except the FoldSeek search."
        )
    return found


@lru_cache(maxsize=16)
def _supports_chain_tm_threshold(foldseek_bin: str) -> bool:
    result = subprocess.run(
        [foldseek_bin, "easy-multimersearch", "-h"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    return "--chain-tm-threshold" in (result.stdout + result.stderr)


def run_foldseek(argv: list[str], dry_run: bool = False) -> None:
    """Run FoldSeek and reject command failures; print the command in dry-run mode."""
    if dry_run:
        print(f"[DRY-RUN] {' '.join(argv)}")
        return
    effective_argv = list(argv)
    if "--chain-tm-threshold" in effective_argv and not _supports_chain_tm_threshold(effective_argv[0]):
        index = effective_argv.index("--chain-tm-threshold")
        del effective_argv[index:index + 2]
        print(
            "[FoldSeek compatibility] omitted --chain-tm-threshold 0; "
            "this build has no such easy-multimersearch option and the frozen value adds no cutoff."
        )
    subprocess.run(effective_argv, check=True)


def log_ilddt_thresholds(counts: dict[float, int]) -> None:
    """Print counts of hits passing each I-LDDT threshold."""
    print("  I-LDDT pass counts (whole report, diagnostic):")
    for thr in ILDDT_THRESHOLDS:
        print(f"    I-LDDT >= {thr}: {counts.get(thr, 0)}")
