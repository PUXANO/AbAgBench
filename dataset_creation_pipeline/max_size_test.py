"""Build sequence-only machine-capacity test inputs from a SAbDab snapshot."""
from __future__ import annotations

import csv
import datetime as dt
import json
import re
import shutil
import tempfile
from pathlib import Path

from . import __version__
from .dataset_summary import summarize_dataset
from .release import (
    CAPACITY_ARTIFACTS, CAPACITY_FEATURE_FIELDS, CAPACITY_REGISTRY_FIELDS,
    _sha256, capacity_selection_report, validate_capacity_selection, validate_release,
)
from .subset import artifact_record


def selection_criteria(*, minimum=None, maximum=None, bin_width=None, bin_edges=None,
                       per_bin=None, pdb_ids=None, complex_ids=None) -> dict:
    if bin_width is not None:
        if (bin_edges is not None or type(bin_width) is not int or bin_width < 1
                or type(minimum) is not int or type(maximum) is not int or minimum >= maximum):
            raise ValueError("--bin-width requires positive width, min < max, and no --bins")
        bin_edges = [*range(minimum, maximum, bin_width), maximum]
    elif bin_edges is not None:
        if minimum is not None or maximum is not None:
            raise ValueError("Use --bins or --min-residues/--max-residues, not both")
        if not isinstance(bin_edges, list) or len(bin_edges) < 2:
            raise ValueError("--bins requires at least two increasing edges")
        minimum, maximum = bin_edges[0], bin_edges[-1]
    if per_bin is not None and bin_edges is None:
        raise ValueError("--per-bin requires --bin-width or --bins")
    return {"min_residues": minimum, "max_residues": maximum, "bin_edges": bin_edges,
            "per_bin": (1 if per_bin is None else per_bin) if bin_edges is not None else None,
            "pdb_ids": [t.lower() for t in pdb_ids] if pdb_ids is not None else None,
            "complex_ids": complex_ids, "rule": "largest_up_to_per_bin_then_complex_id"}


def _source_cif(structures: Path, pdb: str) -> Path:
    stem = f"pdb_0000{pdb}"
    matches = [p for p in (structures / stem / f"{stem}_sabdab.cif",
                          structures / stem / f"{stem}_sabdab.cif.gz",
                          structures / f"{stem}_sabdab.cif",
                          structures / f"{stem}_sabdab.cif.gz") if p.is_file()]
    if len(matches) != 1:
        raise ValueError(f"PDB {pdb}: expected exactly one source CIF, found {len(matches)}")
    return matches[0]


def _catalog(source: Path, summary: Path, structures: Path, criteria: dict) -> tuple[dict, dict]:
    import pandas as pd
    from .utils import chain_full_sequences, normalize_sabdab2_summary, parse_sabdab_cif

    parse_sabdab_cif.cache_clear()
    table = normalize_sabdab2_summary(pd.read_csv(summary, dtype=str, keep_default_na=False))
    missing = {"Hchain", "Lchain", "antigen_chain", "antigen_type"} - set(table.columns)
    if missing:
        raise ValueError(f"SAbDab summary lacks required columns: {sorted(missing)}")
    if table.empty:
        raise ValueError("SAbDab summary is empty")
    # Explicit selections only require parsing the requested PDB entries.
    requested = set(criteria["pdb_ids"] or [t.split("_", 1)[0] for t in criteria["complex_ids"] or []])
    if requested:
        absent = requested - set(table["pdb"])
        if absent:
            raise ValueError(f"Requested PDB IDs absent from source summary: {sorted(absent)}")
        table = table[table["pdb"].isin(requested)]
    candidates, excluded, sources = {}, [], {}
    for index, row in table.sort_values("pdb", kind="stable").iterrows():
        pdb = row["pdb"]
        h, light, antigen = (str(row[k]).strip() for k in ("Hchain", "Lchain", "antigen_chain"))
        light = "" if light.lower() in {"", "nan", "na", "none"} else light
        target = f"{pdb}_{h}{light}{antigen}"
        if target in candidates:
            raise ValueError(f"Duplicate complex ID in source summary: {target}")
        try:
            chains = [h, antigen] + ([light] if light else [])
            if (any(not re.fullmatch(r"[0-9A-Za-z]", c) for c in chains)
                    or len(set(chains)) != len(chains) or "protein" not in row["antigen_type"].lower()):
                raise ValueError("Requires distinct single-character H, optional L, and one protein antigen chain")
            if pdb not in sources:
                path = _source_cif(structures, pdb)
                sources[pdb] = (path, _sha256(path))
            path, digest = sources[pdb]
            sequences = chain_full_sequences(path, set(chains))
            if any(not s or set(s) - set("ACDEFGHIKLMNPQRSTVWY") for s in sequences.values()):
                raise ValueError("Empty or unsupported protein sequence")
            registry = dict(zip(CAPACITY_REGISTRY_FIELDS, (
                target, pdb, h, light, antigen, str(sum(map(len, sequences.values()))))))
            entry = {"registry": registry, "chains": sequences,
                     "source_cif": path.relative_to(source).as_posix(), "source_cif_sha256": digest}
            candidates[target] = entry
        except (ValueError, FileNotFoundError) as error:
            # Every exclusion is retained in the release and in dry-run output.
            excluded.append({"summary_row": int(index) + 2, "pdb_id": pdb,
                             "complex_id": target, "reason": str(error)})
    return {"schema_version": 1, "candidates": [candidates[t] for t in sorted(candidates)],
            "excluded": excluded}, sources


def export_max_size_test(source: Path, output: Path, *, criteria: dict,
                         summary_file="sabdab_summary_all.csv", structures_dir="sabdab_all_structures",
                         dry_run: bool = False) -> dict:
    source, output = source.expanduser().resolve(), output.expanduser().resolve()
    summary, structures = (source / summary_file).resolve(), (source / structures_dir).resolve()
    if (not summary.is_relative_to(source) or not structures.is_relative_to(source)
            or source.is_relative_to(output) or output.is_relative_to(source)):
        raise ValueError("Source, output and configured snapshot paths must not overlap or escape the snapshot")
    if not summary.is_file() or not structures.is_dir():
        raise FileNotFoundError("SAbDab snapshot needs its summary CSV and structure directory")
    if output.exists() and (not output.is_dir() or output.is_symlink()):
        raise FileExistsError(f"Invalid destination experiment: {output}")
    def check_destination():
        for name in ("inputs", "outputs"):
            path = output / name
            if path.is_symlink() or (path.exists() and (not path.is_dir() or any(path.iterdir()))):
                raise FileExistsError(f"Destination {name}/ must be absent or empty: {path}")
    check_destination()
    validate_capacity_selection(criteria)
    summary_hash = _sha256(summary)
    catalog, sources = _catalog(source, summary, structures, criteria)
    candidates = {entry["registry"]["complex_id"]: entry for entry in catalog["candidates"]}
    try:
        selected, coverage = capacity_selection_report(candidates, criteria)
    except ValueError as error:
        raise ValueError(f"{error}. Input exclusions ({len(catalog['excluded'])}): "
                         + json.dumps(catalog["excluded"], ensure_ascii=False)) from error
    warnings = [f"Bin [{b['lower']}, {b['upper']}{']' if b['upper_inclusive'] else ')'}: "
                f"selected {b['selected']} of {b['requested']} requested ({b['available']} available)"
                for b in coverage if b["shortfall"]]
    plan = {"release_kind": "max-size-test", "purpose": "machine_capacity_test", "selection": criteria,
            "bin_coverage": coverage, "warnings": warnings,
            "selected": [{"complex_id": t, "residue_length": int(candidates[t]["registry"]["residue_length"])}
                         for t in selected], "eligible_candidates": len(candidates), "excluded": catalog["excluded"]}
    if dry_run:
        return {"dry_run": True, **plan}
    if not selected:
        raise ValueError("No eligible complexes across the requested capacity bins; no dataset exported. "
                         + "; ".join(warnings))
    from .utils import write_chain_subset_cif
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        dataset = staging / "inputs"
        benchmark = dataset / "benchmark"
        (benchmark / "structures").mkdir(parents=True)
        (dataset / "provenance").mkdir()
        (dataset / "provenance/candidates.json").write_text(json.dumps(catalog, indent=2, sort_keys=True) + "\n")
        shutil.copyfile(summary, dataset / "provenance/source_summary.csv")
        if _sha256(dataset / "provenance/source_summary.csv") != summary_hash:
            raise ValueError("SAbDab summary changed during capacity export")
        with (benchmark / "registry.tsv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CAPACITY_REGISTRY_FIELDS, delimiter="\t")
            writer.writeheader()
            writer.writerows(candidates[t]["registry"] for t in selected)
        with (benchmark / "dataset_features.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CAPACITY_FEATURE_FIELDS)
            writer.writeheader()
            for target in selected:
                row = candidates[target]["registry"]
                writer.writerow({"complex_id": target, "pdb_id": row["pdb"],
                                 "is_nanobody": str(not row["Lchain"]), "residue_length": row["residue_length"]})
        with (benchmark / "benchmark.fasta").open("w") as handle:
            for target in selected:
                entry = candidates[target]
                for chain, sequence in sorted(entry["chains"].items()):
                    handle.write(f">{target}_{chain}|protein\n{sequence}\n")
                write_chain_subset_cif(source / entry["source_cif"], benchmark / "structures" / f"{target}.cif",
                                       set(entry["chains"]))
        manifest = {"schema_version": 1, "release_kind": "max-size-test", "purpose": "machine_capacity_test",
                    "status": "completed", "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "execution": {"canonical": False, "command": "python -m dataset_creation_pipeline max-size-test"},
                    "software": {"abagbench": __version__}, "selection": criteria, "selected_targets": selected,
                    "bin_coverage": coverage,
                    "dataset_summary": summarize_dataset(dataset),
                    "max_size_test_export": {"status": "completed", "outputs": [
                        artifact_record(dataset, p) for p in CAPACITY_ARTIFACTS]}}
        (dataset / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        validate_release(dataset, "max-size-test")
        if _sha256(summary) != summary_hash or any(_sha256(path) != digest for path, digest in sources.values()):
            raise ValueError("SAbDab snapshot changed during capacity export")
        check_destination()
        (staging / "outputs").mkdir()
        if not output.exists():
            staging.rename(output)
        else:
            dataset.rename(output / "inputs")
            (output / "outputs").mkdir(exist_ok=True)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {**validate_release(output / "inputs", "max-size-test"), "experiment_dir": str(output), **plan}
