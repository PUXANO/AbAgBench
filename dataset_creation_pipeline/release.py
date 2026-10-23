#!/usr/bin/env python3
"""Validate full, subset and max-size-test releases using the standard library.

The portable release-v1 validator is also distributed by universal_runner; neither
repository imports the other at runtime.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable


REQUIRED_FILES = (
    "benchmark/registry.tsv",
    "benchmark/benchmark.fasta",
    "benchmark/dataset_features.csv",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON field: {key}")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError(f"Invalid JSON constant: {value}")
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique, parse_constant=invalid)


def _hash_directory(path: Path) -> tuple[str, int, int]:
    files = sorted(item for item in path.rglob("*") if item.is_file())
    digest = hashlib.sha256()
    size = 0
    for item in files:
        relative = item.relative_to(path).as_posix()
        item_hash = _sha256(item)
        item_size = item.stat().st_size
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(item_hash.encode("ascii"))
        digest.update(b"\n")
        size += item_size
    return digest.hexdigest(), len(files), size


def _require_output_record(
    manifest: dict[str, Any], stage: str, relative_path: str
) -> dict[str, Any]:
    record = manifest.get("stages", {}).get(stage)
    if not isinstance(record, dict) or record.get("status") != "completed":
        raise ValueError(f"Dataset manifest stage {stage} is not completed")
    matches = [
        item
        for item in record.get("outputs", [])
        if isinstance(item, dict) and item.get("path") == relative_path
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Dataset manifest stage {stage} must contain exactly one "
            f"{relative_path!r} output record"
        )
    return matches[0]


def _verify_record(release_dir: Path, record: dict[str, Any]) -> None:
    path = release_dir / str(record["path"])
    if record.get("kind") == "file":
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Dataset output is missing or empty: {path}")
        observed = {
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
    elif record.get("kind") == "directory":
        if not path.is_dir():
            raise FileNotFoundError(f"Dataset output directory is missing: {path}")
        digest, count, size = _hash_directory(path)
        observed = {"sha256": digest, "file_count": count, "size_bytes": size}
    else:
        raise ValueError(f"Unsupported dataset manifest artifact kind: {record!r}")
    mismatches = {
        key: (record.get(key), value)
        for key, value in observed.items()
        if record.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Dataset artifact differs from its manifest: {path}: {mismatches}")


def _read_delimited_ids(path: Path, delimiter: str) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if "complex_id" not in (reader.fieldnames or []):
            raise ValueError(f"{path} lacks complex_id")
        ids = [(row.get("complex_id") or "").strip() for row in reader]
    if not ids or any(not value for value in ids):
        raise ValueError(f"{path} contains no valid complex IDs")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path} contains duplicate complex IDs")
    return ids


def _fasta_ids(path: Path) -> list[str]:
    ids: list[str] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or not line.startswith(">"):
            continue
        header = line[1:]
        if not header.endswith("|protein"):
            raise ValueError(f"{path}:{line_number}: invalid FASTA header {header!r}")
        complex_id, separator, chain = header[: -len("|protein")].rpartition("_")
        if not separator or not complex_id or len(chain) != 1:
            raise ValueError(f"{path}:{line_number}: invalid FASTA header {header!r}")
        ids.append(complex_id)
    if not ids:
        raise ValueError(f"{path} contains no FASTA records")
    return ids


def _require_equal_sets(label: str, expected: Iterable[str], observed: Iterable[str]) -> None:
    expected_set = set(expected)
    observed_set = set(observed)
    if expected_set != observed_set:
        raise ValueError(
            f"{label} target set differs from registry: "
            f"missing={sorted(expected_set - observed_set)[:20]}, "
            f"unexpected={sorted(observed_set - expected_set)[:20]}"
        )


ARTIFACTS = (*REQUIRED_FILES, "benchmark/structures")
SOURCE_FILES = ("run_manifest.json", *REQUIRED_FILES)
RELEASE_KINDS = ("full", "subset", "max-size-test")
CAPACITY_ARTIFACTS = (*ARTIFACTS, "provenance/candidates.json", "provenance/source_summary.csv")
CAPACITY_REGISTRY_FIELDS = ("complex_id", "pdb", "Hchain", "Lchain", "antigen_chain", "residue_length")
CAPACITY_FEATURE_FIELDS = ("complex_id", "pdb_id", "is_nanobody", "residue_length")


def full_records(manifest: dict) -> list[dict]:
    if (manifest.get("schema_version") != 1 or manifest.get("status") != "completed"
            or manifest.get("release_kind", "full") != "full"):
        raise ValueError("Source must be a completed full AbAgBench release")
    execution = manifest.get("execution")
    if not isinstance(execution, dict) or execution.get("canonical") is not True:
        raise ValueError("AbAgBench release is non-canonical; a canonical GPU run is required")
    return [_require_output_record(manifest, stage, path) for stage, path in (
        ("6", "benchmark/registry.tsv"), ("6", "benchmark/benchmark.fasta"),
        ("7", "benchmark/dataset_features.csv"), ("6", "benchmark/structures"))]


def read_table(path: Path, delimiter: str) -> tuple[list[str], dict[str, dict[str, str]]]:
    _read_delimited_ids(path, delimiter)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if len(set(fields)) != len(fields) or any(None in row or any(v is None for v in row.values()) for row in rows):
        raise ValueError(f"Malformed table: {path}")
    if any(row["complex_id"] != row["complex_id"].strip() for row in rows):
        raise ValueError(f"Unnormalized complex ID: {path}")
    return fields, {row["complex_id"]: row for row in rows}


def read_fasta(path: Path) -> dict[str, dict[str, str]]:
    records: dict[str, dict[str, str]] = {}
    current = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if not line.endswith("|protein"):
                raise ValueError(f"Invalid protein FASTA header: {line}")
            target, separator, chain = line[1:-len("|protein")].rpartition("_")
            if not separator or not target or len(chain) != 1 or chain in records.get(target, {}):
                raise ValueError(f"Invalid/duplicate FASTA header: {line}")
            records.setdefault(target, {})[chain] = ""
            current = (target, chain)
        else:
            if current is None:
                raise ValueError(f"Sequence before FASTA header: {path}")
            records[current[0]][current[1]] += line
    if not records or any(not seq for chains in records.values() for seq in chains.values()):
        raise ValueError(f"Missing/empty FASTA sequences: {path}")
    return records


def select_targets(registry: dict, features: dict, criteria: dict) -> list[str]:
    if not isinstance(criteria, dict) or set(criteria) != {"ids", "complex_type", "max_residues", "count", "seed"}:
        raise ValueError("Invalid subset selection criteria")
    ids, kind, maximum, count, seed = (criteria[k] for k in
        ("ids", "complex_type", "max_residues", "count", "seed"))
    if kind not in {"all", "nanobody", "antibody"}:
        raise ValueError("complex_type must be all, nanobody or antibody")
    if ids is not None:
        if (not isinstance(ids, list) or not ids or any(not isinstance(t, str) for t in ids)
                or len(ids) != len(set(ids))):
            raise ValueError("Requested IDs must be nonempty and unique")
        missing = set(ids) - set(registry)
        if missing:
            raise ValueError(f"Unknown complex IDs: {sorted(missing)}")
    if maximum is not None and (type(maximum) is not int or maximum < 1):
        raise ValueError("max_residues must be a positive integer")
    if count is not None and (type(count) is not int or count < 1):
        raise ValueError("count must be a positive integer")
    if (count is None) != (seed is None) or (seed is not None and (type(seed) is not int or seed < 0)):
        raise ValueError("Random sampling requires both count and a nonnegative seed")
    if ids is None and kind == "all" and maximum is None and count is None:
        raise ValueError("Specify at least one subset selection criterion")
    candidates = []
    for target in sorted(ids if ids is not None else registry):
        row, feature = registry[target], features[target]
        h, light, antigen = (row.get(k, "").strip() for k in ("Hchain", "Lchain", "antigen_chain"))
        light = "" if light.lower() == "nan" else light
        chains = [h, antigen] + ([light] if light else [])
        if (not re.fullmatch(r"[0-9a-z]{4}_[0-9A-Za-z]{2,3}", target)
                or any(not re.fullmatch(r"[0-9A-Za-z]", c) for c in chains)
                or len(set(chains)) != len(chains) or target.split("_", 1)[1] != h + light + antigen):
            raise ValueError(f"Unsupported heavy/light/single-antigen chain mapping: {target}")
        nanobody = not light
        flag = feature.get("is_nanobody", "").lower()
        if flag not in {"true", "false"} or (flag == "true") != nanobody:
            raise ValueError(f"Feature/registry nanobody status differs: {target}")
        try:
            size = float(feature["residue_length"])
        except (KeyError, ValueError) as error:
            raise ValueError(f"Invalid residue_length for {target}") from error
        if not math.isfinite(size) or size < 1 or not size.is_integer():
            raise ValueError(f"Invalid residue_length for {target}")
        if (kind == "nanobody" and not nanobody) or (kind == "antibody" and nanobody):
            continue
        if maximum is not None and size > maximum:
            continue
        candidates.append(target)
    if not candidates:
        raise ValueError("Selection contains no complexes")
    if count is not None:
        if count > len(candidates):
            raise ValueError(f"Requested {count} complexes, but only {len(candidates)} qualify")
        # Seeded hash ranking is reproducible across Python versions and input order.
        candidates = sorted(candidates, key=lambda target: (
            hashlib.sha256(f"{seed}\0{target}".encode("utf-8")).digest(), target))[:count]
    return sorted(candidates)


def subset_records(release_dir: Path, manifest: dict) -> list[dict]:
    export = manifest.get("subset_export")
    if (not isinstance(export, dict) or export.get("status") != "completed"
            or manifest.get("execution", {}).get("canonical") is not False
            or manifest.get("stages")):
        raise ValueError("Invalid completed subset export record")
    outputs = export.get("outputs", [])
    if (not isinstance(outputs, list) or any(not isinstance(r, dict) or not isinstance(r.get("path"), str) for r in outputs)
            or sorted(r.get("path", "") for r in outputs) != sorted(ARTIFACTS)):
        raise ValueError("Subset must declare exactly the required dataset outputs")
    expected_kinds = {p: ("directory" if p.endswith("/structures") else "file") for p in ARTIFACTS}
    if any(r.get("kind") != expected_kinds[r["path"]] for r in outputs):
        raise ValueError("Invalid subset output artifact kind")
    source = manifest.get("source")
    if not isinstance(source, dict) or not re.fullmatch(r"[0-9a-f]{64}", str(source.get("manifest_sha256", ""))):
        raise ValueError("Missing source release manifest identity")
    source_root = release_dir / "provenance/source"
    if _sha256(source_root / "run_manifest.json") != source["manifest_sha256"]:
        raise ValueError("Source release manifest hash differs")
    parent = json.loads((source_root / "run_manifest.json").read_text(encoding="utf-8"))
    for record in full_records(parent):
        if record["path"] in REQUIRED_FILES:
            _verify_record(source_root, record)
    return outputs


def validate_subset_contents(release_dir: Path, manifest: dict, registry_ids: list[str]) -> None:
    parent = release_dir / "provenance/source/benchmark"
    _, source_registry = read_table(parent / "registry.tsv", "\t")
    _, source_features = read_table(parent / "dataset_features.csv", ",")
    source_fasta = read_fasta(parent / "benchmark.fasta")
    _require_equal_sets("source features", source_registry, source_features)
    _require_equal_sets("source FASTA", source_registry, source_fasta)
    selected = select_targets(source_registry, source_features, manifest.get("selection"))
    if manifest.get("selected_targets") != selected or sorted(registry_ids) != selected:
        raise ValueError("Subset targets differ from the recorded selection")
    benchmark = release_dir / "benchmark"
    for name, delimiter in (("registry.tsv", "\t"), ("dataset_features.csv", ",")):
        source_fields, source_rows = read_table(parent / name, delimiter)
        fields, rows = read_table(benchmark / name, delimiter)
        if fields != source_fields or rows != {t: source_rows[t] for t in selected}:
            raise ValueError(f"Subset {name} differs from source rows")
    if read_fasta(benchmark / "benchmark.fasta") != {t: source_fasta[t] for t in selected}:
        raise ValueError("Subset FASTA differs from source sequences")
    for target in selected:
        chains = source_fasta[target]
        row = source_registry[target]
        expected = {row["Hchain"], row["antigen_chain"]}
        if row.get("Lchain", "").strip().lower() not in {"", "nan"}:
            expected.add(row["Lchain"])
        if set(chains) != expected or any(set(seq) - set("ACDEFGHIKLMNPQRSTVWY") for seq in chains.values()):
            raise ValueError(f"Invalid subset protein chains/sequences: {target}")
    native_hashes = manifest.get("source", {}).get("selected_native_sha256")
    observed = {t: _sha256(benchmark / "structures" / f"{t}.cif") for t in selected}
    if native_hashes != observed:
        raise ValueError("Subset structures differ from recorded source structure hashes")


def validate_capacity_selection(criteria: dict) -> None:
    """Validate capacity options independently of candidate availability."""
    if not isinstance(criteria, dict) or set(criteria) != {
        "min_residues", "max_residues", "bin_edges", "per_bin", "pdb_ids", "complex_ids", "rule"
    } or criteria["rule"] not in ("largest_per_bin_then_complex_id", "largest_up_to_per_bin_then_complex_id"):
        raise ValueError("Invalid max-size-test selection criteria")
    lower, upper, edges, per_bin, pdbs, ids = (criteria[k] for k in (
        "min_residues", "max_residues", "bin_edges", "per_bin", "pdb_ids", "complex_ids"))
    for bound in (lower, upper):
        if bound is not None and (type(bound) is not int or bound < 1):
            raise ValueError("Capacity residue bounds must be positive integers")
    if lower is not None and upper is not None and lower > upper:
        raise ValueError("Capacity minimum exceeds maximum")
    for names, pattern, label in ((pdbs, r"[0-9a-z]{4}", "PDB IDs"),
                                  (ids, r"[0-9a-z]{4}_[0-9A-Za-z]{2,3}", "complex IDs")):
        if names is not None and (not isinstance(names, list) or not names
                or any(not isinstance(t, str) or not re.fullmatch(pattern, t) for t in names)
                or len(set(names)) != len(names)):
            raise ValueError(f"Capacity {label} must be nonempty, valid and unique")
    if pdbs is not None and ids is not None:
        raise ValueError("Choose PDB IDs or complex IDs, not both")
    if edges is not None:
        if (not isinstance(edges, list) or len(edges) < 2
                or any(type(n) is not int or n < 1 for n in edges)
                or edges != sorted(set(edges)) or lower != edges[0] or upper != edges[-1]
                or type(per_bin) is not int or per_bin < 1):
            raise ValueError("Capacity bins require increasing edges, matching bounds and positive per_bin")
    elif per_bin is not None:
        raise ValueError("per_bin requires capacity bins")
    if edges is None and pdbs is None and ids is None:
        raise ValueError("Specify capacity bins, PDB IDs or complex IDs")


def capacity_selection_report(candidates: dict, criteria: dict) -> tuple[list[str], list[dict]]:
    """Replay selection and account for every bin, including empty bins."""
    validate_capacity_selection(criteria)
    lower, upper, edges, per_bin, pdbs, ids = (criteria[k] for k in (
        "min_residues", "max_residues", "bin_edges", "per_bin", "pdb_ids", "complex_ids"))
    lengths = {target: sum(map(len, entry["chains"].values())) for target, entry in candidates.items()}
    eligible = {t for t, n in lengths.items() if (lower is None or n >= lower) and (upper is None or n <= upper)}
    if ids is not None:
        missing = set(ids) - eligible
        if missing:
            raise ValueError(f"Requested capacity complexes unavailable or outside bounds: {sorted(missing)}")
        eligible &= set(ids)
    if pdbs is not None:
        chosen = set()
        for pdb in pdbs:
            matching = sorted(t for t in eligible if t.split("_", 1)[0] == pdb)
            if len(matching) != 1:
                raise ValueError(f"PDB {pdb}: expected one eligible complex, found {matching}; use --complex-ids to disambiguate")
            chosen.update(matching)
        eligible = chosen
    partial = criteria["rule"] == "largest_up_to_per_bin_then_complex_id"
    if not eligible and not (partial and edges is not None):
        raise ValueError("Capacity selection contains no complexes")
    coverage = []
    if edges is not None:
        selected, shortages = [], []
        for index, (lo, hi) in enumerate(zip(edges, edges[1:])):
            final = index == len(edges) - 2
            members = sorted((t for t in eligible if lo <= lengths[t] and
                              (lengths[t] < hi or (final and lengths[t] == hi))),
                             key=lambda t: (-lengths[t], t))
            if len(members) < per_bin:
                shortages.append(f"[{lo}, {hi}{']' if final else ')'}: {len(members)} available, {per_bin} required")
            chosen = members[:per_bin]
            coverage.append({"lower": lo, "upper": hi, "upper_inclusive": final,
                             "available": len(members), "requested": per_bin,
                             "selected_targets": chosen, "selected": len(chosen),
                             "shortfall": max(0, per_bin - len(chosen))})
            selected.extend(chosen)
        # Existing manifests retain their original exact-count selection policy.
        if shortages and not partial:
            raise ValueError("Insufficient capacity-bin coverage: " + "; ".join(shortages))
        eligible = set(selected)
    selected = sorted(eligible)
    if len({t.lower() for t in selected}) != len(selected):
        raise ValueError("Selected capacity IDs collide on case-insensitive filesystems")
    return selected, coverage


def select_capacity_targets(candidates: dict, criteria: dict) -> list[str]:
    """Return the targets declared by the recorded capacity selection policy."""
    return capacity_selection_report(candidates, criteria)[0]


def capacity_records(manifest: dict) -> list[dict]:
    export = manifest.get("max_size_test_export")
    if (manifest.get("purpose") != "machine_capacity_test"
            or not isinstance(manifest.get("execution"), dict)
            or manifest["execution"].get("canonical") is not False
            or manifest.get("stages") or manifest.get("subset_export")
            or not isinstance(export, dict) or export.get("status") != "completed"):
        raise ValueError("Invalid completed max-size-test export record")
    outputs = export.get("outputs")
    if (not isinstance(outputs, list) or any(not isinstance(r, dict) or not isinstance(r.get("path"), str) for r in outputs)
            or sorted(r.get("path", "") for r in outputs) != sorted(CAPACITY_ARTIFACTS)):
        raise ValueError("Max-size-test must declare exactly its required outputs")
    if any(r.get("kind") != ("directory" if r["path"].endswith("/structures") else "file") for r in outputs):
        raise ValueError("Invalid max-size-test artifact kind")
    return outputs


def validate_capacity_contents(release_dir: Path, manifest: dict) -> None:
    catalog = _read_json(release_dir / "provenance/candidates.json")
    if (not isinstance(catalog, dict) or type(catalog.get("schema_version")) is not int
            or catalog["schema_version"] != 1 or not isinstance(catalog.get("candidates"), list)
            or not isinstance(catalog.get("excluded"), list)):
        raise ValueError("Invalid capacity candidate catalog")
    for entry in catalog["excluded"]:
        if (not isinstance(entry, dict) or set(entry) != {"summary_row", "pdb_id", "complex_id", "reason"}
                or type(entry["summary_row"]) is not int or entry["summary_row"] < 2
                or any(not isinstance(entry[k], str) or not entry[k] for k in ("pdb_id", "complex_id", "reason"))):
            raise ValueError("Invalid capacity input-exclusion record")
    with (release_dir / "provenance/source_summary.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)) or not {"Hchain", "Lchain", "antigen_chain", "antigen_type"} <= set(fields) or not {"pdb", "PDB"} & set(fields):
            raise ValueError("Invalid capacity source-summary columns")
        source_mappings = set()
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise ValueError("Malformed capacity source-summary row")
            pdb = re.fullmatch(r"(?:pdb_0000)?([0-9a-z]{4})", row.get("pdb", row.get("PDB", "")).strip().lower())
            if pdb is None:
                raise ValueError("Invalid PDB ID in capacity source summary")
            light = row["Lchain"].strip()
            light = "" if light.lower() in {"", "nan", "na", "none"} else light
            if "protein" in row["antigen_type"].lower():
                source_mappings.add((pdb[1], row["Hchain"].strip(), light, row["antigen_chain"].strip()))
    candidates, registry, features, source_hashes = {}, {}, {}, {}
    for entry in catalog["candidates"]:
        if not isinstance(entry, dict) or set(entry) != {"registry", "chains", "source_cif", "source_cif_sha256"}:
            raise ValueError("Invalid capacity candidate record")
        row, chains = entry["registry"], entry["chains"]
        if (not isinstance(row, dict) or set(row) != set(CAPACITY_REGISTRY_FIELDS)
                or any(not isinstance(v, str) for v in row.values())
                or not isinstance(chains, dict) or any(not isinstance(s, str) or not s
                    or set(s) - set("ACDEFGHIKLMNPQRSTVWY") for s in chains.values())):
            raise ValueError("Invalid capacity chains or registry row")
        target = row["complex_id"]
        if tuple(row[k] for k in ("pdb", "Hchain", "Lchain", "antigen_chain")) not in source_mappings:
            raise ValueError(f"Capacity candidate differs from source summary: {target}")
        expected_chains = {row["Hchain"], row["antigen_chain"]} | ({row["Lchain"]} if row["Lchain"] else set())
        if (target in candidates or set(chains) != expected_chains
                or row["pdb"] != target.split("_", 1)[0]
                or row["residue_length"] != str(sum(map(len, chains.values())))):
            raise ValueError(f"Duplicate or inconsistent capacity candidate: {target}")
        source = entry["source_cif"]
        if (not isinstance(source, str) or not source or Path(source).is_absolute() or ".." in Path(source).parts
                or not isinstance(entry["source_cif_sha256"], str)
                or not re.fullmatch(r"[0-9a-f]{64}", entry["source_cif_sha256"])):
            raise ValueError(f"Invalid capacity source identity: {target}")
        if source in source_hashes and source_hashes[source] != entry["source_cif_sha256"]:
            raise ValueError(f"Conflicting capacity source hashes: {source}")
        source_hashes[source] = entry["source_cif_sha256"]
        candidates[target], registry[target] = entry, row
        features[target] = {"complex_id": target, "pdb_id": row["pdb"],
                            "is_nanobody": str(not row["Lchain"]), "residue_length": row["residue_length"]}
    # Reuse chain/ID/type validation without applying benchmark cohort filters.
    select_targets(registry, features, {"ids": sorted(registry), "complex_type": "all", "max_residues": None,
                                       "count": None, "seed": None})
    selected, coverage = capacity_selection_report(candidates, manifest.get("selection"))
    if not selected:
        raise ValueError("Capacity release must contain at least one selected complex")
    if manifest["selection"]["rule"] == "largest_up_to_per_bin_then_complex_id" or "bin_coverage" in manifest:
        if json.dumps(manifest.get("bin_coverage"), sort_keys=True) != json.dumps(coverage, sort_keys=True):
            raise ValueError("Capacity bin coverage differs from recorded selection")
    if manifest.get("selected_targets") != selected:
        raise ValueError("Capacity targets differ from recorded selection")
    for name, delimiter, expected_fields, expected_rows in (
        ("registry.tsv", "\t", CAPACITY_REGISTRY_FIELDS, registry),
        ("dataset_features.csv", ",", CAPACITY_FEATURE_FIELDS, features),
    ):
        fields, rows = read_table(release_dir / "benchmark" / name, delimiter)
        if fields != list(expected_fields) or rows != {t: expected_rows[t] for t in selected}:
            raise ValueError(f"Capacity {name} differs from selected candidates")
    if read_fasta(release_dir / "benchmark/benchmark.fasta") != {t: candidates[t]["chains"] for t in selected}:
        raise ValueError("Capacity FASTA differs from selected candidates")
    lengths = {t: int(registry[t]["residue_length"]) for t in selected}
    low, high = min(lengths.values()), max(lengths.values())
    expected = {"complexes": len(selected), "residue_length": {"min": low, "max": high,
        "min_targets": sorted(t for t in selected if lengths[t] == low),
        "max_targets": sorted(t for t in selected if lengths[t] == high)}}
    if json.dumps(manifest.get("dataset_summary"), sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise ValueError("Capacity dataset summary differs from selected sequences")


def validate_release(release_dir: Path, release_kind: str = "full") -> dict[str, Any]:
    release_dir = release_dir.resolve()
    manifest_path = release_dir / "run_manifest.json"
    if not manifest_path.is_file() or manifest_path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing/empty AbAgBench run manifest: {manifest_path}")
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise ValueError(f"Unsupported AbAgBench manifest schema: {manifest_path}")
    if manifest.get("status") != "completed":
        raise ValueError(f"AbAgBench run is not complete: status={manifest.get('status')!r}")
    if release_kind not in RELEASE_KINDS:
        raise ValueError("DATASET_RELEASE_KIND must be full or subset or max-size-test")
    observed_kind = manifest.get("release_kind", "full")
    if observed_kind != release_kind:
        raise ValueError(f"Release kind mismatch: expected {release_kind}, found {observed_kind}")
    if release_kind == "full":
        records = full_records(manifest)
    elif release_kind == "subset":
        records = subset_records(release_dir, manifest)
    else:
        records = capacity_records(manifest)
    for record in records:
        _verify_record(release_dir, record)

    for relative in REQUIRED_FILES:
        path = release_dir / relative
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing/empty dataset release artifact: {path}")

    benchmark = release_dir / "benchmark"
    registry_ids = _read_delimited_ids(benchmark / "registry.tsv", "\t")
    feature_ids = _read_delimited_ids(benchmark / "dataset_features.csv", ",")
    fasta_ids = _fasta_ids(benchmark / "benchmark.fasta")
    structure_ids = [path.stem for path in sorted((benchmark / "structures").glob("*.cif"))]
    _require_equal_sets("FASTA", registry_ids, fasta_ids)
    _require_equal_sets("dataset features", registry_ids, feature_ids)
    _require_equal_sets("native structures", registry_ids, structure_ids)

    if release_kind == "subset":
        validate_subset_contents(release_dir, manifest, registry_ids)
    elif release_kind == "max-size-test":
        validate_capacity_contents(release_dir, manifest)

    return {
        "release_kind": release_kind,
        "release_dir": str(release_dir),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "complexes": len(registry_ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-release", required=True, type=Path)
    parser.add_argument("--release-kind", choices=RELEASE_KINDS, default="full")
    args = parser.parse_args()
    try:
        summary = validate_release(args.dataset_release, args.release_kind)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
