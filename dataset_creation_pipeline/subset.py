"""Export a self-contained selection from a completed canonical release."""
from __future__ import annotations

import csv
import datetime as dt
import json
import shutil
import tempfile
from pathlib import Path

from . import __version__
from .dataset_summary import summarize_dataset
from .release import (
    ARTIFACTS, SOURCE_FILES, _hash_directory, _sha256, read_fasta, read_table,
    select_targets, validate_release,
)


def artifact_record(root: Path, relative: str) -> dict:
    path = root / relative
    if path.is_dir():
        digest, count, size = _hash_directory(path)
        return {"path": relative, "kind": "directory", "sha256": digest,
                "file_count": count, "size_bytes": size}
    return {"path": relative, "kind": "file", "sha256": _sha256(path),
            "size_bytes": path.stat().st_size}


def export_subset(source: Path, output: Path, *, ids: list[str] | None = None,
                  complex_type: str = "all", max_residues: int | None = None,
                  count: int | None = None, seed: int | None = None) -> dict:
    source, output = source.expanduser().resolve(), output.expanduser().resolve()
    if source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError("Source dataset and destination experiment must not overlap")
    if output.exists() and not output.is_dir():
        raise FileExistsError(f"Destination is not an experiment directory: {output}")
    # An experiment may already contain run.env or a runner checkout. Export only
    # into unused inputs/ with absent or empty outputs/, preserving other content.
    for name in ("inputs", "outputs"):
        path = output / name
        if path.is_symlink() or (path.exists() and (not path.is_dir() or any(path.iterdir()))):
            raise FileExistsError(f"Destination {name}/ must be absent or empty: {path}")
    summary = validate_release(source, "full")
    source_hashes = {relative: _sha256(source / relative) for relative in SOURCE_FILES}
    criteria = {"ids": ids, "complex_type": complex_type, "max_residues": max_residues,
                "count": count, "seed": seed}
    _, registry = read_table(source / "benchmark/registry.tsv", "\t")
    _, features = read_table(source / "benchmark/dataset_features.csv", ",")
    selected = select_targets(registry, features, criteria)
    fastas = read_fasta(source / "benchmark/benchmark.fasta")
    native_hashes = {t: _sha256(source / "benchmark/structures" / f"{t}.cif") for t in selected}

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        dataset = staging / "inputs"
        benchmark = dataset / "benchmark"
        (benchmark / "structures").mkdir(parents=True)
        for relative in SOURCE_FILES:
            destination = dataset / "provenance/source" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / relative, destination)
            if _sha256(destination) != source_hashes[relative]:
                raise ValueError(f"Source changed during export: {relative}")
        for name, delimiter in (("registry.tsv", "\t"), ("dataset_features.csv", ",")):
            fields, rows = read_table(source / "benchmark" / name, delimiter)
            with (benchmark / name).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, delimiter=delimiter)
                writer.writeheader()
                writer.writerows(rows[t] for t in selected)
        with (benchmark / "benchmark.fasta").open("w", encoding="utf-8") as handle:
            for target in selected:
                for chain, sequence in sorted(fastas[target].items()):
                    handle.write(f">{target}_{chain}|protein\n{sequence}\n")
        for target in selected:
            shutil.copyfile(source / "benchmark/structures" / f"{target}.cif",
                            benchmark / "structures" / f"{target}.cif")
        manifest = {
            "schema_version": 1, "release_kind": "subset", "status": "completed",
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "execution": {"canonical": False, "command": "python -m dataset_creation_pipeline subset"},
            "software": {"abagbench": __version__},
            "source": {"manifest_sha256": summary["manifest_sha256"],
                       "selected_native_sha256": native_hashes},
            "selection": criteria, "selected_targets": selected,
            "dataset_summary": summarize_dataset(dataset),
            "subset_export": {"status": "completed", "outputs": [
                artifact_record(dataset, p) for p in ARTIFACTS]},
        }
        (dataset / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        validate_release(dataset, "subset")
        # Validate source again before publication, including all native hashes.
        if validate_release(source, "full")["manifest_sha256"] != summary["manifest_sha256"]:
            raise ValueError("Source release changed during export")
        (staging / "outputs").mkdir()
        if not output.exists():
            staging.rename(output)
        else:
            for name in ("inputs", "outputs"):
                path = output / name
                if path.is_symlink() or (path.exists() and (not path.is_dir() or any(path.iterdir()))):
                    raise FileExistsError(f"Destination {name}/ changed during export: {path}")
            # Replacing an empty directory publishes the dataset atomically.
            dataset.rename(output / "inputs")
            (output / "outputs").mkdir(exist_ok=True)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    result = validate_release(output / "inputs", "subset")
    result["experiment_dir"] = str(output)
    return result
