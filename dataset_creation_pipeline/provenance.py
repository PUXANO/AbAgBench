"""Input fingerprints, atomic manifests, and run metadata."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Iterable

def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def hash_paths(paths: Iterable[Path], root: Path) -> tuple[list[dict[str, Any]], str]:
    """Hash files exactly and return the ordered file manifest plus tree digest."""
    files = sorted({p.resolve() for p in paths}, key=lambda p: str(p))
    records = []
    tree = hashlib.sha256()
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(f"Input file disappeared while hashing: {path}")
        rel = path.relative_to(root.resolve()).as_posix()
        file_hash = sha256_file(path)
        size = path.stat().st_size
        records.append({"path": rel, "size_bytes": size, "sha256": file_hash})
        tree.update(rel.encode("utf-8"))
        tree.update(b"\0")
        tree.update(file_hash.encode("ascii"))
        tree.update(b"\n")
    return records, tree.hexdigest()


def hash_artifact(path: Path, base: Path) -> dict[str, Any]:
    path = path.resolve()
    if path.is_file():
        return {
            "path": Path(os.path.relpath(path, base.resolve())).as_posix(),
            "kind": "file",
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    if path.is_dir():
        files, digest = hash_paths((p for p in path.rglob("*") if p.is_file()), path)
        return {
            "path": Path(os.path.relpath(path, base.resolve())).as_posix(),
            "kind": "directory",
            "sha256": digest,
            "file_count": len(files),
            "size_bytes": sum(item["size_bytes"] for item in files),
        }
    raise FileNotFoundError(f"Expected stage artifact does not exist: {path}")


def fingerprint(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def software_versions(foldseek_version: str | None = None) -> dict[str, Any]:
    packages = (
        "numpy", "pandas", "scipy", "matplotlib", "biopython", "anarcii",
        "torch", "gemmi", "psutil",
    )
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "python": sys.version,
        "python_executable": Path(sys.executable).name,
        "platform": platform.platform(),
        "packages": versions,
        "foldseek": foldseek_version,
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    from .utils import atomic_write_json

    atomic_write_json(path, manifest)


def output_fingerprints(paths: Iterable[Path], base: Path) -> list[dict[str, Any]]:
    return [hash_artifact(path, base) for path in paths]


def output_fingerprints_match(recorded: list[dict[str, Any]], base: Path) -> bool:
    try:
        current = output_fingerprints(
            (base / item["path"] for item in recorded), base
        )
    except (FileNotFoundError, ValueError):
        return False
    return current == recorded
