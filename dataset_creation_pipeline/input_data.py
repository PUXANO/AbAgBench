"""SAbDab structure discovery helpers."""
from __future__ import annotations

from pathlib import Path


def discover_structure_files(structures_dir: Path) -> list[Path]:
    """Find SAbDab CIF files in either the extracted or gzip-compressed layout."""
    paths = [*structures_dir.rglob("*.cif"), *structures_dir.rglob("*.cif.gz")]
    return sorted(set(paths))
