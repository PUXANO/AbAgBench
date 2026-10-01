"""Read-only input, dependency, and output preflight checks."""
from __future__ import annotations

import importlib.metadata
import gzip
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .config import PipelineConfig
from .foldseek_utils import FOLDSEEK_FORMAT
from .input_data import discover_structure_files


REQUIRED_SUMMARY_COLUMNS = {
    "pdb", "date", "Hchain", "Lchain", "antigen_chain", "antigen_type",
    "chainsharing_construct", "compound", "type", "method", "resolution",
    "r_factor", "short_header",
}
REQUIRED_PACKAGE_VERSIONS = {
    "numpy": "1.26.4",
    "pandas": "2.3.3",
    "scipy": "1.16.3",
    "matplotlib": "3.10.6",
    "biopython": "1.87",
    "anarcii": "2.0.8",
    "psutil": "7.0.0",
}


def _missing_foldseek_format_fields(help_text: str) -> list[str]:
    """Return required custom-output fields absent from this FoldSeek build."""
    return [
        field for field in FOLDSEEK_FORMAT.split(",")
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(field)}(?![A-Za-z0-9_])", help_text) is None
    ]


def run_preflight(config: PipelineConfig, require_foldseek: bool = True) -> dict[str, Any]:
    """Validate inputs and runtime dependencies without creating output files."""
    config.validate()
    errors: list[str] = []
    notes: list[str] = []
    if config.input.sabdab_dir is None:
        raise FileNotFoundError("SAbDab2 input is required; pass --sabdab-dir or set it in the TOML config")

    sabdab_dir = config.input.sabdab_dir
    summary_path = sabdab_dir / config.input.summary_file
    structures_dir = sabdab_dir / config.input.structures_dir
    if not sabdab_dir.is_dir():
        errors.append(f"SAbDab2 directory does not exist: {sabdab_dir}")
    if not summary_path.is_file():
        errors.append(f"SAbDab2 summary file does not exist: {summary_path}")
    if not structures_dir.is_dir():
        errors.append(f"SAbDab2 structure directory does not exist: {structures_dir}")

    summary_rows = None
    if summary_path.is_file():
        try:
            import pandas as pd

            frame = pd.read_csv(summary_path, low_memory=False)
            summary_rows = len(frame)
            available = set(frame.columns)
            if "pdb" not in available and "PDB" in available:
                available.add("pdb")
            missing = REQUIRED_SUMMARY_COLUMNS - available
            if missing:
                errors.append(f"Summary is missing required columns: {sorted(missing)}")
            if frame.empty:
                errors.append("SAbDab2 summary has no rows")
            if "pdb" in frame.columns or "PDB" in frame.columns:
                id_col = "pdb" if "pdb" in frame.columns else "PDB"
                ids = frame[id_col].astype(str).str.strip()
                if ids.eq("").any() or ids.str.lower().eq("nan").any():
                    errors.append("SAbDab2 summary contains empty PDB IDs")
            if "date" in frame.columns:
                parsed = pd.to_datetime(frame["date"], format="%Y/%m/%d", errors="coerce")
                if parsed.isna().any():
                    errors.append(f"{int(parsed.isna().sum())} release dates are missing or malformed")
        except Exception as exc:
            errors.append(f"Could not read SAbDab2 summary: {exc}")

    cif_count = 0
    cif_size = 0
    uncompressed_cif_size = 0
    if structures_dir.is_dir():
        for path in discover_structure_files(structures_dir):
            cif_count += 1
            try:
                compressed_size = path.stat().st_size
                cif_size += compressed_size
                if path.name.endswith(".cif.gz"):
                    with path.open("rb") as handle:
                        handle.seek(-4, os.SEEK_END)
                        trailer = handle.read(4)
                    if len(trailer) != 4:
                        raise ValueError("truncated gzip trailer")
                    uncompressed_cif_size += int.from_bytes(trailer, "little")
                else:
                    uncompressed_cif_size += compressed_size
            except OSError as exc:
                errors.append(f"Cannot inspect input structure {path}: {exc}")
            except ValueError as exc:
                errors.append(f"Cannot estimate expanded size for {path}: {exc}")
        if cif_count == 0:
            errors.append(f"No .cif files found under {structures_dir}")

    if sys.version_info < (3, 12):
        errors.append(f"Python 3.12 or newer is required; found {platform.python_version()}")
    package_versions = {}
    for package, required_version in REQUIRED_PACKAGE_VERSIONS.items():
        try:
            found_version = importlib.metadata.version(package)
            package_versions[package] = found_version
            if found_version != required_version:
                errors.append(
                    f"{package} {required_version} is required; found {found_version}"
                )
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = None
            errors.append(
                f"{package} {required_version} is not installed; install the pinned environment before running"
            )

    foldseek_path = config.resources.foldseek_bin or os.environ.get("FOLDSEEK_BIN") or shutil.which("foldseek")
    foldseek_version = None
    chain_tm_threshold_supported = None
    missing_foldseek_format_fields: list[str] = []
    if require_foldseek:
        if not foldseek_path:
            errors.append("FoldSeek executable not found; install FoldSeek or pass --foldseek-bin")
        else:
            try:
                version_result = subprocess.run(
                    [foldseek_path, "version"], capture_output=True, text=True, check=False, timeout=30
                )
                if version_result.returncode != 0:
                    errors.append(f"foldseek version failed: {version_result.stderr.strip()}")
                foldseek_version = (version_result.stdout or version_result.stderr).strip()
                help_result = subprocess.run(
                    [foldseek_path, "easy-multimersearch", "-h"],
                    capture_output=True, text=True, check=False, timeout=30,
                )
                help_text = help_result.stdout + help_result.stderr
                chain_tm_threshold_supported = "--chain-tm-threshold" in help_text
                missing_foldseek_format_fields = _missing_foldseek_format_fields(help_text)
                if missing_foldseek_format_fields:
                    errors.append(
                        "FoldSeek is missing required easy-multimersearch output field(s): "
                        + ", ".join(missing_foldseek_format_fields)
                        + ". Use the pinned Foldseek commit listed in the README; "
                        "these fields are required for the pipeline's interface scoring."
                    )
                if not chain_tm_threshold_supported:
                    notes.append(
                        "This FoldSeek build does not expose --chain-tm-threshold on easy-multimersearch. "
                        "The frozen value is 0 (no additional chain-TM cutoff), so the option is omitted; "
                        "the Foldseek build is still recorded in the run manifest."
                    )
                if config.resources.device == "gpu" and "--gpu" not in help_text:
                    errors.append("Selected FoldSeek build does not advertise GPU execution")
            except (OSError, subprocess.SubprocessError) as exc:
                errors.append(f"Could not inspect FoldSeek executable {foldseek_path}: {exc}")
    if config.resources.device == "gpu" and platform.system() != "Linux":
        errors.append("GPU FoldSeek execution is supported only by the Linux pipeline profile")

    output_parent = config.output.run_dir.parent
    nearest = output_parent
    while not nearest.exists() and nearest != nearest.parent:
        nearest = nearest.parent
    if not nearest.is_dir() or not os.access(nearest, os.W_OK):
        errors.append(f"Output parent is not writable: {nearest}")
    plots_parent = (config.output.plots_dir / config.output.run_name).parent
    nearest_plots = plots_parent
    while not nearest_plots.exists() and nearest_plots != nearest_plots.parent:
        nearest_plots = nearest_plots.parent
    if not nearest_plots.is_dir() or not os.access(nearest_plots, os.W_OK):
        errors.append(f"Plot output parent is not writable: {nearest_plots}")
    try:
        free_disk = shutil.disk_usage(nearest).free
    except OSError as exc:
        errors.append(f"Could not inspect free disk space at {nearest}: {exc}")
        free_disk = None
    if free_disk is not None and free_disk < 1:
        errors.append(f"No free disk space available at {nearest}")
    if free_disk is not None and uncompressed_cif_size and free_disk < uncompressed_cif_size:
        notes.append("Available disk is smaller than the estimated expanded SAbDab2 CIF collection; verify room for the FoldSeek training database and benchmark outputs.")

    if foldseek_version and "8dc75c74ad0eddab73cfd905963d13bf74dc012b" not in foldseek_version:
        notes.append(
            "Foldseek build differs from the pinned supported commit; treat the generated result as a separate dataset version."
        )

    report = {
        "sabdab_dir": sabdab_dir.name,
        "summary_file": summary_path.name,
        "summary_rows": summary_rows,
        "structures_dir": structures_dir.name,
        "structure_file_count": cif_count,
        "structure_size_bytes": cif_size,
        "estimated_uncompressed_structure_size_bytes": uncompressed_cif_size,
        "output_run_dir": config.output.run_dir.name,
        "free_disk_bytes": free_disk,
        # The resolved path is consumed internally by the stage runner; the
        # manifest serializer stores only the executable basename.
        "foldseek_binary": foldseek_path,
        "foldseek_version": foldseek_version,
        "foldseek_chain_tm_threshold_supported": chain_tm_threshold_supported,
        "foldseek_missing_format_fields": missing_foldseek_format_fields,
        "python_version": platform.python_version(),
        "package_versions": package_versions,
        "notes": notes,
        "errors": errors,
    }
    if errors:
        raise RuntimeError("Preflight failed:\n- " + "\n- ".join(errors))
    return report
