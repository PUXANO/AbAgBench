"""Repository-relative data and external input paths."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("ABAG_DATA_DIR", REPOSITORY_ROOT / "inputs")).expanduser().resolve()
OUTPUT_ROOT = Path(os.environ.get("ABAG_OUTPUT_DIR", DATA_ROOT.parent / "outputs" / "abagbench")).expanduser().resolve()
INTERMEDIATE_ROOT = OUTPUT_ROOT / "intermediate"
SCRATCH_ROOT = OUTPUT_ROOT / "scratch"
LOGS_ROOT = OUTPUT_ROOT / "logs"
BENCHMARK_DIR = DATA_ROOT / "benchmark"
PLOTS_ROOT = Path(os.environ.get("ABAG_PLOTS_DIR", OUTPUT_ROOT / "plots")).expanduser().resolve()

_sabdab_root = os.environ.get("SABDAB2_DIR")
SABDAB2_DATA = Path(_sabdab_root).expanduser().resolve() if _sabdab_root else None
SABDAB2_SUMMARY_FILE = os.environ.get("SABDAB2_SUMMARY_FILE", "sabdab_summary_all.csv")
SABDAB2_STRUCTURES_NAME = os.environ.get("SABDAB2_STRUCTURES_DIR", "sabdab_all_structures")
SABDAB2_SUMMARY_CSV = SABDAB2_DATA / SABDAB2_SUMMARY_FILE if SABDAB2_DATA else None
SABDAB2_STRUCTURE_DIR = SABDAB2_DATA / SABDAB2_STRUCTURES_NAME if SABDAB2_DATA else None


@dataclass(frozen=True)
class PipelinePaths:
    repository_root: Path = REPOSITORY_ROOT
    data_root: Path = DATA_ROOT
    output_root: Path = OUTPUT_ROOT
    logs: Path = LOGS_ROOT
    intermediate: Path = INTERMEDIATE_ROOT
    scratch: Path = SCRATCH_ROOT
    benchmark: Path = BENCHMARK_DIR
    plots: Path = PLOTS_ROOT
    sabdab_root: Path | None = SABDAB2_DATA
    sabdab_summary: Path | None = SABDAB2_SUMMARY_CSV
    sabdab_structures: Path | None = SABDAB2_STRUCTURE_DIR


PATHS = PipelinePaths()


def require_sabdab_input() -> tuple[Path, Path, Path]:
    if PATHS.sabdab_root is None or PATHS.sabdab_summary is None or PATHS.sabdab_structures is None:
        raise FileNotFoundError("SAbDab2 input is not configured; pass --sabdab-dir or set SABDAB2_DIR")
    return PATHS.sabdab_root, PATHS.sabdab_summary, PATHS.sabdab_structures
