"""Observed whole-complex sizes of the final exported dataset."""
from __future__ import annotations

import math
from pathlib import Path

from .release import _require_equal_sets, read_fasta, read_table


def summarize_dataset(dataset: Path) -> dict:
    """Count exported complexes and cross-check full-sequence lengths."""
    benchmark = dataset / "benchmark"
    _, registry = read_table(benchmark / "registry.tsv", "\t")
    _, features = read_table(benchmark / "dataset_features.csv", ",")
    sequences = read_fasta(benchmark / "benchmark.fasta")
    _require_equal_sets("dataset features", registry, features)
    _require_equal_sets("FASTA", registry, sequences)
    lengths = {}
    for target, row in features.items():
        try:
            length = float(row["residue_length"])
        except (KeyError, ValueError) as error:
            raise ValueError(f"Invalid residue_length for {target}") from error
        if not math.isfinite(length) or length < 1 or not length.is_integer():
            raise ValueError(f"Invalid residue_length for {target}")
        if length != sum(len(sequence) for sequence in sequences[target].values()):
            raise ValueError(f"Feature residue_length differs from full FASTA sequences: {target}")
        lengths[target] = int(length)
    minimum, maximum = min(lengths.values()), max(lengths.values())
    return {
        "complexes": len(lengths),
        "residue_length": {
            "min": minimum, "max": maximum,
            "min_targets": sorted(t for t, length in lengths.items() if length == minimum),
            "max_targets": sorted(t for t, length in lengths.items() if length == maximum),
        },
    }


def check_summary(summary: dict) -> dict:
    """Reject malformed observed summaries rather than inventing bounds."""
    if not isinstance(summary, dict) or set(summary) != {"complexes", "residue_length"}:
        raise ValueError("Invalid dataset_summary")
    count, lengths = summary["complexes"], summary["residue_length"]
    if type(count) is not int or count < 1:
        raise ValueError("dataset_summary.complexes must be a positive integer")
    if not isinstance(lengths, dict) or set(lengths) != {"min", "max", "min_targets", "max_targets"}:
        raise ValueError("Invalid dataset_summary.residue_length")
    if any(type(lengths[k]) is not int or lengths[k] < 1 for k in ("min", "max")):
        raise ValueError("Dataset residue bounds must be positive integers")
    if lengths["min"] > lengths["max"]:
        raise ValueError("Dataset minimum residue count exceeds maximum")
    for key in ("min_targets", "max_targets"):
        targets = lengths[key]
        if (not isinstance(targets, list) or not targets or len(targets) > count
                or any(not isinstance(t, str) or not t for t in targets)
                or targets != sorted(set(targets))):
            raise ValueError(f"Invalid dataset_summary.residue_length.{key}")
    minimum, maximum = set(lengths["min_targets"]), set(lengths["max_targets"])
    if lengths["min"] == lengths["max"]:
        if minimum != maximum or len(minimum) != count:
            raise ValueError("Equal dataset residue bounds must include every target")
    elif minimum & maximum or len(minimum | maximum) > count:
        raise ValueError("Conflicting minimum and maximum dataset targets")
    return summary
