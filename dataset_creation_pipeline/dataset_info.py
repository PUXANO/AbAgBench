"""Describe full/subset settings and observed contents using the standard library."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .release import RELEASE_KINDS, validate_release
from .dataset_summary import check_summary, summarize_dataset


def _object(parent: dict, key: str) -> dict:
    value = parent.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"Manifest field {key!r} must be an object")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate manifest field: {key}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON constant in manifest: {value}")


def _read_manifest(path: Path) -> tuple[dict, str]:
    body = path.read_bytes()
    manifest = json.loads(body, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    if (not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int
            or manifest["schema_version"] != 1):
        raise ValueError(f"Unsupported AbAgBench manifest schema: {path}")
    if manifest.get("release_kind", "full") not in RELEASE_KINDS:
        raise ValueError(f"Unsupported AbAgBench release kind: {path}")
    return manifest, hashlib.sha256(body).hexdigest()


def _manifest_path(dataset: Path) -> Path:
    path = dataset.expanduser().resolve()
    if path.is_file():
        if path.name != "run_manifest.json":
            raise ValueError("Specify a dataset directory, experiment directory, or run_manifest.json")
        return path
    candidates = [p for p in (path / "run_manifest.json", path / "inputs/run_manifest.json") if p.is_file()]
    if len(candidates) > 1:
        raise ValueError(f"Ambiguous dataset path; specify the intended run_manifest.json: {path}")
    if not candidates:
        raise FileNotFoundError(f"No run_manifest.json in dataset or experiment: {path}")
    return candidates[0]


def _stage_result(manifest: dict, number: str) -> dict:
    return _object(_object(_object(manifest, "stages"), number), "result")


def _representative_rule(config: dict, manifest: dict) -> str | None:
    selection = _object(config, "selection")
    foldseek = _object(config, "foldseek")
    smallest = selection.get("smallest_repr")
    if smallest is not None and type(smallest) is not bool:
        raise ValueError("selection.smallest_repr must be a boolean")
    base = foldseek.get("representative_rule")
    if base is not None and (not isinstance(base, str) or not base):
        raise ValueError("foldseek.representative_rule must be a nonempty string")
    expected = None
    if base is not None and smallest is not None:
        expected = ("whole-complex residue count, " + base) if smallest else base
    recorded = _stage_result(manifest, "6").get("representative_rule")
    if recorded is not None and (not isinstance(recorded, str) or not recorded):
        raise ValueError("Stage-6 representative_rule must be a nonempty string")
    if recorded is not None and expected is not None and recorded != expected:
        raise ValueError("Stage-6 representative rule conflicts with the recorded configuration")
    return recorded if recorded is not None else expected


def _recorded_count(manifest: dict) -> int | None:
    if manifest.get("release_kind", "full") in {"subset", "max-size-test"}:
        targets = manifest.get("selected_targets")
        if targets is None:
            return None
        if (not isinstance(targets, list) or any(not isinstance(t, str) or not t for t in targets)
                or len(set(targets)) != len(targets)):
            raise ValueError("selected_targets must contain unique, nonempty complex IDs")
        return len(targets)
    total = _stage_result(manifest, "7").get("total")
    representatives = _stage_result(manifest, "6").get("n_representatives")
    for value in (total, representatives):
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError("Recorded dataset size must be a nonnegative integer")
    if total is not None and representatives is not None and total != representatives:
        raise ValueError("Recorded stage-6 and stage-7 dataset sizes differ")
    return total if total is not None else representatives


def dataset_info(dataset: Path, *, verify: bool = False) -> dict[str, Any]:
    """Describe saved settings and final contents; artifact verification is opt-in.

    Missing metadata stays absent/null rather than being filled from current
    configuration defaults. Subset generation settings come from its hashed,
    retained source manifest, without requiring the original dataset location.
    Legacy completed releases without a saved summary are summarized from their
    final metadata and sequences when available, without editing the manifest.
    """
    path = _manifest_path(dataset)
    manifest, digest = _read_manifest(path)
    kind = manifest.get("release_kind", "full")
    generation, generation_digest = manifest, digest
    source_path = None
    if kind == "subset":
        source_path = path.parent / "provenance/source/run_manifest.json"
        generation, generation_digest = _read_manifest(source_path)
        if generation_digest != _object(manifest, "source").get("manifest_sha256"):
            raise ValueError("Source release manifest hash differs")
        if generation.get("release_kind", "full") != "full":
            raise ValueError("Subset source manifest must describe a full release")
    config = _object(generation, "resolved_config")
    for section in ("input", "output", "selection", "cdr", "resources", "foldseek"):
        _object(config, section)
    cdr = _stage_result(generation, "4")
    metrics = cdr.get("metrics", [])
    if not isinstance(metrics, list) or any(not isinstance(item, dict) for item in metrics):
        raise ValueError("Stage-4 metrics must be a list of objects")
    coverage_rules = [item["stage"] for item in metrics
                      if isinstance(item.get("stage"), str) and item["stage"].startswith("CDR coverage filter")]
    verification = {"status": "not_requested"}
    count = _recorded_count(manifest)
    if verify:
        verification = {"status": "passed", **validate_release(path.parent, kind)}
        if verification["manifest_sha256"] != digest:
            raise ValueError("Dataset manifest changed during verification")
        if count is not None and count != verification["complexes"]:
            raise ValueError("Recorded dataset size differs from verified artifacts")
    summary = None
    summary_source = None
    if "dataset_summary" in manifest:
        summary = check_summary(manifest["dataset_summary"])
        summary_source = "manifest"
    elif manifest.get("status") == "completed" and (path.parent / "benchmark").exists():
        summary = summarize_dataset(path.parent)
        summary_source = "dataset_files"
    if summary is not None:
        if count is not None and count != summary["complexes"]:
            raise ValueError("Recorded dataset size differs from dataset summary")
        if verify and summary_source == "manifest" and summary != summarize_dataset(path.parent):
            raise ValueError("Recorded dataset summary differs from verified artifacts")
    return {
        "manifest": str(path),
        "manifest_sha256": digest,
        "schema_version": manifest["schema_version"],
        "release_kind": kind,
        "run_name": manifest.get("run_name"),
        "status": manifest.get("status"),
        "started_at": manifest.get("started_at"),
        "completed_at": manifest.get("completed_at"),
        "created_at": manifest.get("created_at"),
        "canonical": _object(manifest, "execution").get("canonical"),
        "complexes_recorded": count,
        "dataset_summary": summary,
        "dataset_summary_source": summary_source,
        "verification": verification,
        "generation": {
            "manifest_sha256": generation_digest,
            "run_name": generation.get("run_name"),
            "status": generation.get("status"),
            "canonical": _object(generation, "execution").get("canonical"),
            "resolved_config": config,
            "effective_representative_rule": _representative_rule(config, generation),
            "cdr_coverage_rule": "; ".join(coverage_rules) or None,
            "cdr_numbering_eligibility": cdr.get("numbering_eligibility"),
            "input_snapshot": generation.get("input_snapshot"),
            "input_fingerprint": generation.get("input_fingerprint"),
            "config_fingerprint": generation.get("config_fingerprint"),
            "software": generation.get("software"),
            "execution": generation.get("execution"),
        },
        "source_manifest": str(source_path) if source_path else None,
        "subset_selection": _object(manifest, "selection") if kind == "subset" else None,
        "subset_export_software": manifest.get("software") if kind == "subset" else None,
        "selected_targets": manifest.get("selected_targets") if kind in {"subset", "max-size-test"} else None,
        "capacity_selection": _object(manifest, "selection") if kind == "max-size-test" else None,
    }


def format_info(report: dict) -> str:
    """Show generation parameters and observed contents; provenance stays in JSON."""
    capacity = report["release_kind"] == "max-size-test"
    lines = ["Machine capacity-test dataset" if capacity else "Dataset generation parameters"]

    def value_text(value: Any) -> str:
        if value is None or value == {}:
            return "not recorded"
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False)

    def row(label: str, value: Any) -> None:
        lines.append(f"  {label}: {value_text(value)}")

    generation = report["generation"]
    config = generation["resolved_config"]
    if capacity:
        row("release_kind", "max-size-test")
        row("purpose", "machine_capacity_test")
        lines.append("Capacity selection (whole-complex AA bounds inclusive)")
        for key, value in report["capacity_selection"].items():
            row(key, "not set" if value is None else value)
        lines.extend(_contents_lines(report))
        if report["verification"]["status"] == "passed":
            row("artifact verification", "passed")
        return "\n".join(lines)
    row("SAbDab source", config.get("input", {}).get("sabdab_dir"))
    units = {"resolution_min": " (Å, inclusive)", "resolution_max": " (Å, exclusive)",
             "xray_r_factor_max": " (exclusive)", "min_residues": " (exclusive)",
             "max_residues": " (exclusive)", "interface_distance_angstrom": " (Å)"}
    for name, title in (("selection", "Selection"), ("cdr", "CDR"),
                        ("foldseek", "Structural filtering and clustering")):
        lines.append(title)
        values = config.get(name, {})
        for key, value in values.items():
            # Show the effective rule once, rather than its default tie-break.
            if name == "foldseek" and key in {"representative_rule", "foldseek_format"}:
                continue
            row(key + units.get(key, ""), value)
        if not values:
            row("settings", None)
        if name == "cdr":
            row("CDR coverage", generation["cdr_coverage_rule"])
            eligibility = generation["cdr_numbering_eligibility"]
            row("CDR-H3 eligibility", {key: eligibility[key] for key in (
                "max_cdrh3_length", "length_scheme", "length_range") if key in eligibility}
                if isinstance(eligibility, dict) else None)
        elif name == "foldseek":
            row("representative_rule", generation["effective_representative_rule"])
    resources = config.get("resources", {})
    row("device", resources.get("device"))
    row("workers / Foldseek threads", f"{value_text(resources.get('workers'))} / "
        f"{value_text(resources.get('foldseek_threads'))}")
    if report["source_manifest"]:
        lines.append("Subset export criteria (max_residues is inclusive)")
        for key, value in report["subset_selection"].items():
            if key == "ids" and isinstance(value, list) and len(value) > 10:
                value = f"{len(value)} IDs (use --json for the list)"
            row(key, "not set" if value is None else value)
    lines.extend(_contents_lines(report))
    if report["verification"]["status"] == "passed":
        row("artifact verification", "passed")
    return "\n".join(lines)


def _contents_lines(report: dict) -> list[str]:
    lines = ["Dataset contents (final exported complexes)"]
    def row(label, value):
        lines.append(f"  {label}: {value if value is not None else 'not recorded'}")
    summary = report["dataset_summary"]
    row("complexes", summary["complexes"] if summary else report["complexes_recorded"])
    if summary:
        lengths = summary["residue_length"]
        row("whole-complex AA range (observed, inclusive)", f"{lengths['min']}–{lengths['max']}")
        for bound, label in (("min", "smallest"), ("max", "largest")):
            targets = lengths[f"{bound}_targets"]
            row(label, ", ".join(targets) if len(targets) <= 10 else
                f"{len(targets)} complexes (use --json for the list)")
    else:
        row("whole-complex AA range (observed, inclusive)", None)
    return lines
