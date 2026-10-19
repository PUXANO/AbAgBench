"""Read recorded full/subset dataset provenance using only the standard library."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .release import validate_release


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
    if manifest.get("release_kind", "full") not in {"full", "subset"}:
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
    if manifest.get("release_kind", "full") == "subset":
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
    """Describe saved settings; full artifact verification is explicitly opt-in.

    Missing metadata stays absent/null rather than being filled from current
    configuration defaults. Subset generation settings come from its hashed,
    retained source manifest, without requiring the original dataset location.
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
        "selected_targets": manifest.get("selected_targets") if kind == "subset" else None,
    }


def format_info(report: dict) -> str:
    """Format saved configuration with bounds/units and explicit missing values."""
    lines = ["AbAgBench dataset info"]

    def value_text(value: Any) -> str:
        if value is None or value == {}:
            return "not recorded"
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False)

    def row(label: str, value: Any) -> None:
        lines.append(f"  {label}: {value_text(value)}")

    def section(title: str, values: Any) -> None:
        lines.append(f"\n{title}")
        if isinstance(values, dict) and values:
            for key, value in values.items():
                if isinstance(value, dict) and value:
                    section(f"{title}.{key}", value)
                else:
                    row(key, value)
        else:
            row("value", values)

    for key in ("manifest", "manifest_sha256", "release_kind", "run_name", "status",
                "started_at", "completed_at", "created_at", "canonical", "complexes_recorded"):
        row(key, report[key])
    row("artifact verification", "passed" if report["verification"]["status"] == "passed" else "not requested")
    if report["source_manifest"]:
        row("source_manifest", report["source_manifest"])
        section("Subset export criteria (max_residues is inclusive)", report["subset_selection"])
        row("selected_targets", report["selected_targets"])
        section("Subset export software", report["subset_export_software"])
    generation = report["generation"]
    section("Generation summary", {key: generation[key] for key in (
        "manifest_sha256", "run_name", "status", "canonical", "effective_representative_rule",
        "cdr_coverage_rule", "cdr_numbering_eligibility")})
    config = generation["resolved_config"]
    units = {"resolution_min": " (Å, inclusive)", "resolution_max": " (Å, exclusive)",
             "xray_r_factor_max": " (exclusive)", "min_residues": " (exclusive)",
             "max_residues": " (exclusive)"}
    for name in ("input", "selection", "cdr", "foldseek", "resources", "output"):
        values = config.get(name)
        if name == "selection" and isinstance(values, dict):
            values = {key + units.get(key, ""): value for key, value in values.items()}
        section(f"Recorded configuration: {name}", values)
    for name in sorted(config.keys() - {"input", "selection", "cdr", "foldseek", "resources", "output"}):
        section(f"Recorded configuration: {name}", config[name])
    for key in ("input_snapshot", "input_fingerprint", "config_fingerprint", "software", "execution"):
        section(key, generation[key])
    return "\n".join(lines)
