"""Top-level orchestration with input fingerprints and resumable stage records."""
from __future__ import annotations

import contextlib
import copy
import datetime as dt
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable

from .config import PipelineConfig, configure_environment
from .preflight import run_preflight
from .provenance import (
    fingerprint,
    hash_artifact,
    hash_paths,
    output_fingerprints,
    output_fingerprints_match,
    sha256_file,
    software_versions,
    write_manifest,
)
from .input_data import discover_structure_files
from .dataset_summary import summarize_dataset

try:
    import resource
except ImportError:  # Windows does not expose the Unix resource module.
    resource = None


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value: str) -> int:
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _peak_rss_bytes() -> int | None:
    if resource is None:
        return None
    try:
        values = [resource.getrusage(resource.RUSAGE_SELF).ru_maxrss]
        if hasattr(resource, "RUSAGE_CHILDREN"):
            values.append(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
        multiplier = 1 if sys.platform == "darwin" else 1024
        return int(max(values) * multiplier)
    except Exception:
        return None


def _sample_process_tree(stop: threading.Event, peak: list[int]) -> None:
    try:
        import psutil

        root = psutil.Process()
        while not stop.is_set():
            try:
                rss = root.memory_info().rss
                try:
                    children = root.children(recursive=True)
                except (psutil.Error, PermissionError):
                    # Some managed macOS environments restrict KERN_PROC_ALL;
                    # process RSS below is still available as a useful fallback.
                    children = []
                for child in children:
                    try:
                        rss += child.memory_info().rss
                    except (psutil.Error, PermissionError):
                        continue
                peak[0] = max(peak[0], rss)
            except (psutil.Error, PermissionError):
                pass
            stop.wait(0.2)
    except ImportError:
        return


def _require_stage_outputs(number: int, paths) -> list[Path]:
    """Require every declared output before recording a stage as completed."""
    declared = list(paths)
    if number != 3 and not declared:
        raise RuntimeError(f"Stage {number} declares no output artifacts")
    missing = [path for path in declared if not path.exists()]
    if missing:
        formatted = ", ".join(str(path) for path in missing)
        raise RuntimeError(
            f"Stage {number} did not produce every required artifact; missing: {formatted}"
        )
    return declared


def _stage_outputs_match(number: int, recorded: list[dict[str, Any]], base: Path) -> bool:
    """Check saved outputs, accounting for legacy stage-6 directory hashes.

    Earlier manifests fingerprinted the whole benchmark directory at stage 6;
    stage 7 later adds ``dataset_features.csv`` to that directory. Ignore only
    that stage-7 output when validating an earlier directory fingerprint.
    """
    if number == 4 and not any(
        item.get("path") in {"intermediate/04_cdr_exclusions.tsv",
                             "../outputs/abagbench/intermediate/04_cdr_exclusions.tsv"}
        for item in recorded
    ):
        # Older completed stages have no durable numbering-exclusion audit.
        # Recompute stage 4 to adopt the eligibility rule and output contract.
        return False
    if number == 6:
        legacy_dir = next((
            item for item in recorded
            if item.get("path") == "benchmark" and item.get("kind") == "directory"
        ), None)
        if legacy_dir is not None:
            directory = base / "benchmark"
            try:
                files, digest = hash_paths(
                    (
                        path for path in directory.rglob("*")
                        if path.is_file()
                        and path.relative_to(directory).as_posix() != "dataset_features.csv"
                    ),
                    directory,
                )
            except (FileNotFoundError, ValueError):
                return False
            current = {
                "path": "benchmark",
                "kind": "directory",
                "sha256": digest,
                "file_count": len(files),
                "size_bytes": sum(item["size_bytes"] for item in files),
            }
            if current != legacy_dir:
                return False
            return output_fingerprints_match(
                [item for item in recorded if item is not legacy_dir], base
            )
    return output_fingerprints_match(recorded, base)


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _manifest_safe_paths(value: Any, base: Path) -> Any:
    """Keep absolute machine paths out of saved stage result metadata."""
    if isinstance(value, dict):
        return {key: _manifest_safe_paths(item, base) for key, item in value.items()}
    if isinstance(value, list):
        return [_manifest_safe_paths(item, base) for item in value]
    if isinstance(value, str):
        path = Path(value)
        if path.is_absolute():
            try:
                return path.resolve().relative_to(base.resolve()).as_posix()
            except ValueError:
                return path.name
    return value


def _resolve_stage(value: str) -> int:
    names = {
        "release-screen": 1, "quality-filter": 2, "sequence-extraction": 3,
        "cdr-filter": 4, "structural-deleak": 5, "clustering": 6,
        "feature-assembly": 7,
    }
    if value.isdigit() and 1 <= int(value) <= 7:
        return int(value)
    if value in names:
        return names[value]
    raise ValueError(f"Unknown stage {value!r}; use 1–7 or one of {', '.join(names)}")


def _source_snapshot(config: PipelineConfig, include_cif_hashes: bool) -> tuple[dict[str, Any], str]:
    root = config.input.sabdab_dir
    if root is None:
        raise FileNotFoundError("SAbDab2 input is not configured")
    summary = root / config.input.summary_file
    structures = root / config.input.structures_dir
    summary_hash = sha256_file(summary)
    cif_paths = discover_structure_files(structures)
    if include_cif_hashes:
        file_records, structures_hash = hash_paths(cif_paths, root)
    else:
        file_records = [
            {"path": path.relative_to(root).as_posix(), "size_bytes": path.stat().st_size}
            for path in cif_paths
        ]
        structures_hash = fingerprint(file_records)
    snapshot = {
        "summary": {
            "path": summary.relative_to(root).as_posix(),
            "size_bytes": summary.stat().st_size,
            "sha256": summary_hash,
        },
        "structures": {
            "directory": structures.relative_to(root).as_posix(),
            "file_count": len(file_records),
            "size_bytes": sum(item["size_bytes"] for item in file_records),
            "sha256": structures_hash,
            "files": file_records if include_cif_hashes else None,
        },
    }
    return snapshot, fingerprint({
        "summary": summary_hash,
        "structures": structures_hash,
    })


def _write_dry_run(config: PipelineConfig, report: dict[str, Any]) -> None:
    print("Preflight passed. Dry run is non-destructive; no output directories or files were created.")
    print(json.dumps({
        "profile": config.output.run_name,
        "data_output": str(config.output.run_dir),
        "plots_output": str(config.output.plots_dir),
        "cutoff_date": config.selection.cutoff_date,
        "methods": list(config.selection.methods),
        "resolution_angstrom": [config.selection.resolution_min, config.selection.resolution_max],
        "xray_r_factor_max": config.selection.xray_r_factor_max,
        "exclude_membrane": config.selection.exclude_membrane,
        "residue_bounds_strict": [config.selection.min_residues, config.selection.max_residues],
        "representative_rule": config.representative_rule,
        "workers": config.resources.workers,
        "foldseek_threads": config.resources.foldseek_threads,
        "device": config.resources.device,
        "foldseek_binary": report.get("foldseek_binary"),
            "foldseek_version": report.get("foldseek_version"),
            "foldseek_chain_tm_threshold_supported": report.get("foldseek_chain_tm_threshold_supported"),
        "frozen_foldseek_protocol": config.as_dict()["foldseek"],
    }, indent=2))


def run_pipeline(
    config: PipelineConfig,
    *,
    force: bool = False,
    resume: bool = False,
    dry_run: bool = False,
    from_stage: str = "1",
    to_stage: str = "7",
) -> dict[str, Any]:
    """Run all or selected pipeline stages after read-only preflight checks."""
    config = config.resolved()
    start = _resolve_stage(from_stage)
    end = _resolve_stage(to_stage)
    if start > end:
        raise ValueError("--from-stage must be at or before --to-stage")
    if start > 1 and not resume:
        raise ValueError("Starting after stage 1 requires --resume and matching stage fingerprints")

    report = run_preflight(config, require_foldseek=True)
    if dry_run:
        _write_dry_run(config, report)
        return {"dry_run": True, "preflight": report}

    run_dir = config.output.run_dir
    manifest_path = run_dir / "run_manifest.json"
    if run_dir.exists() and any(run_dir.iterdir()) and not (force or resume):
        raise FileExistsError(f"Run output already exists: {run_dir}; use --resume or --force")
    if resume:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Cannot resume without a run manifest: {manifest_path}")
        with manifest_path.open() as handle:
            previous_manifest = json.load(handle)
        if previous_manifest.get("status") == "completed" and not force:
            print(f"This run is already complete: {run_dir}")
    else:
        previous_manifest = {}

    configure_environment(config)
    # Import stages after output/input paths and filter settings are resolved.
    from . import plotting
    from .paths import BENCHMARK_DIR, DATA_ROOT, INTERMEDIATE_ROOT, PLOTS_ROOT, SCRATCH_ROOT, LOGS_ROOT
    from .stages import cdr_filter, clustering, feature_assembly, quality_filter
    from .stages import release_screen, sequence_extraction, structural_deleak

    old_cfg = previous_manifest.get("resolved_config")
    resolved_cfg = config.as_dict()
    if resume and old_cfg is not None and old_cfg != resolved_cfg:
        raise ValueError("Cannot resume: resolved configuration differs from run_manifest.json; use a new --run-name or --force")

    input_snapshot, input_fp = _source_snapshot(config, include_cif_hashes=True)

    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir = LOGS_ROOT
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"run_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    previous_records = previous_manifest.get("stages", {}) if resume else {}
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_name": config.output.run_name,
        "status": "running",
        "started_at": previous_manifest.get("started_at", _utc_now()),
        "updated_at": _utc_now(),
        "completed_at": None,
        "resolved_config": resolved_cfg,
        "config_fingerprint": fingerprint(resolved_cfg),
        "input_snapshot": input_snapshot,
        "input_fingerprint": input_fp,
        "software": software_versions(report.get("foldseek_version")),
        "execution": {
            "canonical": config.resources.device == "gpu",
            "command": "python -m dataset_creation_pipeline run",
            "device": config.resources.device,
            "workers": config.resources.workers,
            "foldseek_threads": config.resources.foldseek_threads,
            "foldseek_binary": Path(report["foldseek_binary"]).name if report.get("foldseek_binary") else None,
            "host_platform": report.get("python_version"),
        },
        "notes": report.get("notes", []),
        "run_log": os.path.relpath(log_path, DATA_ROOT),
        "stages": copy.deepcopy(previous_records),
    }

    def persist() -> None:
        manifest["updated_at"] = _utc_now()
        write_manifest(manifest_path, manifest)

    def stage_fingerprint(number: int) -> tuple[str, list[Path]]:
        s1 = INTERMEDIATE_ROOT / "training_data_ids.json"
        s2 = INTERMEDIATE_ROOT / "02_quality_filtered.tsv"
        s4 = INTERMEDIATE_ROOT / "04_cdr_filtered.tsv"
        s5 = INTERMEDIATE_ROOT / "05_deleaked.tsv"
        if number == 1:
            inputs = {
                "summary_sha256": input_snapshot["summary"]["sha256"],
                "cutoff": config.selection.cutoff_date,
            }
            outputs = [s1, INTERMEDIATE_ROOT / "training_data_matches.txt"]
        elif number == 2:
            inputs = {
                "summary_sha256": input_snapshot["summary"]["sha256"],
                "stage1": hash_artifact(s1, DATA_ROOT),
                "selection": resolved_cfg["selection"],
            }
            outputs = [s2]
        elif number == 3:
            inputs = {"stage2": hash_artifact(s2, DATA_ROOT), "source_fingerprint": input_fp}
            outputs = []
        elif number == 4:
            inputs = {"stage2": hash_artifact(s2, DATA_ROOT), "source_fingerprint": input_fp, "cdr": resolved_cfg["cdr"]}
            outputs = [cdr_filter.FILTERED_TSV, cdr_filter.SCRATCH_SURVIVORS, INTERMEDIATE_ROOT / "04_cdr_exclusions.tsv"]
        elif number == 5:
            inputs = {"stage4": hash_artifact(cdr_filter.FILTERED_TSV, DATA_ROOT), "survivors": hash_artifact(cdr_filter.SCRATCH_SURVIVORS, DATA_ROOT), "training_ids": hash_artifact(s1, DATA_ROOT), "foldseek": report.get("foldseek_version")}
            outputs = [s5, structural_deleak.SCRATCH_DELEAKED]
        elif number == 6:
            inputs = {"stage5": hash_artifact(s5, DATA_ROOT), "deleaked": hash_artifact(structural_deleak.SCRATCH_DELEAKED, DATA_ROOT), "foldseek": report.get("foldseek_version"), "protocol": resolved_cfg["foldseek"], "resources": resolved_cfg["resources"]}
            inputs["representative_rule"] = config.representative_rule
            outputs = [
                BENCHMARK_DIR / "registry.tsv",
                BENCHMARK_DIR / "benchmark.fasta",
                BENCHMARK_DIR / "edges.tsv",
                BENCHMARK_DIR / "clusters.tsv",
                BENCHMARK_DIR / "representatives.tsv",
                BENCHMARK_DIR / "linkage.npy",
                BENCHMARK_DIR / "nodes.json",
                BENCHMARK_DIR / "leakage_summary.txt",
                BENCHMARK_DIR / "structures",
            ]
        elif number == 7:
            inputs = {"registry": hash_artifact(BENCHMARK_DIR / "registry.tsv", DATA_ROOT)}
            outputs = [BENCHMARK_DIR / "dataset_features.csv"]
        else:
            raise ValueError(number)
        return fingerprint(inputs), outputs

    def prior_stage_matches(number: int) -> bool:
        if number == 3:
            # Sequence extraction is intentionally recomputed in memory.
            return True
        try:
            stage_fp, _ = stage_fingerprint(number)
        except (FileNotFoundError, ValueError):
            return False
        record = previous_records.get(str(number), {})
        return (
            record.get("status") == "completed"
            and record.get("fingerprint") == stage_fp
            and _stage_outputs_match(number, record.get("outputs", []), DATA_ROOT)
        )

    if resume:
        for number in range(1, start):
            if not prior_stage_matches(number):
                raise ValueError(f"Cannot start from stage {start}: stage {number} artifacts or fingerprint do not match this run")

    stage_calls: dict[int, tuple[str, Callable[[], dict[str, Any]], Callable[[], list[Path]]]] = {
        1: ("release screen", lambda: release_screen.run(cutoff=config.selection.cutoff_date), lambda: [release_screen.IDS_OUTPUT, release_screen.MATCHES_OUTPUT]),
        2: ("quality filter", lambda: quality_filter.run(force=True, lower=config.selection.min_residues, upper=config.selection.max_residues), lambda: [quality_filter.FILTERED_TSV, quality_filter.METRICS_OUTPUT]),
        4: ("CDR completeness", lambda: cdr_filter.run(processes=config.resources.workers, fastas=meta_s3["fastas"]), lambda: [cdr_filter.FILTERED_TSV, cdr_filter.SCRATCH_SURVIVORS, INTERMEDIATE_ROOT / "04_cdr_metrics.json", INTERMEDIATE_ROOT / "04_cdr_exclusions.tsv"]),
        5: ("FoldSeek structural de-leak", lambda: structural_deleak.run(processes=config.resources.workers, threads=config.resources.foldseek_threads, gpu=int(config.resources.device == "gpu"), foldseek_bin=report.get("foldseek_binary")), lambda: [structural_deleak.FILTERED_TSV, structural_deleak.SCRATCH_DELEAKED, INTERMEDIATE_ROOT / "05_deleak_metrics.json"]),
        6: ("FoldSeek clustering", lambda: clustering.run(processes=config.resources.workers, threads=config.resources.foldseek_threads, gpu=int(config.resources.device == "gpu"), foldseek_bin=report.get("foldseek_binary"), fastas=meta_s3["fastas"], smallest_repr=config.selection.smallest_repr), lambda: [
            clustering.FINAL_BENCHMARK_DIR / "registry.tsv",
            clustering.FINAL_BENCHMARK_DIR / "benchmark.fasta",
            clustering.FINAL_BENCHMARK_DIR / "edges.tsv",
            clustering.FINAL_BENCHMARK_DIR / "clusters.tsv",
            clustering.FINAL_BENCHMARK_DIR / "representatives.tsv",
            clustering.FINAL_BENCHMARK_DIR / "linkage.npy",
            clustering.FINAL_BENCHMARK_DIR / "nodes.json",
            clustering.FINAL_BENCHMARK_DIR / "leakage_summary.txt",
            clustering.FINAL_BENCHMARK_DIR / "structures",
            INTERMEDIATE_ROOT / "06_clustering_metrics.json",
        ]),
        7: ("feature assembly", lambda: feature_assembly.run(), lambda: [feature_assembly.OUTPUT_CSV]),
    }

    def invoke_stage(number: int, title: str, call: Callable[[], dict[str, Any]], output_paths: list[Path]):
        stage_fp, expected_paths = stage_fingerprint(number)
        record = previous_records.get(str(number), {}) if resume else {}
        if number != 3 and resume and record.get("fingerprint") == stage_fp and record.get("status") == "completed" and _stage_outputs_match(number, record.get("outputs", []), DATA_ROOT):
            print(f"[RESUME] stage {number} ({title}) matches saved inputs; reusing artifacts")
            return copy.deepcopy(record.get("result", {}))
        if resume and record and record.get("fingerprint") != stage_fp:
            raise ValueError(f"Cannot resume stage {number}: its configuration or input fingerprint changed")
        stage_record = {"status": "running", "fingerprint": stage_fp, "started_at": _utc_now()}
        manifest["stages"][str(number)] = stage_record
        persist()
        rss_before = _peak_rss_bytes()
        rss_peak = [0]
        sampler_stop = threading.Event()
        sampler = threading.Thread(target=_sample_process_tree, args=(sampler_stop, rss_peak), daemon=True)
        sampler.start()
        started = time.perf_counter()
        try:
            result = call()
        finally:
            sampler_stop.set()
            sampler.join()
        duration = time.perf_counter() - started
        rss_fallback = _peak_rss_bytes() or 0
        actual_paths = _require_stage_outputs(number, output_paths)
        stage_record.update({
            "status": "completed",
            "completed_at": _utc_now(),
            "duration_seconds": round(duration, 3),
            "peak_rss_bytes": max(rss_peak[0], rss_fallback) or None,
            "peak_rss_measurement": "sampled_root_and_resource_child_high_water" if rss_peak[0] else "resource_self_and_child_high_water",
            "process_peak_rss_before_bytes": rss_before,
            "result": _manifest_safe_paths(
                _json_safe({key: value for key, value in result.items() if key != "fastas"}),
                DATA_ROOT,
            ),
            "outputs": output_fingerprints(actual_paths, DATA_ROOT),
        })
        persist()
        print(f"[TIMING] stage {number} {title}: {duration:.2f}s; peak RSS {_peak_rss_bytes()}")
        return result

    try:
        with log_path.open("a", buffering=1) as log_handle:
            tee_out = _Tee(sys.stdout, log_handle)
            tee_err = _Tee(sys.stderr, log_handle)
            with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
                print(f"Pipeline run: {config.output.run_name}")
                print(f"Started: {_utc_now()}")
                print(f"SAbDab2 summary SHA-256: {input_snapshot['summary']['sha256']}")
                print(f"SAbDab2 structure files fingerprint: {input_snapshot['structures']['sha256']}")
                persist()
                for number in range(1, 8):
                    if number < start or number > end:
                        continue
                    if number == 3:
                        title = "sequence extraction"
                        result = invoke_stage(
                            3, title, sequence_extraction.run,
                            [],
                        )
                        meta_s3 = result
                        continue
                    if number in {4, 6} and "meta_s3" not in locals():
                        meta_s3 = sequence_extraction.run()
                    title, call, paths_factory = stage_calls[number]
                    result = invoke_stage(number, title, call, paths_factory())
                    if number == 4:
                        meta_s4 = result
                    elif number == 5:
                        meta_s5 = result
                    elif number == 6:
                        meta_s6 = result

                if end == 7:
                    manifest["dataset_summary"] = summarize_dataset(run_dir)
                    plots_dir = PLOTS_ROOT
                    plot_outputs = [plots_dir]
                    plots_fp = fingerprint({
                        "registry": hash_artifact(BENCHMARK_DIR / "registry.tsv", DATA_ROOT),
                        "stage_metrics": hash_artifact(INTERMEDIATE_ROOT / "stage_metrics.json", DATA_ROOT),
                    })
                    previous_plot = previous_records.get("plots", {}) if resume else {}
                    if not (resume and previous_plot.get("fingerprint") == plots_fp and previous_plot.get("status") == "completed" and output_fingerprints_match(previous_plot.get("outputs", []), DATA_ROOT)):
                        plotting.generate_all()
                        manifest["stages"]["plots"] = {
                            "status": "completed",
                            "fingerprint": plots_fp,
                            "completed_at": _utc_now(),
                            "outputs": output_fingerprints(plot_outputs, DATA_ROOT),
                        }
                        persist()
                print(f"Finished: {_utc_now()}")
        manifest["status"] = "completed" if end == 7 else "incomplete"
        manifest["completed_at"] = _utc_now() if end == 7 else None
        persist()
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        persist()
        raise

    return {
        "status": manifest["status"],
        "manifest": str(manifest_path),
        "log": str(log_path),
        "run_dir": str(run_dir),
    }
