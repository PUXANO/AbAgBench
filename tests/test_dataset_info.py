"""Inspect saved provenance independently of current defaults and runtime tools."""
import hashlib
import json
import subprocess
import sys

import pytest

from dataset_creation_pipeline import cli
from dataset_creation_pipeline.dataset_info import dataset_info, format_info
from dataset_creation_pipeline.dataset_summary import summarize_dataset
from dataset_creation_pipeline.subset import export_subset
from subset_support import make_release


def record_settings(root):
    path = root / "run_manifest.json"
    manifest = json.loads(path.read_text())
    manifest.update({
        "run_name": "recorded-baseline",
        "resolved_config": {
            "input": {"sabdab_dir": "snapshot-2026-10-13"},
            "selection": {"cutoff_date": "2021-09-30", "smallest_repr": True,
                          "min_residues": 350, "max_residues": 3500,
                          "resolution_min": 1.0, "resolution_max": 3.5},
            "foldseek": {"representative_rule": "resolution, R-factor, complex ID"},
            "resources": {"device": "gpu"},
        },
        "software": {"foldseek": "recorded-build"},
    })
    manifest["stages"]["6"]["result"] = {
        "n_representatives": 6,
        "representative_rule": "whole-complex residue count, resolution, R-factor, complex ID",
    }
    manifest["stages"]["7"]["result"] = {"total": 6}
    manifest["stages"]["4"] = {"result": {
        "numbering_eligibility": {"max_cdrh3_length": 36, "length_scheme": "imgt"},
        "metrics": [{"stage": "CDR coverage filter (Chothia 100%)"}],
    }}
    path.write_text(json.dumps(manifest))
    return manifest


@pytest.mark.parametrize("location", ["experiment", "inputs", "manifest"])
def test_full_info_uses_saved_settings_and_leaves_dataset_unchanged(tmp_path, monkeypatch, capsys, location):
    root = make_release(tmp_path / "experiment/inputs")
    record_settings(root)
    before = {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}
    def unexpected_config(*args):
        pytest.fail("dataset-info must not load current configuration defaults")
    monkeypatch.setattr(cli, "load_config", unexpected_config)
    path = {"experiment": root.parent, "inputs": root, "manifest": root / "run_manifest.json"}[location]
    assert cli.main(["dataset-info", str(path), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    generation = report["generation"]
    assert report["complexes_recorded"] == 6
    assert report["verification"] == {"status": "not_requested"}
    assert report["dataset_summary"] == {
        "complexes": 6,
        "residue_length": {"min": 3, "max": 10,
                           "min_targets": ["1abc_HA"], "max_targets": ["6abc_HLA"]},
    }
    assert report["dataset_summary_source"] == "dataset_files"
    assert report["manifest_sha256"] == hashlib.sha256((root / "run_manifest.json").read_bytes()).hexdigest()
    assert generation["resolved_config"]["selection"]["max_residues"] == 3500
    assert generation["effective_representative_rule"].startswith("whole-complex residue count,")
    assert generation["cdr_numbering_eligibility"]["max_cdrh3_length"] == 36
    assert "100%" in generation["cdr_coverage_rule"]
    text = format_info(report)
    assert "max_residues (exclusive): 3500" in text
    assert "resolution_max (Å, exclusive): 3.5" in text
    assert "complexes: 6" in text
    assert "whole-complex AA range (observed, inclusive): 3–10" in text
    assert "CDR\n  settings: not recorded" in text
    assert "artifact verification" not in text
    assert {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()} == before


def test_subset_info_reads_retained_source_after_original_is_removed(tmp_path):
    source = make_release(tmp_path / "source")
    record_settings(source)
    output = tmp_path / "subset"
    export_subset(source, output, complex_type="nanobody", max_residues=7, count=2, seed=17)
    source.rename(tmp_path / "original-moved")
    report = dataset_info(output, verify=True)
    assert report["release_kind"] == "subset"
    assert report["canonical"] is False
    assert report["generation"]["canonical"] is True
    assert report["complexes_recorded"] == 2
    assert report["verification"]["complexes"] == 2
    assert report["generation"]["resolved_config"]["selection"]["max_residues"] == 3500
    assert report["subset_selection"]["max_residues"] == 7
    assert report["subset_selection"]["seed"] == 17
    assert report["subset_export_software"]["abagbench"]
    assert report["dataset_summary_source"] == "manifest"
    assert report["dataset_summary"]["complexes"] == 2
    assert report["dataset_summary"]["residue_length"]["max"] <= 7
    assert "Subset export criteria (max_residues is inclusive)" in format_info(report)
    parent = output / "inputs/provenance/source/run_manifest.json"
    parent.write_text(parent.read_text() + "\n")
    with pytest.raises(ValueError, match="Source release manifest hash differs"):
        dataset_info(output)


def test_verify_checks_artifacts_and_metadata_count(tmp_path, capsys):
    root = make_release(tmp_path / "inputs")
    record_settings(root)
    assert cli.main(["dataset-info", str(root), "--verify"]) == 0
    assert "artifact verification: passed" in capsys.readouterr().out
    path = root / "run_manifest.json"
    manifest = json.loads(path.read_text())
    for stage, key in (("6", "n_representatives"), ("7", "total")):
        manifest["stages"][stage]["result"][key] = 7
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Recorded dataset size differs"):
        dataset_info(root, verify=True)
    record_settings(root)
    (root / "benchmark/structures/1abc_HA.cif").write_text("modified")
    assert dataset_info(root)["verification"]["status"] == "not_requested"
    assert cli.main(["dataset-info", str(root), "--verify", "--json"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "differs from its manifest" in captured.err


def test_incomplete_and_sparse_manifests_are_reported_without_inventing_defaults(tmp_path):
    path = tmp_path / "run_manifest.json"
    path.write_text(json.dumps({"schema_version": 1, "status": "incomplete"}))
    report = dataset_info(tmp_path)
    assert report["status"] == "incomplete"
    assert report["complexes_recorded"] is None
    assert report["generation"]["resolved_config"] == {}
    assert report["generation"]["effective_representative_rule"] is None
    assert "not recorded" in format_info(report)
    with pytest.raises(ValueError, match="not complete"):
        dataset_info(tmp_path, verify=True)


@pytest.mark.parametrize("problem,message", [
    ("representative", "representative rule conflicts"),
    ("count", "dataset sizes differ"),
    ("selection", "must be an object"),
    ("schema", "Unsupported.*schema"),
    ("kind", "Unsupported.*kind"),
])
def test_inconsistent_or_malformed_metadata_fails(tmp_path, problem, message):
    root = make_release(tmp_path / "inputs")
    manifest = record_settings(root)
    if problem == "representative":
        manifest["stages"]["6"]["result"]["representative_rule"] = "resolution, R-factor, complex ID"
    elif problem == "count":
        manifest["stages"]["7"]["result"]["total"] = 7
    elif problem == "selection":
        manifest["resolved_config"]["selection"] = []
    elif problem == "schema":
        manifest["schema_version"] = 2
    else:
        manifest["release_kind"] = "unknown"
    (root / "run_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=message):
        dataset_info(root)


@pytest.mark.parametrize("body", ['{"schema_version": 1, "schema_version": 1}',
                                  '{"schema_version": 1, "resolved_config": NaN}', '[]', '{'])
def test_bad_json_fails_cleanly(tmp_path, capsys, body):
    (tmp_path / "run_manifest.json").write_text(body)
    assert cli.main(["dataset-info", str(tmp_path)]) == 2
    captured = capsys.readouterr()
    assert not captured.out
    assert captured.err.startswith("error:")


def test_missing_or_ambiguous_manifest_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="No run_manifest"):
        dataset_info(tmp_path)
    (tmp_path / "inputs").mkdir()
    for path in (tmp_path / "run_manifest.json", tmp_path / "inputs/run_manifest.json"):
        path.write_text('{"schema_version": 1}')
    with pytest.raises(ValueError, match="Ambiguous"):
        dataset_info(tmp_path)


def test_cli_needs_only_standard_library_and_uses_no_preflight(tmp_path):
    (tmp_path / "run_manifest.json").write_text('{"schema_version": 1, "status": "failed"}')
    result = subprocess.run([sys.executable, "-S", "-m", "dataset_creation_pipeline",
                             "dataset-info", str(tmp_path), "--json"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "failed"


def test_recorded_summary_supports_manifest_only_inspection_and_verifies_contents(tmp_path):
    root = make_release(tmp_path / "inputs")
    manifest = record_settings(root)
    manifest["dataset_summary"] = summarize_dataset(root)
    path = root / "run_manifest.json"
    path.write_text(json.dumps(manifest))
    assert dataset_info(root, verify=True)["dataset_summary_source"] == "manifest"
    # Well-formed but stale bounds must fail artifact verification.
    manifest["dataset_summary"]["residue_length"]["max"] = 9
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="summary differs from verified artifacts"):
        dataset_info(root, verify=True)
    (root / "benchmark").rename(tmp_path / "saved-benchmark")
    assert dataset_info(root)["dataset_summary"]["residue_length"]["max"] == 9
    del manifest["dataset_summary"]
    path.write_text(json.dumps(manifest))
    assert dataset_info(root)["dataset_summary"] is None
    assert "observed, inclusive): not recorded" in format_info(dataset_info(root))


@pytest.mark.parametrize("value", [0, True, 1.5, "10", None])
def test_malformed_recorded_bounds_fail(tmp_path, value):
    root = make_release(tmp_path / "inputs")
    manifest = record_settings(root)
    manifest["dataset_summary"] = summarize_dataset(root)
    manifest["dataset_summary"]["residue_length"]["max"] = value
    (root / "run_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="residue bounds must be positive integers"):
        dataset_info(root)


@pytest.mark.parametrize("value", ["0", "nan", "inf", "3.5", "", "4"])
def test_legacy_summary_rejects_invalid_or_inconsistent_feature_lengths(tmp_path, value):
    root = make_release(tmp_path / "inputs")
    path = root / "benchmark/dataset_features.csv"
    path.write_text(path.read_text().replace("True,3,1", f"True,{value},1"))
    with pytest.raises(ValueError, match="residue_length"):
        dataset_info(root)


def test_observed_bounds_retain_all_tied_targets(tmp_path):
    root = make_release(tmp_path / "inputs")
    feature = root / "benchmark/dataset_features.csv"
    feature.write_text(feature.read_text().replace("3abc_HA,3abc,X-ray,True,5,1",
                                                "3abc_HA,3abc,X-ray,True,3,1"))
    fasta = root / "benchmark/benchmark.fasta"
    fasta.write_text(fasta.read_text().replace(">3abc_HA_A|protein\nGGG\n", ">3abc_HA_A|protein\nG\n"))
    report = dataset_info(root)
    assert report["dataset_summary"]["residue_length"]["min_targets"] == ["1abc_HA", "3abc_HA"]
    assert "smallest: 1abc_HA, 3abc_HA" in format_info(report)


def test_default_output_shows_parameters_and_contents_while_json_keeps_provenance(tmp_path, capsys):
    files = [{"path": f"structures/source_{i}.cif", "sha256": "a" * 64} for i in range(11664)]
    snapshot = {"structures": {"file_count": len(files), "sha256": "b" * 64, "files": files}}
    root = make_release(tmp_path / "inputs")
    manifest = record_settings(root)
    manifest.update({"input_snapshot": snapshot, "started_at": "2026-10-14T02:31:54Z"})
    manifest["resolved_config"]["output"] = {"data_dir": "irrelevant-output-path"}
    manifest["resolved_config"]["foldseek"]["foldseek_format"] = "query,target,interfacelddt"
    (root / "run_manifest.json").write_text(json.dumps(manifest))
    assert cli.main(["dataset-info", str(root)]) == 0
    text = capsys.readouterr().out
    assert "max_residues (exclusive): 3500" in text
    assert "cutoff_date: 2021-09-30" in text
    assert "representative_rule: whole-complex residue count, resolution, R-factor, complex ID" in text
    for detail in ("structures/source_", "manifest", "sha256", "input_snapshot", "files:",
                   "started_at", "software", "recorded-build", "irrelevant-output-path", "foldseek_format"):
        assert detail not in text
    assert len(text.splitlines()) < 40
    assert cli.main(["dataset-info", str(root), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["generation"]["input_snapshot"] == snapshot
