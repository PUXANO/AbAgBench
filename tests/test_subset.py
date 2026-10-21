import json
from pathlib import Path

import pytest

from dataset_creation_pipeline.cli import main
from dataset_creation_pipeline.release import read_fasta, read_table, validate_release
from dataset_creation_pipeline.subset import artifact_record, export_subset
from subset_support import make_release, refresh_manifest


def snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_explicit_subset_preserves_source_and_exports_only_selected_rows(tmp_path):
    source = make_release(tmp_path / "source")
    original = snapshot(source)
    output = tmp_path / "subset"
    result = export_subset(source, output, ids=["2abc_HLA", "1abc_HA"])
    assert result["complexes"] == 2
    assert snapshot(source) == original
    for name, delimiter in (("registry.tsv", "\t"), ("dataset_features.csv", ",")):
        assert list(read_table(output / "inputs/benchmark" / name, delimiter)[1]) == ["1abc_HA", "2abc_HLA"]
    assert set(read_fasta(output / "inputs/benchmark/benchmark.fasta")) == {"1abc_HA", "2abc_HLA"}
    assert sorted(p.stem for p in (output / "inputs/benchmark/structures").iterdir()) == ["1abc_HA", "2abc_HLA"]
    manifest = json.loads((output / "inputs/run_manifest.json").read_text())
    assert "stages" not in manifest
    assert manifest["execution"]["canonical"] is False
    assert manifest["selection"]["ids"] == ["2abc_HLA", "1abc_HA"]
    # Runtime validation is portable and does not consult the original release.
    source.rename(tmp_path / "moved-source")
    assert validate_release(output / "inputs", "subset")["complexes"] == 2


def test_filters_then_reproducible_sampling_and_inclusive_size(tmp_path):
    source = make_release(tmp_path / "source")
    for name in ("a", "b"):
        export_subset(source, tmp_path / name, complex_type="nanobody", max_residues=7, count=2, seed=17)
    manifests = [json.loads((tmp_path / n / "inputs/run_manifest.json").read_text()) for n in ("a", "b")]
    assert manifests[0]["selected_targets"] == manifests[1]["selected_targets"]
    assert len(manifests[0]["selected_targets"]) == 2
    assert set(manifests[0]["selected_targets"]) <= {"1abc_HA", "3abc_HA", "5abc_HA"}
    export_subset(source, tmp_path / "antibodies", complex_type="antibody", max_residues=8)
    assert json.loads((tmp_path / "antibodies/inputs/run_manifest.json").read_text())["selected_targets"] == ["2abc_HLA", "4abc_HLA"]


def test_sampling_is_independent_of_requested_id_order(tmp_path):
    source = make_release(tmp_path / "source")
    ids = ["1abc_HA", "2abc_HLA", "3abc_HA", "4abc_HLA"]
    for name, requested in (("a", ids), ("b", list(reversed(ids)))):
        export_subset(source, tmp_path / name, ids=requested, count=2, seed=17)
    targets = [json.loads((tmp_path / n / "inputs/run_manifest.json").read_text())["selected_targets"] for n in ("a", "b")]
    assert targets[0] == targets[1]


@pytest.mark.parametrize("options,message", [
    ({}, "criterion"), ({"ids": []}, "nonempty"),
    ({"ids": ["1abc_HA", "1abc_HA"]}, "unique"),
    ({"ids": ["9abc_HA"]}, "Unknown"),
    ({"max_residues": 0}, "positive"), ({"max_residues": 1}, "no complexes"),
    ({"count": 7, "seed": 1}, "only 6"),
    ({"count": 1}, "both count"), ({"seed": 1}, "both count"),
    ({"count": 1, "seed": -1}, "nonnegative"),
    ({"count": 0, "seed": 1}, "positive"),
])
def test_invalid_selection_creates_no_release(tmp_path, options, message):
    source = make_release(tmp_path / "source")
    with pytest.raises(ValueError, match=message):
        export_subset(source, tmp_path / "output", **options)
    assert not (tmp_path / "output").exists()


def test_rejects_existing_and_overlapping_destinations(tmp_path):
    source = make_release(tmp_path / "source")
    for destination in (source, source / "nested", tmp_path):
        with pytest.raises(ValueError, match="overlap"):
            export_subset(source, destination, ids=["1abc_HA"])
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "inputs").mkdir()
    (existing / "inputs/keep.txt").write_text("preserve")
    with pytest.raises(FileExistsError):
        export_subset(source, existing, ids=["1abc_HA"])


@pytest.mark.parametrize("problem", ["incomplete", "cpu", "changed_file", "missing_stage", "mismatched_ids"])
def test_rejects_invalid_full_source(tmp_path, problem):
    source = make_release(tmp_path / "source")
    manifest = json.loads((source / "run_manifest.json").read_text())
    if problem == "incomplete":
        manifest["status"] = "incomplete"
    elif problem == "cpu":
        manifest["execution"]["canonical"] = False
    elif problem == "missing_stage":
        del manifest["stages"]["7"]
    elif problem == "changed_file":
        (source / "benchmark/structures/1abc_HA.cif").write_text("changed")
    else:
        (source / "benchmark/structures/1abc_HA.cif").unlink()
        refresh_manifest(source)
        manifest = json.loads((source / "run_manifest.json").read_text())
    (source / "run_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises((ValueError, FileNotFoundError)):
        export_subset(source, tmp_path / "output", ids=["1abc_HA"])
    assert not (tmp_path / "output").exists()


def test_cli_ids_file_and_errors(tmp_path, capsys):
    source = make_release(tmp_path / "source")
    ids = tmp_path / "targets.txt"
    ids.write_text("2abc_HLA\n\n1abc_HA\n")
    args = ["subset", "--source-release", str(source), "--output", str(tmp_path / "output"), "--ids-file", str(ids)]
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)["complexes"] == 2
    assert main(["subset", "--source-release", str(source), "--output", str(tmp_path / "bad"), "--count", "1"]) == 2
    assert "both count" in capsys.readouterr().err


def test_rehashed_subset_content_cannot_drift_from_source(tmp_path):
    source = make_release(tmp_path / "source")
    output = tmp_path / "output"
    export_subset(source, output, ids=["1abc_HA"])
    path = output / "inputs/benchmark/benchmark.fasta"
    path.write_text(path.read_text().replace("AC", "AA"))
    manifest = json.loads((output / "inputs/run_manifest.json").read_text())
    manifest["subset_export"]["outputs"] = [artifact_record(output / "inputs", r["path"]) for r in manifest["subset_export"]["outputs"]]
    (output / "inputs/run_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="source sequences"):
        validate_release(output / "inputs", "subset")


def test_failed_staged_export_cleans_up(tmp_path):
    source = make_release(tmp_path / "source")
    path = source / "benchmark/benchmark.fasta"
    path.write_text(path.read_text().replace("AC", "AX"))
    refresh_manifest(source)
    with pytest.raises(ValueError, match="protein chains"):
        export_subset(source, tmp_path / "output", ids=["1abc_HA"])
    assert sorted(p.name for p in tmp_path.iterdir()) == ["source"]


def test_multiple_listed_antigen_chains_are_rejected_before_export(tmp_path):
    source = make_release(tmp_path / "source")
    path = source / "benchmark/registry.tsv"
    fields, rows = read_table(path, "\t")
    rows["1abc_HA"]["antigen_chain"] = "A|B"
    import csv
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows.values())
    refresh_manifest(source)
    with pytest.raises(ValueError, match="single-antigen chain mapping"):
        export_subset(source, tmp_path / "output", ids=["1abc_HA"])
    assert not (tmp_path / "output").exists()
