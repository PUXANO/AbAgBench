"""Capacity input generation bypasses benchmark cohort filters, not validity."""
import csv
import json
from pathlib import Path

import pytest

from dataset_creation_pipeline.cli import main
from dataset_creation_pipeline.dataset_info import dataset_info, format_info
from dataset_creation_pipeline.max_size_test import export_max_size_test, selection_criteria
from dataset_creation_pipeline.release import _sha256, read_fasta, validate_release
from dataset_creation_pipeline.subset import artifact_record


def make_snapshot(root):
    structures = root / "sabdab_all_structures"
    structures.mkdir(parents=True)
    template = (Path(__file__).parent / "fixtures/simple_complex.cif").read_text().replace("'CA atom'", "CA")
    rows = []
    for pdb, length in (("1abc", 4), ("2def", 5), ("3abc", 6)):
        rows.append({"pdb": pdb, "date": "1900/01/01", "Hchain": "X", "Lchain": "",
                     "antigen_chain": "Y", "antigen_type": "protein", "method": "NMR",
                     "resolution": "99", "r_factor": "0.9", "short_header": "MEMBRANE",
                     "type": "VNAR", "chainsharing_construct": "OTHER", "compound": "scfv"})
        extra = "".join(f"2 {i} ALA n\n" for i in range(2, length - 2))
        (structures / f"pdb_0000{pdb}_sabdab.cif").write_text(template.replace("2 1 SER n\n", "2 1 SER n\n" + extra))
    with (root / "sabdab_summary_all.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return root


def hashes(root):
    return {p.relative_to(root).as_posix(): _sha256(p) for p in root.rglob("*") if p.is_file()}


def test_export_real_sequences_bins_and_capacity_identity(tmp_path):
    source = make_snapshot(tmp_path / "snapshot")
    before = hashes(source)
    output = tmp_path / "test"
    output.mkdir()
    (output / "run.env").write_text("existing settings\n")
    criteria = selection_criteria(minimum=4, maximum=6, bin_width=1)
    result = export_max_size_test(source, output, criteria=criteria)
    assert result["complexes"] == 2
    assert [row["complex_id"] for row in result["selected"]] == ["1abc_XY", "3abc_XY"]
    assert [row["residue_length"] for row in result["selected"]] == [4, 6]
    assert hashes(source) == before
    assert (output / "run.env").read_text() == "existing settings\n"
    assert list((output / "outputs").iterdir()) == []
    fasta = read_fasta(output / "inputs/benchmark/benchmark.fasta")
    assert fasta == {"1abc_XY": {"X": "MAG", "Y": "S"}, "3abc_XY": {"X": "MAG", "Y": "SAA"}}
    manifest = json.loads((output / "inputs/run_manifest.json").read_text())
    assert manifest["execution"]["canonical"] is False
    assert manifest["purpose"] == "machine_capacity_test"
    assert "stages" not in manifest and "resolved_config" not in manifest
    source.rename(tmp_path / "source-moved")
    report = dataset_info(output, verify=True)
    assert report["selected_targets"] == ["1abc_XY", "3abc_XY"]
    assert report["dataset_summary"]["residue_length"]["max"] == 6
    assert "Machine capacity-test dataset" in format_info(report)
    assert "whole-complex AA range (observed, inclusive): 4–6" in format_info(report)
    for wrong_kind in ("full", "subset"):
        with pytest.raises(ValueError, match="kind mismatch"):
            validate_release(output / "inputs", wrong_kind)


def test_dry_run_is_read_only_and_explicit_ids_are_normalized(tmp_path, capsys):
    source = make_snapshot(tmp_path / "snapshot")
    output = tmp_path / "test"
    before = hashes(tmp_path)
    assert main(["max-size-test", "--sabdab-dir", str(source), "--output", str(output),
                 "--pdb-ids", "2DEF", "--max-residues", "5", "--dry-run"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["dry_run"] is True
    assert result["selected"] == [{"complex_id": "2def_XY", "residue_length": 5}]
    assert hashes(tmp_path) == before
    assert not output.exists()


def test_ambiguous_pdb_requires_explicit_chain_combination(tmp_path):
    source = make_snapshot(tmp_path / "snapshot")
    summary = source / "sabdab_summary_all.csv"
    with summary.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.append({**rows[0], "Hchain": "B"})  # Same-entity resolved copy is valid.
    with summary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="use --complex-ids"):
        export_max_size_test(source, tmp_path / "bad", criteria=selection_criteria(pdb_ids=["1abc"]))
    result = export_max_size_test(source, tmp_path / "good", criteria=selection_criteria(complex_ids=["1abc_BY"]))
    assert result["selected"] == [{"complex_id": "1abc_BY", "residue_length": 4}]


def test_empty_and_underfilled_bins_export_available_targets_with_coverage(tmp_path):
    source = make_snapshot(tmp_path / "snapshot")
    output = tmp_path / "partial"
    result = export_max_size_test(source, output, criteria=selection_criteria(bin_edges=[4, 5, 7, 8], per_bin=2))
    assert [r["complex_id"] for r in result["selected"]] == ["1abc_XY", "2def_XY", "3abc_XY"]
    assert result["bin_coverage"] == [
        {"lower": 4, "upper": 5, "upper_inclusive": False, "available": 1, "requested": 2,
         "selected_targets": ["1abc_XY"], "selected": 1, "shortfall": 1},
        {"lower": 5, "upper": 7, "upper_inclusive": False, "available": 2, "requested": 2,
         "selected_targets": ["3abc_XY", "2def_XY"], "selected": 2, "shortfall": 0},
        {"lower": 7, "upper": 8, "upper_inclusive": True, "available": 0, "requested": 2,
         "selected_targets": [], "selected": 0, "shortfall": 2},
    ]
    assert len(result["warnings"]) == 2
    report = dataset_info(output, verify=True)
    assert report["bin_coverage"] == result["bin_coverage"]
    assert "0 available; 0/2 selected; shortfall 2" in format_info(report)
    manifest_path = output / "inputs/run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["bin_coverage"][-1]["available"] = 1
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="bin coverage differs"):
        validate_release(output / "inputs", "max-size-test")


def test_all_empty_bins_are_reported_by_dry_run_but_create_no_empty_release(tmp_path):
    source = make_snapshot(tmp_path / "snapshot")
    criteria = selection_criteria(bin_edges=[10, 11, 12])
    result = export_max_size_test(source, tmp_path / "none", criteria=criteria, dry_run=True)
    assert result["selected"] == []
    assert [b["shortfall"] for b in result["bin_coverage"]] == [1, 1]
    with pytest.raises(ValueError, match="No eligible complexes across"):
        export_max_size_test(source, tmp_path / "none", criteria=criteria)
    assert not (tmp_path / "none").exists()


def test_legacy_exact_count_selection_remains_strict(tmp_path):
    source = make_snapshot(tmp_path / "snapshot")
    criteria = selection_criteria(bin_edges=[4, 5, 6, 7, 8])
    criteria["rule"] = "largest_per_bin_then_complex_id"
    with pytest.raises(ValueError, match="Insufficient capacity-bin coverage"):
        export_max_size_test(source, tmp_path / "legacy", criteria=criteria)


def test_invalid_explicit_requests_fail_without_export(tmp_path):
    source = make_snapshot(tmp_path / "snapshot")
    for criteria, message in ((selection_criteria(pdb_ids=["9abc"]), "absent"),
                              (selection_criteria(complex_ids=["1abc_XY"], minimum=5), "outside bounds")):
        with pytest.raises(ValueError, match=message):
            export_max_size_test(source, tmp_path / "bad", criteria=criteria)
        assert not (tmp_path / "bad").exists()


def test_invalid_sequences_are_audited_and_explicit_selection_fails(tmp_path):
    source = make_snapshot(tmp_path / "snapshot")
    path = source / "sabdab_all_structures/pdb_00002def_sabdab.cif"
    path.write_text(path.read_text().replace("2 1 SER n", "2 1 UNK n"))
    criteria = selection_criteria(bin_edges=[4, 5, 6])
    result = export_max_size_test(source, tmp_path / "good", criteria=criteria)
    assert len(result["excluded"]) == 1
    assert result["excluded"][0]["pdb_id"] == "2def"
    assert "unsupported protein sequence" in result["excluded"][0]["reason"]
    with pytest.raises(ValueError, match="Input exclusions"):
        export_max_size_test(source, tmp_path / "bad", criteria=selection_criteria(pdb_ids=["2def"]))


@pytest.mark.parametrize("problem", ["canonical", "purpose", "stages", "targets", "criteria", "summary", "artifact", "sequence"])
def test_capacity_release_validation_fails_closed(tmp_path, problem):
    source = make_snapshot(tmp_path / "snapshot")
    output = tmp_path / "test"
    export_max_size_test(source, output, criteria=selection_criteria(bin_edges=[4, 5, 6]))
    dataset = output / "inputs"
    path = dataset / "run_manifest.json"
    manifest = json.loads(path.read_text())
    if problem == "canonical":
        manifest["execution"]["canonical"] = True
    elif problem == "purpose":
        manifest["purpose"] = "benchmark"
    elif problem == "stages":
        manifest["stages"] = {"6": {"status": "completed"}}
    elif problem == "targets":
        manifest["selected_targets"].pop()
    elif problem == "criteria":
        manifest["selection"]["max_residues"] = 5
    elif problem == "summary":
        manifest["dataset_summary"]["residue_length"]["max"] = 7
    elif problem == "artifact":
        (dataset / "benchmark/structures/1abc_XY.cif").write_text("changed")
    elif problem == "sequence":
        fasta = dataset / "benchmark/benchmark.fasta"
        fasta.write_text(fasta.read_text().replace("MAG", "AAA"))
        manifest["max_size_test_export"]["outputs"] = [artifact_record(dataset, p["path"])
            for p in manifest["max_size_test_export"]["outputs"]]
    path.write_text(json.dumps(manifest))
    with pytest.raises((ValueError, OSError)):
        validate_release(dataset, "max-size-test")


def test_ties_per_bin_and_nonuniform_edges_are_deterministic():
    from dataset_creation_pipeline.release import select_capacity_targets
    candidates = {t: {"chains": {"H": "A" * n}} for t, n in
                  (("1abc_HA", 2000), ("2def_HA", 2050), ("3abc_HA", 2050),
                   ("4abc_HA", 2100), ("5abc_HA", 2400), ("6abc_HA", 3500))}
    criteria = selection_criteria(bin_edges=[2000, 2100, 3500], per_bin=2)
    assert select_capacity_targets(candidates, criteria) == ["2def_HA", "3abc_HA", "5abc_HA", "6abc_HA"]
    assert select_capacity_targets(dict(reversed(list(candidates.items()))), criteria) == select_capacity_targets(candidates, criteria)


@pytest.mark.parametrize("options", [
    {}, {"minimum": 0, "maximum": 6, "bin_width": 1},
    {"minimum": 6, "maximum": 4, "bin_width": 1},
    {"minimum": 4, "maximum": 6, "bin_width": 0},
    {"bin_edges": [4, 4, 6]}, {"bin_edges": [6, 4]},
    {"bin_edges": [4, 6], "per_bin": 0}, {"per_bin": 1},
    {"pdb_ids": ["1abc", "1ABC"]}, {"pdb_ids": ["invalid"]},
    {"complex_ids": ["1abc_XY", "1abc_XY"]},
])
def test_invalid_capacity_options_create_no_output(tmp_path, options):
    source = make_snapshot(tmp_path / "snapshot")
    with pytest.raises(ValueError):
        export_max_size_test(source, tmp_path / "bad", criteria=selection_criteria(**options))
    assert not (tmp_path / "bad").exists()


def test_nonempty_destinations_and_snapshot_mutation_are_rejected(tmp_path, monkeypatch):
    source = make_snapshot(tmp_path / "snapshot")
    output = tmp_path / "occupied"
    (output / "outputs").mkdir(parents=True)
    (output / "outputs/keep").write_text("preserve")
    with pytest.raises(FileExistsError):
        export_max_size_test(source, output, criteria=selection_criteria(pdb_ids=["1abc"]))
    assert (output / "outputs/keep").read_text() == "preserve"
    from dataset_creation_pipeline import utils
    original = utils.write_chain_subset_cif
    def changing_source(path, *args):
        original(path, *args)
        path.write_text(path.read_text() + "\n# mutated\n")
    monkeypatch.setattr(utils, "write_chain_subset_cif", changing_source)
    with pytest.raises(ValueError, match="snapshot changed"):
        export_max_size_test(source, tmp_path / "bad", criteria=selection_criteria(pdb_ids=["1abc"]))
    assert not (tmp_path / "bad").exists()
    assert not list(tmp_path.glob(".bad-*"))
