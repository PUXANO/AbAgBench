"""CDR eligibility regressions using synthetic numbering, without model inference."""
from __future__ import annotations

import copy
import json
import pandas as pd
import pytest

from dataset_creation_pipeline import utils
from dataset_creation_pipeline.pipeline import _stage_outputs_match
from dataset_creation_pipeline.provenance import output_fingerprints
from dataset_creation_pipeline.stages import cdr_filter


def _imgt_entry(h3_length=13, role="H", offset=0):
    # Real IMGT insertion labels split between 111 and 112. Gaps elsewhere
    # exercise residue counting independently of numbering-list length.
    extra = h3_length - 13
    numbering = []
    for pos in range(1, 129):
        if pos == 112:
            numbering.extend(((112, chr(65 + i)), "A") for i in reversed(range(extra // 2)))
        numbering.append(((pos, " "), "-" if pos == 10 else "A"))
        if pos == 111:
            numbering.extend(((111, chr(65 + i)), "A") for i in range((extra + 1) // 2))
    sequence = "".join(aa for _, aa in numbering if aa != "-")
    return "G" * offset + sequence + "GG", {
        "numbering": numbering, "chain_type": role, "score": 1.0,
        "query_start": offset, "query_end": offset + len(sequence) - 1,
        "error": None, "scheme": "imgt",
    }


class _AnarciiWithoutInference:
    def __init__(self, entries):
        self.entries = entries
        self.converted = None

    def number(self, sequences):
        assert set(sequences) == set(self.entries)
        return copy.deepcopy(self.entries)

    def to_scheme(self, scheme):
        assert scheme == "chothia"
        if self.converted is not None:
            return self.converted
        # Exercise the pinned package's actual converter, including its silent
        # empty-numbering result, while avoiding model inference and downloads.
        from anarcii.output_data_processing.schemes import convert_number_scheme

        return convert_number_scheme(self.entries, scheme)


@pytest.mark.parametrize("length", [13, 36, 37, 63])
def test_conversion_limit_and_successful_light_chain(monkeypatch, length):
    heavy, heavy_entry = _imgt_entry(length, offset=2)
    light, light_entry = _imgt_entry(role="K", offset=1)
    model = _AnarciiWithoutInference({"heavy": heavy_entry, "light": light_entry})
    monkeypatch.setattr(utils, "_get_anarcii_model", lambda: model)
    cdrs, lengths, excluded = utils.get_chothia_details_batch(
        {"heavy": heavy, "light": light}, {"heavy": "H", "light": "L"},
    )
    assert cdrs["light"]
    assert "light" not in lengths and "light" not in excluded
    if length > 36:
        assert "heavy" not in cdrs and "heavy" not in lengths
        assert excluded["heavy"]["cdrh3_length_imgt"] == length
        assert excluded["heavy"]["max_supported_length"] == 36
        assert excluded["heavy"]["query_start"] == 2
        converted = model.to_scheme("chothia")["heavy"]
        assert converted["numbering"] == [] and converted["error"] is None
    else:
        assert excluded == {}
        assert lengths["heavy"] == length - 4
        assert len(cdrs["heavy"]) > lengths["heavy"]
        assert min(cdrs["heavy"]) >= 2


def test_gap_inside_h3_does_not_exceed_limit(monkeypatch):
    seq, entry = _imgt_entry(37)
    index = next(i for i, ((pos, _), _) in enumerate(entry["numbering"]) if pos == 106)
    entry["numbering"][index] = ((106, " "), "-")
    seq = seq[:-3] + "GG"  # All V residues are A; remove one and preserve the tail.
    entry["query_end"] -= 1
    monkeypatch.setattr(utils, "_get_anarcii_model", lambda: _AnarciiWithoutInference({"heavy": entry}))
    _, lengths, excluded = utils.get_chothia_details_batch({"heavy": seq}, {"heavy": "H"})
    assert excluded == {} and lengths["heavy"] == 32


@pytest.mark.parametrize("failure", ["initial_error", "conversion_error", "empty", "missing", "mismatch", "short", "imgt_mismatch", "offset"])
def test_unrelated_numbering_failures_still_raise(monkeypatch, failure):
    seq, entry = _imgt_entry()
    model = _AnarciiWithoutInference({"heavy": entry})
    if failure == "initial_error":
        entry["error"] = "unrecognized sequence"
    elif failure == "imgt_mismatch":
        entry["numbering"][0] = ((1, " "), "G")
    elif failure == "offset":
        entry["query_end"] = len(seq)
    else:
        converted = model.to_scheme("chothia")
        if failure == "conversion_error":
            converted["heavy"]["error"] = "conversion failed"
        elif failure == "empty":
            converted["heavy"]["numbering"] = []
        elif failure == "missing":
            converted = {}
        elif failure == "mismatch":
            converted["heavy"]["numbering"][0] = ((1, " "), "G")
        elif failure == "short":
            converted["heavy"]["numbering"].pop()
        model.converted = converted
    monkeypatch.setattr(utils, "_get_anarcii_model", lambda: model)
    with pytest.raises((RuntimeError, ValueError), match="heavy"):
        utils.get_chothia_details_batch({"heavy": seq}, {"heavy": "H"})


@pytest.mark.parametrize("ordinary_status", ["resolved", "incomplete", "absent"])
def test_stage_excludes_and_records_long_h3(monkeypatch, tmp_path, ordinary_status):
    keep_ordinary = ordinary_status == "resolved"
    intermediate = tmp_path / "intermediate"
    intermediate.mkdir()
    long_seq, long_entry = _imgt_entry(63)
    short_seq, short_entry = _imgt_entry()
    rows = [{"pdb": "8edf", "Hchain": "H", "Lchain": "", "antigen_chain": "A", "modality": "EM"}]
    fastas = {"8edf_HA": {"H": long_seq, "A": "G"}}
    entries = {"8edf_HA__H": long_entry}
    if ordinary_status != "absent":
        rows.append({"pdb": "2def", "Hchain": "H", "Lchain": "", "antigen_chain": "A", "modality": "EM"})
        fastas["2def_HA"] = {"H": short_seq, "A": "G"}
        entries["2def_HA__H"] = short_entry
    pd.DataFrame(rows).to_csv(intermediate / "02_quality_filtered.tsv", sep="\t", index=False)
    (intermediate / "stage_metrics.json").write_text(json.dumps({"stages": [], "final": len(rows)}))
    cif = tmp_path / "source.cif"
    cif.write_text("synthetic source\n")
    monkeypatch.setattr(cdr_filter, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(cdr_filter, "LOGS_ROOT", tmp_path / "logs")
    monkeypatch.setattr(cdr_filter, "INTERMEDIATE_ROOT", intermediate)
    monkeypatch.setattr(cdr_filter, "INPUT_TSV", intermediate / "02_quality_filtered.tsv")
    monkeypatch.setattr(cdr_filter, "FILTERED_TSV", intermediate / "04_cdr_filtered.tsv")
    monkeypatch.setattr(cdr_filter, "SCRATCH_SURVIVORS", tmp_path / "survivors")
    monkeypatch.setattr(cdr_filter, "pdb_to_cif_path", lambda _: cif)
    monkeypatch.setattr(cdr_filter, "resolved_positions", lambda *_: set(range(1, len(short_seq) + 1)) if keep_ordinary else set())
    monkeypatch.setattr(cdr_filter, "write_chain_subset_cif", lambda _src, dst, _chains: dst.write_text("synthetic subset\n"))
    monkeypatch.setattr(utils, "_get_anarcii_model", lambda: _AnarciiWithoutInference(entries))

    result = cdr_filter.run(fastas=fastas)
    dropped = len(rows) - int(keep_ordinary)
    assert result["passed"] == int(keep_ordinary) and result["failures"] == dropped
    assert result["numbering_eligibility"]["long_cdrh3_excluded"] == 1
    exclusions = pd.read_csv(intermediate / "04_cdr_exclusions.tsv", sep="\t")
    assert exclusions["pdb_id"].tolist() == (["8edf", "2def"] if ordinary_status == "incomplete" else ["8edf"])
    assert exclusions.iloc[0]["complex_id"] == "8edf_HA"
    assert exclusions.iloc[0]["reason_code"] == "cdrh3_too_long"
    assert "36 residues" in exclusions.iloc[0]["reason"]
    assert exclusions.iloc[0]["cdrh3_length_imgt"] == 63
    if ordinary_status == "incomplete":
        assert exclusions.iloc[1]["reason_code"] == "cdr_incomplete"
    registry = pd.read_csv(cdr_filter.FILTERED_TSV, sep="\t")
    assert registry["pdb"].tolist() == (["2def"] if keep_ordinary else [])
    assert not (cdr_filter.SCRATCH_SURVIVORS / "8edf_HA.cif").exists()
    assert len(list(cdr_filter.SCRATCH_SURVIVORS.glob("*.cif"))) == int(keep_ordinary)
    metrics = json.loads((intermediate / "04_cdr_metrics.json").read_text())
    assert metrics["dropped"] == sum(metrics["reasons"].values()) == dropped
    funnel = json.loads((intermediate / "stage_metrics.json").read_text())
    assert funnel["final"] == int(keep_ordinary)
    assert funnel["stages"][-1]["reasons"] == metrics["reasons"]
    # Resume must check the new audit artifact, including edits/deletion.
    audit = intermediate / "04_cdr_exclusions.tsv"
    recorded = output_fingerprints([audit], tmp_path)
    assert _stage_outputs_match(4, recorded, tmp_path)
    assert not _stage_outputs_match(4, [], tmp_path)
    audit.write_text("tampered\n")
    assert not _stage_outputs_match(4, recorded, tmp_path)
    audit.unlink()
    assert not _stage_outputs_match(4, recorded, tmp_path)
