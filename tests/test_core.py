"""Offline unit and fixed-protocol checks for pipeline primitives."""
from __future__ import annotations

import json
import gzip
from pathlib import Path

import pandas as pd
import pytest

from dataset_creation_pipeline.config import (
    InputConfig,
    OutputConfig,
    PipelineConfig,
    ResourceConfig,
    _config_from_mapping,
)
from dataset_creation_pipeline.foldseek_utils import (
    decide_novelty,
    parse_cigar_to_query_ordinals,
    parse_complex_chains,
    parse_report_row,
    run_foldseek,
)
from dataset_creation_pipeline.preflight import _missing_foldseek_format_fields
from dataset_creation_pipeline.pipeline import (
    _manifest_safe_paths,
    _require_stage_outputs,
    _stage_outputs_match,
)
from dataset_creation_pipeline.provenance import hash_artifact
from dataset_creation_pipeline.stages.clustering import (
    select_representatives,
    select_representatives_by_quality,
)
from dataset_creation_pipeline.utils import (
    chain_full_sequences,
    complex_id_from_row,
    normalize_sabdab2_summary,
    parse_sabdab_cif,
    resolved_positions,
)


FIXTURES = Path(__file__).parent / "fixtures"


def test_default_config_and_fixed_protocol():
    config = PipelineConfig()
    config.validate()
    assert config.selection.cutoff_date == "2023-06-01"
    assert config.selection.methods == ("xray", "em")
    assert config.selection.resolution_min == 1.0
    assert config.selection.resolution_max == 3.5
    assert config.selection.smallest_repr is False
    assert config.representative_rule == "resolution, R-factor, complex ID"
    assert config.foldseek.interface_lddt_threshold == 0.7
    assert config.resources.device == "gpu"
    assert config.output.run_name == "benchmark"

    with pytest.raises(ValueError, match="Frozen FoldSeek"):
        _config_from_mapping({"foldseek": {"interface_lddt_threshold": 0.6}})


def test_config_rejects_invalid_selection_bounds():
    config = PipelineConfig()
    with pytest.raises(ValueError, match="resolution bounds"):
        PipelineConfig(
            selection=config.selection.__class__(resolution_min=3.5, resolution_max=3.5)
        ).validate()


def test_manifest_configuration_omits_machine_specific_paths(tmp_path):
    config = PipelineConfig(
        input=InputConfig(
            sabdab_dir=tmp_path / "private" / "sabdab_database",
        ),
        output=OutputConfig(
            experiment_dir=tmp_path / "private" / "experiment",
        ),
        resources=ResourceConfig(foldseek_bin=str(tmp_path / "env" / "bin" / "foldseek")),
    ).resolved()

    public_config = config.as_dict()
    assert public_config["input"]["sabdab_dir"] == "sabdab_database"
    assert public_config["output"]["experiment_dir"] == "experiment"
    assert public_config["resources"]["foldseek_bin"] == "foldseek"
    assert str(tmp_path) not in json.dumps(public_config)


def test_summary_normalizes_uppercase_pdb_column_and_rejects_bad_ids():
    normalized = normalize_sabdab2_summary(pd.DataFrame({"PDB": [" PDB_00001ABC "]}))
    assert normalized.loc[0, "pdb"] == "1abc"

    with pytest.raises(ValueError, match="unparseable PDB IDs"):
        normalize_sabdab2_summary(pd.DataFrame({"pdb": ["not-an-id"]}))


def test_complex_id_handles_nanobody_and_rejects_multichar_chain_ids():
    assert complex_id_from_row(
        {"pdb": "1ABC", "Hchain": "H", "Lchain": float("nan"), "antigen_chain": "A"}
    ) == "1abc_HA"
    with pytest.raises(ValueError, match="multi-char chain IDs"):
        complex_id_from_row(
            {"pdb": "1abc", "Hchain": "heavy", "Lchain": "L", "antigen_chain": "A"}
        )


def test_cif_parser_resolves_unmodeled_same_entity_chain_and_quoted_values():
    path = FIXTURES / "simple_complex.cif"
    parsed = parse_sabdab_cif(path)
    assert parsed["poly_seq"]["1"] == ["MSE", "ALA", "GLY"]
    assert parsed["chains"]["X"]["label_asym_id"] == "A"
    assert resolved_positions(path, "X") == {1}
    assert chain_full_sequences(path, {"B", "Y"}) == {"B": "MAG", "Y": "S"}


def test_cif_parser_rejects_duplicate_entity_positions():
    with pytest.raises(ValueError, match="duplicate-position"):
        parse_sabdab_cif(FIXTURES / "duplicate_position.cif")


def test_cif_parser_ignores_semicolon_text_in_unrelated_metadata_loop(tmp_path):
    source = (FIXTURES / "simple_complex.cif").read_text()
    path = tmp_path / "metadata_text.cif"
    path.write_text(
        "data_metadata_text\n"
        "loop_\n"
        "_citation_author.citation_id\n"
        "_citation_author.name\n"
        "_citation_author.ordinal\n"
        "primary\n"
        ";O'Connell, S.\n"
        ";\n"
        "8\n"
        + source[source.index("loop_") :]
    )

    parsed = parse_sabdab_cif(path)
    assert parsed["poly_seq"]["1"] == ["MSE", "ALA", "GLY"]
    assert parsed["chains"]["X"]["label_asym_id"] == "A"


def test_cif_parser_reads_gzip_structures(tmp_path):
    source = FIXTURES / "simple_complex.cif"
    compressed = tmp_path / "simple_complex.cif.gz"
    with source.open("rb") as raw, gzip.open(compressed, "wb") as packed:
        packed.write(raw.read())
    assert parse_sabdab_cif(compressed)["poly_seq"]["2"] == ["SER"]


def test_cif_parser_rejects_missing_required_loops(tmp_path):
    malformed = tmp_path / "missing_atom_loop.cif"
    malformed.write_text(
        "data_incomplete\n"
        "loop_\n"
        "_entity_poly_seq.entity_id\n"
        "_entity_poly_seq.num\n"
        "_entity_poly_seq.mon_id\n"
        "_entity_poly_seq.hetero\n"
        "1 1 ALA n\n"
    )
    with pytest.raises(ValueError, match="no _atom_site loop"):
        parse_sabdab_cif(malformed)


def test_foldseek_cigar_query_ordinals_and_chain_id_parsing():
    assert parse_cigar_to_query_ordinals(3, "2M1I2M1D") == {3, 4, 6, 7}
    with pytest.raises(ValueError, match="Unsupported CIGAR op"):
        parse_cigar_to_query_ordinals(1, "2S")
    assert parse_complex_chains("1abc_HLA") == ("H", "L", "A")
    assert parse_complex_chains("1abc_HA") == ("H", None, "A")


@pytest.mark.parametrize("missing_value", ["na", "-na"])
def test_foldseek_missing_interface_lddt_is_conservatively_unmeasured(missing_value):
    row = ["8rtf_LF", "4ksd", "L,F", "A,B", "0", "0", "0", "1", "2", "2", "2M", "0", missing_value]
    query, target, qchains, tchains, ilddt = parse_report_row(row)

    assert (query, target, qchains, tchains) == ("8rtf_LF", "4ksd", ("L", "F"), ("A", "B"))
    assert ilddt != ilddt  # NaN means this hit cannot qualify as overlap evidence.


@pytest.mark.parametrize(
    ("top_full", "best_coverage", "expected"),
    [
        (None, None, "nan"),
        ((0.69, "target"), None, "novel"),
        ((0.9, "target"), None, "novel"),
        ((0.9, "target"), (0.8, "target", 0.6), "leaked"),
    ],
)
def test_structural_novelty_decision(top_full, best_coverage, expected):
    assert decide_novelty(top_full, best_coverage) == expected


def test_representatives_use_resolution_then_r_factor_then_id():
    registry = pd.DataFrame(
        [
            {"complex_id": "a", "resolution": 2.0, "r_factor": 0.30},
            {"complex_id": "b", "resolution": 2.0, "r_factor": 0.25},
            {"complex_id": "c", "resolution": float("nan"), "r_factor": 0.1},
            {"complex_id": "d", "resolution": 2.0, "r_factor": 0.25},
        ]
    )
    assert select_representatives_by_quality({1: ["a", "b", "d"], 2: ["c"]}, registry) == {
        1: "b",
        2: "c",
    }


def test_smallest_representatives_prioritize_size_then_quality_then_id():
    registry = pd.DataFrame([
        {"complex_id": "a", "residue_length": 800, "resolution": 1.0, "r_factor": 0.1},
        {"complex_id": "b", "residue_length": 400, "resolution": 3.0, "r_factor": 0.3},
        {"complex_id": "c", "residue_length": 400, "resolution": 2.0, "r_factor": None},
        {"complex_id": "d", "residue_length": 400, "resolution": 2.0, "r_factor": 0.2},
        {"complex_id": "e", "residue_length": 400, "resolution": 2.0, "r_factor": 0.2},
        {"complex_id": "f", "residue_length": 500, "resolution": None, "r_factor": None},
    ])
    clusters = {1: ["e", "d", "c", "b", "a"], 2: ["f"]}
    assert select_representatives(clusters, registry) == {1: "a", 2: "f"}
    assert select_representatives(clusters, registry, smallest_repr=True) == {1: "d", 2: "f"}
    assert select_representatives({1: ["a", "b"]}, registry, smallest_repr=True) == {1: "b"}


@pytest.mark.parametrize("size", [None, float("nan"), float("inf"), "unknown", 0, -1, 400.5])
def test_smallest_representatives_reject_invalid_sizes(size):
    registry = pd.DataFrame([
        {"complex_id": "a", "residue_length": size, "resolution": 2.0, "r_factor": 0.2}
    ])
    with pytest.raises(ValueError, match="Invalid residue_length.*a"):
        select_representatives({1: ["a"]}, registry, smallest_repr=True)


def test_smallest_representatives_require_sizes_and_registered_members():
    registry = pd.DataFrame([{"complex_id": "a", "resolution": 2.0, "r_factor": 0.2}])
    with pytest.raises(ValueError, match="lacks residue_length"):
        select_representatives({1: ["a"]}, registry, smallest_repr=True)
    registry["residue_length"] = 400
    with pytest.raises(ValueError, match="Cluster member b missing"):
        select_representatives({1: ["b"]}, registry, smallest_repr=True)


def test_smallest_representative_config_is_boolean_and_records_selection():
    config = _config_from_mapping({"selection": {"smallest_repr": True}})
    assert config.as_dict()["selection"]["smallest_repr"] is True
    assert config.representative_rule == "whole-complex residue count, resolution, R-factor, complex ID"
    with pytest.raises(ValueError, match="smallest_repr must be a boolean"):
        _config_from_mapping({"selection": {"smallest_repr": "true"}})


def test_foldseek_dry_run_prints_without_launching_process(monkeypatch, capsys):
    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("dry-run must not start FoldSeek")

    monkeypatch.setattr("dataset_creation_pipeline.foldseek_utils.subprocess.run", fail_if_called)
    run_foldseek(["foldseek", "version"], dry_run=True)
    assert "[DRY-RUN] foldseek version" in capsys.readouterr().out


def test_stable_foldseek_omits_unsupported_zero_chain_threshold(monkeypatch):
    captured = []
    monkeypatch.setattr(
        "dataset_creation_pipeline.foldseek_utils._supports_chain_tm_threshold",
        lambda _binary: False,
    )
    monkeypatch.setattr(
        "dataset_creation_pipeline.foldseek_utils.subprocess.run",
        lambda argv, **_kwargs: captured.extend(argv),
    )

    run_foldseek([
        "foldseek", "easy-multimersearch", "query", "target", "output", "tmp",
        "--chain-tm-threshold", "0",
    ])

    assert "--chain-tm-threshold" not in captured


def test_foldseek_preflight_requires_all_pipeline_output_fields():
    help_text = "query,target,complexqtmscore,complexttmscore,qstart,qend,qlen,cigar"
    assert _missing_foldseek_format_fields(help_text) == ["interfacelddt", "qchains", "tchains"]


def test_legacy_stage6_resume_ignores_only_stage7_feature_file(tmp_path):
    benchmark = tmp_path / "benchmark"
    benchmark.mkdir()
    (benchmark / "representatives.tsv").write_text("complex_id\n1abc_HLA\n")
    saved = hash_artifact(benchmark, tmp_path)
    (benchmark / "dataset_features.csv").write_text("complex_id\n1abc_HLA\n")

    assert _stage_outputs_match(6, [saved], tmp_path)

    (benchmark / "representatives.tsv").write_text("complex_id\n2def_HLA\n")
    assert not _stage_outputs_match(6, [saved], tmp_path)


def test_stage_completion_requires_every_declared_output(tmp_path):
    result = tmp_path / "result.tsv"
    metrics = tmp_path / "metrics.json"
    scratch = tmp_path / "scratch"
    result.write_text("result\n")
    metrics.write_text("{}\n")

    with pytest.raises(RuntimeError, match="every required artifact") as exc_info:
        _require_stage_outputs(4, [result, metrics, scratch])
    assert str(scratch) in str(exc_info.value)

    scratch.mkdir()
    assert _require_stage_outputs(4, [result, metrics, scratch]) == [
        result,
        metrics,
        scratch,
    ]


def test_stage_three_may_declare_no_persistent_outputs():
    assert _require_stage_outputs(3, []) == []


def test_manifest_stage_results_store_relative_or_basename_paths(tmp_path):
    root = tmp_path / "data"
    internal = root / "run" / "intermediate" / "result.tsv"
    external = tmp_path / "Downloads" / "summary.csv"
    result = _manifest_safe_paths(
        {"output_path": str(internal), "reference_path": str(external)}, root
    )
    assert result == {
        "output_path": "run/intermediate/result.tsv",
        "reference_path": "summary.csv",
    }
