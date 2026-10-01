"""Redistributable functional coverage for the public SAbDab workflow."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


FIXTURES = Path(__file__).parent / "fixtures"


def _materialize_mini_sabdab(tmp_path: Path) -> Path:
    root = tmp_path / "sabdab2"
    structures = root / "sabdab_all_structures"
    root.mkdir(parents=True)
    shutil.copy2(FIXTURES / "mini_sabdab_summary.csv", root / "sabdab_summary_all.csv")
    for pdb_id in ("1abc", "2def"):
        stem = f"pdb_0000{pdb_id}"
        destination = structures / stem / f"{stem}_sabdab.cif"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(FIXTURES / "simple_complex.cif", destination)
    return root


def test_public_fixture_executes_all_seven_stage_handoffs(
    monkeypatch, tmp_path
):
    """Exercise all stages while replacing only model inference and Foldseek."""
    from dataset_creation_pipeline import utils
    from dataset_creation_pipeline.stages import (
        cdr_filter,
        clustering,
        feature_assembly,
        quality_filter,
        release_screen,
        sequence_extraction,
        structural_deleak,
    )

    sabdab = _materialize_mini_sabdab(tmp_path)
    summary = sabdab / "sabdab_summary_all.csv"
    structures = sabdab / "sabdab_all_structures"
    data_root = tmp_path / "data" / "fixture"
    intermediate = data_root / "intermediate"

    monkeypatch.setattr(release_screen, "RAW_SUMMARY", summary)
    monkeypatch.setattr(release_screen, "IDS_OUTPUT", intermediate / "training_data_ids.json")
    monkeypatch.setattr(release_screen, "MATCHES_OUTPUT", intermediate / "training_data_matches.txt")

    monkeypatch.setattr(quality_filter, "SUMMARY_CSV", summary)
    monkeypatch.setattr(quality_filter, "DATA_ROOT", data_root)
    monkeypatch.setattr(quality_filter, "INTERMEDIATE_ROOT", intermediate)
    monkeypatch.setattr(quality_filter, "TRAINING_IDS_JSON", release_screen.IDS_OUTPUT)
    monkeypatch.setattr(quality_filter, "FILTERED_TSV", intermediate / "02_quality_filtered.tsv")
    monkeypatch.setattr(quality_filter, "METRICS_OUTPUT", intermediate / "02_quality_metrics.json")

    monkeypatch.setattr(sequence_extraction, "DATA_ROOT", data_root)
    monkeypatch.setattr(sequence_extraction, "FILTERED_TSV", quality_filter.FILTERED_TSV)
    monkeypatch.setattr(utils, "SABDAB2_STRUCTURE_DIR", structures)
    utils.parse_sabdab_cif.cache_clear()

    stage1 = release_screen.run(cutoff="2023-06-01")
    stage2 = quality_filter.run(force=True, lower=0, upper=10)
    stage3 = sequence_extraction.run()

    scratch = data_root / "scratch"
    monkeypatch.setattr(cdr_filter, "DATA_ROOT", data_root)
    monkeypatch.setattr(cdr_filter, "INTERMEDIATE_ROOT", intermediate)
    monkeypatch.setattr(cdr_filter, "INPUT_TSV", quality_filter.FILTERED_TSV)
    monkeypatch.setattr(cdr_filter, "FILTERED_TSV", intermediate / "04_cdr_filtered.tsv")
    monkeypatch.setattr(cdr_filter, "SCRATCH_SURVIVORS", scratch / "s4_survivors")

    def fake_chothia_details(sequences, roles):
        cdr_indices = {key: set() for key in sequences}
        h3_lengths = {key: 1 for key, role in roles.items() if role == "H"}
        return cdr_indices, h3_lengths

    monkeypatch.setattr(cdr_filter, "get_chothia_details_batch", fake_chothia_details)
    stage4 = cdr_filter.run(fastas=stage3["fastas"])

    monkeypatch.setattr(structural_deleak, "DATA_ROOT", data_root)
    monkeypatch.setattr(structural_deleak, "INTERMEDIATE_ROOT", intermediate)
    monkeypatch.setattr(structural_deleak, "INPUT_TSV", cdr_filter.FILTERED_TSV)
    monkeypatch.setattr(structural_deleak, "FILTERED_TSV", intermediate / "05_deleaked.tsv")
    monkeypatch.setattr(structural_deleak, "STAGE_METRICS_PATH", intermediate / "stage_metrics.json")
    monkeypatch.setattr(structural_deleak, "SCRATCH_SURVIVORS", cdr_filter.SCRATCH_SURVIVORS)
    monkeypatch.setattr(structural_deleak, "SCRATCH_TRAINING", scratch / "s5_training_db")
    monkeypatch.setattr(structural_deleak, "SCRATCH_FOLDSEEK", scratch / "s5_foldseek")
    monkeypatch.setattr(structural_deleak, "SCRATCH_DELEAKED", scratch / "s5_deleaked")
    monkeypatch.setattr(structural_deleak, "build_training_db", lambda: {"cutoff": "2023-06-01", "n_training": 1})
    monkeypatch.setattr(structural_deleak, "run_foldseek_search", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        structural_deleak,
        "parse_and_decide",
        lambda _processes: (
            {},
            {"cov_values": [], "ilddt_pass": {}},
            {"2def_XY": (0.5, "1abc")},
        ),
    )
    monkeypatch.setattr(structural_deleak, "_write_leakage_summary", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(structural_deleak, "_write_log", lambda *_args, **_kwargs: None)
    stage5 = structural_deleak.run(processes=1, threads=1, gpu=1, foldseek_bin="fixture-foldseek")

    final_benchmark = data_root / "benchmark"
    monkeypatch.setattr(clustering, "DATA_ROOT", data_root)
    monkeypatch.setattr(clustering, "INTERMEDIATE_ROOT", intermediate)
    monkeypatch.setattr(clustering, "INPUT_TSV", structural_deleak.FILTERED_TSV)
    monkeypatch.setattr(clustering, "STAGE_METRICS_PATH", intermediate / "stage_metrics.json")
    monkeypatch.setattr(clustering, "SCRATCH_DELEAKED", structural_deleak.SCRATCH_DELEAKED)
    monkeypatch.setattr(clustering, "SCRATCH_FOLDSEEK", scratch / "s6_foldseek")
    monkeypatch.setattr(clustering, "FINAL_BENCHMARK_DIR", final_benchmark)
    monkeypatch.setattr(clustering, "BENCHMARK_DIR", final_benchmark)
    monkeypatch.setattr(clustering, "run_all_vs_all_search", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        clustering,
        "compute_edges",
        lambda _processes, _ids: (
            pd.DataFrame([{"complex1": "2def_XY", "complex2": "2def_XY", "ilddt": 1.0}]),
            {},
        ),
    )
    monkeypatch.setattr(
        clustering,
        "cluster_complexes",
        lambda *_args, **_kwargs: (
            {1: ["2def_XY"]},
            np.zeros((1, 1)),
            {"2def_XY": 0},
            np.empty((0, 4)),
            ["2def_XY"],
        ),
    )

    def write_fixture_leakage(*_args, **_kwargs):
        clustering.LEAKAGE_SUMMARY_PATH.write_text("fixture leakage summary\n")

    monkeypatch.setattr(clustering, "_write_leakage_summary", write_fixture_leakage)
    monkeypatch.setattr(clustering, "_write_log", lambda *_args, **_kwargs: None)
    stage6 = clustering.run(
        processes=1,
        threads=1,
        gpu=1,
        foldseek_bin="fixture-foldseek",
        fastas=stage3["fastas"],
    )

    monkeypatch.setattr(feature_assembly, "DATA_ROOT", data_root)
    monkeypatch.setattr(feature_assembly, "FILTERED_TSV", final_benchmark / "registry.tsv")
    monkeypatch.setattr(feature_assembly, "OUTPUT_CSV", final_benchmark / "dataset_features.csv")
    stage7 = feature_assembly.run()

    ids = json.loads(release_screen.IDS_OUTPUT.read_text())
    registry = pd.read_csv(quality_filter.FILTERED_TSV, sep="\t")
    assert stage1["pre_cutoff_rows"] == 1
    assert stage1["post_cutoff_rows"] == 1
    assert ids["ids"] == ["1abc"]
    assert stage2["total"] == 1
    assert registry["complex_id"].tolist() == ["2def_XY"]
    assert stage3["fastas"] == {"2def_XY": {"X": "MAG", "Y": "S"}}
    assert stage4["passed"] == 1
    assert stage5["n_novel"] == 1
    assert stage6["n_representatives"] == 1
    assert stage7["total"] == 1
    assert (final_benchmark / "structures" / "2def_XY.cif").is_file()
    assert pd.read_csv(final_benchmark / "dataset_features.csv")["complex_id"].tolist() == ["2def_XY"]
