"""Offline checks for representative selection and stage-6 exports."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dataset_creation_pipeline.stages import clustering


@pytest.mark.parametrize("flag", ["--smallest-repr", "--smallest_repr"])
def test_standalone_clustering_cli_passes_smallest_flag(monkeypatch, flag):
    captured = {}
    monkeypatch.setattr(sys, "argv", ["clustering", flag])
    monkeypatch.setattr(clustering, "run", lambda **kwargs: captured.update(kwargs))
    clustering.main()
    assert captured["smallest_repr"] is True


@pytest.mark.parametrize(
    ("smallest_repr", "expected"), [(False, "1abc_XY"), (True, "2def_XY")]
)
def test_clustering_exports_representative_using_requested_rule(
    monkeypatch, tmp_path, smallest_repr, expected
):
    intermediate = tmp_path / "intermediate"
    intermediate.mkdir()
    input_tsv = intermediate / "05_deleaked.tsv"
    pd.DataFrame([
        {"complex_id": "1abc_XY", "resolution": 1.5, "r_factor": 0.1,
         "modality": "xray", "cdrh3_length": 1, "residue_length": 800},
        {"complex_id": "2def_XY", "resolution": 3.0, "r_factor": 0.2,
         "modality": "xray", "cdrh3_length": 1, "residue_length": 400},
    ]).to_csv(input_tsv, sep="\t", index=False)
    stage_metrics = intermediate / "stage_metrics.json"
    stage_metrics.write_text(json.dumps({"stages": [], "final": 2}))
    deleaked = tmp_path / "scratch" / "s5_deleaked"
    deleaked.mkdir(parents=True)
    nodes = ["1abc_XY", "2def_XY"]
    for cid in nodes:
        shutil.copy2(Path(__file__).parent / "fixtures" / "simple_complex.cif", deleaked / f"{cid}.cif")
    final = tmp_path / "benchmark"
    for name, value in {
        "DATA_ROOT": tmp_path,
        "INTERMEDIATE_ROOT": intermediate,
        "INPUT_TSV": input_tsv,
        "STAGE_METRICS_PATH": stage_metrics,
        "SCRATCH_DELEAKED": deleaked,
        "FINAL_BENCHMARK_DIR": final,
        "BENCHMARK_DIR": final,
    }.items():
        monkeypatch.setattr(clustering, name, value)
    # run() rebinds these paths to its staging directory; restore them after each test.
    for name in ("STRUCTURES_DIR", "FASTA_PATH", "EDGES_PATH", "CLUSTERS_PATH",
                 "REPS_PATH", "LINKAGE_PATH", "NODES_PATH", "LEAKAGE_SUMMARY_PATH", "FILTERED_TSV"):
        monkeypatch.setattr(clustering, name, getattr(clustering, name))
    monkeypatch.setattr(clustering, "run_all_vs_all_search", lambda *_args: None)
    edges = pd.DataFrame([{"complex1": nodes[0], "complex2": nodes[1], "ilddt": 0.9}])
    monkeypatch.setattr(clustering, "compute_edges", lambda *_args: (edges, {}))
    monkeypatch.setattr(clustering, "cluster_complexes", lambda *_args, **_kwargs: (
        {1: nodes}, np.zeros((2, 2)), {cid: i for i, cid in enumerate(nodes)},
        np.array([[0, 1, 0.1, 2]]), nodes,
    ))
    monkeypatch.setattr(clustering, "_write_leakage_summary", lambda *_args:
                        clustering.LEAKAGE_SUMMARY_PATH.write_text("fixture\n"))
    result = clustering.run(
        processes=1, threads=1, fastas={cid: {"X": "MAG", "Y": "S"} for cid in nodes},
        smallest_repr=smallest_repr,
    )
    assert result["n_representatives"] == 1
    assert pd.read_csv(final / "representatives.tsv", sep="\t")["complex_id"].tolist() == [expected]
    assert pd.read_csv(final / "registry.tsv", sep="\t")["complex_id"].tolist() == [expected]
    assert sorted(path.name for path in (final / "structures").glob("*.cif")) == [f"{expected}.cif"]
    assert f">{expected}" in (final / "benchmark.fasta").read_text()
    log = next((tmp_path / "logs").glob("s6_foldseek_cluster_*.log"))
    assert f"Representative rule: {result['representative_rule']}" in log.read_text()
    assert result["representative_rule"].startswith("whole-complex residue count") is smallest_repr
