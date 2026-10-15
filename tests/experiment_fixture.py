"""Exercise real pipeline orchestration with offline scientific-tool fixtures."""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from test_public_fixture import _materialize_mini_sabdab
from dataset_creation_pipeline.config import InputConfig, OutputConfig, PipelineConfig, configure_environment


def run(root: Path) -> None:
    sabdab = _materialize_mini_sabdab(root)
    config = PipelineConfig(input=InputConfig(sabdab_dir=sabdab),
                            output=OutputConfig(experiment_dir=root / "experiment"))
    config = dataclasses.replace(config, selection=dataclasses.replace(config.selection, min_residues=0, max_residues=10))
    configure_environment(config)
    # Stage modules resolve their paths at import, just as in a CLI invocation.
    from dataset_creation_pipeline import pipeline, plotting
    from dataset_creation_pipeline.paths import PLOTS_ROOT
    from dataset_creation_pipeline.stages import cdr_filter, clustering, structural_deleak

    def numbering(sequences, roles):
        return ({key: set() for key in sequences},
                {key: 1 for key, role in roles.items() if role == "H"}, {})

    def plots():
        PLOTS_ROOT.mkdir(parents=True, exist_ok=True)
        (PLOTS_ROOT / "fixture.txt").write_text("plot fixture")

    with (
        patch.object(pipeline, "run_preflight", return_value={"foldseek_binary": "fixture-foldseek", "foldseek_version": "fixture"}),
        patch.object(cdr_filter, "get_chothia_details_batch", numbering),
        patch.object(structural_deleak, "build_training_db", return_value={"cutoff": "2023-06-01", "n_training": 1}),
        patch.object(structural_deleak, "run_foldseek_search"),
        patch.object(structural_deleak, "parse_and_decide", return_value=({}, {"cov_values": [], "ilddt_pass": {}}, {"2def_XY": (0.5, "1abc")})),
        patch.object(structural_deleak, "_write_leakage_summary"),
        patch.object(clustering, "run_all_vs_all_search"),
        patch.object(clustering, "compute_edges", return_value=(pd.DataFrame([{"complex1": "2def_XY", "complex2": "2def_XY", "ilddt": 1.0}]), {})),
        patch.object(clustering, "cluster_complexes", return_value=({1: ["2def_XY"]}, np.zeros((1, 1)), {"2def_XY": 0}, np.empty((0, 4)), ["2def_XY"])),
        patch.object(clustering, "_write_leakage_summary", lambda *_: clustering.LEAKAGE_SUMMARY_PATH.write_text("fixture summary")),
        patch.object(plotting, "generate_all", plots),
    ):
        pipeline.run_pipeline(config)
        # Real fingerprint validation must reuse the outputs in their new paths.
        pipeline.run_pipeline(config, resume=True)


if __name__ == "__main__":
    run(Path(sys.argv[1]))
