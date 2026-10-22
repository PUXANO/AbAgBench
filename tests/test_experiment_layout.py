import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from dataset_creation_pipeline.cli import _resolved_config, build_parser, main
from dataset_creation_pipeline.config import InputConfig, OutputConfig, PipelineConfig, load_config
from dataset_creation_pipeline.release import validate_release
from dataset_creation_pipeline.subset import export_subset
from subset_support import make_release


def test_full_pipeline_routes_dataset_and_results_and_resumes(tmp_path):
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root), "PYTHONDONTWRITEBYTECODE": "1", "MPLCONFIGDIR": str(tmp_path / "matplotlib")}
    result = subprocess.run([sys.executable, str(Path(__file__).with_name("experiment_fixture.py")), str(tmp_path)],
                            cwd=root, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "[RESUME] stage 6" in result.stdout
    assert "[RESUME] stage 4" in result.stdout
    experiment = tmp_path / "experiment"
    assert {p.name for p in experiment.iterdir()} == {"inputs", "outputs"}
    assert {p.name for p in (experiment / "inputs").iterdir()} == {"run_manifest.json", "benchmark"}
    results = experiment / "outputs/abagbench"
    assert {p.name for p in results.iterdir()} == {"intermediate", "scratch", "logs", "plots"}
    assert list((results / "logs").glob("*.log"))
    assert (results / "plots/fixture.txt").is_file()
    assert validate_release(experiment / "inputs")["complexes"] == 1
    manifest = json.loads((experiment / "inputs/run_manifest.json").read_text())
    assert (experiment / "inputs" / manifest["run_log"]).is_file()
    assert manifest["status"] == "completed"
    assert manifest["dataset_summary"] == {
        "complexes": 1,
        "residue_length": {"min": 4, "max": 4,
                           "min_targets": ["2def_XY"], "max_targets": ["2def_XY"]},
    }
    # The database remains external to the experiment and is independently reusable.
    assert (tmp_path / "sabdab2/sabdab_summary_all.csv").is_file()
    subset = tmp_path / "smoke-test"
    export_subset(experiment / "inputs", subset, ids=["2def_XY"])
    assert validate_release(subset / "inputs", "subset")["complexes"] == 1
    assert list((subset / "outputs").iterdir()) == []


@pytest.mark.parametrize("placeholders", [False, True])
def test_subset_accepts_existing_experiment_and_preserves_user_files(tmp_path, placeholders):
    source = make_release(tmp_path / "full/inputs")
    output = tmp_path / "smoke"
    output.mkdir()
    (output / "run.env").write_text("user configuration")
    if placeholders:
        (output / "inputs").mkdir()
        (output / "outputs").mkdir()
    export_subset(source, output, ids=["1abc_HA"])
    assert (output / "run.env").read_text() == "user configuration"
    assert list((output / "outputs").iterdir()) == []
    assert validate_release(output / "inputs", "subset")["complexes"] == 1


@pytest.mark.parametrize("name", ["inputs", "outputs"])
def test_subset_rejects_nonempty_managed_folders_without_changes(tmp_path, name):
    source = make_release(tmp_path / "full/inputs")
    experiment = tmp_path / "smoke"
    managed = experiment / name
    managed.mkdir(parents=True)
    (managed / "keep.txt").write_text("keep")
    with pytest.raises(FileExistsError):
        export_subset(source, experiment, ids=["1abc_HA"])
    assert (managed / "keep.txt").read_text() == "keep"
    assert sorted(p.name for p in experiment.iterdir()) == [name]


def test_output_is_explicit_and_config_paths_are_relative_to_toml(tmp_path, capsys):
    assert main(["run"]) == 2
    assert "experiment folder" in capsys.readouterr().err
    profile = tmp_path / "profile.toml"
    profile.write_text('[output]\nexperiment_dir = "my-experiment"\nrun_name = "a-label"\n')
    config = load_config(profile)
    assert config.output.run_dir == tmp_path / "my-experiment/inputs"
    assert config.output.plots_dir == tmp_path / "my-experiment/outputs/abagbench/plots"
    override = _resolved_config(build_parser().parse_args(["run", "--config", str(profile), "--output", str(tmp_path / "override")]))
    assert override.output.run_dir == tmp_path / "override/inputs"


def test_database_can_be_inside_or_outside_experiment_but_not_managed_paths(tmp_path):
    experiment = tmp_path / "experiment"
    for database in (tmp_path / "external-sabdab", experiment / "sabdab", experiment):
        config = PipelineConfig(input=InputConfig(sabdab_dir=database), output=OutputConfig(experiment_dir=experiment))
        config.validate()
    for database in (experiment / "inputs/sabdab", experiment / "outputs/sabdab"):
        config = PipelineConfig(input=InputConfig(sabdab_dir=database), output=OutputConfig(experiment_dir=experiment))
        with pytest.raises(ValueError, match="overlap"):
            config.validate()
