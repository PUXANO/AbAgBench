"""CLI option, dispatch, and lightweight functional smoke tests."""
from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from dataset_creation_pipeline import cli
from dataset_creation_pipeline.config import PipelineConfig, configure_environment
from dataset_creation_pipeline.pipeline import _resolve_stage, run_pipeline
from dataset_creation_pipeline.preflight import REQUIRED_SUMMARY_COLUMNS, run_preflight
from dataset_creation_pipeline.utils import StageMetrics, make_quality_masks


def test_all_common_cli_options_override_the_config(tmp_path):
    config_path = tmp_path / "profile.toml"
    config_path.write_text(
        """
[output]
run_name = "from-config"

[selection]
resolution_max = 3.4

[resources]
device = "cpu"
""".strip()
    )
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "run",
            "--config", str(config_path),
            "--sabdab-dir", str(tmp_path / "sabdab2"),
            "--summary-file", "summary.csv",
            "--structures-dir", "structures",
            "--data-dir", str(tmp_path / "data"),
            "--plots-dir", str(tmp_path / "plots"),
            "--run-name", "cli-overrides",
            "--cutoff", "2024-01-02",
            "--methods", "xray",
            "--resolution-min", "1.2",
            "--resolution-max", "3.2",
            "--xray-r-factor-max", "0.25",
            "--include-membrane",
            "--include-engineered-constructs",
            "--include-vnar",
            "--min-residues", "400",
            "--max-residues", "2000",
            "--anarcii-batch-size", "7",
            "--workers", "3",
            "--foldseek-threads", "4",
            "--device", "gpu",
            "--foldseek-bin", str(tmp_path / "foldseek"),
            "--resume",
            "--from-stage", "cdr-filter",
            "--to-stage", "feature-assembly",
        ]
    )
    config = cli._resolved_config(args)

    assert config.input.sabdab_dir == (tmp_path / "sabdab2").resolve()
    assert config.input.summary_file == "summary.csv"
    assert config.input.structures_dir == "structures"
    assert config.output.data_dir == (tmp_path / "data").resolve()
    assert config.output.plots_dir == (tmp_path / "plots").resolve()
    assert config.output.run_name == "cli-overrides"
    assert config.selection.cutoff_date == "2024-01-02"
    assert config.selection.methods == ("xray",)
    assert config.selection.resolution_min == 1.2
    assert config.selection.resolution_max == 3.2
    assert config.selection.xray_r_factor_max == 0.25
    assert config.selection.exclude_membrane is False
    assert config.selection.exclude_engineered_constructs is False
    assert config.selection.exclude_vnar is False
    assert config.selection.min_residues == 400
    assert config.selection.max_residues == 2000
    assert config.cdr.batch_size == 7
    assert config.resources.workers == 3
    assert config.resources.foldseek_threads == 4
    assert config.resources.device == "gpu"
    assert config.resources.foldseek_bin == str(tmp_path / "foldseek")
    assert args.resume is True
    assert args.force is False
    assert args.dry_run is False
    assert args.from_stage == "cdr-filter"
    assert args.to_stage == "feature-assembly"


@pytest.mark.parametrize(
    ("flags", "field", "expected"),
    [
        (["--exclude-membrane"], "exclude_membrane", True),
        (["--include-membrane"], "exclude_membrane", False),
        (["--exclude-engineered-constructs"], "exclude_engineered_constructs", True),
        (["--include-engineered-constructs"], "exclude_engineered_constructs", False),
        (["--exclude-vnar"], "exclude_vnar", True),
        (["--include-vnar"], "exclude_vnar", False),
    ],
)
def test_each_boolean_selection_flag(flags, field, expected):
    args = cli.build_parser().parse_args(["run", *flags])
    config = cli._resolved_config(args)
    assert getattr(config.selection, field) is expected


def test_configured_values_reach_stage_environment(monkeypatch, tmp_path):
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "run",
            "--sabdab-dir", str(tmp_path / "sabdab2"),
            "--summary-file", "summary.csv",
            "--structures-dir", "structures",
            "--data-dir", str(tmp_path / "data"),
            "--plots-dir", str(tmp_path / "plots"),
            "--run-name", "environment-smoke",
            "--methods", "em",
            "--resolution-min", "1.3",
            "--resolution-max", "3.1",
            "--xray-r-factor-max", "0.24",
            "--include-membrane",
            "--include-engineered-constructs",
            "--include-vnar",
            "--min-residues", "401",
            "--max-residues", "1999",
            "--anarcii-batch-size", "9",
            "--workers", "2",
            "--foldseek-threads", "5",
            "--device", "gpu",
            "--foldseek-bin", str(tmp_path / "foldseek"),
        ]
    )
    config = cli._resolved_config(args)

    expected = {
        "SABDAB2_DIR": str((tmp_path / "sabdab2").resolve()),
        "SABDAB2_SUMMARY_FILE": "summary.csv",
        "SABDAB2_STRUCTURES_DIR": "structures",
        "ABAG_DATA_DIR": str((tmp_path / "data" / "environment-smoke").resolve()),
        "ABAG_PLOTS_DIR": str((tmp_path / "plots" / "environment-smoke").resolve()),
        "ABAG_METHODS": "em",
        "ABAG_RESOLUTION_MIN": "1.3",
        "ABAG_RESOLUTION_MAX": "3.1",
        "ABAG_XRAY_R_FACTOR_MAX": "0.24",
        "ABAG_EXCLUDE_MEMBRANE": "0",
        "ABAG_EXCLUDE_ENGINEERED": "0",
        "ABAG_EXCLUDE_VNAR": "0",
        "ABAG_MIN_RESIDUES": "401",
        "ABAG_MAX_RESIDUES": "1999",
        "ABAG_ANARCII_BATCH_SIZE": "9",
        "ABAG_WORKERS": "2",
        "ABAG_FOLDSEEK_THREADS": "5",
        "ABAG_DEVICE": "gpu",
        "FOLDSEEK_BIN": str(tmp_path / "foldseek"),
    }
    # Register each variable with pytest's monkeypatch before the production
    # helper overwrites it so the process environment is restored afterwards.
    for name in expected:
        monkeypatch.setenv(name, "pre-test-value")
    configure_environment(config)
    for name, value in expected.items():
        assert __import__("os").environ[name] == value

    # Confirm stage filtering consumes the resolved method and quality settings.
    frame = pd.DataFrame(
        [
            {"method": "XRAY", "resolution": 2.0, "r_factor": 0.2, "short_header": "SOLUBLE"},
            {"method": "ELECTRON_MICROSCOPY", "resolution": 2.0, "r_factor": None, "short_header": "MEMBRANE"},
        ]
    )
    xray, em = make_quality_masks(frame)
    assert xray.tolist() == [False, False]
    assert em.tolist() == [False, True]


def test_quality_selection_flags_change_filter_membership():
    frame = pd.DataFrame(
        [
            {"method": "XRAY", "resolution": 1.1, "r_factor": 0.20, "short_header": "SOLUBLE"},
            {"method": "XRAY", "resolution": 2.0, "r_factor": 0.25, "short_header": "SOLUBLE"},
            {"method": "XRAY", "resolution": 3.0, "r_factor": 0.20, "short_header": "SOLUBLE"},
            {"method": "XRAY", "resolution": 2.0, "r_factor": 0.20, "short_header": "MEMBRANE"},
            {"method": "ELECTRON_MICROSCOPY", "resolution": 2.0, "r_factor": None, "short_header": "SOLUBLE"},
        ]
    )
    xray, em = make_quality_masks(
        frame,
        resolution_min=1.2,
        resolution_max=3.0,
        r_factor_max=0.25,
        exclude_membrane=True,
        methods=("xray", "em"),
    )
    assert xray.tolist() == [False, False, False, False, False]
    assert em.tolist() == [False, False, False, False, True]

    xray, em = make_quality_masks(
        frame,
        resolution_min=1.0,
        resolution_max=3.5,
        r_factor_max=0.3,
        exclude_membrane=False,
        methods=("xray",),
    )
    assert xray.tolist() == [True, True, True, True, False]
    assert em.tolist() == [False] * 5


def test_construct_vnar_and_residue_bound_flag_behaviors(monkeypatch):
    from dataset_creation_pipeline.stages import quality_filter

    constructs = pd.DataFrame(
        [
            {"chainsharing_construct": "SCFV", "compound": "ordinary", "Hchain": "H", "Lchain": "L"},
            {"chainsharing_construct": "", "compound": "ordinary", "Hchain": "H", "Lchain": "L"},
        ]
    )
    kept_constructs = quality_filter.apply_construct_removal(
        constructs, StageMetrics("constructs")
    )
    assert kept_constructs.index.tolist() == [1]

    vnars = pd.DataFrame([{"type": "VNAR"}, {"type": "FAB"}])
    kept_vnars = quality_filter.apply_vnar_removal(vnars, StageMetrics("vnars"))
    assert kept_vnars.index.tolist() == [1]

    lengths = pd.Series({0: 400.0, 1: 401.0, 2: 1998.0, 3: 1999.0})
    monkeypatch.setattr(
        quality_filter,
        "compute_complex_residue_lengths",
        lambda frame, cache=None: lengths.loc[frame.index],
    )
    filtered = quality_filter.filter_by_complex_residue_count(
        pd.DataFrame(index=lengths.index), lower=400, upper=1999
    )
    assert filtered.index.tolist() == [1, 2]


def test_cutoff_flag_uses_strict_before_boundary(monkeypatch, tmp_path):
    from dataset_creation_pipeline.stages import release_screen

    summary = tmp_path / "sabdab_summary_all.csv"
    pd.DataFrame(
        [
            {"pdb": "1aaa", "date": "2023/05/31"},
            {"pdb": "1aab", "date": "2023/06/01"},
            {"pdb": "1aac", "date": "2023/06/02"},
        ]
    ).to_csv(summary, index=False)
    monkeypatch.setattr(release_screen, "RAW_SUMMARY", summary)
    monkeypatch.setattr(release_screen, "IDS_OUTPUT", tmp_path / "intermediate" / "training_data_ids.json")
    monkeypatch.setattr(release_screen, "MATCHES_OUTPUT", tmp_path / "intermediate" / "training_data_matches.txt")
    result = release_screen.run(cutoff="2023-06-01")
    saved = json.loads(release_screen.IDS_OUTPUT.read_text())
    assert result["pre_cutoff_rows"] == 1
    assert result["post_cutoff_rows"] == 2
    assert saved["ids"] == ["1aaa"]


def test_anarcii_batch_size_flag_reaches_model(monkeypatch):
    import dataset_creation_pipeline.utils as utils

    captured = {}

    class FakeAnarcii:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake_module = types.ModuleType("anarcii")
    fake_module.Anarcii = FakeAnarcii
    monkeypatch.setitem(sys.modules, "anarcii", fake_module)
    monkeypatch.setattr("importlib.metadata.version", lambda package: "2.0.8")
    monkeypatch.setattr(utils, "_ANARCII_MODEL", None)
    monkeypatch.setenv("ABAG_ANARCII_BATCH_SIZE", "19")
    utils._get_anarcii_model()
    assert captured == {
        "seq_type": "antibody",
        "mode": "accuracy",
        "batch_size": 19,
        "cpu": True,
        "verbose": False,
    }


def test_main_dispatches_validate_with_resolved_flags(monkeypatch, tmp_path, capsys):
    captured = {}

    def fake_preflight(config, require_foldseek):
        captured["config"] = config
        captured["require_foldseek"] = require_foldseek
        return {"ok": True}

    monkeypatch.setattr(cli, "run_preflight", fake_preflight)
    result = cli.main(
        [
            "validate",
            "--sabdab-dir", str(tmp_path / "sabdab2"),
            "--device", "cpu",
            "--workers", "6",
        ]
    )
    assert result == 0
    assert captured["require_foldseek"] is True
    assert captured["config"].input.sabdab_dir == (tmp_path / "sabdab2").resolve()
    assert captured["config"].resources.device == "cpu"
    assert captured["config"].resources.workers == 6
    assert json.loads(capsys.readouterr().out) == {"ok": True}


@pytest.mark.parametrize(
    ("mode_flag", "resume", "force"),
    [("--resume", True, False), ("--force", False, True)],
)
def test_main_dispatches_run_control_flags(
    monkeypatch, tmp_path, capsys, mode_flag, resume, force
):
    captured = {}

    def fake_run_pipeline(config, **kwargs):
        captured["config"] = config
        captured.update(kwargs)
        return {"status": "smoke-passed"}

    import dataset_creation_pipeline.pipeline as pipeline

    monkeypatch.setattr(pipeline, "run_pipeline", fake_run_pipeline)
    result = cli.main(
        [
            "run",
            "--sabdab-dir", str(tmp_path / "sabdab2"),
            mode_flag,
            "--from-stage", "quality-filter",
            "--to-stage", "clustering",
        ]
    )
    assert result == 0
    assert captured["resume"] is resume
    assert captured["force"] is force
    assert captured["dry_run"] is False
    assert captured["from_stage"] == "quality-filter"
    assert captured["to_stage"] == "clustering"
    assert json.loads(capsys.readouterr().out) == {"status": "smoke-passed"}


def test_main_dispatches_dry_run(monkeypatch, tmp_path):
    captured = {}

    def fake_run_pipeline(config, **kwargs):
        captured.update(kwargs)
        return {"dry_run": True}

    import dataset_creation_pipeline.pipeline as pipeline

    monkeypatch.setattr(pipeline, "run_pipeline", fake_run_pipeline)
    assert cli.main(["run", "--sabdab-dir", str(tmp_path), "--dry-run"]) == 0
    assert captured["dry_run"] is True


def test_cpu_mode_emits_noncanonical_warning(monkeypatch, tmp_path, capsys):
    import dataset_creation_pipeline.pipeline as pipeline

    monkeypatch.setattr(
        pipeline,
        "run_pipeline",
        lambda _config, **_kwargs: {"status": "smoke-passed"},
    )
    assert cli.main([
        "run",
        "--sabdab-dir", str(tmp_path / "sabdab2"),
        "--device", "cpu",
    ]) == 0
    assert "non-canonical" in capsys.readouterr().err


def test_cli_rejects_incompatible_modes_and_run_only_validate_flags(capsys):
    assert cli.main(["run", "--dry-run", "--force"]) == 2
    assert "--dry-run cannot be combined" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["run", "--resume", "--force"])
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["validate", "--dry-run"])


@pytest.mark.parametrize("argv", [["--help"], ["validate", "--help"], ["run", "--help"]])
def test_help_flags_exit_successfully(argv, capsys):
    with pytest.raises(SystemExit) as exc_info:
        cli.build_parser().parse_args(argv)
    assert exc_info.value.code == 0
    assert "usage:" in capsys.readouterr().out


def test_all_stage_names_and_numbers_and_invalid_ranges():
    names = [
        "release-screen",
        "quality-filter",
        "sequence-extraction",
        "cdr-filter",
        "structural-deleak",
        "clustering",
        "feature-assembly",
    ]
    assert [_resolve_stage(name) for name in names] == list(range(1, 8))
    assert [_resolve_stage(str(number)) for number in range(1, 8)] == list(range(1, 8))
    with pytest.raises(ValueError, match="Unknown stage"):
        _resolve_stage("not-a-stage")
    with pytest.raises(ValueError, match="at or before"):
        run_pipeline(PipelineConfig(), from_stage="7", to_stage="1")
    with pytest.raises(ValueError, match="requires --resume"):
        run_pipeline(PipelineConfig(), from_stage="2")


def test_pipeline_dry_run_is_non_writing(monkeypatch, tmp_path, capsys):
    data_dir = tmp_path / "not-created-data"
    plots_dir = tmp_path / "not-created-plots"
    config = PipelineConfig(
        output=PipelineConfig().output.__class__(
            data_dir=data_dir,
            plots_dir=plots_dir,
            run_name="fixture-dry-run",
        )
    )
    monkeypatch.setattr(
        "dataset_creation_pipeline.pipeline.run_preflight",
        lambda _config, require_foldseek: {
            "foldseek_binary": "foldseek-pipeline",
            "foldseek_version": "8dc75c74ad0eddab73cfd905963d13bf74dc012b",
            "foldseek_chain_tm_threshold_supported": True,
        },
    )
    result = run_pipeline(config, dry_run=True)
    assert result["dry_run"] is True
    assert not data_dir.exists()
    assert not plots_dir.exists()
    assert "no output directories or files were created" in capsys.readouterr().out


def test_preflight_functional_smoke_accepts_public_fixture_layout(
    monkeypatch, tmp_path
):
    sabdab = tmp_path / "sabdab2"
    structures = sabdab / "sabdab_all_structures" / "pdb_00001abc"
    structures.mkdir(parents=True)
    (structures / "pdb_00001abc_sabdab.cif").write_text("data_1abc\n")
    row = {column: "value" for column in REQUIRED_SUMMARY_COLUMNS}
    row.update(
        {
            "pdb": "1abc",
            "date": "2024/01/01",
            "Hchain": "H",
            "Lchain": "L",
            "antigen_chain": "A",
            "resolution": "2.0",
            "r_factor": "0.2",
        }
    )
    row["PDB"] = row.pop("pdb")
    pd.DataFrame([row]).to_csv(sabdab / "sabdab_summary_all.csv", index=False)
    foldseek = tmp_path / "foldseek-pipeline"
    foldseek.write_text("test executable placeholder")

    def fake_subprocess_run(argv, **_kwargs):
        if argv[-1] == "version":
            return subprocess.CompletedProcess(
                argv, 0, "8dc75c74ad0eddab73cfd905963d13bf74dc012b\n", ""
            )
        return subprocess.CompletedProcess(
            argv,
            0,
            "query target complexqtmscore complexttmscore interfacelddt "
            "qchains tchains qstart qend qlen cigar --chain-tm-threshold --gpu",
            "",
        )

    monkeypatch.setattr("dataset_creation_pipeline.preflight.subprocess.run", fake_subprocess_run)
    config = cli._resolved_config(
        cli.build_parser().parse_args(
            [
                "validate",
                "--sabdab-dir", str(sabdab),
                "--device", "cpu",
                "--foldseek-bin", str(foldseek),
                "--data-dir", str(tmp_path / "data"),
                "--plots-dir", str(tmp_path / "plots"),
            ]
        )
    )
    report = run_preflight(config)
    assert report["summary_rows"] == 1
    assert report["structure_file_count"] == 1
    assert report["foldseek_missing_format_fields"] == []
    assert report["errors"] == []
