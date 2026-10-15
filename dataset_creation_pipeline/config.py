"""Validated configuration for the benchmark protocol and local runs."""
from __future__ import annotations

import os
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "default.toml"
QUALITY_REPRESENTATIVE_RULE = "resolution, R-factor, complex ID"
SMALLEST_REPRESENTATIVE_RULE = "whole-complex residue count, " + QUALITY_REPRESENTATIVE_RULE


@dataclass(frozen=True)
class InputConfig:
    sabdab_dir: Path | None = None
    summary_file: str = "sabdab_summary_all.csv"
    structures_dir: str = "sabdab_all_structures"


@dataclass(frozen=True)
class OutputConfig:
    experiment_dir: Path | None = None
    run_name: str = "benchmark"

    @property
    def root(self) -> Path:
        if self.experiment_dir is None:
            raise ValueError("Specify the experiment folder with --output or [output].experiment_dir")
        return self.experiment_dir

    @property
    def run_dir(self) -> Path:
        return self.root / "inputs"

    @property
    def results_dir(self) -> Path:
        return self.root / "outputs" / "abagbench"

    @property
    def plots_dir(self) -> Path:
        return self.results_dir / "plots"


@dataclass(frozen=True)
class SelectionConfig:
    cutoff_date: str = "2023-06-01"
    methods: tuple[str, ...] = ("xray", "em")
    resolution_min: float = 1.0
    resolution_max: float = 3.5
    xray_r_factor_max: float = 0.3
    exclude_membrane: bool = True
    exclude_engineered_constructs: bool = True
    exclude_vnar: bool = True
    min_residues: int = 350
    max_residues: int = 2200
    smallest_repr: bool = False


@dataclass(frozen=True)
class CDRConfig:
    numbering_scheme: str = "chothia"
    anarcii_mode: str = "accuracy"
    batch_size: int = 32


@dataclass(frozen=True)
class ResourceConfig:
    workers: int = max(1, os.cpu_count() or 1)
    foldseek_threads: int = max(1, os.cpu_count() or 1)
    device: str = "gpu"
    foldseek_bin: str | None = None


@dataclass(frozen=True)
class FrozenFoldseekProtocol:
    """Scientific FoldSeek settings. These are recorded but not CLI-overridable."""

    interface_lddt_threshold: float = 0.7
    interface_coverage_threshold: float = 0.5
    interface_distance_angstrom: float = 8.0
    full_complex_required: bool = True
    linkage_method: str = "complete"
    linkage_cutoff: float = 0.3
    # Default rule; selection.smallest_repr optionally prioritizes complex size.
    representative_rule: str = QUALITY_REPRESENTATIVE_RULE
    chain_tm_threshold: float = 0.0
    foldseek_format: str = (
        "query,target,complexqtmscore,complexttmscore,interfacelddt,"
        "qchains,tchains,qstart,qend,qlen,cigar"
    )


@dataclass(frozen=True)
class PipelineConfig:
    input: InputConfig = field(default_factory=InputConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    cdr: CDRConfig = field(default_factory=CDRConfig)
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    foldseek: FrozenFoldseekProtocol = field(default_factory=FrozenFoldseekProtocol)

    @property
    def representative_rule(self) -> str:
        return SMALLEST_REPRESENTATIVE_RULE if self.selection.smallest_repr else self.foldseek.representative_rule

    def validate(self) -> None:
        import datetime

        try:
            datetime.date.fromisoformat(self.selection.cutoff_date)
        except ValueError as exc:
            raise ValueError("cutoff_date must be an ISO date (YYYY-MM-DD)") from exc
        if not self.selection.methods or not set(self.selection.methods) <= {"xray", "em"}:
            raise ValueError("methods must contain xray and/or em")
        if self.selection.resolution_min < 0 or self.selection.resolution_max <= self.selection.resolution_min:
            raise ValueError("resolution bounds must satisfy 0 <= minimum < maximum")
        if not 0 < self.selection.xray_r_factor_max <= 1:
            raise ValueError("xray_r_factor_max must be in (0, 1]")
        if not isinstance(self.selection.exclude_engineered_constructs, bool) or not isinstance(self.selection.exclude_vnar, bool):
            raise ValueError("engineered-construct and VNAR filter settings must be booleans")
        if self.selection.min_residues < 0 or self.selection.max_residues <= self.selection.min_residues:
            raise ValueError("residue bounds must satisfy 0 <= minimum < maximum")
        if not isinstance(self.selection.smallest_repr, bool):
            raise ValueError("smallest_repr must be a boolean")
        if self.resources.workers < 1 or self.resources.foldseek_threads < 1:
            raise ValueError("worker and FoldSeek thread counts must be at least 1")
        if self.resources.device not in {"cpu", "gpu"}:
            raise ValueError("device must be cpu or gpu")
        if self.cdr.numbering_scheme != "chothia" or self.cdr.anarcii_mode != "accuracy":
            raise ValueError("This release freezes CDR numbering to Anarcii Chothia accuracy mode")
        if self.cdr.batch_size < 1:
            raise ValueError("CDR batch_size must be at least 1")
        if self.foldseek != FrozenFoldseekProtocol():
            raise ValueError("Frozen FoldSeek scoring and clustering protocol values cannot be changed in this release")
        if self.input.summary_file != Path(self.input.summary_file).name:
            raise ValueError("summary_file must be a filename inside --sabdab-dir")
        if not self.output.run_name or Path(self.output.run_name).name != self.output.run_name:
            raise ValueError("run_name must be one path component")
        if self.input.sabdab_dir is not None and self.output.experiment_dir is not None:
            source = self.input.sabdab_dir.expanduser().resolve()
            for managed in (self.output.run_dir, self.output.root / "outputs"):
                managed = managed.expanduser().resolve()
                for artifact in (source / self.input.summary_file, source / self.input.structures_dir):
                    artifact = artifact.resolve()
                    if artifact.is_relative_to(managed) or managed.is_relative_to(artifact):
                        raise ValueError("SAbDab database files must not overlap experiment inputs/ or outputs/")

    def resolved(self) -> "PipelineConfig":
        """Expand user paths without changing the scientific configuration."""
        input_cfg = InputConfig(
            sabdab_dir=self.input.sabdab_dir.expanduser().resolve() if self.input.sabdab_dir else None,
            summary_file=self.input.summary_file,
            structures_dir=self.input.structures_dir,
        )
        output_cfg = OutputConfig(
            experiment_dir=self.output.experiment_dir.expanduser().resolve() if self.output.experiment_dir else None,
            run_name=self.output.run_name,
        )
        resources_cfg = ResourceConfig(
            workers=self.resources.workers,
            foldseek_threads=self.resources.foldseek_threads,
            device=self.resources.device,
            foldseek_bin=str(Path(self.resources.foldseek_bin).expanduser()) if self.resources.foldseek_bin else None,
        )
        result = PipelineConfig(input_cfg, output_cfg, self.selection, self.cdr, resources_cfg, self.foldseek)
        result.validate()
        return result

    def as_dict(self) -> dict[str, Any]:
        """Return portable, machine-path-free settings for logs and manifests."""
        result = asdict(self)
        for section in ("input", "output"):
            for key, value in result[section].items():
                if isinstance(value, Path):
                    result[section][key] = str(value)
        if self.input.sabdab_dir is not None:
            result["input"]["sabdab_dir"] = self.input.sabdab_dir.name
        if self.output.experiment_dir is not None:
            result["output"]["experiment_dir"] = self.output.experiment_dir.name
        if self.resources.foldseek_bin is not None:
            result["resources"]["foldseek_bin"] = Path(self.resources.foldseek_bin).name
        result["selection"]["methods"] = list(self.selection.methods)
        return result


def _config_from_mapping(data: dict[str, Any]) -> PipelineConfig:
    allowed = {"input", "output", "selection", "cdr", "resources", "foldseek"}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"Unknown top-level configuration section(s): {sorted(unknown)}")

    def section(name: str) -> dict[str, Any]:
        value = data.get(name, {})
        if not isinstance(value, dict):
            raise ValueError(f"Configuration section [{name}] must be a table")
        return value

    inp = section("input")
    out = section("output")
    selection = section("selection")
    cdr = section("cdr")
    resources = section("resources")
    foldseek = section("foldseek")
    if isinstance(selection.get("methods"), list):
        selection["methods"] = tuple(str(x).lower() for x in selection["methods"])
    if "sabdab_dir" in inp and inp["sabdab_dir"] is not None:
        inp["sabdab_dir"] = Path(inp["sabdab_dir"])
    for key in ("experiment_dir",):
        if key in out:
            out[key] = Path(out[key])
    cfg = PipelineConfig(
        input=InputConfig(**inp),
        output=OutputConfig(**out),
        selection=SelectionConfig(**selection),
        cdr=CDRConfig(**cdr),
        resources=ResourceConfig(**resources),
        foldseek=FrozenFoldseekProtocol(**foldseek),
    )
    cfg.validate()
    return cfg


def load_config(path: Path | None = None) -> PipelineConfig:
    path = path or DEFAULT_CONFIG
    with path.expanduser().open("rb") as handle:
        raw = tomllib.load(handle)
    output = raw.get("output", {})
    for key in ("experiment_dir",):
        if key in output:
            value = Path(output[key]).expanduser()
            output[key] = value if value.is_absolute() else path.expanduser().resolve().parent / value
    config = _config_from_mapping(raw)
    return config.resolved()


def configure_environment(config: PipelineConfig) -> None:
    """Set runtime paths before importing stage modules, which resolve paths once."""
    if config.input.sabdab_dir is None:
        os.environ.pop("SABDAB2_DIR", None)
    else:
        os.environ["SABDAB2_DIR"] = str(config.input.sabdab_dir)
    os.environ["SABDAB2_SUMMARY_FILE"] = config.input.summary_file
    os.environ["SABDAB2_STRUCTURES_DIR"] = config.input.structures_dir
    os.environ["ABAG_DATA_DIR"] = str(config.output.run_dir)
    os.environ["ABAG_OUTPUT_DIR"] = str(config.output.results_dir)
    os.environ["ABAG_PLOTS_DIR"] = str(config.output.plots_dir)
    os.environ["ABAG_MIN_RESIDUES"] = str(config.selection.min_residues)
    os.environ["ABAG_MAX_RESIDUES"] = str(config.selection.max_residues)
    os.environ["ABAG_RESOLUTION_MIN"] = str(config.selection.resolution_min)
    os.environ["ABAG_RESOLUTION_MAX"] = str(config.selection.resolution_max)
    os.environ["ABAG_XRAY_R_FACTOR_MAX"] = str(config.selection.xray_r_factor_max)
    os.environ["ABAG_EXCLUDE_MEMBRANE"] = "1" if config.selection.exclude_membrane else "0"
    os.environ["ABAG_EXCLUDE_ENGINEERED"] = "1" if config.selection.exclude_engineered_constructs else "0"
    os.environ["ABAG_EXCLUDE_VNAR"] = "1" if config.selection.exclude_vnar else "0"
    os.environ["ABAG_METHODS"] = ",".join(config.selection.methods)
    os.environ["ABAG_WORKERS"] = str(config.resources.workers)
    os.environ["ABAG_FOLDSEEK_THREADS"] = str(config.resources.foldseek_threads)
    os.environ["ABAG_DEVICE"] = config.resources.device
    os.environ["ABAG_ANARCII_BATCH_SIZE"] = str(config.cdr.batch_size)
    if config.resources.foldseek_bin:
        os.environ["FOLDSEEK_BIN"] = config.resources.foldseek_bin
