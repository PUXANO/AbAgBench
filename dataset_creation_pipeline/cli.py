"""Command-line interface for local validation and full dataset construction."""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from .config import DEFAULT_CONFIG, PipelineConfig, load_config
from .preflight import run_preflight


def _common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="TOML configuration file; CLI values override it.")
    parser.add_argument("--sabdab-dir", type=Path, help="SAbDab2 directory containing the summary CSV and structures folder; it may be inside or outside the repository.")
    parser.add_argument("--summary-file", help="Summary CSV filename inside --sabdab-dir (default: sabdab_summary_all.csv).")
    parser.add_argument("--structures-dir", help="Structure directory name inside --sabdab-dir (default: sabdab_all_structures).")
    parser.add_argument("--output", "--data-dir", dest="experiment_dir", type=Path,
                        help="Destination experiment folder; creates inputs/ and outputs/. Required unless set in [output].experiment_dir.")
    parser.add_argument("--run-name", help="Manifest profile label (default: benchmark); does not add a directory.")
    parser.add_argument("--cutoff", help="PDB release cutoff YYYY-MM-DD; entries released strictly before it are excluded and form the FoldSeek reference set.")
    parser.add_argument("--methods", nargs="+", choices=("xray", "em"), help="Allowed methods: xray, em, or both. Changing this changes dataset membership.")
    parser.add_argument("--resolution-min", type=float, help="Inclusive minimum resolution in Å; default 1.0. Changing this changes dataset membership.")
    parser.add_argument("--resolution-max", type=float, help="Exclusive maximum resolution in Å; default 3.5. Changing this changes dataset membership.")
    parser.add_argument("--xray-r-factor-max", type=float, help="Exclusive X-ray R-factor maximum; default 0.3. Changing this changes dataset membership.")
    membrane = parser.add_mutually_exclusive_group()
    membrane.add_argument("--exclude-membrane", dest="exclude_membrane", action="store_true", help="Exclude membrane proteins (default; changes dataset membership).")
    membrane.add_argument("--include-membrane", dest="exclude_membrane", action="store_false", help="Include membrane proteins (changes dataset membership).")
    parser.set_defaults(exclude_membrane=None)
    constructs = parser.add_mutually_exclusive_group()
    constructs.add_argument("--exclude-engineered-constructs", dest="exclude_engineered_constructs", action="store_true", help="Exclude engineered constructs (default; changing this changes dataset membership).")
    constructs.add_argument("--include-engineered-constructs", dest="exclude_engineered_constructs", action="store_false", help="Keep engineered constructs (changes dataset membership).")
    parser.set_defaults(exclude_engineered_constructs=None)
    vnar = parser.add_mutually_exclusive_group()
    vnar.add_argument("--exclude-vnar", dest="exclude_vnar", action="store_true", help="Exclude VNAR/IgNAR entries (default; changing this changes dataset membership).")
    vnar.add_argument("--include-vnar", dest="exclude_vnar", action="store_false", help="Keep VNAR/IgNAR entries (changes dataset membership).")
    parser.set_defaults(exclude_vnar=None)
    parser.add_argument("--min-residues", type=int, help="Strict lower whole-complex sequence length bound; default 350 residues (requires length > bound). Changes membership.")
    parser.add_argument("--max-residues", type=int, help="Strict upper whole-complex sequence length bound; default 2200 residues (requires length < bound). Changes membership.")
    parser.add_argument("--smallest-repr", "--smallest_repr", action="store_true", default=None, help="Select the smallest whole complex in each stage-6 cluster by total residue count, breaking ties by resolution, R-factor, then complex ID. Default: select by quality only.")
    parser.add_argument("--anarcii-batch-size", type=int, help="Anarcii sequence batch size; default 32. Numbering scheme and accuracy mode remain frozen.")
    parser.add_argument("--workers", type=int, help="CPU worker count for Python stages; default: available CPU count.")
    parser.add_argument("--foldseek-threads", type=int, help="FoldSeek thread count; default: available CPU count.")
    parser.add_argument("--device", choices=("cpu", "gpu"), help="FoldSeek device; GPU is required for canonical benchmark construction (default: gpu).")
    parser.add_argument("--foldseek-bin", help="FoldSeek executable path; otherwise use FOLDSEEK_BIN or PATH.")


def _resolved_config(args: argparse.Namespace) -> PipelineConfig:
    config = load_config(args.config)
    inp = config.input
    out = config.output
    selection = config.selection
    resources = config.resources
    if args.sabdab_dir is not None:
        inp = dataclasses.replace(inp, sabdab_dir=args.sabdab_dir.expanduser().resolve())
    if args.summary_file is not None:
        inp = dataclasses.replace(inp, summary_file=args.summary_file)
    if args.structures_dir is not None:
        inp = dataclasses.replace(inp, structures_dir=args.structures_dir)
    if args.experiment_dir is not None:
        out = dataclasses.replace(out, experiment_dir=args.experiment_dir.expanduser().resolve())
    if args.run_name is not None:
        out = dataclasses.replace(out, run_name=args.run_name)
    selection_values = {}
    if args.cutoff is not None:
        selection_values["cutoff_date"] = args.cutoff
    if args.methods is not None:
        selection_values["methods"] = tuple(args.methods)
    for cli_name, field_name in (
        ("resolution_min", "resolution_min"),
        ("resolution_max", "resolution_max"),
        ("xray_r_factor_max", "xray_r_factor_max"),
        ("exclude_membrane", "exclude_membrane"),
        ("exclude_engineered_constructs", "exclude_engineered_constructs"),
        ("exclude_vnar", "exclude_vnar"),
        ("min_residues", "min_residues"),
        ("max_residues", "max_residues"),
        ("smallest_repr", "smallest_repr"),
    ):
        value = getattr(args, cli_name)
        if value is not None:
            selection_values[field_name] = value
    if selection_values:
        selection = dataclasses.replace(selection, **selection_values)
    resource_values = {}
    for key in ("workers", "foldseek_threads", "device", "foldseek_bin"):
        value = getattr(args, key)
        if value is not None:
            resource_values[key] = value
    if resource_values:
        resources = dataclasses.replace(resources, **resource_values)
    cdr = config.cdr
    if args.anarcii_batch_size is not None:
        cdr = dataclasses.replace(cdr, batch_size=args.anarcii_batch_size)
    resolved = dataclasses.replace(config, input=inp, output=out, selection=selection, resources=resources, cdr=cdr).resolved()
    return resolved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m dataset_creation_pipeline",
        description="Construct the antibody–antigen benchmark. Filter or device changes create a different dataset definition; FoldSeek scoring and clustering parameters are fixed.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser("validate", help="Validate SAbDab2 input, dependencies, FoldSeek options, disk, and output permissions without writing artifacts.")
    _common_options(validate_parser)
    run_parser = subparsers.add_parser("run", help="Run the pipeline locally or resume a fingerprint-matched run.")
    _common_options(run_parser)
    mode = run_parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true", help="Resume only stages whose configuration, input, and output hashes match the run manifest.")
    mode.add_argument("--force", action="store_true", help="Overwrite this run's existing artifacts after preflight succeeds.")
    run_parser.add_argument("--dry-run", action="store_true", help="Run read-only preflight and print resolved settings; create no output files.")
    run_parser.add_argument("--from-stage", default="1", help="First stage to run (1–7 or release-screen, quality-filter, sequence-extraction, cdr-filter, structural-deleak, clustering, feature-assembly); requires a matching resume manifest.")
    run_parser.add_argument("--to-stage", default="7", help="Last stage to run (same values as --from-stage).")
    subset = subparsers.add_parser("subset", help="Export selected complexes from a completed canonical release.")
    subset.add_argument("--source-release", required=True, type=Path, help="Completed source dataset directory containing run_manifest.json, normally another experiment's inputs/.")
    subset.add_argument("--output", required=True, type=Path, help="Destination experiment folder; creates inputs/ with the subset and an empty outputs/.")
    ids = subset.add_mutually_exclusive_group()
    ids.add_argument("--ids", nargs="+", help="Explicit complex IDs; other filters apply within this selection.")
    ids.add_argument("--ids-file", type=Path, help="UTF-8 file with one complex ID per nonempty line.")
    subset.add_argument("--complex-type", choices=("all", "nanobody", "antibody"), default="all",
                        help="Filter by heavy-only versus heavy/light antibody chains (default: all).")
    subset.add_argument("--max-residues", type=int, help="Inclusive maximum whole-complex residue count.")
    subset.add_argument("--count", type=int, help="Randomly sample this many qualifying complexes; requires --seed.")
    subset.add_argument("--seed", type=int, help="Explicit nonnegative random seed; requires --count.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "subset":
            from .subset import export_subset

            ids = args.ids
            if args.ids_file is not None:
                ids = [line.strip() for line in args.ids_file.read_text(encoding="utf-8").splitlines() if line.strip()]
            result = export_subset(args.source_release, args.output, ids=ids,
                complex_type=args.complex_type, max_residues=args.max_residues,
                count=args.count, seed=args.seed)
            print(json.dumps(result, indent=2))
            return 0
        config = _resolved_config(args)
        if args.command == "run" and args.dry_run and (args.force or args.resume):
            raise ValueError("--dry-run cannot be combined with --force or --resume")
        config.output.root
        if config.resources.device == "cpu":
            print(
                "WARNING: CPU and GPU FoldSeek modes are not dataset-equivalent. "
                "This CPU run is non-canonical and must be versioned separately.",
                file=sys.stderr,
            )
        if args.command == "validate":
            report = run_preflight(config, require_foldseek=True)
            print(json.dumps(report, indent=2))
            return 0
        from .pipeline import run_pipeline

        result = run_pipeline(
            config,
            force=args.force,
            resume=args.resume,
            dry_run=args.dry_run,
            from_stage=args.from_stage,
            to_stage=args.to_stage,
        )
        print(json.dumps(result, indent=2))
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
