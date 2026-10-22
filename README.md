# AbAgBench

AbAgBench is a reproducible pipeline for building nonredundant SAbDab2 datasets to
benchmark antibody–protein antigen complex structure prediction. The repository contains
code and synthetic test fixtures only; it does not distribute SAbDab2 or generated
datasets. It also exports validated subsets of completed canonical releases for
smoke tests and selected-complex experiments. `dataset-info` displays recorded
settings and observed contents. `max-size-test` builds separate
machine-capacity test inputs directly from a SAbDab2 snapshot.

## Citation

If you use this pipeline, please cite:

> Venturini, M. (2026). *AbAgBench: Antibody–protein antigen complex structure prediction
> benchmark builder* (Version 0.1.0)
> [Computer software]. PUXANO and Hasselt University.
> https://github.com/maven2306/AbAgBench

## Pipeline

| Stage | Name | Operation |
| ---: | --- | --- |
| 1 | `release-screen` | Split pre-cutoff reference structures from post-cutoff candidates. |
| 2 | `quality-filter` | Apply construct, antigen, method, quality, structure, deduplication, and size filters. |
| 3 | `sequence-extraction` | Extract full chain sequences. |
| 4 | `cdr-filter` | Apply Chothia numbering and require complete CDRs. |
| 5 | `structural-deleak` | Remove candidates structurally overlapping the pre-cutoff reference set. |
| 6 | `clustering` | Cluster by structural similarity and select representatives. |
| 7 | `feature-assembly` | Export the final dataset, features, and plots. |

Targets retain the full antibody–antigen interface **as annotated by SAbDab**.
Stage 2 keeps candidates with one listed protein antigen chain and excludes
candidates annotated with multiple antigen chains (`antigen_chain` containing
`|`). Stage 4 extracts the listed heavy chain, optional light chain and antigen
chain into a target mmCIF. A larger source structure can supply several targets.
An antigen chain can belong to a multimeric antigen: its other partners are
omitted when SAbDab does not list them as antigen chains for that antibody.

Within each source PDB entry, entity-copy deduplication retains the candidate
with the most resolved residues for each heavy/light/antigen entity combination,
breaking ties by summary row order. This selects among entity-equivalent copies
regardless of whether their origin is biological or crystallographic. Subsequent
structural de-leaking and clustering select a nonredundant interface dataset.

## Defaults

| Setting | Default |
| --- | --- |
| Device | GPU |
| Release cutoff | `2023-06-01` |
| Methods | X-ray and EM |
| Resolution | `1.0 <= resolution < 3.5 Å` |
| X-ray R-factor | `< 0.3` |
| Membrane proteins | Excluded |
| Engineered constructs | Excluded |
| VNAR/IgNAR | Excluded |
| Whole-complex length | `350 < residues < 2200` |
| CDR numbering | ANARCII 2.0.8; Chothia; accuracy mode |
| CDR-H3 eligibility | At most 36 residues in IMGT 105–117, including insertions |
| Structural overlap | Interface LDDT `>= 0.7`; query-interface coverage `>= 0.5` |
| Clustering | Complete linkage; distance `1 - interface LDDT`; cutoff `0.3` |
| Representative | Resolution, then R-factor, then complex ID |
| Run name | `benchmark` |
| Workers and Foldseek threads | Available CPU count |

Scientific defaults are defined in `configs/default.toml`. CPU and GPU Foldseek modes
can produce different membership; GPU output is canonical. CPU mode is intended for
development and must be treated as a separate dataset version.

## Requirements

For full benchmark construction:

- Linux x86_64 with an NVIDIA GPU; or Windows with WSL2, Ubuntu, and an NVIDIA GPU
  visible inside WSL2.
- Conda or Miniforge.
- A SAbDab2 download containing:

```text
sabdab2/
├── sabdab_summary_all.csv
└── sabdab_all_structures/
    ├── pdb_00001abc/
    │   └── pdb_00001abc_sabdab.cif
    └── ...
```

The `sabdab2/` directory can be anywhere accessible, including outside the experiment
or in a separate folder inside it. Pass its location with `--sabdab-dir`. It must not
overlap the managed `inputs/` or `outputs/` directories. Use `--summary-file` and
`--structures-dir` if the extracted names differ.

## Installation

Run from the repository root:

```bash
conda env create --file environment.yaml
conda activate ab-ag-benchmark-gpu
python -m pip install --no-deps --no-build-isolation -e .
bash scripts/install_foldseek_linux.sh --gpu
```

This installs the pinned Python environment, registers the `abagbench` command,
and builds the required Foldseek commit as `$CONDA_PREFIX/bin/foldseek-pipeline`.
Activate the environment to make `abagbench` available on `PATH`. After pulling
changes to command entry points, rerun the `pip install` command above.

The `abag-dataset-pipeline` command is an alias for `abagbench`.

For Windows, run the commands above in Ubuntu under WSL2. If the checkout is on the
Windows filesystem, the Foldseek build may instead be launched from PowerShell:

```powershell
.\scripts\install_foldseek_windows.ps1 -Distribution Ubuntu -Device gpu
```

For non-canonical CPU development, use `environment-cpu.yaml` and
`scripts/install_foldseek_linux.sh --cpu`.

## Validate and run

With the installed environment active, run from any working directory:

```bash
abagbench validate \
  --sabdab-dir /path/to/sabdab2 \
  --output /path/to/my-experiment \
  --foldseek-bin "$CONDA_PREFIX/bin/foldseek-pipeline"

abagbench run \
  --sabdab-dir /path/to/sabdab2 \
  --output /path/to/my-experiment \
  --foldseek-bin "$CONDA_PREFIX/bin/foldseek-pipeline" \
  --run-name benchmark
```

`validate` checks configuration, the input layout and summary columns, dependency
versions, the Foldseek interface, output permissions, and disk availability. It does
not run an alignment or fully parse every CIF.

## Slurm

An `.sbatch` file may be stored anywhere. It only needs to change to the cloned
repository, activate the environment, and invoke the same command:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=ab-ag-benchmark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --chdir=/absolute/path/to/AbAgBench

set -euo pipefail
source /absolute/path/to/miniforge3/etc/profile.d/conda.sh
conda activate ab-ag-benchmark-gpu

abagbench run \
  --sabdab-dir /absolute/path/to/sabdab2 \
  --output /absolute/path/to/my-experiment \
  --foldseek-bin "$CONDA_PREFIX/bin/foldseek-pipeline" \
  --workers "$SLURM_CPUS_PER_TASK" \
  --foldseek-threads "$SLURM_CPUS_PER_TASK" \
  --run-name benchmark
```

Submit with `sbatch /path/to/run_pipeline.sbatch`. Scheduler directives, partitions,
accounts, modules, memory, and time limits are cluster-specific. If the login node
cannot see a GPU, build Foldseek once from an interactive GPU allocation.

## Options

Run `abagbench run --help` for the complete interface.

- Input/output: `--config`, `--sabdab-dir`, `--summary-file`, `--structures-dir`,
  `--output` (alias: `--data-dir`), `--run-name`
- Selection: `--cutoff`, `--methods`, `--resolution-min`, `--resolution-max`,
  `--xray-r-factor-max`, membrane/construct/VNAR inclusion flags, `--min-residues`,
  `--max-residues`, `--smallest-repr` (alias: `--smallest_repr`)
- Resources: `--device`, `--workers`, `--foldseek-threads`, `--anarcii-batch-size`,
  `--foldseek-bin`
- Run control: `--dry-run`, `--resume`, `--from-stage`, `--to-stage`, `--force`

Changing an input, selection option, device, or fixed scientific setting defines a
different dataset. Use a separate experiment folder for each dataset.

By default, stage 6 selects each cluster's representative by lowest resolution,
then lowest R-factor (missing values last), then complex ID. Add `--smallest_repr`
to select the complex with the fewest total residues across its antibody and antigen
chains instead. Size uses the registry's `residue_length`, computed from full chain
sequences, including unresolved residues. Equal sizes use the default quality order
as a tie-break. Clustering itself is unchanged.

```bash
abagbench run \
  --sabdab-dir /absolute/path/to/sabdab2 \
  --output /absolute/path/to/my-experiment \
  --run-name benchmark-smallest \
  --smallest_repr
```

For TOML profiles, set `smallest_repr = true` under `[selection]`.
The resolved selection flag and effective stage-6 rule are recorded in the run manifest.

## Resume

To restart a failed stage 5 while reusing verified stages 1–4:

```bash
abagbench run \
  --sabdab-dir /path/to/sabdab2 \
  --output /path/to/my-experiment \
  --foldseek-bin "$CONDA_PREFIX/bin/foldseek-pipeline" \
  --run-name benchmark \
  --resume --from-stage 5
```

Resume requires the same inputs and configuration plus that run's manifest,
intermediate files, and scratch files. Every declared stage output must match its
recorded fingerprint.

Stage 4 excludes complexes whose heavy-chain CDR-H3 exceeds ANARCII 2.0.8's
36-residue Chothia conversion limit. Length is verified from the original IMGT
105–117 numbering, counting insertions and ignoring gaps. This conversion region
is broader than the Chothia 96–101 range used for the final `cdrh3_length` feature.
An empty conversion is never interpreted as complete CDR coverage; other
numbering errors still stop the run.

Every rejected complex is recorded in `outputs/abagbench/intermediate/04_cdr_exclusions.tsv`, with
`pdb_id`, `complex_id`, chain, reason, and (for long CDR-H3) IMGT length and limit.
Counts and the eligibility rule are saved in stage-4 metrics and the run manifest;
the exclusion TSV is a required, hashed stage output. Resuming from stage 4 verifies
the earlier stages and reconstructs stage 3's sequences in memory.

## Inspect dataset settings

Use `dataset-info` to display the parameters recorded when a dataset was generated:

```bash
abagbench dataset-info /path/to/my-experiment/inputs
```

The positional path accepts a dataset directory containing `run_manifest.json`,
an experiment directory containing `inputs/run_manifest.json`, or the manifest
file itself. An ambiguous experiment path requires an explicit manifest path.

The default output shows the generation parameters: SAbDab source, selection
filters, CDR rules, structural filtering and clustering thresholds, effective
representative selection, device and worker settings. Resolution and whole-complex
length bounds are labelled with their inclusivity. The effective representative
rule combines `smallest_repr` with the configured quality tie-break and is checked
against the stage-6 rule when both are recorded. Missing information is shown as
`not recorded`; current configuration defaults are never substituted.

A separate **Dataset contents** section shows the final number of complexes,
the observed minimum and maximum whole-complex AA counts, and the IDs at those
extremes. Counts sum the full exported sequences across all chains, including
unresolved residues. These observed bounds describe the selected output; the
generation bounds describe the filters applied before clustering and
representative selection.

Completed runs and subset exports save these statistics in the manifest's
`dataset_summary`. For older completed releases, `dataset-info` computes them
from the final registry, feature table and FASTA when those files are present,
checking target coverage and agreement between feature and sequence lengths.
It leaves the manifest unchanged. A manifest inspected without those files or
recorded statistics shows `not recorded` for the observed range. JSON includes
`dataset_summary_source` (`manifest` or `dataset_files`).

Use `--json` for detailed provenance, including manifest hashes, timestamps,
recorded completion status and dataset size, software versions and the full
source-file inventory.

For subsets, the report shows source-generation settings from the retained,
hash-checked `provenance/source/run_manifest.json`, alongside separate subset
export criteria. Lists of more than ten explicitly requested IDs are summarized
by count in terminal output; `--json` retains the complete list and export software.
The original source directory is not needed.
Generation length bounds are exclusive; the subset's `max_residues` is inclusive.

| Option | Behaviour |
| --- | --- |
| `DATASET` | Required dataset directory, experiment directory, or manifest path. |
| `--json` | Emit parameters, observed contents and detailed provenance as JSON; default: a compact parameter and contents summary. |
| `--verify` | Run the release validator against artifact hashes and target coverage, and check recorded size statistics against the exported files; default: inspect saved metadata, calculating size statistics for older releases when files are available. |

```bash
abagbench dataset-info /path/to/my-experiment --json
abagbench dataset-info /path/to/my-experiment/inputs --verify
```

Metadata inspection requires Python 3.12+ and the source checkout or installed
package. It uses only the standard library and performs no GPU, Foldseek or SAbDab
preflight checks. If Python is provided by an environment, activate that environment
or invoke its Python 3.12+ executable directly. Incomplete and failed runs can be
inspected. `--verify` requires a completed canonical full release or a completed
subset, or a completed max-size-test export, and checks the release files, including
subset/source consistency. Successful verification adds a confirmation line;
errors return a
nonzero exit code. Both modes are read-only.

## Export a subset

`subset` selects survivors from an existing completed canonical GPU release and
writes a separate, self-contained release. It verifies the source manifest,
artifact hashes and target coverage before exporting. It does not rerun filtering,
CDR numbering, structural de-leaking or clustering, and requires no GPU or Foldseek
execution. Selected complexes retain the source release's sequences, features,
chain assignments and native structures. Supported complexes have one heavy chain,
an optional light chain and one protein antigen chain.

Select individual **complex IDs**, rather than PDB IDs; a PDB entry may contain
several complexes:

```bash
abagbench subset \
  --source-release /path/to/full-experiment/inputs \
  --output /path/to/smoke-test-experiment \
  --ids 1abc_HA 2def_HLA
```

Or sample five nanobody complexes with at most 800 total residues:

```bash
abagbench subset \
  --source-release /path/to/full-experiment/inputs \
  --output /path/to/smoke-test-experiment \
  --complex-type nanobody --max-residues 800 \
  --count 5 --seed 17
```

| Option | Behaviour |
| --- | --- |
| `--source-release` | Required path to a completed canonical dataset, normally another experiment’s `inputs/`. |
| `--output` | Required destination experiment folder. Its `inputs/` and `outputs/` must be absent or empty. Source/destination overlap is rejected. Other files, such as `run.env`, are preserved. |
| `--ids ID ...` | Restrict selection to these unique complex IDs. Unknown IDs are errors. |
| `--ids-file FILE` | Alternative to `--ids`: one complex ID per nonempty UTF-8 line. |
| `--complex-type all\|nanobody\|antibody` | Keep both types (default), heavy-only antibodies, or heavy/light antibodies. Each complex also contains its antigen. |
| `--max-residues N` | Inclusive upper bound on the full-sequence whole-complex residue count. |
| `--count N --seed S` | Sample exactly N qualifying complexes without replacement using an explicit nonnegative seed. Both options are required together. |

Specify at least one selection criterion. ID, type and size filters are combined
before sampling. Empty selections, unsupported chain mappings and requests for more
complexes than qualify are errors. Random sampling uses sorted candidate IDs and
the recorded seed, so the same source and criteria reproduce the selection.

The destination contains:

```text
smoke-test-experiment/
├── inputs/
│   ├── run_manifest.json
│   ├── benchmark/
│   │   ├── registry.tsv
│   │   ├── benchmark.fasta
│   │   ├── dataset_features.csv
│   │   └── structures/<complex_id>.cif
│   └── provenance/source/
│       ├── run_manifest.json
│       └── benchmark/
│           ├── registry.tsv
│           ├── benchmark.fasta
│           └── dataset_features.csv
└── outputs/                         # Empty after subset export
```

Every artifact under `inputs/benchmark/` contains exactly the selected complexes. The
manifest declares `release_kind=subset`, the selected IDs, selection criteria,
source-manifest identity, selected native hashes and exported artifact hashes.
It records a completed subset export rather than completed pipeline stages.
`inputs/provenance/source/` preserves the original manifest and complete source metadata
for independent selection verification; those records do not add prediction targets.
The subset can be validated after the original release has been moved or removed.
The source is never edited and the output is published only after validation.
Source clustering diagnostics, intermediate files and plots are not copied.

For universal_runner, place `run.env` in the destination experiment. The runner
uses its `inputs/` by default, reads the subset type from the manifest and records
the manifest hash automatically. Use `DATASET_DIR` only to select another dataset
location. A five-complex export declares five targets for input preparation,
prediction and evaluation.

## Machine-capacity test inputs

`max-size-test` selects antibody–protein antigen inputs directly from a SAbDab2
snapshot for inference memory and runtime measurements. It uses an input-eligibility
profile: distinct single-character heavy, optional light and single protein antigen
chains; full sequences containing the 20 standard amino acids after the existing
modified-residue mapping; and readable mmCIF records with resolvable chains.
Unresolved sequence positions contribute to the AA count. Chain-sharing constructs
must fit this input representation. Selection uses these eligibility checks and
the requested sizes/IDs independently of benchmark date, structure-quality,
CDR-completeness and structural-similarity criteria.

Run in the installed AbAgBench environment. This command uses CPU sequence/structure
parsing and requires the supplied snapshot's summary CSV and mmCIF files.

```bash
abagbench max-size-test \
  --sabdab-dir /path/to/sabdab-snapshot \
  --output /path/to/capacity-test \
  --min-residues 2000 --max-residues 3500 --bin-width 100 --per-bin 1

# Inspect availability and exclusions without writing files.
abagbench max-size-test \
  --sabdab-dir /path/to/sabdab-snapshot \
  --output /path/to/capacity-test \
  --bins 2000 2100 2300 2700 3500 --dry-run

# Select PDB entries; use exact chain combinations when a PDB is ambiguous.
abagbench max-size-test \
  --sabdab-dir /path/to/sabdab-snapshot \
  --output /path/to/capacity-test \
  --pdb-ids 7z12 9mx1 --max-residues 3500
```

| Option | Behaviour |
| --- | --- |
| `--sabdab-dir DIR` | Required SAbDab2 snapshot directory. |
| `--summary-file FILE` | Snapshot-relative summary CSV; default `sabdab_summary_all.csv`. |
| `--structures-dir DIR` | Snapshot-relative structure directory; default `sabdab_all_structures`. Supports extracted and gzip-compressed SAbDab CIF layouts. |
| `--output DIR` | Required experiment folder. Managed `inputs/` and `outputs/` must be absent or empty; existing `run.env` is preserved. |
| `--min-residues N`, `--max-residues N` | Inclusive whole-complex full-sequence AA bounds. Optional for explicit ID selections; required with `--bin-width`. |
| `--bin-width N` | Positive bin width between the requested bounds. The last bin can be shorter. |
| `--bins N ...` | Alternative explicit increasing edges; supplies the overall bounds. |
| `--per-bin N` | Exactly N complexes per bin; default 1 when bins are supplied. |
| `--pdb-ids ID ...` | PDB entries, each resolving to exactly one eligible complex after size filtering. Case is normalized. |
| `--complex-ids ID ...` | Alternative exact complex IDs, including case-sensitive chain suffixes. |
| `--dry-run` | Print the selected IDs/sizes and input-exclusion audit as JSON without creating output files. |

Choose bins or explicit IDs. IDs can restrict the candidates for binned selection.
Each bin is lower-inclusive and upper-exclusive, except that the final upper
edge is inclusive. Selection prefers the largest eligible complexes in each bin,
breaking size ties by complex ID. Underfilled bins, unavailable requested IDs and
ambiguous PDB IDs are errors; ambiguity messages list the eligible complex IDs.
Invalid inputs have recorded exclusion reasons, included in the command's JSON
result and retained with successful exports.

The export creates `inputs/benchmark/{registry.tsv,benchmark.fasta,dataset_features.csv,structures/}`
and an empty `outputs/`. Its feature table records complex ID, PDB ID, nanobody
status and whole-complex AA count. `inputs/provenance/candidates.json` retains
eligible chain sequences, source CIF paths/hashes and input exclusions;
`inputs/provenance/source_summary.csv` retains the source annotations.
`inputs/run_manifest.json` records `release_kind=max-size-test`,
`purpose=machine_capacity_test`, `execution.canonical=false`, selection criteria,
selected targets, observed bounds and artifact hashes. Validation replays selection
from the retained catalog and checks the exported metadata and sequences. The
original snapshot is not required to validate an exported release.

```bash
abagbench dataset-info /path/to/capacity-test --verify
```

Place `run.env` in the capacity-test experiment to prepare MSAs and run inference
with universal_runner. Its configured protocol and resource requests determine the
conditions measured; the release identifies capacity-test inputs separately from
canonical benchmark datasets.

## Experiment layout

`--output` is the experiment folder, not the dataset directory itself. It is
required unless `[output].experiment_dir` is set in the TOML configuration.
Relative TOML paths are resolved against the configuration file. `--run-name`
sets the manifest profile label; it does not add another directory.

A full run creates:

```text
my-experiment/
├── inputs/
│   ├── run_manifest.json
│   └── benchmark/
│       ├── registry.tsv
│       ├── structures/
│       ├── benchmark.fasta
│       ├── edges.tsv
│       ├── clusters.tsv
│       ├── representatives.tsv
│       ├── linkage.npy
│       ├── nodes.json
│       ├── leakage_summary.txt
│       └── dataset_features.csv
└── outputs/
    └── abagbench/
        ├── intermediate/
        ├── scratch/
        ├── logs/
        └── plots/
```

Directories are created automatically. The manifest records the resolved
configuration, canonical status, input/output hashes, software versions and
per-stage timing and memory. Dataset artifact paths are relative to `inputs/`;
working-artifact and plot records refer to `../outputs/abagbench/`.

Create the experiment's `run.env` from universal_runner's template. The runner
uses the adjacent `inputs/` by default and derives the release type and
manifest hash from `inputs/run_manifest.json`; `DATASET_DIR` overrides the dataset
location. The runner creates its preparation and inference directories under `outputs/`.
AbAgBench does not create `run.env` or choose model settings, inference resources
or prediction scratch storage.

## Tests

```bash
python -m pytest -q
```

Tests use redistributable fixtures and do not require SAbDab2 or a GPU.

## Citations

- Capel et al., [SAbDab2: The structural antibody database in the age of machine learning](https://doi.org/10.64898/2026.06.16.732554) (2026 preprint); Dunbar et al., [SAbDab: the structural antibody database](https://doi.org/10.1093/nar/gkt1043) (2014).
- [SAbDab2 data and downloads](https://sabdab.opig.stats.ox.ac.uk/about); [wwPDB data usage policy](https://www.rcsb.org/pages/policies).
- van Kempen et al., [Rapid and sensitive protein complex alignment with Foldseek-Multimer](https://doi.org/10.1038/s41592-025-02593-7) (2025).
- Foldseek, [multimer search output fields](https://github.com/steineggerlab/foldseek#multimer-search-output).
- Greenshields-Watson et al., [ANARCII enables alignment-free antigen receptor numbering using a generalised language model](https://doi.org/10.1038/s42003-026-10186-z) (2026).
- Virtanen et al., [SciPy 1.0: fundamental algorithms for scientific computing in Python](https://doi.org/10.1038/s41592-019-0686-2) (2020).

## License

Copyright 2026 PUXANO. The source code is licensed under Apache-2.0; see `LICENSE` and
`NOTICE`.
