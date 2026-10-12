# AbAgBench

AbAgBench is a reproducible pipeline for building nonredundant SAbDab2 datasets to
benchmark antibody–protein antigen complex structure prediction. The repository contains
code and synthetic test fixtures only; it does not distribute SAbDab2 or generated
datasets.

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
| Structural overlap | Interface LDDT `>= 0.7`; query-interface coverage `>= 0.5` |
| Clustering | Complete linkage; distance `1 - interface LDDT`; cutoff `0.3` |
| Representative | Resolution, then R-factor, then complex ID |
| Run name | `benchmark` |
| Workers and Foldseek threads | Available CPU count |

Scientific defaults are defined in `configs/default.toml`. CPU and GPU Foldseek modes
can produce different membership; GPU output is canonical. CPU mode is intended for
development and must be treated as a separate dataset version.

## Requirements

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

The `sabdab2/` directory may be inside the repository (it is Git-ignored) or elsewhere.
Pass its location with `--sabdab-dir`. Use `--summary-file` and `--structures-dir` if
the extracted names differ.

## Installation

Run from the repository root:

```bash
conda env create --file environment.yaml
conda activate ab-ag-benchmark-gpu
bash scripts/install_foldseek_linux.sh --gpu
```

This installs the pinned Python environment and builds the required Foldseek commit as
`$CONDA_PREFIX/bin/foldseek-pipeline`.

For Windows, run the commands above in Ubuntu under WSL2. If the checkout is on the
Windows filesystem, the Foldseek build may instead be launched from PowerShell:

```powershell
.\scripts\install_foldseek_windows.ps1 -Distribution Ubuntu -Device gpu
```

For non-canonical CPU development, use `environment-cpu.yaml` and
`scripts/install_foldseek_linux.sh --cpu`.

## Validate and run

From the repository root:

```bash
python -m dataset_creation_pipeline validate \
  --sabdab-dir /path/to/sabdab2 \
  --foldseek-bin "$CONDA_PREFIX/bin/foldseek-pipeline"

python -m dataset_creation_pipeline run \
  --sabdab-dir /path/to/sabdab2 \
  --foldseek-bin "$CONDA_PREFIX/bin/foldseek-pipeline" \
  --run-name benchmark
```

`validate` checks configuration, the input layout and summary columns, dependency
versions, the Foldseek interface, output permissions, and disk availability. It does
not run an alignment or fully parse every CIF.

## Slurm

An `.sbatch` file may be stored anywhere. It only needs to change to the cloned
repository, activate the environment, and invoke the same module entry point:

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

python -m dataset_creation_pipeline run \
  --sabdab-dir /absolute/path/to/sabdab2 \
  --foldseek-bin "$CONDA_PREFIX/bin/foldseek-pipeline" \
  --workers "$SLURM_CPUS_PER_TASK" \
  --foldseek-threads "$SLURM_CPUS_PER_TASK" \
  --run-name benchmark
```

Submit with `sbatch /path/to/run_pipeline.sbatch`. Scheduler directives, partitions,
accounts, modules, memory, and time limits are cluster-specific. If the login node
cannot see a GPU, build Foldseek once from an interactive GPU allocation.

## Options

Run `python -m dataset_creation_pipeline run --help` for the complete interface.

- Input/output: `--config`, `--sabdab-dir`, `--summary-file`, `--structures-dir`,
  `--data-dir`, `--plots-dir`, `--run-name`
- Selection: `--cutoff`, `--methods`, `--resolution-min`, `--resolution-max`,
  `--xray-r-factor-max`, membrane/construct/VNAR inclusion flags, `--min-residues`,
  `--max-residues`, `--smallest-repr` (alias: `--smallest_repr`)
- Resources: `--device`, `--workers`, `--foldseek-threads`, `--anarcii-batch-size`,
  `--foldseek-bin`
- Run control: `--dry-run`, `--resume`, `--from-stage`, `--to-stage`, `--force`

Changing an input, selection option, device, or fixed scientific setting defines a
different dataset. Use a different run name rather than mixing results.

By default, stage 6 selects each cluster's representative by lowest resolution,
then lowest R-factor (missing values last), then complex ID. Add `--smallest_repr`
to select the complex with the fewest total residues across its antibody and antigen
chains instead. Size uses the registry's `residue_length`, computed from full chain
sequences, including unresolved residues. Equal sizes use the default quality order
as a tie-break. Clustering itself is unchanged.

```bash
python -m dataset_creation_pipeline run \
  --sabdab-dir /absolute/path/to/sabdab2 \
  --run-name benchmark-smallest \
  --smallest_repr
```

For TOML profiles, set `smallest_repr = true` under `[selection]`. The option also
works when running `python -m dataset_creation_pipeline.stages.clustering` directly.
The resolved selection flag and effective stage-6 rule are recorded in the run manifest.

## Resume

To restart a failed stage 5 while reusing verified stages 1–4:

```bash
python -m dataset_creation_pipeline run \
  --sabdab-dir /path/to/sabdab2 \
  --foldseek-bin "$CONDA_PREFIX/bin/foldseek-pipeline" \
  --run-name benchmark \
  --resume --from-stage 5
```

Resume requires the same inputs and configuration plus that run's manifest,
intermediate files, and scratch files. Every declared stage output must match its
recorded fingerprint.

## Outputs

```text
data/<run-name>/
├── run_manifest.json
├── intermediate/
├── scratch/
├── logs/
└── benchmark/
    ├── registry.tsv
    ├── structures/
    ├── benchmark.fasta
    ├── edges.tsv
    ├── clusters.tsv
    ├── representatives.tsv
    ├── linkage.npy
    ├── nodes.json
    ├── leakage_summary.txt
    └── dataset_features.csv

plots/<run-name>/
```

The manifest records the resolved configuration, canonical status, input and output
hashes, software versions, and per-stage timing and memory.

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
