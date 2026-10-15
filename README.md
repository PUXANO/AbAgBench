# AbAgBench

AbAgBench is a reproducible pipeline for building nonredundant SAbDab2 datasets to
benchmark antibody–protein antigen complex structure prediction. The repository contains
code and synthetic test fixtures only; it does not distribute SAbDab2 or generated
datasets. It also exports validated subsets of completed canonical releases for
smoke tests and selected-complex experiments.

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

Stage 4 excludes complexes whose heavy-chain CDR-H3 exceeds ANARCII 2.0.8's
36-residue Chothia conversion limit. Length is verified from the original IMGT
105–117 numbering, counting insertions and ignoring gaps. This conversion region
is broader than the Chothia 96–101 range used for the final `cdrh3_length` feature.
An empty conversion is never interpreted as complete CDR coverage; other
numbering errors still stop the run.

Every rejected complex is recorded in `intermediate/04_cdr_exclusions.tsv`, with
`pdb_id`, `complex_id`, chain, reason, and (for long CDR-H3) IMGT length and limit.
Counts and the eligibility rule are saved in stage-4 metrics and the run manifest;
the exclusion TSV is a required, hashed stage output. Resuming from stage 4 verifies
the earlier stages and reconstructs stage 3's sequences in memory.

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
python -m dataset_creation_pipeline subset \
  --source-release /path/to/completed-release \
  --output /path/to/test-release \
  --ids 1abc_HA 2def_HLA
```

Or sample five nanobody complexes with at most 800 total residues:

```bash
python -m dataset_creation_pipeline subset \
  --source-release /path/to/completed-release \
  --output /path/to/test-release \
  --complex-type nanobody --max-residues 800 \
  --count 5 --seed 17
```

| Option | Behaviour |
| --- | --- |
| `--source-release` | Required path to a completed canonical full release. |
| `--output` | Required new directory; existing or overlapping destinations are rejected. |
| `--ids ID ...` | Restrict selection to these unique complex IDs. Unknown IDs are errors. |
| `--ids-file FILE` | Alternative to `--ids`: one complex ID per nonempty UTF-8 line. |
| `--complex-type all\|nanobody\|antibody` | Keep both types (default), heavy-only antibodies, or heavy/light antibodies. Each complex also contains its antigen. |
| `--max-residues N` | Inclusive upper bound on the full-sequence whole-complex residue count. |
| `--count N --seed S` | Sample exactly N qualifying complexes without replacement using an explicit nonnegative seed. Both options are required together. |

Specify at least one selection criterion. ID, type and size filters are combined
before sampling. Empty selections, unsupported chain mappings and requests for more
complexes than qualify are errors. Random sampling uses sorted candidate IDs and
the recorded seed, so the same source and criteria reproduce the selection.

The output contains:

```text
test-release/
├── run_manifest.json
├── benchmark/
│   ├── registry.tsv
│   ├── benchmark.fasta
│   ├── dataset_features.csv
│   └── structures/<complex_id>.cif
└── provenance/source/
    ├── run_manifest.json
    └── benchmark/
        ├── registry.tsv
        ├── benchmark.fasta
        └── dataset_features.csv
```

Every artifact under `benchmark/` contains exactly the selected complexes. The
manifest declares `release_kind=subset`, the selected IDs, selection criteria,
source-manifest identity, selected native hashes and exported artifact hashes.
It records a completed subset export rather than completed pipeline stages.
`provenance/source/` preserves the original manifest and complete source metadata
for independent selection verification; those records do not add prediction targets.
The subset can be validated after the original release has been moved or removed.
The source is never edited and the output is published only after validation.
Source clustering diagnostics, intermediate files and plots are not copied.

For universal_runner, point `OPENDDE_DATASET_RELEASE_DIR` at the exported directory,
set `DATASET_RELEASE_KIND=subset`, and set `OPENDDE_DATASET_MANIFEST_SHA256` to the
new `run_manifest.json` hash. A five-complex export declares five targets for input
preparation, prediction and evaluation.

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
