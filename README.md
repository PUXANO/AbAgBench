# AbAgBench

Build reproducible, nonredundant datasets for benchmarking antibody–protein antigen
complex structure prediction from [SAbDab2](https://sabdab.opig.stats.ox.ac.uk/about).

AbAgBench filters structures by release date, quality, construct type and size,
checks CDR completeness with ANARCII, removes structural overlap with pre-cutoff
structures, and clusters the remaining complexes with Foldseek-Multimer. Each
target contains a heavy chain, an optional light chain and one SAbDab-listed
protein antigen chain.

## What you can do

| Command | Purpose |
| --- | --- |
| `run` | Build a dataset, or resume a run with verified inputs and outputs. |
| `validate` | Check inputs, configuration and dependencies before running. |
| `dataset-info` | Inspect a dataset's settings and contents; optionally verify artifact hashes and target coverage. |
| `subset` | Export selected complexes from a completed canonical release by ID, antibody type, size or seeded sampling. |
| `max-size-test` | Select inputs directly from SAbDab2 by size bins or IDs for inference capacity measurements. |

## Install

Benchmark construction requires Linux x86_64 (or Ubuntu under Windows WSL2),
an NVIDIA GPU with a working driver, and Conda or Miniforge. The environment
provides Python 3.12 and pinned dependencies; the installer builds the required
Foldseek revision.

From the repository root:

```bash
conda env create --file environment.yaml
conda activate ab-ag-benchmark-gpu
python -m pip install --no-deps --no-build-isolation -e .
bash scripts/install_foldseek_linux.sh --gpu
```

## Build a dataset

Download a SAbDab2 snapshot containing `sabdab_summary_all.csv` and
`sabdab_all_structures/`. With the environment active, run from any directory:

```bash
abagbench run \
  --sabdab-dir /path/to/sabdab2 \
  --output /path/to/my-experiment \
  --foldseek-bin "$CONDA_PREFIX/bin/foldseek-pipeline"
```

The dataset is written to `my-experiment/inputs/`: mmCIF structures, FASTA
sequences, a registry, feature and clustering tables, and a manifest recording
settings, software versions and artifact hashes. Logs, plots and working files
go under `my-experiment/outputs/abagbench/`.

## Configure

Defaults are in [configs/default.toml](configs/default.toml): release cutoff
`2023-06-01`, X-ray and EM structures, `1.0 ≤ resolution < 3.5 Å`,
and `350 < whole-complex residues < 2200`. Cluster representatives are selected
by resolution, then R-factor, then complex ID; `--smallest-repr` prioritizes size.
GPU mode produces the canonical dataset; CPU mode can produce different membership.

Use command-line options or supply a TOML profile with `--config`; command-line
values override the profile. Add `--resume` to reuse verified completed stages.
Run `abagbench --help` for commands and `abagbench <command> --help` for options.

## Citation and license

If you use AbAgBench, please cite:

> Venturini, M. (2026). *AbAgBench: Antibody–protein antigen complex structure prediction
> benchmark builder* (Version 0.1.0). PUXANO and Hasselt University.
> https://github.com/maven2306/AbAgBench

Citation metadata is available in [CITATION.cff](CITATION.cff).

Method references: [SAbDab2](https://doi.org/10.64898/2026.06.16.732554),
[SAbDab](https://doi.org/10.1093/nar/gkt1043),
[Foldseek-Multimer](https://doi.org/10.1038/s41592-025-02593-7),
[ANARCII](https://doi.org/10.1038/s42003-026-10186-z),
and [SciPy](https://doi.org/10.1038/s41592-019-0686-2).

Copyright 2026 PUXANO. Source code: [Apache-2.0](LICENSE); see [NOTICE](NOTICE).
SAbDab2 data are downloaded separately and remain subject to the
[wwPDB data usage policy](https://www.rcsb.org/pages/policies).
