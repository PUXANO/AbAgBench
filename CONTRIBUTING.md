# Contributing

## Before changing the pipeline

The default profile defines the selection order, scientific thresholds, tie-break rules, Foldseek scoring, clustering, representative selection, and canonical GPU device. Changes to those behaviors need a documented methodology decision, updated fixtures, and a new dataset version. Keep source row order where the current protocol uses it for ties.

## Development checks

Create the appropriate environment, install the repository in editable mode without resolving a second dependency set, and run `python -m pytest -q`. Add small redistributable fixtures for parser, filtering, Foldseek-report, clustering, or output changes; tests must not require a full SAbDab2 download.

For changes that affect run behavior, include the command used, platform and dependency versions, relevant stage counts, and fixture results. Do not describe CPU and GPU outputs as equivalent datasets.

## Data and generated files

SAbDab2 source files may be kept in the ignored root `sabdab2/` or `sabdab_database/` directory, or outside the repository. Do not commit downloaded structures, derived benchmark structures, run outputs, caches, local environment paths, or logs containing private infrastructure details. Review `git diff --check` and `git status --short` before opening a change.

## Reporting an issue

Include a minimal reproduction, the relevant sanitized run-manifest fields, the affected stage, and the error text. Remove access tokens and machine-specific paths before sharing logs. SAbDab2 licensing and access questions should be directed to the database maintainers.
