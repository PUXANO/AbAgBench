#!/usr/bin/env python3
"""Plot dataset construction summaries.

- Read training-cutoff metadata, filter-stage counts, and the final registry.
- Plot attrition, experimental modality, residue counts, resolution, and release-date groups.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .paths import BENCHMARK_DIR, INTERMEDIATE_ROOT, PLOTS_ROOT

PLOTS_DIR = PLOTS_ROOT
SUMMARY_TSV = BENCHMARK_DIR / "registry.tsv"
TRAINING_IDS_JSON = INTERMEDIATE_ROOT / "training_data_ids.json"
STAGE_METRICS_JSON = INTERMEDIATE_ROOT / "stage_metrics.json"

COLOR_COMBINED = "#4a7fb5"
COLOR_EM = "#e07a5f"
COLOR_XRAY = "#81b29a"
COLOR_TRAINING = "#e07a5f"
COLOR_NOT_IN_TRAINING = "#81b29a"


STAGE_LABELS = {
    "Raw SAbDab2 summary": "Raw SAbDab2 summary",
    "Reference-matched raw SAbDab2 summary": "Reference-matched SAbDab2 complexes\n(exact PDB and chain identity)",
    "Training data de-leak (pre-cutoff)": "Not in training complexes\n(PDB released on/after cutoff)",
    "H chain + single protein antigen": "H chain + single protein antigen\n(L chain optional \u2014 nanobodies)",
    "Engineered-construct removal": "Engineered-construct removal\n(scFv/diabody/CODV/DVD/SCDB)",
    "VNAR exclusion (shark IgNAR)": "VNAR exclusion\n(shark IgNAR)",
    "Method/resolution/R-factor/membrane": "Method (X-ray & EM only) + resolution + r-factor\n+ membrane",
    "SAbDab2 mmCIF exists": "SAbDab2 mmCIF exists",
    "Strict CIF eligibility": "Strict CIF eligibility",
    "Entity-copy dedup (one row per entity triple)": "Entity-copy dedup\n(one row per complex)",
    "Whole-complex residue bounds (350 < n < 2200)": "Whole-complex residue bounds\n(350\u20132200 residues)",
    "Case-collision dedup (identical chain copies)": "Case-collision dedup",
    "CDR loop coverage (Chothia 100%)": "CDR loop coverage\n(Chothia 100%, Anarcii)",
    "FoldSeek structural de-leak (I-LDDT>=0.7 & cov>=0.5)": "Structurally novel set\n(FoldSeek screen: I-LDDT\u22650.7 & cov\u22650.5)",
    "FoldSeek clustering (complete linkage, cut 0.3)": "Structurally independent set\n(complete-linkage clustering, cut 0.3)",
}

REQUIRED_FINAL_STAGES = {
    "FoldSeek structural de-leak (I-LDDT>=0.7 & cov>=0.5)",
    "FoldSeek clustering (complete linkage, cut 0.3)",
}


def load_training_ids() -> dict:
    """Load the training-cutoff metadata written by s1_training_data_check."""
    if not TRAINING_IDS_JSON.is_file():
        raise FileNotFoundError(
            f"Training PDB ID index not found: {TRAINING_IDS_JSON}\n"
            f"Run s1_training_data_check first."
        )
    with open(TRAINING_IDS_JSON) as f:
        return json.load(f)


def load_stage_metrics() -> list[dict]:
    """Load filter-stage complex counts; reject missing stage metrics."""
    if not STAGE_METRICS_JSON.is_file():
        raise FileNotFoundError(
            f"Stage metrics not found: {STAGE_METRICS_JSON}\n"
            f"Run s2 through s6 first."
        )
    with open(STAGE_METRICS_JSON) as f:
        data = json.load(f)
    stages = data.get("stages")
    if not isinstance(stages, list) or len(stages) == 0:
        raise ValueError(
            f"{STAGE_METRICS_JSON}: missing/empty 'stages' — re-run s2_quality_filter"
        )
    # Every stage records before/after/dropped integers.
    for s in stages:
        if not ({"stage", "before", "after", "dropped"} <= set(s)):
            raise RuntimeError(f"{STAGE_METRICS_JSON}: malformed stage record: {s!r}")
        s["before"], s["after"], s["dropped"] = (
            int(s["before"]), int(s["after"]), int(s["dropped"]),
        )
    stage_names = {s["stage"] for s in stages}
    missing = REQUIRED_FINAL_STAGES - stage_names
    if missing:
        raise RuntimeError(f"{STAGE_METRICS_JSON}: missing post-s6 stages: {sorted(missing)} — "
        f"run s5 and s6 before generating the plots")
    return stages


def plot_pipeline_funnel(
    stages: list[dict],
    n_final: int,
) -> None:
    """Plot surviving complex counts across the filtering stages."""
    last = stages[-1]
    if not (last["after"] == n_final):
        raise RuntimeError(f"stage_metrics.json final stage ({last['after']}) != registry rows "
        f"({n_final}) — pipeline and plots disagree, re-run the pipeline.")
    def label_for(stage: str) -> str:
        if stage.startswith("Whole-complex residue bounds ("):
            lower = int(os.environ.get("ABAG_MIN_RESIDUES", "350"))
            upper = int(os.environ.get("ABAG_MAX_RESIDUES", "2200"))
            return f"Whole-complex residue bounds\n({lower} < n < {upper})"
        if stage == "Method/resolution/R-factor/membrane":
            methods = os.environ.get("ABAG_METHODS", "xray,em").split(",")
            names = {"xray": "X-ray", "em": "EM"}
            method_label = " & ".join(names.get(method, method) for method in methods)
            res_max = float(os.environ.get("ABAG_RESOLUTION_MAX", "3.5"))
            r_max = float(os.environ.get("ABAG_XRAY_R_FACTOR_MAX", "0.3"))
            membrane = " + membrane exclusion" if os.environ.get("ABAG_EXCLUDE_MEMBRANE", "1") == "1" else ""
            return f"{method_label}; resolution < {res_max:g} Å; R-factor < {r_max:g}{membrane}"
        if stage not in STAGE_LABELS:
            raise ValueError(f"No plot label defined for pipeline stage {stage!r}")
        return STAGE_LABELS[stage]

    labels = [label_for(s["stage"]) for s in stages]
    totals = [s["after"] for s in stages]

    fig, ax = plt.subplots(figsize=(10, max(6.5, len(stages) * 0.75)))
    bar_h = 0.55

    for i, (label, tot) in enumerate(zip(labels, totals)):
        ax.barh(i, tot, height=bar_h, color=COLOR_COMBINED, zorder=2)
        ax.text(tot + 20, i, f"{tot:,}", va="center", fontsize=9.5, color="#222")

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Number of complexes (rows)", fontsize=11)
    ax.set_title("SAbDab filtering pipeline", fontsize=13, fontweight="bold")
    ax.set_xlim(0, max(totals) * 1.42)
    ax.grid(axis="x", alpha=0.3, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "plot_pipeline_funnel.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_final_em_xray(n_em: int, n_xray: int) -> None:
    """Plot X-ray and EM complex counts in the final registry."""
    fig, ax = plt.subplots(figsize=(5, 5))
    methods = ["EM", "X-ray"]
    counts = [n_em, n_xray]
    colors = [COLOR_EM, COLOR_XRAY]

    bars = ax.bar(methods, counts, color=colors, width=0.45, edgecolor="white")
    for bar, c in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
            f"{c:,}", ha="center", va="bottom", fontsize=12,
        )

    ax.set_ylabel("Number of complexes", fontsize=11)
    ax.set_title("Benchmark dataset: EM vs X-ray", fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(counts) * 1.15)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "plot_final_em_xray.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_residue_distribution(residue_length: pd.Series, n_em: int, n_xray: int) -> None:
    """Plot full-sequence complex residue counts in the final registry."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residue_length, bins=60, alpha=0.75, color=COLOR_COMBINED,
            edgecolor="white", label=f"All (n={len(residue_length):,})")

    min_res = int(os.environ.get("ABAG_MIN_RESIDUES", "350"))
    max_res = int(os.environ.get("ABAG_MAX_RESIDUES", "2200"))
    ax.axvline(min_res, color="#555", linestyle="--", linewidth=1.2, label=f"Min ({min_res} AA)")
    ax.axvline(max_res, color="#999", linestyle="--", linewidth=1.2, label=f"Max ({max_res} AA)")

    ax.set_xlabel("Total residues per complex (SEQRES, all chains)", fontsize=11)
    ax.set_ylabel("Number of complexes", fontsize=11)
    ax.set_title(f"Benchmark dataset residue distribution (EM {n_em:,} / X-ray {n_xray:,})",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "plot_residue_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_resolution_distributions(df_post_s6: pd.DataFrame) -> None:
    """Plot experimental resolution distributions for EM and X-ray complexes."""
    required = {"modality", "resolution"}
    if not (required <= set(df_post_s6.columns)):
        raise RuntimeError(f"Post-s6 registry missing columns for resolution plots: "
        f"{sorted(required - set(df_post_s6.columns))}")
    df_post_s6 = df_post_s6.copy()
    df_post_s6["resolution"] = pd.to_numeric(
        df_post_s6["resolution"], errors="coerce"
    )
    if not (df_post_s6["resolution"].notna().all()):
        raise RuntimeError("Post-s6 registry contains missing/non-numeric resolution values")

    for method, color, label_short in [
        ("EM", COLOR_EM, "EM"),
        ("X-ray", COLOR_XRAY, "X-ray"),
    ]:
        res = df_post_s6.loc[
            df_post_s6["modality"] == method, "resolution"
        ]
        if len(res) == 0:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(res, bins=25, color=color, edgecolor="white", alpha=0.85)
        res_min = float(os.environ.get("ABAG_RESOLUTION_MIN", "1.0"))
        res_max = float(os.environ.get("ABAG_RESOLUTION_MAX", "3.5"))
        ax.axvline(res_min, color="#555", linestyle=":", linewidth=1.2,
                   label=f"Quality-filter minimum: {res_min:g} Å")
        ax.axvline(
            res_max,
            color="crimson", linestyle="--", linewidth=1.5,
            label=f"Quality-filter upper bound: {res_max:g} Å",
        )

        ax.set_xlabel("Resolution (Å)", fontsize=11)
        ax.set_ylabel("Number of benchmark complexes", fontsize=11)
        ax.set_title(
            f"Benchmark {label_short}: resolution distribution (n={len(res):,})",
            fontsize=13, fontweight="bold",
        )
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"plot_resolution_{label_short}.png", dpi=150, bbox_inches="tight")
        plt.close()


def plot_training_overlap(
    cutoff: str,
    pre_cutoff: int,
    post_cutoff: int,
    raw_unique: int,
) -> None:
    """Plot pre- and post-cutoff counts of unique SAbDab2 PDB IDs."""
    categories = [
        "In training\n(PDB released before cutoff)",
        "Not in training\n(PDB released on/after cutoff)",
    ]
    counts = [pre_cutoff, post_cutoff]
    colors = [COLOR_TRAINING, COLOR_NOT_IN_TRAINING]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        categories, counts, color=colors, width=0.5,
        edgecolor="white",
    )
    for bar, c in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
            f"{c:,}", ha="center", va="bottom", fontsize=11,
        )

    ax.set_ylabel("Unique PDB IDs in raw SAbDab", fontsize=11)
    ax.set_title(f"Training cutoff split (cutoff {cutoff}, {raw_unique:,} unique PDB IDs)",
                 fontsize=12, fontweight="bold")
    ax.legend([], frameon=False)
    ax.set_ylim(0, max(counts) * 1.15)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "plot_training_overlap.png", dpi=150, bbox_inches="tight")
    plt.close()


def generate_all() -> None:
    """Generate dataset summary plots from the final representative registry."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    training_data = load_training_ids()
    cutoff = training_data.get("cutoff")
    raw_unique = training_data.get("raw_unique_pdb")
    if not (cutoff and raw_unique):
        raise RuntimeError(f"training_data_ids.json missing cutoff/raw_unique keys — "
        f"re-run s1_training_data_check")

    registry = pd.read_csv(SUMMARY_TSV, sep="\t", low_memory=False)
    if not (len(registry) > 0):
        raise RuntimeError(f"Registry is empty: {SUMMARY_TSV} — run the pipeline first.")
    required_registry = {"modality", "residue_length", "resolution"}
    if not (required_registry <= set(registry.columns)):
        raise RuntimeError(f"Post-s6 registry missing required plot columns: "
        f"{sorted(required_registry - set(registry.columns))}")
    n_final_em = int((registry["modality"] == "EM").sum())
    n_final_xray = int((registry["modality"] == "X-ray").sum())
    if not (n_final_em + n_final_xray == len(registry)):
        raise RuntimeError(f"Unexpected modality values in post-s6 registry: "
        f"{registry['modality'].value_counts(dropna=False).to_dict()}")

    stages = load_stage_metrics()
    plot_pipeline_funnel(stages, len(registry))

    plot_resolution_distributions(registry)

    plot_final_em_xray(n_final_em, n_final_xray)
    plot_residue_distribution(registry["residue_length"], n_final_em, n_final_xray)

    pre_cutoff = training_data["pre_cutoff_unique_pdb"]
    post_cutoff = raw_unique - pre_cutoff
    plot_training_overlap(
        cutoff=cutoff,
        pre_cutoff=pre_cutoff,
        post_cutoff=post_cutoff,
        raw_unique=raw_unique,
    )

    print(f"\n  All plots saved to {PLOTS_DIR}/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate dataset construction plots.")
    parser.parse_args()
    generate_all()


if __name__ == "__main__":
    main()
