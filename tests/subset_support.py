"""Small completed release fixture with both supported antibody types."""
import csv
import json

from dataset_creation_pipeline.subset import artifact_record


def make_release(root):
    benchmark = root / "benchmark"
    (benchmark / "structures").mkdir(parents=True)
    registry, features, fasta = [], [], []
    for index in range(1, 7):
        light = "L" if index % 2 == 0 else ""
        target = f"{index}abc_H{light}A"
        chains = {"H": "AC", "A": "G" * index}
        if light:
            chains["L"] = "DE"
        size = sum(map(len, chains.values()))
        registry.append({"complex_id": target, "pdb": f"{index}abc", "Hchain": "H", "Lchain": light,
                         "antigen_chain": "A", "modality": "X-ray", "residue_length": str(size), "cdrh3_length": "1"})
        features.append({"complex_id": target, "pdb_id": f"{index}abc", "modality": "X-ray",
                         "is_nanobody": str(not light), "residue_length": str(size), "cdrh3_length": "1"})
        fasta.extend(f">{target}_{chain}|protein\n{seq}\n" for chain, seq in chains.items())
        (benchmark / "structures" / f"{target}.cif").write_text(f"data_{target}\n#\n")
    for name, rows, delimiter in (("registry.tsv", registry, "\t"), ("dataset_features.csv", features, ",")):
        with (benchmark / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter=delimiter)
            writer.writeheader()
            writer.writerows(rows)
    (benchmark / "benchmark.fasta").write_text("".join(fasta))
    refresh_manifest(root)
    return root


def refresh_manifest(root):
    manifest = {"schema_version": 1, "status": "completed", "execution": {"canonical": True},
                "stages": {str(stage): {"status": "completed", "outputs": [artifact_record(root, p) for p in paths]}
                    for stage, paths in ((6, ["benchmark/registry.tsv", "benchmark/benchmark.fasta", "benchmark/structures"]),
                                         (7, ["benchmark/dataset_features.csv"]))}}
    (root / "run_manifest.json").write_text(json.dumps(manifest))
