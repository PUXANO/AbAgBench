"""Provide shared SAbDab2 sequence, structure, and filtering utilities.

- Read summary records and IMGT-numbered mmCIF structures.
- Extract full entity sequences and resolved residue positions.
- Resolve equivalent chain copies and write chain subsets.
- Apply eligibility masks and derive Chothia CDR features with Anarcii.
- Record filter-stage counts.
"""
from __future__ import annotations

import logging
import os
import re
import shlex
import tempfile
import json
import gzip
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

from .paths import (
    DATA_ROOT,
    INTERMEDIATE_ROOT,
    SABDAB2_DATA,
    SABDAB2_SUMMARY_CSV,
    SABDAB2_STRUCTURE_DIR,
)

logger = logging.getLogger(__name__)


def atomic_write_json(path: Path, value: dict) -> None:
    """Write JSON beside its destination and atomically publish it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def atomic_write_dataframe(df: pd.DataFrame, path: Path, **kwargs) -> None:
    """Write a table to a temporary sibling and atomically publish it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        df.to_csv(temp_name, **kwargs)
        with open(temp_name, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise

LOWER_RESIDUES = int(os.environ.get("ABAG_MIN_RESIDUES", "350"))
UPPER_RESIDUES = int(os.environ.get("ABAG_MAX_RESIDUES", "2200"))

# ── Non-standard residue mapping ─────────────────────────────────────────────
# Common modified amino acids whose 3-letter component IDs appear in
# _entity_poly_seq.mon_id.  Without this mapping they become 'X', which
# silently degrades SEQRES-vs-ATOM calculations.  
NONSTANDARD_MAP: dict[str, str] = {
    "MSE": "M",   # selenomethionine  → methionine
    "SEC": "U",   # selenocysteine
    "CSE": "C",   # selenocysteine variant
    "HYP": "P",   # hydroxyproline
    "TPO": "T",   # phosphothreonine
    "TYS": "Y",   # O-sulfo-L-tyrosine (tyrosine O-sulfate)
    "SEP": "S",   # phosphoserine
    "PTR": "Y",   # phosphotyrosine
    "PCA": "Q",   # pyroglutamate (cyclized N-terminal glutamine)
    "CSO": "C",   # cysteine sulfoxide (oxidized cysteine)
    "HIP": "H",   # doubly protonated histidine
}


def residue_to_one_letter(resname: str) -> str:
    """Convert a residue name using standard and modified-residue mappings; return X if unknown."""
    from Bio.PDB.Polypeptide import protein_letters_3to1
    return protein_letters_3to1.get(resname, NONSTANDARD_MAP.get(resname, "X"))


# ── SAbDab2 summary handling ─────────────────────────────────────────────────

PDB_BASE_RE = re.compile(r"^(?:pdb_0000)?([0-9a-z]{4})$")



def normalize_sabdab2_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize summary columns and lowercase four-character PDB IDs; reject invalid IDs."""
    df = df.copy()
    if "pdb" not in df.columns:
        if "PDB" not in df.columns:
            raise ValueError(
                f"SAbDab2 summary has neither 'pdb' nor 'PDB' column; got "
                f"{sorted(df.columns)[:12]}..."
            )
        df["pdb"] = df["PDB"]
    df["pdb"] = df["pdb"].astype(str).str.strip().str.lower()
    base = df["pdb"].str.extract(PDB_BASE_RE)[0]
    bad = df["pdb"].isna() | base.isna()
    if bad.any():
        raise ValueError(
            f"{int(bad.sum())} SAbDab2 rows have unparseable PDB IDs, e.g. "
            f"{sorted(df.loc[bad, 'pdb'].unique())[:5]}"
        )
    df["pdb"] = base
    return df


# ── Chain IDs & complex IDs ─────────────────────────────────────────────────

def chain_ids_from_row(row) -> list[str]:
    """Return heavy, optional light, and antigen chain IDs in order."""
    return [
        str(c).strip()
        for c in (row["Hchain"], row["Lchain"], row["antigen_chain"])
        if pd.notna(c)
    ]


def complex_id_from_row(row) -> str:
    """Build '<pdb>_<HLA>' from single-character chain IDs; omit L for nanobodies."""
    ids = chain_ids_from_row(row)
    if not (len(ids) in (2, 3)):
        raise RuntimeError(f"Row for {row.get('pdb')}: expected 2 or 3 chains, got {ids}")
    bad = [c for c in ids if len(c) != 1]
    if bad:
        raise ValueError(
            f"Row for {row.get('pdb')}: multi-char chain IDs {bad} reached "
            f"complex_id construction — strict CIF eligibility must drop them"
        )
    return f"{str(row['pdb']).strip().lower()}_{''.join(ids)}"


# ── SAbDab2 mmCIF parsing ────────────────────────────────────────────────────

def pdb_to_cif_path(pdb_id: str) -> Path:
    """Find an extracted or gzip-compressed SAbDab2 CIF for a PDB ID."""
    pdb_id = pdb_id.lower()
    if not PDB_BASE_RE.match(pdb_id):
        raise ValueError(f"Not a 4-character PDB ID: {pdb_id!r}")
    stem = f"pdb_0000{pdb_id}"
    if SABDAB2_STRUCTURE_DIR is None:
        raise FileNotFoundError("SAbDab2 structure directory is not configured; pass --sabdab-dir")
    candidates = (
        SABDAB2_STRUCTURE_DIR / stem / f"{stem}_sabdab.cif",
        SABDAB2_STRUCTURE_DIR / stem / f"{stem}_sabdab.cif.gz",
        SABDAB2_STRUCTURE_DIR / f"{stem}_sabdab.cif",
        SABDAB2_STRUCTURE_DIR / f"{stem}_sabdab.cif.gz",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def materialize_cif(src_path: Path, dst_path: Path) -> None:
    """Write a source CIF to an uncompressed `.cif` path for FoldSeek."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if src_path.name.endswith(".cif.gz"):
        with gzip.open(src_path, "rb") as source, dst_path.open("wb") as destination:
            shutil.copyfileobj(source, destination, length=1024 * 1024)
    else:
        shutil.copy2(src_path, dst_path)


@lru_cache(maxsize=8)
def parse_sabdab_cif(cif_path: Path) -> dict:
    """Read full entity sequences, chain identities, and resolved positions from mmCIF.

    - Return poly_seq and chains mappings.
    - Require sequential entity positions and consistent author/label chain mappings.
    - Keep a small LRU because stages often inspect multiple chains in one CIF.
    """
    if not cif_path.is_file():
        raise FileNotFoundError(f"SAbDab2 CIF not found: {cif_path}")

    poly_seq: dict[str, list[str]] = {}
    label_to_entity: dict[str, str] = {}
    auth_to_labels: dict[str, set] = {}
    resolved_by_label: dict[str, set] = {}

    current_loop: Optional[str] = None
    headers: list[str] = []
    saw_atom_site = False
    saw_poly_seq = False
    in_text_field = False

    opener = gzip.open if cif_path.name.endswith(".gz") else open
    with opener(cif_path, "rt") as f:
        lines = f.readlines()

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].rstrip("\n")
        stripped = line.strip()

        # Semicolon-delimited text fields may contain arbitrary prose, quotes,
        # and tokens such as ``loop_``.  None of the three structural loops we
        # consume uses multiline values, so skip these metadata blocks without
        # trying to tokenize their contents.
        if line.startswith(";"):
            in_text_field = not in_text_field
            current_loop = None
            i += 1
            continue
        if in_text_field:
            i += 1
            continue

        if line.startswith("loop_"):
            # Read the loop header (the _tag lines that follow).
            headers = []
            j = i + 1
            while j < n and lines[j].startswith("_"):
                headers.append(lines[j].strip().split(".")[0])
                j += 1
            if not headers:
                raise ValueError(f"{cif_path}: empty loop header after 'loop_'")
            if headers[0] == "_entity_poly_seq":
                current_loop, saw_poly_seq = "poly_seq", True
            elif headers[0] == "_struct_asym":
                current_loop = "struct_asym"
            elif headers[0] == "_atom_site":
                current_loop, saw_atom_site = "atom_site", True
            else:
                current_loop = None
            i = j
            continue

        if line.startswith("_") or not stripped:
            # Single-value section or blank line: not a data row.
            current_loop = None
            i += 1
            continue

        # Most SAbDab mmCIF metadata is irrelevant to this parser.  Applying
        # shell quoting rules to citation, title, compound, or symmetry text
        # can reject the entire structure. Tokenize only loops we consume.
        if current_loop is None:
            i += 1
            continue

        try:
            parts = shlex.split(stripped, comments=False, posix=True)
        except ValueError as exc:
            raise ValueError(f"{cif_path}: malformed quoted mmCIF row: {line!r}") from exc
        if current_loop == "poly_seq" and len(parts) == 4:
            entity_id, num, mon_id, _hetero = parts
            poly_seq.setdefault(entity_id, []).append((int(num), mon_id))
        elif current_loop == "struct_asym" and len(parts) == 2:
            label_asym_id, entity_id = parts
            label_to_entity[label_asym_id] = entity_id
        elif current_loop == "atom_site" and len(parts) >= 21:
            group = parts[0]
            label_asym_id = parts[6]
            label_seq_id = parts[8]
            auth_asym_id = parts[19]
            if group in ("ATOM", "HETATM"):
                auth_to_labels.setdefault(auth_asym_id, set()).add(label_asym_id)
                if group == "ATOM":
                    if label_seq_id in (".", "?"):
                        raise ValueError(
                            f"{cif_path}: ATOM row without numeric label_seq_id "
                            f"for auth chain {auth_asym_id!r}: {line!r}"
                        )
                    resolved_by_label.setdefault(label_asym_id, set()).add(
                        int(label_seq_id)
                    )
        i += 1

    if not saw_poly_seq:
        raise ValueError(f"{cif_path}: no _entity_poly_seq loop found (malformed CIF)")
    if not saw_atom_site:
        raise ValueError(f"{cif_path}: no _atom_site loop found (malformed CIF)")
    if not poly_seq:
        raise ValueError(f"{cif_path}: _entity_poly_seq is empty")
    if not resolved_by_label:
        raise ValueError(f"{cif_path}: no polymer ATOM records found")

    # Require unique sequential entity positions (1..N).
    for entity_id, rows in poly_seq.items():
        seen: dict[int, str] = {}
        for num, mon_id in rows:
            if num in seen:
                raise ValueError(
                    f"{cif_path}: entity {entity_id} lists position {num} "
                    f"twice ({seen[num]} and {mon_id}) — duplicate-position "
                    f"file, must be dropped by strict CIF eligibility"
                )
            seen[num] = mon_id
        nums = sorted(seen)
        if nums != list(range(1, len(nums) + 1)):
            raise ValueError(
                f"{cif_path}: entity {entity_id} has non-contiguous positions "
                f"{nums[:6]}{'...' if len(nums) > 6 else ''}"
            )
        poly_seq[entity_id] = [seen[n] for n in nums]

    # One author chain may be split into several label chains by SAbDab2
    # (e.g. scFv segments, or phantom segments).  The chain a complex refers
    # to is the label chain with resolved atoms; when several label chains
    # have atoms under one author chain they are identical copies — pick the
    # one with the most resolved residues (ties -> lexicographically smallest
    # label id, deterministic).  Author chains with NO resolved atoms are
    # un-modeled segments: they are omitted here, and a caller requesting one
    # is resolved to a same-entity copy by `resolve_chain` (or raises if no
    # copy exists).
    chains = {}
    for auth_asym_id, labels in auth_to_labels.items():
        with_atoms = [l for l in labels if resolved_by_label.get(l)]
        if not with_atoms:
            continue
        best_label = sorted(
            with_atoms,
            key=lambda l: (-len(resolved_by_label[l]), l),
        )[0]
        entity_id = label_to_entity.get(best_label)
        if entity_id is None:
            raise ValueError(
                f"{cif_path}: label chain {best_label!r} has no "
                f"_struct_asym entity mapping"
            )
        chains[auth_asym_id] = {
            "label_asym_id": best_label,
            "entity_id": entity_id,
            "resolved": resolved_by_label[best_label],
        }

    return {
        "poly_seq": poly_seq,
        "chains": chains,
        # struct_asym label_id -> entity_id for EVERY label (incl. the ones
        # with zero atoms).  Lets an un-modeled auth chain (auth == label,
        # e.g. pdb_000010bt 'I') be mapped to its entity so a resolved
        # same-entity copy can be substituted by `resolve_chain`.
        "entity_id_of": label_to_entity,
    }


def resolve_chain(data: dict, chain: str) -> Optional[dict]:
    """Return a resolved chain entry or its most resolved same-entity copy; return None if unavailable."""
    own = data["chains"].get(chain)
    if own is not None:
        return own
    entity = data["entity_id_of"].get(chain)
    if entity is None:
        return None
    candidates = [
        info for info in data["chains"].values() if info["entity_id"] == entity
    ]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda c: (-len(c["resolved"]), c["label_asym_id"]),
    )[0]


# ── Chain resolution / sequences ────────────────────────────────────────────
# An un-modeled requested chain (no coordinates) is resolved to the most
# resolved same-entity copy via `resolve_chain`.  Two requested chains that
# share ONE entity are copies or split segments of the same polymer — the
# entity sequence describes the copy, not each segment (pdb_00008pmy scFv),
# so s2's strict CIF eligibility DROPS those complexes; anyone reaching
# sequence construction with a shared entity aborts loudly below.


def chain_full_sequences(cif_path: Path, chains: set[str]) -> dict[str, str]:
    """Return full sequences, including unresolved residues; reject missing chains or shared entities."""
    data = parse_sabdab_cif(cif_path)
    out: dict[str, str] = {}
    seen_entities: set[str] = set()
    for chain in sorted(chains):
        info = resolve_chain(data, chain)
        if info is None:
            raise ValueError(
                f"{cif_path}: chain {chain!r} has no coordinates and no "
                f"resolved same-entity copy — cannot build its sequence"
            )
        entity_id = info["entity_id"]
        if entity_id in seen_entities:
            raise ValueError(
                f"{cif_path}: chains {sorted(chains)} include two chains of "
                f"the same entity ({entity_id}) — split or copy chains; must "
                f"be dropped by s2's strict CIF eligibility"
            )
        seen_entities.add(entity_id)
        mon_ids = data["poly_seq"].get(entity_id)
        if not mon_ids:
            raise ValueError(
                f"{cif_path}: entity {entity_id} (chain {chain}) has no "
                f"_entity_poly_seq sequence"
            )
        out[chain] = "".join(residue_to_one_letter(m) for m in mon_ids)
    return out


def complex_residue_count(pdb_id: str, chains) -> int:
    """Sum full-sequence residue counts across the requested chains, including unresolved residues."""
    cif_path = pdb_to_cif_path(pdb_id)
    seqs = chain_full_sequences(cif_path, set(chains))
    return sum(len(s) for s in seqs.values())


def resolved_positions(cif_path: Path, auth_chain: str) -> set[int]:
    """Return one-based resolved sequence positions, using a same-entity copy when necessary."""
    data = parse_sabdab_cif(cif_path)
    chain = resolve_chain(data, auth_chain)
    if chain is None:
        raise ValueError(
            f"{cif_path}: chain {auth_chain!r} has no coordinates and no "
            f"resolved same-entity copy"
        )
    return chain["resolved"]


# ── Chain-subset CIF extraction (for all-atom foldseek staging) ─────────────

def write_chain_subset_cif(src_path: Path, dst_path: Path, auth_chains: set[str]) -> None:
    """Write an mmCIF with atom records restricted to the requested chains.

    - Preserve other mmCIF sections.
    - Substitute unresolved chains with resolved same-entity copies.
    - Require at least one retained ATOM record.
    """
    data = parse_sabdab_cif(src_path)
    label_ids = set()
    for c in auth_chains:
        info = resolve_chain(data, c)
        if info is None:
            raise ValueError(
                f"{src_path}: target chain {c!r} has no coordinates and no "
                f"resolved same-entity copy"
            )
        label_ids.add(info["label_asym_id"])

    opener = gzip.open if src_path.name.endswith(".gz") else open
    with opener(src_path, "rt") as f:
        lines = f.readlines()

    out_lines: list[str] = []
    in_atom_loop = False
    kept_atoms = 0
    for line in lines:
        if line.startswith("_atom_site."):
            in_atom_loop = True
            out_lines.append(line)
            continue
        if in_atom_loop:
            if line.startswith("loop_") or (
                line.startswith("_") and not line.startswith(" ")
            ):
                in_atom_loop = False
                out_lines.append(line)
                continue
            if line.strip():
                parts = line.split()
                if parts and parts[0] in ("ATOM", "HETATM"):
                    if parts[6] in label_ids:
                        out_lines.append(line)
                        if parts[0] == "ATOM":
                            kept_atoms += 1
                    continue
            else:
                out_lines.append(line)
            continue
        out_lines.append(line)

    if kept_atoms == 0:
        raise ValueError(
            f"{src_path}: no ATOM record kept for chains {sorted(auth_chains)}"
        )

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_path, "w") as f:
        f.writelines(out_lines)


# ── SAbDab2 filter masks (shared: s2 quality filter AND plots) ────────────────
# Single source of truth so plots.py can never drift from the filter again.

def make_abag_mask(df: pd.DataFrame) -> pd.Series:
    """Antibody-antigen selection mask (H chain, single protein antigen)."""
    mask_has_H = df["Hchain"].notna()
    mask_has_antigen = df["antigen_chain"].notna()
    mask_multimeric = df["antigen_chain"].str.contains("|", regex=False, na=False)
    mask_protein = df["antigen_type"].str.contains("protein", case=False, na=False)
    return mask_has_H & mask_has_antigen & ~mask_multimeric & mask_protein


# Excluded engineered chain-sharing constructs in SAbDab2.
EXCLUDED_CONSTRUCTS = {"SCFV", "DIABODY", "CODV", "DVD", "SCDB", "OTHER"}


def make_scfv_mask(df: pd.DataFrame) -> pd.Series:
    """Identify engineered constructs from SAbDab2 annotations, shared chain IDs, and compound names."""
    required = {"chainsharing_construct", "compound"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Required SAbDab2 columns missing for construct filter: "
            f"{sorted(missing_cols)}"
        )
    mask_construct = df["chainsharing_construct"].isin(EXCLUDED_CONSTRUCTS)
    mask_scfv_heuristic = (
        df["Lchain"].notna()
        & (df["Hchain"].str.upper() == df["Lchain"].str.upper())
    )
    mask_scfv_compound = df["compound"].astype(str).str.contains(
        r"scfv", case=False, regex=True, na=False
    )
    return mask_construct | mask_scfv_heuristic | mask_scfv_compound


def make_quality_masks(
    df: pd.DataFrame,
    resolution_min: float | None = None,
    resolution_max: float | None = None,
    r_factor_max: float | None = None,
    exclude_membrane: bool | None = None,
    methods: tuple[str, ...] | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Return X-ray and EM masks using the resolved run selection settings."""
    resolution_min = float(os.environ.get("ABAG_RESOLUTION_MIN", "1.0")) if resolution_min is None else resolution_min
    resolution_max = float(os.environ.get("ABAG_RESOLUTION_MAX", "3.5")) if resolution_max is None else resolution_max
    r_factor_max = float(os.environ.get("ABAG_XRAY_R_FACTOR_MAX", "0.3")) if r_factor_max is None else r_factor_max
    exclude_membrane = os.environ.get("ABAG_EXCLUDE_MEMBRANE", "1") == "1" if exclude_membrane is None else exclude_membrane
    methods = tuple(x.lower() for x in os.environ.get("ABAG_METHODS", "xray,em").split(",") if x) if methods is None else methods
    res = pd.to_numeric(df["resolution"], errors="coerce")
    r_factor = pd.to_numeric(df["r_factor"], errors="coerce")
    not_membrane = (
        ~df["short_header"].str.contains("MEMBRANE", case=False, na=False)
        if exclude_membrane else pd.Series(True, index=df.index)
    )
    mask_xray = (
        (df["method"] == "XRAY")
        & (res < resolution_max) & (res >= resolution_min)
        & (r_factor < r_factor_max) & not_membrane & ("xray" in methods)
    )
    mask_em = (
        (df["method"] == "ELECTRON_MICROSCOPY")
        & (res < resolution_max) & (res >= resolution_min)
        & not_membrane & ("em" in methods)
    )
    return mask_xray, mask_em


# Chothia CDR intervals include insertion codes within each numbered range.
CHOTHIA_CDR_RANGES: dict[str, list[tuple[int, int]]] = {
    "H": [(26, 32), (52, 56), (96, 101)],
    "L": [(26, 32), (50, 52), (91, 96)],
}
CDRH3_RANGE: tuple[int, int] = (96, 101)
# ANARCII 2.0.8 converts IMGT 105–117 (including insertions, excluding gaps)
# as one region; chothia_heavy returns [] above this length. This is broader
# than our Chothia 96–101 feature range, so do not apply the limit to that feature.
IMGT_CDRH3_RANGE: tuple[int, int] = (105, 117)
CHOTHIA_MAX_CDRH3_LENGTH = 36
LONG_CDRH3_REASON = "CDRH3 exceeds ANARCII Chothia conversion limit (36 residues)"

# Anarcii singleton — loaded lazily, shared across the process.
_ANARCII_MODEL = None  # type: ignore
_ANARCII_VERSION_REQUIRED = "2.0.8"


def _get_anarcii_model():
    """Load and cache Anarcii 2.0.8 in CPU accuracy mode; reject missing or incompatible versions."""
    global _ANARCII_MODEL
    if _ANARCII_MODEL is not None:
        return _ANARCII_MODEL
    try:
        import anarcii  # noqa: F401
        from anarcii import Anarcii
    except ImportError as e:
        raise ImportError(
            "anarcii is required for Chothia numbering (CDR filter + cdrh3_length). "
            "Install the pinned version: pip install anarcii==2.0.8"
        ) from e
    import importlib.metadata as _im

    try:
        ver = _im.version("anarcii")
    except Exception:
        ver = "unknown"
    if ver != _ANARCII_VERSION_REQUIRED:
        raise RuntimeError(
            f"anarcii version mismatch: required {_ANARCII_VERSION_REQUIRED!r}, "
            f"found {ver!r}.  Pin the version for reproducibility: "
            f"pip install anarcii=={_ANARCII_VERSION_REQUIRED}"
        )
    _ANARCII_MODEL = Anarcii(
        seq_type="antibody",
        mode="accuracy",
        batch_size=int(os.environ.get("ABAG_ANARCII_BATCH_SIZE", "32")),
        cpu=True,
        verbose=False,
    )
    return _ANARCII_MODEL


def _chothia_cdr_indices_from_numbering(
    numbering: list[tuple[tuple[int, str], str]],
    seq: str,
    chain_role: str,
    query_start: int | None = None,
    query_end: int | None = None,
) -> set[int]:
    """Map Chothia CDR positions to zero-based full-sequence indices, accounting for V-domain offsets."""
    if chain_role not in ("H", "L"):
        raise ValueError(f"chain_role must be 'H' or 'L', got {chain_role!r}")
    ranges = CHOTHIA_CDR_RANGES[chain_role]
    # Determine the V-domain slice that Anarcii actually numbered.
    if query_start is None or query_end is None:
        v_seq = seq
        offset = 0
    else:
        if not (0 <= query_start <= query_end < len(seq)):
            raise ValueError(
                f"Invalid query_start/query_end {query_start}/{query_end} "
                f"for seq len {len(seq)} (chain_role={chain_role!r})"
            )
        v_seq = seq[query_start : query_end + 1]
        offset = query_start
    cdr_indices: set[int] = set()
    v_idx = 0
    for (pos, icode), aa in numbering:
        if aa == "-":
            continue
        if v_idx >= len(v_seq):
            raise ValueError(
                f"Numbering longer than V domain ({len(v_seq)}): "
                f"extra aa {aa!r} at Chothia {pos}{icode.strip()}"
            )
        if aa != v_seq[v_idx]:
            raise ValueError(
                f"Sequence/numbering mismatch at V idx {v_idx} (full {offset+v_idx}): "
                f"seq {v_seq[v_idx]!r} != numbering {aa!r} "
                f"(Chothia {pos}{icode.strip()}, chain_role={chain_role!r})"
            )
        for lo, hi in ranges:
            if lo <= pos <= hi:
                cdr_indices.add(offset + v_idx)
                break
        v_idx += 1
    if v_idx != len(v_seq):
        raise ValueError(
            f"Numbering shorter than V domain: numbering covers {v_idx} residues "
            f"but V domain length is {len(v_seq)} (chain_role={chain_role!r})"
        )
    return cdr_indices


def get_chothia_cdr_indices_batch(
    seqs: dict[str, str],
    chain_roles: dict[str, str],
) -> dict[str, set[int]]:
    """Return zero-based full-sequence CDR indices per antibody; reject numbering failures or ambiguity."""
    if set(seqs) != set(chain_roles):
        raise ValueError(
            f"seqs and chain_roles keys mismatch: "
            f"seqs {sorted(seqs)[:5]}... vs roles {sorted(chain_roles)[:5]}..."
        )
    if not seqs:
        return {}
    for name, role in chain_roles.items():
        if role not in ("H", "L"):
            raise ValueError(f"chain_roles[{name!r}] must be 'H' or 'L', got {role!r}")
        if not seqs[name]:
            raise ValueError(f"Empty sequence for {name!r}")

    model = _get_anarcii_model()
    # Anarcii's number() mutates internal state; to_scheme converts the last
    # result.  We call once for the whole batch for efficiency.
    model.number(seqs)
    chothia = model.to_scheme("chothia")
    if not isinstance(chothia, dict):
        raise RuntimeError(f"Anarcii to_scheme('chothia') returned {type(chothia)}, expected dict")

    out: dict[str, set[int]] = {}
    for name, seq in seqs.items():
        entry = chothia.get(name)
        if entry is None:
            raise RuntimeError(f"Anarcii returned no entry for {name!r}")
        if entry.get("error") is not None:
            raise RuntimeError(
                f"Anarcii failed to number {name!r} (role {chain_roles[name]!r}, "
                f"len {len(seq)}): {entry['error']}"
            )
        numbering = entry.get("numbering")
        if numbering is None:
            raise RuntimeError(f"Anarcii entry for {name!r} has no 'numbering'")
        qs = entry.get("query_start")
        qe = entry.get("query_end")
        # Anarcii may return None for qs/qe when it numbers the whole seq (e.g. VHH)
        # — in that case the helper treats the V domain as the full seq.
        if qs is None or qe is None:
            logger.warning(
                "Anarcii returned None query_start/query_end for %r (len %d, role %r) — "
                "assuming V domain == full seq",
                name, len(seq), chain_roles[name],
            )
        # Optional warning if Anarcii's predicted chain_type disagrees with SAbDab role.
        pred_type = entry.get("chain_type")
        expected = chain_roles[name]
        if pred_type == "K":
            pred_type = "L"  # Kappa is a light chain; treat as L for comparison
        if pred_type not in (expected, None):
            # Don't abort — SAbDab's H/L assignment is authoritative — but log
            # loudly so a systematic mis-numbering is not silent.
            logger.warning(
                "Anarcii predicted chain_type %r for %r (role %r, seq len %d) — "
                "using SAbDab role %r for CDR ranges",
                entry.get("chain_type"), name, expected, len(seq), expected,
            )
        out[name] = _chothia_cdr_indices_from_numbering(
            numbering, seq, expected, query_start=qs, query_end=qe
        )
    return out


def get_chothia_cdr_indices(seq: str, chain_role: str) -> set[int]:
    """Single-sequence wrapper around :func:`get_chothia_cdr_indices_batch`."""
    name = "__single__"
    batch = get_chothia_cdr_indices_batch({name: seq}, {name: chain_role})
    return batch[name]


def get_chothia_details_batch(
    seqs: dict[str, str],
    chain_roles: dict[str, str],
) -> tuple[dict[str, set[int]], dict[str, int], dict[str, dict]]:
    """Return CDR indices, Chothia H3 lengths, and confirmed long-H3 exclusions.

    Exclusions use the original IMGT numbering, never empty Chothia CDR sets.
    All other numbering failures and sequence inconsistencies remain errors.
    """
    if set(seqs) != set(chain_roles):
        raise ValueError(f"seqs and chain_roles keys mismatch")
    if not seqs:
        return {}, {}, {}
    for name, role in chain_roles.items():
        if role not in ("H", "L"):
            raise ValueError(f"chain_roles[{name!r}] must be 'H' or 'L', got {role!r}")
        if not seqs[name]:
            raise ValueError(f"Empty sequence for {name!r}")

    model = _get_anarcii_model()
    imgt = model.number(seqs)
    if not isinstance(imgt, dict):
        raise RuntimeError(f"Anarcii number() returned {type(imgt)}, expected dict")
    exclusions: dict[str, dict] = {}
    for name, seq in seqs.items():
        original = imgt.get(name)
        if original is None or original.get("error") is not None or not original.get("numbering"):
            raise RuntimeError(f"Anarcii failed to number {name!r} in IMGT: {original}")
        if chain_roles[name] != "H" or original.get("chain_type") != "H":
            continue
        qs, qe = original.get("query_start"), original.get("query_end")
        if qs is None or qe is None:
            v_seq = seq
        else:
            if not (0 <= qs <= qe < len(seq)):
                raise ValueError(f"Invalid IMGT query_start/query_end {qs}/{qe} for {name!r}")
            v_seq = seq[qs : qe + 1]
        numbering = original["numbering"]
        if "".join(aa for _, aa in numbering if aa != "-") != v_seq:
            raise ValueError(f"IMGT sequence/numbering mismatch for {name!r}")
        imgt_lo, imgt_hi = IMGT_CDRH3_RANGE
        h3_length = sum(aa != "-" and imgt_lo <= pos <= imgt_hi for (pos, _), aa in numbering)
        if h3_length > CHOTHIA_MAX_CDRH3_LENGTH:
            exclusions[name] = {
                "reason_code": "cdrh3_too_long",
                "reason": LONG_CDRH3_REASON,
                "cdrh3_length_imgt": h3_length,
                "max_supported_length": CHOTHIA_MAX_CDRH3_LENGTH,
                "query_start": qs,
                "query_end": qe,
            }
    chothia = model.to_scheme("chothia")
    if not isinstance(chothia, dict):
        raise RuntimeError(f"Anarcii to_scheme('chothia') returned {type(chothia)}")

    cdr_by_key: dict[str, set[int]] = {}
    h3_by_key: dict[str, int] = {}
    lo, hi = CDRH3_RANGE
    for name, seq in seqs.items():
        entry = chothia.get(name)
        if entry is None or entry.get("error") is not None:
            raise RuntimeError(f"Anarcii failed to number {name!r}: {entry}")
        if name in exclusions:
            continue
        numbering = entry["numbering"]
        if not numbering:
            raise RuntimeError(f"Anarcii returned empty Chothia numbering for {name!r} without a confirmed long CDRH3")
        qs = entry.get("query_start")
        qe = entry.get("query_end")
        if qs is None or qe is None:
            logger.warning(
                "Anarcii returned None query_start/query_end for %r — assuming full seq",
                name,
            )
            v_seq = seq
            offset = 0
        else:
            if not (0 <= qs <= qe < len(seq)):
                raise ValueError(f"Invalid query_start/query_end {qs}/{qe} for {name!r} len {len(seq)}")
            v_seq = seq[qs : qe + 1]
            offset = qs
        # One pass over numbering to collect both CDR and H3
        cdr_set: set[int] = set()
        h3_count = 0
        v_idx = 0
        role = chain_roles[name]
        cdr_ranges = CHOTHIA_CDR_RANGES[role]
        for (pos, icode), aa in numbering:
            if aa == "-":
                continue
            if v_idx >= len(v_seq):
                raise ValueError(f"Numbering longer than V domain for {name!r}")
            if aa != v_seq[v_idx]:
                raise ValueError(f"Mismatch at V idx {v_idx} for {name!r}: {v_seq[v_idx]!r} != {aa!r}")
            for clo, chi in cdr_ranges:
                if clo <= pos <= chi:
                    cdr_set.add(offset + v_idx)
                    break
            if role == "H" and lo <= pos <= hi:
                h3_count += 1
            v_idx += 1
        if v_idx != len(v_seq):
            raise ValueError(f"Numbering length mismatch for {name!r}: {v_idx} vs {len(v_seq)}")
        cdr_by_key[name] = cdr_set
        if role == "H":
            h3_by_key[name] = h3_count
        # Validate predicted chain_type
        pred = entry.get("chain_type")
        expected = role
        if pred == "K":
            pred = "L"
        if pred not in (expected, None):
            logger.warning(
                "Anarcii predicted %r for %r (role %r) — using SAbDab role",
                entry.get("chain_type"), name, expected, expected,
            )
    return cdr_by_key, h3_by_key, exclusions


def get_cdrh3_length(seq: str) -> int:
    """Return heavy-chain CDRH3 length at Chothia positions 96–101, including insertions."""
    if not seq:
        raise ValueError("Empty sequence for CDRH3 length")
    model = _get_anarcii_model()
    name = "__cdrh3__"
    model.number({name: seq})
    chothia = model.to_scheme("chothia")
    entry = chothia.get(name)
    if entry is None or entry.get("error") is not None:
        raise RuntimeError(f"Anarcii failed to number CDRH3 sequence (len {len(seq)}): {entry}")
    numbering = entry["numbering"]
    qs = entry.get("query_start")
    qe = entry.get("query_end")
    if qs is None or qe is None:
        v_seq = seq
    else:
        if not (0 <= qs <= qe < len(seq)):
            raise ValueError(f"Invalid query_start/query_end {qs}/{qe} for len {len(seq)}")
        v_seq = seq[qs : qe + 1]
    lo, hi = CDRH3_RANGE
    count = 0
    v_idx = 0
    for (pos, icode), aa in numbering:
        if aa == "-":
            continue
        if v_idx >= len(v_seq):
            raise ValueError(f"CDRH3 numbering longer than V domain ({len(v_seq)})")
        if aa != v_seq[v_idx]:
            raise ValueError(
                f"CDRH3 numbering mismatch at V idx {v_idx}: seq {v_seq[v_idx]!r} != {aa!r}"
            )
        if lo <= pos <= hi:
            count += 1
        v_idx += 1
    if v_idx != len(v_seq):
        raise ValueError(f"CDRH3 numbering length mismatch: {v_idx} vs {len(v_seq)}")
    return count


def get_cdrh3_lengths_batch(seqs: dict[str, str]) -> dict[str, int]:
    """Return Chothia CDRH3 lengths per heavy-chain sequence; reject numbering failures."""
    if not seqs:
        return {}
    for name, seq in seqs.items():
        if not seq:
            raise ValueError(f"Empty sequence for CDRH3 length: {name!r}")
    model = _get_anarcii_model()
    model.number(seqs)
    chothia = model.to_scheme("chothia")
    if not isinstance(chothia, dict):
        raise RuntimeError(f"Anarcii to_scheme('chothia') returned {type(chothia)}")
    lo, hi = CDRH3_RANGE
    out: dict[str, int] = {}
    for name, seq in seqs.items():
        entry = chothia.get(name)
        if entry is None or entry.get("error") is not None:
            raise RuntimeError(f"Anarcii failed to number CDRH3 for {name!r}: {entry}")
        numbering = entry["numbering"]
        qs = entry.get("query_start")
        qe = entry.get("query_end")
        if qs is None or qe is None:
            v_seq = seq
        else:
            v_seq = seq[qs : qe + 1]
        count = 0
        v_idx = 0
        for (pos, icode), aa in numbering:
            if aa == "-":
                continue
            if v_idx >= len(v_seq):
                raise ValueError(f"CDRH3 numbering longer than V domain for {name!r}")
            if aa != v_seq[v_idx]:
                raise ValueError(
                    f"CDRH3 mismatch at V idx {v_idx} for {name!r}: "
                    f"{v_seq[v_idx]!r} != {aa!r}"
                )
            if lo <= pos <= hi:
                count += 1
            v_idx += 1
        if v_idx != len(v_seq):
            raise ValueError(f"CDRH3 numbering length mismatch for {name!r}: {v_idx} vs {len(v_seq)}")
        out[name] = count
    return out


def is_cdr_fully_resolved(
    seq: str,
    resolved: set[int],
    chain_role: str,
) -> tuple[bool, set[int]]:
    """Return CDR completeness and missing zero-based indices from one-based resolved positions."""
    cdr_indices = get_chothia_cdr_indices(seq, chain_role)
    resolved_0 = {r - 1 for r in resolved}
    missing = {i for i in cdr_indices if i not in resolved_0}
    return (len(missing) == 0, missing)


# ── Stage metrics ────────────────────────────────────────────────────────────

class StageMetrics:
    """Track ordered filter-stage counts and optional exclusion reasons."""

    def __init__(self, title: str) -> None:
        self.title = title
        self.stages: list[dict] = []

    def add(self, name: str, n_before: int, n_after: int, reasons: Optional[dict] = None) -> None:
        """Record a stage.  Reasons map a loss label to a count."""
        if n_before < 0 or n_after < 0 or n_after > n_before:
            raise ValueError(
                f"StageMetrics[{name}]: invalid counts before={n_before} after={n_after}"
            )
        self.stages.append({
            "stage": name,
            "before": n_before,
            "after": n_after,
            "dropped": n_before - n_after,
            "reasons": reasons or {},
        })

    def report(self) -> None:
        """Print the loss table to stdout."""
        print(f"\n  ┌─ LOSS BY STAGE: {self.title}")
        for s in self.stages:
            line = (
                f"  │ {s['stage']:<46} {s['before']:>7,} → {s['after']:>7,} "
                f"  (−{s['dropped']:>7,})"
            )
            print(line)
            for reason, count in sorted(s["reasons"].items(), key=lambda kv: -kv[1]):
                print(f"  │      {reason}: {count:,}")
        print("  └" + "─" * 70)

    def to_dict(self) -> list[dict]:
        return self.stages
