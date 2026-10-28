"""Utilities for the Song -> Cross-atlas gamma-delta LR study.

The module deliberately separates receptor evidence, supervised labels, feature
selection, model fitting, and query cohorts.  This prevents annotation-only or
receptor-unknown cells from silently entering the gold standard.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import anndata as ad
import joblib
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TCR_PREFIXES = ("TRA", "TRB", "TRG", "TRD")
TCR_PANEL_DEFAULT = (
    "TRAC", "TRBC1", "TRBC2", "TRDC", "TRGC1", "TRGC2",
    "TRDV1", "TRDV2", "TRDV3",
)
T_LINEAGE_GENES = ("CD3D", "CD3E", "CD247", "LCK", "PTPRC")
NON_T_MARKERS = (
    "MS4A1", "CD79A", "CD79B", "CD14", "LYZ", "FCER1G", "TYROBP",
    "EPCAM", "KRT19", "COL1A1", "COL1A2",
)


def set_reproducibility(seed: int = 20260806, threads: int = 8) -> None:
    """Set deterministic seeds and conservative BLAS thread counts."""
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(threads)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def ensure_dirs(project_root: Path) -> None:
    for rel in (
        "manifests", "features", "models", "predictions", "logs",
        "results/tables", "results/figures",
    ):
        (project_root / rel).mkdir(parents=True, exist_ok=True)


def first_existing(candidates: Sequence[str | Path], kind: str = "any") -> Path:
    checked = [Path(x) for x in candidates]
    for path in checked:
        if kind == "dir" and path.is_dir():
            return path.resolve()
        if kind == "file" and path.is_file():
            return path.resolve()
        if kind == "any" and path.exists():
            return path.resolve()
    raise FileNotFoundError("No candidate exists:\n" + "\n".join(map(str, checked)))


def sha256_file(path: str | Path, chunk_size: int = 2**20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_true(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    return values.astype("string").str.strip().str.lower().isin(
        {"true", "t", "1", "yes", "y", "productive"}
    )


def canonical_barcode(barcode: object) -> str:
    text = str(barcode).strip()
    return re.sub(r"-\d+$", "", text)


def canonical_cell_id(dataset: str, donor: str, barcode: object) -> str:
    return f"{dataset}|{donor}|{canonical_barcode(barcode)}"


def find_obs_column(obs: pd.DataFrame, candidates: Sequence[str], required: bool = True) -> str | None:
    lookup = {str(c).lower(): str(c) for c in obs.columns}
    for candidate in candidates:
        if candidate in obs.columns:
            return candidate
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    if required:
        raise KeyError(f"None of {list(candidates)} found. Available: {list(obs.columns)}")
    return None


def read_contigs(
    path: str | Path,
    donor: str,
    dataset: str,
    require_full_length: bool = False,
    require_vj: bool = True,
) -> pd.DataFrame:
    """Read and harmonize a Cell Ranger contig annotation CSV.

    Gold-eligible rows require cell association, high confidence, productivity,
    chain identity, CDR3, positive reads/UMIs, and (by default) V/J calls.
    Full length is optional because field completeness can differ by source.
    """
    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    required = {"barcode", "chain"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")

    keep = pd.Series(True, index=df.index)
    for col in ("is_cell", "high_confidence", "productive"):
        if col in df:
            keep &= as_true(df[col])
    keep &= df["chain"].astype("string").isin(TCR_PREFIXES)
    if "cdr3" in df:
        keep &= df["cdr3"].notna() & df["cdr3"].astype("string").str.len().gt(0)
    if "cdr3_nt" in df:
        keep &= df["cdr3_nt"].notna() & df["cdr3_nt"].astype("string").str.len().gt(0)
    for col in ("reads", "umis"):
        if col in df:
            keep &= pd.to_numeric(df[col], errors="coerce").fillna(0).gt(0)
    if require_vj:
        for col in ("v_gene", "j_gene"):
            if col in df:
                keep &= df[col].notna() & df[col].astype("string").str.len().gt(0)
    if require_full_length and "full_length" in df:
        keep &= as_true(df["full_length"])

    out = df.loc[keep].copy()
    out["dataset"] = dataset
    out["donor"] = donor
    out["barcode_raw"] = out["barcode"].astype("string")
    out["barcode_core"] = out["barcode_raw"].map(canonical_barcode)
    out["cell_id"] = [canonical_cell_id(dataset, donor, x) for x in out["barcode_raw"]]
    out["source_file"] = str(Path(path).resolve())
    return out


def aggregate_contigs(contigs: pd.DataFrame) -> pd.DataFrame:
    """Produce one immutable receptor-evidence row per canonical cell."""
    if contigs.empty:
        cols = ["dataset", "donor", "barcode_core", "cell_id"]
        return pd.DataFrame(columns=cols).set_index("cell_id")
    group_cols = ["dataset", "donor", "barcode_core", "cell_id"]
    base = contigs[group_cols].drop_duplicates("cell_id").set_index("cell_id")
    counts = pd.crosstab(contigs["cell_id"], contigs["chain"]).reindex(
        columns=list(TCR_PREFIXES), fill_value=0
    )
    counts.columns = [f"n_{c}" for c in counts.columns]
    base = base.join(counts, how="left")
    for chain in TCR_PREFIXES:
        sub = contigs.loc[contigs["chain"].eq(chain)]
        if sub.empty:
            continue
        for source, target in (
            ("cdr3", f"{chain}_cdr3"), ("cdr3_nt", f"{chain}_cdr3_nt"),
            ("v_gene", f"{chain}_v_gene"), ("j_gene", f"{chain}_j_gene"),
        ):
            if source in sub:
                values = sub.groupby("cell_id")[source].agg(
                    lambda x: "|".join(sorted(set(x.dropna().astype(str))))
                )
                base[target] = values
        for source in ("reads", "umis"):
            if source in sub:
                base[f"{chain}_{source}"] = pd.to_numeric(
                    sub[source], errors="coerce"
                ).groupby(sub["cell_id"]).sum()
    for col in [f"n_{x}" for x in TCR_PREFIXES]:
        base[col] = base.get(col, 0).fillna(0).astype(int)
    return assign_receptor_groups(base)


def assign_receptor_groups(evidence: pd.DataFrame) -> pd.DataFrame:
    out = evidence.copy()
    for chain in TCR_PREFIXES:
        col = f"n_{chain}"
        if col not in out:
            out[col] = 0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)
    tra, trb, trg, trd = (out[f"n_{x}"] for x in TCR_PREFIXES)
    has_ab = (tra.gt(0) | trb.gt(0))
    has_gd = (trg.gt(0) | trd.gt(0))
    group = np.select(
        [
            trg.gt(0) & trd.gt(0) & ~has_ab,
            tra.gt(0) & trb.gt(0) & ~has_gd,
            trg.gt(0) & trd.eq(0) & ~has_ab,
            trd.gt(0) & trg.eq(0) & ~has_ab,
            tra.gt(0) & trb.eq(0) & ~has_gd,
            trb.gt(0) & tra.eq(0) & ~has_gd,
            has_ab & has_gd,
            ~has_ab & ~has_gd,
        ],
        [
            "paired_gdT", "paired_abT", "orphan_TRG", "orphan_TRD",
            "single_TRA", "single_TRB", "mixed_ab_gd", "none",
        ],
        default="ambiguous",
    )
    out["receptor_group"] = pd.Categorical(group)
    out["gold_A_paired_gdT"] = trg.eq(1) & trd.eq(1) & tra.eq(0) & trb.eq(0)
    out["gold_B_paired_gdT"] = trg.ge(1) & trd.ge(1) & tra.eq(0) & trb.eq(0)
    out["gold_A_paired_abT"] = tra.eq(1) & trb.eq(1) & trg.eq(0) & trd.eq(0)
    out["gold_B_paired_abT"] = tra.ge(1) & trb.ge(1) & trg.eq(0) & trd.eq(0)
    out["candidate_orphan_TRD_nonoverlap"] = trd.ge(1) & trg.eq(0) & tra.eq(0) & trb.eq(0)
    out["candidate_orphan_TRG_nonoverlap"] = trg.ge(1) & trd.eq(0) & tra.eq(0) & trb.eq(0)
    return out


def ir_obs_to_evidence(obs: pd.DataFrame, dataset: str, default_donor: str = "unknown") -> pd.DataFrame:
    """Reconstruct reported receptor chains from Scirpy IR_*_locus fields.

    This evidence tier is *reported locus plus a CDR3 field*.  It must not be
    described as Cell-Ranger high-confidence/productive unless those flags are
    separately available in the object.
    """
    donor_col = find_obs_column(obs, ["donor", "donor_id", "patient no", "patient", "sample"], False)
    rows: list[dict] = []
    locus_cols = [c for c in obs.columns if re.fullmatch(r"IR_[A-Za-z0-9]+_locus", str(c))]
    for barcode, row in obs.iterrows():
        donor = str(row[donor_col]) if donor_col and pd.notna(row[donor_col]) else default_donor
        cell_id = canonical_cell_id(dataset, donor, barcode)
        for locus_col in locus_cols:
            chain = str(row[locus_col]) if pd.notna(row[locus_col]) else ""
            if chain not in TCR_PREFIXES:
                continue
            prefix = locus_col[:-6]
            cdr3_cols = [f"{prefix}_junction_aa", f"{prefix}_junction", f"{prefix}_cdr3"]
            cdr3 = next((row[c] for c in cdr3_cols if c in obs.columns and pd.notna(row[c])), None)
            if cdr3 is None or not str(cdr3).strip():
                continue
            productive_col = next((c for c in (f"{prefix}_productive", f"{prefix}_is_productive") if c in obs.columns), None)
            if productive_col and not as_true(pd.Series([row[productive_col]])).iloc[0]:
                continue
            v_col = next((c for c in (f"{prefix}_v_call", f"{prefix}_v_gene") if c in obs.columns), None)
            j_col = next((c for c in (f"{prefix}_j_call", f"{prefix}_j_gene") if c in obs.columns), None)
            v_gene = row[v_col] if v_col and pd.notna(row[v_col]) else None
            j_gene = row[j_col] if j_col and pd.notna(row[j_col]) else None
            # If the object exposes V/J columns, incomplete calls are not gold eligible.
            if v_col and (v_gene is None or not str(v_gene).strip()):
                continue
            if j_col and (j_gene is None or not str(j_gene).strip()):
                continue
            available = ["locus", "CDR3"]
            if productive_col: available.append("productive")
            if v_col: available.append("V")
            if j_col: available.append("J")
            rows.append({
                "dataset": dataset, "donor": donor, "barcode_raw": str(barcode),
                "barcode_core": canonical_barcode(barcode), "cell_id": cell_id,
                "chain": chain, "cdr3": str(cdr3), "v_gene": v_gene, "j_gene": j_gene,
                "source_slot": prefix,
                "evidence_tier": "reported_IR_" + "_".join(available),
            })
    return aggregate_contigs(pd.DataFrame(rows)) if rows else assign_receptor_groups(pd.DataFrame())


def attach_evidence(adata: ad.AnnData, evidence: pd.DataFrame, dataset: str,
                    donor_col: str, barcode_col: str | None = None) -> ad.AnnData:
    out = adata.copy()
    barcode_values = out.obs[barcode_col] if barcode_col else pd.Series(out.obs_names, index=out.obs_names)
    out.obs["cell_id"] = [
        canonical_cell_id(dataset, str(d), b)
        for d, b in zip(out.obs[donor_col].astype(str), barcode_values.astype(str))
    ]
    if out.obs["cell_id"].duplicated().any():
        dup = out.obs.loc[out.obs["cell_id"].duplicated(False), "cell_id"].head().tolist()
        raise ValueError(f"Non-unique canonical cell IDs; examples: {dup}")
    out.obs_names = out.obs["cell_id"].astype(str)
    out.obs.index.name = None
    joined = evidence.reindex(out.obs_names)
    for col in joined.columns:
        if col not in out.obs or col.startswith("n_") or col.startswith("gold_") or col.startswith("candidate_"):
            out.obs[col] = joined[col].to_numpy()
    for chain in TCR_PREFIXES:
        col = f"n_{chain}"
        out.obs[col] = pd.to_numeric(out.obs.get(col, 0), errors="coerce").fillna(0).astype(int)
    relabeled = assign_receptor_groups(out.obs)
    for col in relabeled.columns:
        out.obs[col] = relabeled[col]
    return out


def load_song_dataset(song_root: str | Path, donors: Sequence[str] = ("hd1", "hd2", "hd3", "hd4"),
                      require_full_length: bool = False) -> tuple[ad.AnnData, pd.DataFrame, pd.DataFrame]:
    """Load Song GEX plus both alpha-beta and gamma-delta filtered contigs."""
    song_root = Path(song_root)
    adatas = []
    contig_parts = []
    input_rows = []
    for donor in donors:
        donor_dir = song_root / donor
        matrix_dir = donor_dir / "filtered_feature_bc_matrix"
        if not matrix_dir.is_dir():
            raise FileNotFoundError(matrix_dir)
        obj = sc.read_10x_mtx(matrix_dir, var_names="gene_symbols", cache=False)
        obj.var_names_make_unique()
        obj.obs["dataset"] = "Song_D1"
        obj.obs["donor"] = donor
        obj.obs["barcode_raw"] = obj.obs_names.astype(str)
        obj.obs_names = [canonical_cell_id("Song_D1", donor, x) for x in obj.obs["barcode_raw"]]
        obj.obs.index.name = None
        adatas.append(obj)
        for library, filename in (
            ("ab", "ab_tcr_filtered_contig_annotations.csv"),
            ("gd", "gd_tcr_filtered_contig_annotations.csv"),
        ):
            path = donor_dir / filename
            if not path.is_file():
                raise FileNotFoundError(path)
            part = read_contigs(path, donor, "Song_D1", require_full_length=require_full_length)
            part["source_library"] = library
            contig_parts.append(part)
            input_rows.append({
                "dataset": "Song_D1", "donor": donor, "modality": f"{library}_VDJ",
                "path": str(path.resolve()), "sha256": sha256_file(path),
                "n_gold_eligible_contigs": len(part),
            })
    gex = ad.concat(adatas, join="outer", merge="same", label=None, index_unique=None)
    contigs = pd.concat(contig_parts, ignore_index=True)
    evidence = aggregate_contigs(contigs)
    gex = attach_evidence(gex, evidence, "Song_D1", donor_col="donor", barcode_col="barcode_raw")
    return gex, contigs, pd.DataFrame(input_rows)


def combine_evidence_tables(tables: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Combine receptor-count evidence from separate assay objects by cell ID."""
    usable = [x.copy() for x in tables if x is not None and len(x)]
    if not usable:
        return assign_receptor_groups(pd.DataFrame())
    meta_cols = ["dataset", "donor", "barcode_core"]
    frames = []
    for table in usable:
        frame = table.reset_index().rename(columns={table.index.name or "index": "cell_id"})
        detail = [c for c in frame.columns if any(c.startswith(f"{chain}_") for chain in TCR_PREFIXES)]
        keep = ["cell_id"] + [c for c in meta_cols if c in frame] + [f"n_{x}" for x in TCR_PREFIXES if f"n_{x}" in frame] + detail
        keep = list(dict.fromkeys(keep))
        frames.append(frame[keep])
    combined = pd.concat(frames, ignore_index=True)
    counts = combined.groupby("cell_id")[[f"n_{x}" for x in TCR_PREFIXES]].sum()
    meta = combined.drop_duplicates("cell_id").set_index("cell_id")[[c for c in meta_cols if c in combined]]
    detail_cols = [c for c in combined.columns if any(c.startswith(f"{chain}_") for chain in TCR_PREFIXES)
                   and c not in counts.columns]
    if detail_cols:
        details = combined.groupby("cell_id")[detail_cols].agg(
            lambda x: "|".join(sorted(set(v for v in x.dropna().astype(str) if v))) or np.nan
        )
        meta = meta.join(details, how="outer")
    return assign_receptor_groups(meta.join(counts, how="outer"))


def read_h5ad_obs(path: str | Path) -> pd.DataFrame:
    handle = ad.read_h5ad(path, backed="r")
    obs = handle.obs.copy()
    handle.file.close()
    return obs


def load_dataset2_ir_evidence(ab_path: str | Path, gd_path: str | Path,
                              dataset: str = "CrossAtlas_D2") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load Cross-atlas alpha-beta and gamma-delta IR metadata and combine by cell ID."""
    ab_obs = read_h5ad_obs(ab_path)
    gd_obs = read_h5ad_obs(gd_path)
    ab_ev = ir_obs_to_evidence(ab_obs, dataset)
    gd_ev = ir_obs_to_evidence(gd_obs, dataset)
    combined = combine_evidence_tables([ab_ev, gd_ev])
    def audit_row(name, path, obs, ev):
        columns = list(map(str, obs.columns))
        return {
            "object": name, "path": str(Path(path).resolve()), "n_obs": len(obs),
            "n_evidence_cells": len(ev),
            "has_productive_field": any(c.endswith(("_productive", "_is_productive")) for c in columns),
            "has_V_call_field": any(c.endswith(("_v_call", "_v_gene")) for c in columns),
            "has_J_call_field": any(c.endswith(("_j_call", "_j_gene")) for c in columns),
            "has_high_confidence_field": any("high_confidence" in c for c in columns),
            "has_read_or_UMI_field": any(c.endswith(("_reads", "_umis")) for c in columns),
            "minimum_evidence": "reported IR locus plus non-null CDR3; productive/V/J enforced when exposed",
        }
    audit = pd.DataFrame([
        audit_row("ab", ab_path, ab_obs, ab_ev), audit_row("gd", gd_path, gd_obs, gd_ev)
    ])
    return combined, audit


def load_dataset2_gex(full_path: str | Path, organ: str, evidence: pd.DataFrame,
                      dataset: str = "CrossAtlas_D2") -> ad.AnnData:
    """Load one organ from the Cross-atlas GEX object and attach receptor evidence."""
    obj = ad.read_h5ad(full_path)
    organ_col = find_obs_column(obj.obs, ["organ code", "organ_code", "tissue", "organ", "sample_type"])
    donor_col = find_obs_column(obj.obs, ["donor", "donor_id", "patient no", "patient", "sample"])
    organ_values = obj.obs[organ_col].astype(str).str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
    organ_values = organ_values.replace({"SPLEEN": "SPL", "BONEMARROW": "BMA"})
    target = re.sub(r"[^A-Z0-9]", "", organ.upper())
    aliases = {"SPLEEN": "SPL", "BONE MARROW": "BMA", "BONEMARROW": "BMA"}
    target = aliases.get(organ.upper(), target)
    mask = organ_values.eq(target) | organ_values.eq(organ.upper())
    if not mask.any():
        available = sorted(obj.obs[organ_col].astype(str).unique())
        raise ValueError(f"Organ {organ!r} not found in {organ_col}. Available: {available[:100]}")
    out = obj[mask].copy()
    out.obs["dataset"] = f"{dataset}_{target}"
    out.obs["donor"] = out.obs[donor_col].astype("string").fillna("unknown").astype(str)
    out.obs["barcode_raw"] = out.obs_names.astype(str)
    # Evidence keys use the base dataset name; re-key to the organ-specific dataset.
    ev = evidence.reset_index().copy()
    id_col = "cell_id" if "cell_id" in ev else evidence.index.name or "index"
    ev["cell_id"] = [canonical_cell_id(out.obs["dataset"].iloc[0], str(d), b)
                     for d, b in zip(ev["donor"].astype(str), ev["barcode_core"].astype(str))]
    ev["dataset"] = out.obs["dataset"].iloc[0]
    ev = ev.set_index("cell_id")
    return attach_evidence(out, ev, out.obs["dataset"].iloc[0], donor_col="donor", barcode_col="barcode_raw")


def get_counts(adata: ad.AnnData):
    for layer in ("counts", "raw_counts"):
        if layer in adata.layers:
            return adata.layers[layer]
    if adata.raw is not None and adata.raw.shape[1] == adata.shape[1]:
        return adata.raw.X
    return adata.X


def make_log_normalized(adata: ad.AnnData, target_sum: float = 1e4) -> ad.AnnData:
    out = ad.AnnData(X=get_counts(adata).copy(), obs=adata.obs.copy(), var=adata.var.copy())
    out.var_names_make_unique()
    sc.pp.normalize_total(out, target_sum=target_sum)
    sc.pp.log1p(out)
    return out


def add_qc_and_lineage_flags(adata: ad.AnnData, min_genes: int = 200,
                             max_mt_pct: float = 20.0, require_t_gene: bool = True) -> ad.AnnData:
    out = adata.copy()
    out.var["mt"] = out.var_names.astype(str).str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(out, qc_vars=["mt"], inplace=True, percent_top=None)
    log = make_log_normalized(out)
    present = [g for g in T_LINEAGE_GENES if g in log.var_names]
    if present:
        X = log[:, present].X
        detected = np.asarray((X > 0).sum(axis=1)).ravel()
        mean_score = np.asarray(X.mean(axis=1)).ravel()
    else:
        detected = np.zeros(out.n_obs, dtype=int)
        mean_score = np.zeros(out.n_obs)
    out.obs["n_Tlineage_markers_detected"] = detected
    out.obs["Tlineage_mean_expression"] = mean_score
    doublet_col = find_obs_column(out.obs, ["predicted_doublet", "doublet", "is_doublet"], False)
    doublet = as_true(out.obs[doublet_col]) if doublet_col else pd.Series(False, index=out.obs_names)
    out.obs["rna_doublet"] = doublet.to_numpy()
    qc = out.obs["n_genes_by_counts"].ge(min_genes) & out.obs["pct_counts_mt"].le(max_mt_pct) & ~out.obs["rna_doublet"]
    if require_t_gene and present:
        qc &= out.obs["n_Tlineage_markers_detected"].ge(1)
    out.obs["qc_Tlineage_pass"] = qc.astype(bool)
    return out


def finalize_cohorts(adata: ad.AnnData) -> ad.AnnData:
    out = adata.copy()
    passed = out.obs.get("qc_Tlineage_pass", pd.Series(True, index=out.obs_names)).fillna(False).astype(bool)
    obs = out.obs
    cohort = np.select(
        [
            passed & obs["gold_B_paired_gdT"].fillna(False),
            passed & obs["gold_B_paired_abT"].fillna(False),
            passed & obs["candidate_orphan_TRD_nonoverlap"].fillna(False),
            passed & obs["candidate_orphan_TRG_nonoverlap"].fillna(False),
            passed & obs["receptor_group"].astype(str).eq("mixed_ab_gd"),
            passed & obs["receptor_group"].astype(str).eq("none"),
            ~passed,
        ],
        [
            "gold_B_paired_gdT", "gold_B_paired_abT",
            "candidate_orphan_TRD_nonoverlap", "query_orphan_TRG",
            "audit_mixed_ab_gd", "query_none_Tlineage", "audit_qc_failed",
        ],
        default="query_other_receptor",
    )
    out.obs["immutable_cohort"] = pd.Categorical(cohort)
    return out


def is_tcr_gene(gene: str) -> bool:
    gene = str(gene).upper()
    return gene.startswith(TCR_PREFIXES)


def common_genes(adatas: Sequence[ad.AnnData]) -> list[str]:
    common = set(map(str, adatas[0].var_names))
    for obj in adatas[1:]:
        common &= set(map(str, obj.var_names))
    return sorted(common)


def subset_unique_genes(adata: ad.AnnData, genes: Sequence[str]) -> ad.AnnData:
    first = ~pd.Index(adata.var_names.astype(str)).duplicated()
    obj = adata[:, first].copy()
    keep = [g for g in genes if g in obj.var_names]
    return obj[:, keep].copy()


def select_hvgs(
    adata: ad.AnnData,
    n_top: int,
    batch_key: str | None,
    eligible_cells: pd.Series | np.ndarray,
    exclude_tcr: bool = True,
    seed: int = 20260806,
) -> tuple[list[str], dict]:
    """Select HVGs from eligible cells with a deterministic documented fallback."""
    obj = ad.AnnData(X=get_counts(adata)[np.asarray(eligible_cells)].copy(),
                     obs=adata.obs.loc[np.asarray(eligible_cells)].copy(),
                     var=adata.var.copy())
    obj.var_names_make_unique()
    if exclude_tcr:
        obj = obj[:, [not is_tcr_gene(x) for x in obj.var_names]].copy()
    methods = ["seurat_v3", "cell_ranger"]
    errors = {}
    for method in methods:
        try:
            if method == "seurat_v3":
                sc.pp.highly_variable_genes(obj, flavor=method, n_top_genes=n_top, batch_key=batch_key)
            else:
                sc.pp.normalize_total(obj, target_sum=1e4)
                sc.pp.log1p(obj)
                sc.pp.highly_variable_genes(obj, flavor=method, n_top_genes=n_top, batch_key=batch_key)
            genes = obj.var_names[obj.var["highly_variable"]].astype(str).tolist()
            return genes, {"method": method, "n_top": n_top, "seed": seed, "fallback_errors": errors}
        except Exception as exc:  # document fallback rather than silently changing methods
            errors[method] = repr(exc)
    raise RuntimeError(f"HVG selection failed: {errors}")


def model_registry(lambda_trd: float = 0.5) -> pd.DataFrame:
    rows = []
    idx = 1
    for hvg in ("H1_song_inductive", "H2_song_spl_target_informed"):
        for genes in ("G1_TCR_free", "G2_TCR_inclusive"):
            for labels in ("L1_paired_only", "L2_paired_plus_TRD_equal", "L3_paired_plus_TRD_weighted"):
                rows.append({
                    "model_id": f"M{idx:02d}", "label_regime": labels,
                    "hvg_regime": hvg, "gene_regime": genes,
                    "lambda_TRD": 0.0 if labels == "L1_paired_only" else (1.0 if labels == "L2_paired_plus_TRD_equal" else lambda_trd),
                    "confirmatory_eligible": hvg == "H1_song_inductive",
                })
                idx += 1
    return pd.DataFrame(rows)


def pca_model_registry(lambda_trd: float = 0.5) -> pd.DataFrame:
    """Registered factorial comparison of inductive/transductive PCA–LR.

    P01/P02 and P05/P06 never use target expression. P03/P04 and P07/P08 may
    use the predeclared unlabeled target-expression pool to fit PCA, but SPL
    labels are not used to fit PCA, LR, C, or any decision threshold. P01-P04
    are TCR-free; P05-P08 are TCR-inclusive positive-control models.
    """
    return pd.DataFrame([
        {"model_id": "P01", "representation": "inductive_song_only_PCA",
         "label_regime": "L1_paired_only", "lambda_TRD": 0.0,
         "gene_regime": "G1_TCR_free"},
        {"model_id": "P02", "representation": "inductive_song_only_PCA",
         "label_regime": "L3_paired_plus_TRD_weighted", "lambda_TRD": float(lambda_trd),
         "gene_regime": "G1_TCR_free"},
        {"model_id": "P03", "representation": "transductive_song_plus_unlabeled_SPL_joint_PCA",
         "label_regime": "L1_paired_only", "lambda_TRD": 0.0,
         "gene_regime": "G1_TCR_free"},
        {"model_id": "P04", "representation": "transductive_song_plus_unlabeled_SPL_joint_PCA",
         "label_regime": "L3_paired_plus_TRD_weighted", "lambda_TRD": float(lambda_trd),
         "gene_regime": "G1_TCR_free"},
        {"model_id": "P05", "representation": "inductive_song_only_PCA",
         "label_regime": "L1_paired_only", "lambda_TRD": 0.0,
         "gene_regime": "G2_TCR_inclusive"},
        {"model_id": "P06", "representation": "inductive_song_only_PCA",
         "label_regime": "L3_paired_plus_TRD_weighted", "lambda_TRD": float(lambda_trd),
         "gene_regime": "G2_TCR_inclusive"},
        {"model_id": "P07", "representation": "transductive_song_plus_unlabeled_SPL_joint_PCA",
         "label_regime": "L1_paired_only", "lambda_TRD": 0.0,
         "gene_regime": "G2_TCR_inclusive"},
        {"model_id": "P08", "representation": "transductive_song_plus_unlabeled_SPL_joint_PCA",
         "label_regime": "L3_paired_plus_TRD_weighted", "lambda_TRD": float(lambda_trd),
         "gene_regime": "G2_TCR_inclusive"},
    ]).assign(
        hvg_regime="H1_song_inductive",
        target_labels_used_for_representation=False,
        confirmatory_eligible=lambda x: (
            x["representation"].eq("inductive_song_only_PCA")
            & x["gene_regime"].eq("G1_TCR_free")
        ),
    )


def make_training_table(obs: pd.DataFrame, label_regime: str) -> pd.DataFrame:
    gd = obs["immutable_cohort"].astype(str).eq("gold_B_paired_gdT")
    ab = obs["immutable_cohort"].astype(str).eq("gold_B_paired_abT")
    trd = obs["immutable_cohort"].astype(str).eq("candidate_orphan_TRD_nonoverlap")
    keep = gd | ab | (trd if label_regime != "L1_paired_only" else False)
    tab = obs.loc[keep, [c for c in ("dataset", "donor", "immutable_cohort") if c in obs]].copy()
    tab["y"] = (~ab.loc[keep]).astype(int)
    tab["positive_source"] = np.where(gd.loc[keep], "paired_gdT", np.where(trd.loc[keep], "orphan_TRD", "paired_abT"))
    return tab


def donor_balanced_weights(training: pd.DataFrame, label_regime: str,
                           lambda_trd: float = 0.5) -> pd.Series:
    """Give every donor equal total weight and each class 50% within donor.

    L2 shares positive mass equally per positive cell. L3 divides positive mass
    between paired-gd and orphan-TRD at an aggregate ratio of 1:lambda_trd.
    """
    w = pd.Series(0.0, index=training.index)
    donors = training["donor"].astype(str).unique()
    for donor in donors:
        d = training["donor"].astype(str).eq(donor)
        neg = d & training["y"].eq(0)
        paired = d & training["positive_source"].eq("paired_gdT")
        orphan = d & training["positive_source"].eq("orphan_TRD")
        positive = paired | orphan
        if neg.sum():
            w.loc[neg] = 0.5 / neg.sum()
        if label_regime in ("L1_paired_only", "L2_paired_plus_TRD_equal"):
            if positive.sum():
                w.loc[positive] = 0.5 / positive.sum()
        else:
            if paired.sum() and orphan.sum():
                paired_mass = 0.5 / (1.0 + lambda_trd)
                orphan_mass = 0.5 * lambda_trd / (1.0 + lambda_trd)
                w.loc[paired] = paired_mass / paired.sum()
                w.loc[orphan] = orphan_mass / orphan.sum()
            elif paired.sum():
                w.loc[paired] = 0.5 / paired.sum()
            elif orphan.sum():
                w.loc[orphan] = 0.5 / orphan.sum()
    if (w <= 0).any():
        bad = training.loc[w <= 0, ["donor", "positive_source"]].value_counts()
        raise ValueError(f"Zero-weight training cells:\n{bad}")
    return w * (len(w) / w.sum())


def build_lr(C: float, seed: int = 20260806) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler(with_mean=False)),
        ("lr", LogisticRegression(
            C=float(C), penalty="l2", solver="liblinear", max_iter=4000,
            random_state=seed,
        )),
    ])


def build_pc_lr(C: float, seed: int = 20260806) -> Pipeline:
    """Logistic regression for dense PCA scores."""
    return Pipeline([
        ("scale", StandardScaler(with_mean=True)),
        ("lr", LogisticRegression(
            C=float(C), penalty="l2", solver="liblinear", max_iter=4000,
            random_state=seed,
        )),
    ])


def _matrix_for_genes(adata: ad.AnnData, cells: Sequence[str], genes: Sequence[str]):
    obj = subset_unique_genes(adata[cells], genes)
    if obj.n_vars != len(genes):
        missing = sorted(set(genes) - set(obj.var_names))
        raise ValueError(f"Missing model genes: {missing[:20]} ({len(missing)} total)")
    X = make_log_normalized(obj).X
    return X.tocsr() if sp.issparse(X) else np.asarray(X)


def fit_lr(adata: ad.AnnData, training: pd.DataFrame, genes: Sequence[str], C: float,
           label_regime: str, lambda_trd: float, seed: int = 20260806) -> Pipeline:
    X = _matrix_for_genes(adata, training.index.astype(str), genes)
    weights = donor_balanced_weights(training, label_regime, lambda_trd)
    model = build_lr(C, seed)
    model.fit(X, training["y"].to_numpy(), lr__sample_weight=weights.to_numpy())
    return model


def fit_pca_lr(
    source_adata: ad.AnnData,
    training: pd.DataFrame,
    genes: Sequence[str],
    C: float,
    label_regime: str,
    lambda_trd: float,
    representation_source_cells: Sequence[str],
    target_adata: ad.AnnData | None = None,
    representation_target_cells: Sequence[str] | None = None,
    n_components: int = 50,
    seed: int = 20260806,
) -> dict:
    """Fit a centered PCA representation and Song-label-only LR classifier.

    The target interface intentionally accepts expression and cell IDs only;
    outcome labels are neither required nor accepted. This makes target-label
    access impossible inside this fitting function.
    """
    pca_bundle = fit_pca_basis(
        source_adata=source_adata,
        genes=genes,
        representation_source_cells=representation_source_cells,
        target_adata=target_adata,
        representation_target_cells=representation_target_cells,
        n_components=n_components,
        seed=seed,
    )
    classifier = fit_pc_classifier(
        source_adata=source_adata,
        training=training,
        genes=genes,
        pca=pca_bundle["pca"],
        C=C,
        label_regime=label_regime,
        lambda_trd=lambda_trd,
        seed=seed,
    )
    return pca_bundle | {"model": classifier}


def fit_pca_basis(
    source_adata: ad.AnnData,
    genes: Sequence[str],
    representation_source_cells: Sequence[str],
    target_adata: ad.AnnData | None = None,
    representation_target_cells: Sequence[str] | None = None,
    n_components: int = 50,
    seed: int = 20260806,
) -> dict:
    """Fit PCA without accepting source or target outcome labels."""
    source_cells = list(map(str, representation_source_cells))
    if not source_cells:
        raise ValueError("The source PCA representation pool is empty")
    matrices = [_matrix_for_genes(source_adata, source_cells, genes)]
    target_cells = [] if representation_target_cells is None else list(map(str, representation_target_cells))
    if target_cells:
        if target_adata is None:
            raise ValueError("target_adata is required when target representation cells are supplied")
        matrices.append(_matrix_for_genes(target_adata, target_cells, genes))
    elif target_adata is not None and representation_target_cells is None:
        raise ValueError("Explicit representation_target_cells are required for target expression")

    X_rep = sp.vstack([sp.csr_matrix(x) for x in matrices], format="csr")
    effective_components = min(int(n_components), X_rep.shape[0] - 1, X_rep.shape[1] - 1)
    if effective_components < 2:
        raise ValueError(f"Insufficient PCA rank for shape {X_rep.shape}")
    pca = PCA(n_components=effective_components, svd_solver="arpack", random_state=seed)
    pca.fit(X_rep)
    return {
        "pca": pca,
        "genes": list(genes),
        "n_components_requested": int(n_components),
        "n_components_effective": int(effective_components),
        "n_source_representation_cells": len(source_cells),
        "n_target_representation_cells": len(target_cells),
    }


def fit_pc_classifier(
    source_adata: ad.AnnData,
    training: pd.DataFrame,
    genes: Sequence[str],
    pca: PCA,
    C: float,
    label_regime: str,
    lambda_trd: float,
    seed: int = 20260806,
) -> Pipeline:
    """Fit the supervised part after the expression-only PCA is frozen."""
    X_train = _matrix_for_genes(source_adata, training.index.astype(str), genes)
    Z_train = pca.transform(X_train)
    weights = donor_balanced_weights(training, label_regime, lambda_trd)
    classifier = build_pc_lr(C, seed)
    classifier.fit(Z_train, training["y"].to_numpy(), lr__sample_weight=weights.to_numpy())
    return classifier


def predict_pca_lr(bundle: Mapping, adata: ad.AnnData,
                   cells: Sequence[str] | None = None) -> pd.Series:
    """Apply a frozen PCA basis and its LR classifier."""
    cells = list(map(str, cells if cells is not None else adata.obs_names))
    X = _matrix_for_genes(adata, cells, bundle["genes"])
    Z = bundle["pca"].transform(X)
    return pd.Series(bundle["model"].predict_proba(Z)[:, 1], index=cells, name="p_gdT")


def predict_lr(model: Pipeline, adata: ad.AnnData, genes: Sequence[str],
               cells: Sequence[str] | None = None) -> pd.Series:
    cells = list(map(str, cells if cells is not None else adata.obs_names))
    X = _matrix_for_genes(adata, cells, genes)
    return pd.Series(model.predict_proba(X)[:, 1], index=cells, name="p_gdT")


def predict_lr_zero_fill(model: Pipeline, adata: ad.AnnData, genes: Sequence[str],
                         cells: Sequence[str] | None = None) -> tuple[pd.Series, list[str]]:
    """Predict after explicit zero-fill of a small set of absent external genes."""
    cells = list(map(str, cells if cells is not None else adata.obs_names))
    unique = subset_unique_genes(adata[cells], [g for g in genes if g in adata.var_names])
    present = [g for g in genes if g in unique.var_names]
    missing = [g for g in genes if g not in unique.var_names]
    log = make_log_normalized(unique)
    Xp = log[:, present].X
    Xp = sp.csr_matrix(Xp) if not sp.issparse(Xp) else Xp.tocsr()
    coo = Xp.tocoo()
    destination = np.asarray([genes.index(g) for g in present], dtype=int)
    X = sp.csr_matrix((coo.data, (coo.row, destination[coo.col])), shape=(len(cells), len(genes)))
    p = model.predict_proba(X)[:, 1]
    return pd.Series(p, index=cells, name="p_gdT"), missing


def threshold_max_f1(y: Sequence[int], p: Sequence[float]) -> float:
    precision, recall, thresholds = precision_recall_curve(y, p)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    if thresholds.size == 0:
        return 0.5
    return float(thresholds[min(int(np.nanargmax(f1)), thresholds.size - 1)])


def select_song_module_gate(
    y: Sequence[int],
    gd_score: Sequence[float],
    ab_score: Sequence[float],
    quantiles: Sequence[float] | None = None,
) -> dict:
    """Select the Song two-dimensional module-score gate on source labels only.

    The faithful classification rule is ``gd_score > gd_threshold`` and
    ``ab_score < ab_threshold``. Candidate thresholds are source-score
    quantiles. Ties are resolved deterministically by specificity, recall, a
    higher gamma-delta threshold, and finally a lower alpha-beta threshold.
    The caller is responsible for passing development data only.
    """
    y = np.asarray(y, dtype=int)
    gd_score = np.asarray(gd_score, dtype=float)
    ab_score = np.asarray(ab_score, dtype=float)
    if not (len(y) == len(gd_score) == len(ab_score)):
        raise ValueError("y, gd_score, and ab_score must have identical lengths")
    if len(y) == 0 or set(np.unique(y)) != {0, 1}:
        raise ValueError("Song gate selection requires both binary classes")
    if not (np.isfinite(gd_score).all() and np.isfinite(ab_score).all()):
        raise ValueError("Song gate selection received non-finite scores")
    if quantiles is None:
        quantiles = np.linspace(0.02, 0.98, 97)
    quantiles = np.asarray(quantiles, dtype=float)
    if quantiles.ndim != 1 or len(quantiles) == 0 or np.any((quantiles < 0) | (quantiles > 1)):
        raise ValueError("quantiles must be a non-empty one-dimensional sequence in [0, 1]")

    gd_candidates = np.unique(np.quantile(gd_score, quantiles))
    ab_candidates = np.unique(np.quantile(ab_score, quantiles))
    rows = []
    positive = y == 1
    negative = ~positive
    for gd_threshold in gd_candidates:
        gd_pass = gd_score > gd_threshold
        # Evaluate all alpha-beta thresholds at once. This avoids thousands of
        # repeated sklearn calls for a donor-level grid search.
        predictions = gd_pass[:, None] & (ab_score[:, None] < ab_candidates[None, :])
        tp_values = predictions[positive].sum(axis=0).astype(int)
        fp_values = predictions[negative].sum(axis=0).astype(int)
        fn_values = int(positive.sum()) - tp_values
        tn_values = int(negative.sum()) - fp_values
        for j, ab_threshold in enumerate(ab_candidates):
            tp, fp, fn, tn = (int(tp_values[j]), int(fp_values[j]),
                              int(fn_values[j]), int(tn_values[j]))
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-12)
            rows.append({
                "gd_threshold": float(gd_threshold),
                "ab_threshold": float(ab_threshold),
                "f1": float(f1),
                "specificity": float(tn / max(tn + fp, 1)),
                "recall": float(recall),
                "precision": float(precision),
                "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
            })
    ranked = pd.DataFrame(rows).sort_values(
        ["f1", "specificity", "recall", "gd_threshold", "ab_threshold"],
        ascending=[False, False, False, False, True],
        kind="mergesort",
    )
    return ranked.iloc[0].to_dict()


def extract_celltypist_gdt_scores(
    probability_matrix: pd.DataFrame,
    primary_label: str = "gamma-delta T cells",
    cycling_label: str = "Cycling gamma-delta T cells",
) -> pd.DataFrame:
    """Extract declared CellTypist gamma-delta ranking scores.

    The primary score uses only ``primary_label``. The sensitivity score is the
    row-wise maximum of the primary and cycling labels when the latter exists;
    it is a composite ranking score, not a calibrated probability.
    """
    if not isinstance(probability_matrix, pd.DataFrame):
        raise TypeError("CellTypist probability_matrix must be a pandas DataFrame")
    if probability_matrix.index.has_duplicates or probability_matrix.columns.has_duplicates:
        raise ValueError("CellTypist probability_matrix index and columns must be unique")
    if primary_label not in probability_matrix.columns:
        raise KeyError(f"Required CellTypist class is missing: {primary_label}")
    primary = pd.to_numeric(probability_matrix[primary_label], errors="coerce")
    if primary.isna().any() or not np.isfinite(primary.to_numpy()).all():
        raise ValueError("CellTypist primary gamma-delta score contains non-finite values")
    if cycling_label in probability_matrix.columns:
        cycling = pd.to_numeric(probability_matrix[cycling_label], errors="coerce")
        if cycling.isna().any() or not np.isfinite(cycling.to_numpy()).all():
            raise ValueError("CellTypist cycling gamma-delta score contains non-finite values")
        sensitivity = pd.concat([primary, cycling], axis=1).max(axis=1)
        cycling_available = True
    else:
        sensitivity = primary.copy()
        cycling_available = False
    return pd.DataFrame({
        "celltypist_gamma_score": primary.astype(float),
        "celltypist_gamma_or_cycling_score": sensitivity.astype(float),
        "celltypist_cycling_class_available": cycling_available,
    }, index=probability_matrix.index.copy())


def threshold_for_min_recall(
    y: Sequence[int], p: Sequence[float], target_recall: float
) -> float:
    """Freeze the highest-precision threshold meeting recall in source OOF.

    This helper is intended for prespecified operating-point sensitivity only.
    It must be called on source out-of-fold predictions, never on the external
    SPL gold labels. Ties are resolved toward the higher (more conservative)
    threshold.
    """
    target_recall = float(target_recall)
    if not 0 < target_recall <= 1:
        raise ValueError("target_recall must be in (0, 1]")
    precision, recall, thresholds = precision_recall_curve(y, p)
    if thresholds.size == 0:
        return 0.5
    eligible = np.flatnonzero(recall[:-1] >= target_recall)
    if eligible.size == 0:
        return float(np.min(thresholds))
    best_precision = np.nanmax(precision[eligible])
    best = eligible[np.isclose(precision[eligible], best_precision)]
    return float(np.max(thresholds[best]))


def metrics_at_threshold(y: Sequence[int], p: Sequence[float], threshold: float) -> dict:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pred = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)), "n_positive": int(y.sum()), "prevalence": float(y.mean()),
        "auprc": float(average_precision_score(y, p)),
        "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "specificity": float(tn / max(tn + fp, 1)),
        "false_positive_rate": float(fp / max(tn + fp, 1)),
        "brier": float(brier_score_loss(y, p)), "threshold": float(threshold),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def coefficient_table(model: Pipeline, genes: Sequence[str]) -> pd.DataFrame:
    coef = model.named_steps["lr"].coef_.ravel()
    return pd.DataFrame({"gene": list(genes), "coefficient": coef}).assign(
        abs_coefficient=lambda x: x.coefficient.abs()
    ).sort_values("abs_coefficient", ascending=False)


def write_json(data: Mapping, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(data, handle, indent=2, default=str)


def safe_write_h5ad(adata: ad.AnnData, path: str | Path, compression: str = "gzip") -> None:
    """Write AnnData after removing index/column collisions rejected by anndata."""
    out = adata.copy()
    out.obs_names = out.obs_names.astype(str)
    out.var_names = out.var_names.astype(str)
    out.obs.index.name = None
    out.var.index.name = None
    for frame in (out.obs, out.var):
        for col in list(frame.columns):
            if str(col) in {"_index", "index"} and frame[col].astype(str).equals(pd.Series(frame.index.astype(str), index=frame.index)):
                frame.drop(columns=col, inplace=True)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(path, compression=compression)


def save_model_bundle(model: Pipeline, genes: Sequence[str], metadata: Mapping,
                      path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "genes": list(genes), "metadata": dict(metadata)}, path)


def load_model_bundle(path: str | Path) -> dict:
    return joblib.load(path)
