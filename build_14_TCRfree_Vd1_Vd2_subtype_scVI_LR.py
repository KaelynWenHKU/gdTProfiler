#!/usr/bin/env python3
"""Authoritative generator for the reconciled 544-cell paired-only notebook.

Generation never fits models or writes analysis results. Existing historical
execution metadata is preserved; regenerated code cells have no live outputs.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import tempfile

ANALYSIS_NAME = "14_TCRfree_Vd1_Vd2_subtype_scVI_LR"


def build_notebook():
    cells = []
    cells.append({"cell_type": 'markdown', "metadata": {}, "source": r'''
# 14 — TCR-free Vδ1/Vδ2 subtype scVI + LR

## Research question

Conditional on a cell already being a confirmed or predicted γδ T cell,
does its **non-TCR transcriptome** contain enough reproducible information
to predict a receptor-sequence-defined Vδ1 versus Vδ2 subtype?

This is not a replacement for receptor sequencing. It is the second stage
of an evidence-aware cascade:

1. M02/SV02 remain the already-completed upstream γδ-versus-αβ identity work.
2. A valid unambiguous TRD sequence gives a direct sequence-defined subtype.
3. Only cells without valid TRD evidence are passed to this calibrated
   expression model.

The positive class is Vδ1 (`y=1`); Vδ2 is `y=0`. Vδ3, dual-chain and
unresolved cells are not forced into this binary task. In particular,
**low confidence is not Vd3**.

## Confirmatory estimand and evidence language

Development uses strict paired γδ receptor evidence from Song and SPL.
Adult GSE149356 HC1/HC2 is opened only after source model selection,
calibration, and freeze. Because its processed TCR table does not expose an
explicit productive/high-confidence flag, its labels are described as
**cell-linked sequence-supported**, not as fully reprocessed productive gold.

## Reconciled source and historical execution evidence

The generator is authoritative for the revised runnable code. This notebook was
reconciled against the recovered executed 544-cell notebook; historical sources,
outputs and execution counts are retained explicitly in notebook metadata.
Displayed execution outputs have been cleared: the revised code has **not** been
retrained or executed end to end. Frozen adult inference recovery is separate
from training reproduction. The later 2,183-cell analysis remains a separate run.

Existing unsigned historical results are protected: Run All refuses to overwrite
or trust them. Use an author-approved clean project copy with the same input layout
and `GDT_LR_PROJECT`; do not delete the original archive to bypass this guard.
A validated restart uses the same project, inputs, generator, environment and
notebook, followed by **Restart Kernel and Run All**. Completed donor/seed and
final-seed units are verified by hashes, configuration and schema before reuse.
An interruption preserves committed units; an incomplete unit is recomputed.
The existing `completion_manifest.json` records incremental progress. Only its
`completed` state means every declared analysis output has been assembled.

'''.strip()})
    cells.append({"cell_type": 'code', "metadata": {}, "source": r'''
# 1. Imports, deterministic environment, paths, and publication typography
from __future__ import annotations
import gc, gzip, hashlib, json, os, platform, random, re, sys, zlib, tempfile, shutil
from importlib.metadata import version as package_version
from pathlib import Path

import anndata as ad
import joblib
import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp
import scanpy as sc
import scvi
import torch
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, precision_recall_curve,
    roc_auc_score, roc_curve,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ANALYSIS_NAME = "14_TCRfree_Vd1_Vd2_subtype_scVI_LR"
REMOTE_PROJECT = Path("/data/users/kaelynwen/gdT_project/gdt_lr_12model")
PROJECT = Path(os.environ.get("GDT_LR_PROJECT", REMOTE_PROJECT)).resolve()
if not PROJECT.exists():
    candidate = Path.cwd().resolve()
    PROJECT = candidate.parent if candidate.name == "notebooks" else candidate
if str(PROJECT).startswith("/home/kaelynwen"):
    raise RuntimeError("Project work must use /data/users/kaelynwen, not /home/kaelynwen")

SEED = 20260829
N_THREADS = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[key] = str(N_THREADS)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
scvi.settings.seed = SEED
torch.set_num_threads(N_THREADS)
ACCELERATOR = "gpu" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

allowed_fonts = ["Arial", "Helvetica", "Liberation Sans"]
installed_fonts = {f.name for f in fm.fontManager.ttflist}
FIGURE_FONT = next((f for f in allowed_fonts if f in installed_fonts), None)
if FIGURE_FONT is None:
    raise RuntimeError("Install Arial, Helvetica, or Liberation Sans before making figures")
mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": [FIGURE_FONT],
    "font.size": 7,
    "font.weight": "normal", "text.color": "black", "axes.labelcolor": "black",
    "axes.titlecolor": "black", "xtick.color": "black", "ytick.color": "black",
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "axes.grid": False, "axes.spines.right": False, "axes.spines.top": False,
    "legend.frameon": False,
})

# Subtype colors are fixed across every categorical and probability panel.
# Vδ2 is the probability-zero endpoint; Vδ1 is the probability-one endpoint.
VD1_COLOR = "#E78AC3"  # light pink
VD2_COLOR = "#56B4E9"  # light blue
SUBTYPE_CMAP = LinearSegmentedColormap.from_list(
    "Vd2_to_Vd1", [VD2_COLOR, "#F7F7F7", VD1_COLOR]
)

TABLE_DIR = PROJECT / f"results/tables/{ANALYSIS_NAME}"
FIGURE_DIR = PROJECT / f"results/figures/{ANALYSIS_NAME}"
MODEL_DIR = PROJECT / f"models/{ANALYSIS_NAME}"
PREDICTION_DIR = PROJECT / f"predictions/{ANALYSIS_NAME}"
CHECKPOINT_DIR = PROJECT / f"checkpoints/{ANALYSIS_NAME}"
# Validate provenance before creating or writing any analysis outputs.
manifest_path = TABLE_DIR / "completion_manifest.json"
if any(d.exists() and any(d.iterdir()) for d in
       [TABLE_DIR, FIGURE_DIR, MODEL_DIR, PREDICTION_DIR, CHECKPOINT_DIR]):
    if not manifest_path.is_file():
        raise RuntimeError("Existing unsigned results are protected; use an approved clean project copy.")
    existing_manifest = json.loads(manifest_path.read_text())
    if existing_manifest.get("checkpoint_schema") != 1:
        raise RuntimeError("Historical results have no validated restart contract; no files were changed.")

FIGURE_QA_ROWS = []
def save_publication_figure(fig, stem, panel_axes):
    """Save editable PDF/SVG and PNG; record rendered panel geometry."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    panels = []
    for label, axis in panel_axes:
        bbox = axis.get_position()
        panels.append({
            "panel": label,
            "x0_pt": bbox.x0 * fig.get_figwidth() * 72,
            "y0_pt": bbox.y0 * fig.get_figheight() * 72,
            "width_pt": bbox.width * fig.get_figwidth() * 72,
            "height_pt": bbox.height * fig.get_figheight() * 72,
        })
    # Comparable panels in the same rendered row must have equal plot-area
    # width and height within the registered 1.5-pt tolerance.
    alignment_checks = []
    for i, left in enumerate(panels):
        for right in panels[i + 1:]:
            same_row = abs(left["y0_pt"] - right["y0_pt"]) <= 1.5
            if same_row:
                width_delta = abs(left["width_pt"] - right["width_pt"])
                height_delta = abs(left["height_pt"] - right["height_pt"])
                alignment_checks.append({
                    "panels": [left["panel"], right["panel"]],
                    "width_delta_pt": width_delta, "height_delta_pt": height_delta,
                })
                if width_delta > 1.5 or height_delta > 1.5:
                    raise RuntimeError(
                        f"Panel alignment failed for {stem}: {left['panel']}/{right['panel']} "
                        f"width delta={width_delta:.2f} pt, height delta={height_delta:.2f} pt")
    alignment_path = FIGURE_DIR / f"{stem}.panel_alignment.json"
    alignment_path.write_text(json.dumps({
        "figure": stem, "tolerance_pt": 1.5, "panels": panels,
        "checks": alignment_checks, "status": "PASS",
    }, indent=2), encoding="utf-8")
    pdf = FIGURE_DIR / f"{stem}.pdf"
    svg = FIGURE_DIR / f"{stem}.svg"
    png = FIGURE_DIR / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    fig.savefig(png, dpi=600, bbox_inches="tight")
    FIGURE_QA_ROWS.append({
        "figure": stem, "pdf": str(pdf), "svg": str(svg), "png": str(png),
        "panel_alignment": str(alignment_path),
        "collision_audit": "pending_post_execution_PDF_audit",
    })
    return pdf, svg, png, alignment_path
'''.strip()})
    cells.append({"cell_type": 'code', "metadata": {}, "source": r'''
# 2. Prespecified source-only analysis contract
INPUTS = {
    "song": PROJECT / "manifests/Song_D1_cohorts.h5ad",
    "spl": PROJECT / "manifests/CrossAtlas_D2_SPL_cohorts.h5ad",
    "spl_gdt_ir": PROJECT.parent / "Atlas_dataset 2/adata_TILC_TCRgd_onlyseq.h5ad",
    "h1": PROJECT / "features/H1_song_inductive_TCRfree_backbone.csv",
}
EXPECTED_DEVELOPMENT_COUNTS = {"Vd1": 292, "Vd2": 252}
EXPECTED_SPL_SEQUENCE_COUNTS = {"Vd1": 288, "Vd2": 74, "Vd3": 13, "dual": 3}
EXPECTED_DATASET_COUNTS = {
    "Song": {"Vd1": 4, "Vd2": 178},
    "SPL": {"Vd1": 288, "Vd2": 74},
}
EXPECTED_DONORS = ("hd1", "hd2", "hd3", "hd4", "A29", "A31", "A35", "A36", "A37")
MAX_EPOCHS = 200
EARLY_STOPPING_PATIENCE = 20
BATCH_SIZE = 256
ENCODE_BATCH_SIZE = 256
FIXED_SCVI_CONFIG = {
    "config_id": "Z10_L1_H128", "n_latent": 10, "n_layers": 1, "n_hidden": 128,
}
REGISTERED_SEEDS = (20260829, 20260830, 20260831)
PRIMARY_SEED = REGISTERED_SEEDS[0]
C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]
RUN_CONTRACT = {
    "analysis": ANALYSIS_NAME,
    "outcome": "P(Vd1 | gamma-delta T cell, non-TCR transcriptome)",
    "positive_class": "Vd1", "negative_class": "Vd2",
    "primary development excludes orphan": "orphan_TRD, Vd3, dual, mixed and unresolved",
    "existing_identity_models_not_retrained": ["M02", "SV02"],
    "feature_regime": "H1 TCR-free ordered 2000-gene backbone",
    "fixed_scvi_architecture": FIXED_SCVI_CONFIG,
    "registered_seeds": list(REGISTERED_SEEDS),
    "planned_scvi_fits": 30,
    "LR_selection": "grouped LR C selection within each outer-training latent",
    "calibration": "Platt scaling on source-only OOF predictions",
    "query_adaptation_used": False,
    "external_data_used_for_selection": False,
}
'''.strip()})
    cells.append({"cell_type": 'code', "metadata": {}, "source": r'''
# 3. Reusable evidence, matrix, hashing, and weighting functions
def stable_seed(*parts):
    return (SEED + zlib.crc32("|".join(map(str, parts)).encode())) % (2**31 - 1)

def sha256_file(path, block_size=2**20):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()

def parse_trdv_subtype(value):
    if value is None or pd.isna(value):
        return "other_or_unresolved"
    tokens = set(re.findall(r"(?:^|[|,;\s])(TRDV[123])(?:\*\d+)?(?=$|[|,;\s])", str(value).upper()))
    if len(tokens) > 1: return "dual"
    return {"TRDV1": "Vd1", "TRDV2": "Vd2", "TRDV3": "Vd3"}.get(next(iter(tokens), ""), "other_or_unresolved")

def present(value):
    return pd.notna(value) and str(value).strip() not in {"", "nan", "None"}

def combine_ir_slots(frame, slot_prefix, locus, field):
    """Combine exact released IR slots of one locus without guessing chain identity."""
    values = []
    for slot in [1, 2]:
        locus_column = f"IR_{slot_prefix}_{slot}_locus"
        field_column = f"IR_{slot_prefix}_{slot}_{field}"
        if locus_column not in frame or field_column not in frame:
            raise KeyError(f"Released SPL receptor metadata lacks {locus_column}/{field_column}")
        values.append(frame[field_column].where(frame[locus_column].astype(str).eq(locus)))
    temporary = pd.concat(values, axis=1)
    return temporary.apply(
        lambda row: "|".join(dict.fromkeys(str(value) for value in row if present(value))) or pd.NA,
        axis=1,
    )

def donor_subtype_balanced_weights(frame):
    strata = frame["donor"].astype(str) + "::" + frame["subtype"].astype(str)
    counts = strata.value_counts()
    weights = strata.map(lambda x: 1.0 / counts[x]).astype(float)
    return weights * (len(weights) / weights.sum())

def raw_matrix(adata, cells, genes):
    obj = adata[cells, genes]
    matrix = obj.layers["counts"] if "counts" in obj.layers else obj.X
    matrix = matrix.copy()
    values = matrix.data if sp.issparse(matrix) else np.asarray(matrix)
    if not np.isfinite(values).all() or np.any(values < 0) or not np.allclose(values, np.rint(values)):
        raise AssertionError("scVI input must be finite non-negative integer counts")
    return matrix.tocsr() if sp.issparse(matrix) else sp.csr_matrix(matrix)

def clone_groups(obs):
    candidates = [c for c in ["TRD_cdr3", "TRD_cdr3_nt", "TRG_cdr3", "TRG_cdr3_nt"] if c in obs]
    if not candidates:
        raise KeyError("No receptor CDR3 fields available; clone-group leakage cannot be audited")
    def one(row):
        parts = [str(row[c]) for c in candidates if pd.notna(row[c]) and str(row[c]).strip()]
        if not parts:
            raise ValueError("A strict paired cell lacks all available CDR3 values")
        return f"{row['dataset']}::{row['donor']}::" + "::".join(parts)
    return obs.apply(one, axis=1)

def make_development_table(adata, dataset_name):
    obs = adata.obs.copy()
    obs["dataset"] = dataset_name
    if "TRD_v_gene" not in obs:
        raise KeyError(f"{dataset_name} lacks TRD_v_gene")
    obs["subtype"] = obs["TRD_v_gene"].map(parse_trdv_subtype)
    primary = (
        obs["immutable_cohort"].astype(str).eq("gold_B_paired_gdT")
        & obs["qc_Tlineage_pass"].fillna(False).astype(bool)
        & obs["subtype"].isin(["Vd1", "Vd2"])
    )
    out = obs.loc[primary].copy()
    out["y"] = out["subtype"].eq("Vd1").astype(int)
    out["clone_group"] = clone_groups(out)
    return out

# Each write is committed atomically; the manifest is the final commit marker.
def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)

def atomic_csv(frame, path, **kwargs):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            frame.to_csv(handle, **kwargs)
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)

def canonical_hash(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()

def initialize_progress(contract):
    signature = canonical_hash(contract)
    if manifest_path.exists():
        progress = json.loads(manifest_path.read_text())
        if (progress.get("checkpoint_schema") != 1 or
            progress.get("run_signature") != signature or progress.get("contract") != contract):
            raise RuntimeError("Run/input/cohort/feature/environment signature changed; refusing cached results.")
    else:
        progress = {"analysis": ANALYSIS_NAME, "checkpoint_schema": 1,
                    "run_signature": signature, "contract": contract,
                    "status": "in_progress", "units": {}}
    for directory in [TABLE_DIR, FIGURE_DIR, MODEL_DIR, PREDICTION_DIR, CHECKPOINT_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    progress["status"] = "in_progress"
    atomic_json(manifest_path, progress)
    return progress

def validated_unit(key, paths):
    record = progress["units"].get(key)
    if record is None: return None  # no commit: incomplete unit must be recomputed
    expected = [str(Path(path).relative_to(PROJECT)) for path in paths]
    if record.get("paths") != expected:
        raise RuntimeError(f"Checkpoint schema/path mismatch: {key}")
    for relative, expected_hash in record["sha256"].items():
        path = PROJECT / relative
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise RuntimeError(f"Missing or corrupt committed checkpoint: {path}")
    actual_files = []
    for path in paths:
        path = Path(path)
        actual_files.extend(sorted(path.rglob("*")) if path.is_dir() else [path])
    actual_names = {str(path.relative_to(PROJECT)) for path in actual_files if path.is_file()}
    if actual_names != set(record["sha256"]):
        raise RuntimeError(f"Checkpoint file inventory mismatch: {key}")
    if not record["sha256"]: raise RuntimeError(f"Empty checkpoint: {key}")
    return record

def commit_unit(key, paths, **details):
    files = []
    for path in paths:
        path = Path(path)
        files.extend(sorted(path.rglob("*")) if path.is_dir() else [path])
    hashes = {str(path.relative_to(PROJECT)): sha256_file(path)
              for path in files if path.is_file()}
    if not hashes: raise RuntimeError(f"No files to commit: {key}")
    progress["units"][key] = {"paths": [str(Path(p).relative_to(PROJECT)) for p in paths],
                               "sha256": hashes, **details}
    progress["status"] = "in_progress"
    atomic_json(manifest_path, progress)

def validate_oof(frame, scores, test, seed, donor):
    columns = ["dataset", "donor", "subtype", "y", "clone_group", "p_raw",
               "outer_donor", "seed", "selected_C"]
    if list(frame.columns) != columns or not frame.index.is_unique or list(frame.index) != list(test.index):
        raise RuntimeError("OOF schema/cell order does not match this held-out donor")
    for column in ["dataset", "donor", "subtype", "y", "clone_group"]:
        if not frame[column].astype(str).equals(test[column].astype(str)):
            raise RuntimeError(f"OOF labels/cohort mismatch: {column}")
    if not (frame["seed"].eq(seed).all() and frame["outer_donor"].eq(donor).all()
            and np.isfinite(frame["p_raw"]).all() and frame["p_raw"].between(0, 1).all()):
        raise RuntimeError("Invalid OOF seed/donor/probabilities")
    required = ["seed", "outer_donor", "selected_C", "C", "pooled_inner_OOF_average_precision"]
    if list(scores.columns) != required or len(scores) != len(C_GRID) or set(scores["C"]) != set(C_GRID):
        raise RuntimeError("Invalid C-score checkpoint schema/grid")
    if not (scores["seed"].eq(seed).all() and scores["outer_donor"].eq(donor).all()
            and np.isfinite(scores["pooled_inner_OOF_average_precision"]).all()
            and scores["pooled_inner_OOF_average_precision"].between(0, 1).all()):
        raise RuntimeError("Invalid C-score values")
    best = float(scores.sort_values(["pooled_inner_OOF_average_precision", "C"],
                                   ascending=[False, True]).iloc[0]["C"])
    if not (frame["selected_C"].eq(best).all() and scores["selected_C"].eq(best).all()):
        raise RuntimeError("Selected C does not match registered tie-breaking rule")

'''.strip()})
    cells.append({"cell_type": 'code', "metadata": {}, "source": r'''
# 4. Load and audit development data only
missing = [str(p) for p in INPUTS.values() if not p.is_file()]
if missing:
    raise FileNotFoundError("Missing source inputs:\n" + "\n".join(missing))
song = ad.read_h5ad(INPUTS["song"])
spl = ad.read_h5ad(INPUTS["spl"])

# The frozen SPL expression cohort contains chain counts but not V/J/CDR3
# strings. Those fields remain in the released γδ-receptor object. Link the
# two objects by the exact released barcode stored as spl.obs.barcode_raw.
# This reconstructs receptor labels; it never uses receptor genes as model
# features and therefore preserves the TCR-free predictor contract.
if "barcode_raw" not in spl.obs:
    raise KeyError("SPL expression cohort lacks the registered barcode_raw crosswalk")
spl_ir_backed = ad.read_h5ad(INPUTS["spl_gdt_ir"], backed="r")
required_ir_obs = {"organ_code", "IR_VJ_1_locus", "IR_VDJ_1_locus"}
missing_ir_obs = sorted(required_ir_obs - set(spl_ir_backed.obs.columns))
if missing_ir_obs:
    raise KeyError(f"Released SPL γδ-receptor object lacks {missing_ir_obs}")
spl_ir_obs = spl_ir_backed.obs.loc[
    spl_ir_backed.obs["organ_code"].astype(str).eq("SPL")].copy()
if spl_ir_obs.index.duplicated().any():
    raise AssertionError("Released SPL γδ-receptor object has duplicate barcodes")
released_fields = {}
for chain, slot_prefix, locus in [("TRG", "VJ", "TRG"), ("TRD", "VDJ", "TRD")]:
    for field in ["v_gene", "j_gene", "cdr3", "cdr3_nt"]:
        released_fields[f"{chain}_{field}"] = combine_ir_slots(
            spl_ir_obs, slot_prefix, locus, field)
released_spl = pd.DataFrame(released_fields, index=spl_ir_obs.index)
spl_ir_backed.file.close()

spl_barcodes = spl.obs["barcode_raw"].astype(str)
for field in released_spl.columns:
    spl.obs[field] = spl_barcodes.map(released_spl[field])
spl_sequence_subtype = spl.obs["TRD_v_gene"].map(parse_trdv_subtype)
spl_gold = spl.obs["immutable_cohort"].astype(str).eq("gold_B_paired_gdT")
spl_gold_sequence_counts = spl_sequence_subtype.loc[spl_gold].value_counts().to_dict()
assert spl_gold_sequence_counts == EXPECTED_SPL_SEQUENCE_COUNTS, (
    spl_gold_sequence_counts, EXPECTED_SPL_SEQUENCE_COUNTS)
spl_required_sequence = spl.obs.loc[
    spl_gold, ["TRD_v_gene", "TRD_cdr3", "TRD_cdr3_nt",
               "TRG_v_gene", "TRG_cdr3", "TRG_cdr3_nt"]]
if not spl_required_sequence.apply(lambda column: column.map(present)).all().all():
    raise AssertionError("One or more SPL paired γδ cells failed exact released-sequence linkage")
spl_linkage_audit = pd.DataFrame([
    {"audit": "SPL_expression_cells", "n": int(spl.n_obs)},
    {"audit": "released_SPL_receptor_rows", "n": int(len(spl_ir_obs))},
    {"audit": "exact_barcode_overlap", "n": int(spl_barcodes.isin(released_spl.index).sum())},
    {"audit": "gold_B_paired_gdT", "n": int(spl_gold.sum())},
    *[{"audit": f"gold_B_paired_gdT_{subtype}", "n": int(count)}
      for subtype, count in EXPECTED_SPL_SEQUENCE_COUNTS.items()],
])

h1 = pd.read_csv(INPUTS["h1"])["gene"].astype(str).tolist()
assert len(h1) == 2000 and len(h1) == len(set(h1))
assert not any(g.upper().startswith(("TRA", "TRB", "TRG", "TRD")) for g in h1)
for name, obj in [("Song", song), ("SPL", spl)]:
    absent = sorted(set(h1) - set(obj.var_names.astype(str)))
    if absent:
        raise KeyError(f"{name} lacks {len(absent)} ordered H1 genes")

song_table = make_development_table(song, "Song")
spl_table = make_development_table(spl, "SPL")
development = pd.concat([song_table, spl_table], axis=0)
development.index = development["dataset"].astype(str) + "::" + development.index.astype(str)
observed = development["subtype"].value_counts().to_dict()
assert observed == EXPECTED_DEVELOPMENT_COUNTS, (observed, EXPECTED_DEVELOPMENT_COUNTS)
observed_by_dataset = development.groupby(["dataset", "subtype"]).size().unstack(fill_value=0).to_dict("index")
assert observed_by_dataset == EXPECTED_DATASET_COUNTS
assert set(development["donor"].astype(str)) == set(EXPECTED_DONORS)
assert development["donor"].astype(str).nunique() == len(EXPECTED_DONORS)
assert not development["clone_group"].duplicated().all()

development_counts = development.groupby(["dataset", "donor", "subtype"]).size().rename("n").reset_index()
source_generator = PROJECT / "544_cell_source_archive" / "build_14_TCRfree_Vd1_Vd2_subtype_scVI_LR.py"
if not source_generator.is_file():
    raise FileNotFoundError("Matching authoritative generator is required for the run signature")
software_versions = {name: package_version(name) for name in
    ["scvi-tools", "torch", "anndata", "pandas", "numpy", "scipy", "scanpy", "scikit-learn", "joblib"]}
software_versions["python"] = platform.python_version()
software_versions["accelerator"] = ACCELERATOR
software_versions["cuda"] = str(torch.version.cuda)
software_versions["threads"] = N_THREADS
contract = {"run_contract": RUN_CONTRACT, "input_sha256": {k: sha256_file(v) for k, v in INPUTS.items()},
            "generator_sha256": sha256_file(source_generator), "software_versions": software_versions,
            "ordered_genes": h1, "C_grid": C_GRID, "max_epochs": MAX_EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "batch_size": BATCH_SIZE, "encode_batch_size": ENCODE_BATCH_SIZE,
            "cohort": development[["dataset", "donor", "subtype", "y", "clone_group"]]
                       .reset_index().astype(str).to_dict("split")}
progress = initialize_progress(contract)
atomic_csv(spl_linkage_audit, TABLE_DIR / "SPL_released_TRD_sequence_linkage_audit.csv", index=False)
atomic_csv(development_counts, TABLE_DIR / "development_sequence_defined_subtype_counts.csv", index=False)
print(development_counts.to_string(index=False))
'''.strip()})
    cells.append({"cell_type": 'markdown', "metadata": {}, "source": r'''
## What donor holdout does and does not protect

Each outer test donor is absent from encoder and classifier fitting. Inner LR
selection reuses the outer-training latent representation, so an inner validation
donor has contributed to encoder fitting. This is not fully nested representation
selection. The 2,000-gene backbone was inherited from earlier Song analyses.
Donor holdout also keeps donor-scoped receptor `clone_group` values on one side
of each fold. Some Song donors have only Vδ2; hyperparameters use pooled inner
OOF predictions because single-class donor discrimination is undefined.

'''.strip()})
    cells.append({"cell_type": 'code', "metadata": {}, "source": r'''
# 5. Build source count objects and strict-inductive scVI/LR helpers
source_lookup = {"Song": song, "SPL": spl}

def build_count_adata(frame, genes):
    matrices, obs_parts = [], []
    for dataset_name, part in frame.groupby("dataset", sort=False):
        original_cells = part.index.str.split("::", n=1).str[1].tolist()
        matrices.append(raw_matrix(source_lookup[dataset_name], original_cells, genes))
        keep = part[["dataset", "donor", "subtype", "y", "clone_group"]].copy()
        obs_parts.append(keep)
    matrix = sp.vstack(matrices, format="csr")
    obs = pd.concat(obs_parts, axis=0)
    positions = obs.index.get_indexer(frame.index)
    if (positions < 0).any(): raise AssertionError("Missing source cell during count alignment")
    matrix = matrix[positions]
    obs = obs.loc[frame.index]
    return ad.AnnData(X=matrix, obs=obs, var=pd.DataFrame(index=pd.Index(genes, name="gene")))

def fit_scvi(frame, config, seed):
    scvi.settings.seed = int(seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    rep = build_count_adata(frame, h1)
    scvi.model.SCVI.setup_anndata(rep)  # deliberately no batch_key
    model = scvi.model.SCVI(
        rep, n_latent=config["n_latent"], n_layers=config["n_layers"],
        n_hidden=config["n_hidden"], gene_likelihood="nb", dispersion="gene",
        latent_distribution="normal",
    )
    model.train(
        max_epochs=MAX_EPOCHS, accelerator=ACCELERATOR, devices=1,
        batch_size=BATCH_SIZE, early_stopping=True,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        datasplitter_kwargs={"num_workers": 0},
    )
    return rep, model

def training_latent(model, rep):
    z = np.asarray(model.get_latent_representation(adata=rep, batch_size=ENCODE_BATCH_SIZE))
    return pd.DataFrame(z, index=rep.obs_names)

@torch.inference_mode()
def encode_frozen_matrix(model, matrix, index):
    model.to_device(str(DEVICE)); model.module.eval()
    chunks = []
    for start in range(0, matrix.shape[0], ENCODE_BATCH_SIZE):
        dense = matrix[start:start + ENCODE_BATCH_SIZE].toarray()
        x = torch.as_tensor(dense, dtype=torch.float32, device=DEVICE)
        batch_index = torch.zeros((x.shape[0], 1), dtype=torch.int64, device=DEVICE)
        output = model.module.inference(x=x, batch_index=batch_index, n_samples=1)
        z = output["qz"].loc if "qz" in output else output["qz_m"]
        chunks.append(z.detach().cpu().numpy())
    return pd.DataFrame(np.vstack(chunks), index=index)

def fit_lr(z, frame, C, seed):
    classifier = Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(C=C, solver="liblinear", max_iter=4000, random_state=seed)),
    ])
    classifier.fit(z.loc[frame.index], frame["y"],
                   lr__sample_weight=donor_subtype_balanced_weights(frame))
    return classifier

def clear_accelerator():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
'''.strip()})
    cells.append({"cell_type": 'code', "metadata": {}, "source": r'''
# 6. Fixed-architecture, three-seed donor-LOGO evaluation
logo = LeaveOneGroupOut()
donors = development["donor"].astype(str).drop_duplicates().tolist()

def select_lr_C(z, frame, selection_seed):
    """Donor-grouped LR C selection on a frozen outer-training latent."""
    rows = []
    groups = frame["donor"].astype(str).to_numpy()
    for C in C_GRID:
        inner_parts = []
        for train_pos, valid_pos in logo.split(z, frame["y"], groups):
            inner_train = frame.iloc[train_pos]
            inner_valid = frame.iloc[valid_pos]
            head = fit_lr(z, inner_train, C, stable_seed(selection_seed, "inner_lr", C, groups[valid_pos][0]))
            inner_parts.append(pd.DataFrame({
                "y": inner_valid["y"].to_numpy(),
                "p": head.predict_proba(z.loc[inner_valid.index])[:, 1],
            }))
        pooled = pd.concat(inner_parts, ignore_index=True)
        rows.append({"C": float(C), "pooled_inner_OOF_average_precision":
                     average_precision_score(pooled["y"], pooled["p"])})
    scores = pd.DataFrame(rows).sort_values(
        ["pooled_inner_OOF_average_precision", "C"], ascending=[False, True])
    return float(scores.iloc[0]["C"]), scores

outer_predictions, lr_selection_rows = [], []
for registered_seed in REGISTERED_SEEDS:
    for outer_donor in donors:
        checkpoint = CHECKPOINT_DIR / f"seed{registered_seed}_{outer_donor}_OOF.csv"
        score_checkpoint = CHECKPOINT_DIR / f"seed{registered_seed}_{outer_donor}_C_scores.csv"
        fit = development.loc[development["donor"].astype(str).ne(outer_donor)]
        test = development.loc[development["donor"].astype(str).eq(outer_donor)]
        key = checkpoint.name
        if validated_unit(key, [checkpoint, score_checkpoint]) is not None:
            restored = pd.read_csv(checkpoint, index_col=0)
            restored_scores = pd.read_csv(score_checkpoint)
            validate_oof(restored, restored_scores, test, registered_seed, outer_donor)
            outer_predictions.append(restored)
            lr_selection_rows.extend(restored_scores.to_dict("records"))
            continue
        assert set(fit["clone_group"]).isdisjoint(set(test["clone_group"]))
        rep, vae = fit_scvi(fit, FIXED_SCVI_CONFIG, stable_seed(registered_seed, "outer", outer_donor))
        z_fit = training_latent(vae, rep)
        selected_C, C_scores = select_lr_C(z_fit, fit, stable_seed(registered_seed, outer_donor))
        for row in C_scores.to_dict("records"):
            lr_selection_rows.append({"seed": registered_seed, "outer_donor": outer_donor,
                                      "selected_C": selected_C, **row})
        score_frame = pd.DataFrame([
            {"seed": registered_seed, "outer_donor": outer_donor,
             "selected_C": selected_C, **row}
            for row in C_scores.to_dict("records")
        ])
        test_obj = build_count_adata(test, h1)
        z_test = encode_frozen_matrix(vae, test_obj.X.tocsr(), test.index)
        head = fit_lr(z_fit, fit, selected_C, stable_seed(registered_seed, "outer_lr", outer_donor))
        frame = test[["dataset", "donor", "subtype", "y", "clone_group"]].copy()
        frame["p_raw"] = head.predict_proba(z_test)[:, 1]
        frame["outer_donor"] = outer_donor
        frame["seed"] = registered_seed
        frame["selected_C"] = selected_C
        validate_oof(frame, score_frame, test, registered_seed, outer_donor)
        atomic_csv(score_frame, score_checkpoint, index=False)
        atomic_csv(frame, checkpoint)
        commit_unit(key, [checkpoint, score_checkpoint], seed=int(registered_seed), donor=outer_donor)
        outer_predictions.append(frame)
        del vae, rep, z_fit, z_test; clear_accelerator()

expected_units = {f"seed{seed}_{donor}_OOF.csv" for seed in REGISTERED_SEEDS for donor in donors}
if not expected_units.issubset(progress["units"]):
    raise RuntimeError("Cannot aggregate until every required donor/seed unit is committed")
all_source_oof = pd.concat(outer_predictions)
assert pd.MultiIndex.from_arrays([
    all_source_oof.index.astype(str), all_source_oof["seed"].astype(int)
]).is_unique
for registered_seed in REGISTERED_SEEDS:
    seed_oof = all_source_oof.loc[all_source_oof["seed"].astype(int).eq(registered_seed)]
    assert set(seed_oof.index) == set(development.index) and seed_oof.index.is_unique
atomic_csv(all_source_oof, PREDICTION_DIR / "development_three_seed_donor_held_out_OOF_predictions.csv")
atomic_csv(pd.DataFrame(lr_selection_rows), TABLE_DIR / "outer_training_grouped_LR_C_selection.csv", index=False)

seed_stability = pd.DataFrame([
    {"seed": registered_seed,
     "average_precision": average_precision_score(part["y"], part["p_raw"]),
     "roc_auc": roc_auc_score(part["y"], part["p_raw"])}
    for registered_seed, part in all_source_oof.groupby("seed")
])
atomic_csv(seed_stability, TABLE_DIR / "development_seed_stability.csv", index=False)
source_oof = all_source_oof.loc[all_source_oof["seed"].astype(int).eq(PRIMARY_SEED)].copy()
source_oof = source_oof.loc[development.index]
atomic_csv(source_oof, PREDICTION_DIR / "development_primary_seed_OOF_predictions.csv")
'''.strip()})
    cells.append({"cell_type": 'code', "metadata": {}, "source": r'''
# 7. Source-only calibration, three final seed fits, and source freeze
# Equivalent source-only Platt scaling is used instead of CalibratedClassifierCV with cv="prefit".
eps = 1e-6
oof_logit = np.log(np.clip(source_oof[["p_raw"]].to_numpy(), eps, 1-eps) /
                   (1 - np.clip(source_oof[["p_raw"]].to_numpy(), eps, 1-eps)))
calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=4000, random_state=SEED)
calibrator.fit(oof_logit, source_oof["y"])
source_oof["p_calibrated"] = calibrator.predict_proba(oof_logit)[:, 1]
atomic_csv(source_oof, PREDICTION_DIR / "development_source_only_OOF_calibrated_predictions.csv")

primary_final_model = None
final_selection_rows = []
for registered_seed in REGISTERED_SEEDS:
    seed_model_dir = MODEL_DIR / f"seed_{registered_seed}_scVI"
    seed_head_path = MODEL_DIR / f"seed_{registered_seed}_latent_LR.joblib"
    record = validated_unit(seed_head_path.name, [seed_model_dir, seed_head_path])
    if record is not None:
        bundle = joblib.load(seed_head_path)
        if (bundle["genes"] != h1 or bundle["config"] != FIXED_SCVI_CONFIG
                or bundle["seed"] != registered_seed or bundle["C"] not in C_GRID):
            raise RuntimeError("Final-seed model schema/configuration mismatch")
        final_rep_seed = build_count_adata(development, h1)
        final_vae_seed = scvi.model.SCVI.load(seed_model_dir, adata=final_rep_seed)
        final_head_seed, final_C_seed = bundle["classifier"], bundle["C"]
        final_C_scores = pd.DataFrame(record["C_scores"])
        if (list(final_C_scores.columns) != ["C", "pooled_inner_OOF_average_precision"]
                or len(final_C_scores) != len(C_GRID) or set(final_C_scores["C"]) != set(C_GRID)
                or not np.isfinite(final_C_scores["pooled_inner_OOF_average_precision"]).all()
                or not final_C_scores["pooled_inner_OOF_average_precision"].between(0, 1).all()):
            raise RuntimeError("Final-seed selection-score schema mismatch")
        selected = float(final_C_scores.sort_values(
            ["pooled_inner_OOF_average_precision", "C"], ascending=[False, True]).iloc[0]["C"])
        if selected != final_C_seed:
            raise RuntimeError("Final-seed selected C differs from saved selection scores")
    else:
        final_rep_seed, final_vae_seed = fit_scvi(
            development, FIXED_SCVI_CONFIG, stable_seed(registered_seed, "final"))
        final_z_seed = training_latent(final_vae_seed, final_rep_seed)
        final_C_seed, final_C_scores = select_lr_C(
            final_z_seed, development, stable_seed(registered_seed, "final_C"))
        final_head_seed = fit_lr(
            final_z_seed, development, final_C_seed, stable_seed(registered_seed, "final_lr"))
        equivalence_cells = final_rep_seed.obs_names[:min(32, final_rep_seed.n_obs)]
        reference_z = np.asarray(final_vae_seed.get_latent_representation(
            adata=final_rep_seed, indices=np.arange(len(equivalence_cells)),
            batch_size=min(ENCODE_BATCH_SIZE, len(equivalence_cells))))
        manual_z = encode_frozen_matrix(
            final_vae_seed, final_rep_seed[equivalence_cells].X.tocsr(), equivalence_cells).to_numpy()
        if not np.allclose(reference_z, manual_z, rtol=1e-4, atol=1e-4):
            raise RuntimeError("Frozen encoder failed equivalence check for this scvi-tools version")
        temporary_dir = Path(tempfile.mkdtemp(dir=MODEL_DIR, prefix=seed_model_dir.name + ".", suffix=".tmp"))
        try:
            final_vae_seed.save(temporary_dir, overwrite=True, save_anndata=True)
            # No manifest commit exists: any destination here is this run's incomplete unit.
            if seed_model_dir.exists(): shutil.rmtree(seed_model_dir)
            os.replace(temporary_dir, seed_model_dir)
        finally:
            if temporary_dir.exists(): shutil.rmtree(temporary_dir)
        fd, temporary_head = tempfile.mkstemp(dir=MODEL_DIR, prefix=seed_head_path.name + ".", suffix=".tmp")
        os.close(fd)
        try:
            joblib.dump({"classifier": final_head_seed,
                         "calibrator": calibrator if registered_seed == PRIMARY_SEED else None,
                         "genes": h1, "positive_class": "Vd1", "negative_class": "Vd2",
                         "config": FIXED_SCVI_CONFIG, "C": final_C_seed, "seed": registered_seed}, temporary_head)
            os.replace(temporary_head, seed_head_path)
        finally:
            if os.path.exists(temporary_head): os.unlink(temporary_head)
        commit_unit(seed_head_path.name, [seed_model_dir, seed_head_path],
                    seed=int(registered_seed), C_scores=final_C_scores.to_dict("records"))
        del final_z_seed
    if registered_seed == PRIMARY_SEED:
        primary_final_model = (final_rep_seed, final_vae_seed, final_head_seed, final_C_seed)
    else:
        del final_rep_seed, final_vae_seed, final_head_seed
        clear_accelerator()
    for row in final_C_scores.to_dict("records"):
        final_selection_rows.append({"seed": registered_seed, "selected_C": final_C_seed, **row})
if not all(f"seed_{seed}_latent_LR.joblib" in progress["units"] for seed in REGISTERED_SEEDS):
    raise RuntimeError("Final seed fits are incomplete")
atomic_csv(pd.DataFrame(final_selection_rows), TABLE_DIR / "final_grouped_LR_C_selection.csv", index=False)
if primary_final_model is None:
    raise RuntimeError("Primary final seed model was not retained")
final_rep, final_vae, final_head, final_C = primary_final_model

source_freeze = {
    **RUN_CONTRACT, "fixed_scvi": FIXED_SCVI_CONFIG, "primary_seed": PRIMARY_SEED,
    "primary_final_C": final_C,
    "expected_development_counts": EXPECTED_DEVELOPMENT_COUNTS,
    "input_sha256": {k: sha256_file(v) for k, v in INPUTS.items()},
    "software_versions": software_versions,
    "rule_based_sequence_precedence": True,
}
SOURCE_FREEZE_PATH = TABLE_DIR / "source_freeze.json"
atomic_json(SOURCE_FREEZE_PATH, source_freeze)
assert SOURCE_FREEZE_PATH.is_file()
print("SOURCE FROZEN. External adult data may now be opened:", SOURCE_FREEZE_PATH)
'''.strip()})
    cells.append({"cell_type": 'markdown', "metadata": {}, "source": r'''
## Frozen external evaluation

Everything above this point is development-only. The cells below are not
integrated with Song/SPL and do not update the encoder, linear head,
calibrator, genes, or thresholds. HC1 and HC2 are reported separately and
together; donor-macro summaries prevent the larger donor from dominating.
'''.strip()})
    cells.append({"cell_type": 'code', "metadata": {}, "source": r'''
# 9. Open adult HC1/HC2 only after source_freeze exists
EXTERNAL_ROOT = Path(os.environ.get("GDT_EXTERNAL_ONE", "/data/users/kaelynwen/Subtype_evaluation/External_1")).resolve()
if str(EXTERNAL_ROOT).startswith("/home/kaelynwen"):
    raise RuntimeError("External project data must live under /data/users/kaelynwen")
EXTERNAL_INPUTS = {
    "HC1_matrix": EXTERNAL_ROOT / "PB1:HC1/GSM4498553_PB1matrix.mtx",
    "HC1_barcodes": EXTERNAL_ROOT / "PB1:HC1/GSM4498553_PB1barcodes.tsv.gz",
    "HC1_features": EXTERNAL_ROOT / "PB1:HC1/GSM4498553_PB1features.tsv",
    "HC2_matrix": EXTERNAL_ROOT / "PB2:HC2/GSM4498554_PB2matrix.mtx.gz",
    "HC2_barcodes": EXTERNAL_ROOT / "PB2:HC2/GSM4498554_PB2barcodes.tsv.gz",
    "HC2_features": EXTERNAL_ROOT / "PB2:HC2/GSM4498554_PB2features.tsv.gz",
    "tcr": EXTERNAL_ROOT / "TCR_Label/GSE149356_Processed_TCR.csv.gz",
    "metadata": EXTERNAL_ROOT / "Metadata/GSE149356_family .soft.gz",
}
missing = [str(p) for p in EXTERNAL_INPUTS.values() if not p.is_file()]
if missing: raise FileNotFoundError("Missing external inputs:\n" + "\n".join(missing))

external_hashes = {key: sha256_file(path) for key, path in EXTERNAL_INPUTS.items()}
if "external_input_sha256" in progress and progress["external_input_sha256"] != external_hashes:
    raise RuntimeError("External inputs changed; refusing to mix results")
progress["external_input_sha256"] = external_hashes
progress["status"] = "in_progress"
atomic_json(manifest_path, progress)

def read_text_table(path, **kwargs):
    return pd.read_csv(path, sep="\t", header=None, **kwargs)

def read_adult_rna(donor):
    matrix = scipy.io.mmread(EXTERNAL_INPUTS[f"{donor}_matrix"]).T.tocsr()
    barcodes = read_text_table(
        EXTERNAL_INPUTS[f"{donor}_barcodes"]
    )[0].astype(str).to_numpy()
    features = read_text_table(EXTERNAL_INPUTS[f"{donor}_features"])
    gene_column = 1 if features.shape[1] > 1 else 0
    source_genes = features[gene_column].astype(str).to_numpy()
    if matrix.shape != (len(barcodes), len(source_genes)):
        raise AssertionError(f"{donor} matrix/barcode/feature dimensions disagree")

    # Align the adult matrix to the frozen ordered 2,000-gene H1 panel.
    # Source genes outside H1 are omitted. Duplicate source symbols map
    # to the same H1 column and are summed by the COO-to-CSR conversion.
    target_lookup = {gene: position for position, gene in enumerate(h1)}
    source_to_target = np.array(
        [target_lookup.get(gene, -1) for gene in source_genes], dtype=np.int64
    )
    source_keep = source_to_target >= 0
    source_subset = matrix[:, source_keep].tocoo()
    target_columns = source_to_target[source_keep][source_subset.col]
    aligned_matrix = sp.coo_matrix(
        (source_subset.data, (source_subset.row, target_columns)),
        shape=(matrix.shape[0], len(h1)),
    ).tocsr()
    aligned_matrix.sum_duplicates()

    observed_h1 = set(source_genes[source_keep])
    absent = [gene for gene in h1 if gene not in observed_h1]
    print(
        f"{donor}: RNA cells={matrix.shape[0]:,}; "
        f"source features={matrix.shape[1]:,}; "
        f"H1 observed={len(h1) - len(absent):,}/{len(h1):,}; "
        f"H1 zero-filled={len(absent):,}"
    )
    print(f"{donor}: zero-filled examples={absent[:10]}")

    obj = ad.AnnData(
        X=aligned_matrix,
        obs=pd.DataFrame(index=pd.Index(barcodes, name="barcode")),
        var=pd.DataFrame(index=pd.Index(h1, name="gene")),
    )
    obj.obs["donor"] = donor
    assert obj.shape == (len(barcodes), len(h1))
    assert obj.var_names.tolist() == list(h1)
    assert obj.obs_names.is_unique
    assert obj.var_names.is_unique
    return obj

adult_rna = {donor: read_adult_rna(donor) for donor in ["HC1", "HC2"]}
'''.strip()})
    cells.append({"cell_type": 'code', "metadata": {}, "source": r'''
# 10. Join cell-linked sequence-supported adult subtype evidence
tcr = pd.read_csv(EXTERNAL_INPUTS["tcr"])
required_tcr = ["barcode", "donor", "chain_TRD", "v_gene_TRD", "cdr3_TRD", "cdr3_nt_TRD",
                "chain_TRG", "v_gene_TRG", "cdr3_TRG", "cdr3_nt_TRG"]
absent = [c for c in required_tcr if c not in tcr]
if absent: raise KeyError(f"Processed TCR table lacks: {absent}")
tcr = tcr.loc[tcr["donor"].astype(str).isin(["HC1", "HC2"])].copy()
tcr["subtype"] = tcr["v_gene_TRD"].map(parse_trdv_subtype)
tcr["barcode_core"] = (tcr["barcode"].astype(str).str.replace(r"^(HC1|HC2)_", "", regex=True)
                         .str.replace(r"-\d+$", "", regex=True))
for donor in ["HC1", "HC2"]:
    rna = adult_rna[donor]
    core_to_rna = pd.Series(rna.obs_names, index=rna.obs_names.str.replace(r"-\d+$", "", regex=True))
    donor_mask = tcr["donor"].astype(str).eq(donor)
    tcr.loc[donor_mask, "rna_barcode"] = tcr.loc[donor_mask, "barcode_core"].map(core_to_rna)

has_trd = tcr["chain_TRD"].astype(str).eq("TRD") & tcr["cdr3_TRD"].notna() & tcr["cdr3_nt_TRD"].notna()
has_trg = tcr["chain_TRG"].astype(str).eq("TRG") & tcr["cdr3_TRG"].notna() & tcr["cdr3_nt_TRG"].notna()
binary = tcr["subtype"].isin(["Vd1", "Vd2"])
tcr["paired_TRG_TRD_primary"] = has_trd & has_trg & binary
tcr["TRD_only_sensitivity"] = has_trd & binary
tcr["y"] = tcr["subtype"].eq("Vd1").astype(int)
coverage = tcr.groupby("donor")[["paired_TRG_TRD_primary", "TRD_only_sensitivity"]].sum().reset_index()
coverage.to_csv(TABLE_DIR / "adult_sequence_label_coverage.csv", index=False)
'''.strip()})
    cells.append({"cell_type": 'code', "metadata": {}, "source": r'''
# 11. Frozen-encoder adult prediction; no query adaptation or recalibration
# Score all adult RNA cells for the descriptive UMAP. Quantitative metrics
# remain restricted to cells with sequence-supported Vd1/Vd2 labels.
adult_all_predictions, adult_latent_parts = [], []
for donor in ["HC1", "HC2"]:
    rna = adult_rna[donor][:, h1]
    matrix = rna.X.tocsr() if sp.issparse(rna.X) else sp.csr_matrix(rna.X)
    cell_ids = pd.Index([f"{donor}::{x}" for x in rna.obs_names.astype(str)])
    z = encode_frozen_matrix(final_vae, matrix, cell_ids)
    prediction = pd.DataFrame({
        "cell_id": cell_ids, "donor": donor,
        "rna_barcode": rna.obs_names.astype(str),
        "p_raw": final_head.predict_proba(z)[:, 1],
    }).set_index("cell_id")
    raw_logit = np.log(np.clip(prediction[["p_raw"]].to_numpy(), eps, 1-eps) /
                       (1 - np.clip(prediction[["p_raw"]].to_numpy(), eps, 1-eps)))
    prediction["p_calibrated"] = calibrator.predict_proba(raw_logit)[:, 1]
    adult_all_predictions.append(prediction)
    adult_latent_parts.append(z)
adult_all_predictions = pd.concat(adult_all_predictions)
adult_all_latent = pd.concat(adult_latent_parts).loc[adult_all_predictions.index]
assert adult_all_predictions.index.is_unique
adult_all_predictions.to_csv(PREDICTION_DIR / "GSE149356_adult_all_RNA_frozen_subtype_predictions.csv")

matched_tcr = tcr.loc[tcr["rna_barcode"].notna()].copy()
if matched_tcr.duplicated(["donor", "rna_barcode"]).any():
    raise RuntimeError("Adult TCR table has multiple rows for one RNA cell; resolve receptor evidence first")
adult_predictions = matched_tcr.merge(
    adult_all_predictions.reset_index(), on=["donor", "rna_barcode"],
    how="left", validate="one_to_one")
if adult_predictions["p_calibrated"].isna().any():
    raise RuntimeError("One or more TCR-linked adult cells lack frozen RNA predictions")
adult_predictions.to_csv(PREDICTION_DIR / "GSE149356_adult_frozen_subtype_predictions.csv", index=False)
'''.strip()})
    cells.append({"cell_type": 'code', "metadata": {}, "source": r'''
# 12. External metrics — primary and sensitivity, per donor and donor macro
def safe_metrics(frame, endpoint, donor):
    if len(frame) == 0 or frame["y"].nunique() < 2:
        return {"endpoint": endpoint, "donor": donor, "n": len(frame), "n_Vd1": int(frame["y"].sum()),
                "average_precision": np.nan, "roc_auc": np.nan, "brier": np.nan}
    return {"endpoint": endpoint, "donor": donor, "n": len(frame), "n_Vd1": int(frame["y"].sum()),
            "average_precision": average_precision_score(frame["y"], frame["p_calibrated"]),
            "roc_auc": roc_auc_score(frame["y"], frame["p_calibrated"]),
            "brier": brier_score_loss(frame["y"], frame["p_calibrated"])}

metric_rows = []
for endpoint in ["paired_TRG_TRD_primary", "TRD_only_sensitivity"]:
    eligible = adult_predictions.loc[adult_predictions[endpoint]].copy()
    for donor in ["HC1", "HC2"]:
        metric_rows.append(safe_metrics(eligible.loc[eligible["donor"].eq(donor)], endpoint, donor))
    metric_rows.append(safe_metrics(eligible, endpoint, "pooled"))
external_metrics = pd.DataFrame(metric_rows)
macro = (external_metrics.loc[external_metrics["donor"].isin(["HC1", "HC2"])]
         .groupby("endpoint")[["average_precision", "roc_auc", "brier"]].mean()
         .add_prefix("donor_macro_").reset_index())
external_metrics.to_csv(TABLE_DIR / "GSE149356_adult_external_metrics.csv", index=False)
macro.to_csv(TABLE_DIR / "GSE149356_adult_donor_macro_metrics.csv", index=False)
'''.strip()})
    cells.append({"cell_type": 'code', "metadata": {}, "source": r'''
# 13. PR and ROC validation curves with machine-readable source data
curve_rows = []
def add_curve_rows(frame, cohort, endpoint):
    if len(frame) == 0 or frame["y"].nunique() < 2:
        return
    precision, recall, pr_threshold = precision_recall_curve(frame["y"], frame["p_calibrated"])
    for i, (x, y_value) in enumerate(zip(recall, precision)):
        curve_rows.append({"endpoint": endpoint, "cohort": cohort, "curve": "PR",
                           "x": x, "y": y_value,
                           "threshold": pr_threshold[i] if i < len(pr_threshold) else np.nan})
    fpr, tpr, roc_threshold = roc_curve(frame["y"], frame["p_calibrated"])
    for x, y_value, threshold in zip(fpr, tpr, roc_threshold):
        curve_rows.append({"endpoint": endpoint, "cohort": cohort, "curve": "ROC",
                           "x": x, "y": y_value, "threshold": threshold})

add_curve_rows(source_oof, "Development OOF", "development_primary_seed")
adult_primary = adult_predictions.loc[adult_predictions["paired_TRG_TRD_primary"]].copy()
for donor in ["HC1", "HC2"]:
    add_curve_rows(adult_primary.loc[adult_primary["donor"].eq(donor)], donor,
                   "paired_TRG_TRD_primary")
add_curve_rows(adult_primary, "pooled", "paired_TRG_TRD_primary")
curve_source = pd.DataFrame(curve_rows)
curve_source.to_csv(TABLE_DIR / "PR_ROC_curve_source_data.csv", index=False)

def curve_legend_label(frame, cohort, curve_kind):
    """Display the correct scalar summary for each curve without changing its data."""
    if curve_kind == "PR":
        value = average_precision_score(frame["y"], frame["p_calibrated"])
        return f"{cohort} (AP={value:.3f}, n={len(frame):,})"
    value = roc_auc_score(frame["y"], frame["p_calibrated"])
    return f"{cohort} (AUC={value:.3f}, n={len(frame):,})"

fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.2), constrained_layout=True)
palette = {"Development OOF": "#7507F3", "HC1": "#E78AC3",
           "HC2": "#56B4E9", "pooled": "#06F379"}
panels = [
    (axes[0, 0], "Development OOF", "PR", "a", "Development donor-held-out PR"),
    (axes[1, 0], "Development OOF", "ROC", "c", "Development donor-held-out ROC"),
]
for axis, cohort, curve_kind, label, title in panels:
    part = curve_source.loc[curve_source["cohort"].eq(cohort) & curve_source["curve"].eq(curve_kind)]
    axis.plot(
        part["x"], part["y"], color=palette[cohort], linewidth=1.5,
        label=curve_legend_label(source_oof, cohort, curve_kind),
    )
    if curve_kind == "PR":
        prevalence = source_oof["y"].mean()
        axis.axhline(prevalence, color="#777777", linestyle="--", linewidth=0.9,
                     label=f"No-skill baseline ({prevalence:.3f})")
        axis.set(xlabel="Recall", ylabel="Precision")
    else:
        axis.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=0.9,
                  label="No-skill baseline")
        axis.set(xlabel="False-positive rate", ylabel="True-positive rate")
    axis.set_title(title); axis.legend(frameon=False, loc="best")
    axis.text(-0.16, 1.05, label, transform=axis.transAxes, fontweight="bold", color="black")

for axis, curve_kind, label, title in [
    (axes[0, 1], "PR", "b", "Adult external PR"),
    (axes[1, 1], "ROC", "d", "Adult external ROC"),
]:
    for cohort in ["HC1", "HC2", "pooled"]:
        part = curve_source.loc[curve_source["cohort"].eq(cohort) & curve_source["curve"].eq(curve_kind)]
        metric_frame = (adult_primary if cohort == "pooled" else
                        adult_primary.loc[adult_primary["donor"].eq(cohort)])
        if len(part):
            axis.plot(
                part["x"], part["y"], color=palette[cohort], linewidth=1.35,
                label=curve_legend_label(metric_frame, cohort, curve_kind),
            )
    if curve_kind == "PR":
        prevalence = adult_primary["y"].mean()
        axis.axhline(prevalence, color="#777777", linestyle="--", linewidth=0.9,
                     label=f"Pooled no-skill ({prevalence:.3f})")
        axis.set(xlabel="Recall", ylabel="Precision")
    else:
        axis.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=0.9,
                  label="No-skill baseline")
        axis.set(xlabel="False-positive rate", ylabel="True-positive rate")
    axis.set_title(title); axis.legend(frameon=False, loc="best")
    axis.text(-0.16, 1.05, label, transform=axis.transAxes, fontweight="bold", color="black")
for axis in axes.flat:
    axis.set_xlim(0, 1); axis.set_ylim(0, 1); axis.grid(False)
pr_roc_paths = save_publication_figure(
    fig, "development_and_GSE149356_adult_PR_ROC",
    [("a", axes[0, 0]), ("b", axes[0, 1]), ("c", axes[1, 0]), ("d", axes[1, 1])])
plt.show()
'''.strip()})
    cells.append({"cell_type": 'code', "metadata": {}, "source": r'''
# 14. Probability calibration: source OOF and untouched adult external
calibration_rows = []
def add_calibration(frame, cohort, endpoint):
    if len(frame) == 0 or frame["y"].nunique() < 2: return
    observed, predicted = calibration_curve(
        frame["y"], frame["p_calibrated"], n_bins=min(10, max(3, len(frame) // 20)),
        strategy="quantile")
    counts = pd.qcut(frame["p_calibrated"], q=min(len(observed), frame["p_calibrated"].nunique()),
                     duplicates="drop").value_counts(sort=False).to_numpy()
    for i, (x, y_value) in enumerate(zip(predicted, observed)):
        calibration_rows.append({"endpoint": endpoint, "cohort": cohort,
                                 "mean_predicted_probability": x,
                                 "observed_Vd1_fraction": y_value,
                                 "n_bin": int(counts[i]) if i < len(counts) else np.nan})

add_calibration(source_oof, "Development OOF", "development_primary_seed")
for donor in ["HC1", "HC2"]:
    add_calibration(adult_primary.loc[adult_primary["donor"].eq(donor)], donor,
                    "paired_TRG_TRD_primary")
add_calibration(adult_primary, "pooled", "paired_TRG_TRD_primary")
calibration_source = pd.DataFrame(calibration_rows)
calibration_source.to_csv(TABLE_DIR / "calibration_curve_source_data.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), constrained_layout=True)
for axis, cohorts, label, title in [
    (axes[0], ["Development OOF"], "a", "Development OOF calibration"),
    (axes[1], ["HC1", "HC2", "pooled"], "b", "Adult external calibration"),
]:
    axis.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=0.9,
              label="Perfect calibration")
    for cohort in cohorts:
        part = calibration_source.loc[calibration_source["cohort"].eq(cohort)]
        if len(part): axis.plot(part["mean_predicted_probability"], part["observed_Vd1_fraction"],
                                marker="o", markersize=3, linewidth=1.2,
                                color=palette[cohort], label=cohort)
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted P(Vδ1)",
             ylabel="Observed Vδ1 fraction", title=title)
    axis.legend(frameon=False, loc="best"); axis.grid(False)
    axis.text(-0.16, 1.05, label, transform=axis.transAxes, fontweight="bold", color="black")
calibration_paths = save_publication_figure(
    fig, "development_and_GSE149356_adult_calibration", [("a", axes[0]), ("b", axes[1])])
plt.show()
'''.strip()})
    cells.append({"cell_type": 'code', "metadata": {}, "source": r'''
# 15. Development UMAP: final TCR-free latent geometry, OOF probabilities only
# This is descriptive_not_performance_test. Quantitative claims come from PR/ROC/calibration.
final_z = training_latent(final_vae, final_rep).loc[development.index]
development_map = ad.AnnData(
    X=np.zeros((len(development), 1), dtype=np.float32),
    obs=development[["dataset", "donor", "subtype", "y"]].copy())
development_map.obsm["X_scVI"] = final_z.to_numpy()
sc.pp.neighbors(development_map, use_rep="X_scVI", n_neighbors=15, random_state=PRIMARY_SEED)
sc.tl.umap(development_map, min_dist=0.30, random_state=PRIMARY_SEED)
development_coordinates = development_map.obs.copy()
development_coordinates[["UMAP1", "UMAP2"]] = development_map.obsm["X_umap"]
development_coordinates["p_Vd1_OOF"] = source_oof.loc[development_coordinates.index, "p_calibrated"]
development_coordinates["visualization_role"] = "descriptive_not_performance_test"
development_coordinates.to_csv(TABLE_DIR / "development_UMAP_coordinates.csv")

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), constrained_layout=True)
for subtype, color in [("Vd1", VD1_COLOR), ("Vd2", VD2_COLOR)]:
    part = development_coordinates.loc[development_coordinates["subtype"].eq(subtype)]
    axes[0].scatter(part["UMAP1"], part["UMAP2"], s=9, c=color, alpha=0.75,
                    edgecolors="none", label=f"{subtype} (n={len(part)})")
axes[0].legend(frameon=False, markerscale=1.4)
scatter = axes[1].scatter(
    development_coordinates["UMAP1"], development_coordinates["UMAP2"],
    c=development_coordinates["p_Vd1_OOF"], cmap=SUBTYPE_CMAP, vmin=0, vmax=1,
    s=9, alpha=0.82, edgecolors="none")
cax = inset_axes(axes[1], width="3%", height="38%", loc="lower right", borderpad=1.0)
colorbar = fig.colorbar(scatter, cax=cax)
colorbar.set_label("OOF calibrated P(Vδ1)", color="black")
for axis, label, title in [(axes[0], "a", "Sequence-defined subtype"),
                           (axes[1], "b", "Donor-held-out prediction")]:
    axis.set_title(title); axis.set_xlabel("UMAP1", color="black"); axis.set_ylabel("UMAP2", color="black")
    axis.set_xticks([]); axis.set_yticks([]); axis.grid(False)
    axis.text(-0.12, 1.05, label, transform=axis.transAxes, fontweight="bold", color="black")
development_umap_paths = save_publication_figure(
    fig, "development_OOF_predicted_probability_UMAP", [("a", axes[0]), ("b", axes[1])])
plt.show()
'''.strip()})
    cells.append({"cell_type": 'code', "metadata": {}, "source": r'''
# 16. Adult UMAP across all adult RNA cells; labels are overlays, not model inputs
adult_map = ad.AnnData(
    X=np.zeros((len(adult_all_predictions), 1), dtype=np.float32),
    obs=adult_all_predictions[["donor", "rna_barcode", "p_raw", "p_calibrated"]].copy())
adult_map.obsm["X_scVI"] = adult_all_latent.loc[adult_map.obs_names].to_numpy()
sc.pp.neighbors(adult_map, use_rep="X_scVI", n_neighbors=15, random_state=PRIMARY_SEED)
sc.tl.umap(adult_map, min_dist=0.30, random_state=PRIMARY_SEED)
adult_coordinates = adult_map.obs.copy()
adult_coordinates[["UMAP1", "UMAP2"]] = adult_map.obsm["X_umap"]
sequence_lookup = matched_tcr.set_index(
    matched_tcr["donor"].astype(str) + "::" + matched_tcr["rna_barcode"].astype(str))["subtype"]
adult_coordinates["sequence_subtype"] = adult_coordinates.index.map(sequence_lookup)
adult_coordinates["visualization_role"] = "descriptive_not_performance_test"
adult_coordinates["population"] = "all adult RNA cells"
adult_coordinates.to_csv(TABLE_DIR / "GSE149356_adult_UMAP_coordinates.csv")

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), constrained_layout=True)
scatter = axes[0].scatter(
    adult_coordinates["UMAP1"], adult_coordinates["UMAP2"],
    c=adult_coordinates["p_calibrated"], cmap=SUBTYPE_CMAP, vmin=0, vmax=1,
    s=2.2, alpha=0.65, edgecolors="none", rasterized=True)
cax = inset_axes(axes[0], width="3%", height="38%", loc="lower right", borderpad=1.0)
colorbar = fig.colorbar(scatter, cax=cax)
colorbar.set_label("Frozen calibrated P(Vδ1)", color="black")
axes[1].scatter(adult_coordinates["UMAP1"], adult_coordinates["UMAP2"],
                s=1.4, c="#D9D9D9", alpha=0.30, edgecolors="none", rasterized=True,
                label="No sequence-supported binary label")
for subtype, color in [("Vd1", VD1_COLOR), ("Vd2", VD2_COLOR)]:
    part = adult_coordinates.loc[adult_coordinates["sequence_subtype"].eq(subtype)]
    axes[1].scatter(part["UMAP1"], part["UMAP2"], s=5.5, c=color, alpha=0.82,
                    edgecolors="none", rasterized=True, label=f"{subtype} (n={len(part)})")
axes[1].legend(frameon=False, markerscale=1.8, loc="best")
for axis, label, title in [(axes[0], "a", "Frozen prediction: all adult RNA cells"),
                           (axes[1], "b", "Sequence-supported label overlay")]:
    axis.set_title(title); axis.set_xlabel("UMAP1", color="black"); axis.set_ylabel("UMAP2", color="black")
    axis.set_xticks([]); axis.set_yticks([]); axis.grid(False)
    axis.text(-0.12, 1.05, label, transform=axis.transAxes, fontweight="bold", color="black")
adult_umap_paths = save_publication_figure(
    fig, "GSE149356_adult_predicted_probability_UMAP", [("a", axes[0]), ("b", axes[1])])
plt.show()
'''.strip()})
    cells.append({"cell_type": 'code', "metadata": {}, "source": r'''
# 17. Evidence-aware routing for deployment
def route_subtype(valid_trd, sequence_subtype, p_vd1, lower=0.20, upper=0.80):
    """Sequence wins; ambiguous sequence is never overwritten by ML."""
    if valid_trd:
        if sequence_subtype in {"Vd1", "Vd2", "Vd3"}:
            return sequence_subtype, "sequence_defined"
        if sequence_subtype == "dual":
            return "ambiguous", "ambiguous_sequence"
        return "other_or_unresolved", "sequence_defined"
    if p_vd1 >= upper:
        return "Vd1", "expression_model_predicted"
    if p_vd1 <= lower:
        return "Vd2", "expression_model_predicted"
    return "other_or_unresolved", "expression_model_predicted_low_confidence"

# valid unambiguous TRD sequence has rule_based_sequence_precedence.
# low confidence is not Vd3; the binary expression model cannot identify Vd3.
routing_examples = pd.DataFrame([
    {"valid_trd": True, "sequence_subtype": "Vd3", "p_vd1": 0.99},
    {"valid_trd": True, "sequence_subtype": "dual", "p_vd1": 0.01},
    {"valid_trd": False, "sequence_subtype": "other_or_unresolved", "p_vd1": 0.90},
    {"valid_trd": False, "sequence_subtype": "other_or_unresolved", "p_vd1": 0.50},
])
routing_examples[["assigned_subtype", "evidence_level"]] = routing_examples.apply(
    lambda r: pd.Series(route_subtype(r.valid_trd, r.sequence_subtype, r.p_vd1)), axis=1)
routing_examples.to_csv(TABLE_DIR / "evidence_aware_routing_examples.csv", index=False)
'''.strip()})
    cells.append({"cell_type": 'code', "metadata": {}, "source": r'''
# 14. Publication figure: source OOF and adult donor score distributions
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), constrained_layout=True)
for subtype, color in [("Vd1", VD1_COLOR), ("Vd2", VD2_COLOR)]:
    values = source_oof.loc[source_oof["subtype"].eq(subtype), "p_calibrated"]
    axes[0].hist(values, bins=20, density=True, histtype="step", linewidth=1.4,
                 color=color, label=f"{subtype} (n={len(values)})")
axes[0].set(title="a  Development donor-held-out OOF", xlabel="Calibrated P(Vδ1)", ylabel="Density")
axes[0].legend(frameon=False)
primary = adult_predictions.loc[adult_predictions["paired_TRG_TRD_primary"]]
positions, labels, data, colors = [], [], [], []
for i, donor in enumerate(["HC1", "HC2"]):
    for j, subtype in enumerate(["Vd1", "Vd2"]):
        positions.append(i * 3 + j); labels.append(f"{donor}\n{subtype}")
        data.append(primary.loc[primary["donor"].eq(donor) & primary["subtype"].eq(subtype), "p_calibrated"])
        colors.append(VD1_COLOR if subtype == "Vd1" else VD2_COLOR)
boxes = axes[1].boxplot(data, positions=positions, widths=0.65, patch_artist=True, showfliers=False)
for patch, color in zip(boxes["boxes"], colors):
    patch.set_facecolor(color); patch.set_alpha(0.65); patch.set_edgecolor("none"); patch.set_linewidth(0)
axes[1].set(title="b  GSE149356 adult external", ylabel="Calibrated P(Vδ1)", xticks=positions, xticklabels=labels)
for ax in axes:
    ax.grid(False); ax.tick_params(colors="black")
    for text in [ax.title, ax.xaxis.label, ax.yaxis.label, *ax.get_xticklabels(), *ax.get_yticklabels()]:
        text.set_color("black")
    ax.set_xlabel(ax.get_xlabel(), color="black")
score_distribution_paths = save_publication_figure(
    fig, "development_OOF_and_GSE149356_adult_scores", [("a", axes[0]), ("b", axes[1])])
plt.show()
'''.strip()})
    cells.append({"cell_type": 'code', "metadata": {}, "source": r'''
# 19. Auditable completion and figure-QA manifests
figure_QA_path = TABLE_DIR / "figure_QA_manifest.json"
figure_QA_path.write_text(json.dumps({
    "backend": "python_matplotlib",
    "font": FIGURE_FONT,
    "minimum_configured_font_pt": 7,
    "pdf_editable_text": True,
    "umap_interpretation": "descriptive_not_performance_test",
    "rendered_collision_audit": "must_run_on_Woody_after_notebook_execution",
    "figures": FIGURE_QA_ROWS,
}, indent=2), encoding="utf-8")
figure_outputs = [Path(path) for bundle in [
    pr_roc_paths, calibration_paths, development_umap_paths,
    adult_umap_paths, score_distribution_paths,
] for path in bundle]
output_paths = [
    TABLE_DIR / "development_sequence_defined_subtype_counts.csv",
    TABLE_DIR / "SPL_released_TRD_sequence_linkage_audit.csv",
    TABLE_DIR / "outer_training_grouped_LR_C_selection.csv",
    TABLE_DIR / "final_grouped_LR_C_selection.csv",
    TABLE_DIR / "development_seed_stability.csv",
    TABLE_DIR / "source_freeze.json",
    TABLE_DIR / "adult_sequence_label_coverage.csv",
    TABLE_DIR / "GSE149356_adult_external_metrics.csv",
    TABLE_DIR / "GSE149356_adult_donor_macro_metrics.csv",
    TABLE_DIR / "evidence_aware_routing_examples.csv",
    TABLE_DIR / "PR_ROC_curve_source_data.csv",
    TABLE_DIR / "calibration_curve_source_data.csv",
    TABLE_DIR / "development_UMAP_coordinates.csv",
    TABLE_DIR / "GSE149356_adult_UMAP_coordinates.csv",
    figure_QA_path,
    PREDICTION_DIR / "development_source_only_OOF_calibrated_predictions.csv",
    PREDICTION_DIR / "GSE149356_adult_frozen_subtype_predictions.csv",
    PREDICTION_DIR / "GSE149356_adult_all_RNA_frozen_subtype_predictions.csv",
    PREDICTION_DIR / "development_three_seed_donor_held_out_OOF_predictions.csv",
    *[MODEL_DIR / f"seed_{seed}_latent_LR.joblib" for seed in REGISTERED_SEEDS],
    *figure_outputs,
]
missing_outputs = [str(p) for p in output_paths if not p.exists()]
if missing_outputs: raise RuntimeError("Missing declared outputs:\n" + "\n".join(missing_outputs))
manifest = {
    **progress,
    "analysis": ANALYSIS_NAME, "status": "completed", "seed": SEED,
    "query_adaptation_used": False, "external_data_used_for_selection": False,
    "external_label_language": "cell-linked sequence-supported",
    "inputs": {**{k: str(v) for k, v in INPUTS.items()}, **{k: str(v) for k, v in EXTERNAL_INPUTS.items()}},
    "outputs": [{"path": str(p), "sha256": sha256_file(p)} for p in output_paths if p.is_file()],
}
manifest_path = TABLE_DIR / "completion_manifest.json"
atomic_json(manifest_path, manifest)
print("COMPLETE:", manifest_path)
'''.strip()})
    cells.append({"cell_type": 'code', "metadata": {}, "source": r'''
# Inspect the source freeze produced by this run, not a hard-coded project copy.
freeze = json.loads(SOURCE_FREEZE_PATH.read_text())
print("Final primary-seed C:", freeze["primary_final_C"])
'''.strip()})
    cells.append({"cell_type": 'markdown', "metadata": {}, "source": r'''
## Interpretation and limitations

- The primary result is donor-held-out source discrimination followed by a
  frozen adult external evaluation. A high score supports a portable
  expression correlate of receptor subtype; it does not make receptor
  sequencing unnecessary.
- Song contributes almost no Vδ1, while SPL contributes most Vδ1 cells.
  Dataset and subtype are therefore partly confounded. HC1/HC2 transport is
  the decisive check, and additional tissue/donor replication remains needed.
- Processed GSE149356 labels are cell-linked sequence-supported. Reprocessing
  raw TCR FASTQs is required before claiming the same productive/high-confidence
  gold standard used in datasets with explicit contig QC fields.
- Vδ3 is outside this binary model. A future Vδ1/Vδ2/Vδ3 classifier requires
  multi-donor, multi-tissue sequence-confirmed Vδ3 development data.
- No externally validated abstention rule is established; low confidence is not evidence of Vδ3.
- Inner LR selection reuses each outer-training encoder, and the feature backbone
  predates subtype folds. This is not fully nested model/feature development.
'''.strip()})
    for index, cell in enumerate(cells):
        cell["id"] = hashlib.sha256(f"{ANALYSIS_NAME}:{index}".encode()).hexdigest()[:12]
        if cell["cell_type"] == "code":
            cell.update(outputs=[], execution_count=None)
    return {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "gdt_stage": ANALYSIS_NAME,
    }, "nbformat": 4, "nbformat_minor": 5}



def execute_notebook(output):
    """Execute the frozen in-memory cell list; save stdout/stderr/errors durably.

    This runner needs no nbclient/nbformat installation. It does not intercept
    scientific checkpoints: those remain governed by completion_manifest.json.
    """
    import contextlib
    import io
    import socket
    import sys
    import time
    import traceback
    notebook = json.loads(output.read_text())
    code_cells = [(i, cell, "".join(cell["source"])) for i, cell in enumerate(notebook["cells"])
                  if cell["cell_type"] == "code"]
    execution = {"status": "running", "pid": os.getpid(), "host": socket.gethostname(),
                 "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 "python_executable": sys.executable,
                 "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                 "completed_cell_ids": [],
                 "cell_source_sha256": {str(i): hashlib.sha256(source.encode()).hexdigest()
                                        for i, _, source in code_cells}}
    notebook["metadata"]["execution"] = execution
    last_write = [0.0]
    def flush_notebook(force=False):
        if not force and time.monotonic() - last_write[0] < 5:
            return
        fd, temporary = tempfile.mkstemp(dir=output.parent, prefix=output.name + ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as handle:
                json.dump(notebook, handle, ensure_ascii=False, indent=1)
                handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
            os.replace(temporary, output)
            last_write[0] = time.monotonic()
        finally:
            if os.path.exists(temporary): os.unlink(temporary)
    class Capture(io.TextIOBase):
        def __init__(self, cell, name):
            self.cell, self.name = cell, name
        def write(self, value):
            if value:
                outputs = self.cell["outputs"]
                if not outputs or outputs[-1].get("name") != self.name:
                    outputs.append({"output_type": "stream", "name": self.name, "text": ""})
                outputs[-1]["text"] += value
                flush_notebook()
            return len(value)
        def flush(self):
            flush_notebook()
        def isatty(self):
            return False
    flush_notebook(force=True)
    namespace = {"__name__": "__main__"}
    for count, (index, cell, source) in enumerate(code_cells, 1):
        execution["current_cell_id"] = index
        cell["outputs"] = []
        cell["execution_count"] = count
        flush_notebook(force=True)
        try:
            with contextlib.redirect_stdout(Capture(cell, "stdout")), contextlib.redirect_stderr(Capture(cell, "stderr")):
                exec(compile(source, f"{output.name}:cell-{index}", "exec"), namespace)
        except BaseException as error:
            cell["outputs"].append({"output_type": "error", "ename": type(error).__name__,
                                    "evalue": str(error), "traceback": traceback.format_exc().splitlines()})
            execution["status"] = "failed"
            execution["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            flush_notebook(force=True)
            raise
        execution["completed_cell_ids"].append(index)
        flush_notebook(force=True)
    execution["status"] = "completed"
    execution["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    flush_notebook(force=True)


def main():
    root = Path(__file__).resolve().parent
    output = Path(os.environ.get("GDT_NOTEBOOK_OUTPUT", root / f"{ANALYSIS_NAME}.ipynb"))
    notebook = build_notebook()
    if output.exists():
        previous = json.loads(output.read_text())
        notebook["metadata"].update(previous.get("metadata", {}))
        if len(previous["cells"]) != len(notebook["cells"]):
            raise RuntimeError("Refusing to discard historical cells during regeneration")
        for cell, old in zip(notebook["cells"], previous["cells"]):
            cell["metadata"].update(old.get("metadata", {}))
            if old.get("outputs") or old.get("execution_count") is not None:
                historical = {
                    "source": old["source"], "outputs": old.get("outputs", []),
                    "execution_count": old.get("execution_count"),
                    "status": "Historical execution, not execution of regenerated code",
                }
                if "historical_execution" not in cell["metadata"]:
                    cell["metadata"]["historical_execution"] = historical
                else:
                    cell["metadata"].setdefault("historical_execution_history", []).append(historical)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=output.parent, prefix=output.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(notebook, handle, ensure_ascii=False, indent=1)
            handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)
    print(output)
    import sys
    if "--execute" in sys.argv[1:]:
        execute_notebook(output)


if __name__ == "__main__":
    main()
