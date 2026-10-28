"""Generate the registered Jupyter notebook suite.

Run locally to rebuild notebook JSON after editing the cell sources here.  The
generated notebooks themselves contain only remote paths rooted at
`/data/users/kaelynwen` and are intended to be run in numeric order.
"""

from __future__ import annotations

from pathlib import Path
import textwrap

import nbformat as nbf


HERE = Path(__file__).resolve().parent
NB_DIR = HERE / "notebooks"
NB_DIR.mkdir(parents=True, exist_ok=True)


def md(text: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


COMMON = code(r'''
from pathlib import Path
import os, sys, json, warnings
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

REMOTE_PROJECT = Path("/data/users/kaelynwen/gdT_project/gdt_lr_12model")
PROJECT = Path(os.environ.get("GDT_LR_PROJECT", REMOTE_PROJECT))
if not PROJECT.exists():
    candidate = Path.cwd().resolve()
    PROJECT = candidate.parent if candidate.name == "notebooks" else candidate
sys.path.insert(0, str(PROJECT / "src"))

from gdt_lr_utils import *

with open(PROJECT / "config/paths.yaml") as f:
    PATHS = yaml.safe_load(f)
with open(PROJECT / "config/cohort_rules.yaml") as f:
    RULES = yaml.safe_load(f)
with open(PROJECT / "config/lr_grid.yaml") as f:
    GRID = yaml.safe_load(f)

set_reproducibility(GRID["seed"], GRID["threads"])
ensure_dirs(PROJECT)
sc.settings.figdir = PROJECT / "results/figures"
sc.settings.verbosity = 2
print("PROJECT:", PROJECT)
''')


def notebook(title: str, purpose: str, cells: list):
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
        "gdt_lr_stage": title,
    }
    nb["cells"] = [md(f"# {title}\n\n{purpose}\n\n**Remote project:** `/data/users/kaelynwen/gdT_project/gdt_lr_12model`"), COMMON] + cells
    return nb


NOTEBOOKS = {}


NOTEBOOKS["00_environment_and_paths.ipynb"] = notebook(
    "00 — Environment and immutable paths",
    "Validate dependencies and resolve input paths. This notebook writes no biological labels.",
    [
        code(r'''
import platform, sklearn, scipy, joblib
versions = {
    "python": platform.python_version(), "scanpy": sc.__version__,
    "anndata": ad.__version__, "pandas": pd.__version__,
    "numpy": np.__version__, "scipy": scipy.__version__,
    "sklearn": sklearn.__version__, "joblib": joblib.__version__,
}
display(pd.Series(versions, name="version"))
write_json(versions, PROJECT / "manifests/software_versions.json")
'''),
        code(r'''
song_root = first_existing(PATHS["song_root_candidates"], "dir")
d2_root = first_existing(PATHS["dataset2_root_candidates"], "dir")
d2_files = {k: d2_root / v for k, v in PATHS["dataset2_files"].items()}
for name, path in d2_files.items():
    if not path.is_file():
        raise FileNotFoundError(f"Dataset 2 {name}: {path}")

resolved = {
    "remote_root": PATHS["remote_root"], "project_root": str(PROJECT),
    "song_root": str(song_root), "dataset2_root": str(d2_root),
    **{f"dataset2_{k}": str(v) for k, v in d2_files.items()},
}
display(pd.Series(resolved))
write_json(resolved, PROJECT / "manifests/resolved_paths.json")
'''),
        md('''
## Stop conditions

Do not continue if a resolved path is under `/home/kaelynwen`, if a donor folder
is missing, or if Dataset 2 uses a different object than the registered one.
'''),
    ],
)


NOTEBOOKS["01_build_receptor_evidence_table.ipynb"] = notebook(
    "01 — Build barcode-resolved receptor evidence",
    "Build Song and Cross-atlas receptor evidence without using published cell-type annotations as truth.",
    [
        code(r'''
resolved = json.load(open(PROJECT / "manifests/resolved_paths.json"))
song, song_contigs, song_inputs = load_song_dataset(
    resolved["song_root"], RULES["donors_song"],
    require_full_length=RULES["require_full_length_primary"],
)
print("Song GEX:", song.shape, "gold-eligible contigs:", len(song_contigs))
display(pd.crosstab(song.obs["donor"], song.obs["receptor_group"], margins=True))
'''),
        code(r'''
# Audit whether a partner chain exists only in Song's unfiltered all-contig
# table. This is descriptive and never rescues/relabels a gold cell.
partner_rows = []
song_root = Path(resolved["song_root"])
for donor in RULES["donors_song"]:
    all_path = song_root / donor / "gd_tcr_all_contig_annotations.csv"
    if not all_path.is_file():
        warnings.warn(f"Missing optional all-contig audit file: {all_path}")
        continue
    raw = pd.read_csv(all_path, low_memory=False)
    raw["cell_id"] = [canonical_cell_id("Song_D1", donor, b) for b in raw["barcode"]]
    raw["chain"] = raw["chain"].astype(str)
    raw["is_productive"] = as_true(raw["productive"]) if "productive" in raw else False
    raw["is_high_confidence"] = as_true(raw["high_confidence"]) if "high_confidence" in raw else False
    for orphan_group, partner in (("orphan_TRD", "TRG"), ("orphan_TRG", "TRD")):
        orphan_ids = song.obs.index[
            song.obs["donor"].astype(str).eq(donor) & song.obs["receptor_group"].astype(str).eq(orphan_group)
        ]
        for cell_id in orphan_ids:
            z = raw[raw["cell_id"].eq(cell_id) & raw["chain"].eq(partner)]
            partner_rows.append({"donor": donor, "cell_id": cell_id, "filtered_group": orphan_group,
                                 "missing_partner": partner, "partner_any_all_contig": len(z) > 0,
                                 "partner_productive_all_contig": bool(z["is_productive"].any()),
                                 "partner_highconf_productive_all_contig": bool((z["is_productive"] & z["is_high_confidence"]).any())})
partner_audit = pd.DataFrame(partner_rows)
partner_audit.to_csv(PROJECT / "results/tables/song_all_contig_partner_audit.csv", index=False)
if len(partner_audit):
    display(partner_audit.groupby(["donor", "filtered_group"])[
        ["partner_any_all_contig", "partner_productive_all_contig", "partner_highconf_productive_all_contig"]
    ].agg(["count", "sum", "mean"]))
'''),
        code(r'''
d2_evidence, d2_ir_audit = load_dataset2_ir_evidence(
    resolved["dataset2_ab_ir"], resolved["dataset2_gd_ir"], dataset="CrossAtlas_D2"
)
print("Cross-atlas receptor-evidence cells:", len(d2_evidence))
display(d2_ir_audit)
display(pd.crosstab(d2_evidence["donor"], d2_evidence["receptor_group"], margins=True))
'''),
        code(r'''
spl = load_dataset2_gex(resolved["dataset2_gex"], "SPL", d2_evidence)
print("SPL:", spl.shape)
safe_write_h5ad(spl, PROJECT / "manifests/dataset2_SPL_with_receptor_evidence.h5ad")
del spl

bma = load_dataset2_gex(resolved["dataset2_gex"], "BMA", d2_evidence)
print("BMA:", bma.shape)
safe_write_h5ad(bma, PROJECT / "manifests/dataset2_BMA_with_receptor_evidence.h5ad")
del bma
'''),
        code(r'''
safe_write_h5ad(song, PROJECT / "manifests/song_with_receptor_evidence.h5ad")
song_contigs.to_parquet(PROJECT / "manifests/song_gold_eligible_contigs.parquet", index=False)
d2_evidence.reset_index().to_parquet(PROJECT / "manifests/dataset2_receptor_evidence_cells.parquet", index=False)
pd.concat([song_inputs, d2_ir_audit], ignore_index=True, sort=False).to_csv(
    PROJECT / "manifests/input_receptor_files.csv", index=False
)
print("Saved receptor evidence. Dataset 2 evidence tier is not claimed to expose original productivity flags.")
'''),
    ],
)


NOTEBOOKS["02_build_gold_query_cohorts.ipynb"] = notebook(
    "02 — Freeze gold, query, and audit cohorts",
    "Apply identical RNA QC and immutable receptor definitions. SPL remains external and is never used for fitting H1 models.",
    [
        code(r'''
objects = {
    "Song_D1": ad.read_h5ad(PROJECT / "manifests/song_with_receptor_evidence.h5ad"),
    "CrossAtlas_D2_SPL": ad.read_h5ad(PROJECT / "manifests/dataset2_SPL_with_receptor_evidence.h5ad"),
}
for name, obj in objects.items():
    obj = add_qc_and_lineage_flags(
        obj, min_genes=RULES["min_genes"], max_mt_pct=RULES["max_pct_counts_mt"],
        require_t_gene=RULES["require_detected_Tlineage_marker"],
    )
    objects[name] = finalize_cohorts(obj)
    assert objects[name].obs_names.is_unique
    assert not (objects[name].obs["gold_B_paired_gdT"] & objects[name].obs["gold_B_paired_abT"]).any()
    assert not (objects[name].obs["candidate_orphan_TRD_nonoverlap"] &
                (objects[name].obs["n_TRA"].gt(0) | objects[name].obs["n_TRB"].gt(0))).any()
    print(name, objects[name].shape)
    display(pd.crosstab(objects[name].obs["donor"], objects[name].obs["immutable_cohort"], margins=True))
'''),
        code(r'''
summary = []
for name, obj in objects.items():
    tab = obj.obs.groupby(["donor", "immutable_cohort"], observed=True).size().rename("n_cells").reset_index()
    tab.insert(0, "dataset", name)
    summary.append(tab)
    safe_write_h5ad(obj, PROJECT / f"manifests/{name}_cohorts.h5ad")

summary = pd.concat(summary, ignore_index=True)
summary.to_csv(PROJECT / "results/tables/cohort_counts_by_donor.csv", index=False)
display(summary)
'''),
        code(r'''
gold_rows = []
for name, obj in objects.items():
    keep = obj.obs["immutable_cohort"].astype(str).isin(["gold_B_paired_gdT", "gold_B_paired_abT"])
    cols = ["dataset", "donor", "immutable_cohort", "gold_A_paired_gdT", "gold_A_paired_abT",
            "n_TRA", "n_TRB", "n_TRG", "n_TRD", "qc_Tlineage_pass"]
    frame = obj.obs.loc[keep, cols].copy()
    frame["cell_id"] = frame.index.astype(str)
    gold_rows.append(frame)
pd.concat(gold_rows, ignore_index=True).to_parquet(PROJECT / "manifests/gold_cells.parquet", index=False)
'''),
    ],
)


NOTEBOOKS["02b_bma_receptor_reliability_audit.ipynb"] = notebook(
    "02b — Independent BMA receptor-reliability audit",
    "BMA is excluded from all 12 core models. This notebook only audits receptor overlap, RNA identity, donors, and available evidence quality.",
    [
        code(r'''
bma = ad.read_h5ad(PROJECT / "manifests/dataset2_BMA_with_receptor_evidence.h5ad")
bma = finalize_cohorts(add_qc_and_lineage_flags(
    bma, RULES["min_genes"], RULES["max_pct_counts_mt"], RULES["require_detected_Tlineage_marker"]
))
obs = bma.obs
has_ab = obs["n_TRA"].gt(0) | obs["n_TRB"].gt(0)
has_gd = obs["n_TRG"].gt(0) | obs["n_TRD"].gt(0)
obs["bma_audit_group"] = np.select([
    obs["gold_A_paired_gdT"], obs["gold_B_paired_gdT"],
    obs["candidate_orphan_TRD_nonoverlap"], obs["candidate_orphan_TRG_nonoverlap"],
    has_gd & has_ab & (obs["n_TRA"].gt(0) ^ obs["n_TRB"].gt(0)),
    has_gd & has_ab & obs["n_TRA"].gt(0) & obs["n_TRB"].gt(0),
    ~has_ab & ~has_gd,
], ["strict_1TRG_1TRD", "paired_TRG_TRD", "orphan_TRD_no_ab", "orphan_TRG_no_ab",
    "gd_plus_single_ab", "gd_plus_paired_ab", "no_reported_TCR"], default="other")

group_table = pd.crosstab(obs["donor"], obs["bma_audit_group"], margins=True)
display(group_table)
group_table.to_csv(PROJECT / "results/tables/bma_group_by_donor.csv")
'''),
        code(r'''
audit_cells = obs[["dataset", "donor", "bma_audit_group", "n_TRA", "n_TRB", "n_TRG", "n_TRD",
                   "qc_Tlineage_pass", "n_Tlineage_markers_detected", "Tlineage_mean_expression"]].copy()
audit_cells["cell_id"] = audit_cells.index.astype(str)
audit_cells.to_parquet(PROJECT / "manifests/bma_receptor_audit_cells.parquet", index=False)

overlap = obs.groupby(["donor", "bma_audit_group"], observed=True).agg(
    n=("n_TRG", "size"), any_TRA=("n_TRA", lambda x: (x > 0).mean()),
    any_TRB=("n_TRB", lambda x: (x > 0).mean()),
).reset_index()
overlap.to_csv(PROJECT / "results/tables/bma_ab_overlap_audit.csv", index=False)

ir_audit = pd.read_csv(PROJECT / "manifests/input_receptor_files.csv")
ir_audit = ir_audit[ir_audit["object"].notna()].copy()
availability = {
    "productive": bool(as_true(ir_audit["has_productive_field"]).all()),
    "V_call": bool(as_true(ir_audit["has_V_call_field"]).all()),
    "J_call": bool(as_true(ir_audit["has_J_call_field"]).all()),
    "high_confidence": bool(as_true(ir_audit["has_high_confidence_field"]).all()),
    "reads_or_UMIs": bool(as_true(ir_audit["has_read_or_UMI_field"]).all()),
    "CDR3": True,
}
quality = pd.DataFrame({"field": availability.keys(),
                        "available_in_both_dataset2_IR_objects": availability.values()})
quality["interpretation"] = np.where(
    quality["available_in_both_dataset2_IR_objects"], "enforced or available for audit", "not exposed comparably"
)
quality.to_csv(PROJECT / "results/tables/bma_contig_quality_summary.csv", index=False)
display(quality)
'''),
        code(r'''
log = make_log_normalized(bma)
markers = [g for g in list(T_LINEAGE_GENES) + list(NON_T_MARKERS) + list(TCR_PANEL_DEFAULT) if g in log.var_names]
rows = []
for group in obs["bma_audit_group"].astype(str).unique():
    cells = obs.index[obs["bma_audit_group"].astype(str).eq(group)]
    if len(cells) == 0: continue
    X = log[cells, markers].X
    arr = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    rows.extend({"group": group, "gene": gene, "mean_log_expression": float(arr[:, j].mean()),
                 "fraction_detected": float((arr[:, j] > 0).mean()), "n_cells": len(cells)}
                for j, gene in enumerate(markers))
marker_summary = pd.DataFrame(rows)
marker_summary.to_csv(PROJECT / "results/tables/bma_rna_marker_summary.csv", index=False)

sc.pp.pca(log, n_comps=30)
sc.pp.neighbors(log, n_neighbors=15, n_pcs=30)
sc.tl.umap(log, random_state=GRID["seed"])
sc.pl.umap(log, color=["bma_audit_group"], show=False)
plt.savefig(PROJECT / "results/figures/bma_receptor_groups_umap.pdf", bbox_inches="tight")
plt.show()
sc.pl.dotplot(log, markers, groupby="bma_audit_group", show=False)
plt.savefig(PROJECT / "results/figures/bma_marker_dotplot.pdf", bbox_inches="tight")
plt.show()
'''),
        code(r'''
decision = {
    "dataset": "CrossAtlas_D2_BMA",
    "decision": "excluded_core_unreliable",
    "core_12_models_eligible": False,
    "reasons": [
        "BMA was predeclared as excluded from core model construction",
        "Dataset 2 IR evidence fields are not assumed equivalent to Song Cell Ranger filtered contigs",
        "Unavailable comparable fields: " + ", ".join(quality.loc[~quality.available_in_both_dataset2_IR_objects, "field"]),
        "audit results may be used descriptively but cannot be promoted after inspecting performance",
    ],
}
with open(PROJECT / "manifests/bma_audit_decision.yaml", "w") as handle:
    yaml.safe_dump(decision, handle, sort_keys=False)
safe_write_h5ad(bma, PROJECT / "manifests/CrossAtlas_D2_BMA_audited.h5ad")
display(pd.Series(decision))
'''),
    ],
)


NOTEBOOKS["03_build_gene_universe_and_hvgs.ipynb"] = notebook(
    "03 — Build common gene universe and frozen HVGs",
    "Construct H1 from Song gold cells only and the legacy H2 target-gold-informed HVG sensitivity panel. H2 is not a transductive representation model.",
    [
        code(r'''
song = ad.read_h5ad(PROJECT / "manifests/Song_D1_cohorts.h5ad")
spl = ad.read_h5ad(PROJECT / "manifests/CrossAtlas_D2_SPL_cohorts.h5ad")
genes_common = common_genes([song, spl])
pd.Series(genes_common, name="gene").to_csv(PROJECT / "features/common_gene_universe.csv", index=False)
print("Common genes:", len(genes_common))
song = subset_unique_genes(song, genes_common)
spl = subset_unique_genes(spl, genes_common)
'''),
        code(r'''
gold_names = ["gold_B_paired_gdT", "gold_B_paired_abT"]
song_gold = song.obs["immutable_cohort"].astype(str).isin(gold_names)
H1, meta_H1 = select_hvgs(song, GRID["primary_hvg_k"], "donor", song_gold, exclude_tcr=True)
pd.Series(H1, name="gene").to_csv(PROJECT / "features/H1_song_inductive_TCRfree_backbone.csv", index=False)
write_json(meta_H1 | {"eligible_cells": int(song_gold.sum()), "target_expression_used": False},
           PROJECT / "features/H1_metadata.json")
'''),
        code(r'''
from scipy import sparse

# The released gamma-delta IR assay covers only these registered SPL donors.
# Alpha-beta cells from non-covered donors are excluded from the legacy H2
# target-gold-informed sensitivity panel so donor/assay availability cannot
# define this feature space. Strict unlabeled-target transduction is registered
# separately in notebooks 04b/07b/08c.
spl_gd_assay_donors = list(map(str, RULES["spl_gd_assay_complete_donors"]))
observed_gd_donors = sorted(
    spl.obs.loc[(spl.obs["n_TRG"].gt(0) | spl.obs["n_TRD"].gt(0)), "donor"].astype(str).unique()
)
if set(observed_gd_donors) != set(spl_gd_assay_donors):
    raise ValueError(
        f"Registered SPL gamma-delta assay donors {spl_gd_assay_donors} do not match "
        f"observed TRG/TRD evidence donors {observed_gd_donors}."
    )

coverage = (spl.obs.assign(has_gd_evidence=spl.obs["n_TRG"].gt(0) | spl.obs["n_TRD"].gt(0))
            .groupby("donor", observed=True)
            .agg(n_cells=("donor", "size"), n_cells_with_gd_evidence=("has_gd_evidence", "sum"))
            .reset_index())
coverage["registered_gd_assay_complete"] = coverage["donor"].astype(str).isin(spl_gd_assay_donors)
coverage.to_csv(PROJECT / "manifests/dataset2_SPL_gd_assay_donor_coverage.csv", index=False)
display(coverage)

spl_gold = (
    spl.obs["immutable_cohort"].astype(str).isin(gold_names)
    & spl.obs["donor"].astype(str).isin(spl_gd_assay_donors)
)

# Build both inputs from their own raw-count source before concatenation.
# Song stores raw counts in X; SPL stores raw counts in layers['counts'].
song_hvg_counts = ad.AnnData(
    X=get_counts(song)[np.asarray(song_gold)].copy(),
    obs=song.obs.loc[song_gold].copy(),
    var=song.var.copy(),
)
spl_hvg_counts = ad.AnnData(
    X=get_counts(spl)[np.asarray(spl_gold)].copy(),
    obs=spl.obs.loc[spl_gold].copy(),
    var=spl.var.copy(),
)
both = ad.concat([song_hvg_counts, spl_hvg_counts], join="inner", merge="same", index_unique=None)
both.obs["hvg_batch"] = both.obs["dataset"].astype(str) + "|" + both.obs["donor"].astype(str)

count_values = both.X.data if sparse.issparse(both.X) else np.asarray(both.X).ravel()
count_values = count_values[np.isfinite(count_values)]
integer_fraction = float(np.mean(np.isclose(count_values, np.rint(count_values)))) if count_values.size else 1.0
if integer_fraction < 0.999999:
    raise ValueError(f"H2 input is not raw integer counts: integer_fraction={integer_fraction:.8f}")
if both.n_obs != int(song_gold.sum() + spl_gold.sum()):
    raise ValueError("H2 concatenation changed the expected number of cells")

print("H2 Song gold cells:", int(song_gold.sum()))
print("H2 SPL assay-compatible gold cells:", int(spl_gold.sum()))
print("H2 total eligible cells:", both.n_obs)
print("H2 raw-count integer fraction:", integer_fraction)

H2, meta_H2 = select_hvgs(
    both, GRID["primary_hvg_k"], "hvg_batch",
    np.ones(both.n_obs, dtype=bool), exclude_tcr=True,
)
pd.Series(H2, name="gene").to_csv(PROJECT / "features/H2_song_spl_target_informed_TCRfree_backbone.csv", index=False)
write_json(meta_H2 | {"eligible_cells": int(both.n_obs), "target_expression_used": True,
                      "confirmatory_eligible": False,
                      "song_gold_cells": int(song_gold.sum()),
                      "spl_assay_compatible_gold_cells": int(spl_gold.sum()),
                      "spl_gd_assay_complete_donors": spl_gd_assay_donors,
                      "matrix_provenance": "Song X raw counts + SPL counts layer",
                      "raw_count_integer_fraction": integer_fraction},
           PROJECT / "features/H2_metadata.json")
'''),
        code(r'''
tcr_panel = [g for g in TCR_PANEL_DEFAULT if g in genes_common]
pd.Series(tcr_panel, name="gene").to_csv(PROJECT / "features/prespecified_TCR_panel.csv", index=False)
comparison = pd.DataFrame({
    "panel": ["H1", "H2"], "n_backbone": [len(H1), len(H2)],
    "n_TCR_in_backbone": [sum(is_tcr_gene(g) for g in H1), sum(is_tcr_gene(g) for g in H2)],
    "target_informed": [False, True],
})
comparison.to_csv(PROJECT / "results/tables/HVG_panel_summary.csv", index=False)
display(comparison)
'''),
    ],
)


NOTEBOOKS["04_register_splits_and_configurations.ipynb"] = notebook(
    "04 — Register splits and the 12 LR configurations",
    "Freeze model IDs, donor splits, label weights, gene panels, and confirmatory status before model fitting.",
    [
        code(r'''
registry = model_registry(GRID["primary_lambda_TRD"])
registry.to_csv(PROJECT / "config/model_registry.csv", index=False)
display(registry)
assert len(registry) == 12 and registry["model_id"].is_unique
assert registry.loc[registry.model_id.eq("M01"), "label_regime"].item() == "L1_paired_only"
assert registry.loc[registry.model_id.eq("M03"), "label_regime"].item() == "L3_paired_plus_TRD_weighted"
'''),
        code(r'''
song = ad.read_h5ad(PROJECT / "manifests/Song_D1_cohorts.h5ad", backed="r")
donors = sorted(song.obs["donor"].astype(str).unique())
splits = []
for outer in donors:
    train_donors = [x for x in donors if x != outer]
    splits.append({"outer_fold": outer, "train_donors": "|".join(train_donors), "held_out_donor": outer})
splits = pd.DataFrame(splits)
splits.to_csv(PROJECT / "manifests/song_outer_lodo_splits.csv", index=False)
display(splits)
song.file.close()
'''),
        code(r'''
song = ad.read_h5ad(PROJECT / "manifests/Song_D1_cohorts.h5ad", backed="r")
checks = []
for label in registry["label_regime"].unique():
    tab = make_training_table(song.obs, label)
    w = donor_balanced_weights(tab, label, GRID["primary_lambda_TRD"])
    audit = tab.assign(weight=w).groupby(["donor", "positive_source"], observed=True).agg(
        n=("y", "size"), weight_sum=("weight", "sum")
    ).reset_index()
    audit.insert(0, "label_regime", label)
    checks.append(audit)
weight_audit = pd.concat(checks, ignore_index=True)
weight_audit.to_csv(PROJECT / "results/tables/training_weight_audit.csv", index=False)
display(weight_audit)
song.file.close()
'''),
    ],
)


NOTEBOOKS["04b_register_pca_lr_comparison.ipynb"] = notebook(
    "04b — Register inductive versus transductive PCA–LR factorial",
    "Freeze P01–P08 and expression-only representation pools. P01–P04 are TCR-free primary/sensitivity models; P05–P08 are TCR-inclusive positive controls.",
    [
        md(r'''
## Estimands

- **P01/P02 and P05/P06 (inductive):** PCA is fitted only on QC-passing Song T-lineage
  expression from the available training donors.
- **P03/P04 and P07/P08 (transductive):** the same Song expression is combined with
  QC-passing, otherwise unfiltered expression from the five registered SPL
  assay donors. SPL receptor labels are hidden until notebook 08c.

P01–P04 use the same H1 TCR-free backbone and 50 PCs. P05–P08 add only the
prespecified TCR transcript panel to that backbone. Within each gene regime,
the inductive/transductive pair therefore isolates access to the unlabeled
target expression distribution. P05–P08 are supplementary positive controls,
not receptor-independent confirmatory models. M01/M03 and M04/M06 remain the
matched direct-gene LR references.
'''),
        code(r'''
# Reload the authoritative utility module so a kernel that previously imported
# an older version cannot silently keep stale function definitions.
import importlib
import gdt_lr_utils as _gdt_lr_utils
_gdt_lr_utils = importlib.reload(_gdt_lr_utils)
_required_pca_api = [
    "pca_model_registry", "fit_pca_basis", "fit_pc_classifier", "predict_pca_lr"
]
_missing_pca_api = [name for name in _required_pca_api if not hasattr(_gdt_lr_utils, name)]
if _missing_pca_api:
    raise RuntimeError(
        "Woody has an outdated src/gdt_lr_utils.py. Missing: "
        + ", ".join(_missing_pca_api)
        + ". Upload the current local file to "
          "/data/users/kaelynwen/gdT_project/gdt_lr_12model/src/gdt_lr_utils.py, "
          "restart the kernel, and rerun 04b from the top."
    )
for _name in _required_pca_api:
    globals()[_name] = getattr(_gdt_lr_utils, _name)
PCA_N_COMPONENTS = int(GRID.get("pca_n_components", 50))
if "pca_n_components" not in GRID:
    warnings.warn(
        "lr_grid.yaml predates the PCA-LR branch; using the preregistered "
        "fallback pca_n_components=50. Upload the current config/lr_grid.yaml."
    )
print("PCA-LR utility API preflight: PASS")
print("PCA components:", PCA_N_COMPONENTS)
'''),
        code(r'''
song = ad.read_h5ad(PROJECT / "manifests/Song_D1_cohorts.h5ad", backed="r")
spl = ad.read_h5ad(PROJECT / "manifests/CrossAtlas_D2_SPL_cohorts.h5ad", backed="r")
assay_donors = list(map(str, RULES["spl_gd_assay_complete_donors"]))

# Representation membership is frozen using expression/QC and donor coverage
# only. No receptor group, cohort, chain, CDR3, or outcome field is consulted.
source_mask = song.obs["qc_Tlineage_pass"].fillna(False).astype(bool)
target_mask = (
    spl.obs["qc_Tlineage_pass"].fillna(False).astype(bool)
    & spl.obs["donor"].astype(str).isin(assay_donors)
)
source_ids = song.obs_names[source_mask].astype(str)
target_ids = spl.obs_names[target_mask].astype(str)
if not len(source_ids) or not len(target_ids):
    raise ValueError("A registered PCA representation pool is empty")

pool = pd.concat([
    pd.DataFrame({"cell_id": source_ids, "pool": "source_song_qc_Tlineage"}),
    pd.DataFrame({"cell_id": target_ids, "pool": "target_SPL_unlabeled_qc_Tlineage"}),
], ignore_index=True)
pool_path = PROJECT / "manifests/pca_lr_representation_pool.csv"
pool.to_csv(pool_path, index=False)

registry = pca_model_registry(GRID["primary_lambda_TRD"])
registry["n_components"] = PCA_N_COMPONENTS
registry["feature_file"] = "features/H1_song_inductive_TCRfree_backbone.csv"
registry["tcr_panel_file"] = np.where(
    registry["gene_regime"].eq("G2_TCR_inclusive"),
    "features/prespecified_TCR_panel.csv",
    "",
)
registry["feature_construction"] = np.where(
    registry["gene_regime"].eq("G2_TCR_inclusive"),
    "H1_TCRfree_backbone_plus_fixed_TCR_panel",
    "H1_TCRfree_backbone_only",
)
registry.to_csv(PROJECT / "config/pca_lr_model_registry.csv", index=False)
assert registry.model_id.tolist() == [f"P{i:02d}" for i in range(1, 9)]
assert not registry["target_labels_used_for_representation"].any()
assert registry.loc[registry.model_id.isin(["P01", "P02", "P03", "P04"]), "gene_regime"].eq("G1_TCR_free").all()
assert registry.loc[registry.model_id.isin(["P05", "P06", "P07", "P08"]), "gene_regime"].eq("G2_TCR_inclusive").all()

write_json({
    "source_rule": "Song qc_Tlineage_pass == True",
    "target_rule": "SPL qc_Tlineage_pass == True AND donor in registered gd-assay donors",
    "spl_assay_donors": assay_donors,
    "n_source": len(source_ids), "n_target": len(target_ids),
    "target_receptor_labels_used": False,
    "target_evaluation_cells_may_be_present_unlabeled": True,
    "genes": "H1 Song-inductive TCR-free backbone; P05-P08 append the prespecified TCR transcript panel",
    "tcr_inclusive_models": ["P05", "P06", "P07", "P08"],
    "n_components": PCA_N_COMPONENTS,
    "pool_sha256": sha256_file(pool_path),
}, PROJECT / "manifests/pca_lr_experiment_contract.json")
display(registry)
print("Source representation cells:", len(source_ids))
print("Unlabeled SPL representation cells:", len(target_ids))
song.file.close(); spl.file.close()
'''),
        code(r'''
# Outcome fields are opened only after membership has been written and hashed,
# solely to audit the biological composition of the frozen unlabeled pool.
song = ad.read_h5ad(PROJECT / "manifests/Song_D1_cohorts.h5ad", backed="r")
spl = ad.read_h5ad(PROJECT / "manifests/CrossAtlas_D2_SPL_cohorts.h5ad", backed="r")
pool = pd.read_csv(PROJECT / "manifests/pca_lr_representation_pool.csv")
source_ids = pool.loc[pool.pool.eq("source_song_qc_Tlineage"), "cell_id"]
target_ids = pool.loc[pool.pool.eq("target_SPL_unlabeled_qc_Tlineage"), "cell_id"]
audit = pd.concat([
    (song.obs.loc[source_ids].groupby(["donor", "immutable_cohort"], observed=True)
     .size().rename("n_cells").reset_index().assign(pool="source_song_qc_Tlineage")),
    (spl.obs.loc[target_ids].groupby(["donor", "immutable_cohort"], observed=True)
     .size().rename("n_cells").reset_index().assign(pool="target_SPL_unlabeled_qc_Tlineage")),
], ignore_index=True)
audit.to_csv(PROJECT / "results/tables/pca_lr_representation_pool_audit.csv", index=False)
display(audit)
song.file.close(); spl.file.close()
'''),
    ],
)


NOTEBOOKS["05_nested_cv_hyperparameter_selection.ipynb"] = notebook(
    "05 — Nested Song donor CV",
    "Select regularization using inner leave-one-donor-out CV and estimate outer-donor performance on paired gold cells only.",
    [
        code(r'''
from collections import Counter

song = ad.read_h5ad(PROJECT / "manifests/Song_D1_cohorts.h5ad")
registry = pd.read_csv(PROJECT / "config/model_registry.csv")
tcr_panel = pd.read_csv(PROJECT / "features/prespecified_TCR_panel.csv")["gene"].tolist()
donors = sorted(song.obs["donor"].astype(str).unique())
_fold_gene_cache = {}

def fold_genes(train_ids, row):
    donor_key = tuple(sorted(song.obs.loc[train_ids, "donor"].astype(str).unique()))
    key = (row.hvg_regime, row.gene_regime, donor_key)
    if key in _fold_gene_cache:
        return _fold_gene_cache[key]
    if row.hvg_regime == "H1_song_inductive":
        eligible = song.obs.index.isin(train_ids) & song.obs["immutable_cohort"].astype(str).isin(
            ["gold_B_paired_gdT", "gold_B_paired_abT"]
        )
        backbone, _ = select_hvgs(song, GRID["primary_hvg_k"], "donor", eligible, exclude_tcr=True)
    else:
        backbone = pd.read_csv(PROJECT / "features/H2_song_spl_target_informed_TCRfree_backbone.csv")["gene"].tolist()
    genes = backbone if row.gene_regime == "G1_TCR_free" else backbone + [g for g in tcr_panel if g not in backbone]
    _fold_gene_cache[key] = genes
    return genes

def gold_validation(donor):
    mask = song.obs["donor"].astype(str).eq(donor) & song.obs["immutable_cohort"].astype(str).isin(
        ["gold_B_paired_gdT", "gold_B_paired_abT"]
    )
    tab = song.obs.loc[mask, ["donor", "immutable_cohort"]].copy()
    tab["y"] = tab["immutable_cohort"].astype(str).eq("gold_B_paired_gdT").astype(int)
    return tab
'''),
        code(r'''
selection_rows, outer_predictions = [], []
for row in registry.itertuples(index=False):
    print("Nested CV", row.model_id)
    full_training = make_training_table(song.obs, row.label_regime)
    for outer in donors:
        outer_train_donors = [d for d in donors if d != outer]
        scores = []
        for C in GRID["C_grid"]:
            inner_scores = []
            for inner in outer_train_donors:
                fit_donors = [d for d in outer_train_donors if d != inner]
                train = full_training[full_training["donor"].astype(str).isin(fit_donors)]
                valid = gold_validation(inner)
                genes = fold_genes(train.index, row)
                model = fit_lr(song, train, genes, C, row.label_regime, row.lambda_TRD, GRID["seed"])
                p = predict_lr(model, song, genes, valid.index)
                inner_scores.append(average_precision_score(valid["y"], p.loc[valid.index]))
            scores.append((float(C), float(np.mean(inner_scores))))
        best_C, best_score = max(scores, key=lambda x: (x[1], -x[0]))
        train = full_training[full_training["donor"].astype(str).isin(outer_train_donors)]
        valid = gold_validation(outer)
        genes = fold_genes(train.index, row)
        model = fit_lr(song, train, genes, best_C, row.label_regime, row.lambda_TRD, GRID["seed"])
        p = predict_lr(model, song, genes, valid.index)
        selection_rows.append({"model_id": row.model_id, "outer_donor": outer,
                               "best_C": best_C, "inner_mean_auprc": best_score})
        outer_predictions.append(pd.DataFrame({
            "cell_id": valid.index, "donor": outer, "model_id": row.model_id,
            "y": valid["y"].to_numpy(), "p_gdT": p.loc[valid.index].to_numpy(),
            "selected_C": best_C,
        }))

selection = pd.DataFrame(selection_rows)
oof = pd.concat(outer_predictions, ignore_index=True)
selection.to_csv(PROJECT / "results/tables/nested_cv_C_selection.csv", index=False)
oof.to_parquet(PROJECT / "predictions/song_nested_oof_predictions.parquet", index=False)
display(selection)
'''),
        code(r'''
chosen = (selection.groupby("model_id")["best_C"]
          .agg(lambda x: sorted(Counter(x).items(), key=lambda z: (-z[1], z[0]))[0][0])
          .rename("optimized_C").reset_index())
common_C = float(chosen.loc[chosen.model_id.eq("M01"), "optimized_C"].item())
chosen["controlled_common_C"] = common_C
chosen.to_csv(PROJECT / "results/tables/final_C_choices.csv", index=False)

thresholds = oof.groupby("model_id").apply(
    lambda x: threshold_max_f1(x["y"], x["p_gdT"])
).rename("optimized_oof_threshold").reset_index()
thresholds.to_csv(PROJECT / "results/tables/optimized_song_oof_thresholds.csv", index=False)
print("Controlled common C from M01:", common_C)
display(chosen.merge(thresholds))
'''),
    ],
)


def fit_notebook(controlled: bool):
    label = "controlled" if controlled else "optimized"
    num = "06" if controlled else "07"
    return notebook(
        f"{num} — Fit 12 {label} LR models",
        ("Use one common C chosen by M01 for causal comparability." if controlled else
         "Use each configuration's nested-CV-selected C as a secondary optimized analysis."),
        [
            code(r'''
song = ad.read_h5ad(PROJECT / "manifests/Song_D1_cohorts.h5ad")
registry = pd.read_csv(PROJECT / "config/model_registry.csv")
C_choices = pd.read_csv(PROJECT / "results/tables/final_C_choices.csv")
tcr_panel = pd.read_csv(PROJECT / "features/prespecified_TCR_panel.csv")["gene"].tolist()
donors = sorted(song.obs["donor"].astype(str).unique())
_fold_gene_cache = {}

def final_genes(row):
    filename = ("H1_song_inductive_TCRfree_backbone.csv" if row.hvg_regime == "H1_song_inductive"
                else "H2_song_spl_target_informed_TCRfree_backbone.csv")
    backbone = pd.read_csv(PROJECT / "features" / filename)["gene"].tolist()
    return backbone if row.gene_regime == "G1_TCR_free" else backbone + [g for g in tcr_panel if g not in backbone]

def fold_genes(train_ids, row):
    # H1 feature selection is repeated inside the held-out-donor split.
    donor_key = tuple(sorted(song.obs.loc[train_ids, "donor"].astype(str).unique()))
    key = (row.hvg_regime, row.gene_regime, donor_key)
    if key in _fold_gene_cache:
        return _fold_gene_cache[key]
    if row.hvg_regime == "H1_song_inductive":
        eligible = song.obs.index.isin(train_ids) & song.obs["immutable_cohort"].astype(str).isin(
            ["gold_B_paired_gdT", "gold_B_paired_abT"]
        )
        backbone, _ = select_hvgs(song, GRID["primary_hvg_k"], "donor", eligible, exclude_tcr=True)
    else:
        # Legacy H2 is target-gold-informed HVG sensitivity, not joint PCA.
        backbone = pd.read_csv(PROJECT / "features/H2_song_spl_target_informed_TCRfree_backbone.csv")["gene"].tolist()
    genes = backbone if row.gene_regime == "G1_TCR_free" else backbone + [g for g in tcr_panel if g not in backbone]
    _fold_gene_cache[key] = genes
    return genes
'''),
            code((r'''
analysis_name = "CONTROLLED"
C_column = "controlled_common_C"
''' if controlled else r'''
analysis_name = "OPTIMIZED"
C_column = "optimized_C"
''') + r'''
fit_rows, predictions = [], []
for row in registry.itertuples(index=False):
    C = float(C_choices.loc[C_choices.model_id.eq(row.model_id), C_column].item())
    genes = final_genes(row)
    train_all = make_training_table(song.obs, row.label_regime)
    # LODO OOF with the frozen final panel; used only to freeze this analysis threshold.
    for donor in donors:
        train = train_all[~train_all["donor"].astype(str).eq(donor)]
        valid_mask = song.obs["donor"].astype(str).eq(donor) & song.obs["immutable_cohort"].astype(str).isin(
            ["gold_B_paired_gdT", "gold_B_paired_abT"]
        )
        valid = song.obs.loc[valid_mask]
        y = valid["immutable_cohort"].astype(str).eq("gold_B_paired_gdT").astype(int)
        genes_fold = fold_genes(train.index, row)
        fold_model = fit_lr(song, train, genes_fold, C, row.label_regime, row.lambda_TRD, GRID["seed"])
        p = predict_lr(fold_model, song, genes_fold, valid.index)
        predictions.append(pd.DataFrame({"cell_id": valid.index, "donor": donor,
            "model_id": row.model_id, "y": y.to_numpy(), "p_gdT": p.loc[valid.index].to_numpy()}))
    model = fit_lr(song, train_all, genes, C, row.label_regime, row.lambda_TRD, GRID["seed"])
    model_path = PROJECT / "models" / f"{analysis_name.lower()}_{row.model_id}.joblib"
    save_model_bundle(model, genes, {
        "model_id": row.model_id, "analysis": analysis_name, "C": C,
        "label_regime": row.label_regime, "hvg_regime": row.hvg_regime,
        "gene_regime": row.gene_regime, "lambda_TRD": row.lambda_TRD,
        "confirmatory_eligible": bool(row.confirmatory_eligible),
    }, model_path)
    fit_rows.append({"model_id": row.model_id, "analysis": analysis_name, "C": C,
                     "n_training": len(train_all), "n_genes": len(genes), "model_path": str(model_path)})

oof = pd.concat(predictions, ignore_index=True)
thresholds = oof.groupby("model_id").apply(
    lambda x: threshold_max_f1(x["y"], x["p_gdT"])
).rename("threshold").reset_index()
pd.DataFrame(fit_rows).merge(thresholds).to_csv(
    PROJECT / "results/tables" / f"{analysis_name.lower()}_fit_registry.csv", index=False
)
oof.to_parquet(PROJECT / "predictions" / f"song_{analysis_name.lower()}_oof.parquet", index=False)
display(pd.DataFrame(fit_rows).merge(thresholds))
'''),
        ],
    )


NOTEBOOKS["06_fit_12_controlled_lr_models.ipynb"] = fit_notebook(True)
NOTEBOOKS["07_fit_12_optimized_lr_models.ipynb"] = fit_notebook(False)
NOTEBOOKS["07_fit_12_optimized_lr_models.ipynb"]["cells"].extend([
    md('''
## Prespecified M03 lambda-TRD sensitivity

This is outside the 12-model core. It keeps H1, G1, and the controlled M01 C
fixed while varying only orphan-TRD aggregate contribution (0.25, 0.50, 1.00).
'''),
    code(r'''
row = registry.loc[registry.model_id.eq("M03")].iloc[0]
C = float(C_choices.loc[C_choices.model_id.eq("M01"), "controlled_common_C"].item())
genes = final_genes(row)
lambda_rows = []
for lambda_trd in GRID["lambda_TRD_sensitivity"] + [GRID["primary_lambda_TRD"]]:
    lambda_trd = float(lambda_trd)
    train_all = make_training_table(song.obs, row.label_regime)
    fold_predictions = []
    for donor in donors:
        train = train_all[~train_all["donor"].astype(str).eq(donor)]
        valid_mask = song.obs["donor"].astype(str).eq(donor) & song.obs["immutable_cohort"].astype(str).isin(
            ["gold_B_paired_gdT", "gold_B_paired_abT"]
        )
        valid = song.obs.loc[valid_mask]
        genes_fold = fold_genes(train.index, row)
        model_fold = fit_lr(song, train, genes_fold, C, row.label_regime, lambda_trd, GRID["seed"])
        p = predict_lr(model_fold, song, genes_fold, valid.index)
        fold_predictions.append(pd.DataFrame({"cell_id": valid.index,
            "y": valid["immutable_cohort"].astype(str).eq("gold_B_paired_gdT").astype(int).to_numpy(),
            "p_gdT": p.loc[valid.index].to_numpy()}))
    oof_lambda = pd.concat(fold_predictions, ignore_index=True)
    threshold = threshold_max_f1(oof_lambda.y, oof_lambda.p_gdT)
    final_model = fit_lr(song, train_all, genes, C, row.label_regime, lambda_trd, GRID["seed"])
    tag = str(lambda_trd).replace(".", "p")
    path = PROJECT / f"models/sensitivity_M03_lambdaTRD_{tag}.joblib"
    save_model_bundle(final_model, genes, {"base_model": "M03", "lambda_TRD": lambda_trd,
                                          "C": C, "threshold": threshold}, path)
    lambda_rows.append({"lambda_TRD": lambda_trd, "C": C, "threshold": threshold,
                        "song_oof_auprc": average_precision_score(oof_lambda.y, oof_lambda.p_gdT),
                        "model_path": str(path)})
pd.DataFrame(lambda_rows).sort_values("lambda_TRD").to_csv(
    PROJECT / "results/tables/M03_lambdaTRD_sensitivity_registry.csv", index=False
)
display(pd.DataFrame(lambda_rows).sort_values("lambda_TRD"))
'''),
])


NOTEBOOKS["07b_fit_inductive_transductive_pca_lr.ipynb"] = notebook(
    "07b — Fit inductive and transductive PCA–LR factorial",
    "Fit P01–P08 with nested Song donor CV. SPL expression enters only transductive PCA; SPL receptor labels remain unavailable. P05–P08 are TCR-inclusive positive controls.",
    [
        code(r'''
from collections import Counter
import joblib

PCA_N_COMPONENTS = int(GRID.get("pca_n_components", 50))
if "pca_n_components" not in GRID:
    warnings.warn(
        "lr_grid.yaml predates the PCA-LR branch; using the preregistered "
        "fallback pca_n_components=50."
    )

song = ad.read_h5ad(PROJECT / "manifests/Song_D1_cohorts.h5ad")
spl = ad.read_h5ad(PROJECT / "manifests/CrossAtlas_D2_SPL_cohorts.h5ad")
registry = pd.read_csv(PROJECT / "config/pca_lr_model_registry.csv")
pool = pd.read_csv(PROJECT / "manifests/pca_lr_representation_pool.csv")
contract = json.load(open(PROJECT / "manifests/pca_lr_experiment_contract.json"))
pool_path = PROJECT / "manifests/pca_lr_representation_pool.csv"
if sha256_file(pool_path) != contract["pool_sha256"]:
    raise RuntimeError("The frozen PCA representation-pool manifest has changed")

source_pool = pd.Index(pool.loc[pool.pool.eq("source_song_qc_Tlineage"), "cell_id"].astype(str))
target_pool = pd.Index(pool.loc[pool.pool.eq("target_SPL_unlabeled_qc_Tlineage"), "cell_id"].astype(str))
final_backbone = pd.read_csv(PROJECT / "features/H1_song_inductive_TCRfree_backbone.csv")["gene"].tolist()
tcr_panel = pd.read_csv(PROJECT / "features/prespecified_TCR_panel.csv")["gene"].tolist()
assert not any(is_tcr_gene(g) for g in final_backbone)
assert tcr_panel and all(is_tcr_gene(g) for g in tcr_panel)
donors = sorted(song.obs["donor"].astype(str).unique())

def gold_validation(donor):
    mask = song.obs["donor"].astype(str).eq(donor) & song.obs["immutable_cohort"].astype(str).isin(
        ["gold_B_paired_gdT", "gold_B_paired_abT"]
    )
    tab = song.obs.loc[mask, ["donor", "immutable_cohort"]].copy()
    tab["y"] = tab["immutable_cohort"].astype(str).eq("gold_B_paired_gdT").astype(int)
    return tab

_backbone_cache, _gene_cache, _pca_cache = {}, {}, {}

def fold_genes(fit_donors, gene_regime):
    donor_key = tuple(sorted(map(str, fit_donors)))
    if donor_key not in _backbone_cache:
        eligible = (
            song.obs["donor"].astype(str).isin(donor_key)
            & song.obs["immutable_cohort"].astype(str).isin(["gold_B_paired_gdT", "gold_B_paired_abT"])
        )
        _backbone_cache[donor_key], _ = select_hvgs(
            song, GRID["primary_hvg_k"], "donor", eligible, exclude_tcr=True
        )
    key = (donor_key, str(gene_regime))
    if key not in _gene_cache:
        backbone = _backbone_cache[donor_key]
        _gene_cache[key] = (
            backbone if gene_regime == "G1_TCR_free"
            else backbone + [g for g in tcr_panel if g not in backbone]
        )
    return _gene_cache[key]

def pca_for(fit_donors, representation, genes):
    fit_donors = tuple(sorted(map(str, fit_donors)))
    key = (fit_donors, representation, tuple(genes))
    if key not in _pca_cache:
        source_ids = source_pool.intersection(
            song.obs_names[song.obs["donor"].astype(str).isin(fit_donors)]
        )
        is_transductive = str(representation).startswith("transductive_")
        _pca_cache[key] = fit_pca_basis(
            source_adata=song, genes=genes,
            representation_source_cells=source_ids,
            target_adata=spl if is_transductive else None,
            representation_target_cells=target_pool if is_transductive else None,
            n_components=PCA_N_COMPONENTS, seed=GRID["seed"],
        )
    return _pca_cache[key]
'''),
        code(r'''
# Fully nested Song LODO: neither C nor the threshold is selected on SPL.
selection_rows, outer_predictions = [], []
for row in registry.itertuples(index=False):
    print("Nested PCA-LR", row.model_id, row.representation)
    full_training = make_training_table(song.obs, row.label_regime)
    for outer in donors:
        outer_train_donors = [d for d in donors if d != outer]
        scores = []
        for C in GRID["C_grid"]:
            inner_scores = []
            for inner in outer_train_donors:
                fit_donors = [d for d in outer_train_donors if d != inner]
                train = full_training[full_training["donor"].astype(str).isin(fit_donors)]
                valid = gold_validation(inner)
                genes = fold_genes(fit_donors, row.gene_regime)
                basis = pca_for(fit_donors, row.representation, genes)
                classifier = fit_pc_classifier(
                    song, train, genes, basis["pca"], C,
                    row.label_regime, row.lambda_TRD, GRID["seed"],
                )
                p = predict_pca_lr(basis | {"model": classifier}, song, valid.index)
                inner_scores.append(average_precision_score(valid["y"], p.loc[valid.index]))
            scores.append((float(C), float(np.mean(inner_scores))))
        best_C, best_score = max(scores, key=lambda x: (x[1], -x[0]))

        train = full_training[full_training["donor"].astype(str).isin(outer_train_donors)]
        valid = gold_validation(outer)
        genes = fold_genes(outer_train_donors, row.gene_regime)
        basis = pca_for(outer_train_donors, row.representation, genes)
        classifier = fit_pc_classifier(
            song, train, genes, basis["pca"], best_C,
            row.label_regime, row.lambda_TRD, GRID["seed"],
        )
        p = predict_pca_lr(basis | {"model": classifier}, song, valid.index)
        selection_rows.append({"model_id": row.model_id, "outer_donor": outer,
                               "best_C": best_C, "inner_mean_auprc": best_score})
        outer_predictions.append(pd.DataFrame({
            "cell_id": valid.index, "donor": outer, "model_id": row.model_id,
            "y": valid["y"].to_numpy(), "p_gdT": p.loc[valid.index].to_numpy(),
            "selected_C": best_C,
        }))

selection = pd.DataFrame(selection_rows)
oof = pd.concat(outer_predictions, ignore_index=True)
selection.to_csv(PROJECT / "results/tables/pca_lr_nested_cv_C_selection.csv", index=False)
oof.to_parquet(PROJECT / "predictions/song_pca_lr_nested_oof_predictions.parquet", index=False)

choices = (selection.groupby("model_id")["best_C"]
           .agg(lambda x: sorted(Counter(x).items(), key=lambda z: (-z[1], z[0]))[0][0])
           .rename("optimized_C").reset_index())
thresholds = (oof.groupby("model_id").apply(lambda x: threshold_max_f1(x.y, x.p_gdT))
              .rename("threshold").reset_index())
threshold_recall80 = (
    oof.groupby("model_id")
    .apply(lambda x: threshold_for_min_recall(x.y, x.p_gdT, 0.80))
    .rename("threshold_recall80_source").reset_index()
)
threshold_recall90 = (
    oof.groupby("model_id")
    .apply(lambda x: threshold_for_min_recall(x.y, x.p_gdT, 0.90))
    .rename("threshold_recall90_source").reset_index()
)
choices = choices.merge(thresholds, on="model_id").merge(
    threshold_recall80, on="model_id"
).merge(threshold_recall90, on="model_id")
choices.to_csv(PROJECT / "results/tables/pca_lr_final_choices.csv", index=False)
display(selection)
display(choices)
'''),
        code(r'''
# Final bases use all Song source-expression cells. Only transductive models
# additionally use the frozen unlabeled SPL expression pool. Supervised LR
# still uses Song labels only.
fit_rows = []
for row in registry.itertuples(index=False):
    choice = choices.set_index("model_id").loc[row.model_id]
    is_transductive = str(row.representation).startswith("transductive_")
    final_genes = (
        final_backbone if row.gene_regime == "G1_TCR_free"
        else final_backbone + [g for g in tcr_panel if g not in final_backbone]
    )
    basis = fit_pca_basis(
        source_adata=song, genes=final_genes,
        representation_source_cells=source_pool,
        target_adata=spl if is_transductive else None,
        representation_target_cells=target_pool if is_transductive else None,
        n_components=PCA_N_COMPONENTS, seed=GRID["seed"],
    )
    train = make_training_table(song.obs, row.label_regime)
    classifier = fit_pc_classifier(
        song, train, final_genes, basis["pca"], float(choice.optimized_C),
        row.label_regime, row.lambda_TRD, GRID["seed"],
    )
    bundle = basis | {
        "model": classifier,
        "metadata": {
            "model_id": row.model_id, "representation": row.representation,
            "label_regime": row.label_regime, "lambda_TRD": float(row.lambda_TRD),
            "C": float(choice.optimized_C), "threshold": float(choice.threshold),
            "threshold_recall80_source": float(choice.threshold_recall80_source),
            "threshold_recall90_source": float(choice.threshold_recall90_source),
            "hvg_regime": "H1_song_inductive", "gene_regime": row.gene_regime,
            "target_expression_used": bool(is_transductive),
            "target_labels_used": False,
            "representation_pool_sha256": contract["pool_sha256"],
        },
    }
    model_path = PROJECT / f"models/pca_lr_{row.model_id}.joblib"
    joblib.dump(bundle, model_path)
    fit_rows.append({
        "model_id": row.model_id, "representation": row.representation,
        "label_regime": row.label_regime, "gene_regime": row.gene_regime,
        "C": float(choice.optimized_C), "threshold": float(choice.threshold),
        "threshold_recall80_source": float(choice.threshold_recall80_source),
        "threshold_recall90_source": float(choice.threshold_recall90_source),
        "n_training": len(train),
        "n_genes": len(final_genes), "n_components": basis["n_components_effective"],
        "n_source_representation": basis["n_source_representation_cells"],
        "n_target_representation": basis["n_target_representation_cells"],
        "model_path": str(model_path),
    })

fit_registry = pd.DataFrame(fit_rows)
fit_registry.to_csv(PROJECT / "results/tables/pca_lr_fit_registry.csv", index=False)
display(fit_registry)
'''),
        md(r'''
## Stop conditions

Do not run notebook 08c if any inductive model reports target representation
cells, if any transductive model reports zero target cells, if P01–P04 contain
TCR genes, if P05–P08 lack the prespecified TCR panel, or if C/threshold was
derived from SPL. P05–P08 must remain supplementary positive controls.
'''),
    ],
)


NOTEBOOKS["08_external_gold_evaluation.ipynb"] = notebook(
    "08 — Untouched SPL gold evaluation",
    "Evaluate frozen models only on receptor-confirmed paired SPL gold cells. None/orphan/mixed cells are excluded from PR and FPR denominators.",
    [
        code(r'''
spl = ad.read_h5ad(PROJECT / "manifests/CrossAtlas_D2_SPL_cohorts.h5ad")
assay_donors = list(map(str, RULES["spl_gd_assay_complete_donors"]))
gold_cohort = spl.obs["immutable_cohort"].astype(str).isin(
    ["gold_B_paired_gdT", "gold_B_paired_abT"]
)
assay_covered = spl.obs["donor"].astype(str).isin(assay_donors)
gold = gold_cohort & assay_covered
gold_ids = spl.obs_names[gold]
y = spl.obs.loc[gold_ids, "immutable_cohort"].astype(str).eq("gold_B_paired_gdT").astype(int)
excluded_gold = spl.obs.loc[gold_cohort & ~assay_covered, ["donor", "immutable_cohort"]]

assert set(spl.obs.loc[gold_ids, "donor"].astype(str).unique()) == set(assay_donors)
assert y.nunique() == 2, "Assay-compatible SPL gold must contain both paired gdT and paired abT."
print("Registered SPL gd-assay donors:", assay_donors)
print("External assay-compatible gold cells:", len(gold_ids),
      "positives:", int(y.sum()),
      "negatives:", int((1 - y).sum()),
      "donors:", spl.obs.loc[gold_ids, "donor"].nunique())
print("Excluded paired-ab/gd gold cells from non-covered donors:", len(excluded_gold))
display(pd.crosstab(spl.obs.loc[gold_ids, "donor"],
                    spl.obs.loc[gold_ids, "immutable_cohort"], margins=True))
'''),
        code(r'''
rows, predictions = [], []
for analysis in ("controlled", "optimized"):
    fits = pd.read_csv(PROJECT / f"results/tables/{analysis}_fit_registry.csv")
    for fit in fits.itertuples(index=False):
        bundle = load_model_bundle(PROJECT / f"models/{analysis}_{fit.model_id}.joblib")
        p = predict_lr(bundle["model"], spl, bundle["genes"], gold_ids)
        pred = pd.DataFrame({"cell_id": gold_ids, "dataset": "CrossAtlas_D2_SPL",
            "donor": spl.obs.loc[gold_ids, "donor"].astype(str).to_numpy(),
            "model_id": fit.model_id, "analysis": analysis, "y": y.to_numpy(),
            "p_gdT": p.loc[gold_ids].to_numpy()})
        predictions.append(pred)
        overall = metrics_at_threshold(pred.y, pred.p_gdT, fit.threshold)
        rows.append(overall | {"model_id": fit.model_id, "analysis": analysis, "scope": "all_donors"})
        for donor, sub in pred.groupby("donor"):
            rows.append(metrics_at_threshold(sub.y, sub.p_gdT, fit.threshold) |
                        {"model_id": fit.model_id, "analysis": analysis, "scope": f"donor:{donor}"})

metrics = pd.DataFrame(rows)
predictions = pd.concat(predictions, ignore_index=True)
metrics.to_csv(PROJECT / "results/tables/SPL_external_gold_metrics.csv", index=False)
predictions.to_parquet(PROJECT / "predictions/SPL_external_gold_predictions.parquet", index=False)
display(metrics.query("scope == 'all_donors'").sort_values(["analysis", "auprc"], ascending=[True, False]))
'''),
        code(r'''
# Predeclared ultra-strict Gold A sensitivity: exactly one partner chain of the
# correct lineage and zero chains of the opposite lineage.
gold_A_ids = spl.obs_names[
    (spl.obs["gold_A_paired_gdT"].fillna(False) | spl.obs["gold_A_paired_abT"].fillna(False))
    & assay_covered
]
gold_A_metrics = []
for (analysis, model_id), sub in predictions[predictions.cell_id.isin(gold_A_ids)].groupby(["analysis", "model_id"]):
    threshold = float(pd.read_csv(PROJECT / f"results/tables/{analysis}_fit_registry.csv")
                      .set_index("model_id").loc[model_id, "threshold"])
    gold_A_metrics.append(metrics_at_threshold(sub.y, sub.p_gdT, threshold) |
                          {"analysis": analysis, "model_id": model_id, "scope": "gold_A_exact_1plus1"})
gold_A_metrics = pd.DataFrame(gold_A_metrics)
gold_A_metrics.to_csv(PROJECT / "results/tables/SPL_external_gold_A_sensitivity.csv", index=False)
display(gold_A_metrics.sort_values(["analysis", "auprc"], ascending=[True, False]))
'''),
        code(r'''
# Primary paired comparison on exactly the same SPL cells.
primary = predictions.query("analysis == 'controlled' and model_id in ['M01','M03']")
pivot = primary.pivot(index="cell_id", columns="model_id", values="p_gdT").join(
    primary.drop_duplicates("cell_id").set_index("cell_id")[["y", "donor"]]
)
point = average_precision_score(pivot.y, pivot.M03) - average_precision_score(pivot.y, pivot.M01)
rng = np.random.default_rng(GRID["seed"])
donors = pivot.donor.unique()
boot = []
for _ in range(GRID["bootstrap_replicates"]):
    sampled = rng.choice(donors, len(donors), replace=True)
    pieces = [pivot[pivot.donor.eq(d)].assign(_draw=i) for i, d in enumerate(sampled)]
    b = pd.concat(pieces)
    boot.append(average_precision_score(b.y, b.M03) - average_precision_score(b.y, b.M01))
result = {"contrast": "M03-M01", "endpoint": "SPL gold AUPRC", "estimate": point,
          "donor_bootstrap_ci_low": float(np.quantile(boot, .025)),
          "donor_bootstrap_ci_high": float(np.quantile(boot, .975))}
pd.DataFrame([result]).to_csv(PROJECT / "results/tables/primary_M03_vs_M01.csv", index=False)
display(pd.Series(result))
'''),
        code(r'''
# Clone-balanced sensitivity: repeatedly retain one cell per donor/receptor clone.
# This prevents large expanded clonotypes from dominating the cell-level AUPRC.
obs_gold = spl.obs.loc[gold_ids].copy()
gd_key = (obs_gold.get("TRG_cdr3", pd.Series("NA", index=obs_gold.index)).fillna("NA").astype(str) + "|" +
          obs_gold.get("TRD_cdr3", pd.Series("NA", index=obs_gold.index)).fillna("NA").astype(str))
ab_key = (obs_gold.get("TRA_cdr3", pd.Series("NA", index=obs_gold.index)).fillna("NA").astype(str) + "|" +
          obs_gold.get("TRB_cdr3", pd.Series("NA", index=obs_gold.index)).fillna("NA").astype(str))
obs_gold["clone_key"] = np.where(y.eq(1), gd_key, ab_key)
obs_gold["clone_key"] = obs_gold["donor"].astype(str) + "|" + obs_gold["clone_key"].astype(str)

rng = np.random.default_rng(GRID["seed"])
clone_rows = []
for analysis in ("controlled", "optimized"):
    for model_id in ("M01", "M03"):
        psub = predictions.query("analysis == @analysis and model_id == @model_id").set_index("cell_id")
        for rep in range(GRID["clone_dedup_replicates"]):
            chosen = (obs_gold.assign(_u=rng.random(len(obs_gold)))
                      .sort_values("_u").drop_duplicates("clone_key").index)
            z = psub.loc[chosen]
            clone_rows.append({"analysis": analysis, "model_id": model_id, "replicate": rep,
                               "n_cells": len(z), "auprc": average_precision_score(z.y, z.p_gdT)})
clone_sensitivity = pd.DataFrame(clone_rows)
clone_sensitivity.to_csv(PROJECT / "results/tables/SPL_clone_balanced_sensitivity.csv", index=False)
display(clone_sensitivity.groupby(["analysis", "model_id"]).auprc.agg(["mean", "std", "min", "max"]))
'''),
        code(r'''
lambda_registry = pd.read_csv(PROJECT / "results/tables/M03_lambdaTRD_sensitivity_registry.csv")
lambda_metrics = []
for row in lambda_registry.itertuples(index=False):
    bundle = load_model_bundle(row.model_path)
    p = predict_lr(bundle["model"], spl, bundle["genes"], gold_ids)
    lambda_metrics.append(metrics_at_threshold(y, p.loc[gold_ids], row.threshold) |
                          {"lambda_TRD": row.lambda_TRD, "analysis": "prespecified_sensitivity"})
lambda_metrics = pd.DataFrame(lambda_metrics).sort_values("lambda_TRD")
lambda_metrics.to_csv(PROJECT / "results/tables/SPL_M03_lambdaTRD_sensitivity.csv", index=False)
display(lambda_metrics)
'''),
    ],
)


NOTEBOOKS["08c_inductive_transductive_pca_lr_evaluation.ipynb"] = notebook(
    "08c — Inductive versus transductive PCA–LR factorial evaluation",
    "Evaluate P01–P08 on exactly the same paired receptor-confirmed SPL gold cells as notebook 08. P05–P08 are TCR-inclusive supplementary positive controls.",
    [
        md(r'''
## Interpretation boundary

P01/P02 and P05/P06 are inductive external models: neither SPL expression nor
labels entered their PCA or LR. P03/P04 and P07/P08 are transductive: unlabeled
expression from the target SPL cohort—including evaluation cells—entered joint
PCA, but SPL labels did not enter PCA, LR, C selection, or threshold selection.
Therefore transductive models must not be described as fully independent
external validation. P05–P08 retain prespecified receptor-transcript genes and
are positive controls, not evidence of receptor-independent classification.
'''),
        code(r'''
import joblib

spl = ad.read_h5ad(PROJECT / "manifests/CrossAtlas_D2_SPL_cohorts.h5ad")
fits = pd.read_csv(PROJECT / "results/tables/pca_lr_fit_registry.csv")
tcr_panel = pd.read_csv(PROJECT / "features/prespecified_TCR_panel.csv")["gene"].tolist()
assert fits.model_id.tolist() == [f"P{i:02d}" for i in range(1, 9)]
assay_donors = list(map(str, RULES["spl_gd_assay_complete_donors"]))
gold = (
    spl.obs["immutable_cohort"].astype(str).isin(["gold_B_paired_gdT", "gold_B_paired_abT"])
    & spl.obs["donor"].astype(str).isin(assay_donors)
)
gold_ids = spl.obs_names[gold]
y = spl.obs.loc[gold_ids, "immutable_cohort"].astype(str).eq("gold_B_paired_gdT").astype(int)
assert len(gold_ids) == 5046, f"Expected the registered 5,046 SPL gold cells, found {len(gold_ids)}"
assert int(y.sum()) == 378 and int((1-y).sum()) == 4668
display(pd.crosstab(spl.obs.loc[gold_ids, "donor"],
                    spl.obs.loc[gold_ids, "immutable_cohort"], margins=True))
'''),
        code(r'''
rows, predictions = [], []
for fit in fits.itertuples(index=False):
    bundle = joblib.load(fit.model_path)
    metadata = bundle["metadata"]
    assert metadata["target_labels_used"] is False
    assert str(metadata["gene_regime"]) == str(fit.gene_regime)
    if str(fit.representation).startswith("inductive_"):
        assert bundle["n_target_representation_cells"] == 0
    else:
        assert bundle["n_target_representation_cells"] > 0
    if str(fit.gene_regime) == "G1_TCR_free":
        assert not any(is_tcr_gene(g) for g in bundle["genes"])
    else:
        assert tcr_panel and all(g in bundle["genes"] for g in tcr_panel)
    p = predict_pca_lr(bundle, spl, gold_ids)
    pred = pd.DataFrame({
        "cell_id": gold_ids, "donor": spl.obs.loc[gold_ids, "donor"].astype(str).to_numpy(),
        "model_id": fit.model_id, "representation": fit.representation,
        "label_regime": fit.label_regime, "gene_regime": fit.gene_regime,
        "y": y.to_numpy(),
        "p_gdT": p.loc[gold_ids].to_numpy(),
    })
    predictions.append(pred)
    rows.append(metrics_at_threshold(pred.y, pred.p_gdT, fit.threshold) |
                {"model_id": fit.model_id, "representation": fit.representation,
                 "label_regime": fit.label_regime, "gene_regime": fit.gene_regime,
                 "scope": "all_donors"})
    for donor, sub in pred.groupby("donor"):
        rows.append(metrics_at_threshold(sub.y, sub.p_gdT, fit.threshold) |
                    {"model_id": fit.model_id, "representation": fit.representation,
                     "label_regime": fit.label_regime, "gene_regime": fit.gene_regime,
                     "scope": f"donor:{donor}"})

predictions = pd.concat(predictions, ignore_index=True)
metrics = pd.DataFrame(rows)
predictions.to_parquet(PROJECT / "predictions/SPL_pca_lr_gold_predictions.parquet", index=False)
metrics.to_csv(PROJECT / "results/tables/SPL_pca_lr_gold_metrics.csv", index=False)
display(metrics.query("scope == 'all_donors'").sort_values("auprc", ascending=False))
'''),
        code(r'''
# Prespecified paired contrasts. Resample biological donors, not cells.
def donor_bootstrap_delta(frame, model_b, model_a, endpoint="auprc"):
    wide = (frame[frame.model_id.isin([model_a, model_b])]
            .pivot(index="cell_id", columns="model_id", values="p_gdT")
            .join(frame.drop_duplicates("cell_id").set_index("cell_id")[["y", "donor"]]))
    score = average_precision_score if endpoint == "auprc" else roc_auc_score
    point = float(score(wide.y, wide[model_b]) - score(wide.y, wide[model_a]))
    rng = np.random.default_rng(GRID["seed"])
    donors = wide.donor.unique()
    draws = []
    for _ in range(GRID["bootstrap_replicates"]):
        sampled = rng.choice(donors, len(donors), replace=True)
        z = pd.concat([wide[wide.donor.eq(d)].assign(_draw=i) for i, d in enumerate(sampled)])
        draws.append(score(z.y, z[model_b]) - score(z.y, z[model_a]))
    return {"contrast": f"{model_b}-{model_a}", "endpoint": endpoint,
            "estimate": point, "donor_bootstrap_ci_low": float(np.quantile(draws, .025)),
            "donor_bootstrap_ci_high": float(np.quantile(draws, .975))}

contrast_rows = []
for b, a in (("P03", "P01"), ("P04", "P02"),
             ("P07", "P05"), ("P08", "P06")):
    contrast_rows.append(donor_bootstrap_delta(predictions, b, a, "auprc"))
    contrast_rows.append(donor_bootstrap_delta(predictions, b, a, "auroc"))
contrasts = pd.DataFrame(contrast_rows)
contrasts.to_csv(PROJECT / "results/tables/PCA_LR_transductive_minus_inductive.csv", index=False)
display(contrasts)

# Supplementary receptor-transcript contribution at matched representation and
# label regime. These are diagnostic positive-control contrasts, not primary.
receptor_rows = []
for b, a in (("P05", "P01"), ("P06", "P02"),
             ("P07", "P03"), ("P08", "P04")):
    receptor_rows.append(donor_bootstrap_delta(predictions, b, a, "auprc"))
    receptor_rows.append(donor_bootstrap_delta(predictions, b, a, "auroc"))
receptor_contrasts = pd.DataFrame(receptor_rows)
receptor_contrasts.to_csv(
    PROJECT / "results/tables/PCA_LR_TCRinclusive_minus_TCRfree.csv", index=False
)
display(receptor_contrasts)
'''),
        code(r'''
# Reference contrasts against the existing direct-gene LR models isolate the
# effect of PCA compression from target-distribution adaptation.
gene_pred = pd.read_parquet(PROJECT / "predictions/SPL_external_gold_predictions.parquet")
gene_pred = gene_pred.query(
    "analysis == 'controlled' and model_id in ['M01','M03','M04','M06']"
).copy()
pca_for_reference = predictions[predictions.model_id.isin(["P01", "P02", "P05", "P06"])].copy()
reference = pd.concat([
    gene_pred[["cell_id", "donor", "model_id", "y", "p_gdT"]],
    pca_for_reference[["cell_id", "donor", "model_id", "y", "p_gdT"]],
], ignore_index=True)
reference_rows = []
for b, a in (("P01", "M01"), ("P02", "M03"),
             ("P05", "M04"), ("P06", "M06")):
    reference_rows.append(donor_bootstrap_delta(reference, b, a, "auprc"))
    reference_rows.append(donor_bootstrap_delta(reference, b, a, "auroc"))
reference_contrasts = pd.DataFrame(reference_rows)
reference_contrasts.to_csv(PROJECT / "results/tables/PCA_LR_minus_direct_gene_LR.csv", index=False)
display(reference_contrasts)
'''),
        code(r'''
# Three-way representation comparison requested for the manuscript:
# H1 direct-gene LR versus legacy H2 target-gold-HVG LR versus strict joint-PCA LR.
# The paired-only and paired+TRD label regimes are kept separate.
all_gene_pred = pd.read_parquet(PROJECT / "predictions/SPL_external_gold_predictions.parquet")
all_gene_pred = all_gene_pred.query(
    "analysis == 'controlled' and model_id in ['M01','M03','M07','M09']"
)[["cell_id", "donor", "model_id", "y", "p_gdT"]].copy()
joint_pred = predictions[predictions.model_id.isin(["P03", "P04"])][
    ["cell_id", "donor", "model_id", "y", "p_gdT"]
].copy()
three_way = pd.concat([all_gene_pred, joint_pred], ignore_index=True)
model_info = pd.DataFrame([
    {"model_id": "M01", "label_regime_short": "paired_only", "representation_method": "H1 Song-only HVG direct-gene LR", "target_access": "none"},
    {"model_id": "M07", "label_regime_short": "paired_only", "representation_method": "H2 target-gold-informed HVG direct-gene LR", "target_access": "paired-gold target expression used for HVG selection"},
    {"model_id": "P03", "label_regime_short": "paired_only", "representation_method": "Song + unlabeled SPL joint-PCA LR", "target_access": "unlabeled target expression used for PCA"},
    {"model_id": "M03", "label_regime_short": "paired_plus_TRD_weighted", "representation_method": "H1 Song-only HVG direct-gene LR", "target_access": "none"},
    {"model_id": "M09", "label_regime_short": "paired_plus_TRD_weighted", "representation_method": "H2 target-gold-informed HVG direct-gene LR", "target_access": "paired-gold target expression used for HVG selection"},
    {"model_id": "P04", "label_regime_short": "paired_plus_TRD_weighted", "representation_method": "Song + unlabeled SPL joint-PCA LR", "target_access": "unlabeled target expression used for PCA"},
])

comparison_rows = []
for model_id, sub in three_way.groupby("model_id"):
    comparison_rows.append({
        "model_id": model_id,
        "n": len(sub), "n_positive": int(sub.y.sum()),
        "prevalence": float(sub.y.mean()),
        "auprc": float(average_precision_score(sub.y, sub.p_gdT)),
        "auroc": float(roc_auc_score(sub.y, sub.p_gdT)),
    })
three_way_metrics = model_info.merge(pd.DataFrame(comparison_rows), on="model_id")
three_way_metrics.to_csv(PROJECT / "results/tables/SPL_H1_H2_jointPCA_AP_comparison.csv", index=False)

contrast_rows = []
for b, a in (("M07", "M01"), ("P03", "M01"), ("P03", "M07"),
             ("M09", "M03"), ("P04", "M03"), ("P04", "M09")):
    contrast_rows.append(donor_bootstrap_delta(three_way, b, a, "auprc"))
three_way_contrasts = pd.DataFrame(contrast_rows)
three_way_contrasts.to_csv(
    PROJECT / "results/tables/SPL_H1_H2_jointPCA_AUPRC_contrasts.csv", index=False
)
display(three_way_metrics.sort_values(["label_regime_short", "auprc"], ascending=[True, False]))
display(three_way_contrasts)

fig, ax = plt.subplots(figsize=(10, 5))
plot_table = three_way_metrics.copy()
sns.barplot(data=plot_table, x="label_regime_short", y="auprc",
            hue="representation_method", ax=ax)
ax.axhline(y=float(y.mean()), color="black", ls="--", lw=1, label="prevalence baseline")
ax.set(xlabel="Training positive-label regime", ylabel="SPL gold AUPRC",
       title="H1 gene-LR vs H2 target-HVG LR vs joint-PCA LR")
ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
sns.despine()
fig.savefig(PROJECT / "results/figures/Fig_H1_H2_jointPCA_AUPRC.pdf", bbox_inches="tight")
plt.show()
'''),
        code(r'''
# Source-frozen operating points. These thresholds were selected exclusively
# from nested Song OOF predictions in 07b; SPL labels are used only to measure
# transfer performance, never to choose a threshold.
operating_rows = []
for fit in fits.itertuples(index=False):
    sub = predictions[predictions.model_id.eq(fit.model_id)]
    for operating_point, threshold in (
        ("Song_OOF_maxF1", fit.threshold),
        ("Song_OOF_recall80", fit.threshold_recall80_source),
        ("Song_OOF_recall90", fit.threshold_recall90_source),
    ):
        operating_rows.append(
            metrics_at_threshold(sub.y, sub.p_gdT, float(threshold))
            | {"model_id": fit.model_id, "gene_regime": fit.gene_regime,
               "representation": fit.representation,
               "operating_point": operating_point}
        )
operating = pd.DataFrame(operating_rows)
operating.to_csv(
    PROJECT / "results/tables/SPL_pca_lr_source_frozen_operating_points.csv", index=False
)
display(operating[["model_id", "gene_regime", "operating_point", "threshold",
                   "precision", "recall", "specificity", "false_positive_rate"]])

# External calibration is audited but not corrected here. Fitting a diagnostic
# line to SPL labels does not alter any probability or threshold.
from sklearn.linear_model import LogisticRegression as SkLogisticRegression

def external_calibration_diagnostic(y_true, probability, n_bins=10):
    y_true = np.asarray(y_true, dtype=int)
    probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    logit = np.log(probability / (1 - probability)).reshape(-1, 1)
    diagnostic = SkLogisticRegression(C=1e6, solver="lbfgs", max_iter=4000)
    diagnostic.fit(logit, y_true)
    frame = pd.DataFrame({"y": y_true, "p": probability})
    frame["bin"] = pd.qcut(frame.p, q=n_bins, duplicates="drop")
    bins = frame.groupby("bin", observed=True).agg(n=("y", "size"), observed=("y", "mean"), predicted=("p", "mean"))
    ece = float(np.average(np.abs(bins.observed - bins.predicted), weights=bins.n))
    return {
        "calibration_intercept": float(diagnostic.intercept_[0]),
        "calibration_slope": float(diagnostic.coef_[0, 0]),
        "ece_10_quantile_bins": ece,
        "brier": float(brier_score_loss(y_true, probability)),
    }

calibration_rows = []
for model_id, sub in predictions.groupby("model_id"):
    calibration_rows.append(
        {"model_id": model_id} | external_calibration_diagnostic(sub.y, sub.p_gdT)
    )
calibration = pd.DataFrame(calibration_rows).merge(
    fits[["model_id", "representation", "gene_regime", "label_regime"]], on="model_id"
)
calibration.to_csv(PROJECT / "results/tables/SPL_pca_lr_calibration_diagnostics.csv", index=False)
display(calibration)
'''),
        code(r'''
palette = {"P01": "#377EB8", "P02": "#4DAF4A", "P03": "#984EA3", "P04": "#E41A1C",
           "P05": "#377EB8", "P06": "#4DAF4A", "P07": "#984EA3", "P08": "#E41A1C"}
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)
for ax, (gene_regime, title) in zip(
    axes,
    [("G1_TCR_free", "TCR-free primary/sensitivity"),
     ("G2_TCR_inclusive", "TCR-inclusive positive control")],
):
    subset = predictions[predictions.gene_regime.eq(gene_regime)]
    for model_id, sub in subset.groupby("model_id"):
        precision, recall, _ = precision_recall_curve(sub.y, sub.p_gdT)
        ap = average_precision_score(sub.y, sub.p_gdT)
        ax.plot(recall, precision, lw=2.2, color=palette[model_id],
                label=f"{model_id} (AP={ap:.3f})")
    ax.axhline(float(y.mean()), color="black", ls="--", lw=1, label="prevalence")
    ax.set(xlabel="Recall", ylabel="Precision", title=title)
    ax.legend(frameon=False)
sns.despine()
fig.suptitle("SPL gold: inductive versus transductive PCA–LR")
fig.savefig(PROJECT / "results/figures/Fig_PCA_LR_inductive_transductive_PR.pdf", bbox_inches="tight")
plt.show()
'''),
    ],
)


NOTEBOOKS["08b_15organ_frozen_model_silver_evaluation.ipynb"] = notebook(
    "08b — Frozen-model 15-organ silver-label evaluation",
    "Apply the primary frozen model across organs. Published gamma-delta annotations are silver labels, never receptor-confirmed gold labels.",
    [
        md('''
## Required input contract

The registered He/Yuanhua atlas consists of `*_gene_expression.h5ad` files under
`/data/users/kaelynwen/.../15organs`. The object must contain the published
`Cell_type_in_each_tissue` annotation. This notebook never refits the LR. Because
the atlas may not expose biological donor IDs, `organ` must not be described as a
donor replicate.
'''),
        code(r'''
organs_root = first_existing(PATHS["organs15_root_candidates"], "dir")
organ_files = sorted(organs_root.glob("*_gene_expression.h5ad"))
if len(organ_files) != 15:
    warnings.warn(f"Expected 15 expression files, found {len(organ_files)} in {organs_root}")
if not organ_files:
    raise FileNotFoundError(f"No *_gene_expression.h5ad files in {organs_root}")

parts = []
for path in organ_files:
    organ = path.name.removesuffix("_gene_expression.h5ad")
    x = ad.read_h5ad(path)
    x.var_names_make_unique()
    label_col = find_obs_column(x.obs, ["Cell_type_in_each_tissue", "cell_type", "celltype", "annotation"])
    donor_col = find_obs_column(x.obs, ["donor", "donor_id", "patient"], required=False)
    x.obs["organ"] = organ
    x.obs["donor"] = x.obs[donor_col].astype(str) if donor_col else "donor_not_available"
    x.obs["published_annotation"] = x.obs[label_col].astype(str)
    x.obs["dataset"] = "He15_annotation_only"
    x.obs_names = [f"He15|{organ}|{i}" for i in x.obs_names.astype(str)]
    parts.append(x)
atlas = ad.concat(parts, join="inner", merge="same", index_unique=None)
print("15-organ atlas:", atlas.shape, "organs:", atlas.obs.organ.nunique())
'''),
        code(r'''
bundle = load_model_bundle(PROJECT / "models/controlled_M01.joblib")
common = [g for g in bundle["genes"] if g in atlas.var_names]
coverage = len(common) / len(bundle["genes"])
if coverage < 0.95:
    raise ValueError(f"Only {coverage:.1%} of frozen model genes are present; do not impute silently.")
p, missing_genes = predict_lr_zero_fill(bundle["model"], atlas, bundle["genes"])
pd.Series(missing_genes, name="zero_filled_gene").to_csv(
    PROJECT / "results/tables/15organ_zero_filled_model_genes.csv", index=False
)
fit = pd.read_csv(PROJECT / "results/tables/controlled_fit_registry.csv").set_index("model_id").loc["M01"]
out = atlas.obs[["donor", "organ", "published_annotation"]].copy()
out["p_gdT"] = p
out["predicted_gdT"] = out["p_gdT"].ge(float(fit.threshold))
out["label_evidence_tier"] = "annotation_only_silver"
labels_upper = out["published_annotation"].str.upper()
out["silver_gdT"] = labels_upper.str.contains("GD CELL", regex=False)
out["silver_Tlineage"] = labels_upper.str.contains("T CELL", regex=False) & ~labels_upper.str.contains("NK", regex=False)
out.to_parquet(PROJECT / "predictions/15organ_M01_silver_predictions.parquet")
'''),
        code(r'''
summary = out.groupby(["organ", "published_annotation"], observed=True).agg(
    n=("p_gdT", "size"), mean_p_gdT=("p_gdT", "mean"),
    fraction_predicted_gdT=("predicted_gdT", "mean"), n_donors=("donor", "nunique")
).reset_index()
summary.to_csv(PROJECT / "results/tables/15organ_annotation_prediction_summary.csv", index=False)
display(summary.sort_values("mean_p_gdT", ascending=False).head(50))

silver_metrics = []
for organ, sub in out[out["silver_Tlineage"] | out["silver_gdT"]].groupby("organ"):
    if sub["silver_gdT"].nunique() < 2:
        continue
    silver_metrics.append({"organ": organ, "n": len(sub), "n_silver_gdT": int(sub.silver_gdT.sum()),
                           "prevalence": float(sub.silver_gdT.mean()),
                           "auprc_silver": average_precision_score(sub.silver_gdT, sub.p_gdT),
                           "auroc_silver": roc_auc_score(sub.silver_gdT, sub.p_gdT),
                           "evidence_tier": "published_annotation_only"})
silver_metrics = pd.DataFrame(silver_metrics)
silver_metrics.to_csv(PROJECT / "results/tables/15organ_silver_metrics_by_organ.csv", index=False)
display(silver_metrics)
'''),
    ],
)


NOTEBOOKS["09_query_none_orphan_and_mixed_cells.ipynb"] = notebook(
    "09 — Frozen-model query analysis",
    "Score SPL None, orphan, and mixed cells without relabeling them or including them in gold PR curves.",
    [
        code(r'''
spl = ad.read_h5ad(PROJECT / "manifests/CrossAtlas_D2_SPL_cohorts.h5ad")
query_names = ["query_none_Tlineage", "query_orphan_TRG", "candidate_orphan_TRD_nonoverlap",
               "audit_mixed_ab_gd", "query_other_receptor"]
query = spl.obs["immutable_cohort"].astype(str).isin(query_names)
query_ids = spl.obs_names[query]

fits = pd.read_csv(PROJECT / "results/tables/controlled_fit_registry.csv").set_index("model_id")
rows = []
for model_id in ("M01", "M03"):
    bundle = load_model_bundle(PROJECT / f"models/controlled_{model_id}.joblib")
    p = predict_lr(bundle["model"], spl, bundle["genes"], query_ids)
    frame = spl.obs.loc[query_ids, ["dataset", "donor", "immutable_cohort", "receptor_group",
                                    "n_TRA", "n_TRB", "n_TRG", "n_TRD"]].copy()
    frame["cell_id"] = frame.index.astype(str)
    frame["model_id"] = model_id
    frame["p_gdT"] = p.loc[frame.index]
    frame["above_frozen_threshold"] = frame["p_gdT"].ge(float(fits.loc[model_id, "threshold"]))
    rows.append(frame.reset_index(drop=True))
query_predictions = pd.concat(rows, ignore_index=True)
query_predictions.to_parquet(PROJECT / "predictions/SPL_query_predictions.parquet", index=False)
'''),
        code(r'''
summary = query_predictions.groupby(["model_id", "donor", "immutable_cohort"], observed=True).agg(
    n=("p_gdT", "size"), mean_p_gdT=("p_gdT", "mean"), median_p_gdT=("p_gdT", "median"),
    fraction_above_threshold=("above_frozen_threshold", "mean")
).reset_index()
summary.to_csv(PROJECT / "results/tables/SPL_query_prediction_summary.csv", index=False)
display(summary)
'''),
    ],
)


NOTEBOOKS["10_coefficients_and_biological_modules.ipynb"] = notebook(
    "10 — Coefficients and biological modules",
    "Interpret LR coefficients and compare T-lineage, cytotoxic, NK-like, naive, and contamination programs without treating coefficients as causal effects.",
    [
        code(r'''
modules = {
    "T_lineage": ["CD3D", "CD3E", "CD247", "LCK", "PTPRC"],
    "cytotoxic": ["NKG7", "GNLY", "PRF1", "GZMB", "GZMK", "CTSW"],
    "NK_like": ["KLRD1", "FCGR3A", "TYROBP", "TRAC"],
    "naive_memory": ["CCR7", "IL7R", "LTB", "MAL", "TCF7", "LEF1"],
    "alpha_beta": ["TRAC", "TRBC1", "TRBC2", "CD4", "CD8A", "CD8B"],
    "gamma_delta": ["TRDC", "TRGC1", "TRGC2", "TRDV1", "TRDV2", "TRDV3"],
    "B_myeloid_nonT": list(NON_T_MARKERS),
}
'''),
        code(r'''
coef_parts = []
for analysis in ("controlled", "optimized"):
    for model_id in pd.read_csv(PROJECT / "config/model_registry.csv")["model_id"]:
        bundle = load_model_bundle(PROJECT / f"models/{analysis}_{model_id}.joblib")
        tab = coefficient_table(bundle["model"], bundle["genes"])
        tab["model_id"] = model_id
        tab["analysis"] = analysis
        tab["is_TCR_gene"] = tab["gene"].map(is_tcr_gene)
        coef_parts.append(tab)
coefs = pd.concat(coef_parts, ignore_index=True)
coefs.to_csv(PROJECT / "results/tables/LR_coefficients_all_models.csv", index=False)
display(coefs.query("analysis == 'controlled' and model_id in ['M01','M03']").groupby("model_id").head(20))
'''),
        code(r'''
for dataset_name in ("Song_D1", "CrossAtlas_D2_SPL"):
    obj = ad.read_h5ad(PROJECT / f"manifests/{dataset_name}_cohorts.h5ad")
    log = make_log_normalized(obj)
    for module, genes in modules.items():
        present = [g for g in genes if g in log.var_names]
        if present:
            sc.tl.score_genes(log, present, score_name=f"score_{module}", random_state=GRID["seed"])
    cols = ["dataset", "donor", "immutable_cohort"] + [c for c in log.obs if c.startswith("score_")]
    summary = log.obs[cols].groupby(["donor", "immutable_cohort"], observed=True).agg(["count", "mean", "median"])
    summary.to_csv(PROJECT / f"results/tables/{dataset_name}_biological_module_summary.csv")
'''),
    ],
)


NOTEBOOKS["11_publication_figures_and_tables.ipynb"] = notebook(
    "11 — Publication figures and locked result tables",
    "Generate the confirmatory PR figure, model comparison heatmap, donor panels, and a machine-readable claims table.",
    [
        code(r'''
metrics = pd.read_csv(PROJECT / "results/tables/SPL_external_gold_metrics.csv")
pred = pd.read_parquet(PROJECT / "predictions/SPL_external_gold_predictions.parquet")
registry = pd.read_csv(PROJECT / "config/model_registry.csv")
overall = metrics.query("scope == 'all_donors'").merge(registry, on="model_id")
display(overall.sort_values(["analysis", "auprc"], ascending=[True, False]))
'''),
        code(r'''
fig, ax = plt.subplots(figsize=(7, 6))
for model_id, color in (("M01", "#3B6FB6"), ("M03", "#D84A3A")):
    sub = pred.query("analysis == 'controlled' and model_id == @model_id")
    precision, recall, _ = precision_recall_curve(sub.y, sub.p_gdT)
    ap = average_precision_score(sub.y, sub.p_gdT)
    ax.plot(recall, precision, lw=2.5, color=color, label=f"{model_id} (AP={ap:.3f})")
ax.set(xlabel="Recall", ylabel="Precision", title="External SPL receptor-confirmed gold cells")
ax.legend(frameon=False)
sns.despine()
fig.savefig(PROJECT / "results/figures/Fig_primary_SPL_PR.pdf", bbox_inches="tight")
plt.show()
'''),
        code(r'''
heat = overall.query("analysis == 'controlled'").pivot(index="label_regime", columns=["hvg_regime", "gene_regime"], values="auprc")
plt.figure(figsize=(10, 4))
sns.heatmap(heat, annot=True, fmt=".3f", cmap="viridis")
plt.title("SPL gold AUPRC: controlled LR configurations")
plt.tight_layout()
plt.savefig(PROJECT / "results/figures/Fig_12model_AUPRC_heatmap.pdf", bbox_inches="tight")
plt.show()
'''),
        code(r'''
claims = pd.DataFrame([
    {"claim_id": "primary", "contrast": "M03 versus M01", "test_population": "SPL gold_B paired gdT/abT",
     "eligible_wording": "external receptor-confirmed evaluation", "confirmatory": True},
    {"claim_id": "H2", "contrast": "target-gold-informed HVG sensitivity", "test_population": "SPL",
     "eligible_wording": "secondary target-gold-informed feature sensitivity; not transductive representation or independent validation", "confirmatory": False},
    {"claim_id": "PCA_LR", "contrast": "P03-P01 and P04-P02", "test_population": "SPL paired receptor gold",
     "eligible_wording": "transductive target-expression adaptation versus matched inductive PCA-LR", "confirmatory": False},
    {"claim_id": "15organ", "contrast": "frozen M01 across organs", "test_population": "annotation-only atlas",
     "eligible_wording": "silver-label cross-organ concordance", "confirmatory": False},
    {"claim_id": "BMA", "contrast": "receptor audit", "test_population": "Cross-atlas BMA",
     "eligible_wording": "descriptive audit only; excluded from core model", "confirmatory": False},
])
claims.to_csv(PROJECT / "results/tables/allowed_claims.csv", index=False)
display(claims)
'''),
    ],
)


# Override the legacy zero-fill 08b specification with the registered portable
# M01/M03 workflow.  Keeping the specification in a separate module makes its
# feature/label/threshold contract straightforward to audit.
from portable_08b_notebook import build_portable_08b

NOTEBOOKS["08b_15organ_frozen_model_silver_evaluation.ipynb"] = build_portable_08b(
    notebook, md, code
)


import argparse

parser = argparse.ArgumentParser(
    description="Generate all project notebooks or only explicitly named notebooks."
)
parser.add_argument(
    "notebooks",
    nargs="*",
    help="Exact notebook filenames to regenerate; omit to regenerate all notebooks.",
)
args = parser.parse_args()
selected = args.notebooks or list(NOTEBOOKS)
unknown = sorted(set(selected) - set(NOTEBOOKS))
if unknown:
    parser.error(f"unknown notebook name(s): {', '.join(unknown)}")

for name in selected:
    nb = NOTEBOOKS[name]
    nbf.write(nb, NB_DIR / name)
    print("wrote", NB_DIR / name)

# Keep the Obsidian methods guide synchronized with generated notebook sources.
from build_obsidian_model_guide import write_guide
print("wrote", write_guide())
