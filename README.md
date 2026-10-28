# gdTProfiler

Private manuscript-stage research project for evidence-aware identification of human γδ T cells, receptor-defined subtype modelling, and transcriptional-state analysis. This is a curated source-and-artifact draft, not a published software release or an installable Python package. No public archive DOI or software licence has been assigned.

## Model scope under review

Kaelyn's requested hierarchy is M05 as the primary model, TVAE0 as a complementary model, and SV02 as an exploratory model (the current manuscript's main Figure 3). The subtype analysis uses the reproduced 544-cell scVI + LR workflow; state analysis uses frozen K=8 cNMF. CE and hard-gated models do not replace M05. M02 is not a deployment model; its existing comparison rows remain because they explain the TCR-free benchmark. Historical results have not been renamed or overwritten.

| Model/artifact | Intended role | Verified definition |
|---|---|---|
| `optimized_M05` | Proposed default for exploratory candidate discovery | TCR-inclusive LR, C=1, 2,009 genes; Song paired γδ plus orphan-TRD supervision |
| `controlled_M05` | Reproduction of existing manuscript comparisons | Same registered feature panel, C=10; preserved historical thresholds |
| `TVAE(0)` | Complementary, original TVAE0 | 128 hidden units, 32 latent dimensions; exact `TVAE_lite_LOGO_quick_screen` freeze |
| `SV02` | Exploratory TCR-free representation, main Figure 3 | 2,000 genes, scVI 20 latent dimensions, 2×128 hidden units, LR C=0.01 |
| `14_TCRfree_Vd1_Vd2_subtype_scVI_LR` | Separate subtype stage | Verified 544-cell source: 9 donors, 292 Vδ1 and 252 Vδ2 cells; three registered seeds |
| `primary_K8` | Frozen state programme scores | Eight retained programmes; no claim that programme labels are independently validated functional identities |

The proposed optimized_M05 default is a practical candidate-discovery choice, not evidence of statistically established superiority over controlled_M05. Architecture, feature handling, fitted parameters, and decision threshold are different parts of a model contract. A new threshold changes calls, not the ranking performance.

**TVAE0 is not optimized TVAE.** Its registered original differential Adam configuration omitted some trainable parameters from optimizer updates. This is preserved for faithful reproduction and recorded in `TVAE0_optimizer_parameter_coverage.csv`; it must not be described as the corrected optimizer. Existing spatial results from optimized TVAE cannot be relabelled TVAE0. The TVAE0 historical threshold (0.9923211932182312) is separate from the proposed M05 candidate threshold.

## Threshold recommendation and evidence

For exploratory **optimized_M05** discovery, the proposal is:

- Score ≥0.5: primary candidate set.
- 0.2 ≤ score <0.5: supplementary candidate set, reviewed separately.
- Preserve continuous scores, original annotations, receptor-evidence quality/conflicts, and every T/NK gate state (pass, NK, abstain). The gate does not veto candidate retention.
- Neither a model score nor absence of detected TCR proves cell identity. Paired αβ or mixed-receptor conflicts require separate review.

These thresholds have already been inspected using SPL/PANGEA/15-organ and are **exploratory**, not independently validated optimal thresholds. The original optimized_M05 0.9394744727488467 threshold and controlled_M05 0.9529421705908118 threshold remain historical reproduction metadata, not the only candidate entry rule. Thresholds cannot be transferred to TVAE0, SV02 or the subtype model.

Matched SPL reference: 5,046 cells, comprising 378 receptor-supported γδ T and 4,668 αβ T cells. This is a γδ-versus-αβ reference, not an independently verified NK specificity experiment. The source receptor records also have quality-field limitations.

| optimized_M05 rule | γδ reference detected | αβ reference called positive | Recall | Precision within this reference |
|---|---:|---:|---:|---:|
| Historical frozen threshold | 227/378 | 1 | 60.05% | 99.56% |
| ≥0.5 | 296/378 | 19 | 78.31% | 93.97% |
| ≥0.2 | 323/378 | 35 | 85.45% | 90.22% |

Lowering 0.5 to 0.2 adds 27 receptor-supported γδ cells and 16 αβ-reference calls. It does not establish the identity of atlas NK-like candidates. PANGEA and 15-organ author labels are not treated as independent receptor truth; 15-organ's available receptor evidence supports αβ calls, not γδ-negative truth. Previously examined external cohorts now serve diagnosis; a fresh held-out cohort is needed for confirmation after rule revision.

## Existing results and limits

| Evaluation | Saved result | Interpretation |
|---|---|---|
| SPL, controlled_M05 | AP 0.942523; AUROC 0.992440 | Historical main lineage comparator |
| SPL, optimized_M05 | AP 0.945143; AUROC 0.992968 | Small numerical difference; no superiority claim |
| SPL, TVAE0 | AP 0.879763; AUROC 0.985356 | Complementary model, not overall better than M05 |
| SPL, SV02 | AP 0.641393; AUROC 0.953283 | Higher AP than feature-matched direct-expression LR (0.423346); calibration remains limited |
| GSE149356 adult subtype, primary seed | Donor-mean AUROC ≈0.962 | Two adult donors; not an atlas-wide subtype guarantee |
| K=8 state programmes | Frozen scoring assets retained | K=10 enrichment cannot substitute for missing K=8 enrichment |

AP means average precision; it is threshold independent and depends on the evaluation population. Current manuscript main Figure 3 is the SV02 exploration, although historical asset filenames do not necessarily equal manuscript display numbers. These summaries are not a declaration that every manuscript analysis has been independently reproduced.

## Contents and authoritative sources

- `build_notebooks.py`: authoritative generator of the historical LR notebook sequence (00–09 and further registered comparisons). It includes comparison models for provenance; their presence does not make them deployment defaults.
- `src/gdt_lr_utils.py`, `config/`, `features/`: original LR preprocessing, model registry and fixed features.
- `build_TVAE_lite_LOGO_quick_screen.py`: authoritative source for `notebooks/TVAE_lite_LOGO_quick_screen.ipynb` and the included original TVAE0 checkpoint.
- `build_sv02_sv05_scvi_lr_notebook.py`: authoritative source for `notebooks/12_sv02_sv05_strict_inductive_scvi_lr.ipynb`; historical SV05 comparison is preserved in source, only SV02 weights are packaged here.
- `build_14_TCRfree_Vd1_Vd2_subtype_scVI_LR.py`: copied from the verified **544_cell_source_archive**, not a later overwritten development generator. Authoritative for `notebooks/14_TCRfree_Vd1_Vd2_subtype_scVI_LR.ipynb`.
- `build_16_Gray_adult_state_program_discovery_freeze.py` and `build_19b_cNMF_valdiation.py`: state-discovery/validation sources; the existing `valdiation` spelling is intentionally retained.
- `build_21_T_NK_lineage_gate_and_frozen_M05_deployment.py` and `build_21_gate_LRM05_evaluation_15organs.py`: gate and atlas audit provenance. Historical hard-gate code does not implement a new default candidate policy.
- `models/`, `checkpoints/`, `results/models/`: selected frozen models and state scoring assets. Training expression objects (`adata.h5ad`) are intentionally excluded.
- `results/tables/`: selected aggregate results, original run contracts and freeze records.
- `output_manifest.csv`: exact original source, SHA-256, local destination and intended Woody destination for every copied source/artifact. Original completion manifests describe full research runs, not a promise that every referenced intermediate is shipped here.

## Reproduction status

Original generators have been preserved byte-for-byte; they still contain research-workspace paths and may generate several historical comparisons. Set project paths deliberately before execution. Do not blindly run all generators in a directory containing completed results. They are scientific provenance sources, not a new one-command deployment interface.

The standard LR dependencies are in the original `requirements.txt`; it is **not an environment lock** and does not cover every TVAE/scVI environment. Recorded SV02 software versions are in `results/tables/SV02_SV05_scVI_LR/software_versions.json`. The subtype freeze/completion files record its separate environment and completed rerun. Original M05 deserialization can depend on the scikit-learn version used to save it; use trusted model files only.

A source-only notebook can be regenerated, without launching training, for example:

```sh
cd /Users/kaelynwen/Desktop/FYP_YHH/gdTProfiler
python build_TVAE_lite_LOGO_quick_screen.py
```

Its authoritative source is `build_TVAE_lite_LOGO_quick_screen.py`; the generated runnable artifact is `notebooks/TVAE_lite_LOGO_quick_screen.ipynb`. Intended Woody counterparts are `/data/users/kaelynwen/gdTProfiler/build_TVAE_lite_LOGO_quick_screen.py` and `/data/users/kaelynwen/gdTProfiler/notebooks/TVAE_lite_LOGO_quick_screen.ipynb`. Generation does not perform training or validate the environment. Raw data, cell-level evidence/prediction tables, donor split manifests, and intermediate checkpoints required by complete evaluations must be supplied from the original project. They are not silently reconstructed from atlas annotations. Frozen scVI loading without the excluded training AnnData also requires the matching feature/batch registration and has not yet been packaged as a standalone inference command.

The verified 544-cell rerun used incremental donor×seed and final-seed checkpoints with signatures and hashes. Missing checkpoints in this curated repository must not be treated as completed local computations. A complete clean-environment reproduction and portable inference smoke test remain release tasks; no new training or full scientific rerun was performed during repository curation.

## Data and release status

Original data remain with their original providers; they are not redistributed in this private source bundle. Key resources include Song Zenodo 7989561, SPL E-MTAB-11536, adult GSE149356, 15-organ GSE159929, PANGEA, and Gray's adult CELLxGENE collection `ec691f5f-0aac-433c-8f78-e7f4b85a05e0`. Source annotation, RNA-derived receptor signals, protein measurements, and cell-resolved receptor evidence are distinct evidence levels.

Before public release: finalize the exact main-model contract with the authors, freeze the candidate policy, add a portable tested inference interface and environment locks, reconcile all manuscript source-data mappings, confirm redistribution rights and a software licence, and archive an immutable release. Author contributions, funding, conflicts, contacts/ORCIDs and institutional ethics determination remain author-confirmation items. A private GitHub repository can later be made public; a second repository is not required.

Organization was informed by StatBiomed/SpatialDM (overview, installation/reproduction guidance, examples and documentation); no SpatialDM code or licence was copied.

Local root: `/Users/kaelynwen/Desktop/FYP_YHH/gdTProfiler`.
Intended Woody root: `/data/users/kaelynwen/gdTProfiler` (not transferred).
All relative files map unchanged between these roots; individual paths are in `output_manifest.csv`. Manuscript edits remain local, and this bundle contains no Overleaf project or manuscript ZIP. Private GitHub repository: https://github.com/KaelynWenHKU/gdTProfiler. Created on 2026-10-28 under KaelynWenHKU; this is a private research draft, not a public release.
