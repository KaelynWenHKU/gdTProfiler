#!/usr/bin/env python3
"""Generate the authoritative Gray adult state-program discovery notebook.

This module is deliberately import-safe: importing it exposes tested contracts and
pure helpers but neither reads project data nor launches cNMF.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import textwrap

import nbformat as nbf
import numpy as np
import pandas as pd
from scipy import sparse


ANALYSIS_NAME = "16_Gray_adult_state_program_discovery_freeze"
TOTAL_GRAY_CELLS = 7417
TOTAL_GRAY_DONORS = 24
PRIMARY_ASSAY = "10x 5' v2"
PRIMARY_CELLS = 6466
PRIMARY_DONORS = 22
SENSITIVITY_ASSAY = "10x 3' v3"
SENSITIVITY_CELLS = 951
SENSITIVITY_DONORS = {"D496", "D503"}
AGE_BOUNDARY_DONOR = "694B"
EXPECTED_TISSUES = {
    "spleen", "jejunum", "lung", "mesenteric lymph node", "thoracic lymph node"
}
EXPECTED_INPUT_SHA256 = "b128900e39a39c659c0a8c63d5037fca5ee4e627c114bb1564921916ed689b6d"
REGISTERED_SEEDS = (20260901, 20260902, 20260903)
PILOT_K_GRID = (4, 5, 6, 7, 8, 10, 12, 15)


def is_receptor_locus_gene(symbol: str) -> bool:
    """Return True only for TRA/TRB/TRG/TRD-locus symbols."""
    gene = str(symbol).upper()
    receptor_segment_prefixes = (
        "TRAV", "TRAJ", "TRAC",
        "TRBV", "TRBJ", "TRBC", "TRBD",
        "TRGV", "TRGJ", "TRGC",
        "TRDV", "TRDD", "TRDJ", "TRDC",
    )
    return gene.startswith(receptor_segment_prefixes)


def assert_integer_count_matrix(matrix, tolerance: float = 1e-6) -> None:
    """Fail if a dense or sparse count matrix is invalid or not integer-like."""
    values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix).ravel()
    if values.size == 0:
        raise ValueError("Counts matrix is empty")
    if np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Counts must be finite and nonnegative")
    if np.mean(np.isclose(values, np.rint(values))) < 1 - tolerance:
        raise ValueError("Gray raw matrix is not integer-like")


def deterministic_donor_balance(
    metadata: pd.DataFrame,
    donor_col: str,
    cap: int,
    seed: int,
) -> list[str]:
    """Return stable cell IDs after sampling at most ``cap`` cells per donor."""
    if donor_col not in metadata:
        raise KeyError(donor_col)
    if cap < 1 or not metadata.index.is_unique:
        raise ValueError("cap must be positive and cell IDs must be unique")
    selected: list[str] = []
    for donor, frame in metadata.groupby(donor_col, observed=True, sort=True):
        ids = np.asarray(sorted(frame.index.astype(str)))
        if len(ids) > cap:
            rng = np.random.default_rng(seed + int(hashlib.sha256(str(donor).encode()).hexdigest()[:8], 16))
            ids = np.sort(rng.choice(ids, size=cap, replace=False))
        selected.extend(ids.tolist())
    return selected


def assert_matched_arms(
    primary_cells, inclusive_cells, primary_features, inclusive_features,
    receptor_features, primary_k, inclusive_k, primary_seeds, inclusive_seeds,
    primary_normalization, inclusive_normalization,
) -> None:
    """Enforce that receptor loci are the only difference between cNMF arms."""
    if list(primary_cells) != list(inclusive_cells):
        raise AssertionError("Matched cNMF arms must use identical ordered cell IDs")
    if primary_k != inclusive_k or tuple(primary_seeds) != tuple(inclusive_seeds):
        raise AssertionError("Matched cNMF arms must use identical K and seeds")
    if primary_normalization != inclusive_normalization:
        raise AssertionError("Matched cNMF arms must use identical normalization")
    primary, inclusive, receptor = map(set, (primary_features, inclusive_features, receptor_features))
    if not primary.issubset(inclusive) or inclusive - primary != receptor:
        raise AssertionError("Only the registered receptor-locus genes may differ between arms")
    if any(is_receptor_locus_gene(gene) for gene in primary):
        raise AssertionError("Primary feature universe contains a receptor-locus gene")


def md(source: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(source).strip())


def code(source: str):
    return nbf.v4.new_code_cell(textwrap.dedent(source).strip())


def build_notebook():
    cells = [
        md(r"""
        # 16 — Gray adult state-program discovery and freeze

        **Question.** Which overlapping, receptor-locus-independent transcriptional
        programs are reproducible across adult human γδ T-cell donors and tissues?

        This is unsupervised program discovery, not a mutually exclusive state
        classifier. Author fields `classification`, `specific_phenotype`, and `vd`
        are quarantined until after program weights are frozen. The notebook is a
        staged, auditable workflow: preparation → pilot K review → full consensus →
        immutable source freeze → 3′/age/inclusive sensitivities.

        **Evidence boundary.** Gray contains scRNA expression and author annotations,
        but no per-cell V(D)J contigs; `vd` is not a sequence-defined gold label.
        """),
        code(r"""
        # 1. Imports, deterministic configuration, server-first paths, typography
        from __future__ import annotations
        import hashlib, json, os, platform, random, subprocess, sys
        from datetime import datetime, timezone
        from importlib.metadata import PackageNotFoundError, version
        from pathlib import Path

        import anndata as ad
        import matplotlib as mpl
        import matplotlib.font_manager as fm
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import scipy
        from scipy import sparse
        from scipy.optimize import nnls
        from scipy.stats import pearsonr, spearmanr
        import seaborn as sns

        ANALYSIS_NAME = "16_Gray_adult_state_program_discovery_freeze"
        REMOTE_PROJECT = Path("/data/users/kaelynwen/gdT_project/gdt_lr_12model")
        PROJECT = Path(os.environ.get("GDT_LR_PROJECT", REMOTE_PROJECT)).resolve()
        if not PROJECT.exists():
            local = Path("/Users/kaelynwen/Desktop/FYP_YHH/gdT_project/gdt_lr_12model")
            PROJECT = local if local.exists() else Path.cwd().resolve()
        if str(PROJECT).startswith("/home/kaelynwen"):
            raise RuntimeError("Project work must use /data/users/kaelynwen, not /home/kaelynwen")
        GRAY_PATH = Path(os.environ.get(
            "GRAY_ADULT_H5AD", "/data/users/kaelynwen/gdT_project/Gray data/Adult gdT cells.h5ad"
        )).resolve()
        if not GRAY_PATH.exists():
            local_gray = Path("/Users/kaelynwen/Desktop/FYP_YHH/Gray data/Adult gdT cells.h5ad")
            GRAY_PATH = local_gray if local_gray.exists() else GRAY_PATH
        TABLE_DIR = PROJECT / "results" / "tables" / ANALYSIS_NAME
        FIGURE_DIR = PROJECT / "results" / "figures" / ANALYSIS_NAME
        MODEL_DIR = PROJECT / "results" / "models" / ANALYSIS_NAME
        LOG_DIR = PROJECT / "results" / "logs" / ANALYSIS_NAME
        for directory in (TABLE_DIR, FIGURE_DIR, MODEL_DIR, LOG_DIR): directory.mkdir(parents=True, exist_ok=True)

        REGISTERED_SEEDS = (20260901, 20260902, 20260903)
        random.seed(REGISTERED_SEEDS[0]); np.random.seed(REGISTERED_SEEDS[0])
        def choose_permitted_font():
            for family in ("Arial", "Helvetica", "Liberation Sans"):
                try:
                    fm.findfont(fm.FontProperties(family=family), fallback_to_default=False); return family
                except ValueError: pass
            raise RuntimeError("Arial, Helvetica, or Liberation Sans is required")
        FIGURE_FONT = choose_permitted_font()
        mpl.rcParams.update({"font.family": FIGURE_FONT, "font.weight": "normal", "text.color": "black",
            "axes.labelcolor": "black", "axes.titlecolor": "black", "xtick.color": "black",
            "ytick.color": "black", "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight"})
        sns.set_theme(style="white", font=FIGURE_FONT, rc={"text.color":"black", "axes.labelcolor":"black",
            "axes.titlecolor":"black", "xtick.color":"black", "ytick.color":"black"})
        """),
        md(r"""
        ## Interactive execution control

        Edit only the two user parameters below and then use **Restart Kernel → Run All**.

        - `prepare`: audit inputs and prepare cNMF matrices without factorization.
        - `pilot`: run the registered K-grid pilot and create the K diagnostic PDF/CSV.
        - `full_primary`: after reviewing the pilot, set `FROZEN_K` and fit the frozen primary programs.
        - `tcr_inclusive_sensitivity`: run only the matched receptor-inclusive sensitivity after the primary freeze exists.

        Existing completed outputs are reused unless `FORCE_RECOMPUTE=True`.
        """),
        code(r"""
        # USER-EDITABLE CONTROL CELL
        EXECUTION_STAGE = "prepare"  # prepare | pilot | full_primary | tcr_inclusive_sensitivity
        FROZEN_K = None              # e.g. 8, but only after reviewing pilot diagnostics
        FORCE_RECOMPUTE = False

        # Optional terminal overrides preserve non-interactive batch execution.
        if os.environ.get("RUN_CNMF_PILOT", "0") == "1": EXECUTION_STAGE = "pilot"
        if os.environ.get("RUN_FULL_CNMF", "0") == "1": EXECUTION_STAGE = "full_primary"
        if os.environ.get("RUN_TCR_INCLUSIVE_CNMF", "0") == "1": EXECUTION_STAGE = "tcr_inclusive_sensitivity"
        if os.environ.get("GDT_STATE_FROZEN_K") is not None:
            FROZEN_K = int(os.environ["GDT_STATE_FROZEN_K"])

        VALID_EXECUTION_STAGES = {"prepare", "pilot", "full_primary", "tcr_inclusive_sensitivity"}
        if EXECUTION_STAGE not in VALID_EXECUTION_STAGES:
            raise ValueError(f"EXECUTION_STAGE must be one of {sorted(VALID_EXECUTION_STAGES)}")
        RUN_CNMF_PILOT = EXECUTION_STAGE == "pilot"
        RUN_FULL_CNMF = EXECUTION_STAGE == "full_primary"
        RUN_TCR_INCLUSIVE_CNMF = EXECUTION_STAGE == "tcr_inclusive_sensitivity"
        print({"execution_stage": EXECUTION_STAGE, "frozen_K": FROZEN_K,
            "force_recompute": FORCE_RECOMPUTE})
        """),
        code(r"""
        # 2. Pure helpers embedded for transparent Run All execution
        def sha256sum(path, chunk_size=1024*1024):
            digest = hashlib.sha256()
            with Path(path).open("rb") as handle:
                for chunk in iter(lambda: handle.read(chunk_size), b""): digest.update(chunk)
            return digest.hexdigest()
        def sha256_lines(values):
            return hashlib.sha256(("\n".join(map(str, values))+"\n").encode()).hexdigest()
        def is_receptor_locus_gene(symbol):
            gene=str(symbol).upper()
            receptor_segment_prefixes=(
                "TRAV","TRAJ","TRAC","TRBV","TRBJ","TRBC","TRBD",
                "TRGV","TRGJ","TRGC","TRDV","TRDD","TRDJ","TRDC")
            return gene.startswith(receptor_segment_prefixes)
        def assert_integer_count_matrix(matrix, tolerance=1e-6):
            values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix).ravel()
            if values.size == 0: raise ValueError("Counts matrix is empty")
            if np.any(~np.isfinite(values)) or np.any(values < 0): raise ValueError("Counts must be finite and nonnegative")
            if np.mean(np.isclose(values, np.rint(values))) < 1-tolerance: raise ValueError("Gray raw matrix is not integer-like")
        def deterministic_donor_balance(metadata, donor_col, cap, seed):
            selected=[]
            for donor, frame in metadata.groupby(donor_col, observed=True, sort=True):
                ids=np.asarray(sorted(frame.index.astype(str)))
                if len(ids)>cap:
                    donor_seed=seed+int(hashlib.sha256(str(donor).encode()).hexdigest()[:8],16)
                    ids=np.sort(np.random.default_rng(donor_seed).choice(ids,size=cap,replace=False))
                selected.extend(ids.tolist())
            return selected
        """),
        md("## Phase 0 — input integrity, exact cohort audit, and source preparation"),
        code(r"""
        # 3. Verify immutable input and exact audited Gray schema
        EXPECTED_SHA256 = "b128900e39a39c659c0a8c63d5037fca5ee4e627c114bb1564921916ed689b6d"
        if not GRAY_PATH.is_file(): raise FileNotFoundError(GRAY_PATH)
        observed_sha256 = sha256sum(GRAY_PATH)
        if observed_sha256 != EXPECTED_SHA256: raise ValueError(f"Gray SHA-256 mismatch: {observed_sha256}")
        adata = ad.read_h5ad(GRAY_PATH)
        required_obs = {"tissue","donor_id","author_cell_type","assay","donor_age","vd","classification","specific_phenotype"}
        missing = required_obs-set(adata.obs.columns)
        if missing: raise KeyError(f"Gray obs lacks: {sorted(missing)}")
        assert adata.n_obs == 7417 and adata.n_vars == 35477
        assert adata.obs["donor_id"].nunique() == 24
        assert set(adata.obs["tissue"].astype(str)) == {"spleen","jejunum","lung","mesenteric lymph node","thoracic lymph node"}
        assert set(adata.obs["author_cell_type"].astype(str)) == {"gamma-delta T cell"}
        if adata.raw is None: raise RuntimeError("Gray .raw is required; .X is not the cNMF count source")
        assert_integer_count_matrix(adata.raw.X)
        gene_symbols = pd.Index(
            adata.raw.var["feature_name"].fillna(adata.raw.var_names.to_series()).astype(str)
            if "feature_name" in adata.raw.var else adata.raw.var_names.astype(str), name="gene")
        raw_counts = adata.raw.X.tocsr() if sparse.issparse(adata.raw.X) else sparse.csr_matrix(adata.raw.X)
        raw_gene_columns = len(gene_symbols)
        duplicate_gene_columns = int(gene_symbols.duplicated(keep=False).sum())
        if gene_symbols.has_duplicates:
            # Collapse duplicated symbols deterministically by summing their raw-count columns.
            unique_symbols = pd.Index(pd.unique(gene_symbols), name="gene")
            symbol_codes = unique_symbols.get_indexer(gene_symbols)
            collapse = sparse.csr_matrix(
                (np.ones(raw_gene_columns), (np.arange(raw_gene_columns), symbol_codes)),
                shape=(raw_gene_columns, len(unique_symbols)))
            raw_counts = (raw_counts @ collapse).tocsr()
            gene_symbols = unique_symbols
        assert raw_counts.shape == (adata.n_obs, len(gene_symbols)) and gene_symbols.is_unique
        audit = pd.DataFrame({"metric":["cells","raw_gene_columns","unique_gene_symbols",
            "duplicate_gene_columns_collapsed","donors","tissues","raw_integer_like"],
            "value":[adata.n_obs,raw_gene_columns,len(gene_symbols),duplicate_gene_columns,
                adata.obs.donor_id.nunique(),adata.obs.tissue.nunique(),True]})
        audit.to_csv(TABLE_DIR/"Gray_pre_model_audit.csv",index=False)
        display(audit)
        """),
        code(r"""
        # 4. Freeze exact 5′ discovery and 3′ sensitivity cohorts
        primary_mask = adata.obs["assay"].astype(str).eq("10x 5' v2")
        sensitivity_mask = adata.obs["assay"].astype(str).eq("10x 3' v3")
        assert int(primary_mask.sum()) == 6466
        assert adata.obs.loc[primary_mask,"donor_id"].nunique() == 22
        assert int(sensitivity_mask.sum()) == 951
        assert set(adata.obs.loc[sensitivity_mask,"donor_id"].astype(str)) == {"D496","D503"}
        assert "694B" in set(adata.obs.loc[primary_mask,"donor_id"].astype(str))
        cohort_manifest = adata.obs[["donor_id","tissue","assay","donor_age"]].copy()
        cohort_manifest["cohort_role"] = np.where(primary_mask,"primary_5prime",np.where(sensitivity_mask,"sensitivity_3prime","excluded"))
        cohort_manifest.to_csv(TABLE_DIR/"Gray_discovery_cohort_manifest.csv.gz",compression="gzip")
        primary = ad.AnnData(X=raw_counts[np.asarray(primary_mask)].copy(), obs=adata.obs.loc[primary_mask,["donor_id","tissue","assay","donor_age"]].copy(),
            var=pd.DataFrame(index=pd.Index(gene_symbols,name="gene")))
        # Author annotations are intentionally absent from `primary.obs`.
        del adata
        """),
        code(r"""
        # 5. Donor-balanced cells and registered receptor-locus-excluded feature universe
        donor_counts = primary.obs["donor_id"].astype(str).value_counts().sort_index()
        DONOR_CAP = int(np.floor(donor_counts.quantile(0.75)))
        DONOR_CAP = max(DONOR_CAP, 1)
        selected_ids = deterministic_donor_balance(primary.obs,"donor_id",DONOR_CAP,REGISTERED_SEEDS[0])
        balanced = primary[selected_ids].copy()
        balanced_counts = balanced.obs["donor_id"].value_counts()
        assert balanced_counts.max() <= DONOR_CAP and balanced_counts.size == 22
        receptor_genes = sorted(g for g in balanced.var_names if is_receptor_locus_gene(g))
        non_receptor_genes = [g for g in balanced.var_names if not is_receptor_locus_gene(g)]
        # Expressed-gene filter is label-free and donor-aware; no classification/specific_phenotype/vd is available here.
        expressed_by_donor=[]
        for donor in sorted(balanced.obs.donor_id.astype(str).unique()):
            block=balanced[balanced.obs.donor_id.astype(str).eq(donor)].X
            expressed_by_donor.append(np.asarray((block>0).sum(axis=0)).ravel() >= max(3, int(0.01*block.shape[0])))
        donor_recurrence=np.vstack(expressed_by_donor).sum(axis=0)
        eligible=set(balanced.var_names[donor_recurrence >= 3])
        primary_features=sorted(eligible.intersection(non_receptor_genes))
        inclusive_features=sorted(eligible)
        pd.Series(primary_features,name="gene").to_csv(TABLE_DIR/"primary_receptor_locus_excluded_gene_universe.csv",index=False)
        pd.Series(receptor_genes,name="gene").to_csv(TABLE_DIR/"registered_receptor_locus_genes.csv",index=False)
        balance_record={"donor_cap":DONOR_CAP,"seed":REGISTERED_SEEDS[0],"n_cells":balanced.n_obs,
            "cell_manifest_sha256":sha256_lines(balanced.obs_names),"gene_universe_sha256":sha256_lines(primary_features)}
        (TABLE_DIR/"donor_balance_record.json").write_text(json.dumps(balance_record,indent=2),encoding="utf-8")
        """),
        md("## Phase 1 — small cNMF pilot and an explicit frozen-K gate"),
        code(r"""
        # 6. Export count matrices and deterministic pilot specification; this cell does not fit cNMF
        import scipy.io
        PILOT_K_GRID=[4,5,6,7,8,10,12,15]
        PILOT_ITERATIONS=25
        FULL_ITERATIONS=200
        CONSENSUS_DENSITY_THRESHOLD=0.01
        NORMALIZATION="TPM_1000000_then_variance_scaling_without_centering"
        primary_input=balanced[:,primary_features].copy()
        scipy.io.mmwrite(MODEL_DIR/"Gray_primary_balanced_counts.mtx",primary_input.X.T)
        primary_input.write_h5ad(MODEL_DIR/"Gray_primary_balanced_counts.h5ad",compression="gzip")
        pd.Series(primary_input.obs_names,name="cell_id").to_csv(MODEL_DIR/"Gray_primary_balanced_cells.tsv",sep="\t",index=False)
        pd.Series(primary_features,name="gene").to_csv(MODEL_DIR/"Gray_primary_balanced_genes.tsv",sep="\t",index=False)
        pilot_spec={"K_grid":PILOT_K_GRID,"iterations":PILOT_ITERATIONS,"seeds":REGISTERED_SEEDS,
            "normalization":NORMALIZATION,"author_labels_used":False,"input_sha256":observed_sha256,
            "cell_manifest_sha256":sha256_lines(primary_input.obs_names),"gene_universe_sha256":sha256_lines(primary_features)}
        (TABLE_DIR/"cNMF_pilot_specification.json").write_text(json.dumps(pilot_spec,indent=2),encoding="utf-8")
        print("Preparation complete. Run the registered cNMF pilot on Woody; do not inspect author state labels.")
        """),
        code(r"""
        # 6b. Opt-in cNMF pilot implementation (disabled during notebook generation/tests)
        pilot_diagnostics_path=TABLE_DIR/"cNMF_pilot_K_diagnostics.csv"
        if RUN_CNMF_PILOT and pilot_diagnostics_path.exists() and not FORCE_RECOMPUTE:
            print(f"Reusing completed pilot diagnostics: {pilot_diagnostics_path}")
        elif RUN_CNMF_PILOT:
            from cnmf import cNMF
            from cnmf.cnmf import load_df_from_npz
            pilot=cNMF(output_dir=str(MODEL_DIR),name=f"{ANALYSIS_NAME}_pilot")
            pilot.prepare(counts_fn=str(MODEL_DIR/"Gray_primary_balanced_counts.h5ad"),
                components=np.asarray(PILOT_K_GRID),n_iter=PILOT_ITERATIONS,
                seed=REGISTERED_SEEDS[0],num_highvar_genes=2000)
            pilot.factorize(worker_i=0,total_workers=1)
            pilot.combine()
            pilot.k_selection_plot(close_fig=True)
            raw_diagnostics=load_df_from_npz(pilot.paths["k_selection_stats"]).reset_index()
            lowered={str(column).lower():column for column in raw_diagnostics.columns}
            # Prefer the explicit cNMF component-count field.  After reset_index(),
            # the synthetic ``index`` column is only the row number (0..n-1) and
            # must never be reported as K.
            k_column=next((lowered[key] for key in ("k","n_components") if key in lowered),None)
            stability_column=next((column for key,column in lowered.items() if "stability" in key or "silhouette" in key),None)
            error_column=next((column for key,column in lowered.items() if "error" in key),None)
            if None in (stability_column,error_column):
                raise RuntimeError(f"Unrecognized cNMF K-stat columns: {raw_diagnostics.columns.tolist()}")
            if k_column is None:
                if len(raw_diagnostics) != len(PILOT_K_GRID):
                    raise RuntimeError(
                        "cNMF diagnostics lack an explicit K column and cannot be aligned "
                        f"to the registered grid {PILOT_K_GRID}: {raw_diagnostics.columns.tolist()}"
                    )
                k_values=pd.Series(PILOT_K_GRID,index=raw_diagnostics.index,dtype=int)
            else:
                k_values=raw_diagnostics[k_column].astype(int)
            diagnostics=pd.DataFrame({"K":k_values,
                "reconstruction_error":raw_diagnostics[error_column].astype(float),
                "consensus_stability":raw_diagnostics[stability_column].astype(float)})
            if diagnostics["K"].tolist() != PILOT_K_GRID:
                raise RuntimeError(
                    f"Observed K values {diagnostics['K'].tolist()} do not match "
                    f"the registered pilot grid {PILOT_K_GRID}"
                )
            diagnostics.to_csv(TABLE_DIR/"cNMF_pilot_K_diagnostics.csv",index=False)
            print("Pilot complete. Review the registered stability/error table and PDF before freezing K.")
        else:
            print(f"EXECUTION_STAGE={EXECUTION_STAGE!r}: pilot is prepared but not launched.")
        """),
        code(r"""
        # 7. Pilot review contract: reconstruction error and consensus stability only
        PILOT_DIAGNOSTICS = TABLE_DIR/"cNMF_pilot_K_diagnostics.csv"
        FROZEN_K_RECORD = TABLE_DIR/"frozen_K_record.json"
        if PILOT_DIAGNOSTICS.exists():
            diagnostics=pd.read_csv(PILOT_DIAGNOSTICS)
            required={"K","reconstruction_error","consensus_stability"}
            if not required.issubset(diagnostics): raise KeyError(required-set(diagnostics))
            observed_k=diagnostics["K"].astype(int).tolist()
            legacy_row_index=list(range(len(PILOT_K_GRID)))
            if observed_k == legacy_row_index:
                # Repair diagnostics produced by the earlier extraction bug that
                # recorded reset_index() row numbers instead of component counts.
                diagnostics["K"]=PILOT_K_GRID
                diagnostics.to_csv(PILOT_DIAGNOSTICS,index=False)
                print(
                    "Repaired legacy K labels 0..7 to registered grid "
                    f"{PILOT_K_GRID}; cNMF factorization was not rerun."
                )
            elif observed_k != PILOT_K_GRID:
                raise RuntimeError(
                    f"Pilot K values {observed_k} do not match registered grid "
                    f"{PILOT_K_GRID}"
                )
            fig,ax1=plt.subplots(figsize=(4.8,3.2)); ax2=ax1.twinx()
            ax1.plot(diagnostics.K,diagnostics.reconstruction_error,"o-",color="black")
            ax2.plot(diagnostics.K,diagnostics.consensus_stability,"s--",color="black")
            ax1.set(xlabel="K",ylabel="Reconstruction error"); ax2.set_ylabel("Consensus stability")
            fig.savefig(FIGURE_DIR/"cNMF_pilot_K_diagnostics.pdf")
            display(fig)
            plt.close(fig)
        else:
            print(f"Awaiting {PILOT_DIAGNOSTICS}; full consensus remains locked.")
        """),
        md(r"""
        ### Human K-freeze checkpoint

        After reviewing the pilot diagnostics, create `frozen_K_record.json` with:
        `K`, `adjacent_sensitivity_K`, `pilot_input_sha256`, `gene_universe_sha256`,
        `cell_manifest_sha256`, `normalization`, `seeds`, `pilot_iterations`,
        `full_iterations`, `cnmf_parameters`, and UTC `timestamp`. K selection must
        not consult `classification`, `specific_phenotype`, or `vd`.
        """),
        code(r"""
        # 8. Validate the frozen-K record before any full consensus job can launch
        REQUIRED_K_FIELDS={"K","adjacent_sensitivity_K","pilot_input_sha256","gene_universe_sha256",
            "cell_manifest_sha256","normalization","seeds","pilot_iterations","full_iterations","cnmf_parameters","timestamp"}
        requested_frozen_k=FROZEN_K
        if requested_frozen_k is not None and not FROZEN_K_RECORD.exists():
            # Set this only after visually reviewing the label-free K diagnostic plot/table.
            selected_k=int(requested_frozen_k)
            if selected_k not in PILOT_K_GRID: raise ValueError(f"Frozen K must be in {PILOT_K_GRID}")
            selected_index=PILOT_K_GRID.index(selected_k)
            adjacent=[PILOT_K_GRID[i] for i in (selected_index-1,selected_index+1) if 0 <= i < len(PILOT_K_GRID)]
            frozen_record={"K":selected_k,"adjacent_sensitivity_K":adjacent,
                "pilot_input_sha256":observed_sha256,"gene_universe_sha256":sha256_lines(primary_features),
                "cell_manifest_sha256":sha256_lines(primary_input.obs_names),"normalization":NORMALIZATION,
                "seeds":list(REGISTERED_SEEDS),"pilot_iterations":PILOT_ITERATIONS,
                "full_iterations":FULL_ITERATIONS,"cnmf_parameters":pilot_spec,
                "timestamp":pd.Timestamp.now(tz="UTC").isoformat(),"selection_source":"human_reviewed_GDT_STATE_FROZEN_K"}
            FROZEN_K_RECORD.write_text(json.dumps(frozen_record,indent=2),encoding="utf-8")
        elif requested_frozen_k is not None and FROZEN_K_RECORD.exists():
            registered_k=int(json.loads(FROZEN_K_RECORD.read_text())["K"])
            if int(requested_frozen_k) != registered_k:
                raise RuntimeError(f"K is already frozen as {registered_k}; requested {requested_frozen_k}")
        frozen_k=None
        if FROZEN_K_RECORD.exists():
            frozen_k=json.loads(FROZEN_K_RECORD.read_text())
            missing=REQUIRED_K_FIELDS-set(frozen_k)
            if missing: raise RuntimeError(f"Incomplete frozen-K record: {sorted(missing)}")
            if frozen_k["pilot_input_sha256"] != observed_sha256: raise RuntimeError("Frozen-K input checksum mismatch")
            if frozen_k["gene_universe_sha256"] != sha256_lines(primary_features): raise RuntimeError("Frozen-K gene checksum mismatch")
            if frozen_k["cell_manifest_sha256"] != sha256_lines(primary_input.obs_names): raise RuntimeError("Frozen-K cell checksum mismatch")
            if frozen_k["normalization"] != NORMALIZATION or tuple(frozen_k["seeds"]) != REGISTERED_SEEDS: raise RuntimeError("Frozen-K preprocessing mismatch")
            print(f"Full consensus gate unlocked for K={frozen_k['K']}.")
        else:
            print("No frozen-K record: full consensus is intentionally not launched.")
        """),
        md("## Phase 2 — full consensus, immutable program freeze, and matched sensitivities"),
        code(r"""
        # 9. Registered full-consensus command manifest (expensive execution is opt-in on Woody)
        if RUN_FULL_CNMF and frozen_k is None: raise RuntimeError("Cannot run full cNMF before frozen_K_record.json")
        full_outputs_ready=all(path.exists() for path in (
            MODEL_DIR/"Gray_primary_frozen_spectra.tsv", MODEL_DIR/"Gray_primary_frozen_usage.tsv",
            TABLE_DIR/"Gray_primary_frozen_ranked_genes.csv"))
        full_command={"api":"cnmf.cNMF","name":f"{ANALYSIS_NAME}_full",
            "K":frozen_k["K"] if frozen_k else "LOCKED","n_iter":FULL_ITERATIONS,
            "worker_i":0,"total_workers":1}
        (LOG_DIR/"registered_full_cNMF_command.json").write_text(json.dumps({"call":full_command,"run_enabled":RUN_FULL_CNMF},indent=2))
        if RUN_FULL_CNMF and full_outputs_ready and not FORCE_RECOMPUTE:
            print("Reusing completed primary full-consensus artifacts.")
        elif RUN_FULL_CNMF:
            from cnmf import cNMF
            full=cNMF(output_dir=str(MODEL_DIR),name=f"{ANALYSIS_NAME}_full")
            full.prepare(counts_fn=str(MODEL_DIR/"Gray_primary_balanced_counts.h5ad"),
                components=np.asarray([frozen_k["K"]]),n_iter=FULL_ITERATIONS,
                seed=REGISTERED_SEEDS[0],num_highvar_genes=2000)
            full.factorize(worker_i=0,total_workers=1)
            full.combine()
            full.consensus(k=frozen_k["K"],density_threshold=CONSENSUS_DENSITY_THRESHOLD,
                show_clustering=True,close_clustergram_fig=True)
            def unique_cnmf_output(pattern):
                hits=list(MODEL_DIR.rglob(pattern))
                if len(hits)!=1: raise RuntimeError(f"Expected one cNMF output for {pattern}; found {hits}")
                return hits[0]
            k=frozen_k["K"]
            spectra_source=unique_cnmf_output(f"*gene_spectra_tpm.k_{k}.dt_0_01.txt")
            usage_source=unique_cnmf_output(f"*usages.k_{k}.dt_0_01.consensus.txt")
            score_source=unique_cnmf_output(f"*gene_spectra_score.k_{k}.dt_0_01.txt")
            spectra=pd.read_csv(spectra_source,sep="\t",index_col=0)
            usage=pd.read_csv(usage_source,sep="\t",index_col=0)
            scores=pd.read_csv(score_source,sep="\t",index_col=0)
            spectra.to_csv(MODEL_DIR/"Gray_primary_frozen_spectra.tsv",sep="\t")
            usage.to_csv(MODEL_DIR/"Gray_primary_frozen_usage.tsv",sep="\t")
            ranked=pd.concat([pd.DataFrame({"program":program,"rank":np.arange(1,101),
                "gene":scores.loc[program].nlargest(100).index,"score":scores.loc[program].nlargest(100).values})
                for program in scores.index],ignore_index=True)
            ranked.to_csv(TABLE_DIR/"Gray_primary_frozen_ranked_genes.csv",index=False)
        else:
            print(f"EXECUTION_STAGE={EXECUTION_STAGE!r}: primary full cNMF is not launched.")
        """),
        code(r"""
        # 10. Cryptographic source freeze; required before any 3′ projection or post-hoc annotation
        SPECTRA_PATH=MODEL_DIR/"Gray_primary_frozen_spectra.tsv"
        USAGE_PATH=MODEL_DIR/"Gray_primary_frozen_usage.tsv"
        RANKED_GENES_PATH=TABLE_DIR/"Gray_primary_frozen_ranked_genes.csv"
        SOURCE_FREEZE_PATH=TABLE_DIR/"state_program_freeze_manifest.json"
        required_artifacts=[SPECTRA_PATH,USAGE_PATH,RANKED_GENES_PATH,FROZEN_K_RECORD]
        source_freeze=None
        if all(path.exists() for path in required_artifacts):
            shared_genes_path=TABLE_DIR/"primary_receptor_locus_excluded_gene_universe.csv"
            source_freeze={"schema_version":1,"analysis":ANALYSIS_NAME,"freeze_complete":True,
                "K":frozen_k["K"],
                "adjacent_sensitivity_K":frozen_k["adjacent_sensitivity_K"],"seeds":REGISTERED_SEEDS,
                "preprocessing":{"normalization":NORMALIZATION,"target_sum":1000000.0,
                    "log1p":False,"query_count_source_policy":"counts_layer_else_raw_else_integer_X",
                    "spectra_orientation":"program_by_gene","receptor_loci_excluded":True,
                    "author_labels_used_for_fitting":False,
                    "consensus_density_threshold":CONSENSUS_DENSITY_THRESHOLD},
                "cohort":{"dataset":"Gray adult","assay":"10x 5' v2","n_cells_before_balance":6466,
                    "n_cells_after_balance":primary_input.n_obs,"n_donors":22,"donor_cap":DONOR_CAP},
                "author_labels_used_for_fitting":False,"package_versions":{},"artifacts":{},
                "timestamp":datetime.now(timezone.utc).isoformat()}
            for package in ("anndata","numpy","pandas","scipy","cnmf"):
                try: source_freeze["package_versions"][package]=version(package)
                except PackageNotFoundError: source_freeze["package_versions"][package]="not-installed"
            source_freeze["spectra"]={"path":str(SPECTRA_PATH),"sha256":sha256sum(SPECTRA_PATH)}
            source_freeze["ranked_genes"]={"path":str(RANKED_GENES_PATH),"sha256":sha256sum(RANKED_GENES_PATH)}
            source_freeze["shared_genes"]={"path":str(shared_genes_path),"sha256":sha256sum(shared_genes_path)}
            source_freeze["checksums"]={"input":observed_sha256,
                "cells":sha256_lines(primary_input.obs_names),"genes":sha256_lines(primary_features),
                "usage":sha256sum(USAGE_PATH),"spectra":source_freeze["spectra"]["sha256"],
                "ranked_genes":source_freeze["ranked_genes"]["sha256"],
                "shared_genes":source_freeze["shared_genes"]["sha256"]}
            for path in required_artifacts[:3]: source_freeze["artifacts"][path.name]=sha256sum(path)
            SOURCE_FREEZE_PATH.write_text(json.dumps(source_freeze,indent=2),encoding="utf-8")
            print(f"Immutable source freeze written: {SOURCE_FREEZE_PATH}")
        else:
            print("Freeze withheld; missing full-consensus artifacts:",[str(p) for p in required_artifacts if not p.exists()])
        """),
        code(r"""
        # 11. Matched TCR-inclusive arm: identical cells/K/seeds/normalization; only receptor loci differ
        def assert_matched_arms(primary_cells,inclusive_cells,primary_features,inclusive_features,receptor_features,
            primary_k,inclusive_k,primary_seeds,inclusive_seeds,primary_normalization,inclusive_normalization):
            if list(primary_cells)!=list(inclusive_cells): raise AssertionError("cell IDs differ")
            if primary_k!=inclusive_k or tuple(primary_seeds)!=tuple(inclusive_seeds): raise AssertionError("K/seeds differ")
            if primary_normalization!=inclusive_normalization: raise AssertionError("normalization differs")
            p,i,r=set(primary_features),set(inclusive_features),set(receptor_features)
            if not p.issubset(i) or i-p!=r: raise AssertionError("Only registered receptor genes may differ")
        if source_freeze is not None:
            inclusive_receptor_features=sorted(set(inclusive_features)-set(primary_features))
            assert_matched_arms(primary_input.obs_names,primary_input.obs_names,primary_features,inclusive_features,
                inclusive_receptor_features,source_freeze["K"],source_freeze["K"],REGISTERED_SEEDS,REGISTERED_SEEDS,NORMALIZATION,NORMALIZATION)
            inclusive_spec={"cells_sha256":sha256_lines(primary_input.obs_names),"K":source_freeze["K"],
                "seeds":REGISTERED_SEEDS,"normalization":NORMALIZATION,"only_allowed_feature_addition":inclusive_receptor_features}
            (TABLE_DIR/"matched_TCR_inclusive_specification.json").write_text(json.dumps(inclusive_spec,indent=2))
            inclusive_outputs_ready=(MODEL_DIR/"Gray_TCR_inclusive_spectra.tsv").exists() and (MODEL_DIR/"Gray_TCR_inclusive_usage.tsv").exists()
            if RUN_TCR_INCLUSIVE_CNMF and inclusive_outputs_ready and not FORCE_RECOMPUTE:
                print("Reusing completed matched TCR-inclusive sensitivity artifacts.")
            elif RUN_TCR_INCLUSIVE_CNMF:
                from cnmf import cNMF
                inclusive_input=balanced[:,inclusive_features].copy()
                inclusive_input.write_h5ad(MODEL_DIR/"Gray_TCR_inclusive_counts.h5ad",compression="gzip")
                inclusive=cNMF(output_dir=str(MODEL_DIR),name=f"{ANALYSIS_NAME}_TCR_inclusive")
                inclusive.prepare(counts_fn=str(MODEL_DIR/"Gray_TCR_inclusive_counts.h5ad"),
                    components=np.asarray([source_freeze["K"]]),n_iter=FULL_ITERATIONS,
                    seed=REGISTERED_SEEDS[0],num_highvar_genes=2000)
                inclusive.factorize(worker_i=0,total_workers=1); inclusive.combine()
                inclusive.consensus(k=source_freeze["K"],density_threshold=CONSENSUS_DENSITY_THRESHOLD,
                    show_clustering=True,close_clustergram_fig=True)
                k=source_freeze["K"]
                inc_spectra=list(MODEL_DIR.rglob(f"*TCR_inclusive*gene_spectra_tpm.k_{k}.dt_0_01.txt"))
                inc_usage=list(MODEL_DIR.rglob(f"*TCR_inclusive*usages.k_{k}.dt_0_01.consensus.txt"))
                if len(inc_spectra)!=1 or len(inc_usage)!=1: raise RuntimeError("Cannot resolve unique inclusive cNMF outputs")
                pd.read_csv(inc_spectra[0],sep="\t",index_col=0).to_csv(MODEL_DIR/"Gray_TCR_inclusive_spectra.tsv",sep="\t")
                pd.read_csv(inc_usage[0],sep="\t",index_col=0).to_csv(MODEL_DIR/"Gray_TCR_inclusive_usage.tsv",sep="\t")
        """),
        code(r"""
        # 12. Program matching and receptor-dependence comparison outputs
        PRIMARY_SPECTRA=SPECTRA_PATH
        INCLUSIVE_SPECTRA=MODEL_DIR/"Gray_TCR_inclusive_spectra.tsv"
        PRIMARY_USAGE=USAGE_PATH
        INCLUSIVE_USAGE=MODEL_DIR/"Gray_TCR_inclusive_usage.tsv"
        if source_freeze is not None and INCLUSIVE_SPECTRA.exists() and INCLUSIVE_USAGE.exists():
            Wp=pd.read_csv(PRIMARY_SPECTRA,sep="\t",index_col=0); Wi=pd.read_csv(INCLUSIVE_SPECTRA,sep="\t",index_col=0)
            Hp=pd.read_csv(PRIMARY_USAGE,sep="\t",index_col=0); Hi=pd.read_csv(INCLUSIVE_USAGE,sep="\t",index_col=0)
            shared=Wp.columns.intersection(Wi.columns); rows=[]
            for p in Wp.index:
                for q in Wi.index:
                    rho=spearmanr(Wp.loc[p,shared],Wi.loc[q,shared]).statistic
                    top_p=set(Wp.loc[p].nlargest(50).index); top_q=set(Wi.loc[q].nlargest(50).index)
                    jaccard=len(top_p&top_q)/len(top_p|top_q)
                    receptor_contribution=float(Wi.loc[q,[g for g in Wi.columns if is_receptor_locus_gene(g)]].sum()/Wi.loc[q].sum())
                    usage_r=pearsonr(Hp[p],Hi[q]).statistic if p in Hp and q in Hi else np.nan
                    rows.append({"primary_program":p,"inclusive_program":q,"spectra_spearman":rho,
                        "top50_jaccard":jaccard,"usage_pearson":usage_r,"receptor_gene_contribution":receptor_contribution})
            pd.DataFrame(rows).to_csv(TABLE_DIR/"primary_inclusive_program_matching.csv",index=False)
        """),
        code(r"""
        # 13. Frozen 5′ programs project into 3′ cells; no refitting is permitted
        if source_freeze is None:
            print("3′ projection locked until state_program_freeze_manifest.json exists.")
        else:
            gray=ad.read_h5ad(GRAY_PATH)
            mask=gray.obs["assay"].astype(str).eq("10x 3' v3")
            assert int(mask.sum())==951 and set(gray.obs.loc[mask,"donor_id"].astype(str))=={"D496","D503"}
            W=pd.read_csv(SPECTRA_PATH,sep="\t",index_col=0)
            mask_array=mask.to_numpy(dtype=bool)
            raw_symbols=pd.Index(
                gray.raw.var["feature_name"].fillna(gray.raw.var_names.to_series()).astype(str)
                if "feature_name" in gray.raw.var else gray.raw.var_names.astype(str),name="gene")
            projection_counts=gray.raw.X.tocsr() if sparse.issparse(gray.raw.X) else sparse.csr_matrix(gray.raw.X)
            if raw_symbols.has_duplicates:
                # Match the discovery preprocessing: sum duplicate gene-symbol
                # columns before projecting the frozen program weights.
                unique_raw_symbols=pd.Index(pd.unique(raw_symbols),name="gene")
                raw_symbol_codes=unique_raw_symbols.get_indexer(raw_symbols)
                projection_collapse=sparse.csr_matrix(
                    (np.ones(len(raw_symbols)),(np.arange(len(raw_symbols)),raw_symbol_codes)),
                    shape=(len(raw_symbols),len(unique_raw_symbols)))
                projection_counts=(projection_counts@projection_collapse).tocsr()
                raw_symbols=unique_raw_symbols
            shared=W.columns.intersection(raw_symbols)
            coverage=len(shared)/W.shape[1]
            if coverage < 0.70: raise RuntimeError(f"3′ frozen-program gene coverage too low: {coverage:.1%}")
            # NNLS uses fixed W; program weights are never updated.
            raw_position=pd.Series(np.arange(len(raw_symbols)),index=raw_symbols)
            cell_positions=np.flatnonzero(mask_array)
            gene_positions=raw_position.loc[shared].to_numpy(dtype=int)
            X=projection_counts[cell_positions][:,gene_positions]
            row_totals=np.asarray(X.sum(axis=1)).ravel()
            if np.any(row_totals <= 0): raise RuntimeError("3′ projection contains zero-count cells")
            X=sparse.diags(1000000.0/row_totals)@X if sparse.issparse(X) else X*(1000000.0/row_totals)[:,None]
            X=np.asarray(X.toarray() if sparse.issparse(X) else X)
            W_shared=W[shared].to_numpy()
            projected=np.vstack([nnls(W_shared.T,row)[0] for row in X])
            pd.DataFrame(projected,index=gray.obs_names[mask_array],columns=W.index).to_csv(TABLE_DIR/"Gray_3prime_fixed_program_usage.csv.gz",compression="gzip")
            pd.DataFrame({"dataset":["Gray 10x 3' v3"],"gene_coverage":[coverage]}).to_csv(TABLE_DIR/"Gray_3prime_gene_coverage.csv",index=False)
        """),
        code(r"""
        # 14. Age-boundary sensitivity excludes 694B without changing frozen program weights
        if source_freeze is not None:
            usage=pd.read_csv(USAGE_PATH,sep="\t",index_col=0)
            meta=primary.obs.loc[usage.index,["donor_id","tissue","donor_age"]].copy()
            age_sensitivity=usage.loc[~meta.donor_id.astype(str).eq("694B")].copy()
            assert "694B" not in set(meta.loc[age_sensitivity.index,"donor_id"].astype(str))
            age_sensitivity.to_csv(TABLE_DIR/"Gray_usage_excluding_694B.csv.gz",compression="gzip")
            pd.DataFrame({"analysis":["frozen usage excluding 694B"],"weights_refit":[False],
                "excluded_donor":["694B"]}).to_csv(TABLE_DIR/"Gray_age_boundary_sensitivity_manifest.csv",index=False)
        """),
        code(r"""
        # 15. Post-freeze biological interpretation only
        if source_freeze is not None:
            gray=ad.read_h5ad(GRAY_PATH)
            usage=pd.read_csv(USAGE_PATH,sep="\t",index_col=0)
            interpretation=gray.obs.loc[usage.index,["donor_id","tissue","classification","specific_phenotype","vd"]].join(usage)
            interpretation.to_csv(TABLE_DIR/"Gray_frozen_program_posthoc_annotations.csv.gz",compression="gzip")
            # These fields annotate frozen programs; they never alter genes, K, weights, or program matching.
        """),
        code(r"""
        # 16. Completion manifest and declared publication outputs
        declared=[TABLE_DIR/"Gray_pre_model_audit.csv",TABLE_DIR/"Gray_discovery_cohort_manifest.csv.gz",
            TABLE_DIR/"cNMF_pilot_specification.json",TABLE_DIR/"frozen_K_record.json",TABLE_DIR/"state_program_freeze_manifest.json",
            TABLE_DIR/"primary_inclusive_program_matching.csv",TABLE_DIR/"Gray_3prime_fixed_program_usage.csv.gz",
            TABLE_DIR/"Gray_usage_excluding_694B.csv.gz",FIGURE_DIR/"cNMF_pilot_K_diagnostics.pdf"]
        completion={"analysis":ANALYSIS_NAME,"generated_utc":datetime.now(timezone.utc).isoformat(),
            "execution_stage":EXECUTION_STAGE,"expensive_cnmf_executed":RUN_FULL_CNMF,
            "source_frozen":SOURCE_FREEZE_PATH.exists(),
            "outputs":[{"path":str(p),"exists":p.exists(),"sha256":sha256sum(p) if p.exists() else None} for p in declared],
            "limitations":["Gray vd is not sequence-defined","Programs are continuous and non-mutually-exclusive",
                "Donor and tissue effects must not be inferred from cell-level pseudoreplication"]}
        COMPLETION_MANIFEST_PATH = LOG_DIR / "completion_manifest.json"
        COMPLETION_MANIFEST_PATH.write_text(json.dumps(completion,indent=2),encoding="utf-8")
        print(f"Completion manifest: {COMPLETION_MANIFEST_PATH}")
        display(pd.DataFrame(completion["outputs"]))
        """),
        md(r"""
        ## Interpretation and limitations

        A stable component is evidence for a reproducible expression program, not
        proof of a discrete cell state. Donors—not cells—are the biological
        replicates. Receptor-inclusive components dominated by TR genes are reported
        as receptor/subtype-associated rather than functional. External datasets may
        project the frozen spectra but must never refit or rename them.
        """),
    ]
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata = {
        "kernelspec": {"display_name": "Python (gdT cNMF)", "language": "python", "name": "gdt_cnmf"},
        "language_info": {"name": "python", "version": "3.10"},
        "analysis": ANALYSIS_NAME,
        "authoritative_generator": "build_16_Gray_adult_state_program_discovery_freeze.py",
    }
    return notebook


def main() -> None:
    default = Path(__file__).resolve().parent / "notebooks" / f"{ANALYSIS_NAME}.ipynb"
    output = Path(os.environ.get("GDT_NOTEBOOK_OUTPUT", default)).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(build_notebook(), output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
