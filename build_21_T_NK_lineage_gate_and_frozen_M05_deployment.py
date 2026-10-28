#!/usr/bin/env python3
"""Build notebook 21: T/NK lineage gate followed by the unchanged frozen M05.

This generator is the authoritative source.  The generated notebook is the
runnable artifact for Woody. Inner donor fits are saved incrementally inside
each outer-donor x candidate-C checkpoint.
"""

from __future__ import annotations

from pathlib import Path
import textwrap

import nbformat as nbf


NOTEBOOK_NAME = "21_T_NK_lineage_gate_and_frozen_M05_deployment.ipynb"
LOCAL_PROJECT = Path("/Users/kaelynwen/Desktop/FYP_YHH/gdT_project/gdt_lr_12model")
DEFAULT_OUTPUT = LOCAL_PROJECT / "notebooks" / NOTEBOOK_NAME


def _clean(value: str) -> str:
    return textwrap.dedent(value).strip("\n")


def md(value: str):
    return nbf.v4.new_markdown_cell(_clean(value))


def code(value: str):
    return nbf.v4.new_code_cell(_clean(value))


def build_notebook(output_path: str | Path = DEFAULT_OUTPUT) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cells = [
        md(r"""
        # 21 — T/NK lineage gate and frozen M05 deployment

        **Purpose.** Test whether the apparent NK calls produced by M05 reflect
        out-of-domain use of a γδT-versus-αβT classifier, while preserving the
        already frozen M05 model, ordered features, coefficients, and threshold.

        The end-to-end decision is deliberately hierarchical:

        1. a new RNA-only **T-versus-NK lineage gate** decides whether a cell is
           eligible for the T-cell classifier;
        2. cells confidently assigned to NK are reported as `predicted NK`;
        3. uncertain or unsupported cells are `abstain/out-of-domain`;
        4. only T-eligible cells enter the unchanged M05 and become
           `predicted gdT` or `predicted abT`.

        This does **not** rewrite the earlier results. It asks a different and
        necessary question: *was M05 used inside the population for which it was
        trained?*
        """),
        md(r"""
        ## Research question

        ### Primary question

        Among cells that M05 calls γδT in PANGEA, 15-organ, and GSE199411, how
        many remain γδT-positive after an independently trained T/NK eligibility
        gate is applied?

        ### Competing explanations

        - **Out-of-domain M05 use:** M05 was trained to distinguish γδT from αβT,
          so a cytotoxic NK cell can receive a high M05 score when it is forced
          through the classifier.
        - **Annotation impurity:** some author-labelled NK cells may contain T
          cells, doublets, or low-level T-receptor RNA contamination.
        - **Shared biological state:** NK and cytotoxic T cells share GNLY, NKG7,
          PRF1 and granzymes; state similarity is not lineage identity.

        These are not mutually exclusive. The notebook therefore reports a
        graded evidence table rather than declaring every discordant cell an
        annotation error.
        """),
        md(r"""
        ## Frozen contracts

        **M05 is never retrained here.** The notebook reads
        `models/optimized_M05.joblib`, checks M05 in `config/model_registry.csv`,
        and reads its already selected Song-LODO threshold from
        `results/tables/optimized_fit_registry.csv`. Target annotations are not
        used for either M05 training or threshold selection.

        The lineage gate is separate. Its only training reference is GSE164378,
        selected because T and NK cells were measured in the same CITE-seq study
        across the same volunteers. This prevents the fatal design in which all
        T cells come from one study and all NK cells from another study.
        """),
        md(r"""
        ## Evidence and data provenance

        - **GSE164378 / Hao et al. PBMC CITE-seq:** source-only lineage-gate
          training and internal donor-blocked validation. The official scvi-tools
          processed object contains raw RNA UMI counts, 161,764 cells, eight
          volunteers and three time points (24 batches). We read the H5AD
          directly; importing scvi-tools is not required.
        - **GSE199411, GSE130430, GSE149356, PANGEA and 15-organ:** external-only.
          Their labels may be inspected after predictions, but can never enter
          fitting, feature selection, C selection, or threshold selection.

        Primary source links:

        - Hao et al., Cell 2021: https://doi.org/10.1016/j.cell.2021.04.048
        - GSE164378: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE164378
        - scvi-tools data documentation: https://docs.scvi-tools.org/en/stable/api/reference/scvi.data.pbmc_seurat_v4_cite_seq.html
        """),
        md(r"""
        ### Reproducibility contract

        A clean **Run All** creates all declared outputs. The nested CV cell is
        resumable: each inner donor fit is saved atomically inside its
        `outer_donor × candidate_C` checkpoint, and completed units are
        reused only if the run signature, input hash, feature contract and
        schema match. If Woody disconnects, rerun the same notebook; validated
        units are skipped and incomplete units resume from their saved inner fits.

        No long training job is started merely by opening this notebook.
        """),
        code(r"""
        import importlib.util
        import subprocess
        import sys

        AUTO_INSTALL_MISSING = False
        REQUIRED = {
            "anndata": "anndata", "certifi": "certifi", "joblib": "joblib",
            "matplotlib": "matplotlib",
            "nbformat": "nbformat", "numpy": "numpy", "pandas": "pandas",
            "pyarrow": "pyarrow", "scanpy": "scanpy", "scipy": "scipy",
            "seaborn": "seaborn", "sklearn": "scikit-learn", "yaml": "pyyaml",
        }
        missing = [pkg for module, pkg in REQUIRED.items()
                   if importlib.util.find_spec(module) is None]
        print("Missing packages:", missing)
        if missing and AUTO_INSTALL_MISSING:
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
        elif missing:
            raise ModuleNotFoundError(
                "Install the missing packages in the Woody environment, restart "
                "the kernel, and rerun: " + " ".join(missing)
            )
        """),
        code(r"""
        from datetime import datetime, timezone
        from pathlib import Path
        from urllib.request import Request, urlopen
        import hashlib
        import json
        import os
        import platform
        import re
        import ssl
        import sys

        import anndata as ad
        import certifi
        import joblib
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        import numpy as np
        import pandas as pd
        import scanpy as sc
        import seaborn as sns
        from scipy import sparse
        from sklearn.metrics import (
            average_precision_score, balanced_accuracy_score, confusion_matrix,
            f1_score, precision_score, recall_score, roc_auc_score,
        )
        from sklearn.model_selection import GroupKFold
        import yaml

        permitted_fonts = ["Arial", "Helvetica", "Liberation Sans"]
        selected_font = next(
            (name for name in permitted_fonts
             if any(item.name == name for item in font_manager.fontManager.ttflist)),
            None,
        )
        if selected_font is None:
            raise RuntimeError("Install Arial, Helvetica, or Liberation Sans before plotting")
        sns.set_theme(style="white", context="paper", font=selected_font)
        plt.rcParams.update({
            "font.family": selected_font, "font.weight": "normal",
            "pdf.fonttype": 42, "ps.fonttype": 42,
            "text.color": "black", "axes.labelcolor": "black",
            "axes.edgecolor": "black", "axes.titlecolor": "black",
            "xtick.color": "black", "ytick.color": "black",
            "legend.labelcolor": "black", "savefig.dpi": 300,
        })
        """),
        code(r"""
        REMOTE_PROJECT = Path("/data/users/kaelynwen/gdT_project/gdt_lr_12model")
        LOCAL_PROJECT = Path("/Users/kaelynwen/Desktop/FYP_YHH/gdT_project/gdt_lr_12model")
        PROJECT = Path(os.environ.get("GDT_LR_PROJECT", REMOTE_PROJECT))
        if not PROJECT.exists() and LOCAL_PROJECT.exists():
            PROJECT = LOCAL_PROJECT
        if not PROJECT.exists():
            raise FileNotFoundError(f"Project root not found: {PROJECT}")

        sys.path.insert(0, str(PROJECT / "src"))
        from gdt_lr_utils import (
            TCR_PANEL_DEFAULT, build_lr, coefficient_table, make_log_normalized,
            metrics_at_threshold, select_hvgs, set_reproducibility,
            sha256_file, threshold_max_f1,
        )

        ANALYSIS_ID = "21_T_NK_lineage_gate_and_frozen_M05_deployment"
        TABLE_DIR = PROJECT / "results/tables" / ANALYSIS_ID
        FIGURE_DIR = PROJECT / "results/figures" / ANALYSIS_ID
        CHECKPOINT_DIR = PROJECT / "checkpoints" / ANALYSIS_ID
        PREDICTION_DIR = PROJECT / "predictions" / ANALYSIS_ID
        MODEL_DIR = PROJECT / "models" / ANALYSIS_ID
        LOG_DIR = PROJECT / "logs" / ANALYSIS_ID
        for directory in (TABLE_DIR, FIGURE_DIR, CHECKPOINT_DIR, PREDICTION_DIR,
                          MODEL_DIR, LOG_DIR):
            directory.mkdir(parents=True, exist_ok=True)

        SEED = 20260918
        TARGET_SUM = 1e4
        N_HVG = 2000
        MAX_SOURCE_NK_FPR = 0.01
        MAX_SOURCE_T_FNR = 0.01
        MIN_FEATURE_COVERAGE = 0.95
        BATCH_SIZE = 4096
        set_reproducibility(SEED, threads=8)
        print("PROJECT:", PROJECT)
        print("Python:", sys.version.split()[0], "Platform:", platform.platform())
        print("Figure font:", selected_font)
        """),
        code(r"""
        TRAINING_REFERENCE = "GSE164378"
        EXTERNAL_ONLY_DATASETS = (
            "GSE199411", "GSE130430", "GSE149356", "PANGEA", "15-organ"
        )
        SOURCE_URL = "https://ndownloader.figshare.com/files/27458840"
        SOURCE_DIR = PROJECT / "data" / TRAINING_REFERENCE
        SOURCE_H5AD = SOURCE_DIR / "pbmc_seurat_v4.h5ad"

        FROZEN_M05_PATH = PROJECT / "models/optimized_M05.joblib"
        MODEL_REGISTRY_PATH = PROJECT / "config/model_registry.csv"
        M05_THRESHOLD_REGISTRY = PROJECT / "results/tables/optimized_fit_registry.csv"

        # Exact source labels are audited before this allowlist is used.
        # They are not silently relabelled in exported tables.
        SOURCE_T_LABELS = ("CD4 T", "CD8 T", "other T")
        SOURCE_NK_LABELS = ("NK",)

        def assert_external_only_contract(training_reference, training_rows):
            if training_reference != TRAINING_REFERENCE:
                raise AssertionError("Only GSE164378 may train the lineage gate")
            if "dataset" in training_rows:
                observed = set(training_rows["dataset"].astype(str).unique())
                forbidden = observed.intersection(EXTERNAL_ONLY_DATASETS)
                if forbidden:
                    raise AssertionError(f"External-only datasets entered training: {forbidden}")
            return True

        contract = pd.DataFrame([
            {
                "component": "lineage_gate", "training_reference": TRAINING_REFERENCE,
                "target_labels_used_for_training": False,
                "target_labels_used_for_threshold_selection": False,
            },
            {
                "component": "M05", "training_reference": "Song",
                "target_labels_used_for_training": False,
                "target_labels_used_for_threshold_selection": False,
            },
        ])
        display(contract)
        """),
        code(r"""
        output_records = []

        def record_output(path, kind, description):
            output_records.append({
                "path": str(Path(path)), "kind": kind, "description": description
            })

        def atomic_json(payload, path):
            path = Path(path); temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            os.replace(temporary, path)
            return path

        def atomic_csv(frame, path, **kwargs):
            path = Path(path); temporary = path.with_suffix(path.suffix + ".tmp")
            frame.to_csv(temporary, index=False, **kwargs)
            os.replace(temporary, path)
            return path

        def atomic_parquet(frame, path):
            path = Path(path); temporary = path.with_suffix(path.suffix + ".tmp")
            frame.to_parquet(temporary, index=False)
            os.replace(temporary, path)
            return path

        def atomic_joblib(payload, path):
            path = Path(path); temporary = path.with_suffix(path.suffix + ".tmp")
            joblib.dump(payload, temporary)
            os.replace(temporary, path)
            return path

        def save_publication_figure(fig, stem, description):
            pdf = FIGURE_DIR / f"{stem}.pdf"
            png = FIGURE_DIR / f"{stem}.png"
            fig.savefig(pdf, format="pdf", bbox_inches="tight")
            fig.savefig(png, dpi=300, bbox_inches="tight")
            record_output(pdf, "figure_pdf", description)
            record_output(png, "figure_png_preview", description)
            return pdf

        def stable_hash(payload):
            encoded = json.dumps(payload, sort_keys=True, default=str).encode()
            return hashlib.sha256(encoded).hexdigest()
        """),
        md(r"""
        ### Download the training reference once

        The download is atomic: data first enter a `.part` file and become the
        declared H5AD only after completion. A partial network transfer is never
        mistaken for a valid dataset. Set `DOWNLOAD_SOURCE = True` on the first
        Woody run if the file is absent. An existing `.part` file is resumed when
        the server honours HTTP range requests.
        """),
        code(r"""
        DOWNLOAD_SOURCE = True
        SOURCE_DIR.mkdir(parents=True, exist_ok=True)

        def atomic_download(url, destination, chunk_size=2**20):
            destination = Path(destination)
            temporary = destination.with_suffix(destination.suffix + ".part")
            completed = temporary.stat().st_size if temporary.exists() else 0
            request = Request(url, headers={"Range": f"bytes={completed}-"}) if completed else Request(url)
            # Woody's Python 3.13 can point OpenSSL at the obsolete
            # /home/kaelynwen/ENTER path. Use the CA bundle installed in the
            # active /data/conda_envs/kaelynwen/ENTER environment instead of
            # disabling TLS certificate verification.
            tls_context = ssl.create_default_context(cafile=certifi.where())
            response = urlopen(request, context=tls_context, timeout=60)
            supports_resume = completed > 0 and getattr(response, "status", None) == 206
            if completed and not supports_resume:
                completed = 0
            mode = "ab" if supports_resume else "wb"
            print(f"Downloading {url} -> {temporary} from byte {completed:,}")
            with response, open(temporary, mode) as handle:
                written = completed
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    handle.write(chunk)
                    written += len(chunk)
                    if written // (512 * 2**20) != (written - len(chunk)) // (512 * 2**20):
                        print(f"  downloaded {written / 2**30:.2f} GiB")
            os.replace(temporary, destination)
            return destination

        if not SOURCE_H5AD.exists():
            if DOWNLOAD_SOURCE:
                atomic_download(SOURCE_URL, SOURCE_H5AD)
            else:
                raise FileNotFoundError(f"Missing {SOURCE_H5AD}; set DOWNLOAD_SOURCE=True")

        source_input_audit = pd.DataFrame([{
            "dataset": TRAINING_REFERENCE, "path": str(SOURCE_H5AD),
            "bytes": SOURCE_H5AD.stat().st_size, "sha256": sha256_file(SOURCE_H5AD),
            "download_url": SOURCE_URL,
        }])
        source_input_path = TABLE_DIR / "GSE164378_input_audit.csv"
        atomic_csv(source_input_audit, source_input_path)
        record_output(source_input_path, "table", "GSE164378 source file provenance")
        display(source_input_audit)
        """),
        md(r"""
        ### Audit exact source annotations and donors

        No label is guessed. The cell-level `celltype.l1`, `celltype.l2` and
        `donor` and `orig.ident` fields must exist; all observed categories are
        exported. The published `donor` field supplies the eight biological
        volunteers, while `orig.ident` remains the 24 donor-by-time-point batches.
        The notebook stops rather than silently treating those 24 batches as
        independent donors.
        """),
        code(r"""
        source = ad.read_h5ad(SOURCE_H5AD)
        required_obs = ["celltype.l1", "celltype.l2", "donor", "orig.ident"]
        missing_obs = [column for column in required_obs if column not in source.obs]
        if missing_obs:
            raise KeyError(f"GSE164378 missing required source fields: {missing_obs}")

        source.var_names = source.var_names.astype(str)
        source.obs_names = source.obs_names.astype(str)
        source.obs["dataset"] = TRAINING_REFERENCE
        source.obs["source_celltype_l1"] = source.obs["celltype.l1"].astype(str)
        source.obs["source_celltype_l2"] = source.obs["celltype.l2"].astype(str)
        source.obs["source_batch"] = source.obs["orig.ident"].astype(str)
        source.obs["donor"] = source.obs["donor"].astype(str)
        if source.obs["donor"].isna().any() or source.obs["donor"].eq("").any():
            raise ValueError("GSE164378 contains missing donor identifiers")
        if source.obs["donor"].nunique() != 8:
            raise ValueError(f"Expected eight volunteers; found {source.obs.donor.unique()}")

        label_audit = pd.concat([
            source.obs["source_celltype_l1"].value_counts(dropna=False).rename("n_cells")
                .rename_axis("source_label").reset_index().assign(source_field="celltype.l1"),
            source.obs["source_celltype_l2"].value_counts(dropna=False).rename("n_cells")
                .rename_axis("source_label").reset_index().assign(source_field="celltype.l2"),
        ], ignore_index=True)
        label_audit_path = TABLE_DIR / "GSE164378_exact_source_label_audit.csv"
        atomic_csv(label_audit, label_audit_path)
        record_output(label_audit_path, "table", "Exact GSE164378 source annotation categories")
        display(label_audit.loc[label_audit.source_field.eq("celltype.l1")])
        """),
        md(r"""
        ### Define the source T/NK learning cohort

        The source labels remain visible in every export. A derived binary `y`
        is used only for the gate: `1 = T-lineage`, `0 = NK`. Cells labelled as
        Doublet at level 2 are excluded. All other cell types are retained as an
        **OOD stress-control**, not used to fit the LR or choose its threshold.

        The RNA QC mirrors the published loader where fields are available:
        mitochondrial fraction below 12%, adequate detected genes, and removal
        of source doublets. QC exclusions are counted, never silently dropped.
        """),
        code(r"""
        sc.pp.calculate_qc_metrics(source, qc_vars=[], inplace=True, log1p=False)
        mitochondrial = source.var_names.str.upper().str.startswith("MT-")
        if mitochondrial.any():
            mt_counts = np.asarray(source.X[:, mitochondrial].sum(axis=1)).ravel()
            totals = np.asarray(source.X.sum(axis=1)).ravel()
            source.obs["pct_counts_mt_recomputed"] = np.divide(
                100 * mt_counts, totals, out=np.zeros_like(mt_counts, dtype=float), where=totals > 0
            )
        else:
            source.obs["pct_counts_mt_recomputed"] = np.nan

        observed_l1 = set(source.obs["source_celltype_l1"].unique())
        missing_labels = set(SOURCE_T_LABELS + SOURCE_NK_LABELS) - observed_l1
        if missing_labels:
            raise ValueError(
                f"Expected exact source labels absent: {sorted(missing_labels)}; "
                f"observed={sorted(observed_l1)}"
            )

        source.obs["source_doublet"] = source.obs["source_celltype_l2"].eq("Doublet")
        protein_qc_pass = pd.Series(True, index=source.obs_names)
        if "Protein log library size" in source.obs:
            protein_library = pd.to_numeric(source.obs["Protein log library size"], errors="coerce")
            protein_qc_pass &= protein_library.gt(7.6) & protein_library.lt(10.3)
        if "Number proteins detected" in source.obs:
            n_proteins = pd.to_numeric(source.obs["Number proteins detected"], errors="coerce")
            protein_qc_pass &= n_proteins.gt(150)
        source.obs["published_multimodal_qc_pass"] = protein_qc_pass
        source.obs["rna_qc_pass"] = (
            source.obs["n_genes_by_counts"].ge(200)
            & (~source.obs["source_doublet"])
            & (source.obs["pct_counts_mt_recomputed"].fillna(0).lt(12))
            & source.obs["published_multimodal_qc_pass"]
        )
        source.obs["gate_training_eligible"] = (
            source.obs["rna_qc_pass"]
            & source.obs["source_celltype_l1"].isin(SOURCE_T_LABELS + SOURCE_NK_LABELS)
        )
        source.obs["y"] = np.where(
            source.obs["source_celltype_l1"].isin(SOURCE_T_LABELS), 1,
            np.where(source.obs["source_celltype_l1"].isin(SOURCE_NK_LABELS), 0, np.nan),
        )
        training = source.obs.loc[source.obs.gate_training_eligible, [
            "dataset", "donor", "source_batch", "source_celltype_l1",
            "source_celltype_l2", "y", "n_genes_by_counts", "total_counts",
            "pct_counts_mt_recomputed",
        ]].copy()
        training["y"] = training["y"].astype(int)
        assert_external_only_contract(TRAINING_REFERENCE, training)
        if training.groupby("donor")["y"].nunique().min() != 2:
            raise ValueError("Every volunteer must contribute both T and NK cells")

        cohort_audit = (
            training.groupby(["donor", "source_celltype_l1", "y"], observed=True)
            .size().rename("n_cells").reset_index()
        )
        cohort_path = TABLE_DIR / "GSE164378_gate_training_cohort_by_donor.csv"
        atomic_csv(cohort_audit, cohort_path)
        record_output(cohort_path, "table", "Source-only gate training cohort by donor")
        display(cohort_audit)
        """),
        md(r"""
        ### LR preprocessing and donor-balanced weights

        This follows the M05 LR family: library-size normalisation to 10,000,
        `log1p`, sparse standardisation without centring, and L2 logistic
        regression with `liblinear`. Feature selection is repeated inside every
        outer fold so that the held-out donor cannot influence the genes.

        Each donor has equal total weight and, within each donor, T and NK each
        receive half of that donor's weight. A large cell-rich donor therefore
        cannot dominate the fit.
        """),
        code(r"""
        def gate_training_weights(frame):
            weights = pd.Series(0.0, index=frame.index, dtype=float)
            donors = frame["donor"].astype(str).unique()
            for donor in donors:
                donor_mask = frame["donor"].astype(str).eq(donor)
                for label in (0, 1):
                    mask = donor_mask & frame["y"].eq(label)
                    if not mask.any():
                        raise ValueError(f"Donor {donor} lacks class {label}")
                    weights.loc[mask] = 0.5 / mask.sum()
            return weights * (len(weights) / weights.sum())

        def matrix_for_genes(adata, cells, genes, allow_zero_fill=False):
            row_positions = pd.Index(adata.obs_names.astype(str)).get_indexer(list(map(str, cells)))
            if (row_positions < 0).any():
                missing_cells = np.asarray(list(map(str, cells)))[row_positions < 0][:20]
                raise KeyError(f"Cells absent from AnnData: {missing_cells.tolist()}")
            symbol_positions = {}
            for position, symbol in enumerate(adata.var_names.astype(str)):
                symbol_positions.setdefault(symbol, []).append(position)
            # Column-oriented storage avoids scanning the entire sparse matrix
            # again for each gene. Values and the ordered gene panel are unchanged.
            matrix = sparse.csc_matrix(adata.X[row_positions])
            columns, missing = [], []
            for gene in map(str, genes):
                indices = symbol_positions.get(gene, [])
                if len(indices) == 1:
                    columns.append(matrix[:, indices[0]])
                elif len(indices) > 1:
                    columns.append(sparse.csr_matrix(matrix[:, indices].sum(axis=1)))
                elif allow_zero_fill:
                    columns.append(sparse.csr_matrix((len(cells), 1), dtype=np.float32))
                    missing.append(gene)
                else:
                    missing.append(gene)
            if missing and not allow_zero_fill:
                raise ValueError(f"Missing model genes: {missing[:20]} ({len(missing)} total)")
            panel = sparse.hstack(columns, format="csr", dtype=np.float32)
            totals = np.asarray(panel.sum(axis=1)).ravel()
            if (totals <= 0).any():
                raise ValueError("At least one cell has zero counts over the model panel")
            panel = sparse.diags((TARGET_SUM / totals).astype(np.float32)) @ panel
            panel.data = np.log1p(panel.data)
            return panel, missing

        def source_only_genes(adata, eligible_index, donors, n_top=N_HVG):
            eligible = adata.obs_names.isin(eligible_index) & adata.obs["donor"].isin(donors)
            hvgs, hvg_meta = select_hvgs(
                adata, n_top=n_top, batch_key="donor", eligible_cells=eligible,
                exclude_tcr=True, seed=SEED,
            )
            genes = list(dict.fromkeys(hvgs + [g for g in TCR_PANEL_DEFAULT if g in adata.var_names]))
            return genes, hvg_meta

        def fit_gate(adata, frame, genes, C):
            X, _ = matrix_for_genes(adata, frame.index.tolist(), genes)
            model = build_lr(C=float(C), seed=SEED)
            model.fit(X, frame["y"].to_numpy(),
                      lr__sample_weight=gate_training_weights(frame).to_numpy())
            return model

        def predict_gate(model, adata, cell_ids, genes, allow_zero_fill=False):
            X, missing = matrix_for_genes(adata, list(cell_ids), genes, allow_zero_fill)
            return model.predict_proba(X)[:, 1], missing
        """),
        md(r"""
        ## Source-only nested donor CV

        Outer folds estimate generalisation to an unseen volunteer. Within each
        outer training set, inner donor folds select `C`. Neither the held-out
        outer donor nor any external dataset can influence the feature panel or
        `C`. The OOF probabilities produced here are the only probabilities used
        to freeze the gate operating points.
        """),
        code(r"""
        grid_path = PROJECT / "config/lr_grid.yaml"
        if grid_path.exists():
            grid_config = yaml.safe_load(grid_path.read_text()) or {}
            candidate_C_values = grid_config.get(
                "C_grid", grid_config.get("C_values", grid_config.get("C", None))
            )
        else:
            candidate_C_values = None
        if not candidate_C_values:
            candidate_C_values = [0.01, 0.1, 1.0, 10.0]
        candidate_C_values = [float(value) for value in candidate_C_values]

        source_hash = sha256_file(SOURCE_H5AD)
        run_parameters = {
            "analysis_id": ANALYSIS_ID, "training_reference": TRAINING_REFERENCE,
            "source_sha256": source_hash, "seed": SEED, "n_hvg": N_HVG,
            "candidate_C": candidate_C_values, "normalization_target_sum": TARGET_SUM,
            "source_T_labels": SOURCE_T_LABELS, "source_NK_labels": SOURCE_NK_LABELS,
            "external_only": EXTERNAL_ONLY_DATASETS,
        }
        run_signature = stable_hash(run_parameters)
        signature_path = TABLE_DIR / "run_signature.json"
        atomic_json(run_parameters | {"run_signature": run_signature}, signature_path)
        record_output(signature_path, "manifest", "Lineage-gate immutable run signature")
        print("run_signature:", run_signature)
        print("candidate_C:", candidate_C_values)
        """),
        code(r"""
        def inner_cv_score(adata, outer_training, genes, candidate_C,
                           checkpoint_path, payload):
            rows = payload.get("inner_scores", pd.DataFrame()).to_dict("records")
            groups = outer_training["donor"].astype(str).to_numpy()
            splitter = GroupKFold(n_splits=len(np.unique(groups)))
            for inner_fold, (train_pos, validation_pos) in enumerate(
                splitter.split(outer_training, outer_training.y, groups=groups)
            ):
                inner_train = outer_training.iloc[train_pos]
                inner_validation = outer_training.iloc[validation_pos]
                validation_donor = str(inner_validation.donor.iloc[0])
                if any(row["inner_fold"] == inner_fold and
                       str(row["validation_donor"]) == validation_donor for row in rows):
                    continue
                print(f"outer={payload['outer_donor']} C={candidate_C:g} "
                      f"inner={inner_fold + 1}/{len(np.unique(groups))} "
                      f"validation={validation_donor}", flush=True)
                model = fit_gate(adata, inner_train, genes, candidate_C)
                probability, _ = predict_gate(
                    model, adata, inner_validation.index, genes, allow_zero_fill=False
                )
                rows.append({
                    "inner_fold": inner_fold,
                    "validation_donor": validation_donor,
                    "auprc": average_precision_score(inner_validation.y, probability),
                    "auroc": roc_auc_score(inner_validation.y, probability),
                })
                payload["inner_scores"] = pd.DataFrame(rows)
                atomic_joblib(payload, checkpoint_path)
                print(f"Saved {checkpoint_path.name}: {len(rows)} inner folds", flush=True)
            return pd.DataFrame(rows)

        checkpoint_manifest_rows = []
        outer_oof_parts = []
        donors = sorted(training["donor"].astype(str).unique())
        for outer_donor in donors:
            outer_training = training.loc[~training.donor.astype(str).eq(outer_donor)].copy()
            outer_test = training.loc[training.donor.astype(str).eq(outer_donor)].copy()
            fold_genes, hvg_meta = source_only_genes(
                source, outer_training.index, sorted(outer_training.donor.unique())
            )
            feature_hash = stable_hash(fold_genes)
            candidate_summaries = []
            for candidate_C in candidate_C_values:
                unit_name = f"outer_{outer_donor}__C_{candidate_C:g}.joblib"
                checkpoint_path = CHECKPOINT_DIR / unit_name
                expected = {
                    "run_signature": run_signature, "outer_donor": outer_donor,
                    "candidate_C": candidate_C, "feature_hash": feature_hash,
                }
                reuse = False
                loaded = expected | {"genes": fold_genes, "hvg_metadata": hvg_meta}
                if checkpoint_path.exists():
                    saved = joblib.load(checkpoint_path)
                    valid = all(saved.get(key) == value for key, value in expected.items())
                    scores = saved.get("inner_scores", pd.DataFrame())
                    valid &= set(scores.columns) >= {
                        "inner_fold", "validation_donor", "auprc", "auroc"
                    }
                    if valid:
                        inner_donors = sorted(outer_training.donor.astype(str).unique())
                        # GroupKFold may order equally sized groups differently;
                        # reconstruct its actual split contract before trusting rows.
                        splitter = GroupKFold(n_splits=len(inner_donors))
                        expected_folds = {
                            i: str(outer_training.iloc[val].donor.iloc[0])
                            for i, (_, val) in enumerate(splitter.split(
                                outer_training, outer_training.y,
                                groups=outer_training.donor.astype(str).to_numpy()))
                        }
                        valid = (not scores.inner_fold.duplicated().any()
                                 and not scores.validation_donor.duplicated().any()
                                 and all(expected_folds.get(row.inner_fold) == str(row.validation_donor)
                                         for row in scores.itertuples())
                                 and np.isfinite(scores[["auprc", "auroc"]].to_numpy()).all())
                        if valid:
                            loaded = saved
                            reuse = len(scores) == len(inner_donors)
                print(f"outer={outer_donor} C={candidate_C:g}: "
                      f"{'reusing completed checkpoint' if reuse else 'resuming or starting'}", flush=True)
                if not reuse:
                    inner_scores = inner_cv_score(source, outer_training, fold_genes, candidate_C,
                                                  checkpoint_path, loaded)
                    loaded = expected | {
                        "genes": fold_genes, "hvg_metadata": hvg_meta,
                        "inner_scores": inner_scores,
                    }
                    atomic_joblib(loaded, checkpoint_path)
                candidate_summaries.append({
                    "candidate_C": candidate_C,
                    "mean_inner_auprc": loaded["inner_scores"].auprc.mean(),
                    "mean_inner_auroc": loaded["inner_scores"].auroc.mean(),
                })
                checkpoint_manifest_rows.append(
                    expected | {"path": str(checkpoint_path), "validated_reuse": reuse}
                )
                atomic_csv(pd.DataFrame(checkpoint_manifest_rows), TABLE_DIR / "checkpoint_manifest.csv")

            candidate_table = pd.DataFrame(candidate_summaries).sort_values(
                ["mean_inner_auprc", "mean_inner_auroc", "candidate_C"],
                ascending=[False, False, True],
            )
            selected_C = float(candidate_table.iloc[0].candidate_C)
            outer_model = fit_gate(source, outer_training, fold_genes, selected_C)
            outer_probability, _ = predict_gate(
                outer_model, source, outer_test.index, fold_genes, allow_zero_fill=False
            )
            outer_oof_parts.append(pd.DataFrame({
                "cell_id": outer_test.index, "donor": outer_test.donor.to_numpy(),
                "source_celltype_l1": outer_test.source_celltype_l1.to_numpy(),
                "y": outer_test.y.to_numpy(), "p_T_lineage_OOF": outer_probability,
                "outer_donor": outer_donor, "selected_C": selected_C,
                "n_fold_genes": len(fold_genes), "feature_hash": feature_hash,
                "run_signature": run_signature,
            }))

        checkpoint_manifest = pd.DataFrame(checkpoint_manifest_rows)
        checkpoint_manifest_path = TABLE_DIR / "checkpoint_manifest.csv"
        atomic_csv(checkpoint_manifest, checkpoint_manifest_path)
        record_output(checkpoint_manifest_path, "manifest", "Validated outer-donor x C checkpoints")
        print("Completed outer donors:", donors)
        """),
        code(r"""
        OOF = pd.concat(outer_oof_parts, ignore_index=True)
        if OOF.cell_id.duplicated().any() or set(OOF.cell_id) != set(training.index):
            raise AssertionError("OOF predictions must cover every training cell exactly once")
        oof_path = PREDICTION_DIR / "GSE164378_T_NK_gate_donor_blocked_OOF.parquet"
        atomic_parquet(OOF, oof_path)
        record_output(oof_path, "predictions", "Donor-blocked source-only OOF gate predictions")

        donor_metrics = []
        for donor, frame in OOF.groupby("donor", observed=True):
            donor_metrics.append({
                "donor": donor, "n_cells": len(frame),
                "auprc": average_precision_score(frame.y, frame.p_T_lineage_OOF),
                "auroc": roc_auc_score(frame.y, frame.p_T_lineage_OOF),
            })
        donor_metrics = pd.DataFrame(donor_metrics)
        donor_metrics_path = TABLE_DIR / "GSE164378_gate_OOF_metrics_by_donor.csv"
        atomic_csv(donor_metrics, donor_metrics_path)
        record_output(donor_metrics_path, "table", "Gate OOF discrimination by held-out donor")
        display(donor_metrics)
        """),
        md(r"""
        ### Freeze two-sided eligibility operating points from OOF only

        - `t_T`: the smallest OOF threshold whose NK false-positive rate is at
          most 1%; values at or above it are T-eligible.
        - `t_NK`: the largest low-side threshold whose T false-NK rate is at most
          1%; values at or below it are confidently NK.
        - values between them abstain. If the independently selected acceptance
          regions overlap, the entire closed overlap abstains: move the T cutoff
          just above the original NK cutoff and the NK cutoff just below the
          original T cutoff. This only removes accepted cells, so neither error
          rate can increase. A one-step floating-point offset handles exact ties.

        The traditional max-F1 threshold is also saved as a sensitivity analysis,
        but it is not allowed to erase the abstention zone. These error limits are
        empirical pooled source-OOF constraints, not statistical guarantees for
        new donors or external atlases. The conservative overlap policy is a
        documented repair after inspecting the source OOF failure; it has not
        been selected using external labels.
        """),
        code(r"""
        def threshold_with_max_nk_fpr(y, p, max_fpr=MAX_SOURCE_NK_FPR):
            y = np.asarray(y); p = np.asarray(p, dtype=float)
            if (y.ndim != 1 or p.shape != y.shape or set(np.unique(y)) != {0, 1}
                    or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any()
                    or not 0 <= max_fpr <= 1):
                raise ValueError("Expected aligned binary labels, finite probabilities and a valid error limit")
            # Include rejecting every cell, even if a probability equals one.
            candidates = np.append(np.unique(p), np.nextafter(p.max(), np.inf))
            nk = np.sort(p[y == 0])
            fpr = (len(nk) - np.searchsorted(nk, candidates, side="left")) / len(nk)
            return float(candidates[np.flatnonzero(fpr <= max_fpr)[0]])

        def threshold_with_max_t_fnr(y, p, max_fnr=MAX_SOURCE_T_FNR):
            y = np.asarray(y); p = np.asarray(p, dtype=float)
            if (y.ndim != 1 or p.shape != y.shape or set(np.unique(y)) != {0, 1}
                    or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any()
                    or not 0 <= max_fnr <= 1):
                raise ValueError("Expected aligned binary labels, finite probabilities and a valid error limit")
            candidates = np.insert(np.unique(p), 0, np.nextafter(p.min(), -np.inf))
            t = np.sort(p[y == 1])
            fnr = np.searchsorted(t, candidates, side="right") / len(t)
            return float(candidates[np.flatnonzero(fnr <= max_fnr)[-1]])

        gate_threshold_max_f1 = float(threshold_max_f1(OOF.y, OOF.p_T_lineage_OOF))
        gate_threshold_T = threshold_with_max_nk_fpr(OOF.y, OOF.p_T_lineage_OOF)
        gate_threshold_NK = threshold_with_max_t_fnr(OOF.y, OOF.p_T_lineage_OOF)
        print("Independent OOF cutoffs:", {"T": gate_threshold_T, "NK": gate_threshold_NK})
        if gate_threshold_NK >= gate_threshold_T:
            gate_threshold_NK, gate_threshold_T = (
                float(np.nextafter(gate_threshold_T, -np.inf)),
                float(np.nextafter(gate_threshold_NK, np.inf)),
            )
            print("Overlapping acceptance regions assigned to abstain/out-of-domain.")
        assert gate_threshold_NK < gate_threshold_T
        assert (OOF.loc[OOF.y.eq(0), "p_T_lineage_OOF"] >= gate_threshold_T).mean() <= MAX_SOURCE_NK_FPR
        assert (OOF.loc[OOF.y.eq(1), "p_T_lineage_OOF"] <= gate_threshold_NK).mean() <= MAX_SOURCE_T_FNR

        OOF["gate_call"] = np.select(
            [OOF.p_T_lineage_OOF >= gate_threshold_T,
             OOF.p_T_lineage_OOF <= gate_threshold_NK],
            ["T-lineage eligible", "predicted NK"],
            default="abstain/out-of-domain",
        )
        threshold_table = pd.DataFrame([
            {"operating_point": "T-lineage eligible", "threshold": gate_threshold_T,
             "source_constraint": f"NK FPR <= {MAX_SOURCE_NK_FPR:.3f}",
             "threshold_selection_data": "GSE164378 donor-blocked OOF only"},
            {"operating_point": "predicted NK", "threshold": gate_threshold_NK,
             "source_constraint": f"T false-NK rate <= {MAX_SOURCE_T_FNR:.3f}",
             "threshold_selection_data": "GSE164378 donor-blocked OOF only"},
            {"operating_point": "max-F1 sensitivity", "threshold": gate_threshold_max_f1,
             "source_constraint": "maximise pooled OOF F1",
             "threshold_selection_data": "GSE164378 donor-blocked OOF only"},
        ])
        threshold_path = TABLE_DIR / "frozen_lineage_gate_operating_points.csv"
        atomic_csv(threshold_table, threshold_path)
        record_output(threshold_path, "parameter", "OOF-only frozen lineage-gate thresholds")
        display(threshold_table)
        """),
        code(r"""
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        for label, color in [(0, "#D55E00"), (1, "#0072B2")]:
            values = OOF.loc[OOF.y.eq(label), "p_T_lineage_OOF"]
            ax.hist(values, bins=50, density=True, histtype="stepfilled", alpha=0.45,
                    color=color, edgecolor="none",
                    label="source NK" if label == 0 else "source T")
        ax.axvline(gate_threshold_NK, color="black", linestyle=":", linewidth=1)
        ax.axvline(gate_threshold_T, color="black", linestyle="--", linewidth=1)
        ax.set(xlabel="Donor-blocked OOF P(T-lineage)", ylabel="Density",
               title="Source-only gate separation and abstention interval")
        ax.legend(frameon=False)
        sns.despine(ax=ax)
        fig.tight_layout()
        save_publication_figure(fig, "GSE164378_gate_OOF_probability_distribution",
                                "Donor-blocked gate probabilities and frozen operating points")
        plt.show()
        """),
        md(r"""
        ### Fit and freeze the final lineage gate

        The final feature panel is selected from all eligible GSE164378 T/NK
        cells only. The final `C` is the median/mode-compatible value selected
        across outer donors. The bundle records source hashes, ordered genes,
        exact source labels and thresholds. It is a new upstream model; it does
        not alter M05.
        """),
        code(r"""
        selected_C_values = pd.Series([part.selected_C.iloc[0] for part in outer_oof_parts])
        final_C = float(selected_C_values.mode().sort_values().iloc[0])
        final_genes, final_hvg_metadata = source_only_genes(
            source, training.index, sorted(training.donor.unique())
        )
        final_gate = fit_gate(source, training, final_genes, final_C)
        gate_bundle = {
            "model": final_gate, "genes": final_genes, "C": final_C,
            "threshold_T": gate_threshold_T, "threshold_NK": gate_threshold_NK,
            "threshold_max_f1": gate_threshold_max_f1,
            "training_reference": TRAINING_REFERENCE,
            "source_T_labels": SOURCE_T_LABELS, "source_NK_labels": SOURCE_NK_LABELS,
            "source_sha256": source_hash, "run_signature": run_signature,
            "hvg_metadata": final_hvg_metadata, "seed": SEED,
            "target_labels_used_for_training": False,
            "target_labels_used_for_threshold_selection": False,
        }
        gate_model_path = MODEL_DIR / f"{ANALYSIS_ID}_frozen_lineage_gate.joblib"
        atomic_joblib(gate_bundle, gate_model_path)
        record_output(gate_model_path, "model", "Frozen source-only T/NK lineage gate")

        coefficients = coefficient_table(final_gate, final_genes)
        coefficient_path = TABLE_DIR / "frozen_lineage_gate_coefficients.csv"
        atomic_csv(coefficients, coefficient_path)
        record_output(coefficient_path, "table", "Frozen lineage-gate coefficients")
        print("Frozen gate:", gate_model_path)
        """),
        md(r"""
        ## Load and verify the unchanged frozen M05

        The following cell is intentionally defensive. It requires the exact M05
        ID and TCR-inclusive gene regime in the registered design, verifies class
        order `0=abT, 1=gdT`, and reads the already frozen threshold. Nothing in
        this notebook is permitted to call `.fit()` on M05.
        """),
        code(r"""
        required_m05_inputs = {
            "frozen_M05": FROZEN_M05_PATH,
            "model_registry": MODEL_REGISTRY_PATH,
            "M05_threshold_registry": M05_THRESHOLD_REGISTRY,
        }
        missing = [f"{name}: {path}" for name, path in required_m05_inputs.items()
                   if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing frozen M05 inputs:\n- " + "\n- ".join(missing))

        design_registry = pd.read_csv(MODEL_REGISTRY_PATH)
        m05_design = design_registry.loc[design_registry.model_id.eq("M05")]
        if len(m05_design) != 1 or m05_design.iloc[0].gene_regime != "G2_TCR_inclusive":
            raise ValueError("config/model_registry.csv does not contain the expected M05 contract")
        m05_bundle = joblib.load(FROZEN_M05_PATH)
        if list(m05_bundle["model"].classes_) != [0, 1]:
            raise ValueError("M05 must have class order 0=abT, 1=gdT")
        threshold_registry = pd.read_csv(M05_THRESHOLD_REGISTRY)
        m05_row = threshold_registry.loc[threshold_registry.model_id.eq("M05")]
        if len(m05_row) != 1:
            raise ValueError("Expected exactly one M05 row in optimized_fit_registry.csv")
        threshold_candidates = [
            "threshold", "threshold_max_f1_song_lodo", "frozen_threshold"
        ]
        threshold_column = next((column for column in threshold_candidates
                                 if column in m05_row.columns), None)
        if threshold_column is None:
            raise KeyError(f"No frozen threshold column in {M05_THRESHOLD_REGISTRY}")
        M05_THRESHOLD = float(m05_row.iloc[0][threshold_column])
        M05_SHA256 = sha256_file(FROZEN_M05_PATH)
        song_reference=ad.read_h5ad(PROJECT/"manifests/Song_D1_cohorts.h5ad",backed="r")
        if "gene_ids" not in song_reference.var or not song_reference.var_names.is_unique:
            raise ValueError("Song reference must have unique symbols and stable gene_ids")
        M05_GENE_IDS=song_reference.var.gene_ids.astype(str).to_dict()
        song_reference.file.close()

        m05_contract = pd.DataFrame([{
            "model_id": "M05", "path": str(FROZEN_M05_PATH), "sha256": M05_SHA256,
            "n_genes": len(m05_bundle["genes"]), "threshold": M05_THRESHOLD,
            "threshold_source": str(M05_THRESHOLD_REGISTRY),
            "target_labels_used_for_training": False,
            "target_labels_used_for_threshold_selection": False,
            "retrained_in_this_notebook": False,
        }])
        m05_contract_path = TABLE_DIR / "unchanged_frozen_M05_contract.csv"
        atomic_csv(m05_contract, m05_contract_path)
        record_output(m05_contract_path, "parameter", "Hash and threshold of unchanged frozen M05")
        display(m05_contract)
        """),
        md(r"""
        ### Exact external feature alignment and missing-weight audit

        M05 features use stable Ensembl IDs from Song, with exact-symbol fallback,
        matching notebook 20. The gate retains exact-symbol mapping. Duplicate
        symbols are summed only when they do not reuse a target across model genes. Missing genes are explicit zeros in the frozen order. Coverage
        alone is not enough: the notebook also reports the fraction of absolute
        LR coefficient mass that was zero-filled. Coverage below 95% prevents
        decisions at the affected node. Supported root NK calls do not require
        child coverage. Missing coefficient mass is descriptive, not an additional
        acceptance rule. The 95% threshold has not been lowered.
        """),
        code(r"""
        def lr_coefficients(model):
            estimator = model.named_steps["lr"] if hasattr(model, "named_steps") else model
            return np.asarray(estimator.coef_).reshape(-1)

        def resolve_external_panel(target, genes, reference_gene_ids=None):
            reference_gene_ids=reference_gene_ids or {}
            symbols={}; ids={}
            for i,symbol in enumerate(target.var_names.astype(str)): symbols.setdefault(symbol,[]).append(i)
            if "gene_ids" in target.var:
                for i,gid in enumerate(target.var.gene_ids.astype(str)): ids.setdefault(gid,[]).append(i)
            columns=[]; rows=[]; used=[]
            matrix=sparse.csc_matrix(target.X)
            for gene in map(str,genes):
                gid=reference_gene_ids.get(gene)
                idx=ids.get(gid,[]) if gid else []
                method="training_ensembl_id"
                if not idx: idx=symbols.get(gene,[]); method="exact_symbol_fallback"
                if not idx: method="explicit_zero_fill"
                used.extend(idx)
                columns.append(sparse.csr_matrix(matrix[:,idx].sum(axis=1)) if idx else sparse.csr_matrix((target.n_obs,1),dtype=np.float32))
                rows.append({"model_gene":gene,"training_ensembl_id":gid,"mapping_method":method,"resolved":bool(idx)})
            if len(used)!=len(set(used)): raise ValueError("Target feature reused across model genes; reconcile mapping")
            panel=sparse.hstack(columns,format="csr",dtype=np.float32)
            totals=np.asarray(panel.sum(axis=1)).ravel()
            if (totals<=0).any(): raise ValueError("Cell has no counts over the frozen model panel")
            panel=sparse.diags((1e4/totals).astype(np.float32))@panel
            panel.data=np.log1p(panel.data)
            return panel,pd.DataFrame(rows)

        def external_probability(bundle, target, batch_size=BATCH_SIZE):
            genes=list(map(str,bundle["genes"]))
            reference=M05_GENE_IDS if bundle is m05_bundle else None
            panel,mapping=resolve_external_panel(target,genes,reference)
            resolved=mapping.resolved.to_numpy(bool)
            coefficients=np.abs(lr_coefficients(bundle["model"]))
            probabilities=np.concatenate([bundle["model"].predict_proba(panel[i:i+batch_size])[:,1]
                                          for i in range(0,target.n_obs,batch_size)])
            audit={"n_model_genes":len(genes),"n_resolved":int(resolved.sum()),
                "feature_coverage":float(resolved.mean()),
                "zero_filled_abs_coefficient_fraction":float(coefficients[~resolved].sum()/coefficients.sum()),
                "n_zero_filled":int((~resolved).sum()),
                "zero_filled_genes":";".join(mapping.loc[~resolved,"model_gene"]),
                "mapping_methods":mapping.mapping_method.value_counts().to_dict()}
            return probabilities,audit

        def compose_end_to_end_calls(p_T, p_M05, gate_audit, m05_audit,
                                     threshold_T=gate_threshold_T,
                                     threshold_NK=gate_threshold_NK,
                                     threshold_M05=M05_THRESHOLD):
            if gate_audit["feature_coverage"] < MIN_FEATURE_COVERAGE:
                return np.repeat("abstain/out-of-domain",len(p_T))
            if m05_audit["feature_coverage"] < MIN_FEATURE_COVERAGE:
                return np.where(p_T<=threshold_NK,"predicted NK","abstain/out-of-domain")
            return np.select(
                [p_T <= threshold_NK,
                 (p_T >= threshold_T) & (p_M05 >= threshold_M05),
                 (p_T >= threshold_T) & (p_M05 < threshold_M05)],
                ["predicted NK", "predicted gdT", "predicted abT"],
                default="abstain/out-of-domain",
            )
        """),
        md(r"""
        ## External validation

        The table below is an adapter registry, not a licence to substitute one
        dataset for another. Every adapter must preserve exact original barcodes,
        source labels, donor/sample fields and raw-count provenance. Missing input
        produces an actionable error rather than a silent fallback.
        """),
        code(r"""
        external_registry = pd.DataFrame([
            {"dataset": "GSE199411", "role": "NK-enriched negative-arm stress test",
             "training_allowed": False,
             "expected_input": "/data/users/kaelynwen/gdT_project/GSE199411_NK/GSE199411_RAW/GSM5972359_filtered_feature_bc_matrix.h5"},
            {"dataset": "GSE130430", "role": "independent sorted-NK negative arm",
             "training_allowed": False,
             "expected_input": "/data/users/kaelynwen/gdT_project/GSE130430"},
            {"dataset": "GSE149356", "role": "γδT-positive receptor-evidence arm",
             "training_allowed": False,
             "expected_input": "/data/users/kaelynwen/Subtype_evaluation/External_1"},
            {"dataset": "PANGEA", "role": "multi-study atlas deployment",
             "training_allowed": False,
             "expected_input": "/data/users/kaelynwen/gdT_project/pangea"},
            {"dataset": "15-organ", "role": "cross-organ atlas deployment",
             "training_allowed": False,
             "expected_input": "/data/users/kaelynwen/gdT_project"},
        ])
        if external_registry.training_allowed.any():
            raise AssertionError("An external dataset was accidentally authorised for training")
        external_registry_path = TABLE_DIR / "external_only_dataset_registry.csv"
        atomic_csv(external_registry, external_registry_path)
        record_output(external_registry_path, "parameter", "External-only dataset roles and expected paths")
        display(external_registry)
        """),
        md(r"""
        ### GSE199411 end-to-end deployment

        This is the first negative-arm stress test because the existing notebook
        already established the exact 6,525 author post-QC cells and their four
        ADTs. The cell-level evidence file contains M05/M02 results, but the gate
        requires raw GEX counts; therefore the original 10x H5 is re-read and
        joined by exact barcode.

        The author-defined NK label is used **only after prediction** to estimate
        apparent NK false-positive rates before and after gating.
        """),
        code(r"""
        GSE199411_ROOT = Path("/data/users/kaelynwen/gdT_project/GSE199411_NK")
        if not GSE199411_ROOT.exists():
            local_candidate = Path("/Users/kaelynwen/Desktop/FYP_YHH/gdT_project/GSE199411_NK")
            if local_candidate.exists():
                GSE199411_ROOT = local_candidate
        GSE199411_H5 = GSE199411_ROOT / "GSE199411_RAW/GSM5972359_filtered_feature_bc_matrix.h5"
        GSE199411_META = GSE199411_ROOT / "GSE199411_sce_filtered_cell_metadata.csv"
        GSE199411_PRIOR = PROJECT / "predictions/GSE199411_NK_M05_M02_cell_level_evidence.parquet"

        missing = [str(path) for path in (GSE199411_H5, GSE199411_META, GSE199411_PRIOR)
                   if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "GSE199411 adapter inputs missing. Complete notebook 20 first and verify:\n- "
                + "\n- ".join(missing)
            )

        gse199411 = sc.read_10x_h5(GSE199411_H5, gex_only=True)
        gse199411.obs_names = gse199411.obs_names.astype(str)
        gse_meta = pd.read_csv(GSE199411_META, dtype={"barcode": str})
        if gse_meta.barcode.duplicated().any():
            raise ValueError("GSE199411 author metadata contains duplicate barcodes")
        positions = pd.Index(gse199411.obs_names).get_indexer(gse_meta.barcode)
        if (positions < 0).any():
            raise ValueError("Some author post-QC GSE199411 barcodes are absent from the H5")
        gse199411 = gse199411[positions].copy()
        gse199411.obs = gse_meta.set_index("barcode").loc[gse199411.obs_names].copy()

        p_T, gate_external_audit = external_probability(gate_bundle, gse199411)
        p_M05, m05_external_audit = external_probability(m05_bundle, gse199411)
        display(pd.DataFrame([{"component":"lineage_gate",**gate_external_audit},{"component":"M05",**m05_external_audit}]))
        end_to_end = compose_end_to_end_calls(
            p_T, p_M05, gate_external_audit, m05_external_audit
        )
        # Prior notebook-20 columns describe controlled_M05, not optimized_M05.
        prior = pd.read_parquet(GSE199411_PRIOR)
        barcode_column = next((c for c in ("barcode", "cell_barcode", "barcode_raw")
                               if c in prior), None)
        if barcode_column is None:
            raise KeyError("Prior GSE199411 evidence lacks a barcode column")
        result_199411 = gse_meta.copy()
        result_199411["p_T_lineage_gate"] = p_T
        result_199411["p_gdT_M05_recomputed"] = p_M05
        result_199411["end_to_end_call"] = end_to_end
        result_199411 = result_199411.merge(
            prior, left_on="barcode", right_on=barcode_column, how="left",
            validate="one_to_one", suffixes=("", "_prior"),
        )
        if result_199411["p_gdT_M05"].isna().any(): raise ValueError("Unmatched notebook-20 M05 reference prediction")
        controlled_bundle=joblib.load(PROJECT/"models/controlled_M05.joblib")
        controlled_panel,_=resolve_external_panel(gse199411,controlled_bundle["genes"],M05_GENE_IDS)
        controlled_probability=controlled_bundle["model"].predict_proba(controlled_panel)[:,1]
        if not np.allclose(controlled_probability,result_199411.p_gdT_M05,rtol=1e-5,atol=1e-6):
            raise ValueError("controlled_M05 no longer reproduces notebook 20; inspect mapping and model hash")
        print("controlled_M05 reproduces notebook 20. The before/after comparison below uses optimized_M05 on both sides.")
        result_path = PREDICTION_DIR / "GSE199411_gate_then_frozen_M05_cell_level.parquet"
        atomic_parquet(result_199411, result_path)
        record_output(result_path, "predictions", "GSE199411 gate then unchanged M05 calls")
        display(pd.Series(end_to_end).value_counts().rename("n_cells").to_frame())
        """),
        md(r"""
        ## GSE199411 receptor-RNA interpretation

        `TRDC detected in 80.8%`, `median TRDC = 3 UMI in M05+ versus 1 UMI in
        M05−`, and `TRDC detected in 75.2% of ILC and 27.8% of CD34+` **do not
        establish that the author misannotated most NK cells**.

        Why? A lineage-defining transcript should be coherent with other
        T-lineage/receptor evidence and should not appear broadly at tiny counts
        in biologically incompatible compartments. Widespread one-to-few-UMI
        signal in ILC and CD34+ is a warning for ambient RNA, barcode leakage,
        doublets, or gene-mapping artefact. It makes `TRDC >= 1` an unsuitable
        ground truth. The correct inference is:

        - high M05 in author NK is an **apparent false positive before gating**;
        - cells remaining T-eligible after gating are candidates for annotation
          impurity or rare contaminating T cells;
        - without matched TCRγδ protein or productive paired TRG+TRD, none of
          these cells can be called definitively γδT from receptor RNA alone.
        """),
        code(r"""
        receptor_columns = [
            "TRDC_count", "TRGC1_count", "TRGC2_count", "CD3D_count",
            "CD3E_count", "TRAC_count", "TRBC1_count", "TRBC2_count",
        ]
        available_receptor_columns = [c for c in receptor_columns if c in result_199411]
        receptor_summary_rows = []
        for call, frame in result_199411.groupby("end_to_end_call", observed=True):
            for column in available_receptor_columns:
                values = pd.to_numeric(frame[column], errors="coerce").fillna(0)
                receptor_summary_rows.append({
                    "end_to_end_call": call, "measurement": column,
                    "n_cells": len(frame), "fraction_detected": float(values.gt(0).mean()),
                    "median_count": float(values.median()), "mean_count": float(values.mean()),
                })
        receptor_summary = pd.DataFrame(receptor_summary_rows)
        receptor_summary_path = TABLE_DIR / "GSE199411_receptor_RNA_by_end_to_end_call.csv"
        atomic_csv(receptor_summary, receptor_summary_path)
        record_output(receptor_summary_path, "table", "Descriptive receptor RNA after lineage gating")
        display(receptor_summary)
        """),
        code(r"""
        before_positive = result_199411.p_gdT_M05_recomputed.ge(M05_THRESHOLD)
        after_positive = result_199411.end_to_end_call.eq("predicted gdT")
        comparison = pd.DataFrame({
            "analysis": ["optimized_M05 alone", "T/NK gate then optimized_M05"],
            "n_positive": [int(before_positive.sum()), int(after_positive.sum())],
            "n_total": [len(result_199411), len(result_199411)],
            "positive_fraction": [float(before_positive.mean()), float(after_positive.mean())],
        })
        comparison_path = TABLE_DIR / "GSE199411_apparent_gdT_fraction_before_after_gate.csv"
        atomic_csv(comparison, comparison_path)
        record_output(comparison_path, "table", "GSE199411 apparent gdT fraction before and after gate")

        fig, ax = plt.subplots(figsize=(4.8, 3.4))
        ax.bar(comparison.analysis, comparison.positive_fraction, color=["#999999", "#0072B2"],
               edgecolor="none", linewidth=0)
        ax.set(ylabel="Fraction of evaluated cells", xlabel="",
               title="Apparent γδT calls before and after lineage gating", ylim=(0, 1))
        ax.tick_params(axis="x", rotation=20)
        for label in ax.get_xticklabels():
            label.set_ha("right")
        sns.despine(ax=ax)
        fig.tight_layout()
        save_publication_figure(fig, "GSE199411_apparent_gdT_fraction_before_after_gate",
                                "Apparent gdT fraction before versus after lineage gating")
        plt.show()
        display(comparison)
        """),
        md(r"""
        ### Remaining external adapters

        Run the exact same frozen gate and M05 on each target. For PANGEA and
        15-organ, always report the denominator per source dataset, donor and
        organ; a pooled fraction can be dominated by one study. For GSE149356,
        report sensitivity separately in cells with processed/productive TRD,
        processed/productive TRG and paired TRG+TRD. For GSE130430, report NK FPR
        by donor, not only pooled cells.

        The adapter below deliberately accepts only an AnnData with raw counts and
        a declared donor column. It never reads target labels until predictions
        are complete.
        """),
        code(r"""
        def deploy_frozen_hierarchy(target, dataset, donor_column):
            if dataset not in EXTERNAL_ONLY_DATASETS:
                raise ValueError(f"Dataset is not registered external-only: {dataset}")
            if donor_column not in target.obs:
                raise KeyError(f"{dataset}: missing donor column {donor_column}")
            if target.raw is not None:
                raise ValueError(
                    f"{dataset}: pass an explicit raw-count AnnData; do not rely on .raw implicitly"
                )
            p_T, gate_audit = external_probability(gate_bundle, target)
            p_M05, m05_audit = external_probability(m05_bundle, target)
            calls = compose_end_to_end_calls(p_T, p_M05, gate_audit, m05_audit)
            predictions = pd.DataFrame({
                "dataset": dataset, "donor": target.obs[donor_column].astype(str).to_numpy(),
                "cell_id": target.obs_names.astype(str), "p_T_lineage_gate": p_T,
                "p_gdT_M05": p_M05, "end_to_end_call": calls,
            })
            audit = pd.DataFrame([
                {"dataset": dataset, "component": "lineage_gate", **gate_audit},
                {"dataset": dataset, "component": "M05", **m05_audit},
            ])
            return predictions, audit

        # Examples after loading exact raw-count objects:
        # pangea_predictions, pangea_audit = deploy_frozen_hierarchy(pangea_raw, "PANGEA", "donor")
        # organ_predictions, organ_audit = deploy_frozen_hierarchy(organ15_raw, "15-organ", "donor")
        # gse149356_predictions, gse149356_audit = deploy_frozen_hierarchy(gse149356_raw, "GSE149356", "donor")
        # gse130430_predictions, gse130430_audit = deploy_frozen_hierarchy(gse130430_raw, "GSE130430", "donor")
        """),
        md(r"""
        ### Required reporting table for every external dataset

        Do not publish only a UMAP. For each dataset × donor × organ/source label,
        report:

        - evaluated cells and feature coverage;
        - fraction `predicted NK`, `predicted gdT`, `predicted abT`, and
          `abstain/out-of-domain`;
        - the original M05-positive fraction before gating;
        - the retained M05-positive fraction after gating;
        - receptor/protein support where it truly exists;
        - donor-level uncertainty (bootstrap donors when there are enough donors).

        A dramatic reduction after gating supports out-of-domain M05 use. A
        reproducible retained subset with coherent T-lineage and independent
        receptor/protein evidence supports annotation impurity or overlooked γδT.
        """),
        code(r"""
        def summarize_external_predictions(predictions, group_columns):
            required = set(group_columns) | {"end_to_end_call", "cell_id", "p_gdT_M05"}
            missing = required - set(predictions.columns)
            if missing:
                raise KeyError(f"Prediction table missing: {sorted(missing)}")
            counts = (
                predictions.groupby(group_columns + ["end_to_end_call"], observed=True)
                .size().rename("n_cells").reset_index()
            )
            denominators = (
                predictions.groupby(group_columns, observed=True).size()
                .rename("denominator").reset_index()
            )
            counts = counts.merge(denominators, on=group_columns, validate="many_to_one")
            counts["fraction"] = counts.n_cells / counts.denominator
            return counts
        """),
        md(r"""
        ## Limitations and interpretation

        1. **The gate is not new γδT ground truth.** It only prevents a T-cell
           subtype model from being applied indiscriminately to NK-like cells.
        2. **GSE164378 source labels are reference annotations.** Same-study
           multimodal labels reduce dataset confounding, but are not perfect.
           Donor-blocked external sorted-NK validation remains mandatory.
        3. **Abstention is a scientific result.** A cell outside the training
           support should not be forced into αβT, γδT, or NK merely to make a
           colourful UMAP.
        4. **TRDC RNA is not receptor proof.** Low isolated receptor-gene counts,
           especially when widespread in ILC/CD34+ cells, can reflect ambient RNA
           or barcode contamination. Stronger evidence is paired productive
           TRG+TRD, TCRγδ protein, or validated sorting.
        5. **NC-level claim boundary.** The strongest defensible claim is a
           hierarchical, abstaining classifier with donor-blocked source
           validation and independent positive/negative arms. Do not claim that
           PANGEA or 15-organ annotations are wrong from UMAP proximity alone.

        ### Decision rule

        - If strict independent NK donors are mostly `predicted NK` after gating
          and GSE149356 receptor-supported γδT remain T-eligible/M05+, the original
          M05 is useful **within T cells** and the main problem was deployment.
        - If strict NK donors remain T-eligible and M05+ across donors, the model
          still contains an NK/cytotoxic shortcut and must be redesigned.
        - If true γδT are rejected by the gate, the gate is too conservative or
          source-biased; do not deploy it until that sensitivity loss is fixed.
        """),
        md(r"""
        ## Output manifest and exact resume command

        The final cell records versions, hashes, thresholds and every generated
        output. If interrupted, reopen this notebook and use **Run All**; validated
        `outer_donor × candidate_C` checkpoints are reused.

        Non-interactive Woody command:

        ```bash
        cd /data/users/kaelynwen/gdT_project/gdt_lr_12model
        jupyter nbconvert --to notebook --execute \
          --ExecutePreprocessor.timeout=-1 \
          --output 21_T_NK_lineage_gate_and_frozen_M05_deployment.executed.ipynb \
          notebooks/21_T_NK_lineage_gate_and_frozen_M05_deployment.ipynb
        ```
        """),
        code(r"""
        software = {
            "python": sys.version.split()[0], "numpy": np.__version__,
            "pandas": pd.__version__, "anndata": ad.__version__,
            "scanpy": sc.__version__, "joblib": joblib.__version__,
        }
        run_manifest = {
            "analysis_id": ANALYSIS_ID,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "training_reference": TRAINING_REFERENCE,
            "external_only_datasets": EXTERNAL_ONLY_DATASETS,
            "run_signature": run_signature, "seed": SEED,
            "source_sha256": source_hash,
            "frozen_M05_sha256": M05_SHA256,
            "M05_threshold": M05_THRESHOLD,
            "lineage_gate_threshold_T": gate_threshold_T,
            "lineage_gate_threshold_NK": gate_threshold_NK,
            "software": software,
            "models_retrained": {"lineage_gate": True, "M05": False},
            "outputs": output_records,
            "limitations": [
                "T/NK source annotations are not receptor ground truth",
                "external donor-level validation is required",
                "low-count TRDC RNA is not sufficient for gamma-delta lineage",
                "whole-atlas cells outside T/NK support should abstain",
            ],
        }
        manifest_path = TABLE_DIR / "run_manifest.json"
        atomic_json(run_manifest, manifest_path)
        record_output(manifest_path, "manifest", "Parameters, hashes, versions and limitations")
        output_manifest_path = TABLE_DIR / "output_manifest.csv"
        record_output(output_manifest_path, "manifest", "Complete output artifact manifest")
        atomic_csv(pd.DataFrame(output_records), output_manifest_path)
        print("Run manifest:", manifest_path)
        print("Output manifest:", output_manifest_path)
        print("Checkpoint directory:", CHECKPOINT_DIR)
        """),
    ]


    # Reuse the exact deployed feature mapping and decision functions.
    frozen_functions = next(c.source for c in cells if c.cell_type == "code" and "def resolve_external_panel" in c.source)
    cells += [md(r"""
        ## External label evidence and frozen-gate retention audit

        Run this section from its setup cell without refitting models. All comparisons
        use frozen optimized_M05 and the frozen LR gate; no external threshold tuning.
        This short audit saves results in notebook outputs (one cell per donor).
        Rerun the setup and subsequent audit cells to reproduce; no cache is reused.
        Input hashes, versions, thresholds and mapping audits are displayed below.
        Exact count tables are used because denominators and evidence restrictions
        are central; no new figures are generated by this section.

        **GSE199411 provenance:** one healthy bone-marrow donor, magnetic NK enrichment,
        RNA clustering and four CITE-seq antibodies (CD56, CXCR6, CD117, CD34).
        The publication reports removal of 21 low-quality cells and 454 doublets.
        These support author annotation, but the four-protein panel does not establish
        per-cell CD3/TCR lineage. No paired V(D)J library is described for this dataset.
        Separate flow experiments are not barcode-matched truth for these cells.
        Source: https://doi.org/10.3389/fimmu.2022.1044398 (Methods; Figure 1).
        Supplementary Figure S1 and Table S2 were inspected: enrichment QC and
        the four-antibody CITE-seq panel do not supply barcode-matched CD3/TCR status.
        Figure S4 CD3-negative gating concerns separate spectral-cytometry samples.
        Supplement accessed through https://www.ebi.ac.uk/europepmc/webservices/rest/PMC9730327/supplementaryFiles .

        **GSE149356 provenance:** sorted CD3+ TCR-gamma-delta+ cells with paired
        transcriptome and TRG/TRD libraries. The supplied processed TCR CSV lacks
        productive/high-confidence/full-length flags and contig read/UMI QC.
        Exact TRD+TRG calls with both CDR3s provide receptor support, not independently
        revalidated productive-pair ground truth. Multi-chain calls are excluded from
        that restricted denominator and reported separately. Unmatched cells are not NK.
        Sources: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM4634904 and
        https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE149356 .
        PB1:HC1 and PB2:HC2 mappings follow the project input contract. Only the
        terminal -1 suffix is removed; donor prefixes are retained before matching.

        RNA marker comparisons are descriptive and potentially circular: genes used
        by either classifier are explicitly flagged. No marker cutoff relabels cells.
        One NK donor and two adult gamma-delta donors cannot establish general accuracy.
        """), code(_clean(r"""
        from pathlib import Path
        import hashlib, json, platform, importlib.metadata
        import numpy as np
        import pandas as pd
        import scanpy as sc
        import anndata as ad
        import joblib
        from scipy import sparse, io
        from IPython.display import display
        PROJECT = Path('/data/users/kaelynwen/gdT_project/gdt_lr_12model')
        if not PROJECT.exists():
            PROJECT = Path('/Users/kaelynwen/Desktop/FYP_YHH/gdT_project/gdt_lr_12model')
        _external = Path('/data/users/kaelynwen/Subtype_evaluation/External_1')
        if not _external.exists():
            _external = Path('/Users/kaelynwen/Desktop/FYP_YHH/gdT_project/Subtype_evaluation/External_1')
        def _audit_hash(path):
            h=hashlib.sha256()
            with open(path,'rb') as f:
                for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
            return h.hexdigest()
        _gate_path=PROJECT/'models/21_T_NK_lineage_gate_and_frozen_M05_deployment/21_T_NK_lineage_gate_and_frozen_M05_deployment_frozen_lineage_gate.joblib'
        _m05_path=PROJECT/'models/optimized_M05.joblib'
        _registry=PROJECT/'results/tables/optimized_fit_registry.csv'
        gate_bundle=joblib.load(_gate_path)
        m05_bundle=joblib.load(_m05_path)
        gate_threshold_T=gate_bundle['threshold_T']; gate_threshold_NK=gate_bundle['threshold_NK']
        M05_THRESHOLD=float(pd.read_csv(_registry).query('model_id == "M05"').iloc[0]['threshold'])
        MIN_FEATURE_COVERAGE=0.95; BATCH_SIZE=2048
        _ref=PROJECT/'manifests/Song_D1_cohorts.h5ad'
        _song=sc.read_h5ad(_ref,backed='r')
        assert _song.var_names.is_unique
        M05_GENE_IDS=dict(zip(_song.var_names.astype(str),_song.var.gene_ids.astype(str)))
        _song.file.close()
        _tcr_path=_external/'TCR_Label/GSE149356_Processed_TCR.csv.gz'
        _tcr=pd.read_csv(_tcr_path)
        assert _tcr.barcode.is_unique and _tcr.barcode.notna().all()
        assert (_tcr.barcode.str.split('_').str[0] == _tcr.donor).all()
        _audit_manifest={'schema':1,'models_refitted':False,'thresholds_tuned':False,
            'threshold_T':gate_threshold_T,'threshold_NK':gate_threshold_NK,'threshold_M05':M05_THRESHOLD,
            'python':platform.python_version(),
            'packages':{k:importlib.metadata.version(k) for k in ['numpy','pandas','scipy','scanpy','anndata','scikit-learn']},
            'inputs':{str(f):_audit_hash(f) for f in [_gate_path,_m05_path,_registry,_ref,_tcr_path,PROJECT/'build_21_T_NK_lineage_gate_and_frozen_M05_deployment.py']},
            'source':str(PROJECT/'build_21_T_NK_lineage_gate_and_frozen_M05_deployment.py'),
            'completed_donors':[]}
        print(json.dumps(_audit_manifest,indent=2))
        print('Unavailable TCR QC columns:',[c for c in ['productive','high_confidence','full_length','umis','reads'] if c not in _tcr])
        """) + "\n" + frozen_functions), md("### GSE199411: paired comparison within author-defined NK cells"),code(r"""
        _pred_path=PROJECT/'predictions/21_T_NK_lineage_gate_and_frozen_M05_deployment/GSE199411_gate_then_frozen_M05_cell_level.parquet'
        _pred=pd.read_parquet(_pred_path)
        assert _pred.barcode.is_unique and len(_pred)==6525
        _nk=_pred.loc[_pred.author_defined_NK.eq(True)].copy()
        _nk['M05_positive']=_nk.p_gdT_M05_recomputed.ge(M05_THRESHOLD)
        assert len(_nk)==6266
        print('Frozen optimized M05; not the prior controlled-M05 probability column.')
        display(pd.crosstab(_nk.M05_positive,_nk.end_to_end_call))
        display(pd.crosstab(_nk.author_cell_type,_nk.M05_positive))
        display(_nk.groupby('M05_positive')[['author_total_counts','author_n_genes','author_mito_percent','CD56_TotalA_CLR','CXCR6_TotalA_CLR']].median().round(3))
        _raw_path=PROJECT.parent/'GSE199411_NK/GSE199411_RAW/GSM5972359_filtered_feature_bc_matrix.h5'
        _raw=sc.read_10x_h5(_raw_path,gex_only=True)
        assert _raw.obs_names.is_unique
        _x=sparse.csr_matrix(_raw[_nk.barcode.to_numpy(),:].X)
        _genes=_raw.var_names.astype(str).to_numpy()
        _tot=np.asarray(_x.sum(axis=1)).ravel()
        assert np.allclose(_tot,_nk.author_total_counts.to_numpy())
        _marker_rows=[]
        _marker_counts={}
        for gene in ['CD3D','CD3E','CD3G','TRDC','TRGC1','TRGC2','TRAC','KLRD1','FCER1G','TYROBP','NKG7','GNLY']:
            _idx=np.flatnonzero(_genes==gene)
            assert len(_idx)>0, gene
            _cnt=np.asarray(_x[:,_idx].sum(axis=1)).ravel(); _marker_counts[gene]=_cnt
            for positive in [False,True]:
                _mask=_nk.M05_positive.to_numpy()==positive
                _marker_rows.append({'gene':gene,'M05_positive':positive,'n':int(_mask.sum()),
                    'detected_percent':100*float((_cnt[_mask]>0).mean()),
                    'median_logCP10K':float(np.median(np.log1p(1e4*_cnt[_mask]/_tot[_mask]))),
                    'in_gate':gene in gate_bundle['genes'],'in_M05':gene in m05_bundle['genes']})
        display(pd.DataFrame(_marker_rows).round(3))
        _cd3=(_marker_counts['CD3D']>0)|(_marker_counts['CD3E']>0)|(_marker_counts['CD3G']>0)
        _gd=(_marker_counts['TRDC']>0)&((_marker_counts['TRGC1']>0)|(_marker_counts['TRGC2']>0))
        display(pd.crosstab(_nk.M05_positive,[_cd3,_gd],rownames=['M05_positive'],colnames=['any_CD3_RNA','TRDC_and_TRGC_RNA']))
        print('Co-detection is descriptive RNA evidence; not a lineage rule or productive receptor call.')
        print('One donor only. No per-cell doublet score in supplied metadata; depth is not a doublet diagnosis.')
        _audit_manifest['inputs'].update({str(f):_audit_hash(f) for f in [_pred_path,_raw_path]})
        del _raw,_x
        """),md("### GSE149356: frozen inference, donor-scoped matching and receptor evidence")]
    cells += [code(r"""
        _audit_results=[]
        def _audit_donor(sub, donor, prefix):
            folder=_external/sub
            def pick(suffix):
                files=list(folder.glob(prefix+suffix+'*'))
                assert len(files)==1,(suffix,files)
                return files[0]
            paths=[pick('matrix.mtx'),pick('features.tsv'),pick('barcodes.tsv')]
            X=sparse.csr_matrix(io.mmread(paths[0]).T,dtype=np.float32)
            var=pd.read_csv(paths[1],sep='\t',header=None)
            barcodes=pd.read_csv(paths[2],sep='\t',header=None)[0].astype(str)
            assert X.shape==(len(barcodes),len(var)) and barcodes.is_unique
            assert np.isfinite(X.data).all() and (X.data>=0).all() and np.allclose(X.data,np.round(X.data))
            assert barcodes.str.fullmatch('[ACGT]+-1').all()
            target=ad.AnnData(X,obs=pd.DataFrame(index=barcodes),var=pd.DataFrame({'gene_ids':var[0].to_numpy()},index=var[1].astype(str)))
            if prefix.endswith('_'):
                old_barcodes=pd.read_csv(next(folder.glob(prefix+'PB*barcodes*')),sep='\t',header=None)[0].astype(str)
                print('Alternative vs original barcodes:', {'shared':int(barcodes.isin(old_barcodes).sum()),'alternative_only':int((~barcodes.isin(old_barcodes)).sum()),'original_only':int((~old_barcodes.isin(barcodes)).sum())})
            keys=donor+'_'+barcodes.str.replace(r'-1$','',regex=True)
            tcr=_tcr.loc[_tcr.donor.eq(donor)].set_index('barcode')
            matched=keys.isin(tcr.index).to_numpy()
            joined=tcr.reindex(keys).reset_index(drop=True)
            assert len(joined)==target.n_obs and keys.is_unique
            paired=joined.chain_TRD.eq('TRD') & joined.chain_TRG.eq('TRG')
            for col in ['cdr3_TRD','cdr3_TRG','cdr3_nt_TRD','cdr3_nt_TRG']:
                paired &= joined[col].fillna('').str.len().gt(0)
            multi=joined.chain_TRD.eq('Multi')|joined.chain_TRG.eq('Multi')
            assert not (paired & multi).any()
            pT,aT=external_probability(gate_bundle,target)
            pM,aM=external_probability(m05_bundle,target)
            assert np.isfinite(pT).all() and np.isfinite(pM).all()
            assert ((pT>=0)&(pT<=1)&(pM>=0)&(pM<=1)).all()
            calls=compose_end_to_end_calls(pT,pM,aT,aM)
            counts=np.asarray(X.sum(axis=1)).ravel()
            genes=np.asarray((X>0).sum(axis=1)).ravel()
            mt=np.asarray(X[:,target.var_names.str.startswith('MT-')].sum(axis=1)).ravel()/np.maximum(counts,1)*100
            # Literal GEO sample QC, reported as sensitivity only; does not reproduce
            # author cluster trimming or the complete RDS analysis population.
            qc=(counts>=1500)&(counts<=6000)&(mt<=11)
            print(sub, json.dumps({'expression_cells':target.n_obs,'TCR_rows':len(tcr),'matched_TCR_rows':int(matched.sum()),'unmatched_TCR_rows':int((~tcr.index.isin(keys)).sum()),'no_matched_TCR':int((~matched).sum()),'Multi':int(multi.sum()),'TRD_TRG_with_CDR3':int(paired.sum()),'gate_mapping':{k:v for k,v in aT.items() if k!='zero_filled_genes'},'M05_mapping':{k:v for k,v in aM.items() if k!='zero_filled_genes'}}))
            # No hard calls if either model fails its previously frozen coverage rule.
            _audit_manifest['inputs'].update({str(f):_audit_hash(f) for f in paths})
            _audit_manifest.setdefault('GEO_downloads',{}).update({str(f):'https://www.ncbi.nlm.nih.gov/geo/download/?acc='+prefix.split('_')[0]+'&format=file&file='+f.name for f in paths if '_PB' not in f.name})
            if aT['feature_coverage']<MIN_FEATURE_COVERAGE or aM['feature_coverage']<MIN_FEATURE_COVERAGE:
                row={'donor':donor,'input_prefix':prefix,'denominator':'TRD + TRG with CDR3; Multi excluded','n':int(paired.sum()),'status':'not evaluable: frozen feature coverage rule failed','gate_feature_coverage':aT['feature_coverage'],'M05_feature_coverage':aM['feature_coverage']}
                _audit_results.append(row)
                _audit_manifest['completed_donors'].append(donor)
                display(pd.DataFrame([row]))
                print('Absent receptor genes:',[g for g in ['TRAC','TRBC1','TRBC2','TRDC','TRGC1','TRGC2','TRDV1','TRDV2','TRDV3'] if g not in target.var_names])
                print('Retention is unavailable, not zero. Probabilities under failed coverage are not interpreted.')
                return
            tables=[]
            for evidence,mask in [('all supplied expression cells',np.ones(len(keys),bool)),('matched processed TCR',matched),('TRD + TRG with CDR3; Multi excluded',paired.to_numpy()),('TRD + TRG with CDR3; literal GEO QC',paired.to_numpy()&qc)]:
                n=int(mask.sum()); before=pM>=M05_THRESHOLD; eligible=pT>=gate_threshold_T
                row={'donor':donor,'input_prefix':prefix,'denominator':evidence,'n':n,'T_eligible':int((mask&eligible).sum()),'predicted_NK':int((mask&(pT<=gate_threshold_NK)).sum()),'gate_abstain':int((mask&(pT>gate_threshold_NK)&(pT<gate_threshold_T)).sum()),'M05_positive_before_gate':int((mask&before).sum()),'predicted_gdT_after_gate':int((mask&(calls=='predicted gdT')).sum()),'M05_positive_lost_at_gate':int((mask&before&~eligible).sum())}
                assert row['T_eligible']+row['predicted_NK']+row['gate_abstain']==n
                assert row['M05_positive_before_gate']-row['predicted_gdT_after_gate']==row['M05_positive_lost_at_gate']
                tables.append(row)
            display(pd.DataFrame(tables))
            q=pd.DataFrame({'T_eligible':pT>=gate_threshold_T,'counts':counts,'genes':genes,'mito_percent':mt,'v_gene_TRD':joined.v_gene_TRD,'paired':paired})
            rejected=pd.DataFrame({'barcode':barcodes.to_numpy(),'donor':donor,'p_T_lineage_gate':pT,'p_gdT_M05_recomputed':pM,'end_to_end_call':calls,'v_gene_TRD':joined.v_gene_TRD,'author_total_counts':counts,'author_n_genes':genes,'mito_percent':mt})
            rejected=rejected.loc[paired.to_numpy() & (pT<gate_threshold_T)]
            print('Receptor-supported cells not T-eligible; retain for review, do not delete:')
            display(rejected)
            display(q[q.paired].groupby('T_eligible')[['counts','genes','mito_percent']].median().round(2))
            display(pd.crosstab(q.loc[q.paired,'v_gene_TRD'],q.loc[q.paired,'T_eligible']))
            _audit_manifest['inputs'].update({str(f):_audit_hash(f) for f in paths})
            _audit_manifest.setdefault('GEO_downloads',{}).update({str(f):'https://www.ncbi.nlm.nih.gov/geo/download/?acc='+prefix.split('_')[0]+'&format=file&file='+f.name for f in paths if '_PB' not in f.name})
            _audit_manifest['completed_donors'].append(donor)
            print('Completed donor input hashes:',json.dumps({str(f):_audit_manifest['inputs'][str(f)] for f in paths},indent=2))
            _audit_results.extend(tables)
        """),code("_audit_donor('PB1:HC1','HC1','GSM4498553_PB1')"),code("_audit_donor('PB2:HC2','HC2','GSM4498554_PB2')"),md("### Complete alternative triplets supplied by GEO\nThe PB-prefixed files fail the frozen feature coverage rule. The alternative triplets below are downloaded as a complete matrix/features/barcodes set from each same GEO sample, preserving the official filenames. They are evaluated separately, not merged with the older matrices. Model parameters and thresholds stay fixed."),code("_audit_donor('PB1:HC1','HC1','GSM4498553_')"),code("_audit_donor('PB2:HC2','HC2','GSM4498554_')"),code(r"""
        assert set(_audit_manifest['completed_donors'])=={'HC1','HC2'}
        assert _audit_hash(_gate_path)==_audit_manifest['inputs'][str(_gate_path)]
        assert _audit_hash(_m05_path)==_audit_manifest['inputs'][str(_m05_path)]
        _summary=pd.DataFrame(_audit_results)
        _summary=_summary.loc[_summary.input_prefix.str.endswith('_') & _summary.denominator.eq('TRD + TRG with CDR3; Multi excluded')].copy()
        _summary['gate_retention_percent']=100*_summary.T_eligible/_summary.n
        _summary['M05_before_percent']=100*_summary.M05_positive_before_gate/_summary.n
        _summary['M05_after_percent']=100*_summary.predicted_gdT_after_gate/_summary.n
        _summary['loss_of_M05_positives_percent']=100*_summary.M05_positive_lost_at_gate/_summary.M05_positive_before_gate
        display(_summary[['donor','n','gate_retention_percent','M05_before_percent','M05_after_percent','loss_of_M05_positives_percent']].round(3))
        print('These two donors support high receptor-supported T eligibility; NK specificity remains annotation-relative.')
        print('M05 already misses receptor-supported cells before gating. Do not attribute that entire loss to the gate.')
        print('The original PB-prefixed triplets are not evaluable under the frozen coverage rule. Their unavailable retention is not zero.')
        print('Audit complete; frozen model hashes unchanged. Manifest:')
        print(json.dumps(_audit_manifest,indent=2))
        """),md(r"""
        ### Interpretation limits
        Report donor-specific receptor-supported retention, never combine NK and
        gamma-delta cohorts to estimate overall accuracy or precision. The supplied
        matrices are not assumed to equal the author's final post-clustering population.
        Literal GEO QC is a sensitivity check; productive-chain QC and doublets remain
        incompletely verified. Receptor-negative/unmatched cells remain unresolved.
        If retention is poor, preserve rejected cells and do not deploy the gate as
        a destructive filter. Any future threshold/model revision belongs in source-only
        validation, followed by a new untouched external test. No TVAE training is run here.
        """)]

    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
            "analysis": {
                "id": "21_T_NK_lineage_gate_and_frozen_M05_deployment",
                "authoritative_generator": "build_21_T_NK_lineage_gate_and_frozen_M05_deployment.py",
                "models_retrained": {"lineage_gate": True, "M05": False},
            },
        },
    )
    nbf.write(notebook, output_path)
    return output_path


if __name__ == "__main__":
    target = build_notebook()
    print(target)
