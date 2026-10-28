#!/usr/bin/env python3
"""Generate the strict-inductive SV02 versus SV05 scVI-latent + LR notebook.

This generator is authoritative. Edit this file, then regenerate the notebook;
do not hand-edit the generated notebook JSON.
"""

from __future__ import annotations

from pathlib import Path
import textwrap

import nbformat as nbf


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "notebooks/12_sv02_sv05_strict_inductive_scvi_lr.ipynb"


def md(source: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(source).strip())


def code(source: str):
    return nbf.v4.new_code_cell(textwrap.dedent(source).strip())


cells = [
    md(r'''
    # 12 — SV02 versus SV05: strict-inductive scVI latent + LR

    ## tl;dr

    This is a **standalone experiment notebook**. It does not require notebooks
    00–11 to be rerun, but it requires their immutable manifests, fixed feature
    files, and completed M02/M05 prediction artifacts.

    The confirmatory comparison is:

    - **SV02:** scVI latent representation + weighted LR, using the M02 H1
      TCR-free feature definition.
    - **SV05:** the identical pipeline and frozen hyperparameters, using the
      same H1 backbone plus exactly the nine fixed M05 TCR genes.

    SPL expression and labels are never used to fit scVI, choose scVI/LR
    hyperparameters, calibrate probabilities, or select decision thresholds.
    SPL is opened only after Song donor-LODO OOF probabilities and operating
    points have been frozen.

    **Completed-run result:** on 5,046 SPL external-gold cells (378 γδ T
    positives), AUPRC was M02=0.423, M05=0.943, SV02=0.641, and SV05=0.685.
    Thus scVI improved the TCR-free model (`SV02 > M02`) but did not improve the
    TCR-inclusive model (`SV05 < M05`). SV02/SV05 probabilities were poorly
    calibrated on SPL, so their frozen thresholds are not transport-stable.

    **Remote project:** `/data/users/kaelynwen/gdT_project/gdt_lr_12model`
    '''),
    md(r'''
    ## Context & Methods

    ### Key assumptions and confirmatory contract

    1. M02 and SV02 use the same H1 TCR-free definition; M05 and SV05 use that
       backbone plus exactly nine prespecified TCR genes.
    2. The supervised label regime is M02/M05's `L2_paired_plus_TRD_equal`, with
       the existing donor-balanced class weights and `lambda_TRD=1.0`.
    3. Hyperparameters are selected on **SV02 only** inside nested Song donor
       validation, then frozen unchanged for SV05. This prevents differential
       tuning from masquerading as a TCR-feature effect.
    4. Every outer held-out donor is excluded from both scVI representation
       fitting and LR fitting. The final external models see Song only.
    5. scVI is trained without a donor batch covariate so its frozen encoder can
       be applied inductively to unseen SPL donors. No scArches/query adaptation
       is performed in this confirmatory notebook.
    6. TRDV1/TRDV2 SPL strata are explicitly named **RNA-expression proxies**;
       they are not receptor V-gene truth and are never used for training.
    7. SPL-label-selected curve metrics and bootstrap contrasts describe model
       discrimination. They are not deployment thresholds.
    '''),
    md(r'''
    ### Woody/NSC environment

    Use a Python 3.10–3.12 environment on a GPU compute node. Do not install the
    PyTorch/scvi stack repeatedly inside an allocated notebook kernel.

    ```bash
    cd /data/users/kaelynwen/gdT_project/gdt_lr_12model
    python -m pip install -r requirements-scvi.txt
    jupyter lab notebooks/12_sv02_sv05_strict_inductive_scvi_lr.ipynb
    ```

    Recommended allocation: one GPU, 8 CPU cores, at least 32 GB RAM, and enough
    wall time for nested donor validation. The notebook checkpoints inner scores
    and outer predictions so a completed unit is not recomputed.

    For the final clean, auditable execution inside the allocated GPU job:

    ```bash
    cd /data/users/kaelynwen/gdT_project/gdt_lr_12model
    python -m jupyter nbconvert --execute --to notebook --inplace \
      --ExecutePreprocessor.timeout=-1 \
      notebooks/12_sv02_sv05_strict_inductive_scvi_lr.ipynb
    ```
    '''),
    code(r'''
    # 1. Imports, deterministic execution, and NSC-safe project paths
    from pathlib import Path
    from collections import Counter
    import gc, hashlib, json, os, platform, random, sys, time, warnings, zlib

    missing = []
    for module_name in ["scvi", "torch", "anndata", "scanpy", "pandas",
                        "numpy", "scipy", "sklearn", "joblib", "pyarrow"]:
        try:
            __import__(module_name)
        except Exception as exc:
            missing.append(f"{module_name}: {exc!r}")
    if missing:
        raise RuntimeError(
            "Missing/broken packages. Activate the Woody scVI environment built "
            "from requirements-scvi.txt, restart the kernel, and rerun.\n" +
            "\n".join(missing)
        )

    import anndata as ad
    import joblib
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import numpy as np
    import pandas as pd
    import scanpy as sc
    import scipy.sparse as sp
    import scvi
    import seaborn as sns
    import torch
    from scipy.stats import rankdata
    from sklearn.calibration import calibration_curve
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        average_precision_score, brier_score_loss, confusion_matrix,
        precision_recall_curve, roc_auc_score, roc_curve,
    )
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    REMOTE_PROJECT = Path("/data/users/kaelynwen/gdT_project/gdt_lr_12model")
    PROJECT = Path(os.environ.get("GDT_LR_PROJECT", REMOTE_PROJECT)).resolve()
    if not PROJECT.exists():
        candidate = Path.cwd().resolve()
        PROJECT = candidate.parent if candidate.name == "notebooks" else candidate
    if str(PROJECT).startswith("/home/kaelynwen"):
        raise RuntimeError("NSC project work must use /data/users/kaelynwen, not /home/kaelynwen")
    sys.path.insert(0, str(PROJECT / "src"))
    from gdt_lr_utils import (
        donor_balanced_weights, get_counts, make_training_table,
        select_hvgs, sha256_file,
    )

    SEED = 20260806
    N_THREADS = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
    for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"]:
        os.environ[key] = str(N_THREADS)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    scvi.settings.seed = SEED
    torch.set_num_threads(N_THREADS)

    ACCELERATOR = "gpu" if torch.cuda.is_available() else "cpu"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TABLE_DIR = PROJECT / "results/tables/SV02_SV05_scVI_LR"
    FIGURE_DIR = PROJECT / "results/figures/SV02_SV05_scVI_LR"
    MODEL_DIR = PROJECT / "models/SV02_SV05_scVI_LR"
    PREDICTION_DIR = PROJECT / "predictions/SV02_SV05_scVI_LR"
    CHECKPOINT_DIR = PROJECT / "checkpoints/SV02_SV05_scVI_LR"
    for directory in [TABLE_DIR, FIGURE_DIR, MODEL_DIR, PREDICTION_DIR, CHECKPOINT_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    print("PROJECT:", PROJECT)
    print("accelerator:", ACCELERATOR, "device:", DEVICE,
          "CPUs:", N_THREADS, "SLURM_JOB_ID:", os.environ.get("SLURM_JOB_ID", "interactive"))
    software_versions = {
        "python": platform.python_version(), "scvi_tools": scvi.__version__,
        "torch": torch.__version__, "anndata": ad.__version__, "scanpy": sc.__version__,
        "pandas": pd.__version__, "numpy": np.__version__,
        "sklearn": sys.modules["sklearn"].__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    (TABLE_DIR / "software_versions.json").write_text(
        json.dumps(software_versions, indent=2), encoding="utf-8"
    )
    '''),
    code(r'''
    # 2. Registered analysis parameters — edit only before the first confirmatory run
    LABEL_REGIME = "L2_paired_plus_TRD_equal"
    LAMBDA_TRD = 1.0
    SPL_GOLD_DONORS = ("A29", "A31", "A35", "A36", "A37")
    TARGET_DONOR = "A36"
    MAX_EPOCHS = 200
    EARLY_STOPPING_PATIENCE = 20
    BATCH_SIZE = 256
    LEARNING_RATE = 1e-3
    ENCODE_BATCH_SIZE = 256
    BOOTSTRAP_REPLICATES = 2000
    RESUME = True

    # Compact prespecified grid. Configuration complexity is used only for ties.
    SCVI_GRID = [
        {"config_id": "Z10_L1_H128", "n_latent": 10, "n_layers": 1, "n_hidden": 128},
        {"config_id": "Z20_L1_H128", "n_latent": 20, "n_layers": 1, "n_hidden": 128},
        {"config_id": "Z20_L2_H128", "n_latent": 20, "n_layers": 2, "n_hidden": 128},
    ]
    C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]
    OPERATING_RULES = [
        ("Song_OOF_registered_maxF1", "max_f1", None),
        ("Song_OOF_FPR_le_0.007", "fpr_at_most", 0.007),
        ("Song_OOF_recall_ge_0.800", "recall_at_least", 0.800),
        ("Song_OOF_recall_ge_0.861", "recall_at_least", 0.861),
        ("Song_OOF_recall_ge_0.900", "recall_at_least", 0.900),
        ("Song_OOF_precision_ge_0.800", "precision_at_least", 0.800),
        ("Song_OOF_precision_ge_0.900", "precision_at_least", 0.900),
        ("Song_OOF_maxF2", "max_f2", None),
    ]

    run_contract = {
        "seed": SEED, "threads": N_THREADS, "label_regime": LABEL_REGIME,
        "lambda_TRD": LAMBDA_TRD, "spl_gold_donors": list(SPL_GOLD_DONORS),
        "scvi_grid": SCVI_GRID, "C_grid": C_GRID, "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
        "accelerator": ACCELERATOR,
        "target_expression_used_for_training": False,
        "target_labels_used_for_training_or_selection": False,
        "scarches_or_query_adaptation_used": False,
        "hyperparameters_selected_on": "SV02 nested Song donor CV only",
        "SV05_hyperparameters": "frozen from matched SV02 outer fold",
    }
    (TABLE_DIR / "registered_run_contract.json").write_text(
        json.dumps(run_contract, indent=2), encoding="utf-8"
    )
    display(pd.DataFrame(SCVI_GRID))
    '''),
    md(r'''
    ## Data

    The following cell opens immutable Song/SPL cohort objects and fixed feature
    files. It checks hashes, cell-ID uniqueness, raw-count validity, label/donor
    support, and the exact M02/M05 panel relationship before any model fitting.
    '''),
    code(r'''
    # 3. Load and validate immutable inputs
    INPUTS = {
        "song": PROJECT / "manifests/Song_D1_cohorts.h5ad",
        "spl": PROJECT / "manifests/CrossAtlas_D2_SPL_cohorts.h5ad",
        "h1": PROJECT / "features/H1_song_inductive_TCRfree_backbone.csv",
        "h1_metadata": PROJECT / "features/H1_metadata.json",
        "tcr": PROJECT / "features/prespecified_TCR_panel.csv",
        "registry": PROJECT / "config/model_registry.csv",
        "m_oof": PROJECT / "predictions/song_controlled_oof.parquet",
        "m_spl": PROJECT / "predictions/SPL_external_gold_predictions.parquet",
    }
    missing_inputs = [str(path) for path in INPUTS.values() if not path.is_file()]
    if missing_inputs:
        raise FileNotFoundError("Missing completed artifacts:\n" + "\n".join(missing_inputs))

    input_audit = pd.DataFrame([
        {"artifact": key, "path": str(path), "bytes": path.stat().st_size,
         "sha256": sha256_file(path),
         "modified_utc": pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC")}
        for key, path in INPUTS.items()
    ])
    input_audit.to_csv(TABLE_DIR / "input_artifact_audit.csv", index=False)

    # Strong external-test isolation: do not open the SPL h5ad in this cell.
    song = ad.read_h5ad(INPUTS["song"])
    assert song.obs_names.is_unique and song.var_names.is_unique
    required = {"donor", "immutable_cohort", "qc_Tlineage_pass"}
    assert required.issubset(song.obs.columns), f"Song missing {required - set(song.obs.columns)}"

    h1 = pd.read_csv(INPUTS["h1"])["gene"].astype(str).tolist()
    h1_metadata = json.loads(INPUTS["h1_metadata"].read_text(encoding="utf-8"))
    tcr = pd.read_csv(INPUTS["tcr"])["gene"].astype(str).tolist()
    assert len(h1) == 2000 and len(h1) == len(set(h1))
    assert len(tcr) == 9 and len(tcr) == len(set(tcr))
    assert set(h1).isdisjoint(tcr), "H1 must be TCR-free"
    expected_tcr = {"TRAC", "TRBC1", "TRBC2", "TRDC", "TRGC1", "TRGC2", "TRDV1", "TRDV2", "TRDV3"}
    assert set(tcr) == expected_tcr
    assert h1_metadata["method"] == "cell_ranger", (
        "Fold-specific H1 reconstruction must use the same method as final H1"
    )
    assert set(h1 + tcr).issubset(song.var_names)

    registry = pd.read_csv(INPUTS["registry"]).set_index("model_id")
    for model_id, expected_gene_regime in [("M02", "G1_TCR_free"), ("M05", "G2_TCR_inclusive")]:
        row = registry.loc[model_id]
        assert row.label_regime == LABEL_REGIME
        assert row.hvg_regime == "H1_song_inductive"
        assert row.gene_regime == expected_gene_regime
        assert float(row.lambda_TRD) == LAMBDA_TRD

    def raw_matrix(adata, cells, genes):
        row_idx = adata.obs_names.get_indexer(pd.Index(cells).astype(str))
        col_idx = adata.var_names.get_indexer(pd.Index(genes).astype(str))
        if (row_idx < 0).any() or (col_idx < 0).any():
            raise KeyError("Requested cells or genes are absent")
        matrix = get_counts(adata)[row_idx, :][:, col_idx]
        return matrix.tocsr() if sp.issparse(matrix) else np.asarray(matrix)

    def validate_counts(adata, name, genes):
        sample_cells = adata.obs_names[:min(512, adata.n_obs)]
        matrix = raw_matrix(adata, sample_cells, genes[:min(256, len(genes))])
        values = matrix.data if sp.issparse(matrix) else matrix.ravel()
        assert np.isfinite(values).all() and (values >= 0).all(), f"{name}: invalid counts"
        max_fractional = float(np.max(np.abs(values - np.rint(values)))) if len(values) else 0.0
        assert max_fractional < 1e-4, f"{name}: scVI requires raw integer counts; max deviation={max_fractional}"
        return {"dataset": name, "n_obs": adata.n_obs, "n_vars": adata.n_vars,
                "sparse": sp.issparse(get_counts(adata)), "max_fractional_sample": max_fractional}

    count_audit = pd.DataFrame([validate_counts(song, "Song", h1 + tcr)])
    count_audit.to_csv(TABLE_DIR / "raw_count_audit.csv", index=False)

    donors = sorted(song.obs["donor"].astype(str).unique())
    assert len(donors) >= 4
    song_training = make_training_table(song.obs, LABEL_REGIME)
    song_gold = song.obs["immutable_cohort"].astype(str).isin(["gold_B_paired_gdT", "gold_B_paired_abT"])
    print("Song:", song.shape, "donors:", donors, "L2 training:", len(song_training))
    print("SPL remains unopened until source thresholds and final models are frozen.")
    display(input_audit)
    display(count_audit)
    '''),
    code(r'''
    # 4. Audit Song receptor-defined V-delta coverage before considering weighting
    source_gd = song.obs.loc[
        song.obs["immutable_cohort"].astype(str).eq("gold_B_paired_gdT"),
        ["donor", "TRD_v_gene"]
    ].copy()
    vcall = source_gd["TRD_v_gene"].astype("string").fillna("").str.upper()
    has_v1 = vcall.str.contains(r"(?:^|[|,;])TRDV1(?:[*|,;]|$)", regex=True)
    has_v2 = vcall.str.contains(r"(?:^|[|,;])TRDV2(?:[*|,;]|$)", regex=True)
    source_gd["TRDV_receptor_subtype"] = np.select(
        [has_v1 & ~has_v2, has_v2 & ~has_v1, has_v1 & has_v2],
        ["TRDV1", "TRDV2", "TRDV1+TRDV2"], default="other_or_unresolved"
    )
    source_subtype_audit = (
        source_gd.groupby(["donor", "TRDV_receptor_subtype"], observed=True)
        .size().rename("n_gold_gd").reset_index()
    )
    source_subtype_audit.to_csv(TABLE_DIR / "Song_receptor_TRDV_subtype_audit.csv", index=False)

    weighting_feasibility = []
    for outer in donors:
        train_donors = [d for d in donors if d != outer]
        tab = source_gd[source_gd["donor"].astype(str).isin(train_donors)]
        counts = tab["TRDV_receptor_subtype"].value_counts()
        v1_by_donor = tab[tab.TRDV_receptor_subtype.eq("TRDV1")]["donor"].nunique()
        weighting_feasibility.append({
            "outer_donor": outer, "n_TRDV1": int(counts.get("TRDV1", 0)),
            "n_TRDV2": int(counts.get("TRDV2", 0)),
            "n_donors_with_TRDV1": int(v1_by_donor),
            "minimum_gate_pass": bool(counts.get("TRDV1", 0) >= 20 and v1_by_donor >= 2),
        })
    weighting_feasibility = pd.DataFrame(weighting_feasibility)
    weighting_feasibility.to_csv(TABLE_DIR / "Song_TRDV1_weighting_feasibility.csv", index=False)
    display(pd.crosstab(source_gd["donor"], source_gd["TRDV_receptor_subtype"], margins=True))
    display(weighting_feasibility)
    print("Primary SV02/SV05 remains unweighted by subtype. Any Vδ1 weighting is a separate sensitivity analysis.")
    '''),
    md(r'''
    ## Model implementation

    `encode_strict_inductive` runs only the already-trained scVI inference
    network. It does not call `load_query_data`, fine-tune model weights, estimate
    a target batch effect, or expose SPL labels.
    '''),
    code(r'''
    # 5. Reusable strict-inductive scVI and LR functions
    def stable_seed(*parts):
        return (SEED + zlib.crc32("|".join(map(str, parts)).encode())) % (2**31 - 1)

    def clear_accelerator():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def representation_cells(adata, allowed_donors):
        mask = (
            adata.obs["donor"].astype(str).isin(allowed_donors)
            & adata.obs["qc_Tlineage_pass"].fillna(False).astype(bool)
        )
        return adata.obs_names[mask].astype(str).tolist()

    def build_count_adata(adata, cells, genes):
        if not cells:
            raise ValueError("Cannot build AnnData from zero cells")
        if not genes:
            raise ValueError("Cannot build AnnData from zero genes")
        matrix = raw_matrix(adata, cells, genes)
        obs = adata.obs.loc[cells, ["donor"]].copy()
        # AnnData subsetting preserves unused pandas categorical levels. Scanpy's
        # batch-aware cell_ranger HVG code iterates those levels as empty batches
        # and fails inside np.percentile. Rebuild the categorical from only the
        # donors that are actually present in this training fold.
        obs["donor"] = pd.Categorical(obs["donor"].astype(str))
        var = pd.DataFrame(index=pd.Index(genes, name="gene"))
        out = ad.AnnData(X=matrix.copy(), obs=obs, var=var)
        out.obs_names = pd.Index(cells).astype(str)
        assert list(out.obs["donor"].cat.categories.astype(str)) == sorted(
            out.obs["donor"].astype(str).unique()
        )
        return out

    def fit_scvi_source_only(adata, cells, genes, config, seed):
        scvi.settings.seed = int(seed)
        torch.manual_seed(int(seed)); np.random.seed(int(seed)); random.seed(int(seed))
        rep = build_count_adata(adata, cells, genes)
        # Deliberately no batch_key: unseen SPL donors must not need target adaptation.
        scvi.model.SCVI.setup_anndata(rep)
        model = scvi.model.SCVI(
            rep, n_latent=int(config["n_latent"]), n_layers=int(config["n_layers"]),
            n_hidden=int(config["n_hidden"]), gene_likelihood="nb",
            dispersion="gene", latent_distribution="normal",
        )
        model.train(
            max_epochs=MAX_EPOCHS, accelerator=ACCELERATOR, devices=1,
            batch_size=BATCH_SIZE, early_stopping=True,
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
            check_val_every_n_epoch=1, plan_kwargs={"lr": LEARNING_RATE},
            datasplitter_kwargs={"num_workers": 0},
        )
        return rep, model

    def training_latent(model, rep):
        latent = np.asarray(model.get_latent_representation(adata=rep, batch_size=ENCODE_BATCH_SIZE))
        assert latent.shape == (rep.n_obs, model.module.n_latent)
        return pd.DataFrame(latent, index=rep.obs_names)

    @torch.inference_mode()
    def encode_strict_inductive(model, adata, cells, genes):
        """Frozen-encoder transform for unseen cells; performs no parameter updates."""
        model.to_device(str(DEVICE))
        model.module.eval()
        chunks = []
        for start in range(0, len(cells), ENCODE_BATCH_SIZE):
            batch_cells = cells[start:start + ENCODE_BATCH_SIZE]
            matrix = raw_matrix(adata, batch_cells, genes)
            dense = matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix)
            x = torch.as_tensor(dense, dtype=torch.float32, device=DEVICE)
            batch_index = torch.zeros((len(batch_cells), 1), dtype=torch.int64, device=DEVICE)
            output = model.module.inference(x=x, batch_index=batch_index, n_samples=1)
            if "qz" in output and hasattr(output["qz"], "loc"):
                z = output["qz"].loc
            elif "qz_m" in output:
                z = output["qz_m"]
            else:
                raise RuntimeError(f"Unsupported scvi-tools inference output keys: {list(output)}")
            chunks.append(z.detach().cpu().numpy())
        latent = np.vstack(chunks)
        assert latent.shape[0] == len(cells)
        return pd.DataFrame(latent, index=pd.Index(cells).astype(str))

    def verify_encoder_equivalence(model, rep, genes, n_check=32):
        """Guard the version-sensitive frozen-encoder implementation."""
        cells = rep.obs_names[:min(n_check, rep.n_obs)].astype(str).tolist()
        reference = np.asarray(model.get_latent_representation(
            adata=rep, indices=np.arange(len(cells)),
            batch_size=min(ENCODE_BATCH_SIZE, len(cells))
        ))
        observed = encode_strict_inductive(model, rep, cells, genes).to_numpy()
        if not np.allclose(reference, observed, rtol=1e-4, atol=1e-4):
            raise RuntimeError(
                "Frozen encoder does not match scvi-tools get_latent_representation; "
                "do not continue with this scvi-tools version."
            )

    def fit_latent_lr(latent, training, C, seed):
        cells = training.index.astype(str)
        assert set(cells).issubset(latent.index)
        weights = donor_balanced_weights(training, LABEL_REGIME, LAMBDA_TRD)
        classifier = Pipeline([
            ("scale", StandardScaler(with_mean=True)),
            ("lr", LogisticRegression(
                C=float(C), penalty="l2", solver="liblinear", max_iter=4000,
                random_state=int(seed),
            )),
        ])
        classifier.fit(latent.loc[cells], training["y"].to_numpy(),
                       lr__sample_weight=weights.to_numpy())
        return classifier

    def gold_table(adata, donor_values):
        mask = (
            adata.obs["donor"].astype(str).isin(donor_values)
            & adata.obs["immutable_cohort"].astype(str).isin(["gold_B_paired_gdT", "gold_B_paired_abT"])
        )
        out = adata.obs.loc[mask, ["donor", "immutable_cohort"]].copy()
        out["y"] = out["immutable_cohort"].astype(str).eq("gold_B_paired_gdT").astype(int)
        return out

    _fold_backbone_cache = {}
    def fold_backbone(fit_donors):
        key = tuple(sorted(map(str, fit_donors)))
        if key not in _fold_backbone_cache:
            eligible = (
                song.obs["donor"].astype(str).isin(key)
                & song.obs["immutable_cohort"].astype(str).isin(["gold_B_paired_gdT", "gold_B_paired_abT"])
            )
            # Lock the method recorded in H1_metadata.json. Installing scikit-misc
            # must not silently switch fold panels from cell_ranger to seurat_v3.
            eligible_cells = song.obs_names[eligible].astype(str).tolist()
            candidate_genes = [gene for gene in song.var_names.astype(str)
                               if not gene.startswith(("TRA", "TRB", "TRG", "TRD"))]
            fold_obj = build_count_adata(song, eligible_cells, candidate_genes)
            observed_donors = sorted(fold_obj.obs["donor"].astype(str).unique())
            if observed_donors != list(key):
                raise RuntimeError(
                    f"Fold donor mismatch: expected={list(key)}, observed={observed_donors}"
                )
            print(
                "HVG fold donors:", observed_donors,
                "cells per donor:", fold_obj.obs["donor"].value_counts(sort=False).to_dict(),
            )
            sc.pp.normalize_total(fold_obj, target_sum=1e4)
            sc.pp.log1p(fold_obj)
            sc.pp.highly_variable_genes(
                fold_obj, flavor="cell_ranger", n_top_genes=2000, batch_key="donor"
            )
            genes = fold_obj.var_names[fold_obj.var["highly_variable"]].astype(str).tolist()
            metadata = {"method": "cell_ranger", "n_top": 2000, "fit_donors": list(key)}
            assert len(genes) == 2000 and not set(genes).intersection(tcr)
            _fold_backbone_cache[key] = (genes, metadata)
        return _fold_backbone_cache[key]
    '''),
    md(r'''
    ### Why tune on SV02 only?

    Tuning SV02 and SV05 independently would give each feature arm a different
    optimization opportunity. Here the inner folds select the representation
    architecture and LR `C` using only TCR-free SV02. The chosen pair is then
    copied to SV05 in the matched outer fold. Thus `SV05 − SV02` isolates adding
    the fixed TCR panel as closely as possible.
    '''),
    code(r'''
    # 6. Nested Song donor selection on SV02 only, with incremental checkpoints
    INNER_SCORE_PATH = CHECKPOINT_DIR / "SV02_inner_selection_progress.csv"
    if RESUME and INNER_SCORE_PATH.is_file():
        inner_scores = pd.read_csv(INNER_SCORE_PATH)
    else:
        inner_scores = pd.DataFrame(columns=[
            "outer_donor", "inner_donor", "config_id", "C", "inner_auprc",
            "n_fit_representation", "n_fit_supervised", "n_validation", "seed"
        ])

    for outer in donors:
        outer_train_donors = [d for d in donors if d != outer]
        for inner in outer_train_donors:
            fit_donors = [d for d in outer_train_donors if d != inner]
            genes, _ = fold_backbone(fit_donors)
            rep_cells = representation_cells(song, fit_donors)
            train = song_training[song_training["donor"].astype(str).isin(fit_donors)]
            valid = gold_table(song, [inner])
            for config in SCVI_GRID:
                completed = inner_scores[
                    inner_scores.outer_donor.astype(str).eq(str(outer))
                    & inner_scores.inner_donor.astype(str).eq(str(inner))
                    & inner_scores.config_id.astype(str).eq(config["config_id"])
                ]
                if RESUME and set(np.round(completed.C.astype(float), 12)) == set(np.round(C_GRID, 12)):
                    continue
                seed = stable_seed("inner", outer, inner, config["config_id"])
                print("INNER", outer, inner, config["config_id"], "fit donors", fit_donors)
                rep, vae = fit_scvi_source_only(song, rep_cells, genes, config, seed)
                verify_encoder_equivalence(vae, rep, genes)
                z_train = training_latent(vae, rep)
                z_valid = encode_strict_inductive(vae, song, valid.index.astype(str).tolist(), genes)
                rows = []
                for C in C_GRID:
                    classifier = fit_latent_lr(z_train, train, C, seed)
                    probability = classifier.predict_proba(z_valid.loc[valid.index])[:, 1]
                    rows.append({
                        "outer_donor": outer, "inner_donor": inner,
                        "config_id": config["config_id"], "C": float(C),
                        "inner_auprc": average_precision_score(valid["y"], probability),
                        "n_fit_representation": len(rep_cells), "n_fit_supervised": len(train),
                        "n_validation": len(valid), "seed": seed,
                    })
                inner_scores = pd.concat([
                    inner_scores[~(
                        inner_scores.outer_donor.astype(str).eq(str(outer))
                        & inner_scores.inner_donor.astype(str).eq(str(inner))
                        & inner_scores.config_id.astype(str).eq(config["config_id"])
                    )], pd.DataFrame(rows)
                ], ignore_index=True)
                inner_scores.to_csv(INNER_SCORE_PATH, index=False)
                del classifier, z_train, z_valid, vae, rep
                clear_accelerator()

    expected_units = len(donors) * (len(donors) - 1) * len(SCVI_GRID) * len(C_GRID)
    assert len(inner_scores.drop_duplicates(["outer_donor", "inner_donor", "config_id", "C"])) == expected_units
    inner_summary = (
        inner_scores.groupby(["outer_donor", "config_id", "C"], as_index=False)
        .agg(inner_mean_auprc=("inner_auprc", "mean"),
             inner_min_donor_auprc=("inner_auprc", "min"))
    )
    complexity = {row["config_id"]: i for i, row in enumerate(SCVI_GRID)}
    inner_summary["complexity_rank"] = inner_summary.config_id.map(complexity)
    selected_rows = []
    for outer, group in inner_summary.groupby("outer_donor"):
        chosen = group.sort_values(
            ["inner_mean_auprc", "inner_min_donor_auprc", "complexity_rank", "C"],
            ascending=[False, False, True, True]
        ).iloc[0]
        selected_rows.append(chosen)
    outer_selection = pd.DataFrame(selected_rows).reset_index(drop=True)
    inner_scores.to_csv(TABLE_DIR / "SV02_nested_inner_scores.csv", index=False)
    outer_selection.to_csv(TABLE_DIR / "SV02_outer_fold_hyperparameter_selection.csv", index=False)
    display(outer_selection)
    '''),
    code(r'''
    # 7. Matched outer donor OOF fits: identical hyperparameters for SV02 and SV05
    OOF_PART_DIR = CHECKPOINT_DIR / "outer_oof_parts"
    OOF_PART_DIR.mkdir(parents=True, exist_ok=True)
    config_lookup = {row["config_id"]: row for row in SCVI_GRID}

    for outer in donors:
        choice = outer_selection[outer_selection.outer_donor.astype(str).eq(str(outer))].iloc[0]
        config = config_lookup[choice.config_id]
        C = float(choice.C)
        fit_donors = [d for d in donors if d != outer]
        backbone, _ = fold_backbone(fit_donors)
        rep_cells = representation_cells(song, fit_donors)
        train = song_training[song_training["donor"].astype(str).isin(fit_donors)]
        valid = gold_table(song, [outer])
        for model_id, genes in [("SV02", backbone), ("SV05", backbone + tcr)]:
            part_path = OOF_PART_DIR / f"{model_id}_{outer}.parquet"
            if RESUME and part_path.is_file():
                continue
            seed = stable_seed("outer", outer, model_id)
            print("OUTER", model_id, "held out", outer, config["config_id"], "C", C)
            rep, vae = fit_scvi_source_only(song, rep_cells, genes, config, seed)
            verify_encoder_equivalence(vae, rep, genes)
            z_train = training_latent(vae, rep)
            classifier = fit_latent_lr(z_train, train, C, seed)
            z_valid = encode_strict_inductive(vae, song, valid.index.astype(str).tolist(), genes)
            probability = classifier.predict_proba(z_valid.loc[valid.index])[:, 1]
            pd.DataFrame({
                "cell_id": valid.index.astype(str), "donor": outer,
                "model_id": model_id, "y": valid.y.to_numpy(), "p_gdT": probability,
                "outer_selected_config": config["config_id"], "outer_selected_C": C,
                "n_genes": len(genes), "target_expression_used": False,
            }).to_parquet(part_path, index=False)
            del classifier, z_train, z_valid, vae, rep
            clear_accelerator()

    part_paths = [OOF_PART_DIR / f"{model_id}_{outer}.parquet"
                  for model_id in ["SV02", "SV05"] for outer in donors]
    assert all(path.is_file() for path in part_paths)
    sv_oof = pd.concat([pd.read_parquet(path) for path in part_paths], ignore_index=True)
    assert not sv_oof.duplicated(["model_id", "cell_id"]).any()
    assert sv_oof.groupby("model_id").size().nunique() == 1
    sv_oof.to_parquet(PREDICTION_DIR / "Song_nested_donor_OOF_predictions.parquet", index=False)
    display(sv_oof.groupby(["model_id", "donor"])["y"].agg(["size", "sum"]))
    '''),
    code(r'''
    # 8. Freeze each model's operating points on Song OOF only
    def metric_row(y, probability, threshold):
        y = np.asarray(y, dtype=int); probability = np.asarray(probability, dtype=float)
        prediction = probability >= float(threshold)
        tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else np.nan
        specificity = tn / (tn + fp) if tn + fp else np.nan
        fpr = fp / (fp + tn) if fp + tn else np.nan
        f1 = 2 * precision * recall / max(precision + recall, 1e-15)
        f2 = 5 * precision * recall / max(4 * precision + recall, 1e-15)
        return {"n": len(y), "n_positive": int(y.sum()), "prevalence": float(y.mean()),
                "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
                "precision": precision, "recall": recall, "specificity": specificity,
                "false_positive_rate": fpr, "f1": f1, "f2": f2}

    def threshold_candidates(y, probability):
        thresholds = np.unique(np.r_[np.inf, np.asarray(probability, float), -np.inf])[::-1]
        return pd.DataFrame([{"threshold": threshold, **metric_row(y, probability, threshold)}
                             for threshold in thresholds])

    def choose_threshold(y, probability, rule, target=None):
        table = threshold_candidates(y, probability)
        if rule == "max_f1":
            eligible = table; order = ["f1", "precision", "recall", "threshold"]
            ascending = [False, False, False, False]
        elif rule == "max_f2":
            eligible = table; order = ["f2", "recall", "precision", "threshold"]
            ascending = [False, False, False, False]
        elif rule == "fpr_at_most":
            eligible = table[table.false_positive_rate <= float(target) + 1e-15]
            order = ["recall", "precision", "threshold"]; ascending = [False, False, False]
        elif rule == "recall_at_least":
            eligible = table[table.recall >= float(target) - 1e-15]
            order = ["precision", "recall", "threshold"]; ascending = [False, False, False]
        elif rule == "precision_at_least":
            eligible = table[table.precision >= float(target) - 1e-15]
            order = ["recall", "precision", "threshold"]; ascending = [False, False, False]
        else:
            raise ValueError(rule)
        if eligible.empty:
            raise RuntimeError(f"No threshold satisfies {rule} {target}")
        return eligible.sort_values(order, ascending=ascending).iloc[0]

    threshold_rows, source_metric_rows = [], []
    for model_id, frame in sv_oof.groupby("model_id"):
        for name, rule, target in OPERATING_RULES:
            selected = choose_threshold(frame.y, frame.p_gdT, rule, target)
            threshold_rows.append({"model_id": model_id, "operating_point": name,
                                   "threshold": float(selected.threshold),
                                   "selection_dataset": "Song nested donor OOF",
                                   "deployable": True})
            source_metric_rows.append({"model_id": model_id, "operating_point": name,
                                       "threshold": float(selected.threshold),
                                       **metric_row(frame.y, frame.p_gdT, selected.threshold)})
    sv_thresholds = pd.DataFrame(threshold_rows)
    sv_source_metrics = pd.DataFrame(source_metric_rows)
    sv_thresholds.to_csv(TABLE_DIR / "SV02_SV05_source_frozen_thresholds.csv", index=False)
    sv_source_metrics.to_csv(TABLE_DIR / "SV02_SV05_Song_OOF_operating_point_metrics.csv", index=False)
    display(sv_source_metrics)
    '''),
    code(r'''
    # 9. Select one final configuration by outer-fold vote, then refit both final Song-only models
    pair_vote = (
        outer_selection.groupby(["config_id", "C"], as_index=False)
        .agg(n_outer_votes=("outer_donor", "size"),
             mean_inner_auprc=("inner_mean_auprc", "mean"),
             min_inner_auprc=("inner_min_donor_auprc", "min"))
    )
    pair_vote["complexity_rank"] = pair_vote.config_id.map(complexity)
    final_choice = pair_vote.sort_values(
        ["n_outer_votes", "mean_inner_auprc", "min_inner_auprc", "complexity_rank", "C"],
        ascending=[False, False, False, True, True]
    ).iloc[0]
    final_config = config_lookup[final_choice.config_id]
    final_C = float(final_choice.C)
    pair_vote.to_csv(TABLE_DIR / "SV02_final_hyperparameter_vote.csv", index=False)
    (TABLE_DIR / "final_frozen_hyperparameters.json").write_text(json.dumps({
        **final_config, "C": final_C, "selected_on": "SV02 nested Song donor CV",
        "applied_unchanged_to": ["SV02", "SV05"]
    }, indent=2), encoding="utf-8")
    display(pair_vote)
    print("FROZEN FINAL CHOICE:", final_config, "C=", final_C)

    final_rep_cells = representation_cells(song, donors)
    final_training = song_training.copy()
    final_models = {}
    for model_id, genes in [("SV02", h1), ("SV05", h1 + tcr)]:
        model_path = MODEL_DIR / model_id
        classifier_path = MODEL_DIR / f"{model_id}_latent_LR.joblib"
        seed = stable_seed("final", model_id)
        rep = build_count_adata(song, final_rep_cells, genes)
        if RESUME and model_path.is_dir() and classifier_path.is_file():
            scvi.model.SCVI.setup_anndata(rep)
            vae = scvi.model.SCVI.load(model_path, adata=rep)
            classifier = joblib.load(classifier_path)
        else:
            del rep
            rep, vae = fit_scvi_source_only(song, final_rep_cells, genes, final_config, seed)
            verify_encoder_equivalence(vae, rep, genes)
            z_train = training_latent(vae, rep)
            classifier = fit_latent_lr(z_train, final_training, final_C, seed)
            vae.save(model_path, overwrite=True, save_anndata=False)
            joblib.dump(classifier, classifier_path)
            history = pd.DataFrame({key: pd.Series(np.asarray(value).ravel())
                                    for key, value in vae.history.items()})
            history.to_csv(TABLE_DIR / f"{model_id}_training_history.csv", index=False)
            del z_train
        final_models[model_id] = {"vae": vae, "classifier": classifier,
                                  "rep": rep, "genes": genes}
    '''),
    md(r'''
    ## Results: locked external evaluation

    From this point onward SPL is used only for evaluation. No code below calls
    a training, optimizer, calibration-fit, or threshold-selection method on SPL.
    '''),
    code(r'''
    # 10. Only now open SPL; validate it, then apply frozen Song-only models once
    spl = ad.read_h5ad(INPUTS["spl"])
    assert spl.obs_names.is_unique and spl.var_names.is_unique
    required = {"donor", "immutable_cohort", "qc_Tlineage_pass"}
    assert required.issubset(spl.obs.columns), f"SPL missing {required - set(spl.obs.columns)}"
    assert set(h1 + tcr).issubset(spl.var_names)
    spl_count_audit = validate_counts(spl, "SPL", h1 + tcr)
    pd.DataFrame([spl_count_audit]).to_csv(TABLE_DIR / "SPL_raw_count_audit.csv", index=False)
    spl_gold = (
        spl.obs["donor"].astype(str).isin(SPL_GOLD_DONORS)
        & spl.obs["immutable_cohort"].astype(str).isin(["gold_B_paired_gdT", "gold_B_paired_abT"])
    )
    assert set(spl.obs.loc[spl_gold, "donor"].astype(str).unique()) == set(SPL_GOLD_DONORS)
    print("LOCKED SPL external gold:", int(spl_gold.sum()), "positives:",
          int(spl.obs.loc[spl_gold, "immutable_cohort"].astype(str).eq("gold_B_paired_gdT").sum()))

    external = gold_table(spl, SPL_GOLD_DONORS)
    assert len(external) == int(spl_gold.sum())
    external_rows = []
    for model_id, bundle in final_models.items():
        cells = external.index.astype(str).tolist()
        z_external = encode_strict_inductive(bundle["vae"], spl, cells, bundle["genes"])
        probability = bundle["classifier"].predict_proba(z_external.loc[cells])[:, 1]
        external_rows.append(pd.DataFrame({
            "cell_id": cells, "donor": external.loc[cells, "donor"].astype(str).to_numpy(),
            "model_id": model_id, "y": external.loc[cells, "y"].to_numpy(),
            "p_gdT": probability,
        }))
        del z_external
        clear_accelerator()
    sv_spl = pd.concat(external_rows, ignore_index=True)
    assert not sv_spl.duplicated(["model_id", "cell_id"]).any()
    assert sv_spl.groupby("model_id").size().eq(len(external)).all()
    sv_spl.to_parquet(PREDICTION_DIR / "SPL_external_gold_predictions.parquet", index=False)
    display(sv_spl.groupby(["model_id", "donor"])["y"].agg(["size", "sum"]))
    '''),
    code(r'''
    # 11. Threshold-free and source-frozen operating-point metrics, overall and by donor
    def discrimination_row(frame):
        return {
            "n": len(frame), "n_positive": int(frame.y.sum()),
            "prevalence": float(frame.y.mean()),
            "average_precision": average_precision_score(frame.y, frame.p_gdT),
            "auroc": roc_auc_score(frame.y, frame.p_gdT),
            "brier": brier_score_loss(frame.y, frame.p_gdT),
        }

    threshold_free_rows = []
    for model_id, frame in sv_spl.groupby("model_id"):
        threshold_free_rows.append({"model_id": model_id, "scope": "overall", **discrimination_row(frame)})
        for donor, group in frame.groupby("donor"):
            threshold_free_rows.append({"model_id": model_id, "scope": f"donor:{donor}",
                                        **discrimination_row(group)})
    sv_discrimination = pd.DataFrame(threshold_free_rows)
    sv_discrimination.to_csv(TABLE_DIR / "SV02_SV05_SPL_discrimination_metrics.csv", index=False)

    op_rows, donor_op_rows = [], []
    for threshold in sv_thresholds.itertuples(index=False):
        frame = sv_spl[sv_spl.model_id.eq(threshold.model_id)]
        op_rows.append({"model_id": threshold.model_id,
                        "operating_point": threshold.operating_point,
                        "threshold": threshold.threshold,
                        "selection_dataset": threshold.selection_dataset,
                        **metric_row(frame.y, frame.p_gdT, threshold.threshold)})
        for donor, group in frame.groupby("donor"):
            donor_op_rows.append({"model_id": threshold.model_id,
                                  "operating_point": threshold.operating_point,
                                  "threshold": threshold.threshold, "donor": donor,
                                  **metric_row(group.y, group.p_gdT, threshold.threshold)})
    sv_spl_ops = pd.DataFrame(op_rows)
    sv_spl_donor_ops = pd.DataFrame(donor_op_rows)
    sv_spl_ops.to_csv(TABLE_DIR / "SV02_SV05_SPL_source_frozen_operating_points.csv", index=False)
    sv_spl_donor_ops.to_csv(TABLE_DIR / "SV02_SV05_SPL_source_frozen_operating_points_by_donor.csv", index=False)
    display(sv_discrimination)
    display(sv_spl_ops)
    display(sv_spl_donor_ops[sv_spl_donor_ops.donor.eq(TARGET_DONOR)])
    '''),
    code(r'''
    # 12. Calibration overall and by donor (evaluation only; no calibrator is fitted)
    def ece_quantile(y, probability, n_bins=10):
        frame = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(probability)})
        frame["bin"] = pd.qcut(rankdata(frame.p, method="first"), q=min(n_bins, len(frame)), labels=False)
        bins = frame.groupby("bin", observed=True).agg(
            n_cells=("y", "size"), mean_probability=("p", "mean"),
            observed_gd_fraction=("y", "mean")
        ).reset_index()
        ece = np.average(np.abs(bins.mean_probability - bins.observed_gd_fraction),
                         weights=bins.n_cells)
        return float(ece), bins

    calibration_rows, calibration_bin_rows = [], []
    for model_id, model_frame in sv_spl.groupby("model_id"):
        for scope, frame in [("overall", model_frame)] + [
            (f"donor:{donor}", group) for donor, group in model_frame.groupby("donor")
        ]:
            ece, bins = ece_quantile(frame.y, frame.p_gdT)
            calibration_rows.append({"model_id": model_id, "scope": scope,
                                     "n": len(frame), "brier": brier_score_loss(frame.y, frame.p_gdT),
                                     "ece_10_quantile": ece,
                                     "mean_probability": frame.p_gdT.mean(),
                                     "observed_prevalence": frame.y.mean()})
            bins.insert(0, "scope", scope); bins.insert(0, "model_id", model_id)
            calibration_bin_rows.append(bins)
    calibration_summary = pd.DataFrame(calibration_rows)
    calibration_bins = pd.concat(calibration_bin_rows, ignore_index=True)
    calibration_summary.to_csv(TABLE_DIR / "SV02_SV05_SPL_calibration_summary.csv", index=False)
    calibration_bins.to_csv(TABLE_DIR / "SV02_SV05_SPL_calibration_bins.csv", index=False)
    display(calibration_summary)
    '''),
    code(r'''
    # 13. Expression-proxy TRDV1/TRDV2 subgroup recall, including A36
    external_cells = external.index.astype(str).tolist()
    trdv_matrix = raw_matrix(spl, external_cells, ["TRDV1", "TRDV2", "TRDV3"])
    trdv_dense = trdv_matrix.toarray() if sp.issparse(trdv_matrix) else np.asarray(trdv_matrix)
    detected = trdv_dense > 0
    v1, v2, v3 = detected[:, 0], detected[:, 1], detected[:, 2]
    proxy = np.select(
        [v1 & ~v2 & ~v3, v2 & ~v1 & ~v3, v1 & v2, v3],
        ["TRDV1_RNA_only", "TRDV2_RNA_only", "TRDV1_TRDV2_RNA_both", "TRDV3_RNA_other"],
        default="none_detected"
    )
    subtype_map = pd.DataFrame({"cell_id": external_cells, "TRDV_RNA_proxy": proxy})
    subtype_map.to_csv(TABLE_DIR / "SPL_TRDV_RNA_expression_proxy.csv", index=False)

    subtype_predictions = sv_spl.merge(subtype_map, on="cell_id", validate="many_to_one")
    subtype_predictions = subtype_predictions[subtype_predictions.y.eq(1)].copy()
    def wilson_interval(successes, total, z=1.959963984540054):
        if total == 0:
            return np.nan, np.nan
        p = successes / total
        denominator = 1 + z*z/total
        center = (p + z*z/(2*total)) / denominator
        half = z*np.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
        return center - half, center + half

    subgroup_rows = []
    selected_ops = ["Song_OOF_registered_maxF1", "Song_OOF_FPR_le_0.007"]
    for threshold in sv_thresholds[sv_thresholds.operating_point.isin(selected_ops)].itertuples(index=False):
        frame = subtype_predictions[subtype_predictions.model_id.eq(threshold.model_id)].copy()
        frame["predicted"] = frame.p_gdT.ge(threshold.threshold)
        for scope, scoped in [("overall", frame)] + [
            (f"donor:{donor}", group) for donor, group in frame.groupby("donor")
        ]:
            for subtype, group in scoped.groupby("TRDV_RNA_proxy"):
                tp = int(group.predicted.sum()); total = len(group)
                ci_low, ci_high = wilson_interval(tp, total)
                subgroup_rows.append({
                    "model_id": threshold.model_id, "operating_point": threshold.operating_point,
                    "scope": scope, "TRDV_RNA_proxy": subtype, "n_true_gd": len(group),
                    "tp": tp, "fn": int((~group.predicted).sum()),
                    "recall": float(group.predicted.mean()),
                    "recall_ci_low": ci_low, "recall_ci_high": ci_high,
                    "definition_caveat": "RNA detection proxy; not receptor V-gene truth",
                })
    subgroup_metrics = pd.DataFrame(subgroup_rows)
    subgroup_metrics.to_csv(TABLE_DIR / "SV02_SV05_SPL_TRDV_RNA_stratified_recall.csv", index=False)
    display(subgroup_metrics[subgroup_metrics.scope.isin(["overall", f"donor:{TARGET_DONOR}"])])
    '''),
    code(r'''
    # 14. Bring controlled M02/M05 into the identical evaluation protocol
    m_oof = pd.read_parquet(INPUTS["m_oof"])
    m_spl = pd.read_parquet(INPUTS["m_spl"])

    # Song input is explicitly the controlled OOF artifact. The SPL artifact
    # contains controlled and optimized rows, so filter before any comparison.
    m_oof = m_oof[m_oof["model_id"].isin(["M02", "M05"])].copy()
    if "analysis" not in m_spl.columns:
        raise KeyError("SPL_external_gold_predictions.parquet is missing 'analysis'")
    m_spl = m_spl[
        m_spl["model_id"].isin(["M02", "M05"])
        & m_spl["analysis"].astype(str).eq("controlled")
    ].copy()

    required_prediction_columns = {"cell_id", "donor", "model_id", "y", "p_gdT"}
    for name, frame in [("Song OOF", m_oof), ("SPL", m_spl)]:
        missing_columns = required_prediction_columns - set(frame.columns)
        if missing_columns:
            raise KeyError(f"{name} predictions missing columns: {missing_columns}")

    prediction_key = ["cell_id", "donor", "y", "model_id"]
    if m_oof.duplicated(prediction_key).any():
        raise RuntimeError("Controlled M02/M05 Song OOF predictions contain duplicate keys")
    if m_spl.duplicated(prediction_key).any():
        raise RuntimeError("Controlled M02/M05 SPL predictions contain duplicate keys")

    for model_id in ["M02", "M05"]:
        observed_cells = set(m_spl.loc[m_spl.model_id.eq(model_id), "cell_id"])
        if observed_cells != set(external_cells):
            raise RuntimeError(f"{model_id}: SPL cells do not match the SV evaluation cells")

    all_oof = pd.concat([m_oof[list(required_prediction_columns)],
                         sv_oof[list(required_prediction_columns)]], ignore_index=True)
    all_spl = pd.concat([m_spl[list(required_prediction_columns)],
                         sv_spl[list(required_prediction_columns)]], ignore_index=True)

    expected_models = ["M02", "M05", "SV02", "SV05"]
    model_counts = all_spl.groupby("model_id").size().reindex(expected_models)
    if not model_counts.eq(len(external_cells)).all():
        raise RuntimeError(
            "Each model must have one SPL prediction per external cell; "
            f"observed={model_counts.to_dict()}"
        )
    if all_spl.duplicated(prediction_key).any():
        raise RuntimeError("Four-model SPL table contains duplicate cell-model keys")

    all_threshold_rows = []
    for model_id, frame in all_oof.groupby("model_id"):
        for name, rule, target in OPERATING_RULES:
            selected = choose_threshold(frame.y, frame.p_gdT, rule, target)
            all_threshold_rows.append({"model_id": model_id, "operating_point": name,
                                       "threshold": float(selected.threshold),
                                       "selection_dataset": "Song donor OOF only"})
    all_thresholds = pd.DataFrame(all_threshold_rows)
    all_thresholds.to_csv(TABLE_DIR / "M02_M05_SV02_SV05_source_frozen_thresholds.csv", index=False)

    comparison_rows = []
    for model_id, frame in all_spl.groupby("model_id"):
        base = {"model_id": model_id, **discrimination_row(frame)}
        for op in ["Song_OOF_registered_maxF1", "Song_OOF_FPR_le_0.007"]:
            threshold = all_thresholds.query("model_id == @model_id and operating_point == @op").threshold.item()
            metrics = metric_row(frame.y, frame.p_gdT, threshold)
            suffix = "maxF1" if op.endswith("maxF1") else "FPR007"
            for key in ["tp", "fp", "tn", "fn", "precision", "recall", "false_positive_rate", "f1", "f2"]:
                base[f"{suffix}_{key}"] = metrics[key]
        comparison_rows.append(base)
    four_model_comparison = pd.DataFrame(comparison_rows)
    four_model_comparison.to_csv(TABLE_DIR / "M02_M05_SV02_SV05_SPL_comparison.csv", index=False)
    print("PASS:", len(all_spl), "rows;", len(external_cells), "cells × 4 models")
    display(four_model_comparison)
    '''),
    code(r'''
    # 15. Fast paired donor-cluster bootstrap on SPL discrimination curves
    # With only five external donors, intervals are exploratory transport
    # uncertainty. SPL curve operating points are evaluation-only oracles, not
    # threshold-selection rules for deployment.
    def curve_metrics_fast(frame, model_id):
        y = frame["y"].to_numpy(dtype=int)
        probability = frame[model_id].to_numpy(dtype=float)
        if len(np.unique(y)) != 2:
            raise RuntimeError(f"{model_id}: bootstrap sample lacks both classes")

        precision, recall, _ = precision_recall_curve(y, probability)
        eligible_precision = precision[recall >= 0.80 - 1e-15]
        precision_at_recall_080 = (
            float(np.max(eligible_precision)) if len(eligible_precision) else np.nan
        )

        fpr, tpr, _ = roc_curve(y, probability, drop_intermediate=False)
        eligible_recall = tpr[fpr <= 0.007 + 1e-15]
        recall_at_fpr_0007 = (
            float(np.max(eligible_recall)) if len(eligible_recall) else np.nan
        )
        return {
            "AUPRC": float(average_precision_score(y, probability)),
            "precision_at_recall_0.80": precision_at_recall_080,
            "recall_at_FPR_0.007": recall_at_fpr_0007,
        }

    contrast_pairs = [
        ("M05", "M02", "TCR effect in direct-gene LR"),
        ("SV05", "SV02", "TCR effect in scVI-latent LR"),
        ("SV02", "M02", "scVI representation effect, TCR-free"),
        ("SV05", "M05", "scVI representation effect, TCR-inclusive"),
    ]
    if all_spl.duplicated(["cell_id", "donor", "y", "model_id"]).any():
        raise RuntimeError("all_spl contains duplicate cell-model predictions")
    wide = all_spl.pivot(
        index=["cell_id", "donor", "y"], columns="model_id", values="p_gdT"
    ).reset_index()
    required_models = ["M02", "M05", "SV02", "SV05"]
    assert not wide[required_models].isna().any().any()
    donor_frames = {str(donor): frame.copy() for donor, frame in wide.groupby("donor")}
    donor_names = sorted(donor_frames)
    if len(donor_names) != len(SPL_GOLD_DONORS):
        raise RuntimeError(f"Expected {len(SPL_GOLD_DONORS)} donors; found {donor_names}")
    rng = np.random.default_rng(SEED)

    bootstrap_rows = []
    bootstrap_start = time.perf_counter()
    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled = rng.choice(donor_names, size=len(donor_names), replace=True)
        frame = pd.concat([donor_frames[donor].assign(bootstrap_copy=i)
                           for i, donor in enumerate(sampled)], ignore_index=True)
        metrics_by_model = {
            model_id: curve_metrics_fast(frame, model_id) for model_id in required_models
        }
        for model_a, model_b, estimand in contrast_pairs:
            metrics_a = metrics_by_model[model_a]; metrics_b = metrics_by_model[model_b]
            for metric in metrics_a:
                bootstrap_rows.append({"replicate": replicate, "model_a": model_a,
                                       "model_b": model_b, "estimand": estimand,
                                       "metric": metric, "delta_a_minus_b": metrics_a[metric] - metrics_b[metric]})
        if (replicate + 1) % 250 == 0:
            elapsed = time.perf_counter() - bootstrap_start
            print(f"Completed {replicate + 1}/{BOOTSTRAP_REPLICATES} in {elapsed:.1f} seconds")

    bootstrap = pd.DataFrame(bootstrap_rows)
    observed_metrics = {
        model_id: curve_metrics_fast(wide, model_id) for model_id in required_models
    }
    observed_rows = []
    for model_a, model_b, estimand in contrast_pairs:
        for metric in observed_metrics[model_a]:
            observed_rows.append({
                "model_a": model_a, "model_b": model_b, "estimand": estimand,
                "metric": metric,
                "observed_delta": observed_metrics[model_a][metric] - observed_metrics[model_b][metric],
            })
    observed_contrasts = pd.DataFrame(observed_rows)

    contrast_summary = (
        bootstrap.groupby(["model_a", "model_b", "estimand", "metric"], as_index=False)
        .agg(delta_mean=("delta_a_minus_b", "mean"),
             ci_low=("delta_a_minus_b", lambda x: np.quantile(x, 0.025)),
             ci_high=("delta_a_minus_b", lambda x: np.quantile(x, 0.975)),
             probability_delta_gt_zero=("delta_a_minus_b", lambda x: np.mean(x > 0)))
    )
    contrast_summary = observed_contrasts.merge(
        contrast_summary, on=["model_a", "model_b", "estimand", "metric"],
        validate="one_to_one"
    )
    contrast_summary["n_external_donors"] = len(donor_names)
    contrast_summary["uncertainty_scope"] = (
        "Exploratory five-donor cluster bootstrap; probability_delta_gt_zero is not a p-value"
    )
    bootstrap.to_parquet(TABLE_DIR / "SPL_donor_bootstrap_contrast_replicates.parquet", index=False)
    contrast_summary.to_csv(TABLE_DIR / "SPL_donor_bootstrap_contrasts.csv", index=False)
    display(contrast_summary)
    '''),
    code(r'''
    # 16. Apply the final registered figure design and plot SPL PR/ROC curves
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    selected_font = next(
        (font for font in ["Arial", "Helvetica", "Liberation Sans"] if font in available_fonts),
        None,
    )
    if selected_font is None:
        raise RuntimeError(
            "Arial, Helvetica, and Liberation Sans are unavailable; install one before export"
        )
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [selected_font],
        "font.weight": "normal", "pdf.fonttype": 42, "ps.fonttype": 42,
        "text.color": "black", "axes.labelcolor": "black",
        "axes.titlecolor": "black", "axes.edgecolor": "black",
        "xtick.color": "black", "ytick.color": "black",
        "axes.titleweight": "normal", "axes.labelweight": "normal",
        "figure.titleweight": "normal", "savefig.transparent": False,
        "savefig.facecolor": "white", "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
    palette = {"M02": "#6B7280", "M05": "#E32F16", "SV02": "#2563EB", "SV05": "#7C3AED"}

    def save_publication_figure(fig, stem):
        for text_object in fig.findobj(match=mpl.text.Text):
            text_object.set_color("black")
            text_object.set_fontfamily(selected_font)
            text_object.set_fontweight("normal")
        pdf_path = FIGURE_DIR / f"{stem}.pdf"
        png_path = FIGURE_DIR / f"{stem}.png"
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight", facecolor="white")
        fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight", facecolor="white")
        for output_path in [pdf_path, png_path]:
            if not output_path.is_file() or output_path.stat().st_size == 0:
                raise RuntimeError(f"Figure was not created: {output_path}")
        print("Saved:", pdf_path, "and", png_path, "font:", selected_font)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    for model_id, frame in all_spl.groupby("model_id"):
        precision, recall, _ = precision_recall_curve(frame.y, frame.p_gdT)
        fpr, tpr, _ = roc_curve(frame.y, frame.p_gdT)
        axes[0].plot(recall, precision, lw=2.2, color=palette[model_id],
                     label=f"{model_id} AP={average_precision_score(frame.y, frame.p_gdT):.3f}")
        axes[1].plot(fpr, tpr, lw=2.2, color=palette[model_id],
                     label=f"{model_id} AUROC={roc_auc_score(frame.y, frame.p_gdT):.3f}")
    prevalence = all_spl.drop_duplicates("cell_id").y.mean()
    axes[0].axhline(prevalence, ls="--", color="black", lw=1.2,
                    label=f"SPL prevalence={prevalence:.3f}")
    axes[0].set(xlabel="Recall", ylabel="Precision", title="SPL external gold PR",
                xlim=(0, 1.01), ylim=(0, 1.03))
    axes[1].plot([0, 1], [0, 1], ls="--", color="black", lw=1.2)
    axes[1].set(xlabel="False-positive rate", ylabel="True-positive rate",
                title="SPL external gold ROC", xlim=(0, 1), ylim=(0, 1.01))
    for ax in axes:
        ax.grid(color="#D9D9D9", linewidth=.7)
        ax.legend(loc="lower center", bbox_to_anchor=(.5, -0.42), frameon=True)
    fig.subplots_adjust(bottom=.30, wspace=.28)
    save_publication_figure(fig, "M02_M05_SV02_SV05_SPL_PR_ROC")
    plt.show()
    plt.close(fig)
    '''),
    code(r'''
    # 17. Donor and TRDV-proxy recall with Wilson 95% confidence intervals
    recall_plot = sv_spl_donor_ops[
        sv_spl_donor_ops.operating_point.isin(["Song_OOF_registered_maxF1", "Song_OOF_FPR_le_0.007"])
    ].copy()
    recall_plot["operating_point_display"] = recall_plot.operating_point.map({
        "Song_OOF_registered_maxF1": "Song-selected max-F1 threshold",
        "Song_OOF_FPR_le_0.007": "Threshold selected at Song OOF FPR≤0.7%",
    })
    donor_intervals = recall_plot.apply(
        lambda row: wilson_interval(int(row["tp"]), int(row["n_positive"])), axis=1
    )
    recall_plot[["recall_ci_low", "recall_ci_high"]] = pd.DataFrame(
        donor_intervals.tolist(), index=recall_plot.index
    )

    def grouped_recall_bars(ax, data, category_column, category_order, n_column, title):
        model_order = ["SV02", "SV05"]
        x = np.arange(len(category_order), dtype=float)
        width = 0.36
        for model_index, model_id in enumerate(model_order):
            model_data = (
                data[data["model_id"].eq(model_id)]
                .set_index(category_column).reindex(category_order)
            )
            position = x + (model_index - 0.5) * width
            recall_values = model_data["recall"].to_numpy(dtype=float)
            lower = recall_values - model_data["recall_ci_low"].to_numpy(dtype=float)
            upper = model_data["recall_ci_high"].to_numpy(dtype=float) - recall_values
            bars = ax.bar(
                position, recall_values, width=width, color=palette[model_id],
                label=model_id, yerr=np.vstack([lower, upper]), capsize=3,
                error_kw={"ecolor": "black", "elinewidth": 1, "capthick": 1},
            )
            for bar, n_value in zip(bars, model_data[n_column].to_numpy(dtype=int)):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    min(1.01, bar.get_height() + 0.035),
                    f"n={n_value}", ha="center", va="bottom", fontsize=8, color="black",
                )
        ax.set_xticks(x, category_order)
        ax.set(title=title, ylabel="Recall", ylim=(0, 1.08))
        ax.grid(axis="y", color="#D9D9D9", linewidth=.7)
        ax.legend(title="Model", loc="lower center", bbox_to_anchor=(.5, -0.32),
                  ncol=2, frameon=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    donor_order = list(SPL_GOLD_DONORS)
    operating_point_order = [
        "Song-selected max-F1 threshold",
        "Threshold selected at Song OOF FPR≤0.7%",
    ]
    for ax, operating_point in zip(axes[:2], operating_point_order):
        plot_data = recall_plot[
            recall_plot["operating_point_display"].eq(operating_point)
        ]
        grouped_recall_bars(
            ax, plot_data, "donor", donor_order, "n_positive",
            f"SPL donor recall\n{operating_point}",
        )
        ax.set_xlabel("Donor")

    subgroup_plot = subgroup_metrics[
        subgroup_metrics.scope.isin(["overall", f"donor:{TARGET_DONOR}"])
        & subgroup_metrics.operating_point.eq("Song_OOF_FPR_le_0.007")
        & subgroup_metrics.TRDV_RNA_proxy.isin(["TRDV1_RNA_only", "TRDV2_RNA_only"])
    ].copy()
    subgroup_plot["group"] = subgroup_plot.scope + " | " + subgroup_plot.TRDV_RNA_proxy
    subgroup_order = [
        f"overall | {subtype}" for subtype in ["TRDV1_RNA_only", "TRDV2_RNA_only"]
    ] + [
        f"donor:{TARGET_DONOR} | {subtype}"
        for subtype in ["TRDV1_RNA_only", "TRDV2_RNA_only"]
    ]
    grouped_recall_bars(
        axes[2], subgroup_plot, "group", subgroup_order, "n_true_gd",
        "TRDV RNA-proxy recall\nSong-selected FPR threshold",
    )
    axes[2].set_xlabel("")
    axes[2].tick_params(axis="x", rotation=28)
    axes[2].legend(title="Model", loc="lower center", bbox_to_anchor=(.5, -0.45),
                   ncol=2, frameon=True)
    fig.subplots_adjust(bottom=.32, wspace=.28)
    save_publication_figure(fig, "SV02_SV05_donor_and_TRDV_proxy_recall")
    plt.show()
    plt.close(fig)
    '''),
    code(r'''
    # 18. Calibration plot, including A36
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, scope in zip(axes, ["overall", f"donor:{TARGET_DONOR}"]):
        subset = calibration_bins[calibration_bins.scope.eq(scope)]
        for model_id, group in subset.groupby("model_id"):
            group = group.sort_values("mean_probability")
            ax.plot(group.mean_probability, group.observed_gd_fraction, marker="o",
                    markersize=5, linewidth=2, color=palette[model_id], label=model_id)
        ax.plot([0, 1], [0, 1], ls="--", color="black", lw=1.2,
                label="Perfect calibration")
        ax.set(xlabel="Mean predicted probability", ylabel="Observed γδ fraction",
               title=f"Calibration: {scope}", xlim=(0, 1), ylim=(0, 1))
        ax.grid(color="#D9D9D9", linewidth=.7)
        ax.legend(frameon=True)
    fig.tight_layout()
    save_publication_figure(fig, "SV02_SV05_SPL_calibration_overall_A36")
    plt.show()
    plt.close(fig)
    '''),
    code(r'''
    # 19. Final integrity checks and machine-readable completion manifest
    required_outputs = [
        TABLE_DIR / "SV02_SV05_SPL_discrimination_metrics.csv",
        TABLE_DIR / "SV02_SV05_SPL_source_frozen_operating_points.csv",
        TABLE_DIR / "SV02_SV05_SPL_source_frozen_operating_points_by_donor.csv",
        TABLE_DIR / "SV02_SV05_SPL_calibration_summary.csv",
        TABLE_DIR / "SV02_SV05_SPL_TRDV_RNA_stratified_recall.csv",
        TABLE_DIR / "M02_M05_SV02_SV05_SPL_comparison.csv",
        TABLE_DIR / "SPL_donor_bootstrap_contrasts.csv",
        PREDICTION_DIR / "Song_nested_donor_OOF_predictions.parquet",
        PREDICTION_DIR / "SPL_external_gold_predictions.parquet",
        FIGURE_DIR / "M02_M05_SV02_SV05_SPL_PR_ROC.pdf",
        FIGURE_DIR / "M02_M05_SV02_SV05_SPL_PR_ROC.png",
        FIGURE_DIR / "SV02_SV05_donor_and_TRDV_proxy_recall.pdf",
        FIGURE_DIR / "SV02_SV05_donor_and_TRDV_proxy_recall.png",
        FIGURE_DIR / "SV02_SV05_SPL_calibration_overall_A36.pdf",
        FIGURE_DIR / "SV02_SV05_SPL_calibration_overall_A36.png",
    ]
    assert all(path.is_file() and path.stat().st_size > 0 for path in required_outputs)
    assert set(sv_thresholds.model_id) == {"SV02", "SV05"}
    assert len(sv_thresholds) == 2 * len(OPERATING_RULES)
    assert not sv_spl.duplicated(["model_id", "cell_id"]).any()
    assert sv_spl.groupby("model_id").y.sum().nunique() == 1

    completion = {
        "status": "complete", "completed_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "project": str(PROJECT), "models": ["SV02", "SV05"],
        "external_gold_n": int(len(external)), "external_positive_n": int(external.y.sum()),
        "spl_used_for_training": False, "spl_used_for_hyperparameter_selection": False,
        "spl_used_for_threshold_selection": False, "query_adaptation_used": False,
        "required_outputs": {str(path.relative_to(PROJECT)): sha256_file(path) for path in required_outputs},
    }
    (TABLE_DIR / "completion_manifest.json").write_text(json.dumps(completion, indent=2), encoding="utf-8")
    display(pd.Series(completion).drop("required_outputs"))
    '''),
    md(r'''
    ## Takeaways

    ### Biological and modelling interpretation

    1. On 5,046 SPL external-gold cells (378 positives; prevalence 0.075), M02,
       M05, SV02, and SV05 achieved AUPRC 0.423, 0.943, 0.641, and 0.685,
       respectively. AUPRC is primary because γδ T cells are the minority class.
    2. `SV02 − M02` was positive: the scVI latent representation recovered useful
       coordinated TCR-free transcriptional structure. In contrast, `SV05 − M05`
       was strongly negative. Compressing 2,009 genes into a 20-dimensional
       unsupervised latent space diluted the highly discriminative TCR-gene signal
       that direct-gene M05 could use explicitly.
    3. `SV05 − SV02` was modestly positive but donor-dependent. The pooled AUPRC
       difference was 0.043; the five-donor cluster-bootstrap mean was 0.059 with
       a percentile interval of 0.010–0.157. A36 contributed 229/378 positives but
       improved only from AUPRC 0.669 to 0.684; A35 showed no improvement.
    4. Source-frozen thresholds did not preserve their Song operating properties
       on SPL. The threshold selected at Song OOF FPR≤0.7% produced SPL FPR 2.72%
       for SV02 and 5.63% for SV05. These probability scores are not transport-
       calibrated deployment probabilities.
    5. SV05 recall at that Song-selected threshold was lower for
       `TRDV1_RNA_only` than `TRDV2_RNA_only` overall (0.749 versus 0.941) and in
       A36 (0.772 versus 0.962). This is an RNA-expression proxy diagnostic, not
       receptor V-gene truth, and the proxy genes are themselves SV05 inputs.

    ### Statistical and reproducibility limitations

    - Five external donors provide limited biological replication. The 2,000
      donor-cluster resamples quantify exploratory transport uncertainty; they do
      not create 2,000 independent biological samples. The reported
      `probability_delta_gt_zero` is a bootstrap stability proportion, not a
      conventional p-value.
    - `precision_at_recall_0.80` and `recall_at_FPR_0.007` in the bootstrap are
      SPL curve-based oracle estimands for discrimination. They are distinct from
      the Song-selected frozen thresholds and must not be presented as deployable
      SPL operating points.
    - SV02 and SV05 are separate stochastic scVI fits. Their contrast contains
      both the nine-gene feature difference and single-seed representation-fitting
      variability. Matched repeated seeds are required before attributing the
      entire difference to the TCR panel.
    - M05's receptor-associated input genes are biologically coupled to the
      receptor-defined gold labels. Its result supports recovery of receptor-
      defined cells from receptor-gene expression, not receptor-independent γδ T
      cell discovery or performance in orphan/receptor-negative cells.

    ### Model-use conclusion

    Use M05 when TCR-gene expression is available and the target is receptor-
    defined γδ T-cell recovery. Use SV02 as the stronger tested TCR-free model.
    Do not interpret SV02/SV05 scores as calibrated probabilities or change their
    thresholds using SPL labels; threshold recalibration requires an independent
    calibration cohort.
    '''),
    md(r'''
    ## Standalone appendix: frozen SV02 predictions on the existing SPL UMAP

    Run the cells in this appendix **in order** on Woody. They are independent of
    all earlier notebook memory: the first cell reloads every required artifact
    from `/data/users/kaelynwen/gdT_project/gdt_lr_12model`.

    This appendix uses the project's formal model ID `SV02`: the L2
    (`L2_paired_plus_TRD_equal`), H1, G1 TCR-free scVI-latent + LR model. It does
    not train or fine-tune a model, select a threshold on SPL, or recompute UMAP.
    It applies the saved Song-only encoder and classifier to the same four SPL
    receptor-evidence groups shown in the M05 locator figure, then joins those
    scores to the already exported Figure 1A RNA-only UMAP coordinates.

    The grey background is spatial context only. UMAP location is descriptive,
    not an independent accuracy test, because RNA expression underlies both the
    classifier score and the existing UMAP. Large point clouds are rasterized
    inside the PDF solely to keep the file usable; text, labels, and axes remain
    editable vector objects.
    '''),
    code(r'''
    # U1. Standalone setup: locate and validate all frozen inputs.
    from pathlib import Path
    import hashlib
    import json
    import os
    import platform
    import sys

    import anndata as ad
    import joblib
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    import scvi
    import torch

    SV02_MODEL_ID = "SV02"
    SV02_OPERATING_POINT = "Song_OOF_registered_maxF1"
    SV02_SPL_DONORS = ("A29", "A31", "A35", "A36", "A37")
    SV02_ENCODE_BATCH_SIZE = 512

    project_candidates = [
        Path("/data/users/kaelynwen/gdT_project/gdt_lr_12model"),
        Path.cwd(),
        Path.cwd().parent,
    ]
    SV02_PROJECT = next(
        (path.resolve() for path in project_candidates
         if (path / "manifests/CrossAtlas_D2_SPL_cohorts.h5ad").is_file()),
        None,
    )
    if SV02_PROJECT is None:
        raise FileNotFoundError(
            "Cannot locate gdt_lr_12model. On Woody, start from "
            "/data/users/kaelynwen/gdT_project/gdt_lr_12model."
        )

    SV02_TABLE_DIR = SV02_PROJECT / "results/tables/SV02_SV05_scVI_LR"
    SV02_FIGURE_DIR = SV02_PROJECT / "results/figures/SV02_SV05_scVI_LR"
    SV02_PREDICTION_DIR = SV02_PROJECT / "predictions/SV02_SV05_scVI_LR"
    SV02_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    SV02_PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

    SV02_INPUTS = {
        "spl": SV02_PROJECT / "manifests/CrossAtlas_D2_SPL_cohorts.h5ad",
        "h1": SV02_PROJECT / "features/H1_song_inductive_TCRfree_backbone.csv",
        "model": SV02_PROJECT / "models/SV02_SV05_scVI_LR/SV02",
        "classifier": SV02_PROJECT / "models/SV02_SV05_scVI_LR/SV02_latent_LR.joblib",
        "thresholds": SV02_TABLE_DIR / "SV02_SV05_source_frozen_thresholds.csv",
        "figure1_coordinates": (
            SV02_PROJECT.parent
            / "results/figures/Figure1_dataset2_SPL_UMAP"
            / "Figure1A_cell_coordinates_and_groups.csv"
        ),
    }
    sv02_missing = [
        str(path) for path in SV02_INPUTS.values()
        if not (path.is_dir() if path == SV02_INPUTS["model"] else path.is_file())
    ]
    if sv02_missing:
        raise FileNotFoundError(
            "Missing completed frozen artifacts; do not rerun training:\n- "
            + "\n- ".join(sv02_missing)
        )

    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    SV02_FONT = next(
        (font for font in ("Arial", "Helvetica", "Liberation Sans")
         if font in available_fonts),
        None,
    )
    if SV02_FONT is None:
        raise RuntimeError(
            "Install Arial, Helvetica, or Liberation Sans before plotting; "
            "no unapproved font fallback will be used."
        )
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [SV02_FONT],
        "font.weight": "normal",
        "text.color": "black",
        "axes.labelcolor": "black",
        "axes.titlecolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })

    SV02_GROUP_ORDER = ["paired_abT", "orphan_TRG", "orphan_TRD", "paired_gdT"]
    SV02_GROUP_LABELS = {
        "paired_abT": "Paired αβT",
        "orphan_TRG": "Orphan TRG; no TRA/TRB",
        "orphan_TRD": "Orphan TRD; no TRA/TRB",
        "paired_gdT": "Paired γδT",
    }
    SV02_GROUP_COLORS = {
        "paired_abT": "#0057B8",
        "orphan_TRG": "#00A6D6",
        "orphan_TRD": "#D73027",
        "paired_gdT": "#7B2CBF",
    }
    SV02_BACKGROUND_COLOR = "#D9D9D9"
    SV02_UNSELECTED_COLOR = "#BDBDBD"
    SV02_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def sv02_sha256(path, chunk_size=2**20):
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(chunk_size), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def sv02_save_figure(fig, stem):
        pdf_path = SV02_FIGURE_DIR / f"{stem}.pdf"
        png_path = SV02_FIGURE_DIR / f"{stem}.png"
        fig.savefig(pdf_path, bbox_inches="tight")
        fig.savefig(png_path, dpi=350, bbox_inches="tight")
        return pdf_path, png_path

    print("PROJECT:", SV02_PROJECT)
    print("Frozen model:", SV02_MODEL_ID)
    print("Device:", SV02_DEVICE, "| font:", SV02_FONT)
    '''),
    code(r'''
    # U2. Reload frozen SV02, score the four SPL receptor-evidence groups, and join UMAP.
    sv02_spl = ad.read_h5ad(SV02_INPUTS["spl"])
    sv02_spl.obs_names = sv02_spl.obs_names.astype(str)
    sv02_spl.var_names = sv02_spl.var_names.astype(str)
    if not sv02_spl.obs_names.is_unique or not sv02_spl.var_names.is_unique:
        raise AssertionError("SPL cell IDs and gene names must be unique")

    sv02_required_obs = {
        "donor", "barcode_raw", "immutable_cohort",
        "n_TRA", "n_TRB", "n_TRG", "n_TRD",
    }
    sv02_missing_obs = sv02_required_obs - set(sv02_spl.obs.columns)
    if sv02_missing_obs:
        raise KeyError("SPL obs lacks: " + ", ".join(sorted(sv02_missing_obs)))

    sv02_h1 = pd.read_csv(SV02_INPUTS["h1"])["gene"].astype(str).tolist()
    if len(sv02_h1) != 2000 or len(set(sv02_h1)) != 2000:
        raise AssertionError("SV02 requires the immutable 2,000-gene H1 backbone")
    if any(gene.startswith(("TRA", "TRB", "TRG", "TRD")) for gene in sv02_h1):
        raise AssertionError("SV02 H1 must be TCR-free")
    sv02_missing_genes = set(sv02_h1) - set(sv02_spl.var_names)
    if sv02_missing_genes:
        raise KeyError(f"SPL lacks {len(sv02_missing_genes)} SV02 H1 genes")

    sv02_obs = sv02_spl.obs.copy()
    sv02_assay_mask = sv02_obs["donor"].astype(str).isin(SV02_SPL_DONORS)
    sv02_cohort = sv02_obs["immutable_cohort"].astype(str)
    sv02_group = pd.Series("excluded", index=sv02_obs.index, dtype="object")
    sv02_group.loc[sv02_assay_mask & sv02_cohort.eq("gold_B_paired_abT")] = "paired_abT"
    sv02_group.loc[sv02_assay_mask & sv02_cohort.eq("query_orphan_TRG")] = "orphan_TRG"
    sv02_group.loc[sv02_assay_mask & sv02_cohort.eq("candidate_orphan_TRD_nonoverlap")] = "orphan_TRD"
    sv02_group.loc[sv02_assay_mask & sv02_cohort.eq("gold_B_paired_gdT")] = "paired_gdT"
    sv02_obs["receptor_evidence_group"] = sv02_group

    for sv02_chain in ("TRA", "TRB", "TRG", "TRD"):
        sv02_obs[f"n_{sv02_chain}"] = pd.to_numeric(
            sv02_obs[f"n_{sv02_chain}"], errors="coerce"
        ).fillna(0).astype(int)
    sv02_eval = sv02_obs.loc[
        sv02_obs.receptor_evidence_group.isin(SV02_GROUP_ORDER)
    ].copy()
    sv02_expected_counts = {
        "paired_abT": 4668, "orphan_TRG": 87,
        "orphan_TRD": 65, "paired_gdT": 378,
    }
    sv02_observed_counts = sv02_eval.receptor_evidence_group.value_counts().to_dict()
    if sv02_observed_counts != sv02_expected_counts:
        raise AssertionError(
            "Frozen SPL receptor groups changed. Expected "
            f"{sv02_expected_counts}; observed {sv02_observed_counts}"
        )
    if not ((sv02_eval.receptor_evidence_group.eq("paired_abT")) <=
            (sv02_eval.n_TRA.ge(1) & sv02_eval.n_TRB.ge(1) &
             sv02_eval.n_TRG.eq(0) & sv02_eval.n_TRD.eq(0))).all():
        raise AssertionError("paired_abT chain logic changed")
    if not ((sv02_eval.receptor_evidence_group.eq("paired_gdT")) <=
            (sv02_eval.n_TRG.ge(1) & sv02_eval.n_TRD.ge(1) &
             sv02_eval.n_TRA.eq(0) & sv02_eval.n_TRB.eq(0))).all():
        raise AssertionError("paired_gdT chain logic changed")
    if not ((sv02_eval.receptor_evidence_group.eq("orphan_TRD")) <=
            (sv02_eval.n_TRD.ge(1) & sv02_eval.n_TRG.eq(0) &
             sv02_eval.n_TRA.eq(0) & sv02_eval.n_TRB.eq(0))).all():
        raise AssertionError("orphan_TRD chain logic changed")
    if not ((sv02_eval.receptor_evidence_group.eq("orphan_TRG")) <=
            (sv02_eval.n_TRG.ge(1) & sv02_eval.n_TRD.eq(0) &
             sv02_eval.n_TRA.eq(0) & sv02_eval.n_TRB.eq(0))).all():
        raise AssertionError("orphan_TRG chain logic changed")

    def sv02_counts_matrix(adata, cells, genes):
        row_index = adata.obs_names.get_indexer(pd.Index(cells).astype(str))
        column_index = adata.var_names.get_indexer(pd.Index(genes).astype(str))
        if (row_index < 0).any() or (column_index < 0).any():
            raise KeyError("Missing cells or genes during SV02 encoding")
        if "counts" in adata.layers:
            source = adata.layers["counts"]
        elif "raw_counts" in adata.layers:
            source = adata.layers["raw_counts"]
        elif adata.raw is not None and adata.raw.shape[1] == adata.shape[1]:
            source = adata.raw.X
        else:
            source = adata.X
        return source[row_index, :][:, column_index]

    sv02_eval_ids = sv02_eval.index.astype(str).tolist()
    sv02_representative_ids = sv02_eval_ids[:min(64, len(sv02_eval_ids))]
    sv02_rep_matrix = sv02_counts_matrix(
        sv02_spl, sv02_representative_ids, sv02_h1
    )
    sv02_rep = ad.AnnData(
        X=sv02_rep_matrix.copy(),
        obs=pd.DataFrame(index=pd.Index(sv02_representative_ids, name="cell_id")),
        var=pd.DataFrame(index=pd.Index(sv02_h1, name="gene")),
    )
    scvi.model.SCVI.setup_anndata(sv02_rep)
    sv02_vae = scvi.model.SCVI.load(SV02_INPUTS["model"], adata=sv02_rep)
    sv02_vae.to_device(str(SV02_DEVICE))
    sv02_vae.module.eval()
    sv02_classifier = joblib.load(SV02_INPUTS["classifier"])

    @torch.inference_mode()
    def sv02_encode_frozen(model, adata, cells, genes):
        chunks = []
        for start in range(0, len(cells), SV02_ENCODE_BATCH_SIZE):
            batch_cells = cells[start:start + SV02_ENCODE_BATCH_SIZE]
            matrix = sv02_counts_matrix(adata, batch_cells, genes)
            dense = matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix)
            values = dense.ravel()
            if not np.isfinite(values).all() or (values < 0).any():
                raise AssertionError("SV02 input counts contain invalid values")
            if len(values) and np.max(np.abs(values - np.rint(values))) >= 1e-4:
                raise AssertionError("SV02 requires raw integer counts")
            x = torch.as_tensor(dense, dtype=torch.float32, device=SV02_DEVICE)
            batch_index = torch.zeros(
                (len(batch_cells), 1), dtype=torch.int64, device=SV02_DEVICE
            )
            output = model.module.inference(x=x, batch_index=batch_index, n_samples=1)
            if "qz" in output and hasattr(output["qz"], "loc"):
                latent = output["qz"].loc
            elif "qz_m" in output:
                latent = output["qz_m"]
            else:
                raise RuntimeError(
                    "Unsupported scvi-tools inference output keys: "
                    f"{list(output)}"
                )
            chunks.append(latent.detach().cpu().numpy())
        return pd.DataFrame(
            np.vstack(chunks), index=pd.Index(cells, name="cell_id")
        )

    # Version guard: manual unseen-cell encoding must reproduce the public scvi API.
    sv02_reference_latent = np.asarray(
        sv02_vae.get_latent_representation(
            adata=sv02_rep,
            indices=np.arange(sv02_rep.n_obs),
            batch_size=min(SV02_ENCODE_BATCH_SIZE, sv02_rep.n_obs),
        )
    )
    sv02_manual_latent = sv02_encode_frozen(
        sv02_vae, sv02_rep, sv02_representative_ids, sv02_h1
    ).to_numpy()
    if not np.allclose(sv02_reference_latent, sv02_manual_latent, rtol=1e-4, atol=1e-4):
        raise RuntimeError(
            "Frozen SV02 encoder does not reproduce get_latent_representation "
            "with this scvi-tools version; stop before scoring."
        )

    sv02_latent = sv02_encode_frozen(sv02_vae, sv02_spl, sv02_eval_ids, sv02_h1)
    sv02_probability = sv02_classifier.predict_proba(
        sv02_latent.loc[sv02_eval_ids]
    )[:, 1]
    if not np.isfinite(sv02_probability).all() or not (
        (sv02_probability >= 0) & (sv02_probability <= 1)
    ).all():
        raise AssertionError("SV02 probabilities must be finite and within [0, 1]")

    sv02_threshold_table = pd.read_csv(SV02_INPUTS["thresholds"])
    sv02_threshold_row = sv02_threshold_table.loc[
        sv02_threshold_table.model_id.astype(str).eq(SV02_MODEL_ID)
        & sv02_threshold_table.operating_point.astype(str).eq(SV02_OPERATING_POINT)
    ]
    if len(sv02_threshold_row) != 1:
        raise AssertionError(
            f"Expected one {SV02_MODEL_ID}/{SV02_OPERATING_POINT} threshold row"
        )
    SV02_FROZEN_THRESHOLD = float(sv02_threshold_row.iloc[0]["threshold"])

    sv02_predictions = sv02_eval[[
        "donor", "barcode_raw", "receptor_evidence_group", "immutable_cohort",
        "n_TRA", "n_TRB", "n_TRG", "n_TRD",
    ]].copy()
    sv02_predictions["cell_id"] = sv02_predictions.index.astype(str)
    sv02_predictions["model_id"] = SV02_MODEL_ID
    sv02_predictions["p_gdT"] = sv02_probability
    sv02_predictions["operating_point"] = SV02_OPERATING_POINT
    sv02_predictions["frozen_threshold"] = SV02_FROZEN_THRESHOLD
    sv02_predictions["above_frozen_threshold"] = (
        sv02_predictions.p_gdT.ge(SV02_FROZEN_THRESHOLD)
    )

    sv02_coordinates = pd.read_csv(SV02_INPUTS["figure1_coordinates"])
    sv02_required_coordinates = {
        "cell_id", "UMAP1", "UMAP2", "Figure1_group",
        "Manually_curated_celltype", "source_receptor_validation_group",
    }
    if not sv02_required_coordinates.issubset(sv02_coordinates.columns):
        raise KeyError(
            "Figure 1A coordinates lack: "
            + ", ".join(sorted(sv02_required_coordinates - set(sv02_coordinates.columns)))
        )
    if len(sv02_coordinates) != 38_727 or not sv02_coordinates.cell_id.is_unique:
        raise AssertionError("Expected the immutable 38,727-cell Figure 1A coordinate table")
    sv02_coordinates["cell_id"] = sv02_coordinates["cell_id"].astype(str)
    sv02_crosswalk = pd.DataFrame({
        "cell_id": sv02_spl.obs_names.astype(str),
        "barcode_raw": sv02_spl.obs["barcode_raw"].astype(str).to_numpy(),
    })
    if not sv02_crosswalk.cell_id.is_unique or not sv02_crosswalk.barcode_raw.is_unique:
        raise AssertionError("SPL cell_id/barcode_raw crosswalk must be one-to-one")
    if set(sv02_coordinates.cell_id) != set(sv02_crosswalk.barcode_raw):
        raise AssertionError("Figure 1A coordinates and SPL barcode_raw sets differ")

    sv02_coordinate_payload = sv02_coordinates.rename(columns={"cell_id": "barcode_raw"})
    sv02_overlay = sv02_predictions.reset_index(drop=True).merge(
        sv02_coordinate_payload,
        on="barcode_raw", how="left", validate="one_to_one",
    )
    if len(sv02_overlay) != 5_198 or not sv02_overlay[["UMAP1", "UMAP2"]].notna().all().all():
        raise AssertionError("All 5,198 SV02-evaluated cells must map to Figure 1A UMAP")

    SV02_OVERLAY_PATH = (
        SV02_TABLE_DIR / "SV02_SPL_predictions_with_Figure1_UMAP_coordinates.csv"
    )
    sv02_overlay.to_csv(SV02_OVERLAY_PATH, index=False)
    sv02_counts = pd.crosstab(
        sv02_overlay.donor.astype(str),
        sv02_overlay.receptor_evidence_group,
        margins=True,
    ).reindex(columns=SV02_GROUP_ORDER + ["All"], fill_value=0)
    display(sv02_counts)
    print(
        "Song-frozen SV02 threshold:", f"{SV02_FROZEN_THRESHOLD:.6f}",
        "| above threshold:", int(sv02_overlay.above_frozen_threshold.sum()),
        "/", len(sv02_overlay),
    )
    '''),
    code(r'''
    # U3. Overall SPL UMAP: evidence groups, continuous SV02 score, threshold calls.
    def sv02_draw_background(ax):
        ax.scatter(
            sv02_coordinates.UMAP1, sv02_coordinates.UMAP2,
            s=0.7, c=SV02_BACKGROUND_COLOR, alpha=0.16,
            linewidths=0, rasterized=True, zorder=1,
        )

    def sv02_clean_umap_axis(ax, title):
        ax.set_title(title, loc="left", fontweight="normal", color="black")
        ax.set_xlabel("UMAP 1", color="black")
        ax.set_ylabel("UMAP 2", color="black")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    sv02_umap_fig, sv02_umap_axes = plt.subplots(1, 3, figsize=(15.8, 5.1))

    sv02_draw_background(sv02_umap_axes[0])
    for group_name in SV02_GROUP_ORDER:
        group_frame = sv02_overlay.loc[
            sv02_overlay.receptor_evidence_group.eq(group_name)
        ]
        sv02_umap_axes[0].scatter(
            group_frame.UMAP1, group_frame.UMAP2,
            s=5.0 if group_name != "paired_abT" else 2.0,
            c=SV02_GROUP_COLORS[group_name], alpha=0.82,
            linewidths=0, rasterized=True, zorder=3,
            label=f"{SV02_GROUP_LABELS[group_name]} (n={len(group_frame):,})",
        )
    sv02_clean_umap_axis(sv02_umap_axes[0], "a  SV02 evaluation groups")
    sv02_umap_axes[0].legend(
        frameon=False, fontsize=7, markerscale=2.0,
        loc="upper left", bbox_to_anchor=(0.0, -0.04),
    )

    sv02_draw_background(sv02_umap_axes[1])
    sv02_score_order = sv02_overlay.sort_values("p_gdT")
    sv02_score_artist = sv02_umap_axes[1].scatter(
        sv02_score_order.UMAP1, sv02_score_order.UMAP2,
        s=3.0, c=sv02_score_order.p_gdT, cmap="viridis",
        vmin=0, vmax=1, alpha=0.86, linewidths=0,
        rasterized=True, zorder=3,
    )
    sv02_colorbar = sv02_umap_fig.colorbar(
        sv02_score_artist, ax=sv02_umap_axes[1], fraction=0.046, pad=0.02
    )
    sv02_colorbar.set_label("Frozen SV02 predicted γδ probability", color="black")
    sv02_colorbar.ax.tick_params(colors="black")
    sv02_clean_umap_axis(sv02_umap_axes[1], "b  Continuous frozen SV02 probability")

    sv02_draw_background(sv02_umap_axes[2])
    sv02_below = sv02_overlay.loc[~sv02_overlay.above_frozen_threshold]
    sv02_umap_axes[2].scatter(
        sv02_below.UMAP1, sv02_below.UMAP2,
        s=1.5, c=SV02_UNSELECTED_COLOR, alpha=0.22,
        linewidths=0, rasterized=True, zorder=2,
    )
    for group_name in SV02_GROUP_ORDER:
        positive_frame = sv02_overlay.loc[
            sv02_overlay.above_frozen_threshold
            & sv02_overlay.receptor_evidence_group.eq(group_name)
        ]
        sv02_umap_axes[2].scatter(
            positive_frame.UMAP1, positive_frame.UMAP2,
            s=9.0, c=SV02_GROUP_COLORS[group_name], alpha=0.95,
            linewidths=0.25, edgecolors="black", rasterized=True, zorder=4,
            label=f"{SV02_GROUP_LABELS[group_name]} (n={len(positive_frame):,})",
        )
    sv02_clean_umap_axis(
        sv02_umap_axes[2],
        f"c  Predicted γδT at Song-frozen threshold ({SV02_FROZEN_THRESHOLD:.3f})",
    )
    sv02_umap_axes[2].legend(
        frameon=False, fontsize=7, markerscale=1.7,
        loc="upper left", bbox_to_anchor=(0.0, -0.04),
    )

    sv02_umap_fig.suptitle(
        "Frozen SV02 predictions on the existing Dataset 2 SPL RNA-only UMAP",
        y=1.01, fontsize=13, fontweight="normal", color="black",
    )
    sv02_umap_fig.text(
        0.5, -0.08,
        "Grey points are spatial context. Only the 5,198 receptor-evidence cells were scored; "
        "UMAP location is descriptive, not an independent accuracy test.",
        ha="center", va="top", fontsize=8, color="black",
    )
    sv02_umap_fig.tight_layout()
    SV02_OVERVIEW_PDF, SV02_OVERVIEW_PNG = sv02_save_figure(
        sv02_umap_fig, "SV02_SPL_predictions_on_Figure1_dataset2_SPL_UMAP"
    )
    plt.show()
    plt.close(sv02_umap_fig)
    '''),
    code(r'''
    # U4. Donor-faceted SV02 predictions and machine-readable completion manifest.
    sv02_donor_fig, sv02_donor_axes = plt.subplots(2, 3, figsize=(14.5, 8.6))
    sv02_donor_axes = sv02_donor_axes.ravel()
    for ax, donor in zip(sv02_donor_axes, SV02_SPL_DONORS):
        sv02_draw_background(ax)
        donor_frame = sv02_overlay.loc[sv02_overlay.donor.astype(str).eq(donor)]
        for group_name in SV02_GROUP_ORDER:
            group_frame = donor_frame.loc[
                donor_frame.receptor_evidence_group.eq(group_name)
            ]
            ax.scatter(
                group_frame.UMAP1, group_frame.UMAP2,
                s=2.0 if group_name == "paired_abT" else 5.0,
                c=SV02_GROUP_COLORS[group_name], alpha=0.38,
                linewidths=0, rasterized=True, zorder=2,
            )
            positive_frame = group_frame.loc[group_frame.above_frozen_threshold]
            ax.scatter(
                positive_frame.UMAP1, positive_frame.UMAP2,
                s=13.0, c=SV02_GROUP_COLORS[group_name], alpha=0.98,
                linewidths=0.35, edgecolors="black",
                rasterized=True, zorder=4,
            )
        positive_n = int(donor_frame.above_frozen_threshold.sum())
        sv02_clean_umap_axis(
            ax, f"{donor}, {positive_n:,}/{len(donor_frame):,} above threshold"
        )

    sv02_donor_axes[-1].axis("off")
    sv02_legend_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=5,
            markerfacecolor=SV02_GROUP_COLORS[group_name], markeredgecolor="none",
            label=SV02_GROUP_LABELS[group_name],
        )
        for group_name in SV02_GROUP_ORDER
    ]
    sv02_legend_handles.append(
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=6,
            markerfacecolor="white", markeredgecolor="black",
            label="Black edge: above Song-frozen threshold",
        )
    )
    sv02_donor_axes[-1].legend(
        handles=sv02_legend_handles, frameon=False, loc="center", fontsize=9
    )
    sv02_donor_fig.suptitle(
        "Donor-faceted frozen SV02 predictions on the existing SPL UMAP",
        fontsize=13, fontweight="normal", color="black", y=1.01,
    )
    sv02_donor_fig.tight_layout()
    SV02_DONOR_PDF, SV02_DONOR_PNG = sv02_save_figure(
        sv02_donor_fig, "SV02_SPL_donor_faceted_predictions_on_Figure1_UMAP"
    )
    plt.show()
    plt.close(sv02_donor_fig)

    sv02_output_paths = [
        SV02_OVERLAY_PATH,
        SV02_OVERVIEW_PDF, SV02_OVERVIEW_PNG,
        SV02_DONOR_PDF, SV02_DONOR_PNG,
    ]
    if not all(path.is_file() and path.stat().st_size > 0 for path in sv02_output_paths):
        raise AssertionError("One or more SV02 SPL UMAP outputs were not written")

    sv02_manifest = {
        "status": "complete",
        "completed_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "model_id": SV02_MODEL_ID,
        "formal_contract": {
            "label_regime": "L2_paired_plus_TRD_equal",
            "hvg_regime": "H1_song_inductive",
            "gene_regime": "G1_TCR_free",
        },
        "threshold_operating_point": SV02_OPERATING_POINT,
        "threshold": SV02_FROZEN_THRESHOLD,
        "threshold_selection_dataset": "Song nested donor OOF",
        "spl_used_for_training": False,
        "spl_used_for_threshold_selection": False,
        "umap_recomputed": False,
        "n_scored": int(len(sv02_overlay)),
        "n_above_threshold": int(sv02_overlay.above_frozen_threshold.sum()),
        "group_counts": {key: int(value) for key, value in sv02_observed_counts.items()},
        "donors": list(SV02_SPL_DONORS),
        "font": SV02_FONT,
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "anndata": ad.__version__,
            "scvi_tools": scvi.__version__,
            "torch": torch.__version__,
        },
        "inputs": {
            key: str(path) for key, path in SV02_INPUTS.items()
        },
        "outputs": {
            str(path.relative_to(SV02_PROJECT)): sv02_sha256(path)
            for path in sv02_output_paths
        },
    }
    SV02_MANIFEST_PATH = SV02_TABLE_DIR / "SV02_SPL_UMAP_completion_manifest.json"
    SV02_MANIFEST_PATH.write_text(
        json.dumps(sv02_manifest, indent=2), encoding="utf-8"
    )
    print("Saved outputs:")
    for path in [*sv02_output_paths, SV02_MANIFEST_PATH]:
        print(" -", path)
    display(pd.Series(sv02_manifest).drop(["inputs", "outputs", "software"]))
    '''),
]


notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3 (scvi)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "gdt_stage": "SV02_SV05_strict_inductive_scVI_LR",
        "authoritative_generator": "build_sv02_sv05_scvi_lr_notebook.py",
    },
)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, OUTPUT)
print(OUTPUT)
