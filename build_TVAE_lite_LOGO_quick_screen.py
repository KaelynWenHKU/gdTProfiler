#!/usr/bin/env python3
"""Generate the single-arm TVAE(0) Song-LOGO plus frozen-SPL notebook.

This Python generator is authoritative. Do not hand-edit the generated notebook.
"""

from __future__ import annotations

import os
from pathlib import Path
import textwrap

import nbformat as nbf


HERE = Path(__file__).resolve().parent
OUTPUT = Path(os.environ.get(
    "GDT_NOTEBOOK_OUTPUT",
    HERE / "notebooks/TVAE_lite_LOGO_quick_screen.ipynb",
))


def md(source: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(source).strip())


def code(source: str):
    return nbf.v4.new_code_cell(textwrap.dedent(source).strip())


cells = [
    md(r'''
    # TVAE_lite_LOGO_quick_screen

    ## tl;dr

    This notebook contains **one TVAE only: `TVAE(0)`**. It reproduces Professor
    Yuanhua's registered LOGO mechanism: labelled-only representation fitting,
    the original incomplete differential Adam groups, 500 epochs, and
    input-dimension-upweighted BCE. Its input is the exact controlled-M05 gene
    panel, including the nine prespecified TCR receptor genes.

    Four Song donors are evaluated out of fold. A fifth model is then fitted on
    all Song supervision, frozen, and transported without target fitting to SPL.
    Both Song and SPL are evaluated against M05 on four identical endpoints:
    paired gamma-delta T, orphan TRD, orphan TRG, and their combined endpoint.
    '''),
    md(r'''
    ## Context & Methods

    ### Locked model definition

    | Component | Locked value |
    |---|---|
    | Model | `TVAE(0)` only |
    | Genes | exact controlled-M05 panel: H1 + 9 prespecified TCR genes |
    | Split | Song leave-one-donor-out (`hd1`-`hd4`) |
    | Representation population | L2-labelled fit-donor cells only |
    | Optimizer | original incomplete differential Adam |
    | Loss | summed Gaussian + 0.1 summed KL + `input_dim`-upweighted BCE |
    | External evaluation | final Song-only model frozen before SPL is opened |
    '''),
    md(r'''
    ### Evidence tiers and comparison rule

    Paired gamma-delta versus paired alpha-beta is the strict receptor-gold
    endpoint. Orphan TRD and orphan TRG are reported separately as partial-
    receptor sensitivity endpoints; their labels are not silently promoted to
    paired-receptor gold. Every endpoint uses paired alpha-beta cells as the
    negative class. TVAE(0) and M05 are compared only within the same dataset.
    '''),
    md(r'''
    ### Clean Woody execution

    ```bash
    cd /data/users/kaelynwen/gdT_project/gdt_lr_12model
    python build_TVAE_lite_LOGO_quick_screen.py
    python -m jupyter nbconvert --execute --to notebook --inplace \
      --ExecutePreprocessor.timeout=-1 notebooks/TVAE_lite_LOGO_quick_screen.ipynb
    ```

    Completed donor units are checkpointed and reused after interruption.
    The generator is authoritative; the notebook is the generated runnable artifact.
    '''),
    code(r'''
    # 1. Imports, paths, deterministic setup, and publication figure style
    from pathlib import Path
    import gc, hashlib, json, os, platform, random, sys, time, zlib

    import anndata as ad
    import joblib
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    import torch
    import yaml
    from torch import nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.metrics import (
        average_precision_score, brier_score_loss, precision_recall_curve,
        roc_auc_score, roc_curve,
    )

    REMOTE_PROJECT = Path("/data/users/kaelynwen/gdT_project/gdt_lr_12model")
    PROJECT = Path(os.environ.get("GDT_LR_PROJECT", REMOTE_PROJECT)).resolve()
    if not PROJECT.exists():
        candidate = Path.cwd().resolve()
        PROJECT = candidate.parent if candidate.name == "notebooks" else candidate
    if str(PROJECT).startswith("/home/kaelynwen"):
        raise RuntimeError("Use /data/users/kaelynwen for Woody project work")
    sys.path.insert(0, str(PROJECT / "src"))
    from gdt_lr_utils import (
        donor_balanced_weights, fit_lr, get_counts, is_tcr_gene,
        load_model_bundle, make_training_table, predict_lr, select_hvgs,
        sha256_file, threshold_max_f1,
    )

    ANALYSIS = "TVAE_lite_LOGO_quick_screen"
    TABLE_DIR = PROJECT / "results/tables" / ANALYSIS
    FIGURE_DIR = PROJECT / "results/figures" / ANALYSIS
    PREDICTION_DIR = PROJECT / "predictions" / ANALYSIS
    CHECKPOINT_DIR = PROJECT / "checkpoints" / ANALYSIS
    for directory in [TABLE_DIR, FIGURE_DIR, PREDICTION_DIR, CHECKPOINT_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N_THREADS = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
    torch.set_num_threads(N_THREADS)
    permitted_fonts = ["Arial", "Helvetica", "Liberation Sans"]
    FONT = next((name for name in permitted_fonts
                 if any(item.name == name for item in font_manager.fontManager.ttflist)), None)
    if FONT is None:
        raise RuntimeError("Install Arial, Helvetica, or Liberation Sans")
    mpl.rcParams.update({
        "font.family": FONT, "font.weight": "normal", "pdf.fonttype": 42,
        "ps.fonttype": 42, "text.color": "black", "axes.labelcolor": "black",
        "axes.edgecolor": "black", "axes.titlecolor": "black",
        "xtick.color": "black", "ytick.color": "black", "axes.grid": False,
    })
    print("PROJECT:", PROJECT, "DEVICE:", DEVICE, "CPUs:", N_THREADS)
    '''),
    code(r'''
    # 2. Locked single-model contract
    SEED = 20260806
    MODEL_NAME = "TVAE(0)"
    FEATURE_ARM = "G2_TCR_inclusive"
    LABEL_REGIME = "L2_paired_plus_TRD_equal"
    LAMBDA_TRD = 1.0
    BETA_KL = 0.10
    BATCH_SIZE = 512
    FIXED_EPOCHS = 500
    EXPECTED_LOGO_FITS = 4
    EXPECTED_TOTAL_TVAE_FITS = 5
    SPL_GOLD_DONORS = ("A29", "A31", "A35", "A36", "A37")
    LOGO_UNIT_SEEDS = {"hd1":1180388846,"hd2":1566875481,"hd3":744320715,"hd4":914487917}
    RESUME = True
    run_contract = {
        "analysis": ANALYSIS, "seed": SEED, "feature_arm": FEATURE_ARM,
        "model": MODEL_NAME,
        "label_regime": LABEL_REGIME, "lambda_TRD": LAMBDA_TRD,
        "loss": "common summed Gaussian + 0.1 summed KL + input_dim weighted BCE",
        "epochs": FIXED_EPOCHS,
        "representation_population": "L2-labelled fit-donor cells only",
        "optimizer": "Professor Yuanhua original incomplete differential Adam",
        "optimizer_groups": "encoder 1e-4; decoder 1e-4; predictor 1e-2 with weight_decay 10",
        "expected_logo_fits": EXPECTED_LOGO_FITS,
        "expected_total_tvae_fits": EXPECTED_TOTAL_TVAE_FITS,
        "logo_unit_seeds": LOGO_UNIT_SEEDS,
        "external_design": "final Song-only fit frozen before SPL is opened",
        "comparison_model": "M05",
        "target_expression_used_for_training_or_selection": False,
        "target_labels_used_for_training_or_selection": False,
    }
    RUN_SIGNATURE = hashlib.sha256(
        json.dumps(run_contract, sort_keys=True).encode()
    ).hexdigest()
    run_contract["run_signature"] = RUN_SIGNATURE
    (TABLE_DIR / "registered_run_contract.json").write_text(
        json.dumps(run_contract, indent=2), encoding="utf-8"
    )
    display(pd.Series(run_contract, name="registered_value").to_frame())
    '''),
    code(r'''
    # 3. Load Song and the exact controlled-M05 panel; keep SPL unopened
    INPUTS = {
        "song": PROJECT / "manifests/Song_D1_cohorts.h5ad",
        "spl": PROJECT / "manifests/CrossAtlas_D2_SPL_cohorts.h5ad",
        "h1": PROJECT / "features/H1_song_inductive_TCRfree_backbone.csv",
        "tcr": PROJECT / "features/prespecified_TCR_panel.csv",
        "registry": PROJECT / "config/model_registry.csv",
        "grid": PROJECT / "config/lr_grid.yaml",
        "c_choices": PROJECT / "results/tables/final_C_choices.csv",
        "m05_registry": PROJECT / "results/tables/controlled_fit_registry.csv",
        "m05_bundle": PROJECT / "models/controlled_M05.joblib",
        "m05_song_reference": PROJECT / "predictions/song_controlled_oof.parquet",
        "m05_spl_reference": PROJECT / "predictions/SPL_external_gold_predictions.parquet",
    }
    missing = [str(path) for path in INPUTS.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing immutable inputs:\n" + "\n".join(missing))
    input_audit = pd.DataFrame([
        {"artifact": key, "path": str(path), "bytes": path.stat().st_size,
         "sha256": sha256_file(path)} for key, path in INPUTS.items()
    ])
    input_audit.to_csv(TABLE_DIR / "input_artifact_audit.csv", index=False)

    song = ad.read_h5ad(INPUTS["song"])
    assert song.obs_names.is_unique and song.var_names.is_unique
    assert {"donor", "immutable_cohort", "qc_Tlineage_pass"}.issubset(song.obs.columns)
    h1 = pd.read_csv(INPUTS["h1"])["gene"].astype(str).tolist()
    tcr = pd.read_csv(INPUTS["tcr"])["gene"].astype(str).tolist()
    expected_tcr = {"TRAC", "TRBC1", "TRBC2", "TRDC", "TRGC1", "TRGC2", "TRDV1", "TRDV2", "TRDV3"}
    m05_bundle = load_model_bundle(INPUTS["m05_bundle"])
    genes = list(map(str, m05_bundle["genes"]))
    assert len(h1) == 2000 and len(set(h1)) == 2000
    assert not any(is_tcr_gene(gene) for gene in h1)
    assert len(tcr) == 9 and set(tcr) == expected_tcr and set(h1).isdisjoint(tcr)
    expected_panel = h1 + tcr
    assert len(genes) == 2009 and len(set(genes)) == len(genes)
    assert expected_tcr.issubset(set(genes)), (
        "The controlled-M05 bundle does not contain all nine registered TCR genes"
    )
    panel_order_matches_csv = genes == expected_panel
    missing_from_m05_bundle = sorted(set(expected_panel) - set(genes))
    extra_in_m05_bundle = sorted(set(genes) - set(expected_panel))
    panel_drift_audit = pd.DataFrame(
        [{"comparison":"current_H1_plus_TCR_vs_controlled_M05_bundle",
          "status":"missing_from_M05_bundle","gene":gene}
         for gene in missing_from_m05_bundle]
        + [{"comparison":"current_H1_plus_TCR_vs_controlled_M05_bundle",
            "status":"extra_in_M05_bundle","gene":gene}
           for gene in extra_in_m05_bundle],
        columns=["comparison","status","gene"],
    )
    panel_drift_audit.to_csv(TABLE_DIR/"M05_panel_vs_current_H1_TCR_audit.csv",index=False)
    assert str(m05_bundle["metadata"]["model_id"]) == "M05"
    assert str(m05_bundle["metadata"]["gene_regime"]) == FEATURE_ARM
    assert set(genes).issubset(song.var_names.astype(str))
    configuration_signature = RUN_SIGNATURE
    panel_gene_signature = hashlib.sha256("\n".join(genes).encode()).hexdigest()
    RUN_SIGNATURE = hashlib.sha256(
        "|".join([
            configuration_signature, panel_gene_signature,
            sha256_file(INPUTS["song"]), sha256_file(INPUTS["tcr"]),
            sha256_file(INPUTS["m05_bundle"]),
        ]).encode()
    ).hexdigest()
    run_contract["configuration_signature"] = configuration_signature
    run_contract["panel_gene_signature"] = panel_gene_signature
    run_contract["run_signature"] = RUN_SIGNATURE
    (TABLE_DIR / "registered_run_contract.json").write_text(
        json.dumps(run_contract, indent=2), encoding="utf-8"
    )
    song_training = make_training_table(song.obs, LABEL_REGIME)
    donors = tuple(sorted(song_training.donor.astype(str).unique()))
    assert donors == ("hd1", "hd2", "hd3", "hd4")
    assert len(donors) == EXPECTED_LOGO_FITS
    assert set(song_training.positive_source.unique()) == {"paired_gdT", "orphan_TRD", "paired_abT"}
    display(song_training.groupby(["donor", "positive_source"]).size().unstack(fill_value=0))
    print("TVAE genes:", len(genes), "TCR genes:", tcr)
    print("M05 bundle order matches H1 + TCR CSV order:", panel_order_matches_csv)
    print("Genes missing from M05 bundle relative to current H1 + TCR:",len(missing_from_m05_bundle))
    print("Genes extra in M05 bundle relative to current H1 + TCR:",len(extra_in_m05_bundle))
    print("Using the controlled-M05 bundle gene set and order, which are authoritative for this comparison.")
    print("SPL remains unopened until source_freeze.json exists.")
    '''),
    code(r'''
    # 4. Fold-safe expression preprocessing and supervision mapping
    def raw_matrix(adata, cells, selected_genes):
        cells = pd.Index(cells).astype(str)
        selected_genes = pd.Index(selected_genes).astype(str)
        rows = adata.obs_names.astype(str).get_indexer(cells)
        columns = adata.var_names.astype(str).get_indexer(selected_genes)
        if (rows < 0).any() or (columns < 0).any():
            raise KeyError("Requested cells or genes are absent")
        matrix = get_counts(adata)[rows, :][:, columns]
        return matrix.tocsr() if sp.issparse(matrix) else np.asarray(matrix)

    def log_normalize(matrix, target_sum=1e4):
        matrix = matrix.astype(np.float32)
        if sp.issparse(matrix):
            totals = np.asarray(matrix.sum(axis=1)).ravel()
            out = matrix.multiply((target_sum / np.maximum(totals, 1.0))[:, None]).tocsr()
            out.data = np.log1p(out.data)
            return out
        totals = matrix.sum(axis=1)
        return np.log1p(matrix * (target_sum / np.maximum(totals, 1.0))[:, None])

    class FoldStandardizer:
        def fit(self, matrix):
            self.mean_ = np.asarray(matrix.mean(axis=0)).ravel().astype(np.float32)
            squared = (np.asarray(matrix.power(2).mean(axis=0)).ravel()
                       if sp.issparse(matrix) else np.mean(matrix ** 2, axis=0))
            self.scale_ = np.sqrt(np.maximum(squared - self.mean_ ** 2, 1e-6)).astype(np.float32)
            return self

        def transform(self, matrix):
            dense = matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix)
            return ((dense.astype(np.float32) - self.mean_) / self.scale_).astype(np.float32)

    def transform_expression(adata, cells, selected_genes, standardizer=None, fit=False):
        logged = log_normalize(raw_matrix(adata, cells, selected_genes))
        if fit:
            standardizer = FoldStandardizer().fit(logged)
        if standardizer is None:
            raise ValueError("A fitted fold standardizer is required")
        return standardizer.transform(logged), standardizer

    def representation_cells(adata, fit_table, fit_donors, population):
        if population == "labelled_L2_only":
            return fit_table.index.astype(str).tolist()
        if population == "all_qc_Tlineage_pass":
            mask = (adata.obs.donor.astype(str).isin(fit_donors)
                    & adata.obs.qc_Tlineage_pass.fillna(False).astype(bool))
            return adata.obs_names[mask].astype(str).tolist()
        raise ValueError(population)

    def supervision_arrays(cells, labelled):
        cells = pd.Index(cells).astype(str)
        labelled = labelled.copy(); labelled.index = labelled.index.astype(str)
        positions = cells.get_indexer(labelled.index)
        if (positions < 0).any():
            raise RuntimeError("Every supervised cell must be in the representation population")
        y = np.zeros(len(cells), dtype=np.float32)
        weights = np.ones(len(cells), dtype=np.float32)
        mask = np.zeros(len(cells), dtype=bool)
        y[positions] = labelled.y.to_numpy(np.float32)
        weights[positions] = donor_balanced_weights(
            labelled, LABEL_REGIME, LAMBDA_TRD
        ).to_numpy(np.float32)
        mask[positions] = True
        return y, weights, mask
    '''),
    code(r'''
    # 5. Professor lightweight TVAE(0) and exact optimizer-coverage audit
    class ProfessorLiteTVAE(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU())
            self.fc_z_mean = nn.Linear(128, 32)
            self.fc_z_logstd = nn.Linear(128, 32)
            self.decoder = nn.Sequential(nn.Linear(32, 128), nn.ReLU())
            self.fc_x_mean = nn.Linear(128, input_dim)
            self.predictor = nn.Sequential(nn.Linear(32, 1))

        def encode(self, x):
            hidden = self.encoder(x)
            return self.fc_z_mean(hidden), self.fc_z_logstd(hidden).clamp(-10.0, 5.0)

        def forward(self, x):
            z_mean, z_logstd = self.encode(x)
            z = z_mean + torch.exp(z_logstd) * torch.randn_like(z_mean)
            reconstruction = self.fc_x_mean(self.decoder(z))
            logit = self.predictor(z).squeeze(1)
            return reconstruction, z_mean, z_logstd, logit

    def yuanhua_optimizer(model):
        groups = [
            {"params": list(model.encoder.parameters()), "lr": 1e-4},
            {"params": list(model.decoder.parameters()), "lr": 1e-4},
            {"params": list(model.predictor.parameters()), "lr": 1e-2,
             "weight_decay": 10.0},
        ]

        optimized = {id(parameter) for group in groups for parameter in group["params"]}
        coverage = pd.DataFrame([
            {"parameter": name, "n_scalars": parameter.numel(),
             "optimized": id(parameter) in optimized}
            for name, parameter in model.named_parameters() if parameter.requires_grad
        ])
        missing_trainable_parameters = coverage.loc[~coverage.optimized, "parameter"].tolist()
        expected_missing = {
            "fc_z_mean.weight", "fc_z_mean.bias", "fc_z_logstd.weight",
            "fc_z_logstd.bias", "fc_x_mean.weight", "fc_x_mean.bias",
        }
        assert set(missing_trainable_parameters) == expected_missing
        optimizer = torch.optim.Adam(groups)
        return optimizer, coverage, missing_trainable_parameters
    '''),
    code(r'''
    # 6. Common professor-scale loss, labelled-aware batches, and inference
    def set_seed(seed):
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def professor_loss(reconstruction, x, z_mean, z_logstd, logits, y, weights, labelled_mask):
        reconstruction_per_cell = 0.5 * torch.sum((reconstruction - x) ** 2, dim=1)
        variance = torch.exp(2.0 * z_logstd)
        kl_per_cell = 0.5 * torch.sum(
            z_mean.pow(2) + variance - 1.0 - 2.0 * z_logstd, dim=1
        )
        labelled = labelled_mask.bool()
        if not torch.any(labelled):
            raise RuntimeError("Every batch must contain supervised cells")
        normalized = weights[labelled] / weights[labelled].mean().clamp_min(1e-8)
        classification = torch.mean(
            F.binary_cross_entropy_with_logits(
                logits[labelled], y[labelled], reduction="none"
            ) * normalized
        )
        input_dim = x.shape[1]
        classification_multiplier = input_dim
        total = (reconstruction_per_cell.mean() + BETA_KL * kl_per_cell.mean()
                 + classification_multiplier * classification)
        return total, {
            "loss": float(total.detach().cpu()),
            "reconstruction_sum": float(reconstruction_per_cell.mean().detach().cpu()),
            "kl_sum": float(kl_per_cell.mean().detach().cpu()),
            "classification": float(classification.detach().cpu()),
        }

    def labelled_aware_loader(X, y, weights, labelled_mask, seed):
        dataset = TensorDataset(
            torch.as_tensor(X, dtype=torch.float32),
            torch.as_tensor(y, dtype=torch.float32),
            torch.as_tensor(weights, dtype=torch.float32),
            torch.as_tensor(labelled_mask, dtype=torch.bool),
        )
        labelled = np.flatnonzero(labelled_mask); unlabelled = np.flatnonzero(~labelled_mask)
        rng = np.random.default_rng(int(seed)); rng.shuffle(labelled); rng.shuffle(unlabelled)
        n_batches = max(1, int(np.ceil(len(X) / BATCH_SIZE)))
        batches = []
        for labelled_chunk, unlabelled_chunk in zip(
            np.array_split(labelled, n_batches), np.array_split(unlabelled, n_batches)
        ):
            if len(labelled_chunk) == 0:
                labelled_chunk = rng.choice(labelled, size=1, replace=True)
            batch = np.concatenate([labelled_chunk, unlabelled_chunk]); rng.shuffle(batch)
            batches.append(batch.tolist())
        return DataLoader(dataset, batch_sampler=batches, num_workers=0)

    def fit_model(X, y, weights, labelled_mask, seed):
        set_seed(seed)
        model = ProfessorLiteTVAE(X.shape[1]).to(DEVICE)
        optimizer, coverage, missing = yuanhua_optimizer(model)
        loader = labelled_aware_loader(X, y, weights, labelled_mask, seed)
        history = []
        for epoch in range(1, FIXED_EPOCHS + 1):
            model.train(); parts = []
            for xb, yb, wb, mb in loader:
                xb=xb.to(DEVICE); yb=yb.to(DEVICE); wb=wb.to(DEVICE); mb=mb.to(DEVICE)
                optimizer.zero_grad(set_to_none=True)
                reconstruction, z_mean, z_logstd, logits = model(xb)
                loss, row = professor_loss(
                    reconstruction, xb, z_mean, z_logstd, logits, yb, wb, mb
                )
                loss.backward(); optimizer.step(); parts.append(row)
            history.append({"epoch": epoch, **{
                key: float(np.mean([part[key] for part in parts])) for key in parts[0]
            }})
        model.eval()
        return model, pd.DataFrame(history), coverage, missing

    @torch.inference_mode()
    def head_probability(model, X):
        model.eval(); parts = []
        for batch in DataLoader(torch.as_tensor(X, dtype=torch.float32), batch_size=BATCH_SIZE):
            z_mean, _ = model.encode(batch.to(DEVICE))
            parts.append(torch.sigmoid(model.predictor(z_mean).squeeze(1)).cpu().numpy())
        return np.concatenate(parts)

    def stable_seed(*parts):
        return (SEED + zlib.crc32("|".join(map(str, parts)).encode())) % (2**31 - 1)

    def clear_accelerator():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    '''),
    md(r'''
    ## Results: Song donor-LOGO evaluation

    Every held-out donor is absent from TVAE fitting, standardization, BCE
    supervision, and optimizer updates. The same held-out endpoint cells are
    independently scored by the controlled M05 procedure.
    '''),
    code(r'''
    # 7. Execute or resume the four Professor-Yuanhua TVAE(0) donor folds
    endpoint_cohorts = ["gold_B_paired_abT", "gold_B_paired_gdT",
                        "candidate_orphan_TRD_nonoverlap", "query_orphan_TRG"]
    prediction_parts=[]; history_parts=[]; coverage_parts=[]; runtime_rows=[]
    for outer_donor in donors:
        checkpoint = CHECKPOINT_DIR / f"TVAE(0)__heldout-{outer_donor}.joblib"
        if RESUME and checkpoint.is_file():
            cached = joblib.load(checkpoint)
            if cached.get("run_signature") == RUN_SIGNATURE:
                prediction_parts.append(cached["prediction"]); history_parts.append(cached["history"])
                coverage_parts.append(cached["coverage"]); runtime_rows.append(cached["runtime"])
                print("REUSED", MODEL_NAME, outer_donor); continue
            print("STALE checkpoint ignored:", checkpoint)
        started = time.perf_counter()
        fit = song_training[~song_training.donor.astype(str).eq(outer_donor)].copy()
        rep_cells = fit.index.astype(str).tolist()
        test_mask = (song.obs.donor.astype(str).eq(outer_donor)
                     & song.obs.immutable_cohort.astype(str).isin(endpoint_cohorts))
        test = song.obs.loc[test_mask, ["donor", "immutable_cohort"]].copy()
        X_fit, standardizer = transform_expression(song, rep_cells, genes, fit=True)
        X_test, _ = transform_expression(song, test.index, genes, standardizer=standardizer)
        y_fit, w_fit, labelled_mask = supervision_arrays(rep_cells, fit)
        unit_seed = LOGO_UNIT_SEEDS[outer_donor]
        model, history, coverage, missing = fit_model(X_fit, y_fit, w_fit, labelled_mask, unit_seed)
        prediction = test.copy()
        prediction["p_gdT"] = head_probability(model, X_test)
        prediction["cell_id"] = prediction.index.astype(str); prediction["model"] = MODEL_NAME
        prediction["outer_donor"] = outer_donor; prediction["seed"] = SEED
        prediction = prediction.reset_index(drop=True)
        history = history.assign(model=MODEL_NAME, outer_donor=outer_donor, seed=SEED,
                                 n_fit_representation=len(rep_cells), n_fit_supervised=len(fit))
        coverage = coverage.assign(model=MODEL_NAME, outer_donor=outer_donor,
                                   missing_trainable_parameters="|".join(missing))
        runtime = {"model": MODEL_NAME, "outer_donor": outer_donor,
                   "seconds": float(time.perf_counter()-started),
                   "n_fit_representation": len(rep_cells), "n_fit_supervised": len(fit)}
        joblib.dump({"prediction":prediction,"history":history,"coverage":coverage,
                     "runtime":runtime,"run_signature":RUN_SIGNATURE},checkpoint)
        prediction_parts.append(prediction); history_parts.append(history)
        coverage_parts.append(coverage); runtime_rows.append(runtime)
        print("COMPLETED", MODEL_NAME, outer_donor, f"{runtime['seconds']/60:.1f} min")
        del model, X_fit, X_test; clear_accelerator()
    tvae_song_oof = pd.concat(prediction_parts, ignore_index=True)
    histories = pd.concat(history_parts, ignore_index=True)
    optimizer_coverage = pd.concat(coverage_parts, ignore_index=True)
    runtimes = pd.DataFrame(runtime_rows)
    assert tvae_song_oof.outer_donor.nunique() == EXPECTED_LOGO_FITS
    assert not tvae_song_oof.cell_id.duplicated().any() and len(runtimes) == EXPECTED_LOGO_FITS
    tvae_song_oof.to_parquet(PREDICTION_DIR/"TVAE0_Song_LOGO_endpoint_predictions.parquet",index=False)
    histories.to_csv(TABLE_DIR/"TVAE0_Song_LOGO_training_histories.csv",index=False)
    optimizer_coverage.to_csv(TABLE_DIR/"TVAE0_optimizer_parameter_coverage.csv",index=False)
    runtimes.to_csv(TABLE_DIR/"TVAE0_unit_runtime.csv",index=False)
    display(runtimes)
    '''),
    code(r'''
    # 8. Reproduce controlled M05 donor-OOF scoring on all four Song endpoints
    registry = pd.read_csv(INPUTS["registry"])
    m05_row = registry.loc[registry.model_id.astype(str).eq("M05")].iloc[0]
    c_choices = pd.read_csv(INPUTS["c_choices"])
    m05_C = float(c_choices.loc[c_choices.model_id.astype(str).eq("M05"),"controlled_common_C"].item())
    with open(INPUTS["grid"], encoding="utf-8") as handle: grid = yaml.safe_load(handle)
    assert str(m05_row.label_regime) == LABEL_REGIME and str(m05_row.gene_regime) == FEATURE_ARM
    m05_parts=[]
    for outer_donor in donors:
        fit = song_training[~song_training.donor.astype(str).eq(outer_donor)].copy()
        eligible = (song.obs.index.isin(fit.index)
                    & song.obs.immutable_cohort.astype(str).isin(["gold_B_paired_gdT","gold_B_paired_abT"]))
        fold_h1, _ = select_hvgs(song, int(grid["primary_hvg_k"]), "donor", eligible, exclude_tcr=True)
        fold_genes = fold_h1 + [gene for gene in tcr if gene not in fold_h1]
        model = fit_lr(song, fit, fold_genes, m05_C, LABEL_REGIME, LAMBDA_TRD, int(grid["seed"]))
        test_mask = (song.obs.donor.astype(str).eq(outer_donor)
                     & song.obs.immutable_cohort.astype(str).isin(endpoint_cohorts))
        test = song.obs.loc[test_mask,["donor","immutable_cohort"]].copy()
        probability = predict_lr(model, song, fold_genes, test.index)
        test["p_gdT"] = probability.loc[test.index].to_numpy(); test["cell_id"] = test.index.astype(str)
        test["model"] = "M05"; test["outer_donor"] = outer_donor; test["seed"] = SEED
        m05_parts.append(test.reset_index(drop=True))
    m05_song_oof = pd.concat(m05_parts, ignore_index=True)
    assert not m05_song_oof.cell_id.duplicated().any()
    m05_reference = pd.read_parquet(INPUTS["m05_song_reference"])
    m05_reference = m05_reference[m05_reference.model_id.astype(str).eq("M05")].copy()
    strict_ids = m05_song_oof.loc[m05_song_oof.immutable_cohort.isin(
        ["gold_B_paired_abT","gold_B_paired_gdT"]),"cell_id"]
    observed = m05_song_oof.set_index("cell_id").loc[strict_ids,"p_gdT"]
    expected = m05_reference.assign(cell_id=m05_reference.cell_id.astype(str)).set_index("cell_id").loc[strict_ids,"p_gdT"]
    m05_song_max_abs_difference = float(np.max(np.abs(observed.to_numpy()-expected.to_numpy())))
    assert m05_song_max_abs_difference < 1e-7, m05_song_max_abs_difference
    m05_song_oof.to_parquet(PREDICTION_DIR/"M05_Song_LOGO_endpoint_predictions.parquet",index=False)
    print("M05 strict Song OOF reproduction max |difference|:",m05_song_max_abs_difference)
    '''),
    md(r'''
    ## Source freeze before external evaluation

    The next cell fits one final TVAE(0) on Song only and records its model,
    scaler, gene order and checksum. SPL is not opened until this record exists.
    '''),
    code(r'''
    # 9. Fit/freeze the final Song-only TVAE(0)
    final_path = CHECKPOINT_DIR/"TVAE(0)__final_all_Song.joblib"
    if RESUME and final_path.is_file() and joblib.load(final_path).get("run_signature") == RUN_SIGNATURE:
        final_bundle = joblib.load(final_path); final_runtime_seconds = float(final_bundle["runtime_seconds"])
        print("REUSED final Song TVAE(0)")
    else:
        started=time.perf_counter(); rep_cells=song_training.index.astype(str).tolist()
        X_fit,standardizer=transform_expression(song,rep_cells,genes,fit=True)
        y_fit,w_fit,labelled_mask=supervision_arrays(rep_cells,song_training)
        final_model,final_history,final_coverage,final_missing=fit_model(
            X_fit,y_fit,w_fit,labelled_mask,stable_seed(MODEL_NAME,"final"))
        final_runtime_seconds=float(time.perf_counter()-started)
        final_bundle={"model_state":{k:v.detach().cpu() for k,v in final_model.state_dict().items()},
                      "input_dim":len(genes),"genes":genes,
                      "standardizer":{"mean":standardizer.mean_,"scale":standardizer.scale_},
                      "history":final_history,"coverage":final_coverage,
                      "missing_trainable_parameters":final_missing,
                      "runtime_seconds":final_runtime_seconds,"run_signature":RUN_SIGNATURE}
        joblib.dump(final_bundle,final_path); del final_model,X_fit; clear_accelerator()
    SOURCE_FREEZE_PATH=TABLE_DIR/"source_freeze.json"
    source_freeze={"analysis":ANALYSIS,"model":MODEL_NAME,"run_signature":RUN_SIGNATURE,
                   "genes":len(genes),"m05_bundle_sha256":sha256_file(INPUTS["m05_bundle"]),
                   "final_model_path":str(final_path),"final_model_sha256":sha256_file(final_path),
                   "spl_opened_before_freeze":False,"target_data_used_for_training_or_selection":False,
                   "frozen_utc":pd.Timestamp.now(tz="UTC").isoformat()}
    SOURCE_FREEZE_PATH.write_text(json.dumps(source_freeze,indent=2),encoding="utf-8")
    display(pd.Series(source_freeze,name="value").to_frame())
    '''),
    code(r'''
    # 10. Open SPL after freeze; score the four exact receptor-evidence groups
    assert SOURCE_FREEZE_PATH.is_file()
    spl=ad.read_h5ad(INPUTS["spl"]); assert spl.obs_names.is_unique and spl.var_names.is_unique
    obs=spl.obs.copy(); assay=obs.donor.astype(str).isin(SPL_GOLD_DONORS); cohort=obs.immutable_cohort.astype(str)
    group=pd.Series("excluded",index=obs.index,dtype="object")
    group.loc[assay&cohort.eq("gold_B_paired_abT")]="paired_abT"
    group.loc[assay&cohort.eq("gold_B_paired_gdT")]="paired_gdT"
    group.loc[assay&cohort.eq("candidate_orphan_TRD_nonoverlap")]="orphan_TRD"
    group.loc[assay&cohort.eq("query_orphan_TRG")]="orphan_TRG"
    obs["receptor_evidence_group"]=group
    spl_eval=obs.loc[group.isin(["paired_abT","paired_gdT","orphan_TRD","orphan_TRG"]),
                     ["donor","immutable_cohort","receptor_evidence_group"]].copy()
    expected_counts={"paired_abT":4668,"paired_gdT":378,"orphan_TRD":65,"orphan_TRG":87}
    assert spl_eval.receptor_evidence_group.value_counts().to_dict()==expected_counts
    final_bundle=joblib.load(final_path)
    final_model=ProfessorLiteTVAE(final_bundle["input_dim"]).to(DEVICE)
    final_model.load_state_dict(final_bundle["model_state"]); final_model.eval()
    standardizer=FoldStandardizer(); standardizer.mean_=final_bundle["standardizer"]["mean"]
    standardizer.scale_=final_bundle["standardizer"]["scale"]
    X_spl,_=transform_expression(spl,spl_eval.index,final_bundle["genes"],standardizer=standardizer)
    tvae_spl=spl_eval.copy(); tvae_spl["p_gdT"]=head_probability(final_model,X_spl)
    tvae_spl["cell_id"]=tvae_spl.index.astype(str); tvae_spl["model"]=MODEL_NAME; tvae_spl["seed"]=SEED
    tvae_spl=tvae_spl.reset_index(drop=True)
    m05_probability=predict_lr(m05_bundle["model"],spl,m05_bundle["genes"],spl_eval.index)
    m05_spl=spl_eval.copy(); m05_spl["p_gdT"]=m05_probability.loc[spl_eval.index].to_numpy()
    m05_spl["cell_id"]=m05_spl.index.astype(str); m05_spl["model"]="M05"; m05_spl["seed"]=SEED
    m05_spl=m05_spl.reset_index(drop=True)
    strict_ids=spl_eval.index[spl_eval.immutable_cohort.isin(["gold_B_paired_abT","gold_B_paired_gdT"])].astype(str)
    reference=pd.read_parquet(INPUTS["m05_spl_reference"])
    if "analysis" in reference.columns: reference=reference[reference.analysis.astype(str).str.lower().eq("controlled")]
    reference=reference[reference.model_id.astype(str).eq("M05")].assign(cell_id=lambda x:x.cell_id.astype(str)).set_index("cell_id")
    observed=m05_spl.set_index("cell_id").loc[strict_ids,"p_gdT"]
    m05_spl_max_abs_difference=float(np.max(np.abs(observed.to_numpy()-reference.loc[strict_ids,"p_gdT"].to_numpy())))
    assert m05_spl_max_abs_difference<1e-7,m05_spl_max_abs_difference
    external_predictions=pd.concat([tvae_spl,m05_spl],ignore_index=True)
    external_predictions.to_parquet(PREDICTION_DIR/"SPL_frozen_endpoint_predictions_TVAE0_M05.parquet",index=False)
    display(pd.crosstab(spl_eval.donor,spl_eval.receptor_evidence_group,margins=True))
    del final_model,X_spl; clear_accelerator()
    '''),
    code(r'''
    # 11. Four endpoint definitions, pooled/donor metrics, and source-frozen thresholds
    ENDPOINTS={
      "strict_gold":{"positive_groups":["gold_B_paired_gdT"],"label":"Paired γδT","evidence_tier":"paired_receptor_gold"},
      "orphan_TRD_partial":{"positive_groups":["candidate_orphan_TRD_nonoverlap"],"label":"Orphan TRD (no αβ)","evidence_tier":"partial_receptor_sensitivity_not_gold"},
      "orphan_TRG_partial":{"positive_groups":["query_orphan_TRG"],"label":"Orphan TRG (no αβ)","evidence_tier":"partial_receptor_sensitivity_not_gold"},
      "expanded_partial_evidence":{"positive_groups":["gold_B_paired_gdT","candidate_orphan_TRD_nonoverlap","query_orphan_TRG"],"label":"Paired γδT + orphan TRD/TRG","evidence_tier":"partial_receptor_sensitivity_not_gold"},
    }
    def endpoint_frame(predictions,endpoint):
        positive=ENDPOINTS[endpoint]["positive_groups"]
        frame=predictions[predictions.immutable_cohort.isin(["gold_B_paired_abT",*positive])].copy()
        frame["y_endpoint"]=frame.immutable_cohort.isin(positive).astype(int); return frame
    def metric_row(frame):
        y=frame.y_endpoint.to_numpy(int); p=frame.p_gdT.to_numpy(float)
        return {"n":len(frame),"n_positive":int(y.sum()),"prevalence":float(y.mean()),
                "AUPRC":average_precision_score(y,p),"AUROC":roc_auc_score(y,p),
                "Brier":brier_score_loss(y,p)}
    internal_predictions=pd.concat([tvae_song_oof,m05_song_oof],ignore_index=True)
    datasets={"Song donor-OOF":internal_predictions,"SPL frozen external":external_predictions}
    pooled_rows=[]; donor_rows=[]
    for dataset,predictions in datasets.items():
        for model,pred in predictions.groupby("model",sort=False):
            for endpoint in ENDPOINTS:
                frame=endpoint_frame(pred,endpoint)
                pooled_rows.append({"dataset":dataset,"model":model,"endpoint":endpoint,
                                    "evidence_tier":ENDPOINTS[endpoint]["evidence_tier"],**metric_row(frame)})
                for donor,donor_frame in frame.groupby("donor"):
                    if donor_frame.y_endpoint.nunique()<2: continue
                    donor_rows.append({"dataset":dataset,"model":model,"endpoint":endpoint,
                                       "donor":donor,**metric_row(donor_frame)})
    pooled_metrics=pd.DataFrame(pooled_rows); donor_metrics=pd.DataFrame(donor_rows)
    donor_macro=(donor_metrics.groupby(["dataset","model","endpoint"],as_index=False)
                 .agg(donor_macro_AUPRC=("AUPRC","mean"),min_donor_AUPRC=("AUPRC","min"),
                      donor_macro_AUROC=("AUROC","mean"),n_donors=("donor","nunique")))
    m05_registry=pd.read_csv(INPUTS["m05_registry"])
    m05_threshold=float(m05_registry.loc[m05_registry.model_id.astype(str).eq("M05"),"threshold"].item())
    tvae_threshold=float(threshold_max_f1(
        endpoint_frame(tvae_song_oof,"strict_gold").y_endpoint,
        endpoint_frame(tvae_song_oof,"strict_gold").p_gdT))
    frozen_thresholds=pd.DataFrame({"model":[MODEL_NAME,"M05"],"threshold":[tvae_threshold,m05_threshold],
                                    "selection_dataset":["Song donor-OOF strict_gold"]*2})
    pooled_metrics.to_csv(TABLE_DIR/"TVAE0_M05_pooled_endpoint_metrics.csv",index=False)
    donor_metrics.to_csv(TABLE_DIR/"TVAE0_M05_donor_endpoint_metrics.csv",index=False)
    donor_macro.to_csv(TABLE_DIR/"TVAE0_M05_donor_macro_endpoint_metrics.csv",index=False)
    frozen_thresholds.to_csv(TABLE_DIR/"TVAE0_M05_source_frozen_thresholds.csv",index=False)
    display(pooled_metrics.merge(donor_macro,on=["dataset","model","endpoint"],how="left"))
    '''),
    code(r'''
    # 12. PR/ROC source data and source-frozen operating-point metrics
    CURVE_ORDER=["strict_gold","orphan_TRD_partial","orphan_TRG_partial","expanded_partial_evidence"]
    curve_rows=[]; operating_rows=[]
    threshold_lookup=frozen_thresholds.set_index("model").threshold.to_dict()
    for dataset,predictions in datasets.items():
        for model,pred in predictions.groupby("model",sort=False):
            threshold=threshold_lookup[model]
            for endpoint in CURVE_ORDER:
                frame=endpoint_frame(pred,endpoint); y=frame.y_endpoint.to_numpy(int); p=frame.p_gdT.to_numpy(float)
                precision,recall,pr_t=precision_recall_curve(y,p); fpr,tpr,roc_t=roc_curve(y,p)
                for i,(x,yv) in enumerate(zip(recall,precision)):
                    curve_rows.append({"dataset":dataset,"model":model,"endpoint":endpoint,"curve":"PR",
                                       "point_order":i,"x":float(x),"y":float(yv),
                                       "threshold":float(pr_t[i]) if i<len(pr_t) else np.nan})
                for i,(x,yv,t) in enumerate(zip(fpr,tpr,roc_t)):
                    curve_rows.append({"dataset":dataset,"model":model,"endpoint":endpoint,"curve":"ROC",
                                       "point_order":i,"x":float(x),"y":float(yv),"threshold":float(t)})
                call=p>=threshold; tp=int(np.sum(call&(y==1))); fp=int(np.sum(call&(y==0)))
                fn=int(np.sum((~call)&(y==1))); tn=int(np.sum((~call)&(y==0)))
                operating_rows.append({"dataset":dataset,"model":model,"endpoint":endpoint,"threshold":threshold,
                    "precision":tp/max(tp+fp,1),"recall":tp/max(tp+fn,1),"FPR":fp/max(fp+tn,1),
                    "TPR":tp/max(tp+fn,1),"TP":tp,"FP":fp,"FN":fn,"TN":tn})
    curve_points=pd.DataFrame(curve_rows); operating_metrics=pd.DataFrame(operating_rows)
    curve_points.to_csv(TABLE_DIR/"TVAE0_M05_PR_ROC_curve_points.csv",index=False)
    operating_metrics.to_csv(TABLE_DIR/"TVAE0_M05_source_frozen_operating_metrics.csv",index=False)
    display(operating_metrics)
    '''),
    code(r'''
    # 13. Publication-grade 2 x 4 endpoint figures for Song and SPL
    MODEL_ORDER=[MODEL_NAME,"M05"]; MODEL_COLORS={MODEL_NAME:"#2FD0F8","M05":"#D83416"}
    def save_figure(fig,stem):
        fig.savefig(FIGURE_DIR/f"{stem}.pdf",bbox_inches="tight")
        fig.savefig(FIGURE_DIR/f"{stem}.png",dpi=450,bbox_inches="tight")
    for dataset,predictions in datasets.items():
        fig,axes=plt.subplots(2,4,figsize=(15.8,7.2),sharex=True,sharey="row")
        for column,endpoint in enumerate(CURVE_ORDER):
            summary=pooled_metrics[(pooled_metrics.dataset.eq(dataset))&pooled_metrics.endpoint.eq(endpoint)].set_index("model")
            for model in MODEL_ORDER:
                points=curve_points[(curve_points.dataset.eq(dataset))&curve_points.model.eq(model)&curve_points.endpoint.eq(endpoint)]
                pr=points[points.curve.eq("PR")].sort_values("point_order"); roc=points[points.curve.eq("ROC")].sort_values("point_order")
                axes[0,column].plot(pr.x,pr.y,color=MODEL_COLORS[model],lw=1.8,label=f"{model}  AP={summary.loc[model,'AUPRC']:.3f}")
                axes[1,column].plot(roc.x,roc.y,color=MODEL_COLORS[model],lw=1.8,label=f"{model}  AUC={summary.loc[model,'AUROC']:.3f}")
                point=operating_metrics[(operating_metrics.dataset.eq(dataset))&operating_metrics.model.eq(model)&operating_metrics.endpoint.eq(endpoint)].iloc[0]
                axes[0,column].scatter(point.recall,point.precision,s=25,facecolors="white",edgecolors=MODEL_COLORS[model],linewidths=1,zorder=5)
                axes[1,column].scatter(point.FPR,point.TPR,s=25,facecolors="white",edgecolors=MODEL_COLORS[model],linewidths=1,zorder=5)
            prevalence=float(summary.prevalence.iloc[0])
            axes[0,column].axhline(prevalence,color="black",ls=":",lw=0.9,label=f"Random={prevalence:.3f}")
            axes[1,column].plot([0,1],[0,1],color="black",ls=":",lw=0.9,label="Random=0.500")
            axes[0,column].set_title(ENDPOINTS[endpoint]["label"],fontsize=9,color="black")
            axes[0,column].set_xlabel("Recall"); axes[1,column].set_xlabel("False-positive rate")
            axes[0,column].set_ylabel("Precision" if column==0 else ""); axes[1,column].set_ylabel("True-positive rate" if column==0 else "")
            axes[0,column].legend(frameon=False,fontsize=6.2,loc="lower left"); axes[1,column].legend(frameon=False,fontsize=6.2,loc="lower right")
        for axis in axes.flat:
            axis.set_xlim(-0.02,1.02); axis.set_ylim(-0.02,1.02); axis.grid(False); axis.set_aspect("equal",adjustable="box")
        title=("Song donor-OOF" if dataset=="Song donor-OOF" else "SPL frozen external")
        fig.suptitle(f"{title}: TVAE(0) versus M05",fontsize=12,color="black",fontweight="normal")
        fig.text(0.5,-0.01,"All endpoints use paired alpha-beta T cells as negatives. Open markers are Song-OOF-frozen model thresholds.",ha="center",fontsize=8,color="black")
        fig.tight_layout(); save_figure(fig,"Song_internal_TVAE0_vs_M05_four_endpoints" if dataset=="Song donor-OOF" else "SPL_external_TVAE0_vs_M05_four_endpoints")
        plt.show(); plt.close(fig)
    '''),
    code(r'''
    # 14. Output manifest and completion guard
    completion={"analysis":ANALYSIS,"status":"complete","completed_logo_fits":len(runtimes),
                "expected_logo_fits":EXPECTED_LOGO_FITS,"completed_total_tvae_fits":len(runtimes)+1,
                "expected_total_tvae_fits":EXPECTED_TOTAL_TVAE_FITS,"seed":SEED,"feature_arm":FEATURE_ARM,
                "external_evaluation":"SPL frozen","m05_song_reproduction_max_abs_difference":m05_song_max_abs_difference,
                "m05_spl_reproduction_max_abs_difference":m05_spl_max_abs_difference,
                "total_tvae_runtime_minutes":float((runtimes.seconds.sum()+final_runtime_seconds)/60),
                "software":{"python":platform.python_version(),"torch":torch.__version__,"numpy":np.__version__,"pandas":pd.__version__}}
    assert completion["completed_logo_fits"]==completion["expected_logo_fits"]
    assert completion["completed_total_tvae_fits"]==completion["expected_total_tvae_fits"]
    completion_path=TABLE_DIR/"completion_manifest.json"
    completion_path.write_text(json.dumps(completion,indent=2),encoding="utf-8")
    output_paths=[
      TABLE_DIR/"registered_run_contract.json",TABLE_DIR/"input_artifact_audit.csv",
      TABLE_DIR/"M05_panel_vs_current_H1_TCR_audit.csv",
      TABLE_DIR/"TVAE0_Song_LOGO_training_histories.csv",TABLE_DIR/"TVAE0_optimizer_parameter_coverage.csv",
      TABLE_DIR/"TVAE0_unit_runtime.csv",TABLE_DIR/"source_freeze.json",
      TABLE_DIR/"TVAE0_M05_pooled_endpoint_metrics.csv",TABLE_DIR/"TVAE0_M05_donor_endpoint_metrics.csv",
      TABLE_DIR/"TVAE0_M05_donor_macro_endpoint_metrics.csv",TABLE_DIR/"TVAE0_M05_source_frozen_thresholds.csv",
      TABLE_DIR/"TVAE0_M05_PR_ROC_curve_points.csv",TABLE_DIR/"TVAE0_M05_source_frozen_operating_metrics.csv",
      PREDICTION_DIR/"TVAE0_Song_LOGO_endpoint_predictions.parquet",
      PREDICTION_DIR/"M05_Song_LOGO_endpoint_predictions.parquet",
      PREDICTION_DIR/"SPL_frozen_endpoint_predictions_TVAE0_M05.parquet",
      FIGURE_DIR/"Song_internal_TVAE0_vs_M05_four_endpoints.pdf",
      FIGURE_DIR/"Song_internal_TVAE0_vs_M05_four_endpoints.png",
      FIGURE_DIR/"SPL_external_TVAE0_vs_M05_four_endpoints.pdf",
      FIGURE_DIR/"SPL_external_TVAE0_vs_M05_four_endpoints.png",
      final_path,completion_path,*sorted(CHECKPOINT_DIR.glob("TVAE(0)__heldout-*.joblib")),
    ]
    missing_outputs=[str(path) for path in output_paths if not path.is_file()]
    assert not missing_outputs,missing_outputs
    manifest=pd.DataFrame([{"path":str(path),"bytes":path.stat().st_size,"sha256":sha256_file(path)}
                           for path in output_paths])
    manifest.to_csv(TABLE_DIR/"output_manifest.csv",index=False)
    display(manifest)
    '''),
    md(r'''
    ## Takeaways after execution

    Read Song and SPL rows separately. The decisive external comparison is
    TVAE(0) versus M05 on the same SPL endpoint cells. Paired gamma-delta is the
    confirmatory receptor-gold result; orphan TRD, orphan TRG and the combined
    endpoint remain explicitly labelled partial-receptor sensitivity analyses.
    '''),
]


notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, OUTPUT)
print(OUTPUT)
