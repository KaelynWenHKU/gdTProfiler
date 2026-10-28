#!/usr/bin/env python3
"""Build the authoritative remote-runnable cNMF validation notebook.

The user-approved spelling ``valdiation`` is preserved deliberately. Importing this
module is safe: it defines helpers and the notebook builder but reads no project data
and launches no cNMF computation.
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

try:
    import nbformat as nbf
except ImportError:
    nbf = None  # The RNA-only export command needs base R/Matrix, not Jupyter.


ANALYSIS_NAME = "19b_cNMF_valdiation"
NOTEBOOK_NAME = f"{ANALYSIS_NAME}.ipynb"


def normalize_gse294268_cell_id(value: str) -> str:
    """Restore the 10x ``-1`` suffix changed to ``.1`` by R name repair."""
    return re.sub(r"\.([0-9]+_[0-9]+)$", r"-\1", str(value))


def md(source: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(source).strip())


def code(source: str):
    return nbf.v4.new_code_cell(textwrap.dedent(source).strip())


VALIDATION_HELPERS = r'''
# Additional validation helpers; no historical checkpoint or hypothesis is overwritten.
# These functions do not run the new benchmark automatically.
def _align_training_features(features, training_var, universe):
    """Use the training Ensembl annotation first; audit symbol disagreements.

    Each source row has one destination at most. Multiple source rows for the same
    training symbol are summed, as in the original training preprocessing.
    Ensembl version suffixes alone are stripped; no fuzzy symbol matching is used.
    """
    import re
    import numpy as np
    import pandas as pd
    from scipy import sparse

    def stable_id(value):
        return re.sub(r"^(ENSG[0-9]+)\.[0-9]+$", r"\1", str(value))

    if not pd.Index(universe).is_unique:
        raise ValueError("Training universe contains duplicate features")
    if "feature_name" not in training_var:
        raise ValueError("Training annotation lacks feature_name")
    lookup = {}
    for identifier, symbol in zip(training_var.index, training_var.feature_name):
        identifier, symbol = stable_id(identifier), str(symbol)
        if identifier in lookup and lookup[identifier] != symbol:
            raise ValueError(f"Ambiguous training Ensembl annotation: {identifier}")
        lookup[identifier] = symbol
    positions = {str(gene): i for i, gene in enumerate(universe)}
    records, row_ids, col_ids = [], [], []
    for row, feature in enumerate(features.itertuples(index=False)):
        identifier = stable_id(feature.feature_id)
        symbol = str(feature.feature_name)
        # Known IDs outside the frozen universe must NOT be rescued by an
        # inconsistent reused symbol belonging to a different gene.
        if identifier in lookup:
            target = lookup[identifier]
            route = "training Ensembl annotation"
        elif identifier in positions:
            target, route = identifier, "frozen Ensembl feature"
        else:
            target, route = symbol, "exact symbol fallback"
        included = target in positions
        if included:
            row_ids.append(row)
            col_ids.append(positions[target])
        records.append({
            "feature_id": str(feature.feature_id), "feature_name": symbol,
            "training_feature": target, "mapping_basis": route,
            "in_training_universe": included,
            "symbol_ID_disagreement": identifier in lookup and symbol != target,
        })
    mapping = sparse.csr_matrix((np.ones(len(row_ids)), (row_ids, col_ids)),
                                shape=(len(features), len(universe)))
    if np.asarray(mapping.sum(axis=1)).max(initial=0) > 1:
        raise AssertionError("One input feature was counted more than once")
    return mapping, pd.DataFrame(records)


def _thin_raw_counts(counts, fraction, seed):
    """Binomial UMI thinning, not a reduction of the number of cells."""
    import numpy as np
    if not 0 < fraction <= 1:
        raise ValueError("Retained UMI fraction must be in (0, 1]")
    result = counts.tocsr(copy=True)
    if (not np.isfinite(result.data).all() or (result.data < 0).any()
            or not np.equal(result.data, np.floor(result.data)).all()):
        raise ValueError("Depth sensitivity requires nonnegative integer raw UMI counts")
    result.data = np.random.default_rng(seed).binomial(
        result.data.astype(np.int64), fraction)
    result.eliminate_zeros()
    return result


def _project_validation_counts(counts, scorer, seed, threads=2, observed_hvgs=None):
    """Return raw and compositional usage plus numerical-quality diagnostics.

    observed_hvgs is an experimental observed-feature fit, not the locked scorer.
    Its reconstruction errors must not be compared with the frozen q95 threshold.
    All-zero cells/objectives are missing scores, never apparently perfect fits.
    """
    import warnings
    import numpy as np
    import pandas as pd
    from scipy import sparse
    from sklearn.decomposition import non_negative_factorization
    from sklearn.exceptions import ConvergenceWarning
    from threadpoolctl import threadpool_limits

    counts = counts.tocsr().astype(np.float64)
    if counts.shape[1] != len(scorer["universe"]):
        raise ValueError("Counts are not aligned to the frozen universe")
    if not np.isfinite(counts.data).all() or (counts.data < 0).any():
        raise ValueError("Counts must be finite and nonnegative")
    library = np.asarray(counts.sum(axis=1)).ravel()
    tpm = sparse.diags(np.divide(1e6, library, out=np.zeros_like(library),
                                where=library > 0)) @ counts
    objective = tpm[:, scorer["hvg_positions"]].multiply(
        1 / np.asarray(scorer["sd"])).tocsr()
    basis = np.asarray(scorer["basis"], dtype=np.float64)
    if observed_hvgs is not None:
        mask = np.asarray(observed_hvgs, dtype=bool)
        if mask.shape != (basis.shape[1],) or not mask.any():
            raise ValueError("Observed-feature mask has wrong shape or is empty")
        objective, basis = objective[:, mask], basis[:, mask]
    norm_sq = np.asarray(objective.multiply(objective).sum(axis=1)).ravel()
    valid = (library > 0) & (norm_sq > 0)
    raw = np.full((counts.shape[0], basis.shape[0]), np.nan)
    relative = np.full(counts.shape[0], np.nan)
    n_iter, converged = 0, False
    if valid.any():
        params = dict(scorer["solver_params"])
        params.update(H=basis, update_H=False, n_components=basis.shape[0],
                      random_state=int(seed))
        with warnings.catch_warnings(record=True) as recorded, threadpool_limits(limits=threads):
            warnings.simplefilter("always", ConvergenceWarning)
            fitted, _, n_iter = non_negative_factorization(objective[valid], **params)
        converged = not any(issubclass(w.category, ConvergenceWarning) for w in recorded)
        for warning in recorded:
            if not issubclass(warning.category, ConvergenceWarning):
                warnings.warn(str(warning.message), warning.category)
        raw[valid] = fitted
        cross = np.sum(fitted * (objective[valid] @ basis.T), axis=1)
        fitted_sq = np.sum((fitted @ (basis @ basis.T)) * fitted, axis=1)
        residual_sq = np.maximum(norm_sq[valid] + fitted_sq - 2 * cross, 0)
        relative[valid] = np.sqrt(residual_sq / norm_sq[valid])
    totals = raw.sum(axis=1)
    usage = np.divide(raw, totals[:, None], out=np.full_like(raw, np.nan),
                      where=(totals > 0)[:, None])
    quality = pd.DataFrame({"library_over_training_universe": library,
                            "scaled_HVG_relative_error": relative,
                            "valid_objective": valid,
                            "solver_converged": valid & converged,
                            "solver_n_iter": n_iter})
    return raw, usage, quality


def _aucell_validation_scores(counts, genes, signatures):
    """AUCell via decoupler 2.1.2, separately labelled from the rank baseline.

    Fixed gene universe/order, signatures and n_up prevent test-cohort-dependent
    feature selection. Log normalization preserves per-cell gene ranking.
    """
    import importlib.metadata as im
    import numpy as np
    import pandas as pd
    from scipy import sparse
    import decoupler as dc
    import anndata as ad
    if im.version("decoupler") != "2.1.2":
        raise RuntimeError("Validation requires the audited decoupler 2.1.2 implementation")
    genes = pd.Index(genes).astype(str)
    if not genes.is_unique or counts.shape[1] != len(genes):
        raise ValueError("AUCell requires one aligned column per unique gene")
    net = pd.DataFrame([(str(key), gene) for key, values in signatures.items()
                        for gene in dict.fromkeys(values) if gene in genes],
                       columns=["source", "target"])
    if set(net.source) != set(map(str, signatures)):
        raise ValueError("At least one signature has no observed genes")
    library = np.asarray(counts.sum(axis=1)).ravel().astype(float)
    valid = library > 0
    scores = pd.DataFrame(np.nan, index=np.arange(len(library)),
                          columns=list(map(str, signatures)))
    if valid.any():
        normalized = (sparse.diags(1e4 / library[valid]) @ counts[valid]).tocsr()
        normalized.data = np.log1p(normalized.data)
        data = ad.AnnData(normalized, obs=pd.DataFrame(index=np.flatnonzero(valid).astype(str)),
                          var=pd.DataFrame(index=genes))
        if len(genes) < 2:
            raise ValueError("AUCell requires at least two features")
        dc.mt.aucell(data, net, tmin=1, empty=False, bsize=1000,
                     n_up=min(len(genes), max(2, int(np.ceil(0.05 * len(genes))))))
        result = data.obsm["score_aucell"]
        scores.loc[valid, result.columns] = result.to_numpy()
    return scores


def _strict_multimodal_eligibility(tcr, adt):
    """Eligibility is explicit; missing ADT is not a measured negative endpoint."""
    import pandas as pd
    if not tcr.index.is_unique or not adt.index.is_unique:
        raise ValueError("Duplicate cell IDs in multimodal metadata")
    return pd.DataFrame({
        "single_productive_TRG_TRD": tcr.n_productive_TRG.eq(1)
            & tcr.n_productive_TRD.eq(1) & tcr.TRGC_assignment.isin(["TRGC1", "TRGC2"]),
        "has_ADT_counts": adt.ADT_total_counts.reindex(tcr.index).gt(0),
    }, index=tcr.index)
'''


SERUM_RNA_EXPORT = r'''
args <- commandArgs(trailingOnly=TRUE)
library(Matrix)
obj <- readRDS(args[1])
meta <- slot(obj, "meta.data")
counts <- slot(slot(obj, "assays")[["RNA"]], "counts")
stopifnot(inherits(counts, "sparseMatrix"), identical(colnames(counts), rownames(meta)),
          !anyDuplicated(rownames(counts)), !anyDuplicated(colnames(counts)),
          all(is.finite(counts@x)), all(counts@x >= 0), all(counts@x == round(counts@x)))
keep <- c("donor", "condition", "day_of_seq", "serum", "nCount_RNA", "nFeature_RNA",
          "percent.mt", "RNA_snn_res.0.8")
stopifnot(all(keep %in% colnames(meta)))
writeMM(counts, file.path(args[2], "RNA_counts.mtx"))
write.table(rownames(counts), file.path(args[2], "RNA_genes.tsv"), sep="\t",
            quote=FALSE, row.names=FALSE, col.names=FALSE)
write.csv(data.frame(cell_id=rownames(meta), meta[, keep, drop=FALSE], check.names=FALSE),
          file.path(args[2], "RNA_metadata.csv"), row.names=FALSE)
writeLines(c(R.version.string, paste("Matrix", packageVersion("Matrix"))),
           file.path(args[2], "export_versions.txt"))
'''


FOLLOWUP_HELPERS = r'''
FOLLOWUP_PROGRAMS = ["P1", "P2", "P3", "P4", "P5", "P7", "P9", "P10"]

def _followup_read_unit(directory, signature=None, required=None):
    directory = Path(directory)
    payload = json.loads((directory / "scoring.complete.json").read_text())
    if signature is not None and payload["signature"] != signature:
        raise ValueError("Checkpoint signature mismatch")
    if required is not None and set(payload["outputs"]) != set(required):
        raise ValueError("Checkpoint output schema mismatch")
    if not payload.get("outputs"):
        raise ValueError("Empty checkpoint")
    for name, digest in payload["outputs"].items():
        if sha256sum(directory / name) != digest:
            raise ValueError(f"Checkpoint hash mismatch: {directory / name}")
    return payload

def _followup_start(stage, prefix, inputs, parameters):
    import shutil
    config = {"stage": stage, "source_sha256": sha256sum(PROJECT / "build_19b_cNMF_valdiation.py"),
              "inputs": inputs, "parameters": parameters,
              "versions": {p: im.version(p) for p in ("numpy", "pandas", "scipy", "scikit-learn")}}
    signature = stable_token(config)
    root = CHECKPOINT_DIR / stage / signature
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / "job_spec.json", config)
    snapshot = root / "build_19b_cNMF_valdiation.py"
    if not snapshot.exists(): shutil.copy2(PROJECT / snapshot.name, snapshot)
    if sha256sum(snapshot) != config["source_sha256"]: raise ValueError("Source snapshot differs")
    progress = {"status": "running", "run_id": signature, "configuration": config,
                "checkpoint_root": str(root), "completed_units": []}
    path = TABLE_DIR / f"{prefix}_manifest.json"
    def flush():
        progress["updated_UTC"] = datetime.now(timezone.utc).isoformat()
        atomic_json(path, progress)
    flush()
    def validate_frames(frames, spec):
        if stage == "serum_time_course":
            if set(frames) != {"usage_normalized", "usage_raw", "quality"}: raise ValueError("Projection schema mismatch")
            scores = frames["usage_normalized"]
            if (not scores.index.is_unique or len(scores) != spec["n_cells"]
                or sha256_lines(scores.index) != spec["cell_ids_sha256"]
                or set(scores.columns) != set(FOLLOWUP_PROGRAMS)
                or not frames["quality"].index.equals(scores.index)
                or not frames["usage_raw"].index.equals(scores.index)):
                raise ValueError("Projection checkpoint identity/schema mismatch")
        else:
            if set(frames) != {"donor_metrics", "stratified_metrics"}: raise ValueError("P7 schema mismatch")
            if (set(frames["donor_metrics"].donor) != {spec["donor"]}
                or len(frames["donor_metrics"]) != 3
                or set(frames["donor_metrics"].fraction) != {.1, .25, .5}):
                raise ValueError("P7 checkpoint donor/depth mismatch")
    def unit(label, spec, produce):
        unit_signature = stable_token({"run_id": signature, "specification": spec})
        folder = root / label / unit_signature
        folder.mkdir(parents=True, exist_ok=True)
        atomic_json(folder / "job_spec.json", spec)
        marker = folder / "scoring.complete.json"
        if marker.exists():
            payload = _followup_read_unit(folder, unit_signature)
            frames = {Path(name).stem: pd.read_parquet(folder / name) for name in payload["outputs"]}
        else:
            try:
                frames = produce()
                if not frames or any(not isinstance(v, pd.DataFrame) for v in frames.values()):
                    raise ValueError("Unit must return named data frames")
                validate_frames(frames, spec)
                paths = [atomic_parquet(v, folder / f"{name}.parquet") for name, v in frames.items()]
                write_marker(marker, unit_signature, paths)
            except Exception as error:
                progress.update(status="failed", failed_unit=label, error=f"{type(error).__name__}: {error}")
                flush(); raise
        validate_frames(frames, spec)
        progress["completed_units"].append({"label": label, "path": str(folder), "signature": unit_signature})
        flush()
        print(stage, len(progress["completed_units"]), label, flush=True)
        return frames
    return root, progress, flush, unit

def _followup_finish(progress, flush, prefix, expected, limitations):
    if len(progress["completed_units"]) != expected: raise ValueError("Incomplete stage")
    progress.update(status="completed", expected_units=expected, limitations=limitations)
    progress["outputs"] = {str(p.relative_to(PROJECT)): sha256sum(p)
        for parent, extension in ((TABLE_DIR, "csv"), (FIGURE_DIR, "pdf"))
        for p in sorted(parent.glob(f"{prefix}_*.{extension}"))}
    flush()

def _followup_heatmap(frame, name, title, label, diverging=False):
    # One data axis plus a colorbar; vector cells, no rasterization.
    values = frame.to_numpy(float)
    fig, ax = plt.subplots(figsize=(7.2, max(3.5, .22 * len(frame) + 1.5)))
    limit = max(float(np.nanmax(np.abs(values))), 1e-6) if np.isfinite(values).any() else 1
    mesh = ax.pcolormesh(np.ma.masked_invalid(values), cmap="RdBu_r" if diverging else "YlGnBu",
                        vmin=-limit if diverging else 0, vmax=limit, edgecolors="none", rasterized=False)
    ax.set(xticks=np.arange(len(frame.columns)) + .5, xticklabels=frame.columns,
           yticks=np.arange(len(frame)) + .5, yticklabels=frame.index, title=title)
    ax.invert_yaxis()
    colorbar = fig.colorbar(mesh, ax=ax, label=label)
    # Matplotlib rasterizes long colorbars by default; keep the entire PDF vector.
    if colorbar.solids is not None:
        colorbar.solids.set_rasterized(False)
    save_figure(fig, name)

def _serum_paired(frame, value="median_usage"):
    keys = ["donor", "day", "program", "scenario", "replicate", "gate"]
    late = frame.loc[frame.day.ne(0)].copy()
    if late.duplicated(keys + ["serum"]).any(): raise ValueError("Duplicate donor-condition aggregate")
    rows = []
    for key, part in late.groupby(keys, dropna=False, observed=True):
        row = dict(zip(keys, key)); indexed = part.set_index("serum")
        complete = {"+", "-"} <= set(indexed.index)
        row["paired"] = complete
        row["WITH_SERUM_minus_WO_SERUM"] = float(indexed.loc["+", value] - indexed.loc["-", value]) if complete else np.nan
        rows.append(row)
    return pd.DataFrame(rows)

def _serum_decompose(part, programs, minimum=10):
    # Symmetric decomposition on shared RNA strata, never a causal attribution.
    key = "RNA_snn_res.0.8"
    groups = part.groupby(["serum", key], observed=True)
    counts = groups.size().unstack("serum", fill_value=0)
    if not {"+", "-"} <= set(counts): return pd.DataFrame()
    shared = counts.index[(counts["+"] >= minimum) & (counts["-"] >= minimum)]
    if not len(shared): return pd.DataFrame()
    means = groups[programs].mean()
    n = counts.loc[shared]
    wp, wm = n["+"] / n["+"].sum(), n["-"] / n["-"].sum()
    up, um = means.loc["+"].loc[shared], means.loc["-"].loc[shared]
    within = (up - um).mul((wp + wm) / 2, axis=0).sum()
    composition = ((up + um) / 2).mul(wp - wm, axis=0).sum()
    total = up.mul(wp, axis=0).sum() - um.mul(wm, axis=0).sum()
    if not np.allclose(within + composition, total, atol=1e-12): raise AssertionError("Decomposition does not add up")
    return pd.DataFrame({"program": programs, "within_RNA_stratum": within.reindex(programs).to_numpy(),
        "composition": composition.reindex(programs).to_numpy(), "restricted_mean_difference": total.reindex(programs).to_numpy(),
        "n_shared_strata": len(shared), "WITH_SERUM_retained_fraction": n["+"].sum() / (part.serum == "+").sum(),
        "WO_SERUM_retained_fraction": n["-"].sum() / (part.serum == "-").sum()})
'''


P7_DIAGNOSTICS_CODE = r'''
def run_p7_diagnostics():
    prefix = "Gray_primary_K8_P7_diagnostics"
    depth_path = TABLE_DIR / "Gray_primary_K8_depth_calibration_manifest.json"
    prior_path = TABLE_DIR / "Gray_primary_K8_robustness_validation_manifest.json"
    depth, prior = json.loads(depth_path.read_text()), json.loads(prior_path.read_text())
    if depth["status"] != "completed" or prior["status"] != "completed": raise ValueError("Prior stages incomplete")
    donor_units = {u["donor"]: Path(u["path"]) for u in depth["completed_units"] if u["target"] == "P7"}
    if len(donor_units) != 22: raise ValueError("Need all 22 donor P7 checkpoints")
    prior_root = Path(prior["checkpoint_root"])
    metadata = pd.read_csv(REQUIRED["Gray cohort"], index_col=0)
    frames, baselines, specs, inputs = {}, {}, {}, {str(depth_path): sha256sum(depth_path), str(prior_path): sha256sum(prior_path),
                                                   str(REQUIRED["Gray cohort"]): sha256sum(REQUIRED["Gray cohort"])}
    for donor, directory in donor_units.items():
        specs[donor] = json.loads((directory / "job_spec.json").read_text())
        _followup_read_unit(directory, stable_token(specs[donor]))
        inputs[str(directory / "scoring.complete.json")] = sha256sum(directory / "scoring.complete.json")
        pred = pd.read_parquet(directory / "predictions.parquet")
        baseline_dir = one_file(prior_root / "Gray" / donor / "full_depth", "*/scoring.complete.json").parent
        bspec = json.loads((baseline_dir / "job_spec.json").read_text())
        _followup_read_unit(baseline_dir, stable_token({"run_id": prior["run_id"], "specification": bspec}))
        inputs[str(baseline_dir / "scoring.complete.json")] = sha256sum(baseline_dir / "scoring.complete.json")
        baseline = pd.read_parquet(baseline_dir / "usage_normalized.parquet")
        baselines[donor] = baseline.P7
        if not pred.cell_id.isin(metadata.index).all(): raise ValueError("Unmatched Gray cells")
        pred = pred.join(metadata[["tissue"]], on="cell_id").join(baseline.P7.rename("baseline_P7"), on="cell_id")
        if pred.baseline_P7.isna().any(): raise ValueError("Missing full-depth P7 reference")
        pred["exceeds"] = pred.observed_error > pred.depth_q95
        pred["baseline_usage_stratum"] = pd.cut(pred.baseline_P7, [-1e-9, .05, .2, .5, 1.000001],
            labels=["0–0.05", "0.05–0.20", "0.20–0.50", "0.50–1.00"]).astype(str)
        frames[donor] = pred
    perturbation_path = TABLE_DIR / "Gray_primary_K8_robustness_validation_donor_metrics.csv"
    inputs[str(perturbation_path)] = sha256sum(perturbation_path)
    scorer = load_frozen_scorer_for_gse268991()
    for key in ("spectra_path", "sd_path", "spec_path", "ranked_path", "params_path"):
        inputs[str(scorer[key])] = sha256sum(scorer[key])
    root, progress, flush, unit = _followup_start("p7_diagnostics", prefix, inputs,
        {"fractions": [.1, .25, .5], "reference_strata": [0, .05, .2, .5, 1], "refit": False})
    donor_rows, strata_rows = [], []
    for donor in sorted(frames):
        calibration = specs[donor]["calibration_donors"]
        _depth_assert_split(specs[donor]["basis_train_donors"], calibration, donor)
        def produce():
            pred = frames[donor]; out, strat = [], []
            cal_median = float(np.mean([baselines[d].median() for d in calibration]))
            for fraction, part in pred.groupby("fraction"):
                valid = part.valid_objective & part.in_depth_support & np.isfinite(part.observed_error)
                selected = part.loc[valid]
                cal = pd.concat([frames[d].loc[frames[d].fraction.eq(fraction)].assign(cal_donor=d) for d in calibration])
                cal = cal.loc[cal.valid_objective & np.isfinite(cal.observed_error)]
                cal_mae = cal.groupby("cal_donor").observed_error.mean().mean()
                cal_tissues = set(cal.tissue)
                out.append({"donor": donor, "fold": specs[donor]["fold"], "fraction": fraction,
                    "n_cells": part.cell_id.nunique(), "n_observations": len(part), "n_valid_supported": len(selected),
                    "exceedance_fraction": selected.exceeds.mean(), "observed_MAE": selected.observed_error.mean(),
                    "mean_predicted_bound": selected.depth_q95.mean(), "baseline_P7_median": baselines[donor].median(),
                    "calibration_donor_mean_of_medians": cal_median,
                    "baseline_usage_shift": baselines[donor].median() - cal_median,
                    "calibration_donor_mean_MAE": cal_mae,
                    "fraction_in_depth_support": part.in_depth_support.mean(),
                    "fraction_invalid": (~part.valid_objective | ~np.isfinite(part.observed_error)).mean(),
                    "fraction_unseen_tissue": (~selected.tissue.isin(cal_tissues)).mean(),
                    "n_calibration_donors": len(calibration)})
                for grouping in ("tissue", "baseline_usage_stratum"):
                    for value, group in selected.groupby(grouping, observed=True):
                        cal_group = cal.loc[cal[grouping].eq(value)]
                        strat.append({"donor": donor, "fraction": fraction, "stratification": grouping, "stratum": value,
                            "n_cells": group.cell_id.nunique(), "n_observations": len(group),
                            "exceedance_fraction": group.exceeds.mean(), "observed_MAE": group.observed_error.mean(),
                            "mean_predicted_bound": group.depth_q95.mean(), "calibration_donors_in_stratum": cal_group.cal_donor.nunique(),
                            "calibration_cells_in_stratum": cal_group.cell_id.nunique(),
                            "calibration_donor_mean_MAE": cal_group.groupby("cal_donor").observed_error.mean().mean()})
            return {"donor_metrics": pd.DataFrame(out), "stratified_metrics": pd.DataFrame(strat)}
        result = unit(donor, {"donor": donor, "calibration_donors": calibration}, produce)
        if set(result) != {"donor_metrics", "stratified_metrics"}: raise ValueError("P7 schema mismatch")
        if set(result["donor_metrics"].donor) != {donor}: raise ValueError("P7 donor mismatch")
        donor_rows.append(result["donor_metrics"]); strata_rows.append(result["stratified_metrics"])
    donor_table = pd.concat(donor_rows, ignore_index=True)
    save_table(donor_table, f"{prefix}_donor_metrics")
    save_table(pd.concat(strata_rows, ignore_index=True), f"{prefix}_stratified_metrics")
    old_metrics = pd.read_csv(TABLE_DIR / "Gray_primary_K8_robustness_validation_donor_metrics.csv")
    save_table(old_metrics.loc[old_metrics.program.eq("P7")], f"{prefix}_existing_perturbation_metrics")
    component = next(c for c, p in scorer["component_to_program"].items() if p == "P7")
    weights = scorer["basis"][list(scorer["components"]).index(component)]
    order = np.argsort(-weights, kind="stable")
    loadings = pd.DataFrame({"gene": scorer["hvgs"][order], "scaled_basis_weight": weights[order],
        "weight_fraction": weights[order] / weights.sum(),
        "cumulative_weight_fraction": np.cumsum(weights[order]) / weights.sum()})
    loadings["rank"] = np.arange(1, len(loadings) + 1)
    save_table(loadings, f"{prefix}_frozen_spectrum_concentration")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    part = donor_table.loc[donor_table.fraction.eq(.25)]
    ax.scatter(part.baseline_usage_shift, part.exceedance_fraction, color="#4477AA", s=28)
    ax.axhline(.05, color="black", linestyle="--", linewidth=.8)
    ax.set(xlabel="Test minus calibration P7 usage reference", ylabel="Fraction exceeding technical-error bound",
           title="P7: state-distribution shift and bound exceedance\nOne point per donor; 25% retained UMI")
    save_figure(fig, f"{prefix}_usage_shift")
    view = donor_table.pivot(index="donor", columns="fraction", values="exceedance_fraction")
    view.columns = [f"{x:.0%} UMI" for x in view.columns]
    _followup_heatmap(view, f"{prefix}_exceedance", "P7: all 22 held-out donors", "Fraction exceeding bound")
    _followup_finish(progress, flush, prefix, 22,
        "Retrospective diagnosis; baseline-usage strata use full-depth references, not available prospective truths. No refit or threshold change. Associations are not causal. Existing random-mask sensitivity is not targeted-gene ablation.")
    display(donor_table.loc[donor_table.fraction.eq(.25)].sort_values("exceedance_fraction", ascending=False))

if EXECUTION_STAGE == "p7_diagnostics":
    run_p7_diagnostics()
else:
    print("P7 diagnostics not launched in stage", EXECUTION_STAGE)
'''


SERUM_TIME_COURSE_CODE = r'''
def run_serum_time_course():
    from scipy.io import mmread
    prefix = "Zenodo17590095_primary_K8_time_course"
    export = MODEL_DIR / f"{prefix}_input"
    manifest_path = export / "export_manifest.json"
    if not manifest_path.exists(): raise FileNotFoundError("Run the generator --export-serum-rna command and sync its input directory first")
    manifest = json.loads(manifest_path.read_text())
    if manifest["source_md5"] != "9a9bdf00eb10595f20d7eacfbd9ce27d": raise ValueError("Unexpected source release")
    for name, digest in manifest["outputs"].items():
        if sha256sum(export / name) != digest: raise ValueError(f"RNA export checksum mismatch: {name}")
    metadata = pd.read_csv(export / "RNA_metadata.csv", index_col="cell_id", dtype={"RNA_snn_res.0.8": str})
    genes = pd.Index((export / "RNA_genes.tsv").read_text().splitlines())
    counts = mmread(export / "RNA_counts.mtx.gz").T.tocsr()
    if counts.shape != (7363, 36485) or not genes.is_unique or not metadata.index.is_unique:
        raise ValueError("Unexpected released RNA dimensions/identities")
    if len(metadata) != counts.shape[0] or not np.equal(counts.data, np.floor(counts.data)).all() or (counts.data < 0).any():
        raise ValueError("RNA counts or metadata invalid")
    expected = {"D0": (0, "Day 0"), **{f"D{day}_{tag}": (day, symbol)
        for day in (7, 10, 14) for tag, symbol in (("WITH_SERUM", "+"), ("WO_SERUM", "-"))}}
    if set(metadata.condition) != set(expected) or set(metadata.donor) != {"donorA", "donorB", "donorC"}:
        raise ValueError("Unknown donor/condition labels")
    metadata["day"] = metadata.condition.map(lambda x: expected[x][0])
    if not (metadata.serum == metadata.condition.map(lambda x: expected[x][1])).all(): raise ValueError("Serum metadata disagree")
    if not (metadata.day_of_seq == metadata.day.map(lambda x: f"D{x}")).all(): raise ValueError("Day metadata disagree")
    if len(metadata.groupby(["donor", "condition"])) != 21: raise ValueError("Missing donor-condition samples")
    scorer = load_frozen_scorer_for_gse268991()
    frozen = {k: sha256sum(scorer[k]) for k in ("spectra_path", "sd_path", "spec_path", "ranked_path", "params_path")}
    source_positions = scorer["universe"].get_indexer(genes)
    observed = source_positions >= 0
    mapping = sparse.csr_matrix((np.ones(observed.sum()), (np.flatnonzero(observed), source_positions[observed])),
                               shape=(len(genes), len(scorer["universe"])))
    aligned = (counts @ mapping).tocsr()
    present = np.asarray(mapping.sum(axis=0)).ravel() > 0
    hvg_coverage = float(present[scorer["hvg_positions"]].mean())
    if hvg_coverage < MIN_HVG_COVERAGE: raise ValueError("HVG coverage below frozen minimum")
    coverage = []
    for i, component in enumerate(scorer["components"]):
        w = scorer["basis"][i]; detected = present[scorer["hvg_positions"]]
        signature = scorer["signatures"][component]
        coverage.append({"program": scorer["component_to_program"][component], "HVG_coverage": hvg_coverage,
            "top50_coverage": np.mean([g in genes for g in signature]), "weighted_L1_coverage": w[detected].sum()/w.sum(),
            "missing_HVGs": ";".join(scorer["hvgs"][~detected])})
    inputs = {"export_manifest": sha256sum(manifest_path), **manifest["outputs"], **frozen}
    q95 = json.loads((TABLE_DIR / "Gray_primary_K8_robustness_validation_manifest.json").read_text())["configuration"]["q95"]
    root, progress, flush, unit = _followup_start("serum_time_course", prefix, inputs,
        {"seed": 20260920, "threads": THREADS, "target_expected_UMI": 1000, "replicates": [0, 1, 2],
         "fraction": .5, "q95_historical_only": q95, "RNA_stratum_min_cells": 10})
    save_table(pd.DataFrame(coverage), f"{prefix}_coverage")
    summaries, inventories, native_parts = [], [], []
    native_columns = [scorer["component_to_program"][str(c)] for c in scorer["components"]]
    for sample_number, ((donor, condition), meta) in enumerate(metadata.groupby(["donor", "condition"], sort=True)):
        matrix = aligned[metadata.index.get_indexer(meta.index)].tocsr()
        library = np.asarray(matrix.sum(axis=1)).ravel()
        scenarios = [("native", 0)] + [(s, r) for s in ("half_depth", "matched_depth") for r in range(3)]
        native = None
        for scenario, replicate in scenarios:
            ids = np.flatnonzero(library >= 1000) if scenario == "matched_depth" else np.arange(len(meta))
            if not len(ids): raise ValueError(f"No cells support depth matching: {donor}/{condition}")
            seed = 20260920 + sample_number * 100 + replicate
            def produce():
                data = matrix[ids].copy()
                if scenario == "half_depth": data = _thin_raw_counts(data, .5, seed)
                elif scenario == "matched_depth":
                    probability = np.repeat(1000 / library[ids], np.diff(data.indptr))
                    data.data = np.random.default_rng(seed).binomial(data.data.astype(np.int64), probability)
                    data.eliminate_zeros()
                raw, usage, quality = _project_validation_counts(data, scorer, seed, THREADS)
                quality.index = meta.index[ids]
                scores = pd.DataFrame(usage, index=quality.index, columns=native_columns)
                raw_scores = pd.DataFrame(raw, index=quality.index, columns=native_columns)
                return {"usage_normalized": scores, "usage_raw": raw_scores, "quality": quality}
            output = unit(f"{donor}/{condition}/{scenario}_{replicate}",
                {"donor": donor, "condition": condition, "scenario": scenario, "replicate": replicate,
                 "seed": seed, "cell_ids_sha256": sha256_lines(meta.index[ids]), "n_cells": len(ids)}, produce)
            if set(output) != {"usage_normalized", "usage_raw", "quality"}: raise ValueError("Projection schema mismatch")
            scores, quality = output["usage_normalized"], output["quality"]
            if not scores.index.equals(meta.index[ids]) or not quality.index.equals(scores.index): raise ValueError("Projection cell order differs")
            if set(scores.columns) != set(FOLLOWUP_PROGRAMS): raise ValueError("Program schema differs")
            valid = quality.valid_objective & quality.solver_converged & np.isfinite(scores).all(axis=1)
            accepted = valid & quality.scaled_HVG_relative_error.le(q95)
            if scenario == "native":
                native = scores.copy()
                combined = scores.join(meta); combined["historical_q95_accepted"] = accepted
                native_parts.append(combined.loc[valid])
            for gate, mask in (("all_valid", valid), ("historical_q95", accepted)):
                for program in FOLLOWUP_PROGRAMS:
                    summaries.append({"donor": donor, "condition": condition, "day": int(meta.day.iloc[0]),
                        "serum": meta.serum.iloc[0], "scenario": scenario, "replicate": replicate, "gate": gate,
                        "program": program, "n_cells": int(mask.sum()), "median_usage": scores.loc[mask, program].median(),
                        "mean_usage": scores.loc[mask, program].mean(),
                        "matched_native_median": native.reindex(scores.index).loc[mask, program].median(),
                        "paired_cell_MAE": (scores.loc[mask, program] - native.reindex(scores.index).loc[mask, program]).abs().mean()})
            inventories.append({"donor": donor, "condition": condition, "scenario": scenario, "replicate": replicate,
                "n_original_cells": len(meta), "n_eligible_cells": len(ids), "n_valid_converged": int(valid.sum()),
                "n_historical_q95_accepted": int(accepted.sum()), "median_library": quality.library_over_training_universe.median(),
                "median_reconstruction_error": quality.scaled_HVG_relative_error.median(),
                "depth_matching_eligible_fraction": float((library >= 1000).mean())})
        if os.environ.get("GDT_19B_FOLLOWUP_SMOKE", "0") == "1":
            progress.update(status="smoke_completed_not_full_run", expected_units=147)
            flush()
            print("Seven real-data projection units passed; this is not the full time course.")
            return
    table = pd.DataFrame(summaries); inventory = pd.DataFrame(inventories)
    save_table(table, f"{prefix}_donor_condition_scores")
    save_table(inventory, f"{prefix}_quality")
    original = table.loc[table.scenario.eq("native")].copy()
    reference = original.loc[original.day.eq(0), ["donor", "program", "gate", "median_usage"]].rename(
        columns={"median_usage": "D0_median_usage"})
    changes = original.loc[original.day.ne(0)].merge(reference, on=["donor", "program", "gate"], validate="many_to_one")
    changes["change_from_D0"] = changes.median_usage - changes.D0_median_usage
    save_table(changes, f"{prefix}_D0_changes")
    contrasts = _serum_paired(table)
    save_table(contrasts, f"{prefix}_paired_serum_contrasts")
    matched_reference = _serum_paired(table, "matched_native_median")
    contrast_keys = ["donor", "day", "program", "scenario", "replicate", "gate"]
    sensitivity = contrasts.merge(matched_reference[contrast_keys + ["WITH_SERUM_minus_WO_SERUM"]], on=contrast_keys,
        suffixes=("", "_matched_native"), validate="one_to_one")
    sensitivity["contrast_change_after_thinning"] = sensitivity.WITH_SERUM_minus_WO_SERUM - sensitivity.WITH_SERUM_minus_WO_SERUM_matched_native
    save_table(sensitivity, f"{prefix}_depth_sensitivity")
    native_cells = pd.concat(native_parts)
    composition = []
    for (donor, day), part in native_cells.loc[native_cells.day.ne(0)].groupby(["donor", "day"]):
        out = _serum_decompose(part, FOLLOWUP_PROGRAMS)
        if out.empty:
            out = pd.DataFrame({"program": FOLLOWUP_PROGRAMS, "n_shared_strata": 0})
        out["donor"] = donor; out["day"] = day; composition.append(out)
    save_table(pd.concat(composition, ignore_index=True), f"{prefix}_RNA_stratum_decomposition")
    cluster_counts = native_cells.groupby(["donor", "condition", "RNA_snn_res.0.8"]).size().rename("n_cells").reset_index()
    save_table(cluster_counts, f"{prefix}_RNA_stratum_counts")
    # Preserve each donor and the real source conditions; D0 is never duplicated.
    base = table.loc[table.scenario.eq("native") & table.gate.eq("all_valid")].copy()
    base["sample"] = base.donor + " / " + base.condition
    view = base.pivot(index="sample", columns="program", values="median_usage").reindex(columns=FOLLOWUP_PROGRAMS)
    condition_order = ["D0"] + [f"D{d}_{s}" for d in (7, 10, 14) for s in ("WITH_SERUM", "WO_SERUM")]
    view = view.reindex([f"{d} / {c}" for d in ("donorA", "donorB", "donorC") for c in condition_order])
    _followup_heatmap(view, f"{prefix}_native_usages", "Frozen K8: donor-condition RNA program usages", "Median normalized usage")
    paired = contrasts.loc[contrasts.scenario.eq("native") & contrasts.gate.eq("all_valid")].copy()
    paired["sample"] = paired.donor + " / D" + paired.day.astype(str)
    view = paired.pivot(index="sample", columns="program", values="WITH_SERUM_minus_WO_SERUM").reindex(columns=FOLLOWUP_PROGRAMS)
    view = view.reindex([f"{d} / D{day}" for d in ("donorA", "donorB", "donorC") for day in (7, 10, 14)])
    _followup_heatmap(view, f"{prefix}_serum_contrasts", "Within-donor serum contrasts; three donors", "WITH_SERUM minus WO_SERUM", True)
    # Individual donor trajectories, no population CI inferred from three donors.
    for program in ("P2", "P5", "P7"):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for donor, color in zip(("donorA", "donorB", "donorC"), ("#4477AA", "#CC6677", "#228833")):
            sub = base.loc[base.donor.eq(donor) & base.program.eq(program)]
            d0 = sub.loc[sub.day.eq(0)]
            ax.scatter(d0.day, d0.median_usage, color=color, marker="s", s=28)
            for serum, style, name in (("+", "-", "WITH_SERUM"), ("-", "--", "WO_SERUM")):
                late = sub.loc[sub.serum.eq(serum)].sort_values("day")
                ax.plot(late.day, late.median_usage, style, marker="o", color=color, label=f"{donor} {name}")
        ax.set(xticks=[0, 7, 10, 14], xlabel="Culture day", ylabel=f"Median normalized {program} usage",
               title=f"{program}: donor-specific RNA time course\nD0 shown once; no longitudinal cell tracking")
        ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1))
        save_figure(fig, f"{prefix}_{program}_time_course")
    if frozen != {k: sha256sum(scorer[k]) for k in frozen}: raise ValueError("Frozen model changed")
    _followup_finish(progress, flush, prefix, 147,
        "Three donors; exploratory expression-only projection. Source citeseq and overwritten seurat_clusters excluded. RNA strata are not orthogonal labels. DonorA D0 separate processing; released 7363 differs from paper 8249. No Gray fold calibration transferred. No cell-level inference, killing claim or causal interpretation. Full-depth reference is not ground truth.")
    display(inventory.loc[inventory.scenario.eq("native")])
    display(paired[["donor", "day", "program", "WITH_SERUM_minus_WO_SERUM"]])

if EXECUTION_STAGE == "serum_time_course":
    run_serum_time_course()
else:
    print("Serum time course not launched in stage", EXECUTION_STAGE)
'''


DEPTH_CALIBRATION_HELPERS = r'''
def _depth_weighted_quantile(values, weights, probability):
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    if not len(values) or not np.isfinite(values).all() or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Quantiles require finite observations and positive weights")
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order]) / weights.sum()
    return float(values[order[min(np.searchsorted(cumulative, probability), len(order) - 1)]])

def _depth_assert_split(basis_train, calibration, test):
    a, b, c = set(map(str, basis_train)), set(map(str, calibration)), {str(test)}
    if a & b or a & c or b & c or len(b) < 3:
        raise ValueError("Basis, reliability-calibration and test donors must be disjoint; need >=3 calibration donors")

def _depth_weights(frame):
    # Equal donors, equal depth fractions within donor, equal observations in each.
    size = frame.groupby(["donor", "fraction"], observed=True).donor.transform("size")
    fractions = frame.groupby("donor", observed=True).fraction.transform("nunique")
    weights = 1. / (size.to_numpy(float) * fractions.to_numpy(float))
    return weights / weights.sum()

def _depth_design(log_depth, knots):
    x = np.asarray(log_depth, float)
    return np.column_stack([x, *[np.maximum(x - knot, 0) for knot in knots]])

def _depth_fit(frame, target):
    from sklearn.linear_model import QuantileRegressor
    valid = (np.isfinite(frame[target]) & np.isfinite(frame.library_over_training_universe)
             & frame.library_over_training_universe.gt(0) & frame.valid_objective)
    train = frame.loc[valid].copy()
    if train.donor.nunique() < 3 or len(train) < 10:
        raise ValueError("Insufficient valid calibration observations/donors")
    y = train[target].to_numpy(float)
    if ((y < 0) | (y > 1 + 1e-8)).any():
        raise ValueError("Unexpected reconstruction/program-change range")
    w = _depth_weights(train)
    x = np.log1p(train.library_over_training_universe.to_numpy(float))
    knots = [_depth_weighted_quantile(x, w, q) for q in (.25, .5, .75)]
    design = _depth_design(x, knots)
    center = np.average(design, axis=0, weights=w)
    scale = np.sqrt(np.average((design - center) ** 2, axis=0, weights=w))
    scale[scale < 1e-12] = 1
    fit = QuantileRegressor(quantile=.95, alpha=.001, solver="highs")
    fit.fit((design - center) / scale, y, sample_weight=w)
    return {"knots": knots, "center": center.tolist(), "scale": scale.tolist(),
            "coef": fit.coef_.tolist(), "intercept": float(fit.intercept_),
            "constant_q95": _depth_weighted_quantile(y, w, .95),
            "minimum_log_depth": float(x.min()), "maximum_log_depth": float(x.max()),
            "quantile": .95, "alpha": .001, "n_valid_rows": len(train),
            "n_excluded_rows": int((~valid).sum()), "calibration_donors": sorted(train.donor.unique()),
            "n_original_cells": train.groupby("donor").cell_id.nunique().to_dict()}

def _depth_predict(model, depths):
    depths = np.asarray(depths, float)
    valid = np.isfinite(depths) & (depths > 0)
    log_depth = np.full(len(depths), np.nan)
    log_depth[valid] = np.log1p(depths[valid])
    support = valid & (log_depth >= model["minimum_log_depth"]) & (log_depth <= model["maximum_log_depth"])
    raw = np.full(len(depths), np.nan)
    x = _depth_design(log_depth[support], model["knots"])
    raw[support] = ((x - np.asarray(model["center"])) / np.asarray(model["scale"])) @ np.asarray(model["coef"]) + model["intercept"]
    return np.clip(raw, 0, 1), raw, support
'''


DEPTH_CALIBRATION_CODE = r'''
def run_depth_calibration(smoke=False):
    """Cross-fit reliability on donors unseen by each existing cNMF basis."""
    import shutil
    prefix = "Gray_primary_K8_depth_calibration"
    manifest_path = TABLE_DIR / f"{prefix}_manifest.json"
    previous_path = TABLE_DIR / "Gray_primary_K8_robustness_validation_manifest.json"
    previous = json.loads(previous_path.read_text())
    if previous.get("status") != "completed":
        raise RuntimeError("Completed robustness validation is required")
    prior_root = Path(previous["checkpoint_root"])
    if not prior_root.exists():
        prior_root = CHECKPOINT_DIR / "robustness_validation" / previous["run_id"]
    prior_spec = json.loads((prior_root / "job_spec.json").read_text())
    if stable_token(prior_spec) != previous["run_id"]:
        raise ValueError("Invalid prior run specification")
    scorer = load_frozen_scorer_for_gse268991()
    frozen_hashes = {key: sha256sum(scorer[key]) for key in previous["configuration"]["frozen_model"]}
    if frozen_hashes != previous["configuration"]["frozen_model"]:
        raise ValueError("Frozen model differs from robustness validation")
    programs = ["P1", "P2", "P3", "P4", "P5", "P7", "P9", "P10"]
    inputs, donor_frames, donor_folds, basis_specs = {}, {}, {}, {}
    for fold in range(5):
        path = CHECKPOINT_DIR / "full" / f"fold{fold}_K8" / "job_spec.json"
        basis_specs[fold] = json.loads(path.read_text())
        if not validate_marker(path.parent / "heldout_scoring.complete.json", stable_token(basis_specs[fold])):
            raise ValueError("Missing validated fold-scoring checkpoint")
        inputs[str(path.relative_to(PROJECT))] = sha256sum(path)
    def read_unit(path):
        specification = json.loads((path / "job_spec.json").read_text())
        signature = stable_token({"run_id": previous["run_id"], "specification": specification})
        marker = path / "scoring.complete.json"
        if not validate_marker(marker, signature):
            raise ValueError(f"Missing input marker: {path}")
        inputs[str(marker.relative_to(PROJECT))] = sha256sum(marker)
        quality = pd.read_parquet(path / "quality.parquet")
        usage = pd.read_parquet(path / "usage_normalized.parquet")
        if not quality.index.is_unique or not quality.index.equals(usage.index):
            raise ValueError("Checkpoint cell IDs disagree")
        if not set(programs) <= set(usage):
            raise ValueError("Checkpoint program schema differs")
        return specification, quality, usage
    for donor_dir in sorted((prior_root / "Gray").iterdir()):
        if not donor_dir.is_dir(): continue
        baseline_path = one_file(donor_dir, "full_depth/*/scoring.complete.json").parent
        spec, base_q, base_u = read_unit(baseline_path)
        donor, fold = str(spec["donor"]), int(spec["fold"])
        if donor not in basis_specs[fold]["test_donors"]:
            raise ValueError("Donor was not held out from the cNMF basis")
        donor_folds[donor] = fold
        parts = []
        paths = [baseline_path] + sorted(m.parent for m in donor_dir.glob("UMI_thinning_*/*/scoring.complete.json"))
        if len(paths) != 10: raise ValueError(f"Expected baseline plus nine thinnings: {donor}")
        for path in paths:
            specification, quality, usage = read_unit(path)
            if not quality.index.equals(base_q.index): raise ValueError("Thinning changed cell membership/order")
            part = quality[["library_over_training_universe", "scaled_HVG_relative_error", "valid_objective"]].copy()
            part["cell_id"] = part.index.astype(str)
            part["donor"] = donor; part["fold"] = fold
            part["fraction"] = specification.get("fraction", 1.)
            part["replicate"] = specification.get("replicate", 0)
            for program in programs:
                part[f"absolute_usage_change__{program}"] = (usage[program] - base_u[program]).abs()
            parts.append(part.reset_index(drop=True))
        donor_frames[donor] = pd.concat(parts, ignore_index=True)
    if len(donor_frames) != 22: raise ValueError("Expected all 22 Gray donors")
    configuration = {"stage": "depth_calibration", "schema_version": 1,
        "source_sha256": sha256sum(PROJECT / "build_19b_cNMF_valdiation.py"),
        "prior_run_id": previous["run_id"], "inputs": inputs, "frozen_model": frozen_hashes,
        "quantile": .95, "alpha": .001, "log_depth_hinge_quantiles": [.25, .5, .75],
        "donor_folds": donor_folds, "threads": THREADS,
        "versions": {p: im.version(p) for p in ("numpy", "pandas", "scipy", "scikit-learn")},
        "evidence": "post-evaluation technical reliability; fold-specific, not full-Gray deployment calibration"}
    signature = stable_token(configuration)
    root = CHECKPOINT_DIR / "depth_calibration" / signature
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / "job_spec.json", configuration)
    snapshot = root / "build_19b_cNMF_valdiation.py"
    if not snapshot.exists(): shutil.copy2(PROJECT / snapshot.name, snapshot)
    if sha256sum(snapshot) != configuration["source_sha256"]: raise ValueError("Source snapshot mismatch")
    progress = {"status": "running", "run_id": signature, "configuration": configuration,
                "checkpoint_root": str(root), "expected_units": 198, "completed_units": []}
    def flush():
        progress["updated_UTC"] = datetime.now(timezone.utc).isoformat()
        atomic_json(manifest_path, progress)
    flush()
    targets = {"reconstruction": "scaled_HVG_relative_error", **{p: f"absolute_usage_change__{p}" for p in programs}}
    summaries, splits = [], []
    try:
        for donor in sorted(donor_frames)[:1 if smoke else len(donor_frames)]:
            fold = donor_folds[donor]
            calibration = sorted(d for d in donor_frames if donor_folds[d] == fold and d != donor)
            basis_train = basis_specs[fold]["train_donors"]
            _depth_assert_split(basis_train, calibration, donor)
            splits.append({"test_donor": donor, "fold": fold, "calibration_donors": ";".join(calibration),
                           "basis_training_donors": ";".join(basis_train)})
            for target, column in targets.items():
                unit_spec = {"run_id": signature, "test_donor": donor, "fold": fold, "target": target,
                             "calibration_donors": calibration, "basis_train_donors": basis_train}
                unit_signature = stable_token(unit_spec)
                directory = root / donor / target / unit_signature
                directory.mkdir(parents=True, exist_ok=True)
                marker = directory / "scoring.complete.json"
                atomic_json(directory / "job_spec.json", unit_spec)
                expected = {"model.json", "predictions.parquet", "donor_metrics.parquet"}
                reused = marker.exists()
                if reused:
                    validate_marker(marker, unit_signature)
                    if set(json.loads(marker.read_text())["outputs"]) != expected:
                        raise ValueError("Checkpoint output schema differs")
                    prediction = pd.read_parquet(directory / "predictions.parquet")
                    metrics = pd.read_parquet(directory / "donor_metrics.parquet")
                    model = json.loads((directory / "model.json").read_text())
                else:
                    train = pd.concat([donor_frames[d] for d in calibration], ignore_index=True)
                    test = donor_frames[donor].copy()
                    if target != "reconstruction":
                        train = train.loc[train.fraction.lt(1)]
                        test = test.loc[test.fraction.lt(1)]
                    model = _depth_fit(train, column)
                    bound, unclipped, support = _depth_predict(model, test.library_over_training_universe)
                    prediction = test[["cell_id", "donor", "fold", "fraction", "replicate",
                                       "library_over_training_universe", "valid_objective"]].copy()
                    prediction["observed_error"] = test[column]
                    prediction["in_depth_support"] = support
                    prediction["depth_q95"] = bound
                    prediction["depth_q95_unclipped"] = unclipped
                    prediction["calibration_constant_q95"] = model["constant_q95"]
                    prediction["original_q95"] = previous["configuration"]["q95"] if target == "reconstruction" else np.nan
                    if target == "reconstruction":
                        prediction["mean_absolute_usage_change"] = test[[targets[p] for p in programs]].mean(axis=1)
                    rows = []
                    methods = ["depth_q95", "calibration_constant_q95"]
                    if target == "reconstruction": methods.append("original_q95")
                    for fraction, part in prediction.groupby("fraction"):
                        # Comparable metrics use exactly the same supported observations.
                        valid = (part.valid_objective & part.in_depth_support & np.isfinite(part.observed_error))
                        y = part.loc[valid, "observed_error"].to_numpy(float)
                        for method in methods:
                            bound_values = part.loc[valid, method].to_numpy(float)
                            residual = y - bound_values
                            accepted = residual <= 0
                            row = {"donor": donor, "fold": fold, "target": target, "fraction": fraction,
                                "method": method, "n_calibration_donors": len(calibration),
                                "n_original_cells": part.cell_id.nunique(), "n_observations": len(part),
                                "n_supported_valid_observations": int(valid.sum()),
                                "fraction_in_depth_support": float(part.in_depth_support.mean()),
                                "fraction_invalid": float((~part.valid_objective | ~np.isfinite(part.observed_error)).mean()),
                                "exceedance_fraction": float(np.mean(residual > 0)) if len(y) else np.nan,
                                "pinball_loss": float(np.mean(np.maximum(.95 * residual, -.05 * residual))) if len(y) else np.nan,
                                "mean_bound": float(bound_values.mean()) if len(y) else np.nan,
                                "fraction_bound_saturated": float(np.mean(bound_values >= 1)) if len(y) else np.nan}
                            if target == "reconstruction":
                                change = part.loc[valid, "mean_absolute_usage_change"].to_numpy(float)
                                old_accept = y <= previous["configuration"]["q95"]
                                added = accepted & ~old_accept
                                row.update(accepted_usage_MAE=float(change[accepted].mean()) if accepted.any() else np.nan,
                                    newly_accepted_n=int(added.sum()),
                                    newly_accepted_usage_MAE=float(change[added].mean()) if added.any() else np.nan)
                            rows.append(row)
                    prediction = prediction.reset_index(drop=True)
                    metrics = pd.DataFrame(rows)
                    atomic_json(directory / "model.json", model)
                    atomic_parquet(prediction, directory / "predictions.parquet")
                    atomic_parquet(metrics, directory / "donor_metrics.parquet")
                if (set(prediction.donor.astype(str)) != {donor}
                    or prediction.duplicated(["cell_id", "fraction", "replicate"]).any()
                    or set(model["calibration_donors"]) != set(calibration)
                    or not {"method", "pinball_loss", "target", "donor"} <= set(metrics)):
                    raise ValueError("Checkpoint prediction/model schema or donor identities differ")
                if not reused:
                    write_marker(marker, unit_signature, [directory / x for x in sorted(expected)])
                summaries.append(metrics)
                progress["completed_units"].append({"donor": donor, "target": target, "path": str(directory),
                                                     "signature": unit_signature, "reused": reused})
                flush()
                print(f"depth_calibration {len(progress['completed_units'])}/198: {donor}/{target}", flush=True)
        if smoke:
            progress["status"] = "smoke_completed_not_full_run"; flush(); return
        if len(progress["completed_units"]) != 198: raise ValueError("Incomplete run")
        results = pd.concat(summaries, ignore_index=True)
        save_table(results, f"{prefix}_donor_metrics")
        save_table(pd.DataFrame(splits), f"{prefix}_donor_splits")
        comparison = results.pivot(index=["donor", "fold", "target", "fraction"], columns="method", values="pinball_loss").reset_index()
        comparison["constant_minus_depth_pinball_loss"] = comparison.calibration_constant_q95 - comparison.depth_q95
        save_table(comparison, f"{prefix}_paired_comparison")
        # Each plotted summary is across donors, not across cells or technical repeats.
        colors = {"original_q95": "#777777", "calibration_constant_q95": "#CC6677", "depth_q95": "#4477AA"}
        labels = {"original_q95": "Original q95", "calibration_constant_q95": "Calibration constant q95", "depth_q95": "Depth-dependent q95"}
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for method in colors:
            part = results.loc[results.target.eq("reconstruction") & results.method.eq(method)]
            g = part.groupby("fraction").exceedance_fraction
            x = g.median().index.to_numpy(); median = g.median().to_numpy()
            ax.plot(x, median, marker="o", color=colors[method], label=labels[method])
            ax.fill_between(x, g.quantile(.25), g.quantile(.75), color=colors[method], alpha=.12)
        ax.set(xlabel="Retained UMI fraction", ylabel="Fraction exceeding error threshold", ylim=(0, 1.02), xticks=[.1, .25, .5, 1])
        ax.set_title("Gray: held-out donor reconstruction reliability\nMedian and interquartile range across 22 donors")
        ax.legend(frameon=False)
        save_figure(fig, f"{prefix}_reconstruction")
        fig, ax = plt.subplots(figsize=(7, 4.5))
        subset = results.loc[results.target.isin(programs) & results.fraction.eq(.25) & results.method.eq("depth_q95")].copy()
        subset["coverage"] = 1 - subset.exceedance_fraction
        for i, program in enumerate(programs):
            values = subset.loc[subset.target.eq(program), "coverage"].dropna()
            ax.plot([values.quantile(.25), values.quantile(.75)], [i, i], color="#4477AA", linewidth=2)
            ax.plot(values.median(), i, "o", color="#4477AA")
        ax.axvline(.95, color="black", linestyle="--", linewidth=.7)
        ax.set(yticks=range(len(programs)), yticklabels=programs, xlim=(0, 1.02), xlabel="Fraction within predicted technical-error bound")
        ax.set_title("Gray: program reliability at 25% retained UMI\nDonor median and interquartile range; dashed line: nominal 0.95")
        save_figure(fig, f"{prefix}_program_reliability")
        if frozen_hashes != {key: sha256sum(scorer[key]) for key in frozen_hashes}:
            raise ValueError("Frozen artifacts changed")
        progress["status"] = "completed"
        progress["results"] = {str(x.relative_to(PROJECT)): sha256sum(x) for x in sorted(TABLE_DIR.glob(f"{prefix}_*.csv"))}
        progress["figures"] = {str(x.relative_to(PROJECT)): sha256sum(x) for x in sorted(FIGURE_DIR.glob(f"{prefix}_*.pdf"))}
        progress["limitations"] = "3–4 calibration donors per unit; technical-reference errors, not biological truth; not calibrated for full-Gray K8 or external cohorts"
        flush()
        display(results.loc[results.target.eq("reconstruction")].groupby(["fraction", "method"])[
            ["exceedance_fraction", "pinball_loss", "fraction_in_depth_support", "fraction_bound_saturated"]].median())
        display(comparison.groupby(["target", "fraction"]).constant_minus_depth_pinball_loss.agg(["median", "min", "max"]))
        print("Completed 198 reliability units. Lower rejection alone is not evidence of better biology.")
    except Exception as error:
        progress.update(status="failed", error=f"{type(error).__name__}: {error}")
        flush(); raise

if EXECUTION_STAGE == "depth_calibration":
    run_depth_calibration(smoke=os.environ.get("GDT_19B_DEPTH_SMOKE", "0") == "1")
else:
    print("Depth calibration not launched in stage", EXECUTION_STAGE)
'''


INCREMENTAL_HELPERS = r'''
def _iv_same_rows(expected, actual):
    if not expected.is_unique or not actual.is_unique or len(expected) != len(actual) or set(expected) != set(actual):
        raise ValueError("Historical evaluation observation identities changed")


def _iv_splits(metadata, split, outer_heldout):
    """Leave one inner group out without ever accessing outer-test outcomes."""
    if not metadata.index.is_unique or metadata[[split, "donor"]].isna().any().any():
        raise ValueError("Duplicate IDs or missing split identity")
    pool = metadata.loc[metadata[split].astype(str).ne(str(outer_heldout))]
    groups = sorted(pool[split].astype(str).unique())
    if len(groups) < 2:
        raise ValueError("Nested validation needs at least two inner groups")
    for group in groups:
        test = pool.index[pool[split].astype(str).eq(group)]
        train = pool.index[pool[split].astype(str).ne(group)]
        if set(metadata.loc[train, "donor"]) & set(metadata.loc[test, "donor"]):
            raise ValueError("A donor crosses the inner split")
        yield group, train, test


def _iv_fit(X, Y, metadata, train, test, classification, penalty):
    """Fit one training-only decoder; return predictions, donor losses and scaling."""
    import warnings
    import numpy as np
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.metrics import log_loss, balanced_accuracy_score
    from sklearn.exceptions import ConvergenceWarning
    if (not X.index.is_unique or not X.index.equals(Y.index)
            or not X.index.equals(metadata.index) or set(train) & set(test)
            or set(metadata.loc[train, "donor"]) & set(metadata.loc[test, "donor"])):
        raise ValueError("Unaligned rows or donor leakage")
    numeric = list(X.select_dtypes(include="number").columns)
    categorical = [c for c in X if c not in numeric]
    if not np.isfinite(X[numeric].to_numpy()).all() or X[categorical].isna().any().any():
        raise ValueError("Missing/nonfinite predictors; do not silently drop test rows")
    preprocess = ColumnTransformer([
        ("numeric", StandardScaler(), numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical)])
    estimator = (LogisticRegression(C=float(penalty), max_iter=5000, solver="lbfgs", random_state=20261023)
                 if classification else Ridge(alpha=float(penalty)))
    donors = metadata.loc[train, "donor"]
    weights = donors.map(1 / donors.value_counts()).to_numpy(dtype=float, copy=True)
    weights *= len(weights) / weights.sum()
    preprocess.fit(X.loc[train])
    scaler = preprocess.named_transformers_["numeric"]
    scaler.fit(X.loc[train, numeric], sample_weight=weights)
    train_X = preprocess.transform(X.loc[train])
    test_X = preprocess.transform(X.loc[test])
    audit = []
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        if classification:
            labels = Y.loc[train].iloc[:, 0].astype(str)
            if set(labels) != {"CR", "MO", "HDMAPP"}:
                raise ValueError("Training partition lacks a stimulation class")
            estimator.fit(train_X, labels, sample_weight=weights)
            prediction = estimator.predict_proba(test_X)
            classes = estimator.classes_
            pred = pd.DataFrame(prediction, index=test, columns=[f"probability_{c}" for c in classes])
            pred["observed"] = Y.loc[test].iloc[:, 0].astype(str)
        else:
            values = Y.loc[train].to_numpy(float)
            if not np.isfinite(Y.to_numpy(float)).all(): raise ValueError("Nonfinite protein endpoint")
            center = np.average(values, axis=0, weights=weights)
            scale = np.sqrt(np.average((values - center) ** 2, axis=0, weights=weights))
            if (scale < 1e-12).any(): raise ValueError("Constant training endpoint")
            estimator.fit(train_X, (values - center) / scale, sample_weight=weights)
            prediction = estimator.predict(test_X)
            truth = (Y.loc[test].to_numpy(float) - center) / scale
            pred = pd.DataFrame(prediction, index=test,
                                columns=[f"predicted_standardized__{c}" for c in Y])
            for i, c in enumerate(Y):
                pred[f"observed_standardized__{c}"] = truth[:, i]
                audit.append({"kind": "outcome", "feature": c, "center": center[i], "scale": scale[i]})
    for c, mean, scale in zip(numeric, scaler.mean_, scaler.scale_):
        audit.append({"kind": "predictor", "feature": c, "center": mean, "scale": scale})
    pred["donor"] = metadata.loc[test, "donor"].astype(str)
    if "Experiment" in metadata: pred["Experiment"] = metadata.loc[test, "Experiment"].astype(str)
    metrics = []
    for donor, part in pred.groupby("donor", sort=True):
        row = {"donor": donor, "n_observations": len(part), "n_training_donors": donors.nunique()}
        if classification:
            probabilities = part[[f"probability_{c}" for c in classes]].to_numpy()
            row["loss"] = row["log_loss"] = log_loss(part.observed, probabilities, labels=classes)
            row["balanced_accuracy"] = balanced_accuracy_score(part.observed, classes[probabilities.argmax(axis=1)])
            encoded = part.observed.to_numpy()[:, None] == classes[None, :]
            row["multiclass_Brier_score"] = float(np.square(probabilities - encoded).sum(axis=1).mean())
        else:
            errors = []
            for c in Y:
                error = float(np.square(part[f"predicted_standardized__{c}"] -
                                        part[f"observed_standardized__{c}"]).mean())
                row[f"MSE__{c}"] = error; errors.append(error)
            row["loss"] = row["standardized_MSE"] = float(np.mean(errors))
        if "Experiment" in part:
            if part.Experiment.nunique() != 1: raise ValueError("Donor crosses Experiments")
            row["Experiment"] = part.Experiment.iloc[0]
        metrics.append(row)
    return pred, pd.DataFrame(metrics), pd.DataFrame(audit)


def _iv_select(X, Y, metadata, split, heldout, classification, grid):
    """Same inner folds/budget for all methods; ties favour stronger shrinkage."""
    import numpy as np
    import pandas as pd
    records = []
    for group, train, test in _iv_splits(metadata, split, heldout):
        for penalty in grid:
            _, metrics, _ = _iv_fit(X, Y, metadata, train, test, classification, penalty)
            records.append({"inner_heldout": group, "penalty": float(penalty),
                            "loss": metrics.loss.mean(), "n_test_donors": len(metrics),
                            "training_donors": ";".join(sorted(metadata.loc[train, "donor"].astype(str).unique())),
                            "test_donors": ";".join(sorted(metadata.loc[test, "donor"].astype(str).unique()))})
    frame = pd.DataFrame(records)
    means = frame.groupby("penalty").loss.mean()
    best = means.min()
    candidates = means.index[np.isclose(means, best, atol=1e-12, rtol=0)]
    chosen = float(min(candidates) if classification else max(candidates))
    frame["selected_penalty"] = chosen
    return chosen, frame


def _iv_hash(path):
    import hashlib
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def _iv_checkpoint(folder, signature, schemas, produce):
    """Atomic per-library/per-outer-fit checkpoints with fail-closed reuse."""
    import json, os, uuid
    from pathlib import Path
    import pandas as pd
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    marker = folder / "scoring.complete.json"
    reused = marker.is_file()
    if reused:
        payload = json.loads(marker.read_text())
        if payload.get("signature") != signature or set(payload.get("outputs", {})) != {f"{n}.parquet" for n in schemas}:
            raise ValueError("Checkpoint signature/schema changed")
        for name, digest in payload["outputs"].items():
            if not (folder / name).is_file() or _iv_hash(folder / name) != digest:
                raise ValueError("Checkpoint failed hash validation")
        frames = {name: pd.read_parquet(folder / f"{name}.parquet") for name in schemas}
    else:
        frames = produce()
    if set(frames) != set(schemas): raise ValueError("Checkpoint frame schema mismatch")
    for name, required in schemas.items():
        if not isinstance(frames[name], pd.DataFrame) or not frames[name].index.is_unique or not set(required) <= set(frames[name]):
            raise ValueError(f"Checkpoint table schema/index invalid: {name}")
    if not reused:
        outputs = {}
        for name, frame in frames.items():
            target = folder / f"{name}.parquet"
            temporary = folder / f".{name}.{uuid.uuid4().hex}.tmp"
            frame.to_parquet(temporary); os.replace(temporary, target)
            outputs[target.name] = _iv_hash(target)
        temporary = folder / f".complete.{uuid.uuid4().hex}.tmp"
        temporary.write_text(json.dumps({"signature": signature, "outputs": outputs}, indent=2))
        os.replace(temporary, marker)
    return frames, reused


def _iv_plot_added_value(paired, dataset, gate, split, font):
    """One panel: does adding cNMF improve held-out loss in each biological unit?"""
    import numpy as np
    import pandas as pd
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    if font not in ("Arial", "Helvetica", "Liberation Sans"): raise ValueError("Unapproved figure font")
    bases = [("covariates_only", "Covariates only"),
             ("covariates_plus_rank_signature", "Top-50 rank (frozen)"),
             ("covariates_plus_AUCell", "Top-50 AUCell (frozen)"),
             ("covariates_plus_full_measured_AUCell", "Top-50 AUCell (full measured)"),
             ("covariates_plus_Hallmark_AUCell", "Hallmark AUCell (frozen)"),
             ("covariates_plus_full_measured_Hallmark_AUCell", "Hallmark AUCell (full measured)")]
    source = []
    with mpl.rc_context({"font.family": font, "font.size": 7, "font.weight": "normal", "axes.titleweight": "normal",
                         "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "text.color": "black", "axes.labelcolor": "black", "axes.titlecolor": "black",
                         "xtick.color": "black", "ytick.color": "black", "pdf.fonttype": 42, "svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=(180 / 25.4, 90 / 25.4))
        fig.subplots_adjust(left=.36, right=.97, bottom=.20, top=.84)
        ax.axvline(0, color="black", lw=.6, linestyle="--", zorder=0)
        for y, (baseline, label) in enumerate(bases):
            method = "covariates_plus_cNMF" if baseline == "covariates_only" else baseline + "_plus_cNMF"
            part = paired.loc[paired.baseline.eq(baseline) & paired.model.eq(method)
                              & paired.quality_gate.eq(gate) & paired.split.eq(split)].copy()
            unit = "Experiment" if split == "Experiment" else "donor"
            if part.empty or part.donor.duplicated().any(): raise ValueError("Empty or duplicated paired contrast")
            values = part.groupby(unit, sort=True).error_reduction.mean()
            offsets = np.linspace(-.13, .13, len(values))
            ax.scatter(values, y + offsets, s=12, color="#0072B2", edgecolors="none", zorder=2)
            ax.plot([values.mean()] * 2, [y - .22, y + .22], color="black", lw=1.0, zorder=3)
            for identifier, value in values.items():
                source.append({"dataset": dataset, "quality_gate": gate, "split": split, "unit_type": unit,
                               "unit": identifier, "baseline": baseline, "model": method, "error_reduction": value})
        ax.set_yticks(range(len(bases)), [label for _, label in bases]); ax.invert_yaxis()
        ax.set_xlabel("Baseline loss minus loss after adding cNMF\nPositive values favour adding cNMF")
        gate_label = "All cells" if gate == "all_cells" else "Gray q95 accepted"
        ax.set_title(f"{dataset}: {gate_label}; held-out {split}\nPoints: {len(values)} {unit} units; black marks: equal-unit mean", pad=9)
        ax.spines[["top", "right"]].set_visible(False)
        for text in fig.findobj(mpl.text.Text): text.set_color("black"); text.set_fontweight("normal")
    return fig, pd.DataFrame(source)
'''


EVIDENCE_SYNTHESIS_HELPERS = r'''
def _es_native_serum_contrasts(scores):
    native = scores.loc[scores.scenario.eq('native') & scores.day.gt(0)]
    if set(native.serum.unique()) != {'+', '-'}:
        raise ValueError('Unexpected source serum labels; expected + and -')
    wide = native.pivot(index=['donor','day','program','gate','replicate'],
                        columns='serum', values='median_usage')
    return (wide['+'] - wide['-']).rename('recomputed').reset_index()


def _es_donor_summary(frame, keys, value):
    # Technical replicates are averaged within donor before across-donor summaries.
    donor = frame.groupby(keys + ['donor'], dropna=False)[value].mean().reset_index()
    return donor.groupby(keys, dropna=False)[value].agg(
        n_donors='count', n_donors_total='size', donor_median='median',
        donor_min='min', donor_max='max').assign(
            n_donors_missing=lambda d: d.n_donors_total - d.n_donors).reset_index()


def _es_overlap(predictions, usages):
    if predictions.cell_id.isna().any() or usages.cell_id.isna().any():
        raise ValueError('Missing cell identities')
    a, b = predictions.cell_id.astype(str), usages.cell_id.astype(str)
    if a.duplicated().any() or b.duplicated().any(): raise ValueError('Duplicate cell identities')
    common = set(a) & set(b)
    result = {'n_lineage_cells': len(a), 'n_state_cells': len(b), 'n_overlap': len(common),
              'n_lineage_only': len(set(a) - set(b)), 'n_state_only': len(set(b) - set(a))}
    if 'hard_call' in predictions:
        positive = set(a.loc[predictions.hard_call.eq('predicted gdT')])
        result.update(n_predicted_gdT=len(positive), n_predicted_gdT_with_state=len(positive & set(b)),
                      n_predicted_gdT_without_state=len(positive - set(b)),
                      n_state_not_predicted_gdT=len(set(b) - positive))
    return result


def _es_spectrum_pairs(spectra):
    import numpy as np
    import pandas as pd
    x = spectra.to_numpy(float)
    if not np.isfinite(x).all() or (x < 0).any() or (np.linalg.norm(x, axis=1) == 0).any():
        raise ValueError('Invalid frozen spectra')
    cosine = x @ x.T / np.outer(np.linalg.norm(x, axis=1), np.linalg.norm(x, axis=1))
    return pd.DataFrame([{'program_a': a, 'program_b': b, 'cosine': cosine[i, j],
        'scope': 'Frozen K8 HVG-spectrum similarity; not proof of non-identifiability'}
        for i, a in enumerate(spectra.index) for j, b in enumerate(spectra.index) if i < j])
'''


EVIDENCE_SYNTHESIS_CODE = r'''
def run_evidence_synthesis():
    import pyarrow.parquet as pq
    prefix = 'Gray_primary_K8_evidence_synthesis'
    serum_prefix = 'Zenodo17590095_primary_K8_evidence_synthesis'
    audit_prefix = '19b_cNMF_valdiation_end_to_end_audit'
    inputs = {}
    def register(path):
        path = Path(path)
        inputs[str(path.relative_to(PROJECT))] = sha256sum(path)
        return path
    def read(name):
        return pd.read_csv(register(TABLE_DIR / name))
    manifests = {}
    for name in ['Gray_primary_K8_robustness_validation_manifest.json',
                 'Gray_primary_K8_depth_calibration_manifest.json',
                 'Gray_primary_K8_P7_diagnostics_manifest.json',
                 'Zenodo17590095_primary_K8_time_course_manifest.json',
                 'Gray_primary_K8_usage_representation_validation_manifest.json']:
        manifest = json.loads(register(TABLE_DIR / name).read_text())
        if manifest.get('status') != 'completed': raise ValueError('Incomplete source: ' + name)
        manifests[name] = manifest
    def verify_recorded(path):
        path = Path(path); rel = str(path.relative_to(PROJECT)); recorded = []
        for manifest in manifests.values():
            for key in ('results', 'outputs'):
                for n, digest in manifest.get(key, {}).items():
                    if n in (rel, str(path)): recorded.append(digest)
        if recorded and sha256sum(path) not in recorded: raise ValueError('Source checksum mismatch: ' + rel)
    scorer = load_frozen_scorer_for_gse268991()
    previous = manifests['Gray_primary_K8_usage_representation_validation_manifest.json']
    frozen = {k: sha256sum(register(scorer[k])) for k in previous['configuration']['frozen_model']}
    if frozen != previous['configuration']['frozen_model']: raise ValueError('Frozen model mismatch')
    card = json.loads(register(TABLE_DIR / 'primary_K8_model_card.json').read_text())
    mapping = card['program_mapping_to_K10_reference']
    programs = list(mapping.values())
    if len(programs) != 8 or len(set(programs)) != 8: raise ValueError('Invalid K8 program mapping')
    fold = read('Gray_primary_K8_robustness_validation_fold_matching.csv')
    recurrence = read('Gray_full_cross_K_program_recurrence.csv')
    within = read('Gray_full_within_K_stability.csv')
    composite = read('Gray_full_P9_P10_composite_summary.csv')
    robustness = read('Gray_primary_K8_robustness_validation_donor_metrics.csv')
    calibration = read('Gray_primary_K8_depth_calibration_donor_metrics.csv')
    p7 = read('Gray_primary_K8_P7_diagnostics_donor_metrics.csv')
    ranked = read('Gray_primary_K8_ranked_genes.csv')
    prevalence = read('GSE294268_primary_K8_program_prevalence.csv')
    atlas = read('PANGEA_primary_K8_program_prevalence.csv')
    adt = read('GSE268991_primary_K8_program_ADT_associations.csv')
    serum = read('Zenodo17590095_primary_K8_time_course_paired_serum_contrasts.csv')
    scores = read('Zenodo17590095_primary_K8_time_course_donor_condition_scores.csv')
    quality = read('Zenodo17590095_primary_K8_time_course_quality.csv')
    decomposition = read('Zenodo17590095_primary_K8_time_course_RNA_stratum_decomposition.csv')
    depth = read('Zenodo17590095_primary_K8_time_course_depth_sensitivity.csv')
    coverage = read('Zenodo17590095_primary_K8_time_course_coverage.csv')
    external_coverage = pd.concat([read(n + '_primary_K8_robustness_validation_coverage.csv')
                                  for n in ['GSE268991','GSE294268']], ignore_index=True)
    legacy_path = register(PROJECT / 'results/19_scoring_biology/scoring_biology_v1/tables/biological_evidence_table.csv')
    legacy = pd.read_csv(legacy_path).set_index('program')
    for rel in list(inputs): verify_recorded(PROJECT / rel)
    spectra = scorer['spectra'].copy() if 'spectra' in scorer else pd.read_csv(scorer['spectra_path'], sep='\t', index_col=0)
    spectra.index = [mapping[str(int(float(i)))] for i in spectra.index]
    pairs = _es_spectrum_pairs(spectra)
    # Audit only saved files; names are evidence locators, not evidence of correctness.
    inventory = []
    candidates = sorted(set(PROJECT.glob('predictions/*/*.parquet')) |
        set(PROJECT.glob('predictions/*/*.csv')) |
        set(PROJECT.glob('31_subtype_Examination_PMC13425591/*predictions.parquet')))
    for path in candidates:
        if path.suffix == '.parquet':
            f = pq.ParquetFile(path); columns = f.schema.names; n = f.metadata.num_rows
        else:
            columns = list(pd.read_csv(path, nrows=0)); n = None
        inventory.append({'path': str(path.relative_to(PROJECT)), 'sha256': sha256sum(register(path)),
            'n_rows': n, 'columns': '|'.join(columns),
            'lineage_columns': '|'.join(c for c in columns if c in ['hard_call','P_gdT','p_gdT_M05','end_to_end_call']),
            'subtype_columns': '|'.join(c for c in columns if 'subtype' in c.lower() or c.startswith('p_Vd')),
            'K8_named_program_columns': '|'.join(c for c in columns if c in programs),
            'evidence_warning': 'Column presence is not verified provenance, independent truth or complete pipeline validation'})
    pangea_manifest = json.loads(register(TABLE_DIR / 'PANGEA_primary_K8_projection_manifest.json').read_text())
    state_path = register(Path(pangea_manifest['usage_path']))
    if sha256sum(state_path) != pangea_manifest['usage_sha256']: raise ValueError('PANGEA score checksum mismatch')
    state = pd.read_parquet(state_path)
    if 'cell_id' not in state: state = state.assign(cell_id=state.index.astype(str))
    pred_path = PROJECT / 'predictions/22_LR_hierarchical_CE/PANGEA_LR_hierarchical_predictions.parquet'
    overlap = {}
    if pred_path.exists():
        predictions = pd.read_parquet(register(pred_path))
        if 'cell_id' not in predictions: predictions = predictions.assign(cell_id=predictions.index.astype(str))
        overlap = _es_overlap(predictions, state)
    # Every input hash, including the saved prediction inventory, enters the run signature.
    config = {'stage': 'evidence_synthesis', 'source_sha256': sha256sum(PROJECT / 'build_19b_cNMF_valdiation.py'),
              'inputs': inputs, 'frozen_model': frozen, 'program_mapping': mapping,
              'versions': {k: im.version(k) for k in ['numpy','pandas','matplotlib','pyarrow']},
              'scope': 'Existing-evidence synthesis; no retraining or threshold selection', 'schema_version': 1}
    run_id = stable_token(config); root = CHECKPOINT_DIR / 'evidence_synthesis' / run_id
    root.mkdir(parents=True, exist_ok=True); atomic_json(root / 'job_spec.json', config)
    _ur_snapshot(PROJECT / 'build_19b_cNMF_valdiation.py', root / 'build_19b_cNMF_valdiation.py')
    progress = {'status': 'running', 'run_id': run_id, 'configuration': config,
                'checkpoint_root': str(root), 'completed_units': [], 'expected_units': 11}
    def flush():
        progress['updated_UTC'] = datetime.now(timezone.utc).isoformat()
        atomic_json(TABLE_DIR / (prefix + '_manifest.json'), progress)
    def unit(name, schemas, produce):
        signature = stable_token({'run_id': run_id, 'unit': name})
        frames, reused = _iv_checkpoint(root / name, signature, schemas, produce)
        progress['completed_units'].append({'label': name, 'signature': signature,
                                           'path': str(root / name), 'reused': reused}); flush()
        return frames
    outputs = []
    def table(frame, name):
        save_table(frame, name); outputs.append(TABLE_DIR / (name + '.csv'))
    flush()
    try:
        cards = []
        for component, program in mapping.items():
            def produce_card():
                f = fold.loc[fold.program.eq(program)]
                if len(f) != 5 or f.fold.nunique() != 5: raise ValueError('Incomplete fold evidence: ' + program)
                r = recurrence.loc[recurrence.reference_program.eq(int(program[1:]))].sort_values('K')
                b = robustness.loc[robustness.program.eq(program) & robustness.scenario.eq('UMI_thinning') & robustness.fraction.eq(.25)]
                b = b.groupby('donor')[['usage_MAE','usage_spearman_rho']].mean()
                c = calibration.loc[calibration.target.eq(program) & calibration.method.eq('depth_q95') & calibration.fraction.eq(.25)]
                nearest = pairs.loc[pairs.program_a.eq(program) | pairs.program_b.eq(program)].sort_values('cosine', ascending=False).iloc[0]
                caution = 'Relative expression contribution; not a probability, absolute activity or functional output. '
                if program == 'P7': caution += 'High-P7 Gray donors undercovered; low-P7 serum results do not repair this. '
                if program in ['P9','P10']: caution += 'Interpret with P9/P10 composite and split/merge evidence; do not assert two discrete cycling states. '
                if program == 'P10': caution += 'Weakest fold match and sparse external support require caution. '
                e = prevalence.loc[prevalence.K10_reference_program.eq(program) & prevalence.quality_gate.eq('Gray_q95_accepted')]
                a = atlas.loc[atlas.K10_reference_program.eq(program) & atlas.quality_gate.eq('Gray_q95_accepted')]
                cov = coverage.set_index('program').loc[program]
                ecov = external_coverage.loc[external_coverage.program.eq(program)]
                row = {'program': program, 'K8_component': component,
                    'historical_K10_provisional_interpretation': legacy.loc[int(program[1:]), 'provisional_alternatives'],
                    'annotation_scope': 'Inherited K10 hypothesis via K8 mapping; not a new K8 pathway test or functional validation',
                    'K8_top10_genes': ';'.join(ranked.loc[ranked.K8_component.eq(int(component))].sort_values('rank').head(10).gene),
                    'n_fold_models': len(f), 'fold_to_full_cosine_min': f.cosine_on_common_HVGs.min(),
                    'fold_to_full_cosine_median': f.cosine_on_common_HVGs.median(),
                    'fold_match_caveat': 'Full-Gray reference shares donors; not independent K selection',
                    'cross_K_recurrence': ';'.join(f'K{int(x.K)}:{int(x.n_passing_folds)}/{int(x.n_completed_folds)}' for x in r.itertuples()),
                    'cross_K_caveat': 'Full K10 reference; one-to-one competition penalizes low K; descriptive only',
                    'nearest_spectrum_program': nearest.program_b if nearest.program_a == program else nearest.program_a,
                    'nearest_spectrum_cosine': nearest.cosine,
                    'identifiability_caveat': 'Similarity is not proof of usage ambiguity; historical K10 ablations are not K8 validation',
                    'quarter_UMI_n_donors': len(b), 'quarter_UMI_donor_median_MAE': b.usage_MAE.median(),
                    'quarter_UMI_donor_max_MAE': b.usage_MAE.max(),
                    'quarter_UMI_donor_median_rho': b.usage_spearman_rho.median(),
                    'quarter_UMI_bound_n_donors': c.donor.nunique(),
                    'quarter_UMI_bound_exceedance_median': c.exceedance_fraction.median(),
                    'quarter_UMI_bound_exceedance_max': c.exceedance_fraction.max(),
                    'quarter_UMI_bound_worst_donor': str(c.loc[c.exceedance_fraction.idxmax(), 'donor']),
                    'calibration_scope': 'Gray fold-specific technical errors; NOT calibrated for full K8 or external cohorts',
                    'GSE294268_q95_median_usage': float(e.median_usage.iloc[0]) if len(e) else np.nan,
                    'PANGEA_q95_median_usage': float(a.median_usage.iloc[0]) if len(a) else np.nan,
                    'GSE268991_GSE294268_min_aligned_HVG_coverage': ecov.aligned_HVG_coverage.min(),
                    'GSE268991_GSE294268_min_top50_coverage': ecov.top50_signature_coverage.min(),
                    'GSE268991_GSE294268_min_scaled_L1_coverage': ecov.scaled_basis_L1_coverage.min(),
                    'Zenodo17590095_HVG_coverage': cov.HVG_coverage,
                    'Zenodo17590095_top50_coverage': cov.top50_coverage,
                    'Zenodo17590095_weighted_L1_coverage': cov.weighted_L1_coverage,
                    'coverage_caveat': 'Overall HVG gate does not certify each program; reported coverage concerns available features, not per-cell nonzero expression',
                    'external_scope': 'AIM-selected stimulation and silver-labelled atlas; not population prevalence or independent lineage truth',
                    'input_contract': 'Raw UMI; frozen training-universe normalization and SD; >=80% HVGs; inspect program-specific gene coverage',
                    'quality_contract': 'Historical q95 is a reconstruction gate, not 95% reliability for each program',
                    'when_to_use': 'Descriptive continuous programs in supported gamma-delta populations; retain donor and QC context',
                    'when_to_use_caution': caution,
                    'unsupported_claims': 'Killing; degranulation; causal serum factor; calibrated state probability; universal baseline superiority',
                    'source_manifest': prefix + '_manifest.json'}
                return {'program_card': pd.DataFrame([row])}
            cards.append(unit(program, {'program_card': ['program','K8_component']}, produce_card)['program_card'])
        table(pd.concat(cards, ignore_index=True), prefix + '_program_cards')
        table(pairs, prefix + '_spectrum_pairs')
        table(within, prefix + '_within_K_stability')
        table(composite, prefix + '_P9_P10_composite_summary')
        table(adt, prefix + '_GSE268991_program_ADT_associations')
        table(external_coverage, prefix + '_external_coverage')
        table(_es_donor_summary(robustness, ['program','scenario','fraction','observed_feature_fit'], 'usage_MAE'), prefix + '_perturbation_summary')
        table(calibration, prefix + '_calibration_donor_metrics')
        table(p7, prefix + '_P7_diagnostics_donor_metrics')
        def serum_unit():
            keys = ['donor','day','program','scenario','replicate','gate']
            if serum.duplicated(keys).any(): raise ValueError('Duplicate paired serum contrast')
            recomputed = _es_native_serum_contrasts(scores)
            check = serum.loc[serum.scenario.eq('native')].merge(recomputed, on=['donor','day','program','gate','replicate'], validate='one_to_one')
            if len(check) != len(recomputed) or not np.allclose(check.WITH_SERUM_minus_WO_SERUM, check.recomputed, equal_nan=True):
                raise ValueError('Saved serum contrasts do not reproduce')
            valid = decomposition.restricted_mean_difference.notna()
            if not np.allclose(decomposition.loc[valid,'within_RNA_stratum'] + decomposition.loc[valid,'composition'],
                               decomposition.loc[valid,'restricted_mean_difference']): raise ValueError('Decomposition does not add up')
            q = quality.copy(); q['fraction_historical_q95_accepted'] = q.n_historical_q95_accepted / q.n_valid_converged.replace(0, np.nan)
            return {'paired_contrasts': serum, 'donor_summary': _es_donor_summary(serum, ['program','day','scenario','gate'], 'WITH_SERUM_minus_WO_SERUM'),
                    'quality': q, 'decomposition': decomposition, 'depth_sensitivity': depth, 'coverage': coverage}
        frames = unit('serum_time_course', {'paired_contrasts':['program','donor'], 'donor_summary':['program','n_donors'],
                     'quality':['donor','fraction_historical_q95_accepted'], 'decomposition':['program'],
                     'depth_sensitivity':['program'], 'coverage':list(coverage.columns[:1])}, serum_unit)
        for key, frame in frames.items(): table(frame, serum_prefix + '_' + key)
        def audit_unit():
            rows = [
                {'dataset':'GSE268991', 'available':'Frozen K8, same-cell ADT and productive gamma-delta TCR metadata',
                 'missing_or_limitation':'No matching deployed first-layer predictions identified in inspected prediction directories; absence of TRGC is not a negative truth label',
                 'end_to_end_status':'Not established', 'next_required':'Confirm deployment model/version; score full eligible cohort and obtain independently justified negatives'},
                {'dataset':'PANGEA', 'available':'22_LR_hierarchical_CE predictions and preselected 19b K8 scores',
                 'missing_or_limitation':'Expression-derived silver labels; state scores do not cover all lineage-predicted candidates; no verified donor IDs',
                 'end_to_end_status':'Not established', 'next_required':'Independent labels and full candidate coverage; do not call Series a donor'},
                {'dataset':'Song / SPL / 15-organ', 'available':'Saved hierarchical lineage results and historical K10 state projections',
                 'missing_or_limitation':'Historical K10 projections are not current K8; training/adaptation membership must be checked',
                 'end_to_end_status':'Not established for frozen K8', 'next_required':'Choose approved lineage release and independent evaluation cohort before new projections'},
                {'dataset':'Zenodo17590095', 'available':'RNA-only frozen K8 time course, three donors',
                 'missing_or_limitation':'No matched first-layer evaluation; functional donors differ; citeseq provenance unresolved',
                 'end_to_end_status':'Application only', 'next_required':'Do not use as donor-matched RNA-killing validation'},
                {'dataset':'31_subtype_Examination_PMC13425591', 'available':'Saved subtype probabilities and selected gene-expression diagnostics',
                 'missing_or_limitation':'The filename subtype_state_predictions does not supply the eight frozen K8 usages',
                 'end_to_end_status':'Not established', 'next_required':'Verify layer compatibility rather than infer it from filename'}]
            routing = pd.DataFrame([
                {'question':'Can uncertain subtype retain state outputs?', 'current_evidence':'19b scores without requiring a confident Vd1/Vd2 call',
                 'status':'Scorer-level independence; deployed end-to-end routing not demonstrated',
                 'required_behavior':'Retain subtype uncertainty and state QC separately; do not infer subtype from a state program'},
                {'question':'Does lineage contamination alter donor program distributions?', 'current_evidence':'No complete independently labelled same-cell deployed K8 pipeline found in inspected scope',
                 'status':'Not estimable from current joined evidence',
                 'required_behavior':'Compare independently confirmed gamma-delta population with all predicted positives; include false negatives and unknown truth separately'}])
            return {'cohort_readiness':pd.DataFrame(rows), 'prediction_inventory':pd.DataFrame(inventory),
                    'PANGEA_overlap':pd.DataFrame([overlap]), 'routing_requirements':routing}
        af = unit('end_to_end_audit', {'cohort_readiness':['dataset'], 'prediction_inventory':['path'],
                  'PANGEA_overlap':['n_overlap'], 'routing_requirements':['question']}, audit_unit)
        for key, frame in af.items(): table(frame, audit_prefix + '_' + key)
        legends = []
        def figure(fig, data, name, legend):
            table(data, name + '_source_data')
            with mpl.rc_context({'savefig.bbox':None, 'pdf.fonttype':42}):
                fig.savefig(FIGURE_DIR / (name + '.pdf'), bbox_inches=None)
                fig.savefig(FIGURE_DIR / (name + '.png'), bbox_inches=None, dpi=300)
            outputs.append(FIGURE_DIR / (name + '.pdf')); display(fig); plt.close(fig)
            legends.append({'figure':name, 'legend':legend, 'alignment':'Not applicable: single panel',
                            'QA':'Rendered PDF checks required after execution'})
        with mpl.rc_context({'font.family':FIGURE_FONT,'font.size':7,'axes.titlesize':7,'axes.labelsize':7,
                             'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7}):
            # Each donor line is visible; technical replicates never inflate n.
            for program in ['P2','P7']:
                for gate in ['all_valid','historical_q95']:
                    source = serum.loc[serum.program.eq(program) & serum.gate.eq(gate)].copy()
                    fig, ax = plt.subplots(figsize=(180/25.4,85/25.4)); fig.subplots_adjust(left=.13,right=.75,bottom=.20,top=.83)
                    styles = {'native':('-', 'o'), 'half_depth':('--','s'), 'matched_depth':(':','^')}
                    colors = {'donorA':'#0072B2','donorB':'#D55E00','donorC':'#009E73'}
                    for donor, color in colors.items():
                        for scenario, (style, marker) in styles.items():
                            part = source.loc[source.donor.eq(donor) & source.scenario.eq(scenario)].groupby('day').WITH_SERUM_minus_WO_SERUM.agg(['mean','min','max'])
                            ax.plot(part.index,part['mean'],color=color,linestyle=style,marker=marker,markersize=3,linewidth=.8,label=donor+' / '+scenario)
                            if scenario != 'native': ax.vlines(part.index,part['min'],part['max'],color=color,linewidth=.7)
                    ax.axhline(0,color='black',linewidth=.6); ax.set_xticks([7,10,14]); ax.set_xlabel('Culture day')
                    ax.set_ylabel('WITH_SERUM minus WO_SERUM\nMedian normalized usage')
                    ax.set_title(program+' paired serum contrasts: '+gate+'\nThree donors; thinning ranges are technical, not confidence intervals')
                    ax.legend(loc='upper left',bbox_to_anchor=(1.02,1),frameon=False)
                    figure(fig,source,serum_prefix+'_'+program+'_'+gate,
                        'Three biological donors. Native contrasts shown individually; half-depth and matched-depth lines average three technical replicates per donor, with min-max ranges. No hypothesis test. Positive values indicate higher relative usage with serum, not higher absolute function. Both fixed gate conditions are reported.')
            source = frames['quality'].loc[frames['quality'].scenario.eq('native')].copy()
            fig,ax=plt.subplots(figsize=(180/25.4,85/25.4));fig.subplots_adjust(left=.13,right=.80,bottom=.24,top=.83)
            # Chronological order, exact condition strings with a display-only line break.
            order=sorted(source.condition.unique(), key=lambda s:(int(s.split('_')[0][1:]), s))
            for donor, part in source.groupby('donor'):
                part=part.set_index('condition').reindex(order)
                ax.plot(range(len(order)),part.fraction_historical_q95_accepted,marker='o',linewidth=.8,markersize=3,label=donor)
            ax.set_xticks(range(len(order)),[s.replace('_','_\n',1) for s in order]);ax.set_ylim(0,1.05)
            ax.set_ylabel('Fraction of valid cells retained');ax.set_title('Historical q95 retention in native RNA\nSelection changes the analyzed cell population')
            ax.legend(loc='upper left',bbox_to_anchor=(1.02,1),frameon=False)
            figure(fig,source,serum_prefix+'_q95_retention','Native data only; three donor trajectories. Numerator: historical q95 accepted cells. Denominator: valid converged scores. No error bars or population inference; exclusions are not evidence of biologically invalid cells.')
        table(pd.DataFrame(legends), prefix + '_figure_legends')
        unit('figure_source_data', {'figures':['figure']}, lambda:{'figures':pd.DataFrame(legends)})
        if len(progress['completed_units']) != 11: raise ValueError('Incomplete evidence assembly')
        if any(sha256sum(PROJECT / rel)!=h for rel,h in inputs.items()): raise ValueError('Input changed during assembly')
        table(pd.DataFrame([{'path':k,'sha256':v} for k,v in inputs.items()]),prefix+'_input_inventory')
        progress.update(status='completed', outputs={str(p.relative_to(PROJECT)):sha256sum(p) for p in outputs},
            conclusions={'program_cards':8,'end_to_end_validation_completed':False,
                         'serum_functional_validation_completed':False,'deployment_changed':False},
            limitations=['Retrospective synthesis, not new independent validation','Technical calibration does not transfer automatically to external cohorts',
                         'End-to-end inventory limited to recorded project prediction paths','Historical model-card readiness statements are not current audit conclusions'])
        flush()
        display(pd.concat(cards,ignore_index=True)[['program','K8_top10_genes','fold_to_full_cosine_min','quarter_UMI_bound_exceedance_max','when_to_use_caution']])
        display(af['cohort_readiness']); display(frames['donor_summary'].head(12))
        print('Evidence synthesis completed. End-to-end validation remains incomplete; no model or threshold changed.')
    except Exception as exc:
        progress.update(status='failed', error=str(exc));flush();raise


if EXECUTION_STAGE == 'evidence_synthesis':
    run_evidence_synthesis()
'''


END_TO_END_HELPERS = r'''
def _e2e_verify_evidence(manifest, evidence_hash, prediction_path, receptor_paths, rna_path, current_hashes):
    if not manifest.get('complete') or manifest.get('cell_output_sha256')!=evidence_hash:
        raise ValueError('Incomplete or changed original receptor audit')
    original=manifest.get('contract',{}).get('inputs',{})
    for path in [str(prediction_path)]+[str(p) for p in receptor_paths]:
        if path not in original or path not in current_hashes or original[path]!=current_hashes[path]:
            raise ValueError('Receptor audit input provenance mismatch: '+path)
    path=str(rna_path)
    if path not in current_hashes or manifest.get('raw_inputs',{}).get(path)!=current_hashes[path]:
        raise ValueError('Receptor audit RNA provenance mismatch: '+path)


def _e2e_prepare(predictions, evidence):
    required = ['cell_id','Series','Sample','hard_call']
    for d in [predictions,evidence]:
        if not set(required) <= set(d) or d.cell_id.isna().any() or d.cell_id.duplicated().any():
            raise ValueError('Incomplete or duplicate prediction/evidence identities')
    if set(predictions.cell_id) != set(evidence.cell_id):
        raise ValueError('Prediction and evidence populations differ')
    p=predictions.copy().set_index('cell_id',drop=False)
    e=evidence.set_index('cell_id').loc[p.index]
    for col in required[1:]:
        if not p[col].astype(str).equals(e[col].astype(str)):
            raise ValueError('Prediction/evidence metadata mismatch: '+col)
    if p[['Series','Sample']].isna().any().any():raise ValueError('Missing study/sample proxy')
    p['receptor_evidence']=e.receptor_evidence.fillna('not linked; unknown')
    p['is_receptor_positive']=p.receptor_evidence.isin([
        'strict_1TRG_1TRD','TRD + TRG with CDR3; QC flags unavailable'])
    p['is_receptor_negative']=p.receptor_evidence.eq('strict_paired_abTCR')
    p['is_silver_gdT']=p.annotation_Level2.isin(['T_gdT_TRDV1','T_gdT_TRDV2'])
    p['in_primary_analysis']=p.Series.ne('GSE149356')
    p['include_for_state']=(p.hard_call.eq('predicted gdT')|p.is_silver_gdT|
                            p.is_receptor_positive|p.is_receptor_negative)
    p['subtype_prediction']=pd.NA
    p['subtype_prediction_status']='not_available; no second-layer model selected in this stage'
    return p


def _e2e_route(frame, programs, q95):
    out=frame.copy(); x=out[programs].to_numpy(float)
    finite=np.isfinite(x).all(axis=1)
    if (x[finite]<0).any() or not np.allclose(x[finite].sum(axis=1),1,atol=1e-7):
        raise ValueError('Finite usages must be nonnegative and sum to one')
    out['state_score_valid']=(finite & out.valid_objective.astype(bool) &
        out.solver_converged.astype(bool) & np.isfinite(out.scaled_HVG_relative_error))
    out['within_Gray_q95']=out.state_score_valid & out.scaled_HVG_relative_error.le(q95)
    # Subtype fields are deliberately absent from these routing predicates.
    out['state_output_available']=out.hard_call.eq('predicted gdT') & out.state_score_valid
    out['state_output_q95_available']=out.state_output_available & out.within_Gray_q95
    out['state_output_status']=np.select([
        ~out.hard_call.eq('predicted gdT'),~out.state_score_valid,~out.within_Gray_q95],
        ['reference_only; not lineage-positive','invalid_numerical_score','available_with_reconstruction_warning'],
        default='available_within_Gray_q95')
    return out


def _e2e_contrasts(frame, programs):
    frame=frame.copy()
    # Preserve exact labels without categorical-code overflow in empty arms.
    for key in ['Series','Sample']:frame[key]=frame[key].astype(str)
    rows=[]
    for primary in [True,False]:
        base=frame.loc[frame.in_primary_analysis.eq(primary)]
        scope='excluding GSE149356' if primary else 'GSE149356 previously used sensitivity'
        for gate in ['all_valid','Gray_q95_accepted']:
            use=base.loc[base.state_score_valid if gate=='all_valid' else base.within_Gray_q95]
            pred=use.hard_call.eq('predicted gdT')
            comparisons=[('predicted gdT minus PANGEA gdT annotation',pred,use.is_silver_gdT),
                ('predicted gdT minus receptor-supported gdT',pred,use.is_receptor_positive),
                ('predicted gdT minus predicted gdT excluding strict paired abTCR',pred,pred & ~use.is_receptor_negative)]
            for comparison,left,right in comparisons:
                arms=[]
                for mask in [left,right]:
                    a=use.loc[mask].groupby(['Series','Sample'],observed=True)[programs].median()
                    a['n']=use.loc[mask].groupby(['Series','Sample'],observed=True).size();arms.append(a)
                units=base[['Series','Sample']].drop_duplicates()
                for unit in units.itertuples(index=False):
                    key=(unit.Series,unit.Sample)
                    values=[a.loc[key] if key in a.index else None for a in arms]
                    for program in programs:
                        l=float(values[0][program]) if values[0] is not None else np.nan
                        r=float(values[1][program]) if values[1] is not None else np.nan
                        rows.append({'scope':scope,'quality_gate':gate,'comparison':comparison,
                            'Series':key[0],'Sample':key[1],'program':program,
                            'n_predicted':int(values[0]['n']) if values[0] is not None else 0,
                            'n_reference':int(values[1]['n']) if values[1] is not None else 0,
                            'predicted_sample_median':l,'reference_sample_median':r,'difference':l-r,
                            'interpretation':'Selection-associated difference; not a causal or complete contamination estimate'})
    samples=pd.DataFrame(rows)
    keys=['scope','quality_gate','comparison','Series','program']
    studies=samples.groupby(keys,observed=True,dropna=False).agg(
        n_paired_samples=('difference','count'),n_sample_proxies=('difference','size'),
        mean_paired_sample_difference=('difference','mean'),
        min_sample_difference=('difference','min'),max_sample_difference=('difference','max')).reset_index()
    return samples,studies
'''


END_TO_END_CODE = r'''
def run_end_to_end_validation():
    prefix='PANGEA_primary_K8_end_to_end_validation'
    inputs={}
    def register(path):
        path=Path(path);inputs[str(path)]=sha256sum(path);return path
    pred_path=register(PROJECT/'predictions/22_LR_hierarchical_CE/PANGEA_LR_hierarchical_predictions.parquet')
    pred_manifest=json.loads(register(pred_path.with_suffix('.manifest.json')).read_text())
    if not pred_manifest.get('complete') or sha256sum(pred_path)!=pred_manifest['prediction_sha256']:
        raise ValueError('Incomplete or modified lineage prediction release')
    predictions=pd.read_parquet(pred_path)
    evidence_path=register(PROJECT/'predictions/22_LR_hierarchical_CE/gate_discordance_audit_cells.parquet')
    evidence=pd.read_parquet(evidence_path);evidence=evidence.loc[evidence.dataset.eq('PANGEA')].copy()
    evidence_manifest=json.loads(register(PROJECT/'checkpoints/22_LR_hierarchical_CE/gate_discordance_audit_manifest.json').read_text())
    # Register the independently linked receptor sources, not only derived labels.
    receptor_paths=sorted(set(evidence.receptor_evidence_source.dropna().astype(str))-{''})
    for path in receptor_paths:register(path)
    metadata=_e2e_prepare(predictions,evidence)
    previous=json.loads(register(TABLE_DIR/'PANGEA_primary_K8_projection_manifest.json').read_text())
    old_path=register(Path(previous['usage_path']))
    if sha256sum(old_path)!=previous['usage_sha256']:raise ValueError('Historical PANGEA score hash mismatch')
    old=pd.read_parquet(old_path)
    pangea_path=register(Path(previous['PANGEA_path']))
    if inputs[str(pangea_path)]!=previous['PANGEA_sha256']:raise ValueError('PANGEA input changed')
    _e2e_verify_evidence(evidence_manifest,inputs[str(evidence_path)],pred_path,receptor_paths,pangea_path,inputs)
    if any(v['input_sha256']!=inputs[str(pangea_path)] for v in pred_manifest['units'].values()):
        raise ValueError('Lineage and state source versions differ')
    scorer=load_frozen_scorer_for_gse268991()
    frozen={k:sha256sum(register(scorer[k])) for k in ['spec_path','spectra_path','sd_path','params_path','ranked_path']}
    q95=float(previous['Gray_q95_relative_error_gate']); mapping=scorer['component_to_program']
    programs=list(mapping.values());target=metadata.loc[metadata.include_for_state].copy()
    target['previous_state_score_available']=target.index.isin(old.index.astype(str))
    data=ad.read_h5ad(pangea_path,backed='r')
    try:
        if not data.obs_names.is_unique or not data.var_names.is_unique or 'count' not in data.layers:
            raise ValueError('PANGEA identities/features/raw-count layer invalid')
        positions=data.obs_names.get_indexer(target.index)
        if (positions<0).any():raise ValueError('Missing RNA barcodes for selected cells')
        for col in ['Series','Sample','annotation_Level2']:
            observed=data.obs.iloc[positions][col].astype(str).to_numpy()
            if not np.array_equal(observed,target[col].astype(str).to_numpy()):raise ValueError('RNA metadata mismatch: '+col)
        genes=pd.Index(data.var_names.astype(str));coverage=float(scorer['hvgs'].isin(genes).mean())
        if coverage<MIN_HVG_COVERAGE:raise ValueError('Frozen HVG coverage below approved minimum')
        dest=scorer['universe'].get_indexer(genes);keep=dest>=0
        collapse=sparse.csr_matrix((np.ones(keep.sum()),(np.flatnonzero(keep),dest[keep])),shape=(len(genes),len(scorer['universe'])))
        series=sorted(target.Series.astype(str).unique())
        config={'stage':'end_to_end_validation','source_sha256':sha256sum(PROJECT/'build_19b_cNMF_valdiation.py'),
            'inputs':inputs,'frozen_model':frozen,'lineage_release':'22_LR_hierarchical_CE',
            'lineage_model_hashes':pred_manifest['source_hash'],'cell_ids_sha256':sha256_lines(target.index),
            'program_mapping':mapping,'q95':q95,'seed':SEED,'threads':THREADS,'HVG_coverage':coverage,
            'primary_exclusion':['GSE149356'],'state_population':'predicted gdT OR silver gdT OR linked receptor positive/negative',
            'subtype_prediction':'not_available; no second-layer prediction inferred from reference annotations',
            'versions':{k:im.version(k) for k in ['numpy','pandas','scipy','scikit-learn','anndata']},'schema_version':1}
        run_id=stable_token(config);root=CHECKPOINT_DIR/'end_to_end_validation'/run_id;root.mkdir(parents=True,exist_ok=True)
        atomic_json(root/'job_spec.json',config)
        _ur_snapshot(PROJECT/'build_19b_cNMF_valdiation.py',root/'build_19b_cNMF_valdiation.py')
        manifest={'status':'running','run_id':run_id,'configuration':config,'checkpoint_root':str(root),
            'expected_units':len(series),'completed_units':[]}
        def flush():
            manifest['updated_UTC']=datetime.now(timezone.utc).isoformat()
            atomic_json(TABLE_DIR/(prefix+'_manifest.json'),manifest)
        flush(); frames=[];rows_lookup=pd.Series(positions,index=target.index)
        try:
            layer=data.layers['count']
            for study in series:
                if not re.fullmatch(r'[A-Za-z0-9_.-]+',study):raise ValueError('Unsafe Series checkpoint name')
                part=target.loc[target.Series.astype(str).eq(study)].copy()
                spec={'run_id':run_id,'Series':study,'cell_ids_sha256':sha256_lines(part.index)}
                def produce():
                    raw=sparse.csr_matrix(layer[rows_lookup.loc[part.index].to_numpy(int),:],dtype=np.float64)
                    if not np.isfinite(raw.data).all() or (raw.data<0).any() or not np.allclose(raw.data,np.rint(raw.data)):
                        raise ValueError('PANGEA count layer is not finite nonnegative raw UMI')
                    counts=(raw@collapse).tocsr();coef,usage,quality=_project_validation_counts(counts,scorer,SEED,THREADS)
                    quality.index=part.index
                    u=pd.DataFrame(usage,index=part.index,columns=programs)
                    joined=pd.concat([part,u,quality],axis=1)
                    routed=_e2e_route(joined,programs,q95)
                    return {'cells':routed,'usage_raw':pd.DataFrame(coef,index=part.index,columns=programs)}
                got,reused=_iv_checkpoint(root/study,stable_token(spec),
                    {'cells':['cell_id','state_output_available','subtype_prediction_status']+programs,'usage_raw':programs},produce)
                if list(got['cells'].index)!=list(part.index):raise ValueError('Checkpoint cell order mismatch')
                frames.append(got['cells']);manifest['completed_units'].append({'Series':study,'n_cells':len(part),'reused':reused});flush()
                print('end_to_end_validation',study,len(part),'reused' if reused else 'completed',flush=True)
            scored=pd.concat(frames).loc[target.index]
            if len(manifest['completed_units'])!=len(series) or set(scored.index)!=set(target.index):raise ValueError('Incomplete cohort scoring')
            samples,studies=_e2e_contrasts(scored,programs)
            outputs=[]
            def table(frame,suffix):
                name=prefix+'_'+suffix;save_table(frame,name);outputs.append(TABLE_DIR/(name+'.csv'))
            cells_path=atomic_parquet(scored,MODEL_DIR/(prefix+'_cells.parquet'));outputs.append(cells_path)
            table(samples,'sample_contrasts');table(studies,'study_contrasts')
            table(target[['cell_id','Series','Sample','hard_call','in_primary_analysis','is_silver_gdT','is_receptor_positive','is_receptor_negative','previous_state_score_available']], 'cohort_inventory')
            coverage_rows=[]
            for primary in [True,False]:
                full=metadata.loc[metadata.in_primary_analysis.eq(primary)];s=scored.loc[scored.in_primary_analysis.eq(primary)]
                positive=full.hard_call.eq('predicted gdT');sp=s.hard_call.eq('predicted gdT')
                coverage_rows.append({'scope':'excluding GSE149356' if primary else 'GSE149356 previously used sensitivity',
                    'n_lineage_evaluated':len(full),'n_predicted_gdT':int(positive.sum()),'n_scored_union':len(s),
                    'n_predicted_gdT_scoring_attempted':int(sp.sum()),
                    'n_predicted_gdT_previously_scored':int((sp&s.previous_state_score_available).sum()),
                    'n_predicted_gdT_valid':int(s.state_output_available.sum()),'n_predicted_gdT_q95':int(s.state_output_q95_available.sum()),
                    'n_receptor_positive':int(full.is_receptor_positive.sum()),
                    'n_receptor_positive_predicted_gdT':int((positive&full.is_receptor_positive).sum()),
                    'n_strict_paired_abTCR':int(full.is_receptor_negative.sum()),
                    'n_strict_paired_abTCR_predicted_gdT':int((positive&full.is_receptor_negative).sum()),
                    'HVG_coverage':coverage,'q95':q95})
            table(pd.DataFrame(coverage_rows),'coverage')
            evidence_counts=metadata.groupby(['Series','Sample','receptor_evidence','hard_call'],observed=True).size().rename('n_cells').reset_index()
            table(evidence_counts,'receptor_evidence_counts')
            routing=scored.groupby(['in_primary_analysis','hard_call','subtype_prediction_status','state_output_status'],observed=True,dropna=False).size().rename('n_cells').reset_index()
            table(routing,'routing')
            retention=scored.groupby(['Series','Sample','hard_call'],observed=True).agg(
                n_scored=('cell_id','size'),n_valid=('state_score_valid','sum'),n_q95=('within_Gray_q95','sum'),
                median_error=('scaled_HVG_relative_error','median')).reset_index();table(retention,'quality_retention')
            overlap=scored.index.intersection(old.index.astype(str));old.columns=old.columns.map(lambda c:mapping.get(str(c),str(c)))
            delta=scored.loc[overlap,programs]-old.loc[overlap,programs]
            table(pd.DataFrame({'program':programs,'n_overlap':len(overlap),
                'mean_absolute_difference':delta.abs().mean().reindex(programs).to_numpy(),
                'max_absolute_difference':delta.abs().max().reindex(programs).to_numpy(),
                'scope':'Same frozen basis; larger scoring populations may affect numerical stopping; no historical score overwritten'}),'legacy_overlap')
            legends=[]
            with mpl.rc_context({'font.family':FIGURE_FONT,'font.size':7,'axes.titlesize':7,'axes.labelsize':7,
                'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'pdf.fonttype':42}):
                for gate in ['all_valid','Gray_q95_accepted']:
                    for sensitivity in [False,True]:
                        if sensitivity:
                            chart=samples.loc[samples.scope.eq('GSE149356 previously used sensitivity') & samples.quality_gate.eq(gate)
                                & samples.comparison.eq('predicted gdT minus receptor-supported gdT')].copy()
                            value='difference';unit='Sample';title='GSE149356 sensitivity: prediction versus processed TCR support';stem='GSE149356_'+gate
                        else:
                            chart=studies.loc[studies.scope.eq('excluding GSE149356') & studies.quality_gate.eq(gate)
                                & studies.comparison.eq('predicted gdT minus PANGEA gdT annotation')].copy()
                            value='mean_paired_sample_difference';unit='Series';title='PANGEA: prediction versus silver annotation; excluding GSE149356';stem='silver_'+gate
                        fig,ax=plt.subplots(figsize=(180/25.4,85/25.4));fig.subplots_adjust(left=.14,right=.98,bottom=.21,top=.82)
                        for i,program in enumerate(programs):
                            part=chart.loc[chart.program.eq(program)&chart[value].notna()].sort_values(unit);v=part[value].to_numpy()
                            offsets=np.linspace(-.15,.15,len(v)) if len(v)>1 else np.zeros(len(v))
                            ax.scatter(i+offsets,v,s=10,color='#0072B2',alpha=.65,linewidths=0)
                            if len(v):ax.plot([i-.2,i+.2],[v.mean(),v.mean()],color='black',lw=1.2)
                        ax.axhline(0,color='black',lw=.6);ax.set_xticks(range(len(programs)),programs)
                        ax.set_ylabel('Predicted minus reference normalized usage');ax.set_xlabel('Frozen K8 program')
                        ax.set_title(title+'\n'+gate+'; points: '+unit+' effects; black marks: equal-unit mean')
                        ax.spines[['top','right']].set_visible(False)
                        for tx in fig.findobj(mpl.text.Text):tx.set_color('black');tx.set_fontweight('normal')
                        name=prefix+'_'+stem
                        with mpl.rc_context({'savefig.bbox':None}):
                            fig.savefig(FIGURE_DIR/(name+'.pdf'),bbox_inches=None);fig.savefig(FIGURE_DIR/(name+'.png'),bbox_inches=None,dpi=300)
                        outputs.append(FIGURE_DIR/(name+'.pdf'));display(fig);plt.close(fig);table(chart,stem+'_source_data')
                        legends.append({'figure':name,'legend':'Individual observed unit effects; no hypothesis test or confidence interval. Common sample support only; absent arms remain missing in source data. Prediction-versus-reference differences are selection-associated, not causal or complete contamination estimates. GSE149356 was previously examined and processed TCR lacks full QC flags. q95 is not program-level confidence.','QA':'Rendered text/collision and visual checks required after execution'})
            table(pd.DataFrame(legends),'figure_legends')
            if any(sha256sum(p)!=h for p,h in inputs.items()):raise ValueError('Input changed during execution')
            manifest.update(status='completed',outputs={str(p.relative_to(PROJECT)):sha256sum(p) for p in outputs},
                n_scored=len(scored),limitations=['PANGEA silver labels are not independent lineage truth',
                'Unknown receptors stay unknown; observed strict-ab contamination is only a lower bound',
                'GSE149356 is previously examined sensitivity, not new external validation',
                'No verified biological donor IDs for primary PANGEA; use study/sample proxies',
                'Second-layer model not selected/applied; routing independence tested, subtype accuracy not evaluated',
                'No new serum/time-course, functional validation or model tuning'])
            flush();display(pd.DataFrame(coverage_rows));display(routing)
            print('Completed bounded lineage-to-state evaluation; full independently labelled three-layer validation is not claimed.')
        except Exception as exc:
            manifest.update(status='failed',error=str(exc));flush();raise
    finally:data.file.close()


if EXECUTION_STAGE == 'end_to_end_validation':
    if not (TABLE_DIR/'PANGEA_primary_K8_end_to_end_validation_manifest.json').exists():
        run_end_to_end_validation()
    else:
        print('Preserving saved base run; the follow-up verifies its inputs, outputs and source snapshot before reuse.')
'''


END_TO_END_SUMMARY_HELPERS = r'''
def _e2s_prepare(predictions,evidence,scored,followup,lower,upper):
    full=_e2e_prepare(predictions,evidence)
    for data in [scored,followup]:
        if data.cell_id.isna().any() or data.cell_id.duplicated().any():raise ValueError('Duplicate or missing closure cell identity')
    score=scored.set_index('cell_id',drop=False);follow=followup.set_index('cell_id',drop=False)
    if set(score.index)!=set(full.index[full.include_for_state]):raise ValueError('Incomplete scored reference/selected population')
    if set(follow.index)!=set(score.index[score.Series.eq('GSE149356')]):raise ValueError('Incomplete GSE149356 follow-up population')
    for data in [score,follow]:
        for col in ['Series','Sample','hard_call']:
            if not np.array_equal(data[col].astype(str),full.loc[data.index,col].astype(str)):
                raise ValueError('Closure metadata mismatch: '+col)
    if not np.array_equal(full.silver_y.astype(int),full.is_silver_gdT.astype(int)):
        raise ValueError('Silver-label definition changed')
    full['selected']=full.hard_call.eq('predicted gdT')
    if not full.hard_call.isin(['predicted gdT','predicted abT','predicted NK','abstain/out-of-domain']).all():
        raise ValueError('Unrecognized lineage output')
    full['state_scored']=full.index.isin(score.index)
    for col in ['state_score_valid','within_Gray_q95','state_output_available','state_output_q95_available']:
        full[col]=full.index.to_series().map(score[col]).fillna(False).astype(bool)
        if not np.array_equal(follow[col].astype(bool),full.loc[follow.index,col]):raise ValueError('State routing changed in follow-up')
    if not full.state_output_available.eq(full.selected & full.state_score_valid).all():raise ValueError('Invalid state output dependency')
    if not full.state_output_q95_available.eq(full.state_output_available & full.within_Gray_q95).all():raise ValueError('Invalid q95 routing')
    if (full.selected & ~full.state_scored).any():raise ValueError('Unscored selected cells')
    for col in ['p_calibrated','expression_subtype','sequence_subtype']:
        full[col]=full.index.to_series().map(follow[col])
    full['subtype_available']=full.p_calibrated.notna()
    q=full.p_calibrated
    if not q.dropna().between(0,1).all():raise ValueError('Invalid subtype probability')
    expected=pd.Series(np.select([q.ge(upper),q.le(lower)],['Vd1','Vd2'],default='other_or_unresolved'),index=full.index)
    if not full.loc[full.subtype_available,'expression_subtype'].eq(expected[full.subtype_available]).all():
        raise ValueError('Subtype labels do not match frozen policy')
    full['subtype_assigned']=full.subtype_available & full.expression_subtype.isin(['Vd1','Vd2'])
    full['binary_reference']=full.is_receptor_positive & full.sequence_subtype.isin(['Vd1','Vd2'])
    full['subtype_correct']=full.binary_reference & full.subtype_assigned & full.expression_subtype.eq(full.sequence_subtype)
    for c in ['Series','Sample']:full[c]=full[c].astype(str)
    return full.sort_index()


def _e2s_tables(frame,aggregation):
    if not len(frame):raise ValueError('Empty closure population')
    if aggregation not in ['pooled','Series','Sample']:raise ValueError('Invalid aggregation')
    meta={'aggregation':aggregation,'Series':frame.Series.iloc[0] if aggregation!='pooled' else None,
          'Sample':frame.Sample.iloc[0] if aggregation=='Sample' else None}
    selected=frame.selected;positive=frame.is_receptor_positive;negative=frame.is_receptor_negative
    binary=frame.binary_reference;available=frame.subtype_available;assigned=frame.subtype_assigned;correct=frame.subtype_correct
    count=lambda mask:int(mask.sum())
    row={**meta,'n_input':len(frame),'n_studies':frame.Series.nunique(),
         'n_sample_proxies':len(frame[['Series','Sample']].drop_duplicates()),
         'n_selected':count(selected),'n_predicted_abT':count(frame.hard_call.eq('predicted abT')),
         'n_predicted_NK':count(frame.hard_call.eq('predicted NK')),'n_lineage_abstain':count(frame.hard_call.eq('abstain/out-of-domain')),
         'n_receptor_positive':count(positive),'n_receptor_negative':count(negative),'n_receptor_unknown':count(~(positive|negative)),
         'n_receptor_positive_missed':count(positive & ~selected),'n_receptor_negative_selected':count(negative & selected),
         'n_state_scored':count(frame.state_scored),'n_state_output':count(frame.state_output_available),
         'n_state_q95':count(frame.state_output_q95_available),'n_state_warning':count(frame.state_output_available & ~frame.within_Gray_q95),
         'n_state_unavailable_selected':count(selected & ~frame.state_output_available),
         'n_subtype_unavailable_selected':count(selected & ~available),
         'n_subtype_unresolved_selected':count(selected & available & ~assigned),
         'n_Vd1_selected':count(selected & assigned & frame.expression_subtype.eq('Vd1')),
         'n_Vd2_selected':count(selected & assigned & frame.expression_subtype.eq('Vd2')),
         'n_binary_reference':count(binary),'n_binary_reference_missed':count(binary & ~selected),
         'n_binary_reference_model_unavailable':count(binary & selected & ~available),
         'n_binary_reference_unresolved':count(binary & selected & available & ~assigned),
         'n_binary_reference_wrong':count(binary & selected & assigned & ~correct),
         'n_binary_reference_correct':count(binary & selected & correct)}
    assert row['n_input']==sum(row[k] for k in ['n_selected','n_predicted_abT','n_predicted_NK','n_lineage_abstain'])
    assert row['n_input']==row['n_receptor_positive']+row['n_receptor_negative']+row['n_receptor_unknown']
    assert row['n_selected']==sum(row[k] for k in ['n_subtype_unavailable_selected','n_subtype_unresolved_selected','n_Vd1_selected','n_Vd2_selected'])
    assert row['n_selected']==row['n_state_output']+row['n_state_unavailable_selected']
    assert row['n_state_output']==row['n_state_q95']+row['n_state_warning']
    assert row['n_binary_reference']==sum(row[k] for k in ['n_binary_reference_missed','n_binary_reference_model_unavailable','n_binary_reference_unresolved','n_binary_reference_wrong','n_binary_reference_correct'])
    rows=[]
    def add(reference,metric,numerator,denominator,note,estimable=True):
        n=int(numerator);d=int(denominator)
        if n>d or n<0 or d<0:raise ValueError('Invalid metric denominator')
        rows.append({**meta,'reference_set':reference,'metric':metric,'numerator':n,'denominator':d,
                     'value':n/d if d and estimable else np.nan,
                     'status':'descriptive' if d and estimable else 'not_estimable','interpretation':note})
    add('all','state_output_coverage',count(frame.state_output_available),count(selected),'Availability among lineage-positive cells; not state accuracy')
    add('all','state_q95_coverage',count(frame.state_output_q95_available),count(selected),'Fixed reconstruction gate; not 95% biological accuracy')
    add('all','subtype_assignment_coverage',count(selected & assigned),count(selected),'Includes missing subtype models and abstentions in denominator')
    add('receptor_supported','lineage_recall',count(positive & selected),count(positive),'Processed receptor-positive reference; unknowns are not negatives')
    add('strict_paired_abTCR','lineage_false_positive_rate',count(negative & selected),count(negative),'Strict receptor-negative reference; report study context')
    jointly_observed=all(g.is_receptor_positive.any() and g.is_receptor_negative.any() for _,g in frame.loc[positive|negative].groupby('Series',observed=True)) and (positive|negative).any()
    add('receptor_supported','lineage_precision',count(positive & selected),count((positive|negative)&selected),
        'Restricted labeled subset only; not estimable when reference classes are not jointly observed within study',jointly_observed)
    silver=frame.silver_evaluation_eligible.astype(bool);sy=frame.silver_y.eq(1)
    add('silver_evaluation_eligible','lineage_precision',count(silver & sy & selected),count(silver & selected),'Annotation concordance only; not independent lineage truth')
    add('silver_evaluation_eligible','lineage_recall',count(silver & sy & selected),count(silver & sy),'Annotation concordance only; excludes ineligible annotations')
    add('silver_evaluation_eligible','lineage_false_positive_rate',count(silver & ~sy & selected),count(silver & ~sy),'Annotation concordance only; NK/ILC/other ineligible cells are not reassigned negatives')
    ref='paired_TRG_TRD_primary'
    add(ref,'prediction_availability',count(binary & available),count(binary),'Standalone subtype model availability among binary references, before lineage selection')
    add(ref,'assignment_coverage',count(binary & assigned),count(binary & available),'Standalone subtype assignment conditional on available probabilities')
    add(ref,'assigned_accuracy',count(correct),count(binary & assigned),'Standalone expression-vs-sequence agreement; excludes unresolved and unavailable')
    add(ref,'end_to_end_assignment_coverage',count(binary & selected & assigned),count(binary),'Full original binary-reference denominator, including lineage misses and unavailable subtype models')
    add(ref,'end_to_end_assigned_accuracy',count(correct & selected),count(binary & selected & assigned),'Accuracy conditional on emitted subtype; report full-denominator coverage alongside')
    add(ref,'end_to_end_correct_subtype_recall',count(correct & selected),count(binary),'Correct emitted subtype per original binary reference; unavailable outputs remain in denominator')
    add(ref,'end_to_end_correct_subtype_with_state',count(correct & selected & frame.state_output_available),count(binary),'Correct emitted subtype with available state; does not establish correctness of state score')
    add(ref,'end_to_end_correct_subtype_with_q95',count(correct & selected & frame.state_output_q95_available),count(binary),'Correct emitted subtype with fixed reconstruction pass; not functional validation')
    return pd.DataFrame([row]),pd.DataFrame(rows)
'''


END_TO_END_SUMMARY_CODE = r'''
def run_end_to_end_summary():
    prefix='PANGEA_primary_K8_end_to_end_validation';inputs={}
    def register(path,digest=None):
        path=Path(path);actual=sha256sum(path)
        if digest is not None and actual!=digest:raise ValueError('Closure provenance mismatch: '+str(path))
        inputs[str(path)]=actual;return path
    follow_path=TABLE_DIR/(prefix+'_followup_manifest.json');follow_digest=sha256sum(follow_path);follow=json.loads(follow_path.read_text())
    if follow.get('status')!='completed' or (follow['configuration']['lower'],follow['configuration']['upper'])!=(.3,.7):
        raise ValueError('Complete approved 0.30/0.70 follow-up required')
    for p,h in follow['configuration']['inputs'].items():register(p,h)
    for p,h in follow['outputs'].items():
        if sha256sum(PROJECT/p)!=h:raise ValueError('Follow-up output changed: '+p)
    # Check all parent outputs, but only data dependencies enter the run key.
    # Re-rendered PDF metadata and the parent completion timestamp are not inputs.
    register(TABLE_DIR/(prefix+'_subtype_routing.csv'),follow['outputs'][str((TABLE_DIR/(prefix+'_subtype_routing.csv')).relative_to(PROJECT))])
    register(Path(follow['checkpoint_root'])/'build_19b_cNMF_valdiation.py',follow['configuration']['source_sha256'])
    base=json.loads((TABLE_DIR/(prefix+'_manifest.json')).read_text())
    prediction_path=PROJECT/'predictions/22_LR_hierarchical_CE/PANGEA_LR_hierarchical_predictions.parquet'
    evidence_path=PROJECT/'predictions/22_LR_hierarchical_CE/gate_discordance_audit_cells.parquet'
    predictions=pd.read_parquet(prediction_path);evidence=pd.read_parquet(evidence_path)
    evidence=evidence.loc[evidence.dataset.eq('PANGEA')].copy()
    scored=pd.read_parquet(PROJECT/'results/models/19b_cNMF_valdiation'/(prefix+'_cells.parquet'))
    routed=pd.read_csv(TABLE_DIR/(prefix+'_subtype_routing.csv'))
    full=_e2s_prepare(predictions,evidence,scored,routed,follow['configuration']['lower'],follow['configuration']['upper'])
    config={'stage':'end_to_end_validation','source_sha256':sha256sum(PROJECT/'build_19b_cNMF_valdiation.py'),
        'parent_run_id':follow['run_id'],'inputs':inputs,'lower':.3,'upper':.7,'q95':follow['configuration']['q95'],
        'cell_ids_sha256':sha256_lines(full.index),'schema_version':1,'versions':{k:im.version(k) for k in ['numpy','pandas']}}
    run_id=stable_token(config);root=CHECKPOINT_DIR/'end_to_end_validation'/run_id;root.mkdir(parents=True,exist_ok=True)
    _ur_snapshot(PROJECT/'build_19b_cNMF_valdiation.py',root/'build_19b_cNMF_valdiation.py');atomic_json(root/'job_spec.json',config)
    manifest={'status':'running','run_id':run_id,'configuration':config,'checkpoint_root':str(root),'parent_manifest_sha256':follow_digest,
        'expected_units':full.Series.nunique(),'completed_units':[],
        'models':{'lineage':base['configuration']['lineage_release'],'lineage_hashes':base['configuration']['lineage_model_hashes'],
                  'subtype':follow['configuration']['subtype_release'],'primary_seed':follow['configuration']['primary_seed'],
                  'state':'primary_K8','state_artifact_hashes':base['configuration']['frozen_model'],'program_mapping':base['configuration']['program_mapping']},
        'evaluation_scope':'Existing matched PANGEA T/NK evaluation population; not newly processed whole atlas; development evaluation',
        'input_contract':{'RNA':'Original count layer with exact gene and cell identities; reuse hash-verified frozen outputs',
                          'state_minimum_HVG_coverage':MIN_HVG_COVERAGE,'observed_state_HVG_coverage':base['configuration']['HVG_coverage'],
                          'subtype_policy':'P(Vd1)>=0.70 Vd1; <=0.30 Vd2; intermediate unresolved; no model means unavailable'},
        'output_contract':{'state':'Eight continuous normalized usages; quality warning independent of subtype certainty',
                           'reference':'Unknown receptor labels remain unknown; sequence-assisted outputs do not count as expression prediction success'},
        'limitations':['PANGEA silver labels are annotation-derived, not independent truth',
            'Paired positive and strict negative receptor classes may arise in different studies; pooled independent-label precision is not then estimable',
            'Sample is a proxy, not automatically a biological donor; pooled cell fractions are descriptive without cell-independence confidence intervals',
            'GSE149356 informed the 0.30/0.70 choice; not untouched evaluation',
            'CB1/CB2 and other studies lack approved subtype outputs; structural unavailability remains in system-coverage denominators',
            'State availability and q95 are not biological correctness or perturbation/functional prediction',
            'Serum/time-course is a separate existing application; this run does not establish a new full-pipeline serum validation'],
        'software_release_status':'Research workflow only; no public package created. Clean-environment inference parity, documented feature transforms, licensing, versioned model distribution and new-donor evaluation remain release requirements'}
    def flush():atomic_json(TABLE_DIR/(prefix+'_model_contract.json'),manifest)
    flush();flows=[];metrics=[]
    try:
        for study,part in full.groupby('Series',sort=True,observed=True):
            if not re.fullmatch(r'[A-Za-z0-9_.-]+',study):raise ValueError('Unsafe study key')
            def produce():
                f,m=_e2s_tables(part,'Series');fs=[f];ms=[m]
                for sample,sub in part.groupby('Sample',sort=True,observed=True):
                    f,m=_e2s_tables(sub,'Sample');fs.append(f);ms.append(m)
                return {'cell_flow':pd.concat(fs,ignore_index=True),'performance_summary':pd.concat(ms,ignore_index=True)}
            result,reused=_iv_checkpoint(root/study,stable_token({'run_id':run_id,'Series':study,'cells':sha256_lines(part.index)}),
                {'cell_flow':['aggregation','Series','Sample','n_input','n_selected'],
                 'performance_summary':['aggregation','Series','Sample','reference_set','metric','numerator','denominator','value']},produce)
            flows.append(result['cell_flow']);metrics.append(result['performance_summary'])
            manifest['completed_units'].append({'Series':study,'n_cells':len(part),'reused':reused});flush()
        f,m=_e2s_tables(full,'pooled');flows.append(f);metrics.append(m)
        flow=pd.concat(flows,ignore_index=True);performance=pd.concat(metrics,ignore_index=True)
        if len(manifest['completed_units'])!=manifest['expected_units']:raise ValueError('Incomplete full-population summary')
        if flow.loc[flow.aggregation.eq('Series'),'n_input'].sum()!=len(full):raise ValueError('Study denominators do not reconcile')
        if flow.loc[flow.aggregation.eq('Sample'),'n_input'].sum()!=len(full):raise ValueError('Sample denominators do not reconcile')
        for frame,keys in [(flow,['aggregation','Series','Sample']),
                           (performance,['aggregation','Series','Sample','reference_set','metric'])]:
            if frame.duplicated(keys).any():raise ValueError('Duplicate summary key')
        if any(sha256sum(p)!=h for p,h in inputs.items()) or sha256sum(follow_path)!=follow_digest:
            raise ValueError('Closure input or parent manifest changed during run')
        save_table(flow,prefix+'_cell_flow');save_table(performance,prefix+'_performance_summary')
        paths=[TABLE_DIR/(prefix+'_'+s+'.csv') for s in ['cell_flow','performance_summary']]
        manifest.update(status='completed',n_input=len(full),n_scored=len(scored),n_selected=int(full.selected.sum()),
            n_studies=full.Series.nunique(),n_sample_proxies=len(full[['Series','Sample']].drop_duplicates()),
            outputs={str(p.relative_to(PROJECT)):sha256sum(p) for p in paths})
        flush()
        display(flow.loc[flow.aggregation.eq('pooled')].T)
        display(performance.loc[performance.aggregation.eq('Sample') & performance.Series.eq('GSE149356') &
            performance.metric.isin(['lineage_recall','assigned_accuracy','end_to_end_correct_subtype_with_state']),
            ['Sample','reference_set','metric','numerator','denominator','value','status']])
        print('Complete-denominator summary:',len(full),'input cells;',manifest['expected_units'],'study units; see model_contract.json for limits')
    except Exception as exc:
        manifest.update(status='failed',error=str(exc));flush();raise


if EXECUTION_STAGE == 'end_to_end_validation':
    run_end_to_end_summary()
'''


END_TO_END_FOLLOWUP_HELPERS = r'''
def _e2f_barcode(frame, donor_col, barcode_col):
    out=frame.copy()
    extracted=out[barcode_col].astype(str).str.extract(r'^(?:(CB1|CB2|HC1|HC2)_)?([ACGT]+)(?:-[0-9]+)?$')
    if extracted[1].isna().any() or (extracted[0].notna() & extracted[0].ne(out[donor_col])).any():
        raise ValueError('Malformed or mismatched sample/barcode identity')
    out['barcode_core']=extracted[1]
    if out.duplicated([donor_col,'barcode_core']).any():raise ValueError('Duplicate sample/barcode key')
    return out


def _e2f_join(scored,tcr,predictions,groups,lower=.30,upper=.70):
    if not np.isfinite([lower,upper]).all() or not 0<=lower<upper<=1:
        raise ValueError('Invalid or overlapping subtype probability thresholds')
    out=scored.copy().reset_index(drop=True)
    if not out.Series.eq('GSE149356').all() or out.cell_id.duplicated().any():raise ValueError('Wrong or duplicate scored cohort')
    ids=out.cell_id.str.extract(r'^GSE149356_(CB1|CB2|HC1|HC2)@([ACGT]+-[0-9]+)$')
    if ids.isna().any().any() or not ids[0].eq(out.Sample.astype(str)).all():raise ValueError('Scored identity mismatch')
    out['barcode_core']=ids[1].str.replace(r'-[0-9]+$','',regex=True)
    if out.duplicated(['Sample','barcode_core']).any():raise ValueError('Nonunique scored barcode core')
    t=_e2f_barcode(tcr,'donor','barcode')
    cols=['chain_TRD','chain_TRG','cdr3_TRD','cdr3_nt_TRD','cdr3_TRG','cdr3_nt_TRG','v_gene_TRD']
    out=out.merge(t[['donor','barcode_core']+cols],left_on=['Sample','barcode_core'],right_on=['donor','barcode_core'],how='left',validate='one_to_one').drop(columns='donor')
    def present(col):return out[col].notna() & out[col].astype(str).str.strip().ne('')
    trd=out.chain_TRD.eq('TRD') & present('cdr3_TRD') & present('cdr3_nt_TRD')
    trg=out.chain_TRG.eq('TRG') & present('cdr3_TRG') & present('cdr3_nt_TRG')
    if not (trd & trg).eq(out.is_receptor_positive).all():raise ValueError('Recomputed receptor support differs from frozen audit')
    def subtype(value):
        hits=set(re.findall(r'TRDV([123])(?=$|[^0-9])',str(value)))
        return 'Vd'+next(iter(hits)) if len(hits)==1 else ('ambiguous' if len(hits)>1 else 'other_or_unresolved')
    out['sequence_subtype']=out.v_gene_TRD.map(subtype).where(trd,'not_available')
    out['sequence_support']='processed TRD/CDR3; productive/full-length/QC flags unavailable'
    out.loc[~trd,'sequence_support']='not_available'
    p=_e2f_barcode(predictions,'donor','rna_barcode')
    if not p.donor.isin(['HC1','HC2']).all():raise ValueError('Unapproved subtype prediction population')
    if not np.isfinite(p.p_calibrated).all() or not p.p_calibrated.between(0,1).all():raise ValueError('Invalid subtype probabilities')
    out=out.merge(p[['donor','barcode_core','p_calibrated']],left_on=['Sample','barcode_core'],right_on=['donor','barcode_core'],how='left',validate='one_to_one').drop(columns='donor')
    adult=out.Sample.isin(['HC1','HC2'])
    if out.loc[adult,'p_calibrated'].isna().any():raise ValueError('Missing adult frozen subtype predictions')
    if out.loc[~adult,'p_calibrated'].notna().any():raise ValueError('Adult predictions assigned to cord samples')
    out['expression_subtype']=np.select([out.p_calibrated.isna(),out.p_calibrated.ge(upper),out.p_calibrated.le(lower)],
        ['not_available','Vd1','Vd2'],default='other_or_unresolved')
    out['sequence_assisted_subtype']=out.sequence_subtype.where(trd,out.expression_subtype)
    out['sequence_assisted_evidence']=np.where(trd,'processed sequence supported; QC flags unavailable',
        np.where(out.p_calibrated.isna(),'not_available','expression model prediction'))
    out['biological_group']=out.Sample.map(groups)
    if out.biological_group.isna().any():raise ValueError('Unverified biological group')
    # Keep reference expression probabilities separate: no truth-label overwrite.
    out['subtype_prediction']=out.expression_subtype.replace('not_available',pd.NA)
    out['subtype_prediction_status']=np.where(adult,f'14_TCRfree_Vd1_Vd2_subtype_scVI_LR; seed 20260829; {lower:.2f}/{upper:.2f} policy; development evaluation',
        'not_available; no CB1/CB2 predictions in approved release')
    out['selected']=out.hard_call.eq('predicted gdT')
    # A reference-only cell is not emitted as a lineage-positive deployment result.
    out['deployed_expression_subtype']=out.expression_subtype.where(out.selected,'reference_only')
    out['deployed_sequence_assisted_subtype']=out.sequence_assisted_subtype.where(out.selected,'reference_only')
    if not out.state_output_available.eq(out.selected & out.state_score_valid).all():raise ValueError('State output was coupled to subtype')
    if not out.state_output_q95_available.eq(out.state_output_available & out.within_Gray_q95).all():raise ValueError('State-quality routing mismatch')
    return out


def _e2f_sample_summary(cells,programs):
    rows=[];reference=cells.loc[cells.is_receptor_positive]
    metrics=programs+['total_counts','n_genes_by_counts','pct_counts_mt','scaled_HVG_relative_error']
    for sample,base in reference.groupby('Sample',observed=True):
        strata=[('all',base)]+[(str(v),d) for v,d in base.groupby('v_gene_TRD',observed=True,dropna=False)]
        for stratum,original in strata:
            for gate in ['all_valid','Gray_q95_accepted']:
                use=original.loc[original.state_score_valid if gate=='all_valid' else original.within_Gray_q95]
                selected=use.loc[use.selected];missed=use.loc[~use.selected]
                for metric in metrics:
                    a=float(selected[metric].median()) if len(selected) else np.nan
                    b=float(missed[metric].median()) if len(missed) else np.nan
                    rho=np.nan
                    if metric in programs and len(use)>=3 and use[metric].nunique()>1 and use.total_counts.nunique()>1:
                        rho=float(use[[metric,'total_counts']].corr(method='spearman').iloc[0,1])
                    rows.append({'Sample':sample,'biological_group':original.biological_group.iloc[0],
                        'v_gene_TRD_stratum':stratum,'quality_gate':gate,'metric':metric,
                        'n_reference_before_gate':len(original),'n_selected_before_gate':int(original.selected.sum()),
                        'n_missed_before_gate':int((~original.selected).sum()),'n_selected':len(selected),'n_missed':len(missed),
                        'selected_median':a,'missed_median':b,'missed_minus_selected':b-a,'spearman_total_counts':rho})
    return pd.DataFrame(rows)


def _e2f_group_contrasts(cells,programs):
    rows=[];expected=['CB1','CB2','HC1','HC2']
    for gate in ['all_valid','Gray_q95_accepted']:
        use=cells.loc[cells.state_score_valid if gate=='all_valid' else cells.within_Gray_q95]
        populations={'receptor_supported':use.is_receptor_positive,
            'receptor_supported_selected':use.is_receptor_positive & use.selected,'all_predicted':use.selected}
        for population,mask in populations.items():
            part=use.loc[mask];med=part.groupby('Sample',observed=True)[programs].median().reindex(expected)
            counts=part.groupby('Sample',observed=True).size().reindex(expected,fill_value=0)
            for program in programs:
                adult=med.loc[['HC1','HC2'],program];neonate=med.loc[['CB1','CB2'],program]
                a=float(adult.mean()) if adult.notna().all() else np.nan
                n=float(neonate.mean()) if neonate.notna().all() else np.nan
                rows.append({'quality_gate':gate,'population':population,'program':program,
                    **{s+'_median':med.loc[s,program] for s in expected},**{s+'_n_cells':int(counts.loc[s]) for s in expected},
                    'n_adult_donors':int(adult.notna().sum()),'n_neonate_donors':int(neonate.notna().sum()),
                    'adult_mean_of_donor_medians':a,'neonate_mean_of_donor_medians':n,'difference':a-n,
                    'contrast':'adult peripheral blood minus neonatal cord blood',
                    'interpretation':'Descriptive; two donors per group; age/source confounded; previously examined cohort'})
    result=pd.DataFrame(rows)
    ref=result.loc[result.population.eq('receptor_supported')].set_index(['quality_gate','program']).difference
    result['difference_from_reference_contrast']=[r.difference-ref.loc[(r.quality_gate,r.program)] for r in result.itertuples()]
    return result


def _e2f_soft_groups(text):
    expected={'GSM4498551':('CB1','neonate','cord blood'),'GSM4498552':('CB2','neonate','cord blood'),
              'GSM4498553':('HC1','adult','peripheral blood'),'GSM4498554':('HC2','adult','peripheral blood')}
    groups={}
    for block in text.split('^SAMPLE = ')[1:]:
        accession=block.splitlines()[0].strip()
        if accession in expected:
            sample,age,tissue=expected[accession]
            if '!Sample_characteristics_ch1 = age: '+age not in block or '!Sample_characteristics_ch1 = tissue: '+tissue not in block:
                raise ValueError('SOFT biological group changed: '+accession)
            if sample in groups:raise ValueError('Duplicate SOFT sample')
            groups[sample]=age+'; '+tissue
    if set(groups)!={'CB1','CB2','HC1','HC2'}:raise ValueError('Missing source-defined sample groups')
    return groups
'''


END_TO_END_FOLLOWUP_CODE = r'''
def run_end_to_end_followup():
    import gzip
    prefix='PANGEA_primary_K8_end_to_end_validation'
    inputs={}
    def register(path,expected=None):
        path=Path(path);actual=sha256sum(path)
        if expected is not None and actual!=expected:raise ValueError('Provenance mismatch: '+str(path))
        inputs[str(path)]=actual;return path
    parent_path=register(TABLE_DIR/(prefix+'_manifest.json'))
    parent=json.loads(parent_path.read_text())
    if parent.get('status')!='completed':raise ValueError('Base end-to-end run must be complete before follow-up')
    for path,digest in parent['configuration']['inputs'].items():register(path,digest)
    for path,digest in parent['outputs'].items():register(PROJECT/path,digest)
    register(Path(parent['checkpoint_root'])/'build_19b_cNMF_valdiation.py',parent['configuration']['source_sha256'])
    cells_path=PROJECT/'results/models/19b_cNMF_valdiation'/(prefix+'_cells.parquet')
    all_cells=pd.read_parquet(cells_path);cells=all_cells.loc[all_cells.Series.eq('GSE149356')].copy()
    programs=[parent['configuration']['program_mapping'][str(i)] for i in range(1,9)]
    release='14_TCRfree_Vd1_Vd2_subtype_scVI_LR'
    complete_path=register(PROJECT/'results/tables'/release/'completion_manifest.json')
    complete=json.loads(complete_path.read_text())
    if complete.get('status')!='completed' or complete.get('seed')!=20260829 or complete.get('query_adaptation_used') is not False:
        raise ValueError('Unapproved subtype release')
    recorded={row['path']:row['sha256'] for row in complete['outputs']}
    freeze_path=PROJECT/'results/tables'/release/'source_freeze.json'
    pred_path=PROJECT/'predictions'/release/'GSE149356_adult_all_RNA_frozen_subtype_predictions.csv'
    linked_path=PROJECT/'predictions'/release/'GSE149356_adult_frozen_subtype_predictions.csv'
    head_path=PROJECT/'models'/release/'seed_20260829_latent_LR.joblib'
    for path in [freeze_path,pred_path,linked_path,head_path]:
        if str(path) not in recorded:raise ValueError('Artifact absent from original subtype manifest: '+str(path))
        register(path,recorded[str(path)])
    freeze=json.loads(freeze_path.read_text())
    if freeze.get('primary_seed')!=20260829 or freeze.get('external_data_used_for_selection') is not False:
        raise ValueError('Subtype source-freeze contract changed')
    tcr_path=register(complete['inputs']['tcr'])
    if str(tcr_path) not in parent['configuration']['inputs'] or inputs[str(tcr_path)]!=parent['configuration']['inputs'][str(tcr_path)]:
        raise ValueError('Subtype and lineage receptor sources differ')
    soft_path=register(complete['inputs']['metadata'])
    with gzip.open(soft_path,'rt') as handle:groups=_e2f_soft_groups(handle.read())
    tcr=pd.read_csv(tcr_path);predictions=pd.read_csv(pred_path)
    config={'stage':'end_to_end_validation','extension':'followup','source_sha256':sha256sum(PROJECT/'build_19b_cNMF_valdiation.py'),
        'parent_run_id':parent['run_id'],'inputs':inputs,'subtype_release':release,'primary_seed':20260829,
        'lower':.30,'upper':.70,'programs':programs,'q95':parent['configuration']['q95'],'groups':groups,
        'versions':{p:im.version(p) for p in ['numpy','pandas','matplotlib']},'schema_version':2}
    run_id=stable_token(config);root=CHECKPOINT_DIR/'end_to_end_validation'/run_id;root.mkdir(parents=True,exist_ok=True)
    atomic_json(root/'job_spec.json',config);_ur_snapshot(PROJECT/'build_19b_cNMF_valdiation.py',root/'build_19b_cNMF_valdiation.py')
    # Preserve complete earlier-policy results before publishing this run.
    previous_path=TABLE_DIR/(prefix+'_followup_manifest.json')
    if previous_path.exists():
        previous=json.loads(previous_path.read_text())
        if previous['run_id']!=run_id:
            if previous.get('status')!='completed':raise ValueError('Inspect unfinished prior follow-up before replacing it')
            previous_root=Path(previous['checkpoint_root'])
            for relative,digest in previous['outputs'].items():
                source=PROJECT/relative
                if sha256sum(source)!=digest:raise ValueError('Prior follow-up output changed: '+str(source))
                _ur_snapshot(source,previous_root/source.name)
            _ur_snapshot(previous_path,previous_root/previous_path.name)
    manifest={'status':'running','run_id':run_id,'configuration':config,'checkpoint_root':str(root),'expected_units':4,'completed_units':[]}
    def flush():
        manifest['updated_UTC']=datetime.now(timezone.utc).isoformat();atomic_json(TABLE_DIR/(prefix+'_followup_manifest.json'),manifest)
    flush();frames=[];summaries=[];missed=[]
    try:
        for sample in ['CB1','CB2','HC1','HC2']:
            part=cells.loc[cells.Sample.eq(sample)].copy()
            if not len(part):raise ValueError('Missing expected sample: '+sample)
            def produce():
                joined=_e2f_join(part,tcr,predictions,groups,lower=config['lower'],upper=config['upper'])
                return {'subtype_routing':joined,'missed_reference_cells':joined.loc[joined.is_receptor_positive & ~joined.selected].copy(),
                    'missed_reference_sample_summary':_e2f_sample_summary(joined,programs)}
            result,reused=_iv_checkpoint(root/sample,stable_token({'run_id':run_id,'Sample':sample,'cells':sha256_lines(part.cell_id)}),
                {'subtype_routing':['cell_id','expression_subtype','sequence_assisted_subtype','state_output_available'],
                 'missed_reference_cells':['cell_id','selected'],'missed_reference_sample_summary':['Sample','metric','missed_minus_selected']},produce)
            frames.append(result['subtype_routing']);missed.append(result['missed_reference_cells']);summaries.append(result['missed_reference_sample_summary'])
            manifest['completed_units'].append({'Sample':sample,'n_cells':len(part),'reused':reused});flush()
        joined=pd.concat(frames,ignore_index=True);missed_frame=pd.concat(missed,ignore_index=True);summary=pd.concat(summaries,ignore_index=True)
        if joined.cell_id.duplicated().any() or set(joined.cell_id)!=set(cells.cell_id):raise ValueError('Incomplete follow-up population')
        contrasts=_e2f_group_contrasts(joined,programs);outputs=[]
        for suffix,frame in [('missed_reference_cells',missed_frame),('missed_reference_sample_summary',summary),
                             ('subtype_routing',joined),('group_contrasts',contrasts)]:
            path=TABLE_DIR/(prefix+'_'+suffix+'.csv');save_table(frame,prefix+'_'+suffix);outputs.append(path)
        with mpl.rc_context({'font.family':FIGURE_FONT,'font.size':7,'axes.labelsize':7,'axes.titlesize':7,
                'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'pdf.fonttype':42}):
            # Single-panel companion: all programs are reported, not only P1/P7.
            fig,ax=plt.subplots(figsize=(180/25.4,105/25.4));fig.subplots_adjust(left=.13,right=.98,bottom=.31,top=.84)
            palette=['#0072B2','#D55E00','#009E73','#CC79A7']
            for k,sample in enumerate(['CB1','CB2','HC1','HC2']):
                for gate,marker in [('all_valid','o'),('Gray_q95_accepted','x')]:
                    q=summary.loc[summary.Sample.eq(sample)&summary.v_gene_TRD_stratum.eq('all')&summary.quality_gate.eq(gate)&summary.metric.isin(programs)].set_index('metric').reindex(programs)
                    ax.scatter(np.arange(8)+(k-1.5)*.11,q.missed_minus_selected,s=16,marker=marker,color=palette[k],linewidths=.7,
                        label=sample+'; '+('all valid' if gate=='all_valid' else 'q95'))
            ax.axhline(0,color='black',lw=.6);ax.set_xticks(range(8),programs);ax.set_ylabel('Missed minus selected median usage')
            ax.set_title('GSE149356 receptor-supported cells: first-layer selection\nFour donor samples; descriptive differences, no significance test')
            ax.set_xlabel('Frozen K8 program');ax.legend(loc='upper center',bbox_to_anchor=(.5,-.23),ncol=4,frameon=False,columnspacing=1)
            ax.spines[['top','right']].set_visible(False)
            for tx in fig.findobj(mpl.text.Text):tx.set_color('black');tx.set_fontweight('normal')
            name=prefix+'_missed_reference_sample_summary'
            with mpl.rc_context({'savefig.bbox':None}):
                fig.savefig(FIGURE_DIR/(name+'.pdf'),bbox_inches=None);fig.savefig(FIGURE_DIR/(name+'.png'),bbox_inches=None,dpi=300)
            outputs.extend([FIGURE_DIR/(name+'.pdf'),FIGURE_DIR/(name+'.png')]);display(fig);plt.close(fig)
            fig,ax=plt.subplots(figsize=(180/25.4,105/25.4));fig.subplots_adjust(left=.13,right=.98,bottom=.31,top=.84)
            for k,population in enumerate(['receptor_supported','receptor_supported_selected','all_predicted']):
                for gate,marker in [('all_valid','o'),('Gray_q95_accepted','x')]:
                    q=contrasts.loc[contrasts.population.eq(population)&contrasts.quality_gate.eq(gate)].set_index('program').reindex(programs)
                    ax.scatter(np.arange(8)+(k-1)*.16,q.difference,s=18,marker=marker,color=palette[k],linewidths=.7,
                        label=population.replace('_',' ')+'; '+('all valid' if gate=='all_valid' else 'q95'))
            ax.axhline(0,color='black',lw=.6);ax.set_xticks(range(8),programs);ax.set_ylabel('Adult minus neonate: mean of donor medians')
            ax.set_title('GSE149356: selection changes the magnitude of group contrasts\nTwo donors per group; age and blood source are confounded')
            ax.set_xlabel('Frozen K8 program');ax.legend(loc='upper center',bbox_to_anchor=(.5,-.23),ncol=2,frameon=False,columnspacing=1)
            ax.spines[['top','right']].set_visible(False)
            for tx in fig.findobj(mpl.text.Text):tx.set_color('black');tx.set_fontweight('normal')
            name=prefix+'_group_contrasts'
            with mpl.rc_context({'savefig.bbox':None}):
                fig.savefig(FIGURE_DIR/(name+'.pdf'),bbox_inches=None);fig.savefig(FIGURE_DIR/(name+'.png'),bbox_inches=None,dpi=300)
            outputs.extend([FIGURE_DIR/(name+'.pdf'),FIGURE_DIR/(name+'.png')]);display(fig);plt.close(fig)
        if any(sha256sum(p)!=h for p,h in inputs.items()):raise ValueError('An input changed during follow-up')
        manifest.update(status='completed',outputs={str(p.relative_to(PROJECT)):sha256sum(p) for p in outputs},
            n_cells=len(joined),n_missed_reference=len(missed_frame),
            figure_legends={'missed_reference_sample_summary':'Within-sample missed-minus-selected medians among processed TRG+TRD/CDR3-supported cells. Circles: all valid; crosses: original Gray q95. Each color identifies one sample/donor. No inferential error bars or tests. Subtype-stratified and QC summaries, counts and missing arms are retained in the same CSV.',
            'group_contrasts':'Difference between equal-donor means of sample medians, adult peripheral blood (HC1/HC2) minus neonatal cord blood (CB1/CB2). Three cell-selection definitions; circles: all valid; crosses: original Gray q95. Two donors per group. No p values, confidence intervals, equivalence or pure-age-effect claim. Source CSV includes each donor median and denominator.'},
            limitations=['Previously examined cohort; not untouched validation','Processed receptor evidence lacks full QC flags',
                'Subtype predictions are existing frozen adult predictions only; CB predictions remain unavailable',
                'Expression-only and sequence-assisted subtypes are separate; reference labels never establish model accuracy',
                'Two donors per group; age and source confounded; q95 retention differs between groups',
                '0.30/0.70 routing selected after inspecting GSE149356; this is development evaluation, not untouched threshold validation',
                'Original 0.20/0.80 results are preserved under their run-id checkpoint; no probability recalibration, model fitting, killing validation or serum rerun'])
        flush()
        display(joined.loc[joined.selected].groupby(['Sample','expression_subtype'],observed=True).agg(n_cells=('cell_id','size'),state_available=('state_output_available','sum'),within_q95=('state_output_q95_available','sum')))
        display(contrasts[['quality_gate','population','program','difference']])
        print('Follow-up complete. Routing uses 0.30/0.70; historical results preserved. Frozen probabilities, lineage calls and K8 model unchanged.')
    except Exception as exc:
        manifest.update(status='failed',error=str(exc));flush();raise


if EXECUTION_STAGE == 'end_to_end_validation':
    run_end_to_end_followup()
'''


PERTURBATION_PREDICTION_HELPERS = r'''
def _pr_design():
    return [(f'D{day}_{tag}',day,symbol) for day in [7,10,14]
        for tag,symbol in [('WITH_SERUM','+'),('WO_SERUM','-')]]


def _pr_prepare(table,programs):
    required={'donor','condition','day','serum','scenario','replicate','gate','program','n_cells','mean_usage'}
    if not required<=set(table):raise ValueError('Incomplete serum-summary schema')
    native=table.loc[table.scenario.eq('native')].copy()
    if not native.replicate.eq(0).all():raise ValueError('Native summary has unexpected replicate')
    donors=['donorA','donorB','donorC'];gates=['all_valid','historical_q95']
    expected={'D0':(0,'Day 0'),**{c:(d,s) for c,d,s in _pr_design()}}
    keys=['donor','gate','condition','program']
    index=pd.MultiIndex.from_product([donors,gates,list(expected),programs],names=keys)
    if native.duplicated(keys).any() or set(pd.MultiIndex.from_frame(native[keys]))!=set(index):
        raise ValueError('Duplicate or incomplete donor/gate/condition/program design')
    if not native.apply(lambda r: (r.day,r.serum)==expected[r.condition],axis=1).all():
        raise ValueError('Native day/serum metadata mismatch')
    if (not np.isfinite(native[['n_cells','mean_usage']].to_numpy(float)).all()
        or not native.n_cells.gt(0).all() or not native.n_cells.eq(np.floor(native.n_cells)).all()
        or not native.mean_usage.between(-1e-12,1+1e-12).all()):
        raise ValueError('Missing, empty or invalid native target; never replace with zero')
    g=native.groupby(['donor','gate','condition'],observed=True)
    if not g.n_cells.nunique().eq(1).all() or not np.allclose(g.mean_usage.sum(),1,atol=1e-6):
        raise ValueError('Program means must describe one common normalized-usage population')
    return native.sort_values(keys).reset_index(drop=True)


def _pr_inventory(native,quality):
    cols=['donor','condition','day','serum','scenario','replicate','gate','n_cells']
    inventory=native[cols].drop_duplicates()
    if len(inventory)!=42 or inventory.duplicated(['donor','gate','condition']).any():
        raise ValueError('Expected 42 unique donor/gate/condition inventory rows')
    if quality.duplicated(['donor','condition']).any() or len(quality)!=21:raise ValueError('Native quality inventory differs')
    inventory=inventory.merge(quality,on=['donor','condition','scenario','replicate'],validate='many_to_one')
    expected_counts=np.where(inventory.gate.eq('all_valid'),inventory.n_valid_converged,inventory.n_historical_q95_accepted)
    if not np.array_equal(inventory.n_cells.to_numpy(),expected_counts):raise ValueError('Target denominator and quality inventory differ')
    inventory['retained_fraction']=inventory.n_cells/inventory.n_valid_converged
    return inventory


def _pr_simplex(values):
    a=np.asarray(values,float)
    if a.ndim!=2 or not np.isfinite(a).all() or a.shape[1]<2:raise ValueError('Invalid composition prediction')
    u=np.sort(a,axis=1)[:,::-1];css=np.cumsum(u,axis=1)-1
    ok=u-css/np.arange(1,a.shape[1]+1)>0
    rho=ok.sum(axis=1)-1;theta=css[np.arange(len(a)),rho]/(rho+1)
    return np.maximum(a-theta[:,None],0)


def _pr_forecast(training,baseline,programs,method):
    from sklearn.linear_model import Ridge
    if len(baseline)!=len(programs) or baseline.donor.nunique()!=1 or baseline.gate.nunique()!=1:
        raise ValueError('One complete held-out D0 profile required')
    if not baseline.day.eq(0).all() or baseline.program.duplicated().any() or set(baseline.program)!=set(programs):
        raise ValueError('Held-out input must be D0 only, never future RNA')
    donor=baseline.donor.iloc[0];gate=baseline.gate.iloc[0]
    if training.donor.nunique()!=2 or donor in set(training.donor) or not training.gate.eq(gate).all():
        raise ValueError('Training must contain only the two other donors and same gate')
    if training.duplicated(['donor','condition','program']).any() or len(training)!=2*7*len(programs):
        raise ValueError('Incomplete training design')
    b=baseline.set_index('program').loc[programs,'mean_usage'].to_numpy(float)
    reference=training.loc[training.day.eq(0),['donor','program','mean_usage']].rename(columns={'mean_usage':'D0_mean_usage'})
    train=training.loc[training.day.gt(0)].merge(reference,on=['donor','program'],validate='many_to_one')
    train['change']=train.mean_usage-train.D0_mean_usage
    if not np.isfinite(train[['change','D0_mean_usage']].to_numpy()).all():raise ValueError('Missing training D0')
    design=_pr_design();raw=np.empty((6,len(programs)));params=[];outside=np.zeros((6,len(programs)),bool)
    for j,program in enumerate(programs):
        tr=train.loc[train.program.eq(program)].copy()
        if len(tr)!=12 or not tr.groupby('condition').donor.nunique().reindex([c for c,_,_ in design]).eq(2).all():
            raise ValueError('Each training condition needs both donors')
        outside[:,j]=bool(b[j]<tr.D0_mean_usage.min()-1e-12 or b[j]>tr.D0_mean_usage.max()+1e-12)
        if method=='no change':
            delta=np.zeros(6);params.append({'program':program,'parameter':'D0 coefficient','value':1.})
        elif method=='training-donor mean change':
            delta=np.array([tr.loc[tr.condition.eq(c),'change'].mean() for c,_,_ in design])
            params.extend({'program':program,'parameter':c,'value':float(v)} for (c,_,_),v in zip(design,delta))
        elif method=='ridge regression':
            day=tr.day.to_numpy(float)/14;serum=tr.serum.eq('+').to_numpy(float)
            X=np.column_stack([tr.D0_mean_usage.to_numpy(),day,serum,day*serum])
            Xt=np.array([[b[j],d/14,float(s=='+'),d/14*float(s=='+')] for _,d,s in design])
            model=Ridge(alpha=1.,fit_intercept=True,solver='svd').fit(X,tr.change.to_numpy())
            delta=model.predict(Xt)
            params.extend({'program':program,'parameter':name,'value':float(value)} for name,value in
                zip(['intercept','D0_mean_usage','day_over_14','serum','day_serum_interaction'],[model.intercept_,*model.coef_]))
        else:raise ValueError('Unapproved prediction method')
        raw[:,j]=b[j]+delta
    predicted=_pr_simplex(raw);rows=[]
    for i,(condition,day,serum) in enumerate(design):
        for j,program in enumerate(programs):
            rows.append({'donor':donor,'gate':gate,'method':method,'condition':condition,'day':day,'serum':serum,
                'program':program,'D0_mean_usage':b[j],'D0_n_cells':int(baseline.n_cells.iloc[0]),
                'predicted_mean_usage':predicted[i,j],'unprojected_mean_usage':raw[i,j],
                'predicted_change':predicted[i,j]-b[j],'projection_absolute_change':abs(predicted[i,j]-raw[i,j]),
                'D0_outside_training_donor_range':bool(outside[i,j]),
                'training_donors':';'.join(sorted(training.donor.unique())), 'n_training_donors':2,
                'n_training_conditions':12})
    return pd.DataFrame(rows),pd.DataFrame(params)


def _pr_evaluate(predictions,future):
    keys=['donor','gate','condition','program']
    if future.duplicated(keys).any() or set(map(tuple,future[keys].to_numpy()))!=set(map(tuple,predictions[keys].to_numpy())):
        raise ValueError('Future outcomes do not exactly match prediction keys')
    y=future[keys+['mean_usage','n_cells']].rename(columns={'mean_usage':'observed_mean_usage','n_cells':'n_cells'})
    out=predictions.merge(y,on=keys,validate='one_to_one')
    if not np.isfinite(out.observed_mean_usage).all():raise ValueError('Missing future target')
    out['observed_change']=out.observed_mean_usage-out.D0_mean_usage
    out['absolute_error']=(out.predicted_mean_usage-out.observed_mean_usage).abs()
    out['squared_error']=(out.predicted_mean_usage-out.observed_mean_usage)**2
    out['unprojected_absolute_error']=(out.unprojected_mean_usage-out.observed_mean_usage).abs()
    out['observed_nonzero_change']=out.observed_change.abs().gt(1e-12)
    out['direction_correct']=np.where(out.observed_nonzero_change,
        np.sign(out.predicted_change.where(out.predicted_change.abs().gt(1e-12),0)).eq(np.sign(out.observed_change)).astype(float),np.nan)
    return out


def _pr_metrics(predictions):
    rows=[]
    for (donor,gate,method),part in predictions.groupby(['donor','gate','method'],sort=True):
        groups=[(None,part)]+list(part.groupby('program',sort=False))
        for program,p in groups:
            rows.append({'donor':donor,'gate':gate,'method':method,'program':program,'n_evaluations':len(p),
                'MAE':float(p.absolute_error.mean()),'RMSE':float(np.sqrt(p.squared_error.mean())),
                'direction_agreement':float(p.direction_correct.mean()) if p.direction_correct.notna().any() else np.nan,
                'n_nonzero_changes':int(p.observed_nonzero_change.sum()),
                'unprojected_MAE':float(p.unprojected_absolute_error.mean()),
                'mean_projection_absolute_change':float(p.projection_absolute_change.mean())})
    return pd.DataFrame(rows)
'''


PERTURBATION_PREDICTION_CODE = r'''
def run_perturbation_response_prediction():
    prefix='Zenodo17590095_primary_K8_perturbation_response_prediction';source_prefix='Zenodo17590095_primary_K8_time_course'
    inputs={}
    def register(path,expected=None):
        path=Path(path);digest=sha256sum(path)
        if expected is not None and digest!=expected:raise ValueError('Input provenance mismatch: '+str(path))
        inputs[str(path)]=digest;return path
    parent_path=register(TABLE_DIR/(source_prefix+'_manifest.json'));parent=json.loads(parent_path.read_text())
    if parent.get('status')!='completed' or len(parent['completed_units'])!=147:raise ValueError('Completed 147-unit time course required')
    for name,digest in parent['outputs'].items():register(PROJECT/name,digest)
    register(Path(parent['checkpoint_root'])/'build_19b_cNMF_valdiation.py',parent['configuration']['source_sha256'])
    export=MODEL_DIR/(source_prefix+'_input')
    prior_inputs=parent['configuration']['inputs']
    for name in ['RNA_counts.mtx.gz','RNA_genes.tsv','RNA_metadata.csv','export_versions.txt']:
        register(export/name,prior_inputs[name])
    register(export/'export_manifest.json',prior_inputs['export_manifest'])
    scorer=load_frozen_scorer_for_gse268991()
    for name in ['params_path','ranked_path','sd_path','spec_path','spectra_path']:register(scorer[name],prior_inputs[name])
    programs=[scorer['component_to_program'][str(c)] for c in scorer['components']]
    native=_pr_prepare(pd.read_csv(TABLE_DIR/(source_prefix+'_donor_condition_scores.csv')),programs)
    quality=pd.read_csv(TABLE_DIR/(source_prefix+'_quality.csv'));quality=quality.loc[quality.scenario.eq('native')&quality.replicate.eq(0)]
    inventory=_pr_inventory(native,quality)
    config={'stage':'perturbation_response_prediction','source_sha256':sha256sum(PROJECT/'build_19b_cNMF_valdiation.py'),
        'inputs':inputs,'parent_run_id':parent['run_id'],'programs':programs,'primary_gate':'all_valid',
        'gates':['all_valid','historical_q95'],'q95':parent['configuration']['parameters']['q95_historical_only'],
        'methods':['no change','training-donor mean change','ridge regression'],'donors':['donorA','donorB','donorC'],
        'endpoint':'mean normalized usage; change from same-donor D0','ridge_alpha':1.,'ridge_solver':'svd',
        'features':['D0_mean_usage','day_over_14','serum','day_serum_interaction'],'feature_scaling':'fixed; no fitted scaling',
        'composition_constraint':'Euclidean simplex projection for all methods; retain unprojected prediction',
        'versions':{p:im.version(p) for p in ['numpy','pandas','scikit-learn','matplotlib']},'schema_version':1}
    run_id=stable_token(config);root=CHECKPOINT_DIR/'perturbation_response_prediction'/run_id;root.mkdir(parents=True,exist_ok=True)
    atomic_json(root/'job_spec.json',config);_ur_snapshot(PROJECT/'build_19b_cNMF_valdiation.py',root/'build_19b_cNMF_valdiation.py')
    manifest={'status':'running','run_id':run_id,'configuration':config,'checkpoint_root':str(root),'expected_units':18,'completed_units':[]}
    def flush():
        manifest['updated_UTC']=datetime.now(timezone.utc).isoformat();atomic_json(TABLE_DIR/(prefix+'_manifest.json'),manifest)
    flush();prediction_frames=[];metric_frames=[]
    try:
        for donor in config['donors']:
            for gate in config['gates']:
                train=native.loc[native.donor.ne(donor)&native.gate.eq(gate)].copy()
                baseline=native.loc[native.donor.eq(donor)&native.gate.eq(gate)&native.day.eq(0)].copy()
                future=native.loc[native.donor.eq(donor)&native.gate.eq(gate)&native.day.gt(0)].copy()
                for method in config['methods']:
                    specification={'run_id':run_id,'donor':donor,'gate':gate,'method':method}
                    folder=root/donor/gate/method
                    def produce():
                        forecast,parameters=_pr_forecast(train,baseline,programs,method)
                        evaluated=_pr_evaluate(forecast,future)
                        return {'predictions':evaluated,'donor_metrics':_pr_metrics(evaluated),'summary':parameters}
                    frames,reused=_iv_checkpoint(folder,stable_token(specification),
                        {'predictions':['donor','condition','program','predicted_mean_usage','observed_mean_usage'],
                         'donor_metrics':['donor','method','MAE'],'summary':['program','parameter','value']},produce)
                    p=frames['predictions']
                    if (len(p)!=48 or not p.donor.eq(donor).all() or not p.gate.eq(gate).all()
                        or not p.method.eq(method).all() or p.duplicated(['condition','program']).any()
                        or set(p.program)!=set(programs) or set(p.condition)!={c for c,_,_ in _pr_design()}
                        or not np.isfinite(p[['predicted_mean_usage','absolute_error']]).all().all()
                        or not p.predicted_mean_usage.between(-1e-12,1+1e-12).all()
                        or not np.allclose(p.groupby('condition').predicted_mean_usage.sum(),1)):
                        raise ValueError('Prediction checkpoint identity/composition mismatch')
                    prediction_frames.append(p);metric_frames.append(frames['donor_metrics'])
                    manifest['completed_units'].append({**specification,'path':str(folder),'reused':reused});flush()
        if len(manifest['completed_units'])!=18:raise ValueError('Incomplete forecast units')
        predictions=pd.concat(prediction_frames,ignore_index=True);metrics=pd.concat(metric_frames,ignore_index=True)
        if len(predictions)!=864 or predictions.duplicated(['donor','gate','method','condition','program']).any():
            raise ValueError('Forecast population incomplete or duplicated')
        summaries=[]
        for (gate,method,program),part in metrics.groupby(['gate','method','program'],dropna=False):
            if set(part.donor)!=set(config['donors']) or len(part)!=3:raise ValueError('Incomplete donor metric')
            summaries.append({'gate':gate,'method':method,'program':program,'n_donors':3,
                'mean_donor_MAE':part.MAE.mean(),'min_donor_MAE':part.MAE.min(),'max_donor_MAE':part.MAE.max(),
                'mean_donor_RMSE':part.RMSE.mean(),'mean_donor_direction_agreement':part.direction_agreement.mean(),
                **{d+'_MAE':float(part.loc[part.donor.eq(d),'MAE'].iloc[0]) for d in config['donors']}})
        summary=pd.DataFrame(summaries);outputs=[]
        for suffix,frame in [('input_inventory',inventory),('predictions',predictions),('donor_metrics',metrics),('summary',summary)]:
            save_table(frame,prefix+'_'+suffix);outputs.append(TABLE_DIR/(prefix+'_'+suffix+'.csv'))
        with mpl.rc_context({'font.family':FIGURE_FONT,'font.size':7,'axes.labelsize':7,'axes.titlesize':7,
                'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'pdf.fonttype':42}):
            palette=['#0072B2','#D55E00','#009E73'];markers=['o','s','^']
            fig,ax=plt.subplots(figsize=(180/25.4,105/25.4));fig.subplots_adjust(left=.13,right=.98,bottom=.31,top=.84)
            for i,donor in enumerate(config['donors']):
                for gate,style in [('all_valid','-'),('historical_q95','--')]:
                    part=metrics.loc[metrics.donor.eq(donor)&metrics.gate.eq(gate)&metrics.program.isna()].set_index('method').reindex(config['methods'])
                    ax.plot(range(3),part.MAE,linestyle=style,marker=markers[i],color=palette[i],lw=.8,ms=4,
                        markerfacecolor=palette[i] if gate=='all_valid' else 'white',
                        label=donor+'; '+('all valid' if gate=='all_valid' else 'q95'))
            ax.set_xticks(range(3),['No change','Training-donor\nmean change','Ridge regression']);ax.set_ylabel('Mean absolute error in program usage')
            ax.set_ylim(bottom=0);ax.set_title('Serum time course: held-out-donor prediction error\nThree donors; six conditions and eight programs per donor; development only')
            ax.legend(loc='upper center',bbox_to_anchor=(.5,-.24),ncol=3,frameon=False)
            for tx in fig.findobj(mpl.text.Text):tx.set_color('black');tx.set_fontweight('normal')
            name=prefix+'_donor_metrics'
            with mpl.rc_context({'savefig.bbox':None}):
                fig.savefig(FIGURE_DIR/(name+'.pdf'),bbox_inches=None);fig.savefig(FIGURE_DIR/(name+'.png'),bbox_inches=None,dpi=300)
            outputs.extend([FIGURE_DIR/(name+'.pdf'),FIGURE_DIR/(name+'.png')]);display(fig);plt.close(fig)
            fig,ax=plt.subplots(figsize=(180/25.4,105/25.4));fig.subplots_adjust(left=.13,right=.98,bottom=.31,top=.84)
            for i,donor in enumerate(config['donors']):
                for gate,filled in [('all_valid',True),('historical_q95',False)]:
                    part=metrics.loc[metrics.donor.eq(donor)&metrics.gate.eq(gate)&metrics.program.notna()].pivot(index='program',columns='method',values='MAE').reindex(programs)
                    gain=part['training-donor mean change']-part['ridge regression']
                    ax.scatter(np.arange(8)+(i-1)*.16+(0 if filled else .05),gain,s=19,marker=markers[i],
                        facecolors=palette[i] if filled else 'white',edgecolors=palette[i],linewidths=.8,
                        label=donor+'; '+('all valid' if filled else 'q95'))
            ax.axhline(0,color='black',lw=.6);ax.set_xticks(range(8),programs);ax.set_ylabel('Mean-change MAE minus ridge MAE')
            ax.set_xlabel('Frozen K8 program');ax.set_title('Does ridge improve on training-donor mean change?\nPositive values favor ridge; donor-paired descriptive comparison')
            ax.legend(loc='upper center',bbox_to_anchor=(.5,-.24),ncol=3,frameon=False)
            for tx in fig.findobj(mpl.text.Text):tx.set_color('black');tx.set_fontweight('normal')
            name=prefix+'_summary'
            with mpl.rc_context({'savefig.bbox':None}):
                fig.savefig(FIGURE_DIR/(name+'.pdf'),bbox_inches=None);fig.savefig(FIGURE_DIR/(name+'.png'),bbox_inches=None,dpi=300)
            outputs.extend([FIGURE_DIR/(name+'.pdf'),FIGURE_DIR/(name+'.png')]);display(fig);plt.close(fig)
        if any(sha256sum(path)!=h for path,h in inputs.items()):raise ValueError('Prediction input changed during run')
        manifest.update(status='completed',outputs={str(p.relative_to(PROJECT)):sha256sum(p) for p in outputs},
            n_donors=3,n_prediction_rows=len(predictions),
            figure_legends={'donor_metrics':'Each line is one held-out donor. Solid/filled: all-valid primary population; dashed/open: separately fitted historical-q95 sensitivity. MAE averages six conditions and eight programs equally within each donor. Three biological donors, overlapping training folds, no inferential error bars or p values. Source: donor_metrics.csv.',
            'summary':'Each point is one donor-specific program MAE difference: training-donor mean-change baseline minus ridge. Positive favors ridge. Six conditions per donor/program. Filled: all valid; open: historical q95. All eight frozen programs; three biological donors. Source: donor_metrics.csv; macro summaries in summary.csv.'},
            limitations=['Three donors only; previously examined development cohort, not untouched external validation',
                'Known day/serum settings in a new donor; not unseen-perturbation or future-time extrapolation',
                'Observed frozen cNMF mean usages are targets, not independent functional truth',
                'Population composition and within-cell-state responses are not separated by this forecasting task',
                'Historical q95 changes the target population and is retrospective sensitivity, not a prospective eligibility oracle',
                'Two training donors poorly identify baseline-response relationships; no tuning or superiority claim from aggregate loss alone',
                'Exact sign agreement is descriptive and ignores biological effect-size significance; no calibrated uncertainty intervals',
                'No cNMF refit, subtype threshold change, serum rescoring, protein validation or killing prediction'])
        flush()
        display(summary.loc[summary.program.isna(),['gate','method','mean_donor_MAE','min_donor_MAE','max_donor_MAE']].round(5))
        print('Completed 18 donor/method/gate units. No method is selected for deployment by this pilot.')
    except Exception as exc:
        manifest.update(status='failed',error=str(exc));flush();raise


if EXECUTION_STAGE=='perturbation_response_prediction':
    run_perturbation_response_prediction()
'''


USAGE_REPRESENTATION_HELPERS = r'''
def _ur_snapshot(source, target):
    import hashlib
    import os
    import uuid
    payload = source.read_bytes()
    if not target.exists():
        temporary = target.with_name('.' + target.name + '.' + uuid.uuid4().hex + '.tmp')
        temporary.write_bytes(payload)
        os.replace(temporary, target)
    if hashlib.sha256(target.read_bytes()).digest() != hashlib.sha256(payload).digest():
        raise ValueError('Immutable source snapshot differs')


def _ur_validate_usage(raw, normalized):
    import numpy as np
    if (not raw.index.is_unique or not raw.index.equals(normalized.index)
            or not raw.columns.equals(normalized.columns)
            or not np.isfinite(raw.to_numpy()).all() or (raw.to_numpy() < 0).any()
            or not np.isfinite(normalized.to_numpy()).all() or (raw.sum(axis=1) <= 0).any()
            or not np.allclose(raw.div(raw.sum(axis=1), axis=0), normalized, rtol=1e-10, atol=1e-12)):
        raise ValueError('Raw/normalized usage identity, values or normalization differ')


def _ur_prediction_losses(predictions):
    import numpy as np
    result = predictions.copy()
    if 'observed' in result:
        columns = ['probability_CR', 'probability_HDMAPP', 'probability_MO']
        values = result[columns].to_numpy(float)
        classes = np.array(['CR', 'HDMAPP', 'MO'])
        if (not np.isfinite(values).all() or (values < 0).any() or (values > 1).any()
                or not np.allclose(values.sum(axis=1), 1) or not result.observed.isin(classes).all()):
            raise ValueError('Invalid historical probability simplex/labels')
        positions = np.array([list(classes).index(x) for x in result.observed])
        result['loss'] = -np.log(np.clip(values[np.arange(len(result)), positions], np.finfo(float).eps, 1))
        result['predicted'] = classes[values.argmax(axis=1)]
        result['correct'] = result.predicted.eq(result.observed)
    else:
        columns = [c for c in result if c.startswith('predicted_standardized__')]
        if not columns: raise ValueError('Prediction endpoints absent')
        errors = np.column_stack([(result[c] - result[c.replace('predicted_', 'observed_', 1)]) ** 2 for c in columns])
        if not np.isfinite(errors).all(): raise ValueError('Nonfinite prediction losses')
        result['loss'] = errors.mean(axis=1)
    return result


def _ur_verify_predictions(expected, actual):
    import numpy as np
    if not expected.index.is_unique or not expected.index.equals(actual.index):
        raise ValueError('Historical prediction row identities/order changed')
    for c in expected:
        if c.startswith(('probability_', 'predicted_standardized__', 'observed_standardized__')):
            if c not in actual or not np.allclose(expected[c], actual[c], rtol=1e-7, atol=1e-9):
                raise ValueError('Historical normalized predictions do not reproduce: ' + c)
        elif c in ('donor', 'Experiment', 'observed'):
            if c not in actual or not np.array_equal(expected[c].astype(str), actual[c].astype(str)):
                raise ValueError('Historical metadata changed: ' + c)


def _ur_plot_effects(source, title, xlabel, font):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    if font not in ('Arial', 'Helvetica', 'Liberation Sans'): raise ValueError('Unapproved font')
    with mpl.rc_context({'font.family': font, 'font.size': 7, 'font.weight': 'normal',
                         'axes.titlesize': 7, 'axes.titleweight': 'normal', 'axes.labelsize': 7,
                         'xtick.labelsize': 7, 'ytick.labelsize': 7, 'text.color': 'black',
                         'axes.labelcolor': 'black', 'axes.titlecolor': 'black', 'xtick.color': 'black',
                         'ytick.color': 'black', 'pdf.fonttype': 42, 'svg.fonttype': 'none'}):
        fig, ax = plt.subplots(figsize=(180 / 25.4, 80 / 25.4))
        fig.subplots_adjust(left=.36, right=.97, bottom=.23, top=.82)
        ax.axvline(0, color='black', linestyle='--', linewidth=.6)
        labels = list(source.comparison.drop_duplicates())
        for i, label in enumerate(labels):
            values = source.loc[source.comparison.eq(label), 'error_reduction'].to_numpy()
            ax.scatter(values, i + np.linspace(-.12, .12, len(values)), s=14, color='#0072B2', edgecolors='none')
            ax.plot([values.mean()] * 2, [i-.2, i+.2], color='black', linewidth=1)
        ax.set_yticks(range(len(labels)), labels); ax.invert_yaxis()
        ax.set_xlabel(xlabel); ax.set_title(title, pad=9)
        ax.spines[['top', 'right']].set_visible(False)
        for text in fig.findobj(mpl.text.Text): text.set_color('black'); text.set_fontweight('normal')
    return fig
'''


USAGE_REPRESENTATION_CODE = r'''
def _run_usage_representation_validation(context):
    import shutil
    from scipy.stats import spearmanr
    from threadpoolctl import threadpool_limits
    prefix = 'Gray_primary_K8_usage_representation_validation'
    old_path = TABLE_DIR / 'Gray_primary_K8_incremental_value_validation_manifest.json'
    robust_path = TABLE_DIR / 'Gray_primary_K8_robustness_validation_manifest.json'
    old, robust = [json.loads(p.read_text()) for p in (old_path, robust_path)]
    if old.get('status') != 'completed' or robust.get('status') != 'completed':
        raise ValueError('Required historical stage incomplete')
    if sha256sum(robust_path) != old['configuration']['prior_manifest_sha256']:
        raise ValueError('Historical source manifest changed')
    scorer = load_frozen_scorer_for_gse268991()
    frozen = {k: sha256sum(scorer[k]) for k in old['configuration']['frozen_model']}
    if frozen != old['configuration']['frozen_model']: raise ValueError('Frozen K8 changed')
    versions = {p: im.version(p) for p in old['configuration']['versions']}
    if versions != old['configuration']['versions']: raise ValueError('Historical runtime versions changed')
    for name, digest in old['outputs'].items():
        if sha256sum(PROJECT / name) != digest: raise ValueError('Historical aggregate changed: ' + name)
    old_units = {u['label']: u for u in old['completed_units']}
    if len(old_units) != 651: raise ValueError('Historical units incomplete/duplicated')
    libraries = [u for u in robust['completed_units'] if u['label'].count('/') == 1
                 and u['label'].startswith(('GSE268991/', 'GSE294268/'))]
    if len(libraries) != 27 or len({u['label'] for u in libraries}) != 27: raise ValueError('Library cohort changed')
    def checked_directory(unit):
        directory = Path(unit['path'])
        if not validate_marker(directory / 'scoring.complete.json', unit['signature']):
            raise ValueError('Historical checkpoint failed validation: ' + unit['label'])
        return directory
    config = {'stage': 'usage_representation_validation', 'schema_version': 1,
        'source_sha256': sha256sum(PROJECT / 'build_19b_cNMF_valdiation.py'),
        'prior_manifest_sha256': sha256sum(old_path), 'robustness_manifest_sha256': sha256sum(robust_path),
        'frozen_model': frozen, 'q95': old['configuration']['q95'], 'versions': versions,
        'grids': old['configuration']['grids'], 'seed': 20261023,
        'representation': 'usage_raw', 'transformation': 'training-only donor-weighted StandardScaler; no log transform',
        'libraries': [{'label': u['label'], 'marker_sha256': sha256sum(checked_directory(u) / 'scoring.complete.json')}
                      for u in libraries],
        'evidence_scope': 'post-evaluation development; raw is not absolute RNA or functional activity'}
    run_id = stable_token(config)
    root = CHECKPOINT_DIR / 'usage_representation_validation' / run_id
    root.mkdir(parents=True, exist_ok=True); atomic_json(root / 'job_spec.json', config)
    source = PROJECT / 'build_19b_cNMF_valdiation.py'; snapshot = root / source.name
    _ur_snapshot(source, snapshot)
    if sha256sum(snapshot) != config['source_sha256']: raise ValueError('Source snapshot changed')
    progress = {'status': 'running', 'run_id': run_id, 'checkpoint_root': str(root),
                'configuration': config, 'expected_units': 104, 'completed_units': [], 'active_unit': None}
    def flush():
        progress['updated_UTC'] = datetime.now(timezone.utc).isoformat()
        atomic_json(TABLE_DIR / f'{prefix}_manifest.json', progress)
    context.update(progress=progress, flush=flush); flush()
    external = {}; raw_audit = []
    for u in libraries:
        dataset, library = u['label'].split('/')
        directory = checked_directory(u); scored = checked_directory(old_units[u['label']])
        keys = ['usage_raw', 'usage_normalized', 'quality', 'cell_metadata']
        if dataset == 'GSE268991': keys += ['gamma_delta_TCR_metadata', 'ADT_raw_and_CLR']
        frames = {k: pd.read_parquet(directory / f'{k}.parquet') for k in keys}
        frames['full_measured_AUCell_scores'] = pd.read_parquet(scored / 'full_measured_AUCell_scores.parquet')
        ids = frames['usage_raw'].index
        if any(not f.index.equals(ids) for f in frames.values()): raise ValueError('Library row alignment changed')
        _ur_validate_usage(frames['usage_raw'], frames['usage_normalized'])
        total = frames['usage_raw'].sum(axis=1)
        raw_audit.append({'dataset': dataset, 'library': library, 'n_cells': len(total),
            'sum_raw_q05': total.quantile(.05), 'sum_raw_median': total.median(), 'sum_raw_q95': total.quantile(.95),
            'sum_raw_depth_Spearman': spearmanr(total, frames['quality'].library_over_training_universe).statistic,
            'scope': 'all released cells in library; descriptive, no cell-level inference'})
        for k, f in frames.items(): external.setdefault(dataset, {}).setdefault(k, []).append(f)
    external = {ds: {k: pd.concat(v) for k, v in frames.items()} for ds, frames in external.items()}
    save_table(pd.DataFrame(raw_audit), prefix + '_raw_usage_audit')
    methods = ['covariates_plus_cNMF', 'covariates_plus_full_measured_Hallmark_AUCell_plus_cNMF']
    results = []; historical = []; gates = []; legends = []
    def figure(source_data, name, title, xlabel, unit):
        save_table(source_data, name + '_source_data')
        fig = _ur_plot_effects(source_data, title, xlabel, FIGURE_FONT)
        with mpl.rc_context({'savefig.bbox': None, 'pdf.fonttype': 42}):
            fig.savefig(FIGURE_DIR / (name + '.pdf'), bbox_inches=None)
            fig.savefig(FIGURE_DIR / (name + '.png'), bbox_inches=None, dpi=300)
        display(fig); plt.close(fig)
        legends.append({'figure': name, 'unit': unit, 'n': source_data.unit.nunique(),
            'center': 'equal-unit arithmetic mean', 'spread': 'all unit effects; no inferential error bars',
            'statistics': 'descriptive post-evaluation diagnostic; no confirmatory P values',
            'source_data': name + '_source_data.csv', 'alignment': 'not applicable: single panel',
            'submission_status': 'rendered QA and final Excel source-data packaging required before submission'})
    for dataset, frames in external.items():
        raw = frames['usage_raw']; norm = frames['usage_normalized']; meta = frames['cell_metadata']; quality = frames['quality']
        _ur_validate_usage(raw, norm)
        if not meta.index.is_unique: raise ValueError('Duplicate external cell IDs')
        for gate in ('all_cells', 'Gray_q95_accepted'):
            if dataset == 'GSE268991':
                eligibility = _strict_multimodal_eligibility(frames['gamma_delta_TCR_metadata'], frames['ADT_raw_and_CLR'])
                target = frames['ADT_raw_and_CLR'][['ADT_CLR__CD45RA', 'ADT_CLR__CCR7']]
                mask = eligibility.single_productive_TRG_TRD & eligibility.has_ADT_counts & target.notna().all(axis=1)
            else: mask = meta.Allergic.eq('NonA')
            if gate == 'Gray_q95_accepted': mask &= quality.within_Gray_q95
            if dataset == 'GSE268991':
                X = pd.concat([norm.add_prefix('cNMF_'), frames['full_measured_AUCell_scores'].add_prefix('full_measured_')], axis=1)
                X['covariate_log1p_library_size'] = np.log1p(quality.library_over_training_universe)
                X['covariate_condition'] = meta.condition
                X = X.loc[mask]; Y = target.loc[mask]; metadata = meta.loc[mask].copy()
                raw_X = X.copy(); raw_X[raw.add_prefix('cNMF_').columns] = raw.loc[mask].to_numpy()
                splits = ['donor']
                if metadata.donor.nunique() != 7: raise ValueError('Protein donor cohort changed')
            else:
                scores = pd.read_csv(TABLE_DIR / f'{dataset}_primary_K8_incremental_value_validation_{gate}_donor_scores.csv',
                                     dtype={'donor': str, 'Experiment': str, 'Stim': str})
                metadata = scores[['donor', 'Experiment', 'Stim']].copy(); Y = scores[['Stim']].copy()
                columns = [c for c in scores if c.startswith(('cNMF_', 'covariate_', 'full_measured_'))]
                X = scores[columns].copy(); X['covariate_Experiment'] = metadata.Experiment
                group_keys = ['donor', 'Experiment', 'Stim']
                chosen = raw.loc[mask].join(meta.loc[mask, group_keys]); chosen['donor'] = chosen.donor.astype(str)
                grouped = chosen.groupby(group_keys, observed=True)
                median = grouped[list(raw)].median(); counts = grouped.size()
                wanted = pd.MultiIndex.from_frame(metadata)
                if wanted.has_duplicates or not wanted.isin(median.index).all() or (counts.reindex(wanted) < 20).any():
                    raise ValueError('Raw aggregate cohort/minimum cells differs')
                raw_X = X.copy(); raw_X[['cNMF_' + c for c in raw]] = median.reindex(wanted).to_numpy()
                if metadata.donor.nunique() != 15 or metadata.Experiment.nunique() != 4: raise ValueError('Stimulation cohort changed')
                splits = ['donor', 'Experiment']
            for split in splits:
                for heldout in sorted(metadata[split].astype(str).unique()):
                    train = metadata.index[metadata[split].astype(str).ne(heldout)]
                    test = metadata.index[metadata[split].astype(str).eq(heldout)]
                    for method in methods:
                        label = f'{dataset}/benchmark/{gate}/{split}/{heldout}/{method}'
                        previous = checked_directory(old_units[label])
                        spec = json.loads((previous / 'job_spec.json').read_text()); columns = spec['columns']
                        if stable_token({'run_id': old['run_id'], 'specification': spec}) != old_units[label]['signature']:
                            raise ValueError('Historical benchmark specification changed')
                        if sha256_lines(train) != spec['train_ids_sha256'] or sha256_lines(test) != spec['test_ids_sha256']:
                            raise ValueError('Historical train/test identities changed')
                        old_predictions = pd.read_parquet(previous / 'predictions.parquet')
                        old_metrics = pd.read_parquet(previous / 'donor_metrics.parquet')
                        penalty = float(old_predictions.selected_penalty.iloc[0])
                        with threadpool_limits(limits=THREADS):
                            reproduced, _, _ = _iv_fit(X[columns], Y, metadata, train, test, dataset == 'GSE294268', penalty)
                        _ur_verify_predictions(old_predictions, reproduced)
                        old_metrics['representation'] = 'usage_normalized'; results.append(old_metrics)
                        old_predictions['representation'] = 'usage_normalized'; historical.append(old_predictions)
                        input_signature = stable_token({k: pd.util.hash_pandas_object(f, index=True).astype(str).tolist()
                            for k, f in {'X': raw_X[columns], 'Y': Y, 'metadata': metadata}.items()})
                        specification = {**spec, 'representation': 'usage_raw', 'input_signature': input_signature,
                                         'historical_signature': old_units[label]['signature']}
                        signature = stable_token({'run_id': run_id, 'specification': specification})
                        folder = root / label / signature; folder.mkdir(parents=True, exist_ok=True)
                        atomic_json(folder / 'job_spec.json', specification)
                        progress['active_unit'] = label; flush()
                        def fit_outer():
                            classification = dataset == 'GSE294268'
                            with threadpool_limits(limits=THREADS):
                                selected, inner = _iv_select(raw_X[columns], Y, metadata, split, heldout, classification,
                                    config['grids']['classification_C' if classification else 'regression_alpha'])
                                predictions, metrics, audit = _iv_fit(raw_X[columns], Y, metadata, train, test, classification, selected)
                            for frame in (predictions, metrics, inner, audit):
                                for key in ('dataset', 'quality_gate', 'split', 'heldout', 'method', 'representation'):
                                    frame[key] = specification[key]
                                frame['selected_penalty'] = selected
                            metrics['n_features'] = len(columns)
                            return {'predictions': predictions, 'donor_metrics': metrics, 'inner_losses': inner, 'training_scaling': audit}
                        fitted, reused = _iv_checkpoint(folder, signature,
                            {'predictions': ['donor', 'representation'], 'donor_metrics': ['donor', 'loss'],
                             'inner_losses': ['penalty', 'loss'], 'training_scaling': ['feature', 'center', 'scale']}, fit_outer)
                        if not fitted['predictions'].index.equals(test): raise ValueError('Raw prediction identities changed')
                        results.append(fitted['donor_metrics'])
                        progress['completed_units'].append({'label': label, 'path': str(folder), 'signature': signature, 'reused': reused})
                        flush(); print('usage_representation_validation', len(progress['completed_units']), '/104', label, flush=True)
        if dataset == 'GSE294268':
            cohort = set(metadata.donor.astype(str))
            selected = meta.loc[meta.Allergic.eq('NonA') & meta.donor.astype(str).isin(cohort)].copy()
            selected = selected.join(quality[['within_Gray_q95', 'library_over_training_universe', 'scaled_HVG_relative_error']]).join(norm)
            retention = selected.groupby(['Experiment', 'donor', 'Stim'], observed=True).within_Gray_q95.agg(n_cells='size', n_retained='sum').reset_index()
            retention['fraction_retained'] = retention.n_retained / retention.n_cells
            save_table(retention, f'{dataset}_primary_K8_usage_representation_validation_q95_retention')
            profile = selected.groupby(['Experiment', 'donor', 'Stim', 'within_Gray_q95'], observed=True)[
                ['library_over_training_universe', 'scaled_HVG_relative_error'] + list(norm)].median().reset_index()
            save_table(profile, f'{dataset}_primary_K8_usage_representation_validation_q95_profiles')
    if len(progress['completed_units']) != 104: raise ValueError('Incomplete raw-representation fits')
    benchmark = pd.concat(results, ignore_index=True); paired_all = []; summaries = []
    for dataset, part in benchmark.groupby('dataset'):
        name = f'{dataset}_primary_K8_usage_representation_validation'
        save_table(part, name + '_benchmark')
        keys = ['dataset', 'quality_gate', 'split', 'heldout', 'method', 'donor']
        wide = part.pivot(index=keys, columns='representation', values='loss')
        if wide.isna().any().any(): raise ValueError('Unpaired representations')
        wide['error_reduction'] = wide.usage_normalized - wide.usage_raw
        paired = wide.reset_index()
        if dataset == 'GSE294268':
            mapping = part[['donor', 'Experiment']].drop_duplicates()
            if mapping.donor.duplicated().any(): raise ValueError('Donor crosses Experiments')
            paired = paired.merge(mapping, on='donor', validate='many_to_one')
        save_table(paired, name + '_paired_losses'); paired_all.append(paired)
        for (gate, split), group in paired.groupby(['quality_gate', 'split']):
            sources = []
            for method, values in group.groupby('method'):
                unit = 'Experiment' if split == 'Experiment' else 'donor'
                effects = values.groupby(unit).error_reduction.mean()
                sources.append(pd.DataFrame({'unit': effects.index, 'error_reduction': effects.to_numpy(),
                    'comparison': 'cNMF' if method == methods[0] else 'Hallmark AUCell + cNMF', 'method': method}))
                rng = np.random.default_rng(20261023)
                boot = effects.to_numpy()[rng.integers(0, len(effects), size=(4000, len(effects)))].mean(axis=1)
                summaries.append({'dataset': dataset, 'quality_gate': gate, 'split': split, 'method': method,
                    'n_units': len(effects), 'n_improved': int((effects > 0).sum()), 'mean_error_reduction': effects.mean(),
                    'conditional_95CI_low': np.quantile(boot, .025), 'conditional_95CI_high': np.quantile(boot, .975),
                    'interval_scope': 'fixed-model unit resampling; no refitting or multiplicity adjustment; exploratory'})
            source_data = pd.concat(sources, ignore_index=True)
            figure(source_data, name + f'_{gate}_{split}_paired_effects',
                f'{dataset}: {gate}; held-out {split}\nPoints: {len(effects)} {unit} units; black marks: equal-unit mean',
                'Normalized-usage loss minus raw-usage loss\nPositive values favour raw usage', unit)
    # Diagnose original Hallmark complementarity; do not substitute raw results here.
    predictions = pd.concat(historical, ignore_index=True)
    base = 'covariates_plus_full_measured_Hallmark_AUCell'
    diagnostic = []
    for u in old['completed_units']:
        if '/benchmark/' not in u['label'] or not u['label'].startswith('GSE294268/'): continue
        if u['label'].split('/')[3] != 'Experiment' or u['label'].split('/')[-1] not in (base, base + '_plus_cNMF'): continue
        p = pd.read_parquet(checked_directory(u) / 'predictions.parquet')
        diagnostic.append(_ur_prediction_losses(p))
    diagnostic = pd.concat(diagnostic, ignore_index=True)
    name = 'GSE294268_primary_K8_usage_representation_validation'
    save_table(diagnostic, name + '_historical_prediction_diagnostics')
    wide = diagnostic.pivot(index=['quality_gate', 'Experiment', 'donor', 'observed'], columns='method', values='loss')
    if wide.isna().any().any(): raise ValueError('Historical diagnostic pairing failed')
    wide['error_reduction'] = wide[base] - wide[base + '_plus_cNMF']
    save_table(wide.reset_index(), name + '_historical_stimulus_effects')
    for gate, part in wide.reset_index().groupby('quality_gate'):
        source_data = part.loc[part.Experiment.eq('GD005'), ['donor', 'observed', 'error_reduction']].rename(columns={'donor': 'unit', 'observed': 'comparison'})
        figure(source_data, name + f'_{gate}_GD005_stimulus_effects',
            f'GD005: {gate}\nPoints: 4 donors; black marks: donor mean',
            'Hallmark loss minus Hallmark + cNMF loss\nHistorical normalized usage; positive values favour adding cNMF', 'donor')
    save_table(pd.DataFrame(summaries), prefix + '_summary')
    save_table(pd.DataFrame(legends), prefix + '_figure_legends')
    if frozen != {k: sha256sum(scorer[k]) for k in frozen}: raise ValueError('Frozen K8 changed during run')
    if sha256sum(old_path) != config['prior_manifest_sha256']: raise ValueError('Historical manifest changed during run')
    progress.update(status='completed', active_unit=None, historical_prediction_reproductions=104,
        limitations=['Previously examined development cohorts', 'Only 7 protein donors and 4 stimulation Experiments',
                     'Raw coefficients are not absolute activity', 'q95 changes cell composition',
                     'No killing endpoint or mechanism inference', 'Conditional intervals omit refitting uncertainty'],
        outputs={str(p.relative_to(PROJECT)): sha256sum(p) for directory, ext in ((TABLE_DIR, 'csv'), (FIGURE_DIR, 'pdf'))
                 for p in directory.glob(f'*_primary_K8_usage_representation_validation_*.{ext}')})
    flush(); display(pd.DataFrame(summaries)); print('Completed 104 raw-usage comparisons; no deployment selection made.')


def run_usage_representation_validation():
    context = {}
    try: return _run_usage_representation_validation(context)
    except Exception as error:
        if context:
            context['progress'].update(status='failed', error=f'{type(error).__name__}: {error}')
            context['flush']()
        raise


if EXECUTION_STAGE == 'usage_representation_validation':
    run_usage_representation_validation()
else:
    print('Usage representation comparison not launched in stage', EXECUTION_STAGE)
'''


INCREMENTAL_CODE = r'''
def _run_incremental_value_validation(context):
    """Development-only nested benchmark. No cNMF training or threshold changes."""
    import shutil
    from threadpoolctl import threadpool_limits
    import numba
    numba.set_num_threads(max(1, min(THREADS, numba.config.NUMBA_NUM_THREADS)))
    prefix = "Gray_primary_K8_incremental_value_validation"
    prior_path = TABLE_DIR / "Gray_primary_K8_robustness_validation_manifest.json"
    prior = json.loads(prior_path.read_text())
    if prior.get("status") != "completed": raise ValueError("Robustness stage is incomplete")
    scorer = load_frozen_scorer_for_gse268991()
    frozen = {k: sha256sum(scorer[k]) for k in ("spec_path", "spectra_path", "sd_path", "ranked_path", "params_path")}
    if frozen != prior["configuration"]["frozen_model"]: raise ValueError("Frozen K8 differs from audited model")
    if sha256sum(GSE268991_PREANALYSIS) != GSE268991_PREANALYSIS_SHA256:
        raise ValueError("Original preanalysis changed")
    gmt = SCORING_ROOT / "resources" / "h.all.v2025.1.Hs.symbols.gmt"
    if sha256sum(gmt) != "f22066af72e215ccb7b89d88e492c07e1eef17534c2ca7b0f9902cfecbbdd8e9":
        raise ValueError("Hallmark resource differs from approved preflight")
    hallmark = {}
    for line in gmt.read_text().splitlines():
        name, url, *genes = line.split("\t")
        if name in hallmark or not name.startswith("HALLMARK_"): raise ValueError("Invalid Hallmark GMT")
        hallmark[name] = list(dict.fromkeys(genes))
    if len(hallmark) != 50: raise ValueError("Expected all 50 Hallmark sets")
    selected_units = [u for u in prior["completed_units"]
                      if u["label"].count("/") == 1 and u["label"].startswith(("GSE268991/", "GSE294268/"))]
    if len(selected_units) != 27 or len({u["label"] for u in selected_units}) != 27:
        raise ValueError("Expected 27 distinct external library checkpoints")
    libraries, gene_sets = [], []
    for u in selected_units:
        directory = Path(u["path"])
        marker = directory / "scoring.complete.json"
        if not validate_marker(marker, u["signature"]): raise ValueError("Missing prior checkpoint")
        spec = json.loads((directory / "job_spec.json").read_text())
        for record in spec["inputs"].values():
            if sha256sum(record["path"]) != record["sha256"]: raise ValueError("Raw input changed")
        audit = pd.read_parquet(directory / "feature_mapping.parquet")
        targets = audit.training_feature.astype(str)
        if targets.isna().any(): raise ValueError("Missing mapped gene identity")
        gene_sets.append(set(targets))
        libraries.append((u, spec, audit))
    # Annotation-only intersection: neither expression abundance nor outcome is used.
    full_universe = pd.Index(sorted(set.intersection(*gene_sets)))
    if len(full_universe) <= len(scorer["universe"]):
        raise ValueError("Full measured universe unexpectedly smaller than frozen universe")
    q95 = prior["configuration"]["q95"]
    programs = ["P1", "P2", "P3", "P4", "P5", "P7", "P9", "P10"]
    top50 = {"top50_" + scorer["component_to_program"][c]: genes for c, genes in scorer["signatures"].items()}
    grids = {"classification_C": [.01, .1, 1., 10., 100.], "regression_alpha": [.01, .1, 1., 10., 100.]}
    config = {"stage": "incremental_value_validation", "schema_version": 1,
              "source_sha256": sha256sum(PROJECT / "build_19b_cNMF_valdiation.py"),
              "prior_manifest_sha256": sha256sum(prior_path), "frozen_model": frozen,
              "preanalysis_sha256": sha256sum(GSE268991_PREANALYSIS), "gmt_sha256": sha256sum(gmt),
              "full_universe_sha256": sha256_lines(full_universe), "n_full_genes": len(full_universe),
              "frozen_universe_sha256": sha256_lines(scorer["universe"]), "q95": q95,
              "grids": grids, "seed": 20261023, "AUCell_n_up_fraction": .05,
              "versions": {p: im.version(p) for p in ("numpy", "pandas", "scipy", "scikit-learn", "decoupler", "anndata")},
              "evidence_scope": "development evaluation on previously examined cohorts; no killing endpoint",
              "libraries": [{"label": u["label"], "marker_sha256": sha256sum(Path(u["path"]) / "scoring.complete.json"),
                             "inputs": spec["inputs"]} for u, spec, _ in libraries]}
    run_id = stable_token(config)
    root = CHECKPOINT_DIR / "incremental_value_validation" / run_id
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / "job_spec.json", config)
    for source in (PROJECT / "build_19b_cNMF_valdiation.py", gmt):
        destination = root / source.name
        if not destination.exists(): shutil.copy2(source, destination)
        if sha256sum(destination) != sha256sum(source): raise ValueError("Snapshot differs")
    atomic_parquet(pd.DataFrame({"gene": full_universe}), root / "full_measured_gene_universe.parquet")
    progress = {"status": "running", "run_id": run_id, "checkpoint_root": str(root),
                "configuration": config, "completed_units": [], "expected_units": 651}
    manifest_path = TABLE_DIR / f"{prefix}_manifest.json"
    def flush():
        progress["updated_UTC"] = datetime.now(timezone.utc).isoformat()
        atomic_json(manifest_path, progress)
    context.update(progress=progress, flush=flush)
    flush()
    def unit(label, spec, schemas, produce):
        signature = stable_token({"run_id": run_id, "specification": spec})
        folder = root / label / signature
        folder.mkdir(parents=True, exist_ok=True)
        atomic_json(folder / "job_spec.json", spec)
        progress["active_unit"] = label; flush()
        try:
            frames, reused = _iv_checkpoint(folder, signature, schemas, produce)
        except Exception as error:
            progress.update(status="failed", failed_unit=label, error=f"{type(error).__name__}: {error}")
            flush(); raise
        progress["completed_units"].append({"label": label, "path": str(folder), "signature": signature, "reused": reused})
        flush(); print("incremental_value_validation", len(progress["completed_units"]), "/651", label, flush=True)
        return frames
    external = {}
    for u, old_spec, audit in libraries:
        dataset, library = u["label"].split("/")
        directory = Path(u["path"])
        old = {key: pd.read_parquet(directory / f"{key}.parquet") for key in (
            "usage_normalized", "rank_signature_scores", "AUCell_scores", "quality", "cell_metadata")}
        if dataset == "GSE268991":
            for key in ("gamma_delta_TCR_metadata", "ADT_raw_and_CLR"):
                old[key] = pd.read_parquet(directory / f"{key}.parquet")
        ids = old["usage_normalized"].index
        if not ids.is_unique or any(not frame.index.equals(ids) for frame in old.values()):
            raise ValueError(f"Prior tables do not share cell identity/order: {library}")
        paths = {k: Path(v["path"]) for k, v in old_spec["inputs"].items()}
        spec = {"prior_signature": u["signature"], "cell_ids_sha256": sha256_lines(ids),
                "n_cells": len(ids), "full_universe_sha256": config["full_universe_sha256"]}
        def score_library():
            features = pd.read_csv(paths["features"], sep="\t", header=None)
            gene_mask = features.iloc[:, 2].eq("Gene Expression").to_numpy()
            if features.loc[gene_mask, 0].astype(str).tolist() != audit.feature_id.astype(str).tolist():
                raise ValueError("Raw feature order differs from audited alignment")
            with gzip.open(paths["matrix"], "rb") as handle, threadpool_limits(limits=THREADS):
                matrix = scipy.io.mmread(handle).tocsr()
            barcodes = pd.Index(pd.read_csv(paths["barcodes"], sep="\t", header=None).iloc[:, 0].astype(str))
            if matrix.shape != (len(features), len(barcodes)) or not barcodes.is_unique:
                raise ValueError("Raw library shape/IDs invalid")
            if dataset == "GSE268991": raw_ids = pd.Index([f"{library}:{b}" for b in barcodes])
            else:
                suffixes = {str(value).rsplit("_", 1)[1] for value in ids}
                if len(suffixes) != 1: raise ValueError("Ambiguous released barcode suffix")
                raw_ids = pd.Index(barcodes + "_" + next(iter(suffixes)))
            selected = raw_ids.get_indexer(ids)
            if (selected < 0).any(): raise ValueError("Raw-to-released cell crosswalk failed")
            counts = matrix[gene_mask, :].T.tocsr()[selected]
            if not np.isfinite(counts.data).all() or (counts.data < 0).any() or not np.equal(counts.data, np.floor(counts.data)).all():
                raise ValueError("Scoring input is not raw UMI")
            outputs, coverage_rows = {}, []
            for universe_name, universe, signatures in (
                ("full_measured", full_universe, {**hallmark, **top50}),
                ("frozen", scorer["universe"], hallmark)):
                positions = universe.get_indexer(audit.training_feature.astype(str))
                present = positions >= 0
                mapping = sparse.csr_matrix((np.ones(present.sum()), (np.flatnonzero(present), positions[present])),
                                             shape=(len(audit), len(universe)))
                aligned = (counts @ mapping).tocsr()
                scores = _aucell_validation_scores(aligned, universe, signatures)
                scores.index = ids
                if not np.isfinite(scores.to_numpy()).all(): raise ValueError("Nonfinite AUCell scores")
                outputs[f"{universe_name}_AUCell_scores"] = scores
                measured = set(audit.training_feature) & set(universe)
                for name, genes in signatures.items():
                    observed = set(genes) & measured
                    coverage_rows.append({"dataset": dataset, "library": library, "universe": universe_name,
                        "signature": name, "n_signature_genes": len(set(genes)), "n_observed_genes": len(observed),
                        "coverage": len(observed) / len(set(genes)), "missing_genes": ";".join(sorted(set(genes)-observed))})
            outputs["coverage"] = pd.DataFrame(coverage_rows)
            return outputs
        scores = unit(f"{dataset}/{library}", spec,
                      {"full_measured_AUCell_scores": list(hallmark) + list(top50),
                       "frozen_AUCell_scores": list(hallmark), "coverage": ["signature", "coverage"]}, score_library)
        for key in ("full_measured_AUCell_scores", "frozen_AUCell_scores"):
            if not scores[key].index.equals(ids) or not np.isfinite(scores[key].to_numpy()).all():
                raise ValueError("Reused scores have invalid cell identities/values")
        old.update(scores)
        for key, frame in old.items(): external.setdefault(dataset, {}).setdefault(key, []).append(frame)
    external = {ds: {key: pd.concat(parts, ignore_index=(key == "coverage")) for key, parts in frames.items()}
                for ds, frames in external.items()}

    def feature_methods(X):
        cov = [c for c in X if c.startswith("covariate_")]
        cnmf = [c for c in X if c.startswith("cNMF_")]
        groups = {"rank_signature": [c for c in X if c.startswith("rank_signature_")],
                  "AUCell": [c for c in X if c.startswith("AUCell_")],
                  "full_measured_AUCell": [c for c in X if c.startswith("full_measured_top50_")],
                  "Hallmark_AUCell": [c for c in X if c.startswith("frozen_HALLMARK_")],
                  "full_measured_Hallmark_AUCell": [c for c in X if c.startswith("full_measured_HALLMARK_")]}
        methods = {"covariates_only": cov, "covariates_plus_cNMF": cov + cnmf}
        for name, columns in groups.items():
            if not columns: raise ValueError(f"Empty method features: {name}")
            methods["covariates_plus_" + name] = cov + columns
            methods["covariates_plus_" + name + "_plus_cNMF"] = cov + columns + cnmf
        return methods

    def benchmark(X, Y, metadata, dataset, gate, split, classification):
        methods = feature_methods(X)
        input_signature = stable_token({key: pd.util.hash_pandas_object(frame, index=True).astype(str).tolist()
                                       for key, frame in {"X": X, "Y": Y, "metadata": metadata}.items()})
        results = []
        for heldout in sorted(metadata[split].astype(str).unique()):
            train = metadata.index[metadata[split].astype(str).ne(heldout)]
            test = metadata.index[metadata[split].astype(str).eq(heldout)]
            legacy_label = f"{dataset}/benchmark/{gate}/{split}/{heldout}/covariates_only"
            candidates = [u for u in prior["completed_units"] if u["label"] == legacy_label]
            if len(candidates) != 1: raise ValueError("Historical comparison unit is absent/ambiguous")
            previous = candidates[0]; previous_path = Path(previous["path"])
            if not validate_marker(previous_path / "scoring.complete.json", previous["signature"]):
                raise ValueError("Historical benchmark checkpoint is absent")
            previous_predictions = pd.read_parquet(previous_path / "predictions.parquet")
            _iv_same_rows(previous_predictions.index, test)
            if not previous_predictions.donor.astype(str).reindex(test).equals(metadata.loc[test, "donor"].astype(str)):
                raise ValueError("Historical test donor mapping changed")
            if classification and not np.array_equal(previous_predictions.observed.reindex(test).astype(str), Y.loc[test].iloc[:, 0].astype(str)):
                raise ValueError("Historical test stimulation labels changed")
            for method, columns in methods.items():
                specification = {"dataset": dataset, "quality_gate": gate, "split": split, "heldout": heldout,
                                 "method": method, "columns": columns, "input_signature": input_signature,
                                 "train_ids_sha256": sha256_lines(train), "test_ids_sha256": sha256_lines(test)}
                def fit_outer():
                    selected, inner = _iv_select(X[columns], Y, metadata, split, heldout, classification,
                        grids["classification_C" if classification else "regression_alpha"])
                    predictions, metrics, audit = _iv_fit(X[columns], Y, metadata, train, test, classification, selected)
                    for frame in (predictions, metrics, inner, audit):
                        for key in ("dataset", "quality_gate", "split", "heldout", "method"):
                            frame[key] = specification[key]
                        frame["selected_penalty"] = selected
                    metrics["n_features"] = len(columns)
                    return {"predictions": predictions, "donor_metrics": metrics, "inner_losses": inner, "training_scaling": audit}
                frames = unit(f"{dataset}/benchmark/{gate}/{split}/{heldout}/{method}", specification,
                              {"predictions": ["donor", "method"], "donor_metrics": ["donor", "loss"],
                               "inner_losses": ["inner_heldout", "penalty", "loss"],
                               "training_scaling": ["feature", "center", "scale"]}, fit_outer)
                if not frames["predictions"].index.equals(test): raise ValueError("Different test rows across methods")
                results.append(frames["donor_metrics"])
        return pd.concat(results, ignore_index=True)

    benchmark_tables, denominators = {}, []
    for dataset, frames in external.items():
        name = f"{dataset}_primary_K8_incremental_value_validation"
        save_table(frames["coverage"], name + "_coverage")
        X = pd.concat([frames["usage_normalized"].add_prefix("cNMF_"),
                       frames["rank_signature_scores"].add_prefix("rank_signature_"),
                       frames["AUCell_scores"].add_prefix("AUCell_"),
                       frames["full_measured_AUCell_scores"].add_prefix("full_measured_"),
                       frames["frozen_AUCell_scores"].add_prefix("frozen_")], axis=1)
        meta = frames["cell_metadata"].copy()
        if not X.index.is_unique or not X.index.equals(meta.index): raise ValueError("Duplicated/unmatched cells")
        X["covariate_log1p_library_size"] = np.log1p(frames["quality"].library_over_training_universe)
        tables = []
        for gate in ("all_cells", "Gray_q95_accepted"):
            if dataset == "GSE268991":
                eligibility = _strict_multimodal_eligibility(frames["gamma_delta_TCR_metadata"], frames["ADT_raw_and_CLR"])
                Y = frames["ADT_raw_and_CLR"][["ADT_CLR__CD45RA", "ADT_CLR__CCR7"]]
                mask = eligibility.single_productive_TRG_TRD & eligibility.has_ADT_counts & Y.notna().all(axis=1)
                X["covariate_condition"] = meta.condition
            else: mask = meta.Allergic.eq("NonA")
            before = mask.copy()
            if gate != "all_cells": mask &= frames["quality"].within_Gray_q95
            counts = pd.DataFrame({"eligible_before_gate": before, "selected_after_gate": mask,
                                  "donor": meta.donor, "condition": meta.condition}).groupby(["donor", "condition"]).sum().reset_index()
            counts["dataset"] = dataset; counts["quality_gate"] = gate; denominators.append(counts)
            if dataset == "GSE268991":
                ids = X.index[mask]
                if meta.loc[ids, "donor"].nunique() != 7: raise ValueError("Protein donor cohort changed")
                tables.append(benchmark(X.loc[ids], Y.loc[ids], meta.loc[ids], dataset, gate, "donor", False))
            else:
                selected = X.loc[mask].join(meta.loc[mask, ["donor", "Experiment", "Stim"]])
                grouped = selected.groupby(["donor", "Experiment", "Stim"], observed=True)
                scores = grouped[X.columns].median().loc[grouped.size().ge(20)]
                stimuli = scores.reset_index().groupby("donor").Stim.agg(set)
                donors = stimuli.index[stimuli.map(lambda x: x == {"CR", "MO", "HDMAPP"})]
                scores = scores.loc[scores.index.get_level_values("donor").isin(donors)].reset_index()
                metadata = scores[["donor", "Experiment", "Stim"]].copy()
                if metadata.donor.nunique() != 15 or metadata.Experiment.nunique() != 4:
                    raise ValueError("Stimulation donor/Experiment cohort changed")
                if metadata.groupby("donor").Experiment.nunique().max() != 1: raise ValueError("Donor crosses Experiments")
                aggregate = scores[X.columns].copy()
                aggregate["covariate_Experiment"] = metadata.Experiment
                save_table(scores, name + f"_{gate}_donor_scores")
                for split in ("donor", "Experiment"):
                    tables.append(benchmark(aggregate, scores[["Stim"]], metadata, dataset, gate, split, True))
        benchmark_tables[dataset] = pd.concat(tables, ignore_index=True)
        save_table(benchmark_tables[dataset], name + "_benchmark")
    save_table(pd.concat(denominators, ignore_index=True), prefix + "_denominators")
    if len(progress["completed_units"]) != progress["expected_units"]: raise ValueError("Incomplete required units")
    for dataset, results in benchmark_tables.items():
        name = f"{dataset}_primary_K8_incremental_value_validation"
        paired, summaries = [], []
        for (gate, split), part in results.groupby(["quality_gate", "split"]):
            wide = part.pivot(index="donor", columns="method", values="loss")
            if wide.isna().any().any(): raise ValueError("Unpaired model comparisons")
            bases = ["covariates_only", "covariates_plus_rank_signature", "covariates_plus_AUCell",
                     "covariates_plus_full_measured_AUCell", "covariates_plus_Hallmark_AUCell",
                     "covariates_plus_full_measured_Hallmark_AUCell"]
            for baseline in bases:
                models = ["covariates_plus_cNMF"]
                if baseline != "covariates_only": models.append(baseline + "_plus_cNMF")
                for method in models:
                    delta = wide[baseline] - wide[method]
                    group_metadata = part[["donor"] + (["Experiment"] if "Experiment" in part else [])].drop_duplicates().set_index("donor")
                    donor_frame = delta.rename("error_reduction").to_frame().join(group_metadata).reset_index()
                    for key, value in {"dataset": dataset, "quality_gate": gate, "split": split,
                                       "model": method, "baseline": baseline}.items(): donor_frame[key] = value
                    paired.append(donor_frame)
                    units = (donor_frame.groupby("Experiment").error_reduction.mean() if split == "Experiment" else delta)
                    rng = np.random.default_rng(20261023)
                    boot = units.to_numpy()[rng.integers(0, len(units), size=(4000, len(units)))].mean(axis=1)
                    summaries.append({"dataset": dataset, "quality_gate": gate, "split": split,
                        "model": method, "baseline": baseline, "n_donors": len(delta), "n_resampling_units": len(units),
                        "resampling_unit": split, "mean_error_reduction": units.mean(), "n_units_improved": int((units > 0).sum()),
                        "conditional_95CI_low": np.quantile(boot, .025), "conditional_95CI_high": np.quantile(boot, .975),
                        "interval_scope": "conditional resampling of held-out loss units; models not refitted; exploratory, no multiplicity-adjusted claim"})
        paired_frame = pd.concat(paired, ignore_index=True)
        save_table(paired_frame, name + "_paired_losses")
        save_table(pd.DataFrame(summaries), name + "_incremental_value")
        legend_rows = []
        for gate, split in results[["quality_gate", "split"]].drop_duplicates().itertuples(index=False, name=None):
            figure_name = name + f"_{gate}_{split}_added_value"
            fig, source = _iv_plot_added_value(paired_frame, dataset, gate, split, FIGURE_FONT)
            save_table(source, figure_name + "_source_data")
            # Exact physical page width; all plotted marks and text remain vector.
            with mpl.rc_context({"savefig.bbox": None, "pdf.fonttype": 42, "svg.fonttype": "none"}):
                fig.savefig(FIGURE_DIR / f"{figure_name}.pdf", bbox_inches=None)
                fig.savefig(FIGURE_DIR / f"{figure_name}.png", bbox_inches=None, dpi=300)
            display(fig); plt.close(fig)
            legend_rows.append({"figure": figure_name, "n": source.unit.nunique(),
                "unit": "Experiment (equal mean of donor losses within Experiment)" if split == "Experiment" else "donor",
                "center": "equal-unit arithmetic mean", "spread": "all individual unit effects shown; no inferential error bars",
                "comparison": "same eligible observations and same nested grouped parameter-selection budget; baseline versus baseline plus cNMF",
                "metric": "joint standardized MSE for CD45RA/CCR7" if dataset == "GSE268991" else "multiclass log loss for CR/MO/HDMAPP",
                "test_and_multiplicity": "descriptive development comparison; no significance stars or confirmatory P values",
                "source_data": figure_name + "_source_data.csv",
                "figure_QA": "single panel: alignment not applicable; rendered collision/font QA must be completed before submission",
                "submission_status": "analysis figure; NI Excel source-data packaging and final manuscript numbering remain separate submission tasks"})
        save_table(pd.DataFrame(legend_rows), name + "_figure_legends")
    if frozen != {k: sha256sum(scorer[k]) for k in frozen}: raise ValueError("Frozen artifacts changed")
    if sha256sum(GSE268991_PREANALYSIS) != config["preanalysis_sha256"]: raise ValueError("Preanalysis changed")
    progress.update(status="completed", active_unit=None,
                    limitations=["Previously examined cohorts; not untouched final validation", "Only 7 protein donors and 4 stimulation Experiments",
                                 "RNA-cell-state and stimulation discrimination are not killing-function validation",
                                 "Full measured universe uses annotation availability across libraries, not test expression/outcome",
                                 "Conditional bootstrap intervals ignore model-refitting uncertainty", "No universal superiority or equivalence claim"],
                    outputs={str(p.relative_to(PROJECT)): sha256sum(p) for parent, ext in ((TABLE_DIR, "csv"), (FIGURE_DIR, "pdf"))
                             for p in parent.glob(f"*_primary_K8_incremental_value_validation_*.{ext}")})
    flush()
    for dataset, frame in benchmark_tables.items():
        print(dataset, "equal-donor mean losses (lower is better)")
        display(frame.groupby(["quality_gate", "split", "method"]).loss.mean().unstack("method"))
    print("Completed incremental_value_validation. Interpret paired losses before claiming added value.")

def run_incremental_value_validation():
    context = {}
    try:
        return _run_incremental_value_validation(context)
    except Exception as error:
        if context:
            context["progress"].update(status="failed", error=f"{type(error).__name__}: {error}")
            context["flush"]()
        raise

if EXECUTION_STAGE == "incremental_value_validation":
    run_incremental_value_validation()
else:
    print("Incremental value validation not launched in stage", EXECUTION_STAGE)
'''


ROBUSTNESS_CODE = r'''
def run_robustness_validation():
    """Independent, resumable sensitivity analyses; never refit the frozen model."""
    import yaml
    from cnmf.cnmf import load_df_from_npz
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.metrics import log_loss, balanced_accuracy_score
    import numba
    numba.set_num_threads(max(1, min(THREADS, numba.config.NUMBA_NUM_THREADS)))

    scorer = load_frozen_scorer_for_gse268991()
    if len(GSE268991_SAMPLES) != 14 or raw_inventory.library.nunique() != 13:
        raise RuntimeError("Expected all 14 GSE268991 samples and 13 GSE294268 libraries")
    preanalysis_hash = sha256sum(GSE268991_PREANALYSIS)
    if preanalysis_hash != GSE268991_PREANALYSIS_SHA256:
        raise RuntimeError("Original preanalysis manifest changed")
    preanalysis = json.loads(GSE268991_PREANALYSIS.read_text())
    q95 = float(preanalysis["frozen_model"]["primary_quality_gate"]["maximum"])
    q90 = float(preanalysis["frozen_model"]["strict_sensitivity_gate"]["maximum"])
    programs = ["P1", "P2", "P3", "P4", "P5", "P7", "P9", "P10"]
    annotation_object = ad.read_h5ad(GRAY_PATH, backed="r")
    training_var = annotation_object.raw.var.copy()
    annotation_object.file.close()
    frozen_hashes = {key: sha256sum(scorer[key]) for key in (
        "spec_path", "spectra_path", "sd_path", "ranked_path", "params_path")}
    run_spec = {
        "stage": "robustness_validation", "schema_version": 1,
        "source_sha256": sha256sum(PROJECT / "build_19b_cNMF_valdiation.py"),
        "Gray_sha256": gray_record["sha256"],
        "cohort_sha256": source_records["Gray cohort"]["sha256"],
        "frozen_model": frozen_hashes, "preanalysis_sha256": preanalysis_hash,
        "depth_fractions": [0.5, 0.25, 0.1], "replicates": [0, 1, 2],
        "random_missing_fractions": [0.1, 0.2], "seed": 20260919,
        "q90": q90, "q95": q95, "threads": THREADS,
        "versions": {name: im.version(name) for name in
                     ["numpy", "scipy", "scikit-learn", "cnmf", "decoupler", "anndata"]},
        "classification_C": 1.0, "regression_alpha": 1.0,
        "evidence": "post-evaluation sensitivity; no new untouched cohort",
    }
    run_id = stable_token(run_spec)
    root = CHECKPOINT_DIR / "robustness_validation" / run_id
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / "job_spec.json", run_spec)
    manifest_path = TABLE_DIR / "Gray_primary_K8_robustness_validation_manifest.json"
    progress = {"run_id": run_id, "configuration": run_spec, "status": "running",
                "checkpoint_root": str(root), "completed_units": [],
                "functional_validation": "candidate protein provenance pending; no killing endpoint"}

    def flush_progress():
        progress["updated_UTC"] = datetime.now(timezone.utc).isoformat()
        atomic_json(manifest_path, progress)

    def unit(label, specification, schemas, produce):
        signature = stable_token({"run_id": run_id, "specification": specification})
        path = root / label / signature
        path.mkdir(parents=True, exist_ok=True)
        marker = path / "scoring.complete.json"
        atomic_json(path / "job_spec.json", specification)
        reused = marker.exists()
        if reused:
            validate_marker(marker, signature)
            payload = json.loads(marker.read_text())
            if set(payload.get("outputs", {})) != {f"{name}.parquet" for name in schemas}:
                raise ValueError(f"Incomplete checkpoint output schema: {path}")
            frames = {name: pd.read_parquet(path / f"{name}.parquet") for name in schemas}
        else:
            try:
                frames = produce()
            except Exception as error:
                progress["status"] = "failed"
                progress["failed_unit"] = label
                progress["error"] = f"{type(error).__name__}: {error}"
                flush_progress()
                raise
        if set(frames) != set(schemas):
            raise ValueError(f"Unexpected checkpoint frames: {label}")
        for name, columns in schemas.items():
            if not set(columns) <= set(frames[name].columns) or not frames[name].index.is_unique:
                raise ValueError(f"Invalid checkpoint schema/index: {label}/{name}")
        if not reused:
            outputs = [atomic_parquet(frame, path / f"{name}.parquet")
                       for name, frame in frames.items()]
            write_marker(marker, signature, outputs)
        progress["completed_units"].append({"label": label, "signature": signature,
                                            "path": str(path), "reused": reused})
        flush_progress()
        print(f"robustness_validation {len(progress['completed_units'])}: {label}", flush=True)
        return frames

    def record_inputs(paths):
        return {name: {"path": str(path), "sha256": sha256sum(path)}
                for name, path in paths.items()}

    def coverage_rows(features, mapping, model, dataset, library):
        present = np.asarray(mapping.sum(axis=0)).ravel() > 0
        observed = present[model["hvg_positions"]]
        old = model["hvgs"].isin(features.feature_name.astype(str))
        rows = []
        for index, component in enumerate(model["components"]):
            weights = np.asarray(model["basis"])[index]
            signatures = model["signatures"][component]
            positions = model["universe"].get_indexer(signatures)
            signature_coverage = np.mean([p >= 0 and present[p] for p in positions])
            rows.append({"dataset": dataset, "library": library,
                         "program": model["component_to_program"][component],
                         "symbol_only_HVG_coverage": float(old.mean()),
                         "aligned_HVG_coverage": float(observed.mean()),
                         "top50_signature_coverage": float(signature_coverage),
                         "scaled_basis_L1_coverage": float(weights[observed].sum() / weights.sum()),
                         "scaled_basis_squared_L2_coverage": float(
                             (weights[observed] ** 2).sum() / (weights ** 2).sum()),
                         "missing_HVGs": ";".join(model["hvgs"][~observed])})
        return pd.DataFrame(rows)

    # Validate old checkpoints without writing to them. They define the original
    # cell population, raw-barcode crosswalk and pre-existing paired metadata.
    external = {}
    for dataset in ("GSE268991", "GSE294268"):
        containers = {}
        samples = list(GSE268991_SAMPLES) if dataset == "GSE268991" else raw_inventory.library.tolist()
        for sample in samples:
            if dataset == "GSE268991":
                paths = gse268991_sample_paths(sample)
                legacy = GSE268991_PROJECTION_CHECKPOINTS / sample
            else:
                inventory_row = raw_inventory.set_index("library").loc[sample]
                paths = {"matrix": Path(inventory_row.matrix_path),
                         "features": Path(inventory_row.features_path),
                         "barcodes": Path(inventory_row.barcodes_path)}
                legacy = GSE_PROJECTION_CHECKPOINTS / sample
            old_spec = json.loads((legacy / "job_spec.json").read_text())
            if not validate_marker(legacy / "scoring.complete.json", stable_token(old_spec)):
                raise RuntimeError(f"Missing historical library checkpoint: {legacy}")
            old_usage = pd.read_parquet(legacy / "usage_normalized.parquet")
            old_usage.columns = old_usage.columns.astype(str)
            ids = old_usage.index.astype(str)
            old_quality = pd.read_parquet(legacy / "quality.parquet").loc[ids]
            features = pd.read_csv(paths["features"], sep="\t", header=None)
            features = features.iloc[:, :3].copy()
            features.columns = ["feature_id", "feature_name", "feature_type"]
            gene_mask = features.feature_type.eq("Gene Expression").to_numpy()
            genes = features.loc[gene_mask].reset_index(drop=True)
            mapping, audit = _align_training_features(genes, training_var, scorer["universe"])
            coverage = coverage_rows(genes, mapping, scorer, dataset, sample)
            if coverage.aligned_HVG_coverage.min() < MIN_HVG_COVERAGE:
                raise ValueError(f"Corrected feature coverage is below the frozen gate: {sample}")
            spec = {"dataset": dataset, "library": sample, "inputs": record_inputs(paths),
                    "legacy_marker_sha256": sha256sum(legacy / "scoring.complete.json"),
                    "cell_ids_sha256": sha256_lines(ids), "n_cells": len(ids),
                    "mapping": "training Ensembl first; exact-symbol fallback for unknown IDs",
                    "signatures": scorer["signatures"], "AUCell_n_up_fraction": 0.05}
            schemas = {"usage_normalized": programs, "usage_raw": programs,
                       "rank_signature_scores": programs, "AUCell_scores": programs,
                       "quality": ["donor", "scaled_HVG_relative_error", "within_Gray_q95"],
                       "cell_metadata": ["donor", "condition"],
                       "feature_mapping": ["feature_id", "training_feature"],
                       "coverage": ["program", "aligned_HVG_coverage"]}
            if dataset == "GSE268991":
                schemas.update({"gamma_delta_TCR_metadata": ["n_productive_TRG", "n_productive_TRD"],
                                "ADT_raw_and_CLR": ["ADT_total_counts"]})

            def produce_external():
                with gzip.open(paths["matrix"], "rb") as handle:
                    matrix = scipy.io.mmread(handle).tocsr()
                barcodes = pd.Index(pd.read_csv(paths["barcodes"], sep="\t", header=None).iloc[:, 0].astype(str))
                if not barcodes.is_unique or matrix.shape != (len(features), len(barcodes)):
                    raise ValueError(f"Raw library dimensions/IDs disagree: {sample}")
                raw_ids = (pd.Index([f"{sample}:{x}" for x in barcodes]) if dataset == "GSE268991"
                           else pd.Index(barcodes + f"_{old_spec['barcode_suffix_number']}"))
                selected = raw_ids.get_indexer(ids)
                if (selected < 0).any():
                    raise ValueError(f"Released-cell crosswalk failed: {sample}")
                counts = (matrix[gene_mask, :].T[selected] @ mapping).tocsr()
                raw, usage, quality = _project_validation_counts(counts, scorer, 20260919, THREADS)
                quality.index = ids
                usage = pd.DataFrame(usage, index=ids, columns=scorer["components"]).rename(
                    columns=scorer["component_to_program"])[programs]
                raw = pd.DataFrame(raw, index=ids, columns=scorer["components"]).rename(
                    columns=scorer["component_to_program"])[programs]
                rank = rank_scores(counts, scorer["universe"], scorer["signatures"])
                auc = _aucell_validation_scores(counts, scorer["universe"], scorer["signatures"])
                rank.index = auc.index = ids
                rank = rank.rename(columns=scorer["component_to_program"])[programs]
                auc = auc.rename(columns=scorer["component_to_program"])[programs]
                if not np.allclose(usage.sum(axis=1), 1, atol=1e-8):
                    raise RuntimeError(f"Invalid projected usage: {sample}")
                if dataset == "GSE268991":
                    metadata = pd.read_parquet(legacy / "cell_metadata.parquet").loc[ids].copy()
                else:
                    metadata = annotated.loc[ids, ["Donor.ID", "Stim", "Experiment", "Allergic"]].copy()
                    metadata["donor"] = metadata["Donor.ID"].astype(str)
                    metadata["condition"] = metadata["Stim"].astype(str)
                quality["donor"] = metadata.donor
                quality["condition"] = metadata.condition
                quality["library"] = sample
                quality["within_Gray_q95"] = quality.scaled_HVG_relative_error.le(q95)
                quality["within_Gray_q90"] = quality.scaled_HVG_relative_error.le(q90)
                quality["old_relative_error"] = old_quality.scaled_HVG_relative_error
                quality["old_within_Gray_q95"] = old_quality.scaled_HVG_relative_error.le(q95)
                quality["mean_absolute_usage_change"] = np.abs(usage - old_usage.rename(
                    columns=scorer["component_to_program"])[programs]).mean(axis=1)
                result = {"usage_normalized": usage, "usage_raw": raw,
                          "rank_signature_scores": rank, "AUCell_scores": auc,
                          "quality": quality, "cell_metadata": metadata,
                          "feature_mapping": audit, "coverage": coverage}
                if dataset == "GSE268991":
                    tcr = pd.read_parquet(legacy / "gamma_delta_TCR_metadata.parquet").loc[ids].copy()
                    vdj = pd.read_csv(paths["gd_tcr"])
                    good = (boolean_series(vdj.is_cell) & boolean_series(vdj.high_confidence)
                            & boolean_series(vdj.productive))
                    for chain in ("TRG", "TRD"):
                        number = vdj.loc[good & vdj.chain.eq(chain)].groupby("barcode").size()
                        tcr[f"n_productive_{chain}"] = number.reindex(barcodes, fill_value=0).to_numpy()[selected]
                    adt = pd.read_parquet(legacy / "ADT_raw_and_CLR.parquet").loc[ids]
                    result["gamma_delta_TCR_metadata"] = tcr
                    result["ADT_raw_and_CLR"] = adt
                return result

            frames = unit(f"{dataset}/{sample}", spec, schemas, produce_external)
            for key, frame in frames.items():
                if key == "feature_mapping":
                    continue
                containers.setdefault(key, []).append(frame)
        external[dataset] = {key: pd.concat(parts, ignore_index=(key == "coverage"))
                             for key, parts in containers.items()}
        output = external[dataset]
        if not output["usage_normalized"].index.is_unique:
            raise ValueError(f"Repeated external cells after assembly: {dataset}")
        if dataset == "GSE294268" and set(output["usage_normalized"].index) != set(gse_metadata.index):
            raise ValueError("Corrected GSE294268 projection does not cover all released cells")
        save_table(output["coverage"], f"{dataset}_primary_K8_robustness_validation_coverage")
        q = output["quality"]
        donor_q = q.groupby(["donor", "condition"], observed=True).agg(
            n_cells=("within_Gray_q95", "size"),
            original_fraction_accepted=("old_within_Gray_q95", "mean"),
            corrected_fraction_accepted=("within_Gray_q95", "mean"),
            median_relative_error=("scaled_HVG_relative_error", "median"),
            median_library_size=("library_over_training_universe", "median"),
            mean_absolute_usage_change=("mean_absolute_usage_change", "mean"),
            fraction_solver_converged=("solver_converged", "mean")).reset_index()
        save_table(donor_q, f"{dataset}_primary_K8_robustness_validation_quality")
        depth_q = q.copy()
        depth_q["library_size_bin"] = pd.cut(depth_q.library_over_training_universe,
            [0, 500, 1000, 2000, 5000, 10000, np.inf], include_lowest=True).astype(str)
        depth_audit = depth_q.groupby(["donor", "condition", "library_size_bin"], observed=True).agg(
            n_cells=("within_Gray_q95", "size"),
            fraction_accepted=("within_Gray_q95", "mean"),
            median_relative_error=("scaled_HVG_relative_error", "median")).reset_index()
        save_table(depth_audit, f"{dataset}_primary_K8_robustness_validation_depth_gate_audit")
        composition = []
        for gate, mask in (("all_cells", np.ones(len(q), bool)),
                           ("Gray_q95_accepted", q.within_Gray_q95.to_numpy())):
            summaries = output["usage_normalized"].loc[mask].join(output["cell_metadata"][["donor", "condition"]])
            summaries = summaries.groupby(["donor", "condition"], observed=True)[programs].median().reset_index()
            summaries["quality_gate"] = gate
            composition.append(summaries)
        save_table(pd.concat(composition, ignore_index=True),
                   f"{dataset}_primary_K8_robustness_validation_gate_program_composition")

    # Fold-model loading and matching are diagnostic only. Full-data reference
    # alignment names the fold components; it is never used to fit held-out data.
    fold_scorers, fold_inputs, matching_rows = {}, {}, []
    frozen_tpm = pd.read_csv(scorer["spectra_path"], sep="\t", index_col=0)
    frozen_tpm.index = frozen_tpm.index.map(lambda x: str(int(float(x))))
    for fold in range(N_FOLDS):
        job = CHECKPOINT_DIR / "full" / f"fold{fold}_K8"
        old = json.loads((job / "job_spec.json").read_text())
        if not validate_marker(job / "heldout_scoring.complete.json", stable_token(old)):
            raise RuntimeError(f"Unvalidated Gray fold: {fold}")
        paths = {"spectra": one_file(job, "*gene_spectra_tpm.k_8.dt_0_01.txt"),
                 "stats": one_file(job, "*.tpm_stats.df.npz"),
                 "hvgs": one_file(job, "*.overdispersed_genes.txt"),
                 "params": one_file(job, "*.nmf_idvrun_params.yaml"),
                 "training": job / "training_counts.h5ad"}
        train = ad.read_h5ad(paths["training"], backed="r")
        universe = pd.Index(train.var_names.astype(str)); train.file.close()
        hvgs = pd.Index(paths["hvgs"].read_text().splitlines())
        spectra = pd.read_csv(paths["spectra"], sep="\t", index_col=0)
        spectra.index = spectra.index.map(lambda x: str(int(float(x))))
        sd = load_df_from_npz(paths["stats"]).loc[hvgs, "__std"].astype(float)
        common = hvgs.intersection(scorer["hvgs"])
        similarity = cosine_similarity(spectra[common], frozen_tpm[common])
        left, right = linear_sum_assignment(-similarity)
        mapping = {str(spectra.index[i]): scorer["component_to_program"][str(frozen_tpm.index[j])]
                   for i, j in zip(left, right)}
        for i, j in zip(left, right):
            matching_rows.append({"fold": fold, "component": str(spectra.index[i]),
                                  "program": mapping[str(spectra.index[i])],
                                  "cosine_on_common_HVGs": float(similarity[i, j]),
                                  "n_common_HVGs": len(common)})
        fold_scorers[fold] = {"universe": universe, "hvgs": hvgs,
                             "hvg_positions": universe.get_indexer(hvgs), "sd": sd,
                             "basis": spectra[hvgs].div(sd, axis=1).to_numpy(),
                             "solver_params": yaml.safe_load(paths["params"].read_text()),
                             "components": spectra.index, "component_to_program": mapping}
        fold_inputs[fold] = record_inputs(paths)
    save_table(pd.DataFrame(matching_rows), "Gray_primary_K8_robustness_validation_fold_matching")

    # Every donor/scenario/replicate is a separately validated durable checkpoint.
    sensitivity_parts = []
    mask_features = pd.read_csv(gse268991_sample_paths("N77")["features"], sep="\t", header=None)
    mask_features.columns = ["feature_id", "feature_name", "feature_type"]
    mask_features = mask_features.loc[mask_features.feature_type.eq("Gene Expression")]
    for donor, metadata in primary.groupby("donor_id", observed=True, sort=True):
        fold = int(metadata.outer_fold.iloc[0])
        model = fold_scorers[fold]
        counts = primary_counts[primary.index.get_indexer(metadata.index)][:, gray_genes.get_indexer(model["universe"])].tocsr()
        baseline_spec = {"donor": str(donor), "fold": fold, "inputs": fold_inputs[fold],
                         "cell_ids_sha256": sha256_lines(metadata.index), "n_cells": len(metadata),
                         "scenario": "full_depth", "seed": 20260919}
        columns = [model["component_to_program"][str(c)] for c in model["components"]]
        def project(counts, observed=None, seed=20260919):
            raw, usage, quality = _project_validation_counts(counts, model, seed, THREADS, observed)
            quality.index = metadata.index
            return {"usage_normalized": pd.DataFrame(usage, index=metadata.index, columns=columns),
                    "usage_raw": pd.DataFrame(raw, index=metadata.index, columns=columns),
                    "quality": quality}
        schema = {"usage_normalized": programs, "usage_raw": programs,
                  "quality": ["scaled_HVG_relative_error", "valid_objective", "solver_converged"]}
        baseline = unit(f"Gray/{donor}/full_depth", baseline_spec, schema, lambda: project(counts))
        symbol_mask = model["universe"].isin(mask_features.feature_name.astype(str))
        real_mapping, _ = _align_training_features(mask_features, training_var, model["universe"])
        aligned_mask = np.asarray(real_mapping.sum(axis=0)).ravel() > 0
        scenarios = [("full_depth", 1., 0, None, False)]
        scenarios += [("UMI_thinning", f, rep, None, False) for f in (.5, .25, .1) for rep in range(3)]
        for fraction in (.1, .2):
            for rep in range(3):
                rng = np.random.default_rng(20260919 + fold * 100 + rep + int(fraction * 1000))
                mask = np.ones(len(model["universe"]), bool)
                mask[rng.choice(len(mask), round(fraction * len(mask)), replace=False)] = False
                scenarios.append(("random_missing_genes", fraction, rep, mask, False))
        for name, mask in (("GSE268991_symbol_only_missing", symbol_mask),
                           ("GSE268991_aligned_missing", aligned_mask)):
            scenarios.extend([(name, np.nan, 0, mask, False), (name, np.nan, 0, mask, True)])
        for name, fraction, rep, mask, observed_fit in scenarios:
            scenario_id = f"{name}_{fraction}_{rep}_{int(observed_fit)}"
            specification = {**baseline_spec, "scenario": name, "fraction": fraction,
                             "replicate": rep, "observed_feature_fit": observed_fit,
                             "mask_sha256": sha256_lines(mask) if mask is not None else None}
            def perturb():
                seed = 20260919 + rep + fold * 100
                data = _thin_raw_counts(counts, fraction, seed) if name == "UMI_thinning" else counts
                if mask is not None:
                    data = data.multiply(mask).tocsr()
                    data.eliminate_zeros()
                fitted = (baseline if name == "full_depth" else
                          project(data, mask[model["hvg_positions"]] if observed_fit else None, seed))
                quality = fitted["quality"]
                rows = []
                for program in programs:
                    ref = baseline["usage_normalized"][program]
                    candidate = fitted["usage_normalized"][program]
                    delta = candidate - ref
                    raw_ref = baseline["usage_raw"][program]
                    raw_delta = fitted["usage_raw"][program] - raw_ref
                    rows.append({"donor": str(donor), "fold": fold, "program": program,
                                 "scenario": name, "fraction": fraction, "replicate": rep,
                                 "observed_feature_fit": observed_fit, "n_cells": len(candidate),
                                 "n_valid_cells": int(candidate.notna().sum()),
                                 "usage_spearman_rho": rho(ref, candidate),
                                 "usage_MAE": float(delta.abs().mean()),
                                 "usage_signed_mean_change": float(delta.mean()),
                                 "raw_usage_MAE": float(raw_delta.abs().mean()),
                                 "raw_usage_MAE_over_baseline_mean": float(
                                     raw_delta.abs().mean() / max(raw_ref.mean(), 1e-15)),
                                 "median_relative_error": float(quality.scaled_HVG_relative_error.median()),
                                 "fraction_above_q95": (np.nan if observed_fit else float(
                                     quality.scaled_HVG_relative_error.gt(q95).mean())),
                                 "fraction_solver_converged": float(quality.solver_converged.mean()),
                                 "fraction_invalid_objective": float((~quality.valid_objective).mean()),
                                 "n_tissues": metadata.tissue.nunique()})
                return {"donor_metrics": pd.DataFrame(rows), **fitted}
            result = unit(f"Gray/{donor}/{scenario_id}", specification,
                          {"donor_metrics": ["donor", "program", "usage_MAE"], **schema}, perturb)
            sensitivity_parts.append(result["donor_metrics"])
    sensitivity = pd.concat(sensitivity_parts, ignore_index=True)
    save_table(sensitivity, "Gray_primary_K8_robustness_validation_donor_metrics")

    # Fixed-basis fold sensitivity is scored on the SAME external target cells.
    # Overlapping training folds are not independent biological replicates.
    ensemble_parts = []
    for sample in GSE268991_SAMPLES:
        paths = gse268991_sample_paths(sample)
        frame = external["GSE268991"]
        ids = frame["cell_metadata"].index[frame["cell_metadata"].index.str.startswith(f"{sample}:")]
        eligible = _strict_multimodal_eligibility(frame["gamma_delta_TCR_metadata"].loc[ids],
                                                 frame["ADT_raw_and_CLR"].loc[ids])
        ids = ids[eligible.single_productive_TRG_TRD]
        if not len(ids):
            continue
        # Read only once per sample for five small fold projections.
        features = pd.read_csv(paths["features"], sep="\t", header=None,
                               names=["feature_id", "feature_name", "feature_type"])
        mask = features.feature_type.eq("Gene Expression").to_numpy()
        genes = features.loc[mask].reset_index(drop=True)
        barcodes = pd.read_csv(paths["barcodes"], header=None).iloc[:, 0].astype(str)
        positions = pd.Index([f"{sample}:{b}" for b in barcodes]).get_indexer(ids)
        with gzip.open(paths["matrix"], "rb") as handle:
            source_counts = scipy.io.mmread(handle).tocsr()[mask, :].T[positions].tocsr()
        for fold, model in fold_scorers.items():
            spec = {"sample": sample, "fold": fold, "fold_inputs": fold_inputs[fold],
                    "raw_inputs": record_inputs(paths), "cell_ids_sha256": sha256_lines(ids)}
            def fold_projection():
                mapping, _ = _align_training_features(genes, training_var, model["universe"])
                _, u, _ = _project_validation_counts(source_counts @ mapping, model, 20260919, THREADS)
                usage = pd.DataFrame(u, index=ids, columns=model["components"]).rename(
                    columns=model["component_to_program"])[programs]
                usage["fold"] = fold
                usage["donor"] = frame["cell_metadata"].loc[ids, "donor"]
                usage["cell_id"] = ids
                return {"usage_normalized": usage}
            result = unit(f"GSE268991/{sample}/fold{fold}", spec,
                          {"usage_normalized": [*programs, "fold", "cell_id"]}, fold_projection)
            ensemble_parts.append(result["usage_normalized"])
    ensemble = pd.concat(ensemble_parts, ignore_index=True)
    spread = ensemble.groupby("cell_id")[programs].agg(lambda x: x.max() - x.min())
    spread["donor"] = external["GSE268991"]["cell_metadata"].loc[spread.index, "donor"]
    save_table(spread.groupby("donor")[programs].median().reset_index(),
               "GSE268991_primary_K8_robustness_validation_fold_spread")

    def methods(frame):
        cov = [c for c in frame.columns if c.startswith("covariate_")]
        cnmf = [c for c in frame.columns if c.startswith("cNMF_")]
        rank = [c for c in frame.columns if c.startswith("rank_signature_")]
        auc = [c for c in frame.columns if c.startswith("AUCell_")]
        return {"covariates_only": cov, "cNMF": cnmf, "rank_signature": rank, "AUCell": auc,
                "covariates_plus_cNMF": cov + cnmf,
                "covariates_plus_rank_signature": cov + rank,
                "covariates_plus_AUCell": cov + auc,
                "covariates_plus_rank_signature_plus_cNMF": cov + rank + cnmf,
                "covariates_plus_AUCell_plus_cNMF": cov + auc + cnmf}

    def benchmark(features, outcomes, metadata, classification, dataset, gate, split):
        frames = []
        input_signature = stable_token({
            "features": pd.util.hash_pandas_object(features, index=True).astype(str).tolist(),
            "outcomes": pd.util.hash_pandas_object(outcomes, index=True).astype(str).tolist(),
            "metadata": pd.util.hash_pandas_object(metadata, index=True).astype(str).tolist()})
        for heldout in sorted(metadata[split].astype(str).unique()):
            test = metadata[split].astype(str).eq(heldout)
            train = ~test
            if metadata.loc[train, "donor"].nunique() < 2:
                raise ValueError("Too few training donors")
            for method, columns in methods(features).items():
                spec = {"dataset": dataset, "gate": gate, "split": split, "heldout": heldout,
                        "method": method, "input_signature": input_signature, "features": columns,
                        "classification": classification, "regularization": 1.0}
                def fit_predict():
                    numeric = [c for c in columns if pd.api.types.is_numeric_dtype(features[c])]
                    categorical = [c for c in columns if c not in numeric]
                    preprocess = ColumnTransformer(
                        [("numeric", StandardScaler(), numeric),
                         ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical)],
                        remainder="drop")
                    estimator = (LogisticRegression(C=1., max_iter=5000, solver="lbfgs", random_state=20260919)
                                 if classification else Ridge(alpha=1.))
                    model = Pipeline([("preprocess", preprocess), ("estimator", estimator)])
                    train_donors = metadata.loc[train, "donor"]
                    weights = train_donors.map(1 / train_donors.value_counts()).to_numpy(float)
                    weights *= len(weights) / weights.sum()
                    if classification:
                        ytrain = outcomes.loc[train].iloc[:, 0]
                        if ytrain.nunique() != 3:
                            raise ValueError("Stimulation training split lacks one of CR/MO/HDMAPP")
                        model.fit(features.loc[train], ytrain, estimator__sample_weight=weights)
                        predictions = model.predict_proba(features.loc[test])
                        classes = model.named_steps["estimator"].classes_
                    else:
                        center = np.average(outcomes.loc[train].to_numpy(), axis=0, weights=weights)
                        scale = np.sqrt(np.average((outcomes.loc[train].to_numpy() - center) ** 2,
                                                   axis=0, weights=weights))
                        if not np.isfinite(scale).all() or (scale < 1e-12).any():
                            raise ValueError("Protein outcome has no training variation")
                        model.fit(features.loc[train], (outcomes.loc[train] - center) / scale,
                                  estimator__sample_weight=weights)
                        predictions = model.predict(features.loc[test])
                    records, pred_frames = [], []
                    tested_ids = features.index[test]
                    for donor, part in metadata.loc[test].groupby("donor", observed=True):
                        positions = tested_ids.get_indexer(part.index)
                        truth = outcomes.loc[part.index]
                        pred = predictions[positions]
                        record = {"dataset": dataset, "quality_gate": gate, "split": split,
                                  "heldout": heldout, "donor": str(donor), "method": method,
                                  "n_observations": len(part), "n_training_donors": train_donors.nunique()}
                        if classification:
                            record["log_loss"] = log_loss(truth.iloc[:, 0], pred, labels=classes)
                            record["balanced_accuracy"] = balanced_accuracy_score(
                                truth.iloc[:, 0], classes[pred.argmax(axis=1)])
                            encoded = (truth.iloc[:, 0].to_numpy()[:, None] == classes[None, :]).astype(float)
                            record["multiclass_Brier_score"] = float(np.mean(np.sum((pred - encoded) ** 2, axis=1)))
                            pframe = pd.DataFrame(pred, index=part.index, columns=[f"probability_{c}" for c in classes])
                            pframe["observed"] = truth.iloc[:, 0]
                        else:
                            target = (truth.to_numpy() - center) / scale
                            errors = np.mean((pred - target) ** 2, axis=0)
                            record["standardized_MSE"] = float(errors.mean())
                            for endpoint, error in zip(outcomes.columns, errors):
                                record[f"MSE__{endpoint}"] = float(error)
                            pframe = pd.DataFrame(pred, index=part.index,
                                                  columns=[f"predicted_standardized__{c}" for c in outcomes.columns])
                            for j, endpoint in enumerate(outcomes.columns):
                                pframe[f"observed_standardized__{endpoint}"] = target[:, j]
                        records.append(record)
                        pframe["donor"] = str(donor); pframe["method"] = method
                        pred_frames.append(pframe)
                    return {"donor_metrics": pd.DataFrame(records), "predictions": pd.concat(pred_frames)}
                result = unit(f"{dataset}/benchmark/{gate}/{split}/{heldout}/{method}", spec,
                              {"donor_metrics": ["donor", "method"], "predictions": ["donor", "method"]}, fit_predict)
                frames.append(result["donor_metrics"])
        return pd.concat(frames, ignore_index=True)

    benchmark_tables = {}
    for dataset, frame in external.items():
        features = pd.concat([frame["usage_normalized"].add_prefix("cNMF_"),
                              frame["rank_signature_scores"].add_prefix("rank_signature_"),
                              frame["AUCell_scores"].add_prefix("AUCell_")], axis=1)
        features["covariate_log1p_library_size"] = np.log1p(frame["quality"].library_over_training_universe)
        metadata = frame["cell_metadata"].copy()
        tables = []
        if dataset == "GSE268991":
            eligibility = _strict_multimodal_eligibility(frame["gamma_delta_TCR_metadata"], frame["ADT_raw_and_CLR"])
            eligibility = eligibility.join(metadata[["donor", "condition"]])
            save_table(eligibility.groupby(["donor", "condition"]).agg(
                n_cells=("single_productive_TRG_TRD", "size"),
                n_single_productive_TRG_TRD=("single_productive_TRG_TRD", "sum"),
                n_nonzero_ADT=("has_ADT_counts", "sum")).reset_index(),
                "GSE268991_primary_K8_robustness_validation_eligibility")
            endpoints = [c for c in frame["ADT_raw_and_CLR"] if c.startswith("ADT_CLR__")
                         and ("CD45RA" in c.upper() or "CCR7" in c.upper())]
            if len(endpoints) != 2:
                raise ValueError(f"Expected released CD45RA/CCR7 protein endpoints; found {endpoints}")
            outcomes = frame["ADT_raw_and_CLR"][endpoints]
            features["covariate_condition"] = metadata.condition
            for gate in ("all_cells", "Gray_q95_accepted"):
                mask = eligibility.single_productive_TRG_TRD & eligibility.has_ADT_counts
                if gate != "all_cells": mask &= frame["quality"].within_Gray_q95
                ids = features.index[mask & outcomes.notna().all(axis=1)]
                tables.append(benchmark(features.loc[ids], outcomes.loc[ids], metadata.loc[ids],
                                        False, dataset, gate, "donor"))
        else:
            # Aggregate BEFORE learning. Each row is donor x Experiment x stimulus,
            # never a cell counted as an independent stimulation experiment.
            for gate in ("all_cells", "Gray_q95_accepted"):
                mask = metadata.Allergic.eq("NonA")
                if gate != "all_cells": mask &= frame["quality"].within_Gray_q95
                selected = features.loc[mask].join(metadata.loc[mask, ["donor", "Experiment", "Stim"]])
                grouped = selected.groupby(["donor", "Experiment", "Stim"], observed=True)
                scores = grouped[features.columns].median()
                sizes = grouped.size()
                scores = scores.loc[sizes.ge(20)]
                complete = scores.reset_index().groupby("donor").Stim.agg(set)
                valid_donors = complete.index[complete.map(lambda x: x == {"CR", "MO", "HDMAPP"})]
                scores = scores.loc[scores.index.get_level_values("donor").isin(valid_donors)].reset_index()
                meta = scores[["donor", "Experiment", "Stim"]].copy()
                outcomes = scores[["Stim"]]
                aggregate_features = scores[features.columns].copy()
                aggregate_features["covariate_Experiment"] = meta.Experiment
                save_table(scores, f"GSE294268_primary_K8_robustness_validation_{gate}_donor_scores")
                for split in ("donor", "Experiment"):
                    tables.append(benchmark(aggregate_features, outcomes, meta, True, dataset, gate, split))
        benchmark_tables[dataset] = pd.concat(tables, ignore_index=True)
        save_table(benchmark_tables[dataset], f"{dataset}_primary_K8_robustness_validation_benchmark")
        metric = "standardized_MSE" if dataset == "GSE268991" else "log_loss"
        contrasts = []
        for (gate, split), part in benchmark_tables[dataset].groupby(["quality_gate", "split"]):
            wide = part.pivot(index="donor", columns="method", values=metric)
            pairs = [("covariates_plus_cNMF", "covariates_plus_rank_signature"),
                     ("covariates_plus_cNMF", "covariates_plus_AUCell"),
                     ("covariates_plus_rank_signature_plus_cNMF", "covariates_plus_rank_signature"),
                     ("covariates_plus_AUCell_plus_cNMF", "covariates_plus_AUCell")]
            for model, baseline_name in pairs:
                delta = (wide[baseline_name] - wide[model]).dropna().to_numpy()
                rng = np.random.default_rng(20260919)
                bootstrap = delta[rng.integers(0, len(delta), (4000, len(delta)))].mean(axis=1)
                contrasts.append({"quality_gate": gate, "split": split, "model": model,
                                  "baseline": baseline_name, "metric": metric, "n_donors": len(delta),
                                  "mean_error_reduction_positive_favors_model": delta.mean(),
                                  "conditional_donor_bootstrap_95CI_low": np.quantile(bootstrap, .025),
                                  "conditional_donor_bootstrap_95CI_high": np.quantile(bootstrap, .975),
                                  "interval_scope": "resampled held-out donor errors; models not refitted"})
        save_table(pd.DataFrame(contrasts), f"{dataset}_primary_K8_robustness_validation_incremental_value")

    # Visual summaries preserve donor as the independent biological unit.
    depth = sensitivity.loc[sensitivity.scenario.eq("UMI_thinning")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    donor_depth = depth.groupby(["donor", "fraction"], observed=True).agg(
        usage_MAE=("usage_MAE", "mean"), fraction_above_q95=("fraction_above_q95", "mean")).reset_index()
    for _, part in donor_depth.groupby("donor"):
        part = part.sort_values("fraction")
        axes[0].plot(part.fraction, part.usage_MAE, color="#4477AA", alpha=.35, linewidth=.8)
        axes[1].plot(part.fraction, part.fraction_above_q95, color="#4477AA", alpha=.35, linewidth=.8)
    for ax in axes:
        ax.set_xlabel("Retained UMI fraction"); ax.set_xticks([.1, .25, .5]); ax.set_ylim(bottom=0)
    axes[0].set_ylabel("Mean absolute change in normalized usage")
    axes[1].set_ylabel("Fraction of cells above frozen q95")
    fig.suptitle("Gray held-out donor depth sensitivity\nEach line: one donor; averaged over programs and three thinnings")
    save_figure(fig, "Gray_primary_K8_robustness_validation_depth")
    for dataset, table in benchmark_tables.items():
        metric = "standardized_MSE" if dataset == "GSE268991" else "log_loss"
        selected = table.loc[table.quality_gate.eq("Gray_q95_accepted") & table.split.eq("donor")]
        order = ["covariates_only", "covariates_plus_rank_signature", "covariates_plus_AUCell", "covariates_plus_cNMF"]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for _, part in selected.loc[selected.method.isin(order)].groupby("donor"):
            values = part.set_index("method").reindex(order)[metric]
            ax.plot(range(len(order)), values, color="#4477AA", alpha=.35, marker="o", markersize=3, linewidth=.7)
        ax.set_xticks(range(4), ["Covariates", "+ rank\nsignature", "+ AUCell", "+ cNMF"])
        ax.set_ylabel("Held-out donor standardized MSE" if dataset == "GSE268991" else "Held-out donor multiclass log loss")
        ax.set_ylim(bottom=0)
        ax.set_title(f"{dataset}: matched donor-held-out comparison\nEach line: one donor; lower error is better")
        save_figure(fig, f"{dataset}_primary_K8_robustness_validation_benchmark")
    if frozen_hashes != {key: sha256sum(scorer[key]) for key in frozen_hashes}:
        raise RuntimeError("Frozen model artifacts changed during validation")
    if sha256sum(GSE268991_PREANALYSIS) != preanalysis_hash:
        raise RuntimeError("Preanalysis manifest changed during validation")
    progress["status"] = "completed"
    progress["n_Gray_donors"] = sensitivity.donor.nunique()
    progress["n_GSE268991_libraries"] = len(GSE268991_SAMPLES)
    progress["n_GSE294268_libraries"] = len(raw_inventory)
    progress["results"] = {str(path.relative_to(PROJECT)): sha256sum(path)
                           for path in sorted(TABLE_DIR.glob("*_primary_K8_robustness_validation_*.csv"))}
    progress["figures"] = {str(path.relative_to(PROJECT)): sha256sum(path)
                           for path in sorted(FIGURE_DIR.glob("*_primary_K8_robustness_validation_*.pdf"))}
    flush_progress()
    display(pd.DataFrame({"dataset": list(benchmark_tables),
                          "held_out_donors": [t.donor.nunique() for t in benchmark_tables.values()]}))
    print("Robustness validation completed. K8 and all original thresholds remain frozen.")

if EXECUTION_STAGE == "robustness_validation":
    try:
        run_robustness_validation()
    except Exception as error:
        failure_path = TABLE_DIR / "Gray_primary_K8_robustness_validation_manifest.json"
        failure = json.loads(failure_path.read_text()) if failure_path.exists() else {}
        failure.update(status="failed", error=f"{type(error).__name__}: {error}",
                       updated_UTC=datetime.now(timezone.utc).isoformat())
        atomic_json(failure_path, failure)
        raise
else:
    print("Additional robustness validation not launched in stage", EXECUTION_STAGE)
'''


def build_notebook():
    cells = [
        md(r"""
        # 19b · cNMF cell-state model validation

        **Scope.** This notebook validates a continuous, fixed-basis cNMF
        **cell-state scorer**. It does not train the γδ lineage classifier, infer
        Vδ1/Vδ2 subtype, or turn P1–P10 into mutually exclusive cell labels.

        The main model-selection evidence is donor-separated Gray refitting across
        `K = 5, 8, 10, 12, 15`. Each outer fold learns its genes, HVGs, TPM standard
        deviations and cNMF spectra from training donors only. Held-out donors are
        scored by the scoring procedure reconstructed in notebook 19:

        `raw counts → TPM over the training universe → training HVGs → divide by
        training TPM SD → native fixed-H NMF → continuous usages`.

        GSE294268 is an external perturbation dataset. Its downloaded SCT matrix is
        suitable for eligibility and metadata QC, but is **not** substituted for raw
        UMI counts in the final scorer. External data are locked validation evidence;
        they are never used to choose K.

        PANGEA is a heterogeneous atlas stress test. Only PANGEA cells annotated
        `T_gdT_TRDV1` or `T_gdT_TRDV2` are projected, using `layers["count"]` raw
        UMI counts. PANGEA is never used to choose K, refit the basis, change the
        Gray-derived q90/q95 quality gates, or rename programs. Because PANGEA was
        inspected in notebooks 16–18 and supplies expression-derived silver labels,
        it is external transportability evidence, not an untouched final cohort or
        orthogonal functional validation.

        Gray Data S12 supplies an orthogonal adult flow-cytometry functional panel
        after PMA-ionomycin stimulation. Its donors do not overlap the adult Gray
        scRNA-seq donors. The functional stage therefore tests independent-cohort
        concordance across matched `tissue × Vδ` contexts; it never describes this
        evidence as donor-paired validation.

        GSE268991 is the locked untouched final external cohort. Its preanalysis
        manifest was hashed before any expression or ADT matrix was accessed. Seven
        colorectal-cancer donors contribute paired normal/tumor raw-UMI matrices,
        same-cell CITE-seq ADT and independent γδ-TCR libraries. This stage tests
        frozen P2/P3 hypotheses and benchmarks cNMF against matched rank signatures
        under leave-one-donor-out evaluation. The ADT panel has no GZMB, PRF1,
        CD107a or killing endpoint, so it is paired multimodal/TCR validation rather
        than a substitute for a prospective cytotoxicity assay.
        """),
        md(r"""
        ## Execution contract

        Restart the kernel and use **Run All**. The default `input_qc` stage is fast
        and does not require the cNMF package. Long stages are explicit:

        - `input_qc`: immutable-input, frozen-scorer and GSE294268 eligibility audit.
        - `gray_screen`: 5 donor folds × 5 K values × 30 starts; implementation and
          resource screen only, not publication evidence.
        - `gray_full`: 5 donor folds × 5 K values × 200 starts; registered final run.
        - `assemble_screen` / `assemble_full`: verify all available checkpoints,
          aggregate donor-level evidence and write publication-grade PDF figures.
        - `gray_primary_k8`: after full assembly selects K=8, fit and freeze one
          full-Gray K=8 basis with 200 starts for locked external projection.
        - `gse294268_score`: project each raw 10x library independently onto the
          frozen K=8 basis, checkpoint by library, and assemble donor-level tests.
        - `pangea_score`: project PANGEA gamma-delta cells onto the same frozen K=8
          basis, checkpoint by Series, and assemble Series-level transportability
          evidence. `GSE149356` is excluded from the primary PANGEA analysis because
          it was already used in earlier project evaluation.
        - `gray_functional_validation`: compare donor-held-out Gray K=8 program
          usages with the independent-donor Data S12 Areg, GzmB and IFNg flow
          readouts across matched tissue × Vδ contexts. Every fold is mapped to the
          frozen P1/P2/P3/P4/P5/P7/P9/P10 reference and compared with its own
          train-only rank-signature baseline.
        - `gse268991_final_validation`: project all 14 paired normal/tumor raw-UMI
          matrices using the frozen K=8 scorer, join same-cell ADT and productive
          γδ-TCR evidence, test the locked P2/P3 hypotheses, and compare cNMF with
          matched rank-signature and covariate baselines using outer donor holdout.
        - `robustness_validation`: independently checkpoint Gray donor-depth/missing
          gene sensitivity, Ensembl-aligned external projections, AUCell baselines,
          and donor-held-out surface-protein/stimulation benchmarks. This is
          post-evaluation sensitivity work, not a second untouched evaluation.

        A single job can be selected with `GDT_19B_ONLY_JOB=fold0_K5`. This supports
        scheduler arrays without changing the notebook. cNMF's completed per-start
        factor files are retained, and `skip_completed_tasks=True` is used when the
        installed cNMF version exposes it. A stage marker is reused only when its run
        signature and output checksums validate.
        """),
        code(r"""
        # 1. Imports, server-first paths, deterministic configuration and typography
        from __future__ import annotations

        import contextlib, gzip, hashlib, inspect, itertools, json, os, platform, random, re
        import shlex, sys, uuid, warnings, zipfile
        from datetime import datetime, timezone
        from importlib import metadata as im
        from pathlib import Path

        import anndata as ad
        import matplotlib as mpl
        import matplotlib.font_manager as fm
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import scipy
        from IPython.display import display
        from scipy import sparse
        from scipy.optimize import linear_sum_assignment
        from scipy.stats import rankdata, spearmanr
        from sklearn.decomposition import non_negative_factorization

        ANALYSIS_NAME = "19b_cNMF_valdiation"  # user-approved spelling
        REMOTE_PROJECT = Path("/data/users/kaelynwen/gdT_project/gdt_lr_12model")
        PROJECT = Path(os.environ.get("GDT_LR_PROJECT", REMOTE_PROJECT)).resolve()
        if not PROJECT.exists():
            local = Path("/Users/kaelynwen/Desktop/FYP_YHH/gdT_project/gdt_lr_12model")
            PROJECT = local if local.exists() else PROJECT
        if str(PROJECT).startswith("/home/kaelynwen"):
            raise RuntimeError("Project work must use /data/users/kaelynwen, not /home/kaelynwen")

        GRAY_PATH = Path(os.environ.get(
            "GRAY_ADULT_H5AD", "/data/users/kaelynwen/gdT_project/Gray data/Adult gdT cells.h5ad"
        )).resolve()
        if not GRAY_PATH.exists():
            local_gray = Path("/Users/kaelynwen/Desktop/FYP_YHH/Gray data/Adult gdT cells.h5ad")
            GRAY_PATH = local_gray if local_gray.exists() else GRAY_PATH

        GRAY_SUPPLEMENT_DIR = Path(os.environ.get(
            "GRAY_SUPPLEMENT_DIR", "/data/users/kaelynwen/gdT_project/Gray data"
        )).resolve()
        if not GRAY_SUPPLEMENT_DIR.exists():
            local_gray_supplement = Path("/Users/kaelynwen/Desktop/FYP_YHH/Gray data")
            GRAY_SUPPLEMENT_DIR = (
                local_gray_supplement if local_gray_supplement.exists()
                else GRAY_SUPPLEMENT_DIR
            )
        GRAY_DATA_S1 = GRAY_SUPPLEMENT_DIR / "NIHMS2012612-supplement-Data_S1.xlsx"
        GRAY_DATA_S12 = GRAY_SUPPLEMENT_DIR / "NIHMS2012612-supplement-Data_S12.xlsx"

        GSE294268_DIR = Path(os.environ.get(
            "GSE294268_DIR", "/data/users/kaelynwen/gdT_project/GSE294268"
        )).resolve()
        if not GSE294268_DIR.exists():
            local_gse = Path("/Users/kaelynwen/Desktop/FYP_YHH/gdT_project/GSE294268")
            GSE294268_DIR = local_gse if local_gse.exists() else GSE294268_DIR

        GSE268991_DIR = Path(os.environ.get(
            "GSE268991_DIR", "/data/users/kaelynwen/gdT_project/GSE268991"
        )).resolve()
        if not GSE268991_DIR.exists():
            local_gse268991 = Path("/Users/kaelynwen/Desktop/FYP_YHH/gdT_project/GSE268991")
            GSE268991_DIR = local_gse268991 if local_gse268991.exists() else GSE268991_DIR

        PANGEA_PATH = Path(os.environ.get(
            "GDT_PANGEA_TANDNK", "/data/users/kaelynwen/gdT_project/pangea/Level2_TandNK.h5ad"
        )).resolve()
        if not PANGEA_PATH.exists():
            local_pangea = Path("/Users/kaelynwen/Desktop/FYP_YHH/pangea/Level2_TandNK.h5ad")
            PANGEA_PATH = local_pangea if local_pangea.exists() else PANGEA_PATH

        TABLE_DIR = PROJECT / "results" / "tables" / ANALYSIS_NAME
        FIGURE_DIR = PROJECT / "results" / "figures" / ANALYSIS_NAME
        MODEL_DIR = PROJECT / "results" / "models" / ANALYSIS_NAME
        LOG_DIR = PROJECT / "results" / "logs" / ANALYSIS_NAME
        CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
        for directory in (TABLE_DIR, FIGURE_DIR, MODEL_DIR, LOG_DIR, CHECKPOINT_DIR):
            directory.mkdir(parents=True, exist_ok=True)

        DISCOVERY_NAME = "16_Gray_adult_state_program_discovery_freeze"
        DISCOVERY_TABLE = PROJECT / "results" / "tables" / DISCOVERY_NAME
        SCORING_ROOT = PROJECT / "results" / "19_scoring_biology" / "scoring_biology_v1"
        SCORER_MODEL = SCORING_ROOT / "models"

        SEED = 20260912  # retained from notebook 18 so donor folds remain comparable
        K_GRID = (5, 8, 10, 12, 15)
        N_FOLDS = 5
        SCREEN_STARTS = 30
        FULL_STARTS = 200
        DENSITY_THRESHOLD = 0.01
        N_HVG = 2000
        THREADS = int(os.environ.get("GDT_19B_THREADS", "2"))
        MIN_HVG_COVERAGE = 0.80
        REQUIRED_CNMF_VERSION = "1.7.1"
        PRIMARY_K = 8
        PRIMARY_FULL_STARTS = 200
        IMPLEMENTATION_REVISION = 1
        # Revision 4 changes assembly/input auditing only: it corrects the
        # K10-reference recovery denominator, adds K-neutral fold stability and
        # discovers the existing GSE294268_RAW/ 10x triplets. It also adds a
        # Gray-derived reconstruction quality gate, experiment-confounding audit,
        # quality-filtered sensitivity analyses, matched simple-signature tests
        # and a frozen-model card. Numerical job signatures remain unchanged so
        # validated cNMF checkpoints are reusable.
        # Numerical job signatures remain on implementation revision 1 so that
        # validated checkpoints from the original run remain reusable.
        ORCHESTRATION_REVISION = 8
        random.seed(SEED); np.random.seed(SEED)

        def choose_permitted_font():
            for family in ("Arial", "Helvetica", "Liberation Sans"):
                try:
                    fm.findfont(fm.FontProperties(family=family), fallback_to_default=False)
                    return family
                except ValueError:
                    pass
            raise RuntimeError("Install Arial, Helvetica, or Liberation Sans before producing figures")

        FIGURE_FONT = choose_permitted_font()
        mpl.rcParams.update({
            "font.family": FIGURE_FONT, "font.weight": "normal", "text.color": "black",
            "axes.labelcolor": "black", "axes.titlecolor": "black", "xtick.color": "black",
            "ytick.color": "black", "pdf.fonttype": 42, "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight",
        })
        """),
        code(r"""
        # 2. User-visible stage controls
        EXECUTION_STAGE = os.environ.get("GDT_19B_STAGE", "input_qc")
        ONLY_JOB = os.environ.get("GDT_19B_ONLY_JOB", "").strip()
        VALID_STAGES = {
            "input_qc", "gray_screen", "gray_full", "assemble_screen", "assemble_full",
            "gray_primary_k8", "gse294268_score", "pangea_score",
            "gray_functional_validation", "gse268991_final_validation",
            "robustness_validation", "depth_calibration", "p7_diagnostics", "serum_time_course",
            "incremental_value_validation", "usage_representation_validation", "evidence_synthesis", "end_to_end_validation",
            "perturbation_response_prediction",
        }
        if EXECUTION_STAGE not in VALID_STAGES:
            raise ValueError(f"GDT_19B_STAGE must be one of {sorted(VALID_STAGES)}")
        RUN_TIER = "screen" if EXECUTION_STAGE in {"gray_screen", "assemble_screen"} else "full"
        N_STARTS = SCREEN_STARTS if RUN_TIER == "screen" else FULL_STARTS
        if ONLY_JOB and not re.fullmatch(r"fold[0-4]_K(5|8|10|12|15)", ONLY_JOB):
            raise ValueError("GDT_19B_ONLY_JOB must look like fold0_K5")
        print({"stage": EXECUTION_STAGE, "tier": RUN_TIER, "n_starts": N_STARTS,
               "only_job": ONLY_JOB or None, "project": str(PROJECT)})
        """),
        code(r"""
        # 3. Auditable I/O and numerical helpers
        STATUS = []
        SOURCES = []

        def sha256sum(path, chunk_size=1024 * 1024):
            digest = hashlib.sha256()
            with Path(path).open("rb") as handle:
                for chunk in iter(lambda: handle.read(chunk_size), b""):
                    digest.update(chunk)
            return digest.hexdigest()

        def stable_token(value):
            payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
            return hashlib.sha256(payload.encode()).hexdigest()

        def sha256_lines(values):
            return hashlib.sha256(("\n".join(map(str, values)) + "\n").encode()).hexdigest()

        def atomic_text(path, text):
            path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
            temporary.write_text(text)
            os.replace(temporary, path)

        def atomic_json(path, value):
            atomic_text(path, json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")

        def record_source(path, full_hash=True):
            path = Path(path)
            if not path.is_file():
                raise FileNotFoundError(path)
            record = {"path": str(path), "size": path.stat().st_size,
                      "mtime_ns": path.stat().st_mtime_ns,
                      "sha256": sha256sum(path) if full_hash else None}
            SOURCES.append(record)
            return record

        def note(section, state, detail):
            STATUS.append({"section": section, "status": state, "detail": str(detail)})
            pd.DataFrame(STATUS).to_csv(TABLE_DIR / "analysis_status.csv", index=False)
            print(section, state, detail)

        def save_table(frame, name, index=False):
            path = TABLE_DIR / f"{name}.csv"
            temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
            frame.to_csv(temporary, index=index)
            os.replace(temporary, path)
            return path

        def atomic_parquet(frame, path):
            path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
            frame.to_parquet(temporary)
            os.replace(temporary, path)
            return path

        def save_figure(fig, name):
            fig.tight_layout(rect=[0, 0, 1, 0.965] if fig._suptitle is not None else None)
            fig.savefig(FIGURE_DIR / f"{name}.pdf")
            fig.savefig(FIGURE_DIR / f"{name}.svg")
            fig.savefig(FIGURE_DIR / f"{name}.png", dpi=600)
            display(fig); plt.close(fig)

        def rho(a, b):
            a = np.asarray(a, float); b = np.asarray(b, float)
            valid = np.isfinite(a) & np.isfinite(b)
            if valid.sum() < 3 or np.unique(a[valid]).size < 2 or np.unique(b[valid]).size < 2:
                return np.nan
            return float(spearmanr(a[valid], b[valid]).statistic)

        def normalize_usage(frame):
            return frame.div(frame.sum(axis=1).replace(0, np.nan), axis=0)

        def is_receptor_gene(symbol):
            return bool(re.match(
                r"^(TRAV|TRAJ|TRAC|TRBV|TRBJ|TRBC|TRBD|TRGV|TRGJ|TRGC|TRDV|TRDD|TRDJ|TRDC)",
                str(symbol).upper()))

        def count_source(adata):
            choices = []
            if "counts" in adata.layers:
                choices.append((adata.layers["counts"], pd.Index(adata.var_names.astype(str)), "layers[counts]"))
            if adata.raw is not None:
                names = (adata.raw.var["feature_name"].fillna(adata.raw.var_names.to_series()).astype(str)
                         if "feature_name" in adata.raw.var else adata.raw.var_names.astype(str))
                choices.append((adata.raw.X, pd.Index(names), "raw.X"))
            choices.append((adata.X, pd.Index(adata.var_names.astype(str)), "X"))
            for matrix, genes, origin in choices:
                values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix).ravel()
                if (values.size and np.isfinite(values).all() and (values >= 0).all()
                        and np.allclose(values, np.rint(values), atol=1e-6, rtol=0)):
                    return sparse.csr_matrix(matrix), genes, origin
            raise ValueError("No nonnegative integer count source; transformed expression is not accepted")

        def collapse_duplicate_genes(matrix, genes):
            genes = pd.Index(genes.astype(str), name="gene")
            unique = pd.Index(pd.unique(genes), name="gene")
            if len(unique) == len(genes):
                return matrix.tocsr(), genes
            codes = unique.get_indexer(genes)
            collapse = sparse.csr_matrix((np.ones(len(genes)), (np.arange(len(genes)), codes)),
                                         shape=(len(genes), len(unique)))
            return (matrix @ collapse).tocsr(), unique

        def donor_folds(donors):
            donors = np.asarray(sorted(map(str, donors)))
            np.random.default_rng(SEED).shuffle(donors)
            return {donor: index % N_FOLDS for index, donor in enumerate(donors)}

        def normalize_gse_cell_id(value):
            return re.sub(r"\.([0-9]+_[0-9]+)$", r"-\1", str(value))

        def read_r_space_header(path):
            opener = gzip.open if str(path).endswith(".gz") else open
            with opener(path, "rt") as handle:
                return shlex.split(handle.readline())

        def read_r_space_row_names(path):
            opener = gzip.open if str(path).endswith(".gz") else open
            names = []
            with opener(path, "rt") as handle:
                handle.readline()
                for line in handle:
                    boundary = line.find(" ")
                    if boundary < 1:
                        raise ValueError(f"Malformed R-export row in {path}")
                    names.append(line[:boundary].strip('"'))
            return pd.Index(names, name="gene")

        def validate_marker(marker, signature):
            marker = Path(marker)
            if not marker.is_file():
                return False
            payload = json.loads(marker.read_text())
            if payload.get("signature") != signature:
                raise ValueError(f"Checkpoint signature changed: {marker}")
            for relative, expected in payload.get("outputs", {}).items():
                target = marker.parent / relative
                if not target.is_file() or sha256sum(target) != expected:
                    raise ValueError(f"Checkpoint output failed checksum: {target}")
            return True

        def write_marker(marker, signature, outputs):
            marker = Path(marker)
            atomic_json(marker, {"signature": signature, "timestamp": datetime.now(timezone.utc).isoformat(),
                                 "outputs": {str(Path(p).relative_to(marker.parent)): sha256sum(p)
                                             for p in outputs}})
        """),
        md("## 1 · Immutable inputs and GSE294268 eligibility"),
        code(r"""
        # 4. Resolve and validate frozen sources without launching cNMF
        REQUIRED = {
            "Gray cohort": DISCOVERY_TABLE / "Gray_discovery_cohort_manifest.csv.gz",
            "Gray ranked genes": DISCOVERY_TABLE / "Gray_primary_frozen_ranked_genes.csv",
            "Gray freeze manifest": DISCOVERY_TABLE / "state_program_freeze_manifest.json",
            "candidate spectra": SCORER_MODEL / "candidate_HVG_TPM_spectra.tsv",
            "candidate SD": SCORER_MODEL / "candidate_training_sd.csv",
            "candidate scorer spec": SCORER_MODEL / "candidate_scoring_spec.json",
            "GSE294268 expression": GSE294268_DIR / "GSE294268_GEX_SCT_counts.tsv.gz",
            "GSE294268 HTO": GSE294268_DIR / "GSE294268_HTO_normalized_counts.tsv.gz",
            "GSE294268 hashtag map": GSE294268_DIR / "GSE294268_hashtag_list.csv.gz",
            "GSE294268 metadata": GSE294268_DIR / "GSE294268_metadata.csv.gz",
        }
        missing = {name: str(path) for name, path in REQUIRED.items() if not path.is_file()}
        if missing:
            raise FileNotFoundError(json.dumps(missing, indent=2))

        # Large GEO matrices can sit on networked or cloud-backed filesystems where a
        # full sequential hash is needlessly fragile. Their exact structural content
        # is bound below by cell-ID and gene-list hashes; small metadata/model files
        # retain full-file SHA256.
        source_records = {
            name: record_source(path, full_hash=name not in {"GSE294268 expression", "GSE294268 HTO"})
            for name, path in REQUIRED.items()
        }
        gray_record = record_source(GRAY_PATH, full_hash=True)
        freeze = json.loads(REQUIRED["Gray freeze manifest"].read_text())
        scorer_spec = json.loads(REQUIRED["candidate scorer spec"].read_text())
        reference_spectra = pd.read_csv(REQUIRED["candidate spectra"], sep="\t", index_col=0)
        reference_spectra.index = reference_spectra.index.map(lambda x: str(int(float(x))))
        reference_sd = pd.read_csv(REQUIRED["candidate SD"], index_col=0).iloc[:, 0]
        if scorer_spec.get("status") != "reconstruction passed; biological validation pending":
            raise ValueError("Notebook 19 candidate did not pass its reconstruction gate")
        if reference_spectra.shape != (10, 2000) or len(reference_sd) != 2000:
            raise ValueError("Unexpected frozen candidate dimensions")
        if list(reference_spectra.columns) != list(reference_sd.index):
            raise ValueError("Frozen spectra and training SD genes are not identically ordered")
        if not np.isfinite(reference_spectra.to_numpy()).all() or (reference_spectra.to_numpy() < 0).any():
            raise ValueError("Frozen spectra must be finite and nonnegative")
        if not np.isfinite(reference_sd).all() or (reference_sd <= 0).any():
            raise ValueError("Frozen training SD must be finite and positive")
        note("frozen scorer", "ready", "Notebook 19 reconstruction gate passed; biology pending")
        """),
        code(r"""
        # 5. GSE294268 metadata, barcode and feature-coverage audit
        gse_metadata = pd.read_csv(REQUIRED["GSE294268 metadata"], index_col=0)
        gse_metadata.index = gse_metadata.index.astype(str)
        hashtag_map = pd.read_csv(REQUIRED["GSE294268 hashtag map"], encoding="utf-8-sig")
        required_hash_columns = {"Experiment", "MULTI_ID", "Hashtag", "Donor.ID", "Allergic", "Stim"}
        if not required_hash_columns <= set(hashtag_map.columns):
            raise KeyError(f"Hashtag map lacks {sorted(required_hash_columns - set(hashtag_map.columns))}")
        if not hashtag_map["MULTI_ID"].is_unique:
            raise ValueError("Hashtag MULTI_ID must be unique before metadata join")
        annotated = gse_metadata.join(hashtag_map.set_index("MULTI_ID"), on="MULTI_classification")
        if annotated["Donor.ID"].isna().any():
            raise ValueError("Some expression cells do not map to the released hashtag metadata")

        expression_header = read_r_space_header(REQUIRED["GSE294268 expression"])
        expression_ids = pd.Index([normalize_gse_cell_id(x) for x in expression_header])
        if not expression_ids.is_unique or set(expression_ids) != set(gse_metadata.index):
            raise ValueError("GSE294268 expression barcodes do not map one-to-one to metadata")
        hto_header = pd.Index([normalize_gse_cell_id(x) for x in read_r_space_header(REQUIRED["GSE294268 HTO"])])
        if not hto_header.equals(expression_ids):
            raise ValueError("GSE294268 GEX and HTO barcode order differs")

        uncompressed_expression = GSE294268_DIR / "GSE294268_GEX_SCT_counts.tsv"
        feature_source = uncompressed_expression if uncompressed_expression.is_file() else REQUIRED["GSE294268 expression"]
        gse_genes = read_r_space_row_names(feature_source)
        if not gse_genes.is_unique:
            raise ValueError("GSE294268 processed gene names are duplicated")
        frozen_hvgs = pd.Index(scorer_spec["hvgs"])
        frozen_universe = pd.Index(scorer_spec["universe"])
        hvg_coverage = len(set(gse_genes) & set(frozen_hvgs)) / len(frozen_hvgs)
        universe_coverage = len(set(gse_genes) & set(frozen_universe)) / len(frozen_universe)
        gse_structural_fingerprint = {
            "expression_cell_ids_sha256": sha256_lines(expression_ids),
            "hto_cell_ids_sha256": sha256_lines(hto_header),
            "expression_genes_sha256": sha256_lines(gse_genes),
        }
        atomic_json(TABLE_DIR / "GSE294268_structural_fingerprint.json", gse_structural_fingerprint)

        # The extracted GEO archive is authoritative for scoring readiness. A tar
        # file alone is not evidence that the 10x libraries are complete or usable.
        raw_dir = GSE294268_DIR / "GSE294268_RAW"
        raw_matrices = sorted(raw_dir.glob("*_matrix.mtx*")) if raw_dir.is_dir() else []
        raw_rows = []
        raw_gene_hashes = {}
        for matrix_path in raw_matrices:
            library = re.sub(r"_matrix\.mtx(?:\.gz)?$", "", matrix_path.name)
            feature_hits = sorted(raw_dir.glob(f"{library}_features.tsv*"))
            barcode_hits = sorted(raw_dir.glob(f"{library}_barcodes.tsv*"))
            complete_triplet = len(feature_hits) == 1 and len(barcode_hits) == 1
            row = {
                "library": library, "matrix_path": str(matrix_path),
                "features_path": str(feature_hits[0]) if len(feature_hits) == 1 else None,
                "barcodes_path": str(barcode_hits[0]) if len(barcode_hits) == 1 else None,
                "complete_triplet": complete_triplet, "matrix_shape_matches": False,
                "frozen_HVG_coverage": 0.0, "frozen_universe_coverage": 0.0,
            }
            if complete_triplet:
                feature_path, barcode_path = feature_hits[0], barcode_hits[0]
                features = pd.read_csv(feature_path, sep="\t", header=None, compression="infer")
                if features.shape[1] < 2:
                    raise ValueError(f"10x feature table lacks gene symbols: {feature_path}")
                genes = pd.Index(features.iloc[:, 1].astype(str), name="gene")
                barcodes = pd.read_csv(barcode_path, sep="\t", header=None, compression="infer")
                opener = gzip.open if str(matrix_path).endswith(".gz") else open
                with opener(matrix_path, "rt") as handle:
                    dimensions = None
                    for line in handle:
                        if not line.startswith("%"):
                            dimensions = tuple(map(int, line.split()[:3]))
                            break
                if dimensions is None:
                    raise ValueError(f"Matrix Market dimensions missing: {matrix_path}")
                n_matrix_genes, n_matrix_cells, n_nonzero = dimensions
                unique_genes = pd.Index(pd.unique(genes))
                raw_gene_hashes[library] = sha256_lines(unique_genes)
                row.update({
                    "n_matrix_genes": n_matrix_genes, "n_matrix_cells": n_matrix_cells,
                    "n_nonzero": n_nonzero, "n_feature_rows": len(features),
                    "n_unique_gene_symbols": len(unique_genes), "n_barcodes": len(barcodes),
                    "matrix_shape_matches": bool(
                        n_matrix_genes == len(features) and n_matrix_cells == len(barcodes)),
                    "frozen_HVG_coverage": len(set(unique_genes) & set(frozen_hvgs)) / len(frozen_hvgs),
                    "frozen_universe_coverage": (
                        len(set(unique_genes) & set(frozen_universe)) / len(frozen_universe)),
                })
            raw_rows.append(row)
        raw_inventory = pd.DataFrame(raw_rows)
        if not raw_inventory.empty:
            raw_inventory = raw_inventory.sort_values("library").reset_index(drop=True)
        save_table(raw_inventory, "GSE294268_raw_10x_inventory")
        raw_available = bool(
            len(raw_inventory) == 13 and raw_inventory["complete_triplet"].all() and
            raw_inventory["matrix_shape_matches"].all() and
            raw_inventory["frozen_HVG_coverage"].ge(MIN_HVG_COVERAGE).all())
        raw_min_hvg_coverage = (
            float(raw_inventory["frozen_HVG_coverage"].min()) if len(raw_inventory) else np.nan)
        raw_min_universe_coverage = (
            float(raw_inventory["frozen_universe_coverage"].min()) if len(raw_inventory) else np.nan)
        gse_structural_fingerprint.update({
            "raw_directory": str(raw_dir), "raw_library_count": len(raw_inventory),
            "raw_library_gene_hashes": raw_gene_hashes,
        })
        atomic_json(TABLE_DIR / "GSE294268_structural_fingerprint.json", gse_structural_fingerprint)
        eligibility = pd.DataFrame([{
            "dataset": "GSE294268", "n_cells": len(gse_metadata), "n_expression_genes": len(gse_genes),
            "n_donors_in_expression": annotated["Donor.ID"].nunique(),
            "n_experiments_in_expression": annotated["Experiment"].nunique(),
            "processed_frozen_HVG_coverage": hvg_coverage,
            "processed_frozen_universe_coverage": universe_coverage,
            "raw_library_count": len(raw_inventory),
            "raw_min_frozen_HVG_coverage": raw_min_hvg_coverage,
            "raw_min_frozen_universe_coverage": raw_min_universe_coverage,
            "coverage_gate": MIN_HVG_COVERAGE,
            "raw_coverage_pass": bool(raw_available),
            "released_matrix_kind": "SCT processed counts",
            "raw_matrix_kind": "10x raw UMI counts" if raw_available else "incomplete or ineligible",
            "raw_UMI_available": raw_available,
            "final_fixed_basis_scoring_allowed": raw_available,
        }])
        save_table(eligibility, "GSE294268_eligibility")
        save_table(annotated.groupby(["Allergic", "Stim"], observed=True).size().rename("n_cells").reset_index(),
                   "GSE294268_condition_cell_counts")
        save_table(annotated.groupby(["Donor.ID", "Allergic", "Stim"], observed=True).size().rename("n_cells").reset_index(),
                   "GSE294268_donor_condition_cell_counts")
        note("GSE294268", "raw scoring eligible" if raw_available else "raw scoring blocked",
             f"13-library raw audit={raw_available}; minimum raw frozen-HVG coverage={raw_min_hvg_coverage:.1%}")
        if not raw_available:
            note("GSE294268 final projection", "blocked as designed",
                 "SCT data are audited but cannot replace raw UMI counts in the frozen scorer")

        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        plot_data = annotated.groupby(["Allergic", "Stim"], observed=True).size().unstack(fill_value=0)
        plot_data.plot.bar(ax=ax, color=mpl.colormaps["tab10"].colors[:len(plot_data.columns)])
        ax.set(xlabel="Allergy group", ylabel="Cells", title="GSE294268 released cells by group and stimulation")
        ax.tick_params(axis="x", rotation=0); ax.legend(title="Stimulation", frameon=False)
        save_figure(fig, "GSE294268_condition_cell_counts")
        display(eligibility)
        """),
        md(r"""
        ## 2 · Gray donor-separated cross-K refitting

        The screen and full tiers use identical folds and preprocessing. They are
        stored separately so a 30-start screen can never masquerade as the 200-start
        registered analysis. The held-out objective uses fold-specific training HVGs,
        fold-specific TPM SD and the native solver parameters written by cNMF.
        """),
        code(r"""
        # 6. Load Gray counts and freeze outer donor folds
        gray = ad.read_h5ad(GRAY_PATH)
        gray.obs_names = gray.obs_names.astype(str)
        gray_counts, gray_genes, gray_count_origin = count_source(gray)
        gray_counts, gray_genes = collapse_duplicate_genes(gray_counts, gray_genes)
        gray_counts.eliminate_zeros()
        gray_ids = pd.Index(gray.obs_names)
        cohort = pd.read_csv(REQUIRED["Gray cohort"], index_col=0)
        cohort.index = cohort.index.astype(str)
        primary = cohort.loc[cohort["cohort_role"].eq("primary_5prime")].copy()
        if len(primary) != 6466 or primary["donor_id"].nunique() != 22:
            raise ValueError("Gray primary cohort no longer matches the frozen 6,466-cell/22-donor definition")
        if not set(primary.index) <= set(gray_ids):
            raise ValueError("Gray primary cell IDs are missing from the raw count source")
        primary_positions = gray_ids.get_indexer(primary.index)
        primary_counts = gray_counts[primary_positions].tocsr()
        primary["donor_id"] = primary["donor_id"].astype(str)
        fold_map = donor_folds(primary["donor_id"].unique())
        primary["outer_fold"] = primary["donor_id"].map(fold_map).astype(int)
        fold_table = (primary.groupby(["outer_fold", "donor_id"], observed=True).size()
                      .rename("n_cells").reset_index().sort_values(["outer_fold", "donor_id"]))
        save_table(fold_table, "Gray_outer_donor_folds")
        for fold in range(N_FOLDS):
            train = set(primary.loc[primary.outer_fold.ne(fold), "donor_id"])
            test = set(primary.loc[primary.outer_fold.eq(fold), "donor_id"])
            if train & test or not test:
                raise AssertionError("Donor leakage or empty test fold")
        note("Gray outer folds", "frozen", f"{N_FOLDS} folds cover 22 donors without leakage")
        del gray
        """),
        code(r"""
        # 7. Fold-specific preparation, cNMF, exact held-out scoring and reference matching
        def one_file(root, pattern):
            hits = list(Path(root).rglob(pattern))
            if len(hits) != 1:
                raise FileNotFoundError(f"Expected exactly one {pattern} under {root}; found {hits}")
            return hits[0]

        def training_subset(fold):
            train_meta = primary.loc[primary.outer_fold.ne(fold)].copy()
            test_meta = primary.loc[primary.outer_fold.eq(fold)].copy()
            cap = max(1, int(np.floor(train_meta["donor_id"].value_counts().quantile(0.75))))
            selected = []
            for donor, part in train_meta.groupby("donor_id", observed=True, sort=True):
                ids = np.asarray(sorted(part.index.astype(str)))
                donor_seed = SEED + fold + int(hashlib.sha256(donor.encode()).hexdigest()[:8], 16)
                if len(ids) > cap:
                    ids = np.sort(np.random.default_rng(donor_seed).choice(ids, cap, replace=False))
                selected.extend(ids.tolist())
            selected = pd.Index(selected)
            selected_counts = primary_counts[primary.index.get_indexer(selected)].tocsr()
            selected_meta = primary.loc[selected]
            recurrence = np.zeros(len(gray_genes), dtype=int)
            for donor, cell_ids in selected_meta.groupby("donor_id", observed=True).groups.items():
                donor_matrix = selected_counts[selected.get_indexer(cell_ids)]
                threshold = max(3, int(0.01 * donor_matrix.shape[0]))
                recurrence += np.asarray((donor_matrix > 0).sum(axis=0)).ravel() >= threshold
            keep = (recurrence >= 3) & np.array([not is_receptor_gene(g) for g in gray_genes])
            train_genes = pd.Index(gray_genes[keep])
            if len(train_genes) < N_HVG:
                raise ValueError(f"Fold {fold} has fewer than {N_HVG} eligible training genes")
            train_matrix = selected_counts[:, keep].tocsr()
            test_matrix = primary_counts[primary.index.get_indexer(test_meta.index)].tocsr()
            return selected, selected_meta, train_genes, train_matrix, test_meta, test_matrix, cap

        def rank_scores(matrix, genes, signatures, max_rank=1500):
            positions = {gene: index for index, gene in enumerate(genes)}
            valid = {key: [positions[g] for g in values if g in positions] for key, values in signatures.items()}
            output = {key: np.full(matrix.shape[0], np.nan) for key in signatures}
            matrix = matrix.tocsr(copy=True); matrix.eliminate_zeros()
            for row_index in range(matrix.shape[0]):
                row = matrix.getrow(row_index)
                ranked = dict(zip(row.indices, rankdata(-row.data, method="average")))
                for key, indices in valid.items():
                    n_genes = len(indices)
                    if not n_genes:
                        continue
                    denominator = n_genes * max_rank - n_genes * (n_genes + 1) / 2
                    values = np.minimum([ranked.get(index, max_rank) for index in indices], max_rank)
                    output[key][row_index] = np.clip(
                        1 - (sum(values) - n_genes * (n_genes + 1) / 2) / denominator, 0, 1)
            return pd.DataFrame(output)

        def fit_heldout(test_matrix, test_ids, training_universe, hvgs, sd, spectra, solver_params, output_dir):
            positions = gray_genes.get_indexer(training_universe)
            if (positions < 0).any():
                raise ValueError("Training universe absent from held-out Gray count matrix")
            counts = test_matrix[:, positions].astype(float).tocsr()
            library = np.asarray(counts.sum(axis=1)).ravel()
            tpm = sparse.diags(np.divide(1e6, library, out=np.zeros_like(library), where=library > 0)) @ counts
            hvg_positions = training_universe.get_indexer(hvgs)
            if (hvg_positions < 0).any():
                raise ValueError("Fold HVGs absent from fold training universe")
            objective = tpm[:, hvg_positions].multiply(1 / sd.to_numpy()).tocsr().astype(np.float64)
            basis = spectra.loc[:, hvgs].div(sd, axis=1).to_numpy(dtype=np.float64)
            valid = library > 0
            coefficients = np.full((len(test_ids), len(spectra)), np.nan)
            params = dict(solver_params)
            params.update(H=basis, update_H=False, n_components=len(spectra))
            np.random.seed(SEED)
            from threadpoolctl import threadpool_limits
            with threadpool_limits(limits=THREADS):
                fitted, _, n_iter = non_negative_factorization(objective[valid], **params)
            coefficients[valid] = fitted
            raw = pd.DataFrame(coefficients, index=test_ids, columns=spectra.index.astype(str))
            normalized = normalize_usage(raw)
            reconstructed = fitted @ basis
            observed = objective[valid].toarray()
            relative = np.linalg.norm(observed - reconstructed, axis=1) / np.maximum(np.linalg.norm(observed, axis=1), 1e-15)
            quality = pd.DataFrame({"library_over_training_universe": library,
                                    "scaled_HVG_relative_error": np.nan}, index=test_ids)
            quality.loc[np.asarray(test_ids)[valid], "scaled_HVG_relative_error"] = relative
            for frame, name in ((raw, "heldout_usage_raw"), (normalized, "heldout_usage_normalized"),
                                (quality, "heldout_quality")):
                frame.to_parquet(output_dir / f"{name}.parquet")
            return raw, normalized, quality, n_iter

        def run_gray_job(fold, k, tier, n_starts):
            try:
                import cnmf.cnmf as cnmf_module
                from cnmf import cNMF
                from cnmf.cnmf import load_df_from_npz
                import yaml
            except ImportError as error:
                raise RuntimeError("Run Gray stages in the same Woody cNMF kernel used by notebooks 16/18") from error
            if im.version("cnmf") != REQUIRED_CNMF_VERSION:
                raise RuntimeError(f"Expected cnmf {REQUIRED_CNMF_VERSION}; found {im.version('cnmf')}")

            label = f"fold{fold}_K{k}"
            job = CHECKPOINT_DIR / tier / label
            job.mkdir(parents=True, exist_ok=True)
            selected, train_meta, train_genes, train_matrix, test_meta, test_matrix, cap = training_subset(fold)
            spec = {
                "schema_version": 1, "implementation_revision": IMPLEMENTATION_REVISION,
                "analysis": ANALYSIS_NAME, "tier": tier, "label": label,
                "gray_sha256": gray_record["sha256"], "cohort_sha256": source_records["Gray cohort"]["sha256"],
                "fold": fold, "K": k, "n_starts": n_starts, "seed": SEED + fold,
                "density_threshold": DENSITY_THRESHOLD, "n_hvg": N_HVG, "donor_cap": cap,
                "train_donors": sorted(train_meta.donor_id.unique()), "test_donors": sorted(test_meta.donor_id.unique()),
                "train_cell_ids_sha256": sha256_lines(selected), "test_cell_ids_sha256": sha256_lines(test_meta.index),
                "training_genes_sha256": sha256_lines(train_genes), "cnmf_version": im.version("cnmf"),
            }
            signature = stable_token(spec)
            spec_path = job / "job_spec.json"
            if spec_path.exists() and json.loads(spec_path.read_text()) != spec:
                raise ValueError(f"Job definition changed; do not mix checkpoints: {job}")
            atomic_json(spec_path, spec)
            input_path = job / "training_counts.h5ad"
            prepared_marker = job / "prepared.complete.json"
            if not validate_marker(prepared_marker, signature):
                ad.AnnData(train_matrix.copy(), obs=train_meta[["donor_id", "tissue"]].copy(),
                           var=pd.DataFrame(index=train_genes)).write_h5ad(input_path)
                model = cNMF(output_dir=str(job), name="cnmf")
                model.prepare(counts_fn=str(input_path), components=np.array([k]), n_iter=n_starts,
                              seed=SEED + fold, num_highvar_genes=N_HVG)
                hvg_file = one_file(job, "*.overdispersed_genes.txt")
                stats_file = one_file(job, "*.tpm_stats.df.npz")
                tpm_file = one_file(job, "*.tpm.h5ad")
                normalized_file = one_file(job, "*.norm_counts.h5ad")
                parameters_file = one_file(job, "*.nmf_idvrun_params.yaml")
                write_marker(prepared_marker, signature,
                             [input_path, hvg_file, stats_file, tpm_file, normalized_file, parameters_file])

            model = cNMF(output_dir=str(job), name="cnmf")
            factor_marker = job / "factorized.complete.json"
            if not validate_marker(factor_marker, signature):
                kwargs = {"worker_i": 0, "total_workers": 1}
                if "skip_completed_tasks" in inspect.signature(model.factorize).parameters:
                    kwargs["skip_completed_tasks"] = True
                with (LOG_DIR / f"{tier}_{label}_factorize.log").open("a") as log:
                    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                        model.factorize(**kwargs)
                factor_files = sorted(job.rglob("*spectra*.df.npz"))
                if len(factor_files) < n_starts:
                    raise RuntimeError(f"Expected at least {n_starts} factor outputs; found {len(factor_files)}")
                write_marker(factor_marker, signature, factor_files)

            consensus_marker = job / "consensus.complete.json"
            if not validate_marker(consensus_marker, signature):
                model.combine()
                with (LOG_DIR / f"{tier}_{label}_consensus.log").open("a") as log:
                    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                        model.consensus(k=k, density_threshold=DENSITY_THRESHOLD, show_clustering=True,
                                        close_clustergram_fig=True)
                density = str(DENSITY_THRESHOLD).replace(".", "_")
                fold_spectra_file = one_file(job, f"*gene_spectra_tpm.k_{k}.dt_{density}.txt")
                fold_scores_file = one_file(job, f"*gene_spectra_score.k_{k}.dt_{density}.txt")
                write_marker(consensus_marker, signature, [fold_spectra_file, fold_scores_file])

            scoring_marker = job / "heldout_scoring.complete.json"
            if validate_marker(scoring_marker, signature):
                note(label, "reused", f"validated {tier} checkpoint")
                return
            density = str(DENSITY_THRESHOLD).replace(".", "_")
            fold_spectra_file = one_file(job, f"*gene_spectra_tpm.k_{k}.dt_{density}.txt")
            fold_scores_file = one_file(job, f"*gene_spectra_score.k_{k}.dt_{density}.txt")
            hvg_file = one_file(job, "*.overdispersed_genes.txt")
            stats_file = one_file(job, "*.tpm_stats.df.npz")
            params_file = one_file(job, "*.nmf_idvrun_params.yaml")
            hvgs = pd.Index([x for x in hvg_file.read_text().splitlines() if x])
            stats = load_df_from_npz(stats_file)
            sd = stats.loc[hvgs, "__std"].astype(float)
            fold_spectra = pd.read_csv(fold_spectra_file, sep="\t", index_col=0)
            fold_scores = pd.read_csv(fold_scores_file, sep="\t", index_col=0)
            fold_spectra.index = fold_spectra.index.map(lambda x: str(int(float(x))))
            fold_scores.index = fold_scores.index.map(lambda x: str(int(float(x))))
            solver_params = yaml.safe_load(params_file.read_text())
            raw, normalized, quality, solver_iterations = fit_heldout(
                test_matrix, test_meta.index, train_genes, hvgs, sd, fold_spectra, solver_params, job)

            signatures = {program: fold_scores.loc[program].nlargest(50).index.tolist()
                          for program in fold_scores.index}
            marker = rank_scores(test_matrix, gray_genes, signatures)
            marker.index = test_meta.index; marker.columns = marker.columns.astype(str)
            marker.to_parquet(job / "heldout_training_signature_scores.parquet")
            metric_rows = []
            for donor, ids in test_meta.groupby("donor_id", observed=True).groups.items():
                for program in fold_spectra.index:
                    metric_rows.append({
                        "tier": tier, "fold": fold, "K": k, "donor": donor,
                        "fold_program": program, "n_cells": len(ids),
                        "rho_training_signature": rho(normalized.loc[ids, program], marker.loc[ids, program]),
                        "median_scaled_HVG_relative_error": quality.loc[ids, "scaled_HVG_relative_error"].median(),
                        "solver_iterations": solver_iterations,
                    })
            metrics_path = job / "heldout_donor_metrics.csv"
            pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)

            common = sorted(set(reference_spectra.columns) & set(fold_spectra.columns))
            ref = reference_spectra[common].to_numpy(float)
            fitted = fold_spectra[common].to_numpy(float)
            cosine = ref @ fitted.T / np.maximum(
                np.outer(np.linalg.norm(ref, axis=1), np.linalg.norm(fitted, axis=1)), 1e-15)
            ref_ranked = pd.read_csv(REQUIRED["Gray ranked genes"])
            ref_ranked["program"] = ref_ranked["program"].map(lambda x: str(int(float(x))))
            ref_top = {p: set(ref_ranked.loc[ref_ranked.program.eq(p)].sort_values("rank").head(50).gene)
                       for p in reference_spectra.index}
            row_index, column_index = linear_sum_assignment(-cosine)
            assignments = dict(zip(row_index, column_index))
            match_rows = []
            for i, reference_program in enumerate(reference_spectra.index):
                for j, fold_program in enumerate(fold_spectra.index):
                    alternative = np.delete(cosine[i], j)
                    fold_top = set(signatures[fold_program])
                    match_rows.append({
                        "tier": tier, "fold": fold, "K": k,
                        "reference_program": reference_program, "fold_program": fold_program,
                        "cosine": cosine[i, j],
                        "top50_jaccard": len(ref_top[reference_program] & fold_top) /
                                         max(1, len(ref_top[reference_program] | fold_top)),
                        "assigned_one_to_one": assignments.get(i) == j,
                        "margin_to_best_alternative": cosine[i, j] - alternative.max() if len(alternative) else np.nan,
                        "n_common_genes": len(common),
                    })
            matching_path = job / "reference_matching.csv"
            pd.DataFrame(match_rows).to_csv(matching_path, index=False)
            outputs = [job / "heldout_usage_raw.parquet", job / "heldout_usage_normalized.parquet",
                       job / "heldout_quality.parquet", job / "heldout_training_signature_scores.parquet",
                       metrics_path, matching_path]
            write_marker(scoring_marker, signature, outputs)
            note(label, "completed", f"{tier}; exact training-only held-out scoring")
        """),
        code(r"""
        # 8. Execute only the requested Gray refit tier/jobs
        def validated_density_failure(job):
            spec_path = job / "job_spec.json"
            failure_path = job / "consensus_density_filter.failure.json"
            if not (spec_path.is_file() and failure_path.is_file()):
                return False
            spec = json.loads(spec_path.read_text())
            failure = json.loads(failure_path.read_text())
            return (
                failure.get("schema_version") == 1 and
                failure.get("job_signature") == stable_token(spec) and
                failure.get("failure_type") == "zero_components_after_density_filter" and
                float(failure.get("density_threshold")) == float(DENSITY_THRESHOLD)
            )

        if EXECUTION_STAGE in {"gray_screen", "gray_full"}:
            for fold in range(N_FOLDS):
                for k in K_GRID:
                    label = f"fold{fold}_K{k}"
                    if ONLY_JOB and ONLY_JOB != label:
                        continue
                    job = CHECKPOINT_DIR / RUN_TIER / label
                    if validated_density_failure(job):
                        note(label, "reused_expected_failure",
                             f"{RUN_TIER}; zero components at registered density threshold")
                        continue
                    try:
                        run_gray_job(fold, k, RUN_TIER, N_STARTS)
                    except RuntimeError as error:
                        expected = "Zero components remain after density filtering"
                        if expected not in str(error):
                            raise
                        spec_path = job / "job_spec.json"
                        if not spec_path.is_file():
                            raise RuntimeError(f"Cannot bind expected failure to missing job spec: {job}") from error
                        spec = json.loads(spec_path.read_text())
                        atomic_json(job / "consensus_density_filter.failure.json", {
                            "schema_version": 1,
                            "analysis": ANALYSIS_NAME,
                            "tier": RUN_TIER,
                            "label": label,
                            "fold": fold,
                            "K": k,
                            "failure_type": "zero_components_after_density_filter",
                            "density_threshold": DENSITY_THRESHOLD,
                            "job_signature": stable_token(spec),
                            "error_type": type(error).__name__,
                            "error_message": str(error),
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "interpretation": (
                                "Registered consensus failed without adaptive threshold tuning; "
                                "retain as evidence against robust recurrence at this K."
                            ),
                        })
                        note(label, "expected_failure",
                             f"{RUN_TIER}; zero components at registered density threshold")
        else:
            print("No cNMF refit launched in stage", EXECUTION_STAGE)
        """),
        md(r"""
        ## 3 · Freeze the full-Gray primary K=8 basis

        This stage is unlocked only by the completed full donor-fold decision record.
        It reuses the immutable, donor-balanced full-Gray count input prepared by
        notebook 16, but writes a new K=8 model under notebook 19b. It never mutates
        the earlier frozen K=10 discovery. Per-start cNMF factors are durable restart
        units; validated preparation, factorization, consensus and freeze markers are
        reused after interruption.
        """),
        code(r"""
        # 9. Fit and freeze the selected full-Gray K=8 basis for external projection
        PRIMARY_JOB = CHECKPOINT_DIR / "primary_K8_full"
        PRIMARY_FREEZE_DIR = MODEL_DIR / "primary_K8"
        PRIMARY_DECISION = TABLE_DIR / "Gray_full_K_decision_record.json"
        NOTEBOOK16_MODEL = PROJECT / "results" / "models" / DISCOVERY_NAME
        PRIMARY_INPUT = NOTEBOOK16_MODEL / "Gray_primary_balanced_counts.h5ad"

        def freeze_primary_k8():
            try:
                from cnmf import cNMF
                from cnmf.cnmf import load_df_from_npz
                import yaml
            except ImportError as error:
                raise RuntimeError("Run the primary fit in the registered Woody cNMF kernel") from error
            if im.version("cnmf") != REQUIRED_CNMF_VERSION:
                raise RuntimeError(f"Expected cnmf {REQUIRED_CNMF_VERSION}; found {im.version('cnmf')}")
            if not PRIMARY_DECISION.is_file():
                raise FileNotFoundError("Run assemble_full before fitting the primary basis")
            decision = json.loads(PRIMARY_DECISION.read_text())
            if int(decision.get("primary_K_recommendation", -1)) != PRIMARY_K:
                raise RuntimeError(f"Full assembly did not select the registered primary K={PRIMARY_K}")
            if not PRIMARY_INPUT.is_file():
                raise FileNotFoundError(PRIMARY_INPUT)

            PRIMARY_JOB.mkdir(parents=True, exist_ok=True)
            PRIMARY_FREEZE_DIR.mkdir(parents=True, exist_ok=True)
            primary_input = ad.read_h5ad(PRIMARY_INPUT, backed="r")
            try:
                input_cells = pd.Index(primary_input.obs_names.astype(str))
                input_genes = pd.Index(primary_input.var_names.astype(str))
                input_shape = tuple(primary_input.shape)
            finally:
                primary_input.file.close()
            spec = {
                "schema_version": 1, "primary_implementation_revision": 1,
                "analysis": ANALYSIS_NAME, "job": "primary_K8_full",
                "K": PRIMARY_K, "n_starts": PRIMARY_FULL_STARTS, "seed": SEED,
                "density_threshold": DENSITY_THRESHOLD, "n_hvg": N_HVG,
                "input_path": str(PRIMARY_INPUT), "input_sha256": sha256sum(PRIMARY_INPUT),
                "input_shape": input_shape, "cell_ids_sha256": sha256_lines(input_cells),
                "gene_universe_sha256": sha256_lines(input_genes),
                "decision_sha256": sha256sum(PRIMARY_DECISION),
                "cnmf_version": im.version("cnmf"),
            }
            signature = stable_token(spec)
            spec_path = PRIMARY_JOB / "job_spec.json"
            if spec_path.exists() and json.loads(spec_path.read_text()) != spec:
                raise ValueError(f"Primary K8 job definition changed: {PRIMARY_JOB}")
            atomic_json(spec_path, spec)

            prepared_marker = PRIMARY_JOB / "prepared.complete.json"
            model = cNMF(output_dir=str(PRIMARY_JOB), name="cnmf")
            if not validate_marker(prepared_marker, signature):
                model.prepare(counts_fn=str(PRIMARY_INPUT), components=np.asarray([PRIMARY_K]),
                              n_iter=PRIMARY_FULL_STARTS, seed=SEED, num_highvar_genes=N_HVG)
                prepared_outputs = [
                    one_file(PRIMARY_JOB, "*.overdispersed_genes.txt"),
                    one_file(PRIMARY_JOB, "*.tpm_stats.df.npz"),
                    one_file(PRIMARY_JOB, "*.tpm.h5ad"),
                    one_file(PRIMARY_JOB, "*.norm_counts.h5ad"),
                    one_file(PRIMARY_JOB, "*.nmf_idvrun_params.yaml"),
                ]
                write_marker(prepared_marker, signature, prepared_outputs)

            factor_marker = PRIMARY_JOB / "factorized.complete.json"
            if not validate_marker(factor_marker, signature):
                kwargs = {"worker_i": 0, "total_workers": 1}
                if "skip_completed_tasks" in inspect.signature(model.factorize).parameters:
                    kwargs["skip_completed_tasks"] = True
                with (LOG_DIR / "primary_K8_full_factorize.log").open("a") as log:
                    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                        model.factorize(**kwargs)
                factor_files = sorted(PRIMARY_JOB.rglob("*spectra.k_8.iter_*.df.npz"))
                if len(factor_files) != PRIMARY_FULL_STARTS:
                    raise RuntimeError(
                        f"Expected {PRIMARY_FULL_STARTS} primary K8 factor files; found {len(factor_files)}")
                write_marker(factor_marker, signature, factor_files)

            consensus_marker = PRIMARY_JOB / "consensus.complete.json"
            if not validate_marker(consensus_marker, signature):
                model.combine()
                with (LOG_DIR / "primary_K8_full_consensus.log").open("a") as log:
                    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                        model.consensus(k=PRIMARY_K, density_threshold=DENSITY_THRESHOLD,
                                        show_clustering=True, close_clustergram_fig=True)
                density = str(DENSITY_THRESHOLD).replace(".", "_")
                consensus_outputs = [
                    one_file(PRIMARY_JOB, f"*gene_spectra_tpm.k_{PRIMARY_K}.dt_{density}.txt"),
                    one_file(PRIMARY_JOB, f"*gene_spectra_score.k_{PRIMARY_K}.dt_{density}.txt"),
                    one_file(PRIMARY_JOB, f"*usages.k_{PRIMARY_K}.dt_{density}.consensus.txt"),
                ]
                write_marker(consensus_marker, signature, consensus_outputs)

            freeze_marker = PRIMARY_FREEZE_DIR / "frozen.complete.json"
            if validate_marker(freeze_marker, signature):
                note("primary K8 basis", "reused", "validated full-Gray frozen basis")
                return
            density = str(DENSITY_THRESHOLD).replace(".", "_")
            spectra = pd.read_csv(one_file(
                PRIMARY_JOB, f"*gene_spectra_tpm.k_{PRIMARY_K}.dt_{density}.txt"),
                sep="\t", index_col=0)
            scores = pd.read_csv(one_file(
                PRIMARY_JOB, f"*gene_spectra_score.k_{PRIMARY_K}.dt_{density}.txt"),
                sep="\t", index_col=0)
            spectra.index = spectra.index.map(lambda x: str(int(float(x))))
            scores.index = scores.index.map(lambda x: str(int(float(x))))
            hvgs = pd.Index([
                value for value in one_file(PRIMARY_JOB, "*.overdispersed_genes.txt").read_text().splitlines()
                if value
            ])
            stats = load_df_from_npz(one_file(PRIMARY_JOB, "*.tpm_stats.df.npz"))
            sd = stats.loc[hvgs, "__std"].astype(float)
            if not (sd > 0).all() or not np.isfinite(sd).all():
                raise ValueError("Primary K8 training SD must be finite and positive")
            hvg_spectra = spectra.loc[:, hvgs]
            spectra_path = PRIMARY_FREEZE_DIR / "primary_K8_HVG_TPM_spectra.tsv"
            sd_path = PRIMARY_FREEZE_DIR / "primary_K8_training_sd.csv"
            ranked_path = PRIMARY_FREEZE_DIR / "primary_K8_ranked_genes.csv"
            hvg_spectra.to_csv(spectra_path, sep="\t")
            sd.rename("training_TPM_sd").to_csv(sd_path)
            ranked = pd.concat([
                pd.DataFrame({"K8_component": component, "rank": np.arange(1, 101),
                              "gene": scores.loc[component].nlargest(100).index,
                              "score": scores.loc[component].nlargest(100).values})
                for component in scores.index
            ], ignore_index=True)
            ranked.to_csv(ranked_path, index=False)
            save_table(ranked, "Gray_primary_K8_ranked_genes")

            common = sorted(set(reference_spectra.columns) & set(hvg_spectra.columns))
            ref = reference_spectra.loc[:, common].to_numpy(float)
            fitted = hvg_spectra.loc[:, common].to_numpy(float)
            cosine = ref @ fitted.T / np.maximum(
                np.outer(np.linalg.norm(ref, axis=1), np.linalg.norm(fitted, axis=1)), 1e-15)
            rows, columns = linear_sum_assignment(-cosine)
            assignments = dict(zip(rows, columns))
            mapping_rows = []
            for i, reference_program in enumerate(reference_spectra.index):
                for j, component in enumerate(hvg_spectra.index):
                    mapping_rows.append({
                        "K10_reference_program": reference_program, "K8_component": component,
                        "cosine": cosine[i, j], "assigned_one_to_one": assignments.get(i) == j,
                        "n_common_genes": len(common),
                    })
            mapping = pd.DataFrame(mapping_rows)
            mapping_path = PRIMARY_FREEZE_DIR / "primary_K8_to_K10_reference_mapping.csv"
            mapping.to_csv(mapping_path, index=False)
            save_table(mapping, "Gray_primary_K8_to_K10_reference_mapping")
            scoring_spec_path = PRIMARY_FREEZE_DIR / "primary_K8_scoring_spec.json"
            atomic_json(scoring_spec_path, {
                "schema_version": 1, "analysis": ANALYSIS_NAME, "status": "frozen for external validation",
                "K": PRIMARY_K, "n_starts": PRIMARY_FULL_STARTS, "seed": SEED,
                "density_threshold": DENSITY_THRESHOLD, "normalization": "TPM_1000000",
                "scaling": "divide_HVG_TPM_by_training_TPM_sd_without_centering",
                "usage_normalization": "row_sum_to_one",
                "universe": input_genes.tolist(), "hvgs": hvgs.tolist(),
                "spectra_sha256": sha256sum(spectra_path), "sd_sha256": sha256sum(sd_path),
                "source_job_signature": signature,
                "evidence_boundary": "continuous cell-state usages; not lineage or Vd subtype prediction",
            })
            outputs = [spectra_path, sd_path, ranked_path, mapping_path, scoring_spec_path]
            write_marker(freeze_marker, signature, outputs)
            note("primary K8 basis", "completed", "full-Gray 200-start basis frozen for external projection")

        if EXECUTION_STAGE == "gray_primary_k8":
            freeze_primary_k8()
        else:
            print("Primary K8 full-data fit not launched in stage", EXECUTION_STAGE)
        """),
        md(r"""
        ## 4 · Locked external projection into GSE294268 raw UMI counts

        Every raw 10x library is scored independently using the frozen full-Gray
        K=8 HVGs, training SD and spectra. Missing query genes are zero-filled;
        libraries below 80% frozen-HVG coverage are rejected. The released SCT
        matrix is used only to verify cell identities, never as scorer input.

        Inference is performed at donor level. Reconstruction-error thresholds are
        learned only from donor-held-out Gray cells: cells above the Gray 95th
        percentile are rejected, while the Gray 90th percentile is a stricter
        sensitivity gate. CR-allergic and MO-allergic pooled contrasts are exported
        only as experiment-confounded exploratory summaries. Within NonA,
        CR/MO/HDMAPP contrasts are paired by donor. The rank-based top-gene score is
        tested under the same donor design as a simple baseline, not treated as an
        independent biological ground truth.
        """),
        code(r"""
        # 10. Score GSE294268 raw 10x libraries and assemble donor-level evidence
        GSE_PROJECTION_DIR = MODEL_DIR / "GSE294268_primary_K8_projection"
        GSE_PROJECTION_CHECKPOINTS = CHECKPOINT_DIR / "GSE294268_primary_K8_projection"

        def bh_adjust(values):
            values = np.asarray(values, float)
            output = np.full(len(values), np.nan)
            valid = np.flatnonzero(np.isfinite(values))
            if not len(valid):
                return output
            order = valid[np.argsort(values[valid])]
            ranked = values[order] * len(valid) / np.arange(1, len(valid) + 1)
            ranked = np.minimum.accumulate(ranked[::-1])[::-1]
            output[order] = np.minimum(ranked, 1.0)
            return output

        def score_gse294268_library(inventory_row, scorer):
            import scipy.io
            import yaml

            library = str(inventory_row.library)
            orig_ident = re.sub(r"^GSM[0-9]+_", "", library)
            matrix_path = Path(inventory_row.matrix_path)
            feature_path = Path(inventory_row.features_path)
            barcode_path = Path(inventory_row.barcodes_path)
            raw_barcodes = pd.read_csv(
                barcode_path, sep="\t", header=None, compression="infer").iloc[:, 0].astype(str)
            expected_ids = pd.Index(gse_metadata.index[
                gse_metadata["orig.ident"].astype(str).eq(orig_ident)], dtype=str)
            suffix_matches = []
            for suffix_number in range(1, len(raw_inventory) + 1):
                candidate = pd.Index(raw_barcodes + f"_{suffix_number}")
                overlap = len(set(candidate) & set(expected_ids))
                suffix_matches.append((overlap, suffix_number, candidate))
            suffix_matches.sort(key=lambda item: item[0], reverse=True)
            best_overlap, suffix_number, released_ids = suffix_matches[0]
            if best_overlap != len(expected_ids) or (
                    len(suffix_matches) > 1 and suffix_matches[1][0] == best_overlap):
                raise ValueError(f"Cannot uniquely map raw barcodes to released cells for {library}")
            keep = released_ids.isin(expected_ids)
            selected_positions = np.flatnonzero(keep)
            selected_ids = pd.Index(released_ids[keep], dtype=str)
            if not selected_ids.is_unique or set(selected_ids) != set(expected_ids):
                raise ValueError(f"Released-cell mapping is not one-to-one for {library}")

            job = GSE_PROJECTION_CHECKPOINTS / library
            job.mkdir(parents=True, exist_ok=True)
            spec = {
                "schema_version": 1, "analysis": ANALYSIS_NAME, "library": library,
                "orig_ident": orig_ident, "barcode_suffix_number": suffix_number,
                "n_released_cells": len(selected_ids), "cell_ids_sha256": sha256_lines(selected_ids),
                "matrix_sha256": sha256sum(matrix_path), "features_sha256": sha256sum(feature_path),
                "barcodes_sha256": sha256sum(barcode_path),
                "scoring_spec_sha256": sha256sum(scorer["spec_path"]),
                "spectra_sha256": sha256sum(scorer["spectra_path"]),
                "sd_sha256": sha256sum(scorer["sd_path"]),
                "solver_params_sha256": sha256sum(scorer["params_path"]),
            }
            signature = stable_token(spec)
            spec_path = job / "job_spec.json"
            if spec_path.exists() and json.loads(spec_path.read_text()) != spec:
                raise ValueError(f"GSE294268 library definition changed: {job}")
            atomic_json(spec_path, spec)
            marker = job / "scoring.complete.json"
            if validate_marker(marker, signature):
                note(library, "reused", "validated raw-UMI projection checkpoint")
                return

            features = pd.read_csv(feature_path, sep="\t", header=None, compression="infer")
            raw_genes = pd.Index(features.iloc[:, 1].astype(str))
            opener = gzip.open if str(matrix_path).endswith(".gz") else open
            with opener(matrix_path, "rb") as handle:
                raw_matrix = scipy.io.mmread(handle).tocsr()
            if raw_matrix.shape != (len(raw_genes), len(raw_barcodes)):
                raise ValueError(f"10x matrix dimensions disagree with features/barcodes: {library}")
            universe_positions = scorer["universe"].get_indexer(raw_genes)
            valid_features = universe_positions >= 0
            collapse = sparse.csr_matrix((
                np.ones(valid_features.sum()),
                (np.flatnonzero(valid_features), universe_positions[valid_features])),
                shape=(len(raw_genes), len(scorer["universe"])))
            counts = (raw_matrix[:, selected_positions].T @ collapse).tocsr().astype(float)
            library_size = np.asarray(counts.sum(axis=1)).ravel()
            if (library_size <= 0).any():
                raise ValueError(f"Released cells with zero counts over training universe: {library}")
            tpm = sparse.diags(1e6 / library_size) @ counts
            objective = tpm[:, scorer["hvg_positions"]].multiply(
                1 / scorer["sd"].to_numpy()).tocsr().astype(np.float64)
            params = dict(scorer["solver_params"])
            params.update(H=scorer["basis"], update_H=False, n_components=PRIMARY_K)
            usage_blocks, relative_error = [], []
            gram = scorer["basis"] @ scorer["basis"].T
            from threadpoolctl import threadpool_limits
            for start in range(0, objective.shape[0], 5000):
                block = objective[start:start + 5000]
                np.random.seed(SEED)
                with threadpool_limits(limits=THREADS):
                    coefficients, _, _ = non_negative_factorization(block, **params)
                usage_blocks.append(coefficients)
                x_norm_sq = np.asarray(block.multiply(block).sum(axis=1)).ravel()
                cross = np.sum(coefficients * (block @ scorer["basis"].T), axis=1)
                fitted_norm_sq = np.sum((coefficients @ gram) * coefficients, axis=1)
                residual_sq = np.maximum(x_norm_sq + fitted_norm_sq - 2 * cross, 0)
                relative_error.extend(np.sqrt(residual_sq) / np.maximum(np.sqrt(x_norm_sq), 1e-15))
            raw_usage = pd.DataFrame(
                np.vstack(usage_blocks), index=selected_ids, columns=scorer["components"])
            normalized_usage = normalize_usage(raw_usage)
            quality = pd.DataFrame({
                "library_over_training_universe": library_size,
                "scaled_HVG_relative_error": relative_error,
                "orig.ident": orig_ident,
            }, index=selected_ids)
            signature_scores = rank_scores(counts, scorer["universe"], scorer["signatures"])
            signature_scores.index = selected_ids
            usage_path = atomic_parquet(normalized_usage, job / "usage_normalized.parquet")
            quality_path = atomic_parquet(quality, job / "quality.parquet")
            baseline_path = atomic_parquet(signature_scores, job / "rank_signature_scores.parquet")
            write_marker(marker, signature, [usage_path, quality_path, baseline_path])
            note(library, "completed", f"{len(selected_ids)} released cells scored from raw UMI counts")

        def run_gse294268_projection():
            import yaml
            from scipy.stats import mannwhitneyu, wilcoxon

            primary_spec_path = PRIMARY_JOB / "job_spec.json"
            primary_marker = PRIMARY_FREEZE_DIR / "frozen.complete.json"
            if not primary_spec_path.is_file() or not validate_marker(
                    primary_marker, stable_token(json.loads(primary_spec_path.read_text()))):
                raise RuntimeError("The frozen primary K8 basis is absent or failed checksum validation")
            scoring_spec_path = PRIMARY_FREEZE_DIR / "primary_K8_scoring_spec.json"
            spectra_path = PRIMARY_FREEZE_DIR / "primary_K8_HVG_TPM_spectra.tsv"
            sd_path = PRIMARY_FREEZE_DIR / "primary_K8_training_sd.csv"
            params_path = one_file(PRIMARY_JOB, "*.nmf_idvrun_params.yaml")
            scoring_spec = json.loads(scoring_spec_path.read_text())
            spectra = pd.read_csv(spectra_path, sep="\t", index_col=0)
            spectra.index = spectra.index.map(lambda x: str(int(float(x))))
            sd = pd.read_csv(sd_path, index_col=0).iloc[:, 0]
            universe = pd.Index(scoring_spec["universe"])
            hvgs = pd.Index(scoring_spec["hvgs"])
            hvg_positions = universe.get_indexer(hvgs)
            if (hvg_positions < 0).any() or list(spectra.columns) != list(hvgs) or list(sd.index) != list(hvgs):
                raise ValueError("Frozen K8 scorer gene definitions are inconsistent")
            basis = spectra.div(sd, axis=1).to_numpy(np.float64)
            ranked = pd.read_csv(PRIMARY_FREEZE_DIR / "primary_K8_ranked_genes.csv")
            ranked["K8_component"] = ranked["K8_component"].map(lambda x: str(int(float(x))))
            signatures = {
                component: part.sort_values("rank").head(50)["gene"].astype(str).tolist()
                for component, part in ranked.groupby("K8_component", observed=True)
            }
            scorer = {
                "spec_path": scoring_spec_path, "spectra_path": spectra_path, "sd_path": sd_path,
                "params_path": params_path, "universe": universe, "hvgs": hvgs,
                "hvg_positions": hvg_positions, "sd": sd, "basis": basis,
                "components": spectra.index.astype(str), "signatures": signatures,
                "solver_params": yaml.safe_load(params_path.read_text()),
            }
            if not raw_available:
                raise RuntimeError("GSE294268 raw 10x audit did not pass")
            GSE_PROJECTION_DIR.mkdir(parents=True, exist_ok=True)
            GSE_PROJECTION_CHECKPOINTS.mkdir(parents=True, exist_ok=True)
            for row in raw_inventory.itertuples(index=False):
                score_gse294268_library(row, scorer)

            usage_parts, quality_parts, baseline_parts = [], [], []
            for row in raw_inventory.itertuples(index=False):
                job = GSE_PROJECTION_CHECKPOINTS / str(row.library)
                usage_parts.append(pd.read_parquet(job / "usage_normalized.parquet"))
                quality_parts.append(pd.read_parquet(job / "quality.parquet"))
                baseline_parts.append(pd.read_parquet(job / "rank_signature_scores.parquet"))
            usage = pd.concat(usage_parts).loc[gse_metadata.index]
            quality = pd.concat(quality_parts).loc[gse_metadata.index]
            baseline = pd.concat(baseline_parts).loc[gse_metadata.index]
            if not usage.index.is_unique or len(usage) != len(gse_metadata):
                raise ValueError("Projected GSE294268 cells do not match released metadata one-to-one")
            if not np.allclose(usage.sum(axis=1).to_numpy(float), 1.0, atol=1e-8):
                raise ValueError("Normalized K8 usages must sum to one for every projected cell")

            # Calibrate query rejection only from donor-held-out Gray cells. The
            # external dataset never sets its own acceptance threshold.
            gray_quality_paths = [
                CHECKPOINT_DIR / "full" / f"fold{fold}_K8" / "heldout_quality.parquet"
                for fold in range(N_FOLDS)
            ]
            missing_gray_quality = [str(path) for path in gray_quality_paths if not path.is_file()]
            if missing_gray_quality:
                raise FileNotFoundError(
                    "All five full K8 held-out quality files are required: " + ", ".join(missing_gray_quality))
            gray_errors = pd.concat([
                pd.read_parquet(path)["scaled_HVG_relative_error"] for path in gray_quality_paths
            ]).astype(float)
            if not np.isfinite(gray_errors).all():
                raise ValueError("Gray held-out reconstruction errors must be finite")
            gray_q90, gray_q95, gray_q99 = [float(gray_errors.quantile(q)) for q in (0.90, 0.95, 0.99)]
            external_error = quality["scaled_HVG_relative_error"].astype(float)
            quality["within_Gray_q90"] = external_error.le(gray_q90)
            quality["within_Gray_q95"] = external_error.le(gray_q95)
            quality["reconstruction_gate"] = np.select(
                [quality["within_Gray_q90"], quality["within_Gray_q95"]],
                ["high_confidence_within_q90", "accepted_q90_to_q95"],
                default="abstain_above_q95")

            usage_path = atomic_parquet(
                usage, GSE_PROJECTION_DIR / "GSE294268_primary_K8_usage_normalized.parquet")
            quality_path = atomic_parquet(
                quality, GSE_PROJECTION_DIR / "GSE294268_primary_K8_quality.parquet")
            baseline_path = atomic_parquet(
                baseline, GSE_PROJECTION_DIR / "GSE294268_primary_K8_rank_signature_scores.parquet")
            gated_usage = usage.where(quality["within_Gray_q95"], np.nan)
            gated_usage_path = atomic_parquet(
                gated_usage,
                GSE_PROJECTION_DIR / "GSE294268_primary_K8_usage_q95_quality_gated.parquet")

            quality_summary_rows = []
            for source, values in (("Gray_donor_heldout_K8", gray_errors),
                                   ("GSE294268_external", external_error)):
                quality_summary_rows.append({
                    "source": source, "n_cells": len(values),
                    "median_relative_error": float(values.median()),
                    "q90_relative_error": float(values.quantile(0.90)),
                    "q95_relative_error": float(values.quantile(0.95)),
                    "q99_relative_error": float(values.quantile(0.99)),
                    "fraction_above_Gray_q90": float(values.gt(gray_q90).mean()),
                    "fraction_above_Gray_q95": float(values.gt(gray_q95).mean()),
                    "fraction_above_Gray_q99": float(values.gt(gray_q99).mean()),
                })
            quality_summary = pd.DataFrame(quality_summary_rows)
            save_table(quality_summary, "GSE294268_primary_K8_reconstruction_quality_gate")

            metadata = annotated[["Donor.ID", "Allergic", "Stim", "Experiment", "orig.ident"]].copy()
            metadata["Donor.ID"] = metadata["Donor.ID"].astype(str)
            mapping = pd.read_csv(PRIMARY_FREEZE_DIR / "primary_K8_to_K10_reference_mapping.csv")
            mapping["K8_component"] = mapping["K8_component"].map(lambda x: str(int(float(x))))
            assigned_mapping = mapping.loc[mapping.assigned_one_to_one].copy()
            component_to_reference = dict(zip(
                assigned_mapping["K8_component"],
                assigned_mapping["K10_reference_program"].map(lambda x: f"P{int(float(x))}")))

            # Explicitly expose experiment overlap before any disease interpretation.
            design = metadata[["Experiment", "Donor.ID", "Allergic", "Stim"]].drop_duplicates()
            design_counts = (metadata.reset_index(names="cell_id")
                             .groupby(["Experiment", "Allergic", "Stim"], observed=True)
                             .agg(n_cells=("cell_id", "size"), n_donors=("Donor.ID", "nunique"))
                             .reset_index())
            save_table(design_counts, "GSE294268_experiment_design_audit")
            confounding_rows = []
            for allergic_group, stim in (("CR", "CR"), ("MO", "MO")):
                case_design = design.loc[design.Allergic.eq(allergic_group) & design.Stim.eq(stim)]
                control_design = design.loc[design.Allergic.eq("NonA") & design.Stim.eq(stim)]
                case_experiments = sorted(case_design.Experiment.astype(str).unique())
                control_experiments = sorted(control_design.Experiment.astype(str).unique())
                shared = sorted(set(case_experiments) & set(control_experiments))
                shared_case = case_design.loc[case_design.Experiment.astype(str).isin(shared), "Donor.ID"].nunique()
                shared_control = control_design.loc[
                    control_design.Experiment.astype(str).isin(shared), "Donor.ID"].nunique()
                if not shared:
                    status = "not_identifiable_no_shared_experiment"
                else:
                    status = "pooled_contrast_confounded_use_within_experiment_sensitivity_only"
                confounding_rows.append({
                    "contrast": f"{allergic_group}-allergic_vs_NonA_under_{stim}",
                    "case_experiments": ";".join(case_experiments),
                    "control_experiments": ";".join(control_experiments),
                    "shared_experiments": ";".join(shared),
                    "n_shared_experiments": len(shared),
                    "n_case_donors_in_shared_experiments": int(shared_case),
                    "n_control_donors_in_shared_experiments": int(shared_control),
                    "disease_effect_status": status,
                })
            confounding_audit = pd.DataFrame(confounding_rows)
            save_table(confounding_audit, "GSE294268_allergic_experiment_confounding_audit")
            contrast_to_status = dict(zip(
                confounding_audit["contrast"], confounding_audit["disease_effect_status"]))
            contrast_to_shared = dict(zip(
                confounding_audit["contrast"], confounding_audit["shared_experiments"]))

            def aggregate_donor_scores(cell_mask, quality_gate):
                selected_ids = quality.index[np.asarray(cell_mask, dtype=bool)]
                selected_usage = usage.loc[selected_ids]
                selected_baseline = baseline.loc[selected_ids]
                selected_metadata = metadata.loc[selected_ids]
                id_columns = ["cell_id", "Experiment", "Donor.ID", "Allergic", "Stim", "orig.ident"]
                usage_long = (selected_usage.join(selected_metadata).reset_index(names="cell_id")
                              .melt(id_vars=id_columns, value_vars=list(scorer["components"]),
                                    var_name="K8_component", value_name="cNMF_usage"))
                baseline_long = (selected_baseline.join(selected_metadata).reset_index(names="cell_id")
                                 .melt(id_vars=id_columns, value_vars=list(scorer["components"]),
                                       var_name="K8_component", value_name="rank_signature_score"))
                group_keys = ["Experiment", "Donor.ID", "Allergic", "Stim", "K8_component"]
                donor_usage = (usage_long.groupby(group_keys, observed=True)
                               .agg(n_cells=("cell_id", "size"),
                                    median_cNMF_usage=("cNMF_usage", "median"))
                               .reset_index())
                donor_baseline = (baseline_long.groupby(group_keys, observed=True)
                                  .agg(median_rank_signature_score=("rank_signature_score", "median"))
                                  .reset_index())
                output = donor_usage.merge(donor_baseline, on=group_keys, how="inner")
                output["quality_gate"] = quality_gate
                output["K10_reference_program"] = output["K8_component"].map(component_to_reference)
                return output

            gate_masks = {
                "all_cells": np.ones(len(quality), dtype=bool),
                "Gray_q95_accepted": quality["within_Gray_q95"].to_numpy(bool),
                "Gray_q90_strict": quality["within_Gray_q90"].to_numpy(bool),
            }
            donor_scores_by_gate = {
                label: aggregate_donor_scores(mask, label) for label, mask in gate_masks.items()
            }
            donor_scores = pd.concat(donor_scores_by_gate.values(), ignore_index=True)
            save_table(donor_scores_by_gate["all_cells"], "GSE294268_primary_K8_donor_state_scores")
            save_table(donor_scores, "GSE294268_primary_K8_quality_sensitivity_donor_state_scores")

            def bootstrap_median_interval(values, token, n_boot=10000):
                values = np.asarray(values, dtype=float)
                values = values[np.isfinite(values)]
                if len(values) < 2:
                    return np.nan, np.nan
                seed = int(stable_token([SEED, token])[:8], 16)
                rng = np.random.default_rng(seed)
                samples = values[rng.integers(0, len(values), size=(n_boot, len(values)))]
                boot = np.median(samples, axis=1)
                return tuple(np.quantile(boot, [0.025, 0.975]).astype(float))

            def donor_tests_for_metric(value_column, score_type):
                rows = []
                for quality_gate, gate_scores in donor_scores_by_gate.items():
                    for component in scorer["components"]:
                        part = gate_scores.loc[gate_scores.K8_component.eq(component)]
                        for allergic_group, stim in (("CR", "CR"), ("MO", "MO")):
                            contrast = f"{allergic_group}-allergic_vs_NonA_under_{stim}"
                            case = part.loc[
                                part.Allergic.eq(allergic_group) & part.Stim.eq(stim), value_column
                            ].dropna().to_numpy(float)
                            control = part.loc[
                                part.Allergic.eq("NonA") & part.Stim.eq(stim), value_column
                            ].dropna().to_numpy(float)
                            result = mannwhitneyu(case, control, alternative="two-sided", method="auto")
                            rows.append({
                                "score_type": score_type, "quality_gate": quality_gate,
                                "test_family": "allergic_vs_NonA_pooled_experiment_confounded",
                                "contrast": contrast, "Experiment": None,
                                "K8_component": component,
                                "K10_reference_program": component_to_reference.get(component),
                                "n_case_donors": len(case), "n_control_donors": len(control),
                                "effect": 2 * float(result.statistic) / (len(case) * len(control)) - 1,
                                "effect_ci95_low": np.nan, "effect_ci95_high": np.nan,
                                "effect_definition": "Cliffs_delta; positive means allergic higher",
                                "p_value": float(result.pvalue),
                                "inference_status": contrast_to_status[contrast],
                            })
                            shared_experiments = [
                                value for value in contrast_to_shared[contrast].split(";") if value
                            ]
                            for experiment in shared_experiments:
                                within = part.loc[part.Experiment.astype(str).eq(experiment)]
                                within_case = within.loc[
                                    within.Allergic.eq(allergic_group) & within.Stim.eq(stim), value_column
                                ].dropna().to_numpy(float)
                                within_control = within.loc[
                                    within.Allergic.eq("NonA") & within.Stim.eq(stim), value_column
                                ].dropna().to_numpy(float)
                                if len(within_case) and len(within_control):
                                    within_result = mannwhitneyu(
                                        within_case, within_control, alternative="two-sided", method="exact")
                                    rows.append({
                                        "score_type": score_type, "quality_gate": quality_gate,
                                        "test_family": "allergic_vs_NonA_within_experiment_sensitivity",
                                        "contrast": contrast, "Experiment": experiment,
                                        "K8_component": component,
                                        "K10_reference_program": component_to_reference.get(component),
                                        "n_case_donors": len(within_case),
                                        "n_control_donors": len(within_control),
                                        "effect": 2 * float(within_result.statistic) /
                                                  (len(within_case) * len(within_control)) - 1,
                                        "effect_ci95_low": np.nan, "effect_ci95_high": np.nan,
                                        "effect_definition": "Cliffs_delta within one experiment",
                                        "p_value": float(within_result.pvalue),
                                        "inference_status": "small_n_within_experiment_sensitivity_only",
                                    })
                        nona = part.loc[part.Allergic.eq("NonA")].pivot_table(
                            index="Donor.ID", columns="Stim", values=value_column, aggfunc="first")
                        for stim_a, stim_b in (("CR", "MO"), ("CR", "HDMAPP"), ("MO", "HDMAPP")):
                            paired = nona[[stim_a, stim_b]].dropna()
                            difference = (paired[stim_a] - paired[stim_b]).to_numpy(float)
                            p_value = 1.0 if np.allclose(difference, 0) else float(
                                wilcoxon(paired[stim_a], paired[stim_b], alternative="two-sided").pvalue)
                            ci_low, ci_high = bootstrap_median_interval(
                                difference, [score_type, quality_gate, component, stim_a, stim_b])
                            rows.append({
                                "score_type": score_type, "quality_gate": quality_gate,
                                "test_family": "NonA_paired_stimulation",
                                "contrast": f"NonA_{stim_a}_vs_{stim_b}", "Experiment": None,
                                "K8_component": component,
                                "K10_reference_program": component_to_reference.get(component),
                                "n_case_donors": len(paired), "n_control_donors": len(paired),
                                "effect": float(np.median(difference)),
                                "effect_ci95_low": ci_low, "effect_ci95_high": ci_high,
                                "effect_definition": f"median paired score difference; {stim_a} minus {stim_b}",
                                "p_value": p_value,
                                "inference_status": "paired_donor_estimable",
                            })
                output = pd.DataFrame(rows)
                output["q_value_BH_within_family"] = np.nan
                family_keys = ["score_type", "quality_gate", "test_family"]
                for _, indices in output.groupby(family_keys, observed=True).groups.items():
                    output.loc[indices, "q_value_BH_within_family"] = bh_adjust(
                        output.loc[indices, "p_value"].to_numpy(float))
                return output

            cnmf_tests = donor_tests_for_metric("median_cNMF_usage", "cNMF_usage")
            baseline_tests = donor_tests_for_metric(
                "median_rank_signature_score", "rank_signature_score")
            legacy_tests = cnmf_tests.loc[
                cnmf_tests.quality_gate.eq("all_cells") &
                cnmf_tests.test_family.isin([
                    "allergic_vs_NonA_pooled_experiment_confounded", "NonA_paired_stimulation"])
            ].copy()
            legacy_tests["q_value_BH"] = bh_adjust(legacy_tests["p_value"])
            save_table(legacy_tests, "GSE294268_primary_K8_donor_tests")
            save_table(cnmf_tests, "GSE294268_primary_K8_quality_sensitivity_donor_tests")
            save_table(baseline_tests, "GSE294268_primary_K8_rank_signature_donor_tests")

            correlations = []
            for quality_gate, gate_scores in donor_scores_by_gate.items():
                for component, part in gate_scores.groupby("K8_component", observed=True):
                    correlations.append({
                        "quality_gate": quality_gate, "K8_component": component,
                        "K10_reference_program": component_to_reference.get(component),
                        "n_donor_conditions": len(part),
                        "spearman_rho_cNMF_vs_rank_signature": rho(
                            part["median_cNMF_usage"], part["median_rank_signature_score"]),
                    })
            correlations = pd.DataFrame(correlations)
            save_table(correlations, "GSE294268_primary_K8_signature_baseline_comparison")

            comparison_keys = ["quality_gate", "test_family", "contrast", "K8_component"]
            effect_comparison = cnmf_tests.loc[
                cnmf_tests.test_family.eq("NonA_paired_stimulation")
            ].merge(
                baseline_tests.loc[baseline_tests.test_family.eq("NonA_paired_stimulation")],
                on=comparison_keys, suffixes=("_cNMF", "_rank_signature"))
            effect_comparison["same_effect_direction"] = (
                np.sign(effect_comparison["effect_cNMF"]) ==
                np.sign(effect_comparison["effect_rank_signature"]))
            save_table(effect_comparison, "GSE294268_primary_K8_cNMF_vs_rank_signature_effects")

            prevalence_rows = []
            for quality_gate, mask in gate_masks.items():
                selected = usage.loc[quality.index[np.asarray(mask, dtype=bool)]]
                gate_scores = donor_scores_by_gate[quality_gate]
                for component in scorer["components"]:
                    values = selected[component].to_numpy(float)
                    donor_medians = gate_scores.loc[
                        gate_scores.K8_component.eq(component), "median_cNMF_usage"].to_numpy(float)
                    prevalence_rows.append({
                        "quality_gate": quality_gate, "K8_component": component,
                        "K10_reference_program": component_to_reference.get(component),
                        "n_cells": len(values), "fraction_usage_gt_0": float(np.mean(values > 0)),
                        "fraction_usage_gt_0_01": float(np.mean(values > 0.01)),
                        "fraction_usage_gt_0_10": float(np.mean(values > 0.10)),
                        "median_usage": float(np.median(values)),
                        "fraction_donor_condition_median_gt_0": float(np.mean(donor_medians > 0)),
                    })
            prevalence = pd.DataFrame(prevalence_rows)
            save_table(prevalence, "GSE294268_primary_K8_program_prevalence")

            # Quality transportability figure: single-panel ECDF avoids rendering
            # hundreds of thousands of vector points while preserving the distribution.
            fig, ax = plt.subplots(figsize=(6.4, 4.2))
            probabilities = np.linspace(0, 1, 1001)
            for label, values, color in (
                ("Gray donor-held-out K=8", gray_errors, "#0072B2"),
                ("GSE294268 external", external_error, "#D55E00"),
            ):
                ax.plot(np.quantile(values, probabilities), probabilities,
                        label=label, color=color, linewidth=1.6)
            ax.axvline(gray_q90, color="black", linestyle="--", linewidth=0.9,
                       label="Gray q90 / q95 gates")
            ax.axvline(gray_q95, color="black", linestyle=":", linewidth=1.1)
            ax.set(xlabel="Scaled-HVG relative reconstruction error",
                   ylabel="Empirical cumulative fraction",
                   title="External reconstruction-quality shift")
            ax.legend(frameon=False, loc="upper left")
            save_figure(fig, "GSE294268_primary_K8_reconstruction_quality")

            order = (assigned_mapping.assign(
                reference_number=assigned_mapping.K10_reference_program.astype(float))
                .sort_values("reference_number")["K8_component"].tolist())
            paired_plot = cnmf_tests.loc[
                cnmf_tests.quality_gate.eq("Gray_q95_accepted") &
                cnmf_tests.test_family.eq("NonA_paired_stimulation")].copy()
            heat = paired_plot.pivot(
                index="contrast", columns="K8_component", values="effect").reindex(columns=order)
            q_heat = paired_plot.pivot(
                index="contrast", columns="K8_component",
                values="q_value_BH_within_family").reindex(index=heat.index, columns=order)
            limit = max(0.02, float(np.nanmax(np.abs(heat.to_numpy(float)))))
            fig, ax = plt.subplots(figsize=(7.4, 3.8))
            image = ax.imshow(heat.to_numpy(float), cmap="coolwarm", vmin=-limit,
                              vmax=limit, aspect="auto")
            for row_index in range(len(heat.index)):
                for column_index in range(len(heat.columns)):
                    if q_heat.iloc[row_index, column_index] < 0.05:
                        ax.text(column_index, row_index, "•", ha="center", va="center",
                                color="black", fontsize=9)
            labels = [f"{component}→{component_to_reference.get(component, 'unmapped')}"
                      for component in heat.columns]
            paired_contrast_labels = {
                "NonA_CR_vs_HDMAPP": "CR − HDMAPP",
                "NonA_CR_vs_MO": "CR − MO",
                "NonA_MO_vs_HDMAPP": "MO − HDMAPP",
            }
            ax.set(xticks=np.arange(len(labels)), xticklabels=labels,
                   yticks=np.arange(len(heat.index)),
                   yticklabels=[paired_contrast_labels.get(value, value) for value in heat.index],
                   xlabel="K=8 component → K=10 reference", ylabel="Paired donor contrast",
                   title="NonA stimulation responses after Gray-q95 quality gating\n"
                         "Black dot: within-family BH q<0.05; n=15 paired donors")
            ax.tick_params(axis="x", rotation=45)
            colorbar = fig.colorbar(image, ax=ax)
            colorbar.set_label("Median paired usage difference")
            save_figure(fig, "GSE294268_primary_K8_paired_stimulation_effects")

            exploratory_tests = legacy_tests.loc[
                legacy_tests.test_family.eq("allergic_vs_NonA_pooled_experiment_confounded")].copy()
            heat = exploratory_tests.pivot(
                index="contrast", columns="K8_component", values="effect").reindex(columns=order)
            limit = max(0.1, float(np.nanmax(np.abs(heat.to_numpy(float)))))
            fig, ax = plt.subplots(figsize=(7.2, 3.4))
            image = ax.imshow(heat.to_numpy(float), cmap="coolwarm", vmin=-limit,
                              vmax=limit, aspect="auto")
            labels = [f"{component}→{component_to_reference.get(component, 'unmapped')}"
                      for component in heat.columns]
            allergic_contrast_labels = {
                "CR-allergic_vs_NonA_under_CR": "CR-allergic vs NonA (CR)",
                "MO-allergic_vs_NonA_under_MO": "MO-allergic vs NonA (MO)",
            }
            ax.set(xticks=np.arange(len(labels)), xticklabels=labels,
                   yticks=np.arange(len(heat.index)),
                   yticklabels=[allergic_contrast_labels.get(value, value) for value in heat.index],
                   xlabel="K=8 component → K=10 reference", ylabel="Pooled donor contrast",
                   title="Exploratory allergy contrasts (experiment-confounded)")
            ax.tick_params(axis="x", rotation=45)
            colorbar = fig.colorbar(image, ax=ax); colorbar.set_label("Cliff's delta")
            save_figure(fig, "GSE294268_primary_K8_allergic_effects")

            significant_paired = cnmf_tests.loc[
                cnmf_tests.quality_gate.eq("Gray_q95_accepted") &
                cnmf_tests.test_family.eq("NonA_paired_stimulation") &
                cnmf_tests.q_value_BH_within_family.lt(0.05)]
            significant_baseline_paired = baseline_tests.loc[
                baseline_tests.quality_gate.eq("Gray_q95_accepted") &
                baseline_tests.test_family.eq("NonA_paired_stimulation") &
                baseline_tests.q_value_BH_within_family.lt(0.05)]
            q95_prevalence = prevalence.loc[
                prevalence.quality_gate.eq("Gray_q95_accepted")].copy()
            external_program_support = {
                str(row.K10_reference_program): {
                    "K8_component": str(row.K8_component),
                    "fraction_usage_gt_0_10": float(row.fraction_usage_gt_0_10),
                    "median_usage": float(row.median_usage),
                    "fraction_donor_condition_median_gt_0": float(
                        row.fraction_donor_condition_median_gt_0),
                }
                for row in q95_prevalence.itertuples(index=False)
            }
            model_card = {
                "schema_version": 1, "analysis": ANALYSIS_NAME,
                "model": "full-Gray primary K=8 fixed basis",
                "status": "research-grade frozen scorer; biological validation incomplete",
                "intended_use": "continuous gamma-delta T-cell state-program usage scoring from raw UMI counts",
                "not_intended_for": [
                    "gamma-delta lineage classification", "Vd1/Vd2 subtype prediction",
                    "mutually exclusive cell-state labels", "clinical diagnosis",
                    "population-prevalence inference from AIM-selected GSE294268 cells",
                ],
                "required_inputs": {
                    "matrix_kind": "raw UMI counts", "minimum_frozen_HVG_coverage": MIN_HVG_COVERAGE,
                    "gene_identifier": "gene symbol", "normalization": "TPM over frozen training universe",
                },
                "artifacts": {
                    "scoring_spec": str(scoring_spec_path), "scoring_spec_sha256": sha256sum(scoring_spec_path),
                    "spectra": str(spectra_path), "spectra_sha256": sha256sum(spectra_path),
                    "training_sd": str(sd_path), "training_sd_sha256": sha256sum(sd_path),
                },
                "quality_gate": {
                    "metric": "scaled_HVG_relative_error",
                    "high_confidence_max_Gray_q90": gray_q90,
                    "accepted_max_Gray_q95": gray_q95,
                    "abstain_above": gray_q95,
                    "primary_external_analysis_gate": "Gray_q95_accepted",
                    "strict_sensitivity_gate": "Gray_q90_strict",
                    "calibration_population": "donor-held-out Gray cells from five K=8 outer folds",
                    "external_fraction_accepted_q95": float(quality["within_Gray_q95"].mean()),
                    "external_fraction_strict_q90": float(quality["within_Gray_q90"].mean()),
                },
                "program_mapping_to_K10_reference": component_to_reference,
                "external_program_support_after_q95_gate": external_program_support,
                "validation_readout": {
                    "donor_separated_internal_validation": "completed",
                    "external_raw_UMI_projection": "completed for 13 libraries",
                    "external_paired_stimulation_validation": (
                        f"{len(significant_paired)} component-contrast tests at within-family BH q<0.05 "
                        "after Gray-q95 filtering"),
                    "disease_validation": "not established; allergy is confounded with Experiment",
                    "incremental_value_over_rank_signature": (
                        "not established; under the same Gray-q95 paired-donor design, "
                        f"cNMF has {len(significant_paired)}/24 and the rank-signature baseline has "
                        f"{len(significant_baseline_paired)}/24 within-family BH q<0.05 tests"),
                    "orthogonal_protein_or_functional_validation": "not completed",
                },
                "readiness": {
                    "scoring_model_frozen": True,
                    "quality_abstention_defined": True,
                    "external_state_response_supported": True,
                    "disease_effect_identifiable": False,
                    "incremental_value_over_simple_signature_established": False,
                    "orthogonal_functional_validation_completed": False,
                    "NC_claim_package_complete": False,
                },
                "limitations": [
                    "GSE294268 contains AIM-selected stimulated cells and is not a population-prevalence cohort.",
                    "A substantial external reconstruction-error tail exceeds the Gray held-out reference.",
                    "Pooled allergic-versus-NonA effects are confounded by Experiment.",
                    "Rank-signature comparisons are baselines, not orthogonal functional validation.",
                    "Low external usage for a component is absence of support, not proof of biological absence.",
                ],
            }
            model_card_path = PRIMARY_FREEZE_DIR / "primary_K8_model_card.json"
            atomic_json(model_card_path, model_card)
            atomic_json(TABLE_DIR / "primary_K8_model_card.json", model_card)

            atomic_json(TABLE_DIR / "GSE294268_primary_K8_projection_manifest.json", {
                "schema_version": 2, "analysis": ANALYSIS_NAME,
                "model": "full-Gray primary K=8 fixed basis", "n_libraries": len(raw_inventory),
                "n_cells": len(usage),
                "n_donor_condition_rows_all_cells": len(donor_scores_by_gate["all_cells"]),
                "usage_path": str(usage_path), "usage_sha256": sha256sum(usage_path),
                "quality_path": str(quality_path), "quality_sha256": sha256sum(quality_path),
                "quality_gated_usage_path": str(gated_usage_path),
                "quality_gated_usage_sha256": sha256sum(gated_usage_path),
                "baseline_path": str(baseline_path), "baseline_sha256": sha256sum(baseline_path),
                "model_card_path": str(model_card_path), "model_card_sha256": sha256sum(model_card_path),
                "external_data_used_for_K_selection": False,
                "Gray_q90_relative_error_gate": gray_q90,
                "Gray_q95_relative_error_gate": gray_q95,
                "external_fraction_accepted_q95": float(quality["within_Gray_q95"].mean()),
                "limitations": model_card["limitations"],
            })
            note("GSE294268 primary K8 projection", "completed_with_quality_gate",
                 f"{len(usage)} cells; donor-level paired inference; disease contrasts flagged as confounded")
            display(cnmf_tests.sort_values("q_value_BH_within_family").head(24))

        if EXECUTION_STAGE == "gse294268_score":
            run_gse294268_projection()
        else:
            print("GSE294268 raw projection not launched in stage", EXECUTION_STAGE)
        """),
        md(r"""
        ## 5 · PANGEA external-atlas transportability

        This stage is deliberately narrower than a functional validation. PANGEA
        contributes raw UMI counts, broad study/tissue/chemistry diversity and
        expression-derived `T_gdT_TRDV1` / `T_gdT_TRDV2` silver labels, but no
        protein, degranulation, cytokine or killing endpoint in this object.

        The primary analysis excludes exact `Series == "GSE149356"`, which was used
        in earlier project evaluation. The highest consistently available grouping
        variables are `Series` and `Sample`; therefore `Series × Sample` is reported
        as a **sample proxy**, never silently relabelled as a donor. Checkpoints are
        written per Series. Subtype contrasts first aggregate within sample proxy and
        then within Series, so cell count does not become the inferential sample size.
        PANGEA cannot change K, spectra, program names or Gray-derived quality gates.
        """),
        code(r"""
        # 11. Score PANGEA gamma-delta cells with the locked K=8 basis
        PANGEA_PROJECTION_DIR = MODEL_DIR / "PANGEA_primary_K8_projection"
        PANGEA_PROJECTION_CHECKPOINTS = CHECKPOINT_DIR / "PANGEA_primary_K8_projection"
        PANGEA_EXCLUDED_SERIES = ("GSE149356",)

        def load_primary_k8_scorer():
            import yaml

            primary_spec_path = PRIMARY_JOB / "job_spec.json"
            primary_marker = PRIMARY_FREEZE_DIR / "frozen.complete.json"
            if not primary_spec_path.is_file() or not validate_marker(
                    primary_marker, stable_token(json.loads(primary_spec_path.read_text()))):
                raise RuntimeError("The frozen primary K8 basis is absent or failed checksum validation")
            scoring_spec_path = PRIMARY_FREEZE_DIR / "primary_K8_scoring_spec.json"
            spectra_path = PRIMARY_FREEZE_DIR / "primary_K8_HVG_TPM_spectra.tsv"
            sd_path = PRIMARY_FREEZE_DIR / "primary_K8_training_sd.csv"
            params_path = one_file(PRIMARY_JOB, "*.nmf_idvrun_params.yaml")
            scoring_spec = json.loads(scoring_spec_path.read_text())
            spectra = pd.read_csv(spectra_path, sep="\t", index_col=0)
            spectra.index = spectra.index.map(lambda x: str(int(float(x))))
            sd = pd.read_csv(sd_path, index_col=0).iloc[:, 0]
            universe = pd.Index(scoring_spec["universe"])
            hvgs = pd.Index(scoring_spec["hvgs"])
            hvg_positions = universe.get_indexer(hvgs)
            if ((hvg_positions < 0).any() or list(spectra.columns) != list(hvgs)
                    or list(sd.index) != list(hvgs)):
                raise ValueError("Frozen K8 scorer gene definitions are inconsistent")
            ranked = pd.read_csv(PRIMARY_FREEZE_DIR / "primary_K8_ranked_genes.csv")
            ranked["K8_component"] = ranked["K8_component"].map(
                lambda x: str(int(float(x))))
            signatures = {
                component: part.sort_values("rank").head(50)["gene"].astype(str).tolist()
                for component, part in ranked.groupby("K8_component", observed=True)
            }
            return {
                "spec_path": scoring_spec_path, "spectra_path": spectra_path,
                "sd_path": sd_path, "params_path": params_path, "universe": universe,
                "hvgs": hvgs, "hvg_positions": hvg_positions, "sd": sd,
                "basis": spectra.div(sd, axis=1).to_numpy(np.float64),
                "components": spectra.index.astype(str), "signatures": signatures,
                "solver_params": yaml.safe_load(params_path.read_text()),
            }

        def score_pangea_series(series, row_positions, cell_ids, count_layer,
                                 collapse, scorer, source_record):
            series = str(series)
            job = PANGEA_PROJECTION_CHECKPOINTS / series
            job.mkdir(parents=True, exist_ok=True)
            spec = {
                "schema_version": 1, "analysis": ANALYSIS_NAME, "dataset": "PANGEA",
                "Series": series, "n_cells": len(cell_ids),
                "cell_ids_sha256": sha256_lines(cell_ids),
                "PANGEA_sha256": source_record["sha256"],
                "excluded_Series": list(PANGEA_EXCLUDED_SERIES),
                "cell_eligibility": ["T_gdT_TRDV1", "T_gdT_TRDV2"],
                "count_source": 'layers["count"]',
                "scoring_spec_sha256": sha256sum(scorer["spec_path"]),
                "spectra_sha256": sha256sum(scorer["spectra_path"]),
                "sd_sha256": sha256sum(scorer["sd_path"]),
                "solver_params_sha256": sha256sum(scorer["params_path"]),
            }
            signature = stable_token(spec)
            spec_path = job / "job_spec.json"
            if spec_path.exists() and json.loads(spec_path.read_text()) != spec:
                raise ValueError(f"PANGEA Series definition changed: {job}")
            atomic_json(spec_path, spec)
            marker = job / "scoring.complete.json"
            if validate_marker(marker, signature):
                note(f"PANGEA {series}", "reused", "validated raw-UMI Series checkpoint")
                return

            raw = sparse.csr_matrix(count_layer[np.asarray(row_positions, int), :], dtype=float)
            counts = (raw @ collapse).tocsr()
            library_size = np.asarray(counts.sum(axis=1)).ravel()
            if (library_size <= 0).any():
                raise ValueError(f"PANGEA cells with zero counts over training universe: {series}")
            tpm = sparse.diags(1e6 / library_size) @ counts
            objective = tpm[:, scorer["hvg_positions"]].multiply(
                1 / scorer["sd"].to_numpy()).tocsr().astype(np.float64)
            params = dict(scorer["solver_params"])
            params.update(H=scorer["basis"], update_H=False, n_components=PRIMARY_K)
            usage_blocks, relative_error = [], []
            gram = scorer["basis"] @ scorer["basis"].T
            from threadpoolctl import threadpool_limits
            for start in range(0, objective.shape[0], 5000):
                block = objective[start:start + 5000]
                np.random.seed(SEED)
                with threadpool_limits(limits=THREADS):
                    coefficients, _, _ = non_negative_factorization(block, **params)
                usage_blocks.append(coefficients)
                x_norm_sq = np.asarray(block.multiply(block).sum(axis=1)).ravel()
                cross = np.sum(coefficients * (block @ scorer["basis"].T), axis=1)
                fitted_norm_sq = np.sum((coefficients @ gram) * coefficients, axis=1)
                residual_sq = np.maximum(x_norm_sq + fitted_norm_sq - 2 * cross, 0)
                relative_error.extend(
                    np.sqrt(residual_sq) / np.maximum(np.sqrt(x_norm_sq), 1e-15))
            raw_usage = pd.DataFrame(
                np.vstack(usage_blocks), index=cell_ids, columns=scorer["components"])
            normalized_usage = normalize_usage(raw_usage)
            if normalized_usage.isna().any().any():
                raise ValueError(f"PANGEA normalized usages contain missing values: {series}")
            quality = pd.DataFrame({
                "library_over_training_universe": library_size,
                "scaled_HVG_relative_error": relative_error, "Series": series,
            }, index=cell_ids)
            signature_scores = rank_scores(counts, scorer["universe"], scorer["signatures"])
            signature_scores.index = cell_ids
            usage_path = atomic_parquet(normalized_usage, job / "usage_normalized.parquet")
            quality_path = atomic_parquet(quality, job / "quality.parquet")
            baseline_path = atomic_parquet(
                signature_scores, job / "rank_signature_scores.parquet")
            write_marker(marker, signature, [usage_path, quality_path, baseline_path])
            note(f"PANGEA {series}", "completed", f"{len(cell_ids)} gamma-delta cells")

        def run_pangea_projection():
            from scipy.stats import wilcoxon

            scorer = load_primary_k8_scorer()
            if not PANGEA_PATH.is_file():
                raise FileNotFoundError(PANGEA_PATH)
            source_record = record_source(PANGEA_PATH, full_hash=True)
            pangea = ad.read_h5ad(PANGEA_PATH, backed="r")
            required_obs = {
                "Series", "Sample", "Chemistry", "Category", "Disease", "Organ",
                "annotation_Level2",
            }
            missing_obs = sorted(required_obs - set(pangea.obs.columns))
            if missing_obs:
                raise KeyError(f"PANGEA metadata columns missing: {missing_obs}")
            if "count" not in pangea.layers:
                raise KeyError('PANGEA raw counts must be present as layers["count"]')
            genes = pd.Index(pangea.var_names.astype(str), name="gene")
            if not genes.is_unique:
                raise ValueError("PANGEA gene names must be unique before frozen scoring")
            if not pangea.obs_names.is_unique:
                raise ValueError("PANGEA cell identifiers must be unique")

            labels = pangea.obs["annotation_Level2"].astype(str)
            eligible = labels.isin(["T_gdT_TRDV1", "T_gdT_TRDV2"])
            excluded = pangea.obs["Series"].astype(str).isin(PANGEA_EXCLUDED_SERIES)
            primary = eligible & ~excluded
            metadata = pangea.obs.loc[primary, sorted(required_obs)].copy()
            metadata.index = metadata.index.astype(str)
            metadata["sample_proxy"] = (
                metadata["Series"].astype(str) + "::" + metadata["Sample"].astype(str))
            row_numbers = np.flatnonzero(primary.to_numpy())
            row_lookup = pd.Series(row_numbers, index=metadata.index)

            hvg_coverage = float(scorer["hvgs"].isin(genes).mean())
            universe_coverage = float(scorer["universe"].isin(genes).mean())
            if hvg_coverage < MIN_HVG_COVERAGE:
                raise RuntimeError(
                    f"PANGEA frozen-HVG coverage {hvg_coverage:.3f} is below "
                    f"the locked {MIN_HVG_COVERAGE:.2f} minimum")
            universe_positions = scorer["universe"].get_indexer(genes)
            valid_features = universe_positions >= 0
            collapse = sparse.csr_matrix((
                np.ones(valid_features.sum()),
                (np.flatnonzero(valid_features), universe_positions[valid_features])),
                shape=(len(genes), len(scorer["universe"])))

            input_audit = pd.DataFrame([
                {"check": "PANGEA_shape", "value": str(pangea.shape), "status": "pass"},
                {"check": "raw_count_layer", "value": "count", "status": "pass"},
                {"check": "frozen_HVG_coverage", "value": hvg_coverage,
                 "status": "pass" if hvg_coverage >= MIN_HVG_COVERAGE else "fail"},
                {"check": "training_universe_coverage", "value": universe_coverage, "status": "pass"},
                {"check": "eligible_gdT_cells_before_exclusion", "value": int(eligible.sum()), "status": "pass"},
                {"check": "excluded_GSE149356_cells", "value": int((eligible & excluded).sum()), "status": "pass"},
                {"check": "primary_gdT_cells", "value": int(primary.sum()), "status": "pass"},
                {"check": "primary_Series", "value": metadata["Series"].nunique(), "status": "pass"},
                {"check": "primary_sample_proxies", "value": metadata["sample_proxy"].nunique(), "status": "pass"},
                {"check": "independent_unit", "value": "Series (sample proxy nested within Series)", "status": "pass"},
                {"check": "label_evidence", "value": "expression-derived PANGEA silver annotation", "status": "pass"},
                {"check": "untouched_final_cohort", "value": False, "status": "pass"},
            ])
            save_table(input_audit, "PANGEA_primary_K8_input_audit")

            PANGEA_PROJECTION_DIR.mkdir(parents=True, exist_ok=True)
            PANGEA_PROJECTION_CHECKPOINTS.mkdir(parents=True, exist_ok=True)
            count_layer = pangea.layers["count"]
            for series in sorted(metadata["Series"].astype(str).unique()):
                cell_ids = metadata.index[metadata["Series"].astype(str).eq(series)]
                positions = row_lookup.loc[cell_ids].to_numpy(int)
                score_pangea_series(
                    series, positions, cell_ids, count_layer, collapse, scorer, source_record)

            usage_parts, quality_parts, baseline_parts = [], [], []
            for series in sorted(metadata["Series"].astype(str).unique()):
                job = PANGEA_PROJECTION_CHECKPOINTS / series
                usage_parts.append(pd.read_parquet(job / "usage_normalized.parquet"))
                quality_parts.append(pd.read_parquet(job / "quality.parquet"))
                baseline_parts.append(pd.read_parquet(job / "rank_signature_scores.parquet"))
            usage = pd.concat(usage_parts).loc[metadata.index]
            quality = pd.concat(quality_parts).loc[metadata.index]
            baseline = pd.concat(baseline_parts).loc[metadata.index]
            if not np.allclose(usage.sum(axis=1).to_numpy(float), 1.0, atol=1e-8):
                raise ValueError("Normalized PANGEA K8 usages must sum to one")

            gray_quality_paths = [
                CHECKPOINT_DIR / "full" / f"fold{fold}_K8" / "heldout_quality.parquet"
                for fold in range(N_FOLDS)
            ]
            missing_gray = [str(path) for path in gray_quality_paths if not path.is_file()]
            if missing_gray:
                raise FileNotFoundError("Missing Gray K8 quality references: " + ", ".join(missing_gray))
            gray_errors = pd.concat([
                pd.read_parquet(path)["scaled_HVG_relative_error"] for path in gray_quality_paths
            ]).astype(float)
            gray_q90, gray_q95, gray_q99 = [
                float(gray_errors.quantile(q)) for q in (0.90, 0.95, 0.99)]
            external_error = quality["scaled_HVG_relative_error"].astype(float)
            quality["within_Gray_q90"] = external_error.le(gray_q90)
            quality["within_Gray_q95"] = external_error.le(gray_q95)
            quality["reconstruction_gate"] = np.select(
                [quality["within_Gray_q90"], quality["within_Gray_q95"]],
                ["high_confidence_within_q90", "accepted_q90_to_q95"],
                default="abstain_above_q95")

            usage_path = atomic_parquet(
                usage, PANGEA_PROJECTION_DIR / "PANGEA_primary_K8_usage_normalized.parquet")
            quality_path = atomic_parquet(
                quality, PANGEA_PROJECTION_DIR / "PANGEA_primary_K8_quality.parquet")
            baseline_path = atomic_parquet(
                baseline, PANGEA_PROJECTION_DIR / "PANGEA_primary_K8_rank_signature_scores.parquet")
            metadata_path = atomic_parquet(
                metadata, PANGEA_PROJECTION_DIR / "PANGEA_primary_K8_cell_metadata.parquet")
            gated_usage_path = atomic_parquet(
                usage.where(quality["within_Gray_q95"], np.nan),
                PANGEA_PROJECTION_DIR / "PANGEA_primary_K8_usage_q95_quality_gated.parquet")

            mapping = pd.read_csv(PRIMARY_FREEZE_DIR / "primary_K8_to_K10_reference_mapping.csv")
            mapping["K8_component"] = mapping["K8_component"].map(
                lambda x: str(int(float(x))))
            assigned = mapping.loc[mapping.assigned_one_to_one].copy()
            component_to_reference = dict(zip(
                assigned["K8_component"],
                assigned["K10_reference_program"].map(lambda x: f"P{int(float(x))}")))

            quality_summary = []
            for source, values in (("Gray_donor_heldout_K8", gray_errors),
                                   ("PANGEA_external", external_error)):
                quality_summary.append({
                    "source": source, "n_cells": len(values),
                    "median_relative_error": float(values.median()),
                    "q90_relative_error": float(values.quantile(0.90)),
                    "q95_relative_error": float(values.quantile(0.95)),
                    "q99_relative_error": float(values.quantile(0.99)),
                    "fraction_above_Gray_q90": float(values.gt(gray_q90).mean()),
                    "fraction_above_Gray_q95": float(values.gt(gray_q95).mean()),
                    "fraction_above_Gray_q99": float(values.gt(gray_q99).mean()),
                })
            quality_summary = pd.DataFrame(quality_summary)
            save_table(quality_summary, "PANGEA_primary_K8_reconstruction_quality_gate")

            series_quality = (quality.join(metadata[["sample_proxy", "Chemistry", "Organ"]])
                              .groupby("Series", observed=True)
                              .agg(n_cells=("scaled_HVG_relative_error", "size"),
                                   n_sample_proxies=("sample_proxy", "nunique"),
                                   median_relative_error=("scaled_HVG_relative_error", "median"),
                                   fraction_within_Gray_q90=("within_Gray_q90", "mean"),
                                   fraction_within_Gray_q95=("within_Gray_q95", "mean"),
                                   n_chemistries=("Chemistry", "nunique"),
                                   n_organs=("Organ", "nunique"))
                              .reset_index())
            save_table(series_quality, "PANGEA_primary_K8_series_quality")

            context_quality_rows = []
            quality_with_context = quality.join(metadata[[
                "sample_proxy", "Chemistry", "Organ", "Category", "annotation_Level2"]])
            for field in ("Chemistry", "Organ", "Category", "annotation_Level2"):
                for value, part in quality_with_context.groupby(field, observed=True):
                    context_quality_rows.append({
                        "field": field, "value": value, "n_cells": len(part),
                        "n_sample_proxies": part["sample_proxy"].nunique(),
                        "n_Series": part["Series"].nunique(),
                        "median_relative_error": float(
                            part["scaled_HVG_relative_error"].median()),
                        "fraction_within_Gray_q90": float(part["within_Gray_q90"].mean()),
                        "fraction_within_Gray_q95": float(part["within_Gray_q95"].mean()),
                        "inference_status": "descriptive quality audit; contexts may be Series-confounded",
                    })
            context_quality = pd.DataFrame(context_quality_rows)
            save_table(context_quality, "PANGEA_primary_K8_quality_by_context")

            gate_masks = {
                "all_cells": np.ones(len(quality), dtype=bool),
                "Gray_q95_accepted": quality["within_Gray_q95"].to_numpy(bool),
                "Gray_q90_strict": quality["within_Gray_q90"].to_numpy(bool),
            }
            sample_score_parts = []
            for gate_name, mask in gate_masks.items():
                ids = quality.index[mask]
                common = ["Series", "Sample", "sample_proxy", "annotation_Level2"]
                for score_type, frame, value_name in (
                        ("cNMF_usage", usage, "score"),
                        ("rank_signature_score", baseline, "score")):
                    long = (frame.loc[ids].join(metadata[common]).reset_index(names="cell_id")
                            .melt(id_vars=["cell_id"] + common,
                                  value_vars=list(scorer["components"]),
                                  var_name="K8_component", value_name=value_name))
                    summary = (long.groupby(
                        ["Series", "Sample", "sample_proxy", "annotation_Level2", "K8_component"],
                        observed=True)[value_name].median().rename("median_score").reset_index())
                    summary["score_type"] = score_type
                    summary["quality_gate"] = gate_name
                    sample_score_parts.append(summary)
            sample_scores = pd.concat(sample_score_parts, ignore_index=True)
            save_table(sample_scores, "PANGEA_primary_K8_sample_state_scores")

            def median_ci(values, token, draws=4000):
                values = np.asarray(values, float)
                if not len(values):
                    return np.nan, np.nan
                seed = SEED + int(hashlib.sha256(str(token).encode()).hexdigest()[:8], 16)
                rng = np.random.default_rng(seed)
                estimates = np.median(
                    rng.choice(values, size=(draws, len(values)), replace=True), axis=1)
                return tuple(np.quantile(estimates, [0.025, 0.975]))

            test_rows = []
            for (score_type, gate_name, component), part in sample_scores.groupby(
                    ["score_type", "quality_gate", "K8_component"], observed=True):
                wide = part.pivot_table(
                    index=["Series", "sample_proxy"], columns="annotation_Level2",
                    values="median_score", aggfunc="first")
                required = ["T_gdT_TRDV1", "T_gdT_TRDV2"]
                if not set(required).issubset(wide.columns):
                    paired = pd.DataFrame(columns=required)
                else:
                    paired = wide[required].dropna()
                sample_differences = paired[required[0]] - paired[required[1]]
                # ``Series`` is categorical in the source h5ad. Explicitly keep
                # observed groups only; otherwise pandas materialises unused
                # atlas-wide categories as NaN pseudo-Series.
                series_differences = (sample_differences.groupby(
                    level="Series", observed=True).median().dropna())
                if len(series_differences) >= 2 and not np.allclose(series_differences, 0):
                    p_value = float(wilcoxon(series_differences, alternative="two-sided").pvalue)
                else:
                    p_value = 1.0
                ci_low, ci_high = median_ci(
                    series_differences.to_numpy(float),
                    [score_type, gate_name, component, "TRDV1_minus_TRDV2"])
                test_rows.append({
                    "score_type": score_type, "quality_gate": gate_name,
                    "contrast": "T_gdT_TRDV1_minus_T_gdT_TRDV2",
                    "K8_component": component,
                    "K10_reference_program": component_to_reference.get(component),
                    "n_paired_sample_proxies": len(paired),
                    "n_independent_Series": len(series_differences),
                    "effect": float(np.median(series_differences)) if len(series_differences) else np.nan,
                    "effect_ci95_low": ci_low, "effect_ci95_high": ci_high,
                    "effect_definition": "median Series-level difference of paired sample-proxy medians",
                    "p_value": p_value,
                    "inference_unit": "Series",
                    "evidence_boundary": "expression-derived silver-label association; not subtype validation",
                })
            subtype_tests = pd.DataFrame(test_rows)
            if subtype_tests["n_independent_Series"].max() > metadata["Series"].nunique():
                raise AssertionError("Subtype tests contain unobserved PANGEA Series categories")
            estimable = subtype_tests["n_independent_Series"].gt(0)
            if subtype_tests.loc[estimable, "effect"].isna().any():
                raise AssertionError("Estimable PANGEA subtype effects must be finite")
            subtype_tests["q_value_BH_within_score_gate"] = np.nan
            for _, indices in subtype_tests.groupby(
                    ["score_type", "quality_gate"], observed=True).groups.items():
                subtype_tests.loc[indices, "q_value_BH_within_score_gate"] = bh_adjust(
                    subtype_tests.loc[indices, "p_value"].to_numpy(float))
            save_table(subtype_tests, "PANGEA_primary_K8_TRDV1_TRDV2_series_tests")

            comparison = subtype_tests.loc[
                subtype_tests.quality_gate.eq("Gray_q95_accepted")
            ].pivot_table(index=["K8_component", "K10_reference_program"],
                          columns="score_type", values=["effect", "q_value_BH_within_score_gate"])
            comparison.columns = [f"{left}_{right}" for left, right in comparison.columns]
            comparison = comparison.reset_index()
            if {"effect_cNMF_usage", "effect_rank_signature_score"}.issubset(comparison.columns):
                comparison["same_effect_direction"] = (
                    np.sign(comparison["effect_cNMF_usage"]) ==
                    np.sign(comparison["effect_rank_signature_score"]))
            save_table(comparison, "PANGEA_primary_K8_cNMF_vs_rank_signature_subtype_effects")

            prevalence_rows = []
            for gate_name, mask in gate_masks.items():
                ids = quality.index[mask]
                overall_sample = (usage.loc[ids].join(metadata[["Series", "sample_proxy"]])
                                  .groupby(["Series", "sample_proxy"], observed=True)
                                  [list(scorer["components"])].median())
                series_medians = (overall_sample.groupby(
                    level="Series", observed=True).median().dropna(how="all"))
                for component in scorer["components"]:
                    values = usage.loc[ids, component].to_numpy(float)
                    sample_values = overall_sample[component].to_numpy(float)
                    series_values = series_medians[component].to_numpy(float)
                    prevalence_rows.append({
                        "quality_gate": gate_name, "K8_component": component,
                        "K10_reference_program": component_to_reference.get(component),
                        "n_cells": len(values), "n_sample_proxies": len(sample_values),
                        "n_Series": len(series_values),
                        "median_usage": float(np.median(values)) if len(values) else np.nan,
                        "fraction_cells_usage_gt_0_10": float(np.mean(values > 0.10)) if len(values) else np.nan,
                        "fraction_sample_medians_gt_0": float(np.mean(sample_values > 0)) if len(sample_values) else np.nan,
                        "fraction_Series_medians_gt_0": float(np.mean(series_values > 0)) if len(series_values) else np.nan,
                        "fraction_Series_medians_gt_0_10": float(np.mean(series_values > 0.10)) if len(series_values) else np.nan,
                    })
            prevalence = pd.DataFrame(prevalence_rows)
            if prevalence["n_Series"].max() > metadata["Series"].nunique():
                raise AssertionError("Program prevalence contains unobserved PANGEA Series categories")
            save_table(prevalence, "PANGEA_primary_K8_program_prevalence")

            context_rows = []
            for field in ("Organ", "Disease", "Category", "Chemistry", "annotation_Level2"):
                context = metadata.reset_index(names="cell_id").groupby(field, observed=True)
                for value, part in context:
                    context_rows.append({
                        "field": field, "value": value, "n_cells": len(part),
                        "n_sample_proxies": part["sample_proxy"].nunique(),
                        "n_Series": part["Series"].nunique(),
                        "inference_status": "descriptive_only; inspect Series/context confounding before testing",
                    })
            save_table(pd.DataFrame(context_rows), "PANGEA_primary_K8_context_design_audit")

            probabilities = np.linspace(0, 1, 1001)
            fig, ax = plt.subplots(figsize=(6.4, 4.2))
            for label, values, color in (
                    ("Gray donor-held-out K=8", gray_errors, "#0072B2"),
                    ("PANGEA gamma-delta", external_error, "#D55E00")):
                ax.plot(np.quantile(values, probabilities), probabilities,
                        label=label, color=color, linewidth=1.6)
            ax.axvline(gray_q90, color="black", linestyle="--", linewidth=0.9,
                       label="Gray q90 / q95 gates")
            ax.axvline(gray_q95, color="black", linestyle=":", linewidth=1.1)
            ax.set(xlabel="Scaled-HVG relative reconstruction error",
                   ylabel="Empirical cumulative fraction",
                   title="PANGEA external-atlas reconstruction quality")
            ax.legend(frameon=False, loc="upper left")
            save_figure(fig, "PANGEA_primary_K8_reconstruction_quality")

            order = (assigned.assign(
                reference_number=assigned.K10_reference_program.astype(float))
                .sort_values("reference_number")["K8_component"].tolist())
            plot_tests = subtype_tests.loc[
                subtype_tests.quality_gate.eq("Gray_q95_accepted")].copy()
            heat = plot_tests.pivot(
                index="score_type", columns="K8_component", values="effect").reindex(
                    index=["cNMF_usage", "rank_signature_score"], columns=order)
            q_heat = plot_tests.pivot(
                index="score_type", columns="K8_component",
                values="q_value_BH_within_score_gate").reindex(
                    index=heat.index, columns=heat.columns)
            limit = max(0.02, float(np.nanmax(np.abs(heat.to_numpy(float)))))
            fig, ax = plt.subplots(figsize=(7.3, 3.2))
            image = ax.imshow(heat.to_numpy(float), cmap="coolwarm", vmin=-limit,
                              vmax=limit, aspect="auto")
            for row_index in range(len(heat.index)):
                for column_index in range(len(heat.columns)):
                    if q_heat.iloc[row_index, column_index] < 0.05:
                        ax.text(column_index, row_index, "•", ha="center", va="center",
                                color="black", fontsize=9)
            labels = [f"{component}→{component_to_reference.get(component, 'unmapped')}"
                      for component in heat.columns]
            ax.set(xticks=np.arange(len(labels)), xticklabels=labels,
                   yticks=np.arange(2), yticklabels=["cNMF usage", "Rank signature"],
                   xlabel="K=8 component → K=10 reference", ylabel="Frozen score",
                   title="PANGEA TRDV1 − TRDV2 association\nSeries-level inference; silver labels")
            ax.tick_params(axis="x", rotation=45)
            colorbar = fig.colorbar(image, ax=ax)
            colorbar.set_label("Median Series-level difference")
            save_figure(fig, "PANGEA_primary_K8_TRDV1_TRDV2_series_effects")

            support = prevalence.loc[
                prevalence.quality_gate.eq("Gray_q95_accepted")].set_index(
                    "K8_component").reindex(order).reset_index()
            fig, ax = plt.subplots(figsize=(6.8, 4.0))
            x = np.arange(len(support))
            ax.bar(x, support["fraction_Series_medians_gt_0"], color="#56B4E9",
                   label="Series median > 0")
            ax.scatter(x, support["fraction_Series_medians_gt_0_10"], color="black", s=20,
                       label="Series median > 0.10", zorder=3)
            ax.set(xticks=x,
                   xticklabels=[f"{c}→{p}" for c, p in zip(
                       support.K8_component, support.K10_reference_program)],
                   ylim=(0, 1.03), xlabel="K=8 component → K=10 reference",
                   ylabel="Fraction of PANGEA Series",
                   title="Cross-study program support after Gray-q95 gating")
            ax.tick_params(axis="x", rotation=45)
            ax.legend(frameon=False)
            save_figure(fig, "PANGEA_primary_K8_cross_Series_program_support")

            model_card_path = PRIMARY_FREEZE_DIR / "primary_K8_model_card.json"
            if not model_card_path.is_file():
                raise FileNotFoundError(model_card_path)
            model_card = json.loads(model_card_path.read_text())
            model_card["schema_version"] = max(2, int(model_card.get("schema_version", 1)))
            model_card["updated_at"] = datetime.now(timezone.utc).isoformat()
            model_card.setdefault("validation_readout", {})[
                "external_PANGEA_raw_UMI_projection"] = (
                    f"completed for {len(usage)} gamma-delta cells across "
                    f"{metadata['Series'].nunique()} Series after excluding GSE149356")
            model_card["validation_readout"]["external_PANGEA_evidence_tier"] = (
                "atlas transportability and expression-derived silver-label association; "
                "not untouched or orthogonal functional validation")
            model_card.setdefault("readiness", {})[
                "external_atlas_transportability_assessed"] = True
            model_card.setdefault("quality_gate", {})[
                "PANGEA_fraction_accepted_q95"] = float(quality["within_Gray_q95"].mean())
            model_card["quality_gate"][
                "PANGEA_fraction_strict_q90"] = float(quality["within_Gray_q90"].mean())
            model_card["quality_gate"]["PANGEA_Series_with_q95_accepted_cells"] = int(
                metadata.loc[quality["within_Gray_q95"], "Series"].nunique())
            for label in ("T_gdT_TRDV1", "T_gdT_TRDV2"):
                label_ids = metadata.index[metadata["annotation_Level2"].astype(str).eq(label)]
                model_card["quality_gate"][
                    f"PANGEA_{label}_fraction_accepted_q95"] = float(
                        quality.loc[label_ids, "within_Gray_q95"].mean())
            model_card["readiness"]["untouched_external_validation_completed"] = False
            additional_limitations = [
                "PANGEA was inspected in notebooks 16-18 and is not an untouched final cohort.",
                "PANGEA gamma-delta labels are expression-derived silver annotations, not receptor truth.",
                "PANGEA exposes Series and Sample but no verified donor identifier; inference uses Series.",
                "The PANGEA object contains no orthogonal protein or functional endpoint for this scorer.",
            ]
            model_card.setdefault("limitations", [])
            for limitation in additional_limitations:
                if limitation not in model_card["limitations"]:
                    model_card["limitations"].append(limitation)
            atomic_json(model_card_path, model_card)
            atomic_json(TABLE_DIR / "primary_K8_model_card.json", model_card)
            gse_manifest_path = TABLE_DIR / "GSE294268_primary_K8_projection_manifest.json"
            if gse_manifest_path.is_file():
                gse_manifest = json.loads(gse_manifest_path.read_text())
                gse_manifest["model_card_sha256"] = sha256sum(model_card_path)
                atomic_json(gse_manifest_path, gse_manifest)

            manifest_path = TABLE_DIR / "PANGEA_primary_K8_projection_manifest.json"
            atomic_json(manifest_path, {
                "schema_version": 1, "analysis": ANALYSIS_NAME, "dataset": "PANGEA",
                "model": "full-Gray primary K=8 fixed basis",
                "n_cells": len(usage), "n_Series": metadata["Series"].nunique(),
                "n_sample_proxies": metadata["sample_proxy"].nunique(),
                "excluded_Series": list(PANGEA_EXCLUDED_SERIES),
                "PANGEA_path": str(PANGEA_PATH), "PANGEA_sha256": source_record["sha256"],
                "usage_path": str(usage_path), "usage_sha256": sha256sum(usage_path),
                "quality_path": str(quality_path), "quality_sha256": sha256sum(quality_path),
                "quality_gated_usage_path": str(gated_usage_path),
                "quality_gated_usage_sha256": sha256sum(gated_usage_path),
                "baseline_path": str(baseline_path), "baseline_sha256": sha256sum(baseline_path),
                "metadata_path": str(metadata_path), "metadata_sha256": sha256sum(metadata_path),
                "model_card_path": str(model_card_path),
                "model_card_sha256": sha256sum(model_card_path),
                "Gray_q90_relative_error_gate": gray_q90,
                "Gray_q95_relative_error_gate": gray_q95,
                "fraction_accepted_q95": float(quality["within_Gray_q95"].mean()),
                "fraction_strict_q90": float(quality["within_Gray_q90"].mean()),
                "external_data_used_for_K_selection": False,
                "untouched_final_cohort": False,
                "evidence_boundary": (
                    "transportability and silver-label association; not protein/function validation"),
            })
            note("PANGEA primary K8 projection", "completed_with_quality_gate",
                 f"{len(usage)} cells across {metadata['Series'].nunique()} Series; "
                 "Series-level inference and silver-label boundary enforced")
            display(quality_summary)
            display(subtype_tests.loc[
                subtype_tests.quality_gate.eq("Gray_q95_accepted")
            ].sort_values(["score_type", "q_value_BH_within_score_gate"]))
            pangea.file.close()

        if EXECUTION_STAGE == "pangea_score":
            run_pangea_projection()
        else:
            print("PANGEA raw projection not launched in stage", EXECUTION_STAGE)
        """),
        md(r"""
        ## 6 · Gray independent-donor orthogonal functional validation

        Gray Data S12 contains adult flow-cytometry readouts after four hours of
        PMA-ionomycin stimulation: Areg, GzmB and IFNg for Vδ1 and Vδ2 cells. The
        functional donors and adult scRNA-seq donors are disjoint. Consequently,
        this stage does **not** correlate RNA and protein within donors. It compares
        donor-aggregated measurements across the ten shared `tissue × Vδ` contexts.

        The RNA side uses the five donor-held-out K=8 scores, not in-sample usages.
        Fold-specific components are mapped one-to-one to the frozen P1/P2/P3/P4/
        P5/P7/P9/P10 reference. The simple baseline is the top-50 rank signature
        learned from the training donors of that same fold. P2–GzmB is the registered
        primary functional hypothesis because P2 contains `GZMB`, `PRF1`, `GNLY`
        and related cytolytic genes. All other program–endpoint combinations are
        secondary or exploratory and receive multiplicity-adjusted results.
        """),
        code(r"""
        # 12. Validate donor-held-out Gray programs against Data S12 flow function
        FUNCTIONAL_PROGRAMS = ["P1", "P2", "P3", "P4", "P5", "P7", "P9", "P10"]
        FUNCTIONAL_ENDPOINTS = ["Areg", "GzmB", "IFNg"]

        def permutation_spearman(x, y, token, draws=19999):
            x = np.asarray(x, float); y = np.asarray(y, float)
            valid = np.isfinite(x) & np.isfinite(y)
            x = x[valid]; y = y[valid]
            if len(x) < 5 or np.unique(x).size < 2 or np.unique(y).size < 2:
                return np.nan, np.nan
            x_rank = rankdata(x, method="average")
            y_rank = rankdata(y, method="average")
            x_rank = x_rank - x_rank.mean(); y_rank = y_rank - y_rank.mean()
            denominator = np.linalg.norm(x_rank) * np.linalg.norm(y_rank)
            observed = float(np.dot(x_rank, y_rank) / denominator)
            seed = SEED + int(hashlib.sha256(str(token).encode()).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            permutations = np.argsort(rng.random((draws, len(y_rank))), axis=1)
            null = (y_rank[permutations] @ x_rank) / denominator
            p_value = float((1 + np.sum(np.abs(null) >= abs(observed) - 1e-12)) / (draws + 1))
            return observed, p_value

        def parse_gray_functional_panel():
            for path in (GRAY_DATA_S1, GRAY_DATA_S12):
                if not path.is_file():
                    raise FileNotFoundError(path)
                record_source(path, full_hash=True)
            donor_inventory = pd.read_excel(GRAY_DATA_S1, sheet_name="Sheet1", header=1)
            donor_inventory = donor_inventory.loc[donor_inventory["Donor"].notna()].copy()
            donor_inventory["Donor"] = donor_inventory["Donor"].astype(str)
            raw = pd.read_excel(GRAY_DATA_S12, sheet_name="Sheet1", header=None)
            anchors = np.flatnonzero(raw.iloc[:, 0].astype(str).eq("Functional panel"))
            if len(anchors) != 1:
                raise ValueError("Data S12 must contain exactly one Functional panel")
            start = int(anchors[0]) + 2
            panel = raw.iloc[start:, :11].copy()
            panel = panel.loc[panel.iloc[:, 0].notna()].copy()
            panel.columns = [
                "sample", "organ", "TCRgd", "Vd1", "Vd1_Areg", "Vd1_GzmB", "Vd1_IFNg",
                "Vd2", "Vd2_Areg", "Vd2_GzmB", "Vd2_IFNg",
            ]
            panel["donor_id"] = panel["sample"].astype(str).str.extract(r"^(D\d+)")[0]
            organ_map = {
                "Blood": "blood", "BLD": "blood", "Spleen": "spleen", "SPL": "spleen",
                "Lung": "lung", "LNG": "lung", "LLN": "thoracic lymph node",
                "Jej": "jejunum", "JEJ": "jejunum", "MLN": "mesenteric lymph node",
            }
            tissue_code = {
                "blood": "B", "spleen": "S", "lung": "L", "thoracic lymph node": "LL",
                "jejunum": "J", "mesenteric lymph node": "M",
            }
            panel["tissue"] = panel["organ"].map(organ_map)
            if panel["donor_id"].isna().any() or panel["tissue"].isna().any():
                raise ValueError("Data S12 functional sample identifiers or organs could not be parsed")
            inventory = donor_inventory.set_index("Donor")
            audit_rows, long_rows = [], []
            for row in panel.itertuples(index=False):
                if row.donor_id not in inventory.index:
                    raise ValueError(f"Data S12 donor absent from Data S1: {row.donor_id}")
                declared = {
                    token.strip() for token in str(inventory.loc[row.donor_id, "Function"]).split(",")
                    if token.strip() and token.strip().lower() != "nan"
                }
                code = tissue_code[row.tissue]
                audit_rows.append({
                    "sample": row.sample, "donor_id": row.donor_id, "tissue": row.tissue,
                    "Data_S1_function_tissues": ",".join(sorted(declared)),
                    "tissue_declared_in_Data_S1": code in declared,
                })
                for vd, prefix in (("vd1", "Vd1"), ("vd2", "Vd2")):
                    subset_frequency = float(getattr(row, prefix))
                    long_rows.append({
                        "sample": row.sample, "donor_id": row.donor_id,
                        "donor_age": float(inventory.loc[row.donor_id, "Age"]),
                        "tissue": row.tissue, "vd": vd,
                        "subset_frequency_percent": subset_frequency,
                        "Areg_percent": float(getattr(row, f"{prefix}_Areg")),
                        "GzmB_percent": float(getattr(row, f"{prefix}_GzmB")),
                        "IFNg_percent": float(getattr(row, f"{prefix}_IFNg")),
                        "included": subset_frequency > 0,
                        "exclusion_reason": "" if subset_frequency > 0 else "Vd subset frequency is zero",
                    })
            audit = pd.DataFrame(audit_rows)
            if not audit["tissue_declared_in_Data_S1"].all():
                raise ValueError("Data S12 functional tissue is not declared in Data S1")
            functional = pd.DataFrame(long_rows)
            return donor_inventory, panel, audit, functional

        def load_gray_oof_k8_scores():
            matching_path = TABLE_DIR / "Gray_full_reference_matching_all_pairs.csv"
            if not matching_path.is_file():
                raise FileNotFoundError("Run assemble_full before functional validation")
            matching = pd.read_csv(matching_path)
            score_parts = {"cNMF_usage": [], "rank_signature_score": []}
            for fold in range(N_FOLDS):
                job = CHECKPOINT_DIR / "full" / f"fold{fold}_K8"
                spec_path = job / "job_spec.json"
                if not spec_path.is_file():
                    raise FileNotFoundError(spec_path)
                signature = stable_token(json.loads(spec_path.read_text()))
                if not validate_marker(job / "heldout_scoring.complete.json", signature):
                    raise RuntimeError(f"Invalid donor-held-out checkpoint: {job}")
                assigned = matching.loc[
                    matching["fold"].eq(fold) & matching["K"].eq(PRIMARY_K) &
                    matching["assigned_one_to_one"].astype(bool)
                ].copy()
                if len(assigned) != PRIMARY_K:
                    raise ValueError(f"Fold {fold} lacks a complete one-to-one K8 mapping")
                rename = {
                    str(int(float(row.fold_program))): f"P{int(float(row.reference_program))}"
                    for row in assigned.itertuples(index=False)
                }
                if set(rename.values()) != set(FUNCTIONAL_PROGRAMS):
                    raise ValueError(f"Unexpected fold {fold} reference mapping: {rename}")
                paths = {
                    "cNMF_usage": job / "heldout_usage_normalized.parquet",
                    "rank_signature_score": job / "heldout_training_signature_scores.parquet",
                }
                expected_ids = set(primary.index[primary["outer_fold"].eq(fold)].astype(str))
                for score_type, path in paths.items():
                    frame = pd.read_parquet(path)
                    frame.index = frame.index.astype(str)
                    frame.columns = [str(int(float(column))) for column in frame.columns]
                    if set(frame.index) != expected_ids or not frame.index.is_unique:
                        raise ValueError(f"Held-out cell definition changed: {path}")
                    frame = frame.rename(columns=rename).loc[:, FUNCTIONAL_PROGRAMS]
                    frame["outer_fold"] = fold
                    score_parts[score_type].append(frame)
            scores = {key: pd.concat(value).sort_index() for key, value in score_parts.items()}
            if not scores["cNMF_usage"].index.equals(scores["rank_signature_score"].index):
                raise ValueError("cNMF and rank-signature held-out cells differ")
            gray_backed = ad.read_h5ad(GRAY_PATH, backed="r")
            try:
                cell_ids = scores["cNMF_usage"].index
                if not set(cell_ids) <= set(gray_backed.obs_names.astype(str)):
                    raise ValueError("Held-out cell identifiers are absent from Gray h5ad")
                metadata = gray_backed.obs.loc[cell_ids, ["donor_id", "tissue", "vd"]].copy()
            finally:
                gray_backed.file.close()
            metadata.index = metadata.index.astype(str)
            metadata["donor_id"] = metadata["donor_id"].astype(str)
            metadata["tissue"] = metadata["tissue"].astype(str)
            metadata["vd"] = metadata["vd"].astype(str)
            return scores, metadata

        def donor_context_scores(frame, metadata):
            data = frame[FUNCTIONAL_PROGRAMS].join(metadata)
            data = data.loc[data["vd"].isin(["vd1", "vd2"])].copy()
            return (data.groupby(["donor_id", "tissue", "vd"], observed=True)
                    [FUNCTIONAL_PROGRAMS].median().reset_index())

        def resample_donor_context(table, value_columns, rng):
            donor_ids = np.asarray(sorted(table["donor_id"].unique()))
            sampled = rng.choice(donor_ids, size=len(donor_ids), replace=True)
            pieces = []
            for bootstrap_id, donor in enumerate(sampled):
                piece = table.loc[table["donor_id"].eq(donor),
                                  ["tissue", "vd", *value_columns]].copy()
                piece["bootstrap_donor"] = bootstrap_id
                pieces.append(piece)
            sampled_table = pd.concat(pieces, ignore_index=True)
            return (sampled_table.groupby(["tissue", "vd"], observed=True)
                    [value_columns].median().reset_index())

        def bootstrap_primary_functional_delta(cnmf_donor, signature_donor, functional_donor,
                                               draws=4000):
            paired = cnmf_donor[["donor_id", "tissue", "vd", "P2"]].merge(
                signature_donor[["donor_id", "tissue", "vd", "P2"]],
                on=["donor_id", "tissue", "vd"], suffixes=("_cNMF", "_signature"),
                validate="one_to_one")
            function = functional_donor[["donor_id", "tissue", "vd", "GzmB_percent"]].copy()
            rng = np.random.default_rng(SEED + 12012)
            estimates = []
            for _ in range(draws):
                score_context = resample_donor_context(
                    paired, ["P2_cNMF", "P2_signature"], rng)
                function_context = resample_donor_context(function, ["GzmB_percent"], rng)
                merged = score_context.merge(function_context, on=["tissue", "vd"], how="inner")
                if len(merged) < 5:
                    continue
                cnmf_rho = rho(merged["P2_cNMF"], merged["GzmB_percent"])
                signature_rho = rho(merged["P2_signature"], merged["GzmB_percent"])
                estimates.append((cnmf_rho, signature_rho, cnmf_rho - signature_rho))
            estimates = np.asarray(estimates, float)
            if len(estimates) < 0.95 * draws:
                raise RuntimeError("Too few valid donor bootstrap replicates")
            return estimates

        def run_gray_functional_validation():
            donor_inventory, panel, sample_audit, functional_long = parse_gray_functional_panel()
            scores, metadata = load_gray_oof_k8_scores()
            functional = functional_long.loc[functional_long["included"]].copy()
            functional_donors = set(functional["donor_id"])
            scrna_donors = set(metadata["donor_id"])
            overlap = sorted(functional_donors & scrna_donors)
            if overlap:
                raise RuntimeError(
                    "Gray donor overlap is nonzero; replace context-level design with paired validation")

            save_table(sample_audit, "Gray_Data_S12_functional_sample_audit")
            save_table(functional_long, "Gray_Data_S12_functional_panel_long")
            donor_sets = pd.concat([
                pd.DataFrame({"donor_id": sorted(scrna_donors), "cohort": "Gray adult scRNA"}),
                pd.DataFrame({"donor_id": sorted(functional_donors), "cohort": "Gray Data S12 function"}),
            ], ignore_index=True)
            donor_sets["present_in_other_cohort"] = donor_sets["donor_id"].isin(overlap)
            save_table(donor_sets, "Gray_scRNA_Data_S12_donor_overlap")

            functional_donor = (functional.groupby(["donor_id", "tissue", "vd"], observed=True)
                                [[f"{endpoint}_percent" for endpoint in FUNCTIONAL_ENDPOINTS]]
                                .median().reset_index())
            save_table(functional_donor, "Gray_Data_S12_functional_donor_tissue_vd_endpoints")
            score_donors = {
                score_type: donor_context_scores(frame, metadata)
                for score_type, frame in scores.items()
            }
            for score_type, frame in score_donors.items():
                save_table(frame, f"Gray_primary_K8_{score_type}_donor_tissue_vd_scores")

            functional_context = (functional_donor.groupby(["tissue", "vd"], observed=True)
                                  .agg(**{
                                      **{f"{endpoint}_percent": (f"{endpoint}_percent", "median")
                                         for endpoint in FUNCTIONAL_ENDPOINTS},
                                      "n_function_donors": ("donor_id", "nunique"),
                                  }).reset_index())
            context_parts, test_rows = [], []
            for score_type, donor_table in score_donors.items():
                score_context = (donor_table.groupby(["tissue", "vd"], observed=True)
                                 .agg(**{
                                     **{program: (program, "median") for program in FUNCTIONAL_PROGRAMS},
                                     "n_scRNA_donors": ("donor_id", "nunique"),
                                 }).reset_index())
                context = score_context.merge(
                    functional_context, on=["tissue", "vd"], how="inner", validate="one_to_one")
                context["score_type"] = score_type
                context_parts.append(context)
                for program in FUNCTIONAL_PROGRAMS:
                    for endpoint in FUNCTIONAL_ENDPOINTS:
                        observed, permutation_p = permutation_spearman(
                            context[program], context[f"{endpoint}_percent"],
                            [score_type, program, endpoint])
                        analytic = spearmanr(
                            context[program].to_numpy(float),
                            context[f"{endpoint}_percent"].to_numpy(float))
                        hypothesis = (
                            "primary" if (program, endpoint) == ("P2", "GzmB") else
                            "secondary" if (program, endpoint) == ("P2", "IFNg") else
                            "exploratory"
                        )
                        test_rows.append({
                            "score_type": score_type, "program": program, "endpoint": endpoint,
                            "hypothesis": hypothesis, "n_matched_tissue_vd_contexts": len(context),
                            "n_scRNA_donors": donor_table["donor_id"].nunique(),
                            "n_function_donors": functional_donor["donor_id"].nunique(),
                            "n_overlapping_donors": 0,
                            "spearman_rho_context": observed,
                            "analytic_two_sided_p_value": float(analytic.pvalue),
                            "permutation_two_sided_p_value": permutation_p,
                            "permutation_draws": 19999,
                            "inference_unit": "tissue_x_Vd_context",
                            "donor_handling": "median within donor/context, then median across donors",
                            "evidence_boundary": (
                                "independent-donor orthogonal concordance; not donor-paired validation"),
                        })
            context_table = pd.concat(context_parts, ignore_index=True)
            tests = pd.DataFrame(test_rows)
            tests["q_value_BH_all_program_endpoint_within_score_type"] = np.nan
            for _, indices in tests.groupby("score_type", observed=True).groups.items():
                tests.loc[indices, "q_value_BH_all_program_endpoint_within_score_type"] = bh_adjust(
                    tests.loc[indices, "permutation_two_sided_p_value"].to_numpy(float))
            save_table(context_table, "Gray_primary_K8_functional_context_medians")
            save_table(tests, "Gray_primary_K8_functional_context_concordance_tests")

            primary = tests.loc[
                tests["program"].eq("P2") & tests["endpoint"].eq("GzmB")
            ].copy()
            comparison = primary.pivot(
                index=["program", "endpoint"], columns="score_type",
                values="spearman_rho_context").reset_index()
            comparison["cNMF_minus_rank_signature_rho"] = (
                comparison["cNMF_usage"] - comparison["rank_signature_score"])
            bootstrap = bootstrap_primary_functional_delta(
                score_donors["cNMF_usage"], score_donors["rank_signature_score"],
                functional_donor)
            for column, position in (("cNMF_usage", 0), ("rank_signature_score", 1),
                                     ("cNMF_minus_rank_signature_rho", 2)):
                comparison[f"{column}_bootstrap_ci95_low"] = np.quantile(bootstrap[:, position], 0.025)
                comparison[f"{column}_bootstrap_ci95_high"] = np.quantile(bootstrap[:, position], 0.975)
            comparison["bootstrap_draws"] = len(bootstrap)
            comparison["bootstrap_probability_cNMF_gt_signature"] = float(
                np.mean(bootstrap[:, 2] > 0))
            comparison["incremental_value_established"] = (
                comparison["cNMF_minus_rank_signature_rho_bootstrap_ci95_low"] > 0)
            comparison["evidence_boundary"] = (
                "donor bootstrap in two disjoint cohorts; context-level comparison")
            save_table(comparison, "Gray_primary_K8_cNMF_vs_rank_signature_functional_value")

            plot_context = context_table.loc[
                context_table["score_type"].isin(["cNMF_usage", "rank_signature_score"])
            ].copy()
            fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), sharey=True)
            label_offsets = {
                ("mesenteric lymph node", "vd1"): (3, 7, "left"),
                ("thoracic lymph node", "vd1"): (3, -10, "left"),
            }
            for ax, score_type, title in zip(
                    axes, ["cNMF_usage", "rank_signature_score"],
                    ["Donor-held-out cNMF P2", "Train-only P2 rank signature"]):
                part = plot_context.loc[plot_context["score_type"].eq(score_type)]
                ax.scatter(part["P2"], part["GzmB_percent"], s=34, color="#0072B2")
                for row in part.itertuples(index=False):
                    label = f"{row.tissue}, {str(row.vd).replace('vd', 'Vδ')}"
                    dx, dy, horizontal_alignment = label_offsets.get(
                        (str(row.tissue), str(row.vd)), (3, 3, "left"))
                    ax.annotate(
                        label, (row.P2, row.GzmB_percent), xytext=(dx, dy),
                        textcoords="offset points", fontsize=6.5, color="black",
                        ha=horizontal_alignment,
                    )
                result = primary.loc[primary["score_type"].eq(score_type)].iloc[0]
                ax.set(xlabel="Context median score",
                       title=f"{title}\nρ={result.spearman_rho_context:.2f}, "
                             f"permutation P={result.permutation_two_sided_p_value:.3g}")
            axes[0].set_ylabel("GzmB-positive cells (%)")
            fig.suptitle("Gray P2 cytolytic program and independent flow-cytometry function")
            save_figure(fig, "Gray_primary_K8_P2_GzmB_orthogonal_functional_validation")

            model_card_path = PRIMARY_FREEZE_DIR / "primary_K8_model_card.json"
            if not model_card_path.is_file():
                raise FileNotFoundError(model_card_path)
            model_card = json.loads(model_card_path.read_text())
            cnmf_primary = primary.loc[primary["score_type"].eq("cNMF_usage")].iloc[0]
            signature_primary = primary.loc[
                primary["score_type"].eq("rank_signature_score")].iloc[0]
            incremental = bool(comparison["incremental_value_established"].iloc[0])
            model_card["updated_at"] = datetime.now(timezone.utc).isoformat()
            model_card.setdefault("validation_readout", {})[
                "orthogonal_protein_or_functional_validation"] = (
                    "Gray Data S12 independent-donor flow validation completed across 10 matched "
                    f"tissue x Vd contexts; P2-GzmB rho={cnmf_primary.spearman_rho_context:.3f}, "
                    f"permutation P={cnmf_primary.permutation_two_sided_p_value:.5g}; "
                    "zero donor overlap, so this is not donor-paired validation")
            model_card["validation_readout"]["functional_rank_signature_baseline"] = (
                f"P2-GzmB rank-signature rho={signature_primary.spearman_rho_context:.3f}; "
                f"incremental cNMF value established={incremental}")
            readiness = model_card.setdefault("readiness", {})
            readiness["orthogonal_functional_validation_completed"] = True
            readiness["orthogonal_functional_validation_is_donor_paired"] = False
            readiness["incremental_value_over_simple_signature_established"] = incremental
            readiness["NC_claim_package_complete"] = False
            model_card["status"] = (
                "research-grade frozen scorer; independent-donor functional concordance completed; "
                "NC claim package incomplete")
            limitations = [
                "Gray adult scRNA and Data S12 functional donors do not overlap; validation aligns 10 tissue x Vd contexts and cannot estimate paired donor-level correspondence.",
                "P2-GzmB concordance is also recovered by the train-only rank-signature baseline; incremental cNMF value is not established unless the bootstrap delta excludes zero.",
                "Data S12 provides Areg, GzmB and IFNg after PMA-ionomycin but no killing, CD107a, TNF or protein readout for every cNMF program.",
            ]
            model_card.setdefault("limitations", [])
            for limitation in limitations:
                if limitation not in model_card["limitations"]:
                    model_card["limitations"].append(limitation)
            atomic_json(model_card_path, model_card)
            atomic_json(TABLE_DIR / "primary_K8_model_card.json", model_card)

            manifest_path = TABLE_DIR / "Gray_primary_K8_functional_validation_manifest.json"
            output_paths = [
                TABLE_DIR / "Gray_Data_S12_functional_sample_audit.csv",
                TABLE_DIR / "Gray_Data_S12_functional_panel_long.csv",
                TABLE_DIR / "Gray_scRNA_Data_S12_donor_overlap.csv",
                TABLE_DIR / "Gray_Data_S12_functional_donor_tissue_vd_endpoints.csv",
                TABLE_DIR / "Gray_primary_K8_cNMF_usage_donor_tissue_vd_scores.csv",
                TABLE_DIR / "Gray_primary_K8_rank_signature_score_donor_tissue_vd_scores.csv",
                TABLE_DIR / "Gray_primary_K8_functional_context_medians.csv",
                TABLE_DIR / "Gray_primary_K8_functional_context_concordance_tests.csv",
                TABLE_DIR / "Gray_primary_K8_cNMF_vs_rank_signature_functional_value.csv",
            ]
            atomic_json(manifest_path, {
                "schema_version": 1, "analysis": ANALYSIS_NAME,
                "stage": "gray_functional_validation",
                "model": "donor-held-out Gray K8 programs mapped to frozen K10 reference",
                "Data_S1_path": str(GRAY_DATA_S1), "Data_S1_sha256": sha256sum(GRAY_DATA_S1),
                "Data_S12_path": str(GRAY_DATA_S12), "Data_S12_sha256": sha256sum(GRAY_DATA_S12),
                "n_scRNA_donors": len(scrna_donors),
                "n_function_donors": len(functional_donors),
                "n_overlapping_donors": len(overlap),
                "n_matched_tissue_vd_contexts": int(
                    context_table.groupby(["tissue", "vd"], observed=True).ngroups),
                "primary_hypothesis": "P2 positive concordance with GzmB",
                "primary_cNMF_spearman_rho": cnmf_primary.spearman_rho_context,
                "primary_cNMF_permutation_p_value": cnmf_primary.permutation_two_sided_p_value,
                "incremental_value_over_rank_signature_established": incremental,
                "external_data_used_for_K_selection": False,
                "donor_paired_validation": False,
                "evidence_boundary": (
                    "independent-donor orthogonal context concordance; not paired validation"),
                "outputs": {str(path): sha256sum(path) for path in output_paths},
                "model_card_path": str(model_card_path),
                "model_card_sha256": sha256sum(model_card_path),
            })
            note("Gray Data S12 functional validation", "completed_independent_donor_context_level",
                 f"P2-GzmB rho={cnmf_primary.spearman_rho_context:.3f}; "
                 f"permutation P={cnmf_primary.permutation_two_sided_p_value:.5g}; "
                 f"incremental value={incremental}")
            display(primary)
            display(comparison)

        if EXECUTION_STAGE == "gray_functional_validation":
            run_gray_functional_validation()
        else:
            print("Gray functional validation not launched in stage", EXECUTION_STAGE)
        """),
        md(r"""
        ## 7 · Locked GSE268991 untouched paired-multimodal validation

        The immutable preanalysis manifest was written and synchronized before any
        GSE268991 expression or ADT matrix was opened. The primary population is
        evaluated with the frozen full-Gray K=8 scorer and the pre-existing Gray
        q95 reconstruction gate. Each of the 14 `Normal/Tumor × donor` matrices is
        an atomic restart unit. No result from this cohort can change K, HVGs,
        training SD, spectra, program mapping, quality thresholds or hypotheses.

        The three locked biological hypotheses are: P2 is higher in independently
        sequenced TRGC1 than TRGC2 cells; P3 is higher in TRGC2 than TRGC1 cells;
        and P2 is negatively associated with the same-cell CD45RA/CCR7 ADT naive
        composite. The primary incremental-value endpoint is donor-held-out TRGC1
        versus TRGC2 prediction. Receptor genes are absent from the frozen HVGs, so
        the endpoint is not directly copied from TRGC transcript counts.
        """),
        code(r"""
        # 13. Score and validate the locked GSE268991 paired GEX/ADT/gamma-delta-TCR cohort
        GSE268991_SAMPLES = (
            "N77", "N78", "N79", "N81", "N85", "N86", "N91",
            "T77", "T78", "T79", "T81", "T85", "T86", "T91",
        )
        GSE268991_PREANALYSIS = TABLE_DIR / "GSE268991_preanalysis_evaluation_manifest.json"
        GSE268991_PREANALYSIS_SHA256 = (
            "ed5b6a984a164622e2e75d1ee1f7edd7c273f2504c9b925200047dde00cf9136")
        GSE268991_RAW_DIR = GSE268991_DIR / "GSE268991_RAW"
        GSE268991_PROJECTION_DIR = MODEL_DIR / "GSE268991_primary_K8_projection"
        GSE268991_PROJECTION_CHECKPOINTS = (
            CHECKPOINT_DIR / "GSE268991_primary_K8_projection")

        def load_frozen_scorer_for_gse268991():
            import yaml

            primary_spec_path = PRIMARY_JOB / "job_spec.json"
            primary_marker = PRIMARY_FREEZE_DIR / "frozen.complete.json"
            if not primary_spec_path.is_file() or not validate_marker(
                    primary_marker, stable_token(json.loads(primary_spec_path.read_text()))):
                raise RuntimeError("The frozen primary K8 basis is absent or failed checksum validation")
            spec_path = PRIMARY_FREEZE_DIR / "primary_K8_scoring_spec.json"
            spectra_path = PRIMARY_FREEZE_DIR / "primary_K8_HVG_TPM_spectra.tsv"
            sd_path = PRIMARY_FREEZE_DIR / "primary_K8_training_sd.csv"
            ranked_path = PRIMARY_FREEZE_DIR / "primary_K8_ranked_genes.csv"
            params_path = one_file(PRIMARY_JOB, "*.nmf_idvrun_params.yaml")
            scoring_spec = json.loads(spec_path.read_text())
            spectra = pd.read_csv(spectra_path, sep="\t", index_col=0)
            spectra.index = spectra.index.map(lambda value: str(int(float(value))))
            sd = pd.read_csv(sd_path, index_col=0).iloc[:, 0]
            universe = pd.Index(scoring_spec["universe"])
            hvgs = pd.Index(scoring_spec["hvgs"])
            hvg_positions = universe.get_indexer(hvgs)
            if (hvg_positions < 0).any() or list(spectra.columns) != list(hvgs):
                raise ValueError("Frozen K8 scorer gene definitions are inconsistent")
            if list(sd.index) != list(hvgs) or not np.isfinite(sd).all() or not (sd > 0).all():
                raise ValueError("Frozen K8 training SD is inconsistent")
            receptor_hvgs = [gene for gene in hvgs if is_receptor_gene(gene)]
            if receptor_hvgs:
                raise RuntimeError(
                    "TRGC endpoint leakage: receptor genes occur in frozen HVGs: "
                    + ", ".join(receptor_hvgs))
            ranked = pd.read_csv(ranked_path)
            ranked["K8_component"] = ranked["K8_component"].map(
                lambda value: str(int(float(value))))
            signatures = {
                component: part.sort_values("rank").head(50)["gene"].astype(str).tolist()
                for component, part in ranked.groupby("K8_component", observed=True)
            }
            mapping = pd.read_csv(PRIMARY_FREEZE_DIR / "primary_K8_to_K10_reference_mapping.csv")
            assigned = mapping.loc[mapping["assigned_one_to_one"]].copy()
            component_to_program = {
                str(int(float(row.K8_component))): f"P{int(float(row.K10_reference_program))}"
                for row in assigned.itertuples(index=False)
            }
            expected_programs = {"P1", "P2", "P3", "P4", "P5", "P7", "P9", "P10"}
            if set(component_to_program.values()) != expected_programs:
                raise ValueError(f"Unexpected frozen K8 program mapping: {component_to_program}")
            return {
                "spec_path": spec_path, "spectra_path": spectra_path, "sd_path": sd_path,
                "ranked_path": ranked_path, "params_path": params_path,
                "universe": universe, "hvgs": hvgs, "hvg_positions": hvg_positions,
                "sd": sd, "basis": spectra.div(sd, axis=1).to_numpy(np.float64),
                "components": spectra.index.astype(str), "signatures": signatures,
                "component_to_program": component_to_program,
                "solver_params": yaml.safe_load(params_path.read_text()),
            }

        def gse268991_sample_paths(sample):
            paths = {
                "matrix": GSE268991_DIR / f"GSE268991_{sample}_matrix.mtx.gz",
                "features": GSE268991_DIR / f"GSE268991_{sample}_features.tsv.gz",
                "barcodes": GSE268991_DIR / f"GSE268991_{sample}_barcodes.tsv.gz",
            }
            candidates = sorted(GSE268991_RAW_DIR.glob(f"*_{sample}_gd.csv.gz"))
            if len(candidates) != 1:
                raise FileNotFoundError(
                    f"Expected one gamma-delta VDJ file for {sample}; found {candidates}")
            paths["gd_tcr"] = candidates[0]
            for path in paths.values():
                if not path.is_file():
                    raise FileNotFoundError(path)
            return paths

        def boolean_series(values):
            return values.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})

        def score_gse268991_sample(sample, scorer, preanalysis_hash):
            from threadpoolctl import threadpool_limits

            paths = gse268991_sample_paths(sample)
            donor = f"CRC {int(sample[1:])}"
            condition = "Normal" if sample.startswith("N") else "Tumor"
            features = pd.read_csv(paths["features"], sep="\t", header=None, compression="infer")
            if features.shape[1] < 3:
                raise ValueError(f"GSE268991 feature schema is incomplete: {sample}")
            features.columns = ["feature_id", "feature_name", "feature_type", *[
                f"extra_{index}" for index in range(features.shape[1] - 3)]]
            barcodes = pd.Index(pd.read_csv(
                paths["barcodes"], sep="\t", header=None, compression="infer").iloc[:, 0].astype(str))
            if not barcodes.is_unique:
                raise ValueError(f"GSE268991 barcodes are not unique within {sample}")
            gene_mask = features["feature_type"].astype(str).eq("Gene Expression").to_numpy()
            adt_mask = features["feature_type"].astype(str).eq("Antibody Capture").to_numpy()
            genes = pd.Index(features.loc[gene_mask, "feature_name"].astype(str))
            adt_names = pd.Index(features.loc[adt_mask, "feature_name"].astype(str))
            if not adt_names.is_unique:
                raise ValueError(f"GSE268991 ADT names are duplicated in {sample}")
            hvg_coverage = float(scorer["hvgs"].isin(genes).mean())
            if hvg_coverage < MIN_HVG_COVERAGE:
                raise RuntimeError(
                    f"GSE268991 {sample} frozen-HVG coverage {hvg_coverage:.3f} is below "
                    f"the locked {MIN_HVG_COVERAGE:.2f} minimum")
            cell_ids = pd.Index([f"{sample}:{barcode}" for barcode in barcodes], name="cell_id")
            job = GSE268991_PROJECTION_CHECKPOINTS / sample
            job.mkdir(parents=True, exist_ok=True)
            spec = {
                "schema_version": 1, "analysis": ANALYSIS_NAME, "dataset": "GSE268991",
                "sample": sample, "donor": donor, "condition": condition,
                "preanalysis_manifest_sha256": preanalysis_hash,
                "n_cells": len(cell_ids), "cell_ids_sha256": sha256_lines(cell_ids),
                "n_gene_features": int(gene_mask.sum()), "n_ADT_features": int(adt_mask.sum()),
                "frozen_HVG_coverage": hvg_coverage,
                "inputs": {key: sha256sum(path) for key, path in paths.items()},
                "scoring_spec_sha256": sha256sum(scorer["spec_path"]),
                "spectra_sha256": sha256sum(scorer["spectra_path"]),
                "sd_sha256": sha256sum(scorer["sd_path"]),
                "ranked_genes_sha256": sha256sum(scorer["ranked_path"]),
                "solver_params_sha256": sha256sum(scorer["params_path"]),
                "seed": 20260919,
            }
            signature = stable_token(spec)
            spec_path = job / "job_spec.json"
            if spec_path.exists() and json.loads(spec_path.read_text()) != spec:
                raise ValueError(f"GSE268991 sample definition changed: {job}")
            atomic_json(spec_path, spec)
            marker = job / "scoring.complete.json"
            if validate_marker(marker, signature):
                note(f"GSE268991 {sample}", "reused", "validated paired-multimodal checkpoint")
                return

            with gzip.open(paths["matrix"], "rb") as handle:
                raw_matrix = scipy.io.mmread(handle).tocsr()
            if raw_matrix.shape != (len(features), len(barcodes)):
                raise ValueError(f"GSE268991 matrix/features/barcodes disagree: {sample}")
            gene_matrix = raw_matrix[gene_mask, :]
            universe_positions = scorer["universe"].get_indexer(genes)
            valid_features = universe_positions >= 0
            collapse = sparse.csr_matrix((
                np.ones(valid_features.sum()),
                (np.flatnonzero(valid_features), universe_positions[valid_features])),
                shape=(len(genes), len(scorer["universe"])))
            counts = (gene_matrix.T @ collapse).tocsr().astype(float)
            library_size = np.asarray(counts.sum(axis=1)).ravel()
            if (library_size <= 0).any():
                raise ValueError(f"GSE268991 cells with zero frozen-universe counts: {sample}")
            tpm = sparse.diags(1e6 / library_size) @ counts
            objective = tpm[:, scorer["hvg_positions"]].multiply(
                1 / scorer["sd"].to_numpy()).tocsr().astype(np.float64)
            params = dict(scorer["solver_params"])
            params.update(H=scorer["basis"], update_H=False, n_components=PRIMARY_K)
            usage_blocks, relative_error = [], []
            gram = scorer["basis"] @ scorer["basis"].T
            for start in range(0, objective.shape[0], 5000):
                block = objective[start:start + 5000]
                np.random.seed(20260919)
                with threadpool_limits(limits=THREADS):
                    coefficients, _, _ = non_negative_factorization(block, **params)
                usage_blocks.append(coefficients)
                x_norm_sq = np.asarray(block.multiply(block).sum(axis=1)).ravel()
                cross = np.sum(coefficients * (block @ scorer["basis"].T), axis=1)
                fitted_norm_sq = np.sum((coefficients @ gram) * coefficients, axis=1)
                residual_sq = np.maximum(x_norm_sq + fitted_norm_sq - 2 * cross, 0)
                relative_error.extend(
                    np.sqrt(residual_sq) / np.maximum(np.sqrt(x_norm_sq), 1e-15))
            usage = normalize_usage(pd.DataFrame(
                np.vstack(usage_blocks), index=cell_ids, columns=scorer["components"]))
            signature_scores = rank_scores(counts, scorer["universe"], scorer["signatures"])
            signature_scores.index = cell_ids

            adt_counts = raw_matrix[adt_mask, :].T.toarray().astype(float)
            adt_scale = np.exp(np.mean(np.log1p(adt_counts), axis=0))
            adt_clr = np.log1p(adt_counts / np.maximum(adt_scale, 1e-12))
            adt = pd.concat([
                pd.DataFrame(adt_counts, index=cell_ids,
                             columns=[f"ADT_raw__{name}" for name in adt_names]),
                pd.DataFrame(adt_clr, index=cell_ids,
                             columns=[f"ADT_CLR__{name}" for name in adt_names]),
            ], axis=1)
            adt["ADT_total_counts"] = adt_counts.sum(axis=1)

            vdj = pd.read_csv(paths["gd_tcr"], compression="infer")
            required_vdj = {"barcode", "is_cell", "high_confidence", "chain", "c_gene", "productive"}
            if not required_vdj <= set(vdj.columns):
                raise KeyError(f"GSE268991 VDJ columns missing for {sample}: {required_vdj - set(vdj.columns)}")
            productive = (boolean_series(vdj["is_cell"]) &
                          boolean_series(vdj["high_confidence"]) &
                          boolean_series(vdj["productive"]))
            productive_vdj = vdj.loc[productive].copy()
            productive_vdj["barcode"] = productive_vdj["barcode"].astype(str)
            productive_vdj["TRGC_assignment"] = (
                productive_vdj["c_gene"].astype(str).str.extract(r"(TRGC[12])", expand=False))
            trg = productive_vdj.loc[
                productive_vdj["chain"].astype(str).eq("TRG") &
                productive_vdj["TRGC_assignment"].notna()].copy()
            trd = productive_vdj.loc[productive_vdj["chain"].astype(str).eq("TRD")].copy()
            tcr = pd.DataFrame(index=barcodes)
            tcr["n_productive_TRG"] = trg.groupby("barcode").size().reindex(barcodes, fill_value=0)
            tcr["n_productive_TRD"] = trd.groupby("barcode").size().reindex(barcodes, fill_value=0)
            assignments = trg.groupby("barcode")["TRGC_assignment"].agg(
                lambda values: ";".join(sorted(set(values.dropna().astype(str)))))
            tcr["TRGC_assignment"] = assignments.reindex(barcodes).replace({"": np.nan})
            tcr.loc[tcr["TRGC_assignment"].str.contains(";", na=False), "TRGC_assignment"] = np.nan
            tcr.loc[tcr["n_productive_TRD"].eq(0), "TRGC_assignment"] = np.nan
            tcr.index = cell_ids

            metadata = pd.DataFrame({
                "sample": sample, "donor": donor, "condition": condition,
                "raw_barcode": barcodes.to_numpy(),
            }, index=cell_ids)
            quality = pd.DataFrame({
                "GEX_library_size_over_frozen_universe": library_size,
                "scaled_HVG_relative_error": relative_error,
                "frozen_HVG_coverage": hvg_coverage,
                "sample": sample, "donor": donor, "condition": condition,
            }, index=cell_ids)
            outputs = [
                atomic_parquet(usage, job / "usage_normalized.parquet"),
                atomic_parquet(signature_scores, job / "rank_signature_scores.parquet"),
                atomic_parquet(adt, job / "ADT_raw_and_CLR.parquet"),
                atomic_parquet(tcr, job / "gamma_delta_TCR_metadata.parquet"),
                atomic_parquet(metadata, job / "cell_metadata.parquet"),
                atomic_parquet(quality, job / "quality.parquet"),
            ]
            write_marker(marker, signature, outputs)
            note(f"GSE268991 {sample}", "completed", f"{len(cell_ids)} cells scored")

        def sign_permutation_test(values, alternative="two-sided"):
            values = np.asarray(values, float)
            values = values[np.isfinite(values)]
            if len(values) < 3:
                return np.nan, np.nan, 0
            absolute = np.abs(values)
            signs = np.asarray(list(itertools.product([-1.0, 1.0], repeat=len(values))))
            null = (signs * absolute).mean(axis=1)
            observed = float(values.mean())
            tolerance = 1e-15
            if alternative == "greater":
                p_value = float(np.mean(null >= observed - tolerance))
            elif alternative == "less":
                p_value = float(np.mean(null <= observed + tolerance))
            elif alternative == "two-sided":
                p_value = float(np.mean(np.abs(null) >= abs(observed) - tolerance))
            else:
                raise ValueError(alternative)
            return observed, p_value, len(null)

        def bh_adjust_gse268991(p_values):
            p_values = np.asarray(p_values, float)
            result = np.full(len(p_values), np.nan)
            valid = np.isfinite(p_values)
            if not valid.any():
                return result
            p = p_values[valid]
            order = np.argsort(p)
            ranked = p[order]
            adjusted = np.minimum.accumulate(
                (ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1])[::-1]
            restored = np.empty_like(adjusted)
            restored[order] = np.minimum(adjusted, 1.0)
            result[np.flatnonzero(valid)] = restored
            return result

        def equal_donor_class_weights(donors, outcomes):
            frame = pd.DataFrame({"donor": np.asarray(donors), "outcome": np.asarray(outcomes)})
            counts = frame.groupby(["donor", "outcome"], observed=True).size()
            weights = np.asarray([
                1.0 / counts.loc[(donor, outcome)]
                for donor, outcome in zip(frame["donor"], frame["outcome"])
            ])
            return weights / weights.mean()

        def donor_heldout_trgc_benchmark(features, outcome, donors, model_columns):
            from sklearn.compose import ColumnTransformer
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import (balanced_accuracy_score, brier_score_loss,
                                         log_loss, roc_auc_score)
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import OneHotEncoder, StandardScaler

            donors = pd.Series(donors, index=features.index, dtype=str)
            outcome = pd.Series(outcome, index=features.index, dtype=int)
            candidate_c = (0.01, 0.1, 1.0, 10.0)

            def make_model(numeric_columns, categorical_columns, c_value):
                transformers = [("numeric", StandardScaler(), numeric_columns)]
                if categorical_columns:
                    transformers.append((
                        "categorical", OneHotEncoder(handle_unknown="ignore"),
                        categorical_columns))
                return Pipeline([
                    ("preprocess", ColumnTransformer(transformers, remainder="drop")),
                    ("classifier", LogisticRegression(
                        C=c_value, penalty="l2", solver="liblinear", max_iter=5000,
                        random_state=20260919)),
                ])

            prediction_rows, metric_rows = [], []
            unique_donors = sorted(donors.unique())
            for outer_donor in unique_donors:
                outer_test = donors.eq(outer_donor).to_numpy()
                outer_train = ~outer_test
                if outcome.loc[outer_test].nunique() < 2 or outcome.loc[outer_train].nunique() < 2:
                    continue
                train_donors = sorted(donors.loc[outer_train].unique())
                for model_name, specification in model_columns.items():
                    numeric_columns = specification["numeric"]
                    categorical_columns = specification["categorical"]
                    c_scores = []
                    for c_value in candidate_c:
                        inner_scores = []
                        for inner_donor in train_donors:
                            inner_valid = donors.eq(inner_donor).to_numpy() & outer_train
                            inner_fit = outer_train & ~inner_valid
                            if (outcome.loc[inner_valid].nunique() < 2 or
                                    outcome.loc[inner_fit].nunique() < 2):
                                continue
                            model = make_model(numeric_columns, categorical_columns, c_value)
                            inner_weights = equal_donor_class_weights(
                                donors.loc[inner_fit], outcome.loc[inner_fit])
                            model.fit(features.loc[inner_fit], outcome.loc[inner_fit],
                                      classifier__sample_weight=inner_weights)
                            probability = model.predict_proba(features.loc[inner_valid])[:, 1]
                            inner_scores.append(roc_auc_score(outcome.loc[inner_valid], probability))
                        c_scores.append((np.mean(inner_scores) if inner_scores else -np.inf, c_value))
                    best_score, best_c = sorted(c_scores, key=lambda item: (-item[0], item[1]))[0]
                    model = make_model(numeric_columns, categorical_columns, best_c)
                    weights = equal_donor_class_weights(
                        donors.loc[outer_train], outcome.loc[outer_train])
                    model.fit(features.loc[outer_train], outcome.loc[outer_train],
                              classifier__sample_weight=weights)
                    probability = model.predict_proba(features.loc[outer_test])[:, 1]
                    truth = outcome.loc[outer_test].to_numpy()
                    prediction = probability >= 0.5
                    metric_rows.append({
                        "outer_donor": outer_donor, "model": model_name,
                        "n_cells": int(outer_test.sum()), "n_TRGC1": int(truth.sum()),
                        "n_TRGC2": int((1 - truth).sum()), "selected_C": best_c,
                        "inner_mean_ROC_AUC": best_score,
                        "ROC_AUC": roc_auc_score(truth, probability),
                        "balanced_accuracy": balanced_accuracy_score(truth, prediction),
                        "log_loss": log_loss(truth, probability, labels=[0, 1]),
                        "Brier_score": brier_score_loss(truth, probability),
                    })
                    for cell_id, observed, predicted in zip(
                            features.index[outer_test], truth, probability):
                        prediction_rows.append({
                            "cell_id": cell_id, "outer_donor": outer_donor,
                            "model": model_name, "TRGC1": int(observed),
                            "probability_TRGC1": float(predicted),
                        })
            return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)

        def fixed_prediction_permutation_delta(predictions, model_a, model_b, draws=19999):
            from scipy.stats import rankdata

            wide = predictions.loc[predictions["model"].isin([model_a, model_b])].pivot(
                index=["cell_id", "outer_donor", "TRGC1"], columns="model",
                values="probability_TRGC1").reset_index()
            wide = wide.dropna(subset=[model_a, model_b])
            donor_parts = []
            observed_deltas = []
            for donor, part in wide.groupby("outer_donor", observed=True):
                truth = part["TRGC1"].to_numpy(int)
                n_positive = int(truth.sum()); n_negative = int(len(truth) - n_positive)
                if n_positive == 0 or n_negative == 0:
                    continue
                rank_delta = (rankdata(part[model_a].to_numpy(float), method="average") -
                              rankdata(part[model_b].to_numpy(float), method="average"))
                denominator = n_positive * n_negative
                observed_deltas.append(float(rank_delta[truth == 1].sum() / denominator))
                donor_parts.append((rank_delta / denominator, n_positive))
            if len(donor_parts) < 3:
                raise RuntimeError("Too few donors with both TRGC classes for permutation")
            rng = np.random.default_rng(20260919 + 268991)
            null = np.zeros(draws, dtype=float)
            for rank_delta, n_positive in donor_parts:
                for draw in range(draws):
                    selected = rng.choice(len(rank_delta), size=n_positive, replace=False)
                    null[draw] += rank_delta[selected].sum()
            null /= len(donor_parts)
            observed = float(np.mean(observed_deltas))
            p_value = float((1 + np.sum(np.abs(null) >= abs(observed) - 1e-15)) / (draws + 1))
            return observed, p_value, np.asarray(observed_deltas), len(donor_parts)

        def run_gse268991_final_validation():
            from sklearn.metrics import roc_auc_score

            if not GSE268991_PREANALYSIS.is_file():
                raise FileNotFoundError(GSE268991_PREANALYSIS)
            preanalysis_hash = sha256sum(GSE268991_PREANALYSIS)
            if preanalysis_hash != GSE268991_PREANALYSIS_SHA256:
                raise RuntimeError(
                    "GSE268991 preanalysis manifest changed after lock; refusing evaluation")
            preanalysis = json.loads(GSE268991_PREANALYSIS.read_text())
            if preanalysis["untouched_status"]["expression_or_adt_matrix_accessed_before_lock"]:
                raise RuntimeError("GSE268991 is not marked untouched in the locked manifest")
            scorer = load_frozen_scorer_for_gse268991()
            GSE268991_PROJECTION_DIR.mkdir(parents=True, exist_ok=True)
            GSE268991_PROJECTION_CHECKPOINTS.mkdir(parents=True, exist_ok=True)
            for sample in GSE268991_SAMPLES:
                score_gse268991_sample(sample, scorer, preanalysis_hash)

            containers = {key: [] for key in ("usage", "signature", "adt", "tcr", "metadata", "quality")}
            inventory_rows = []
            for sample in GSE268991_SAMPLES:
                job = GSE268991_PROJECTION_CHECKPOINTS / sample
                containers["usage"].append(pd.read_parquet(job / "usage_normalized.parquet"))
                containers["signature"].append(pd.read_parquet(job / "rank_signature_scores.parquet"))
                containers["adt"].append(pd.read_parquet(job / "ADT_raw_and_CLR.parquet"))
                containers["tcr"].append(pd.read_parquet(job / "gamma_delta_TCR_metadata.parquet"))
                containers["metadata"].append(pd.read_parquet(job / "cell_metadata.parquet"))
                containers["quality"].append(pd.read_parquet(job / "quality.parquet"))
                spec = json.loads((job / "job_spec.json").read_text())
                tcr_part = containers["tcr"][-1]
                inventory_rows.append({
                    "sample": sample, "donor": spec["donor"], "condition": spec["condition"],
                    "n_cells": spec["n_cells"], "n_gene_features": spec["n_gene_features"],
                    "n_ADT_features": spec["n_ADT_features"],
                    "frozen_HVG_coverage": spec["frozen_HVG_coverage"],
                    "n_unambiguous_productive_TRGC": int(
                        tcr_part["TRGC_assignment"].isin(["TRGC1", "TRGC2"]).sum()),
                    "checkpoint_signature": stable_token(spec),
                })
            frames = {key: pd.concat(parts, axis=0, sort=False) for key, parts in containers.items()}
            expected_cells = pd.Index(frames["metadata"].index)
            if not expected_cells.is_unique:
                raise ValueError("GSE268991 prefixed cell identifiers are not unique")
            for key, frame in frames.items():
                if set(frame.index) != set(expected_cells):
                    raise ValueError(f"GSE268991 {key} cells do not match metadata")
                frames[key] = frame.loc[expected_cells]
            usage = frames["usage"].rename(columns=scorer["component_to_program"])
            signature = frames["signature"].rename(columns=scorer["component_to_program"])
            programs = ["P1", "P2", "P3", "P4", "P5", "P7", "P9", "P10"]
            usage = usage.loc[:, programs]
            signature = signature.loc[:, programs]
            quality = frames["quality"]
            metadata = frames["metadata"]
            tcr = frames["tcr"]
            adt = frames["adt"]
            if not np.allclose(usage.sum(axis=1), 1.0, atol=1e-8):
                raise ValueError("GSE268991 normalized K8 usages must sum to one")

            gray_quality_paths = [
                CHECKPOINT_DIR / "full" / f"fold{fold}_K8" / "heldout_quality.parquet"
                for fold in range(N_FOLDS)]
            gray_errors = pd.concat([
                pd.read_parquet(path)["scaled_HVG_relative_error"]
                for path in gray_quality_paths]).astype(float)
            gray_q90, gray_q95 = [float(gray_errors.quantile(value)) for value in (0.90, 0.95)]
            locked_q90 = float(preanalysis["frozen_model"]["strict_sensitivity_gate"]["maximum"])
            locked_q95 = float(preanalysis["frozen_model"]["primary_quality_gate"]["maximum"])
            if not (np.isclose(gray_q90, locked_q90) and np.isclose(gray_q95, locked_q95)):
                raise RuntimeError("Gray quality thresholds differ from the locked manifest")
            quality["within_Gray_q90"] = quality["scaled_HVG_relative_error"].le(gray_q90)
            quality["within_Gray_q95"] = quality["scaled_HVG_relative_error"].le(gray_q95)
            quality["reconstruction_gate"] = np.select(
                [quality["within_Gray_q90"], quality["within_Gray_q95"]],
                ["high_confidence_within_q90", "accepted_q90_to_q95"],
                default="abstain_above_q95")

            usage_path = atomic_parquet(
                usage, GSE268991_PROJECTION_DIR / "GSE268991_primary_K8_usage_normalized.parquet")
            signature_path = atomic_parquet(
                signature, GSE268991_PROJECTION_DIR / "GSE268991_primary_K8_rank_signature_scores.parquet")
            quality_path = atomic_parquet(
                quality, GSE268991_PROJECTION_DIR / "GSE268991_primary_K8_quality.parquet")
            adt_path = atomic_parquet(
                adt, GSE268991_PROJECTION_DIR / "GSE268991_ADT_raw_and_CLR.parquet")
            tcr_path = atomic_parquet(
                tcr, GSE268991_PROJECTION_DIR / "GSE268991_gamma_delta_TCR_metadata.parquet")
            metadata_path = atomic_parquet(
                metadata, GSE268991_PROJECTION_DIR / "GSE268991_cell_metadata.parquet")
            input_audit = pd.DataFrame(inventory_rows)
            save_table(input_audit, "GSE268991_primary_K8_input_audit")

            quality_summary = []
            for source, values in (
                    ("Gray_donor_heldout_K8", gray_errors),
                    ("GSE268991_external", quality["scaled_HVG_relative_error"].astype(float))):
                quality_summary.append({
                    "source": source, "n_cells": len(values),
                    "median_relative_error": float(values.median()),
                    "q90_relative_error": float(values.quantile(0.90)),
                    "q95_relative_error": float(values.quantile(0.95)),
                    "fraction_above_Gray_q90": float(values.gt(gray_q90).mean()),
                    "fraction_above_Gray_q95": float(values.gt(gray_q95).mean()),
                })
            quality_summary = pd.DataFrame(quality_summary)
            save_table(quality_summary, "GSE268991_primary_K8_reconstruction_quality_gate")

            joined = usage.join(metadata[["sample", "donor", "condition"]]).join(
                tcr[["TRGC_assignment", "n_productive_TRG", "n_productive_TRD"]])
            accepted = quality["within_Gray_q95"].astype(bool)
            tcr_eligible = accepted & joined["TRGC_assignment"].isin(["TRGC1", "TRGC2"])
            target_quality_rows = []
            population_masks = {
                "locked_primary_all_matched_GEX_ADT": pd.Series(True, index=quality.index),
                "posthoc_productive_TRD_confirmed_gdT": joined["n_productive_TRD"].fillna(0).gt(0),
                "posthoc_unambiguous_TRGC1_or_TRGC2": joined["TRGC_assignment"].isin(
                    ["TRGC1", "TRGC2"]),
            }
            for population, mask in population_masks.items():
                values = quality.loc[mask, "scaled_HVG_relative_error"].astype(float)
                target_quality_rows.append({
                    "population": population,
                    "analysis_status": ("locked_primary" if population.startswith("locked_primary")
                                        else "posthoc_target_population_diagnostic"),
                    "n_cells": len(values),
                    "median_relative_error": float(values.median()),
                    "fraction_within_Gray_q90": float(values.le(gray_q90).mean()),
                    "fraction_within_Gray_q95": float(values.le(gray_q95).mean()),
                    "fraction_above_Gray_q95": float(values.gt(gray_q95).mean()),
                })
            target_quality = pd.DataFrame(target_quality_rows)
            save_table(target_quality, "GSE268991_primary_K8_target_population_quality_sensitivity")
            tcr_scores = (joined.loc[tcr_eligible]
                          .groupby(["donor", "condition", "TRGC_assignment"], observed=True)
                          [programs].median().reset_index())
            donor_tcr_scores = (tcr_scores.groupby(["donor", "TRGC_assignment"], observed=True)
                                [programs].median().reset_index())
            save_table(donor_tcr_scores, "GSE268991_primary_K8_TRGC_donor_scores")
            p2 = donor_tcr_scores.pivot(index="donor", columns="TRGC_assignment", values="P2").dropna()
            p3 = donor_tcr_scores.pivot(index="donor", columns="TRGC_assignment", values="P3").dropna()
            h1_values = (p2["TRGC1"] - p2["TRGC2"]).to_numpy(float)
            h2_values = (p3["TRGC2"] - p3["TRGC1"]).to_numpy(float)
            h1_effect, h1_p, h1_permutations = sign_permutation_test(h1_values, "greater")
            h2_effect, h2_p, h2_permutations = sign_permutation_test(h2_values, "greater")

            required_adt = ["ADT_CLR__CD45RA", "ADT_CLR__CCR7"]
            if not set(required_adt) <= set(adt.columns):
                raise KeyError(f"GSE268991 required ADTs absent: {set(required_adt) - set(adt.columns)}")
            protein = usage.join(metadata[["donor", "condition"]]).join(adt[required_adt])
            protein = protein.loc[accepted].copy()
            protein["naive_ADT_composite"] = protein[required_adt].mean(axis=1)
            h3_rows = []
            for donor, part in protein.groupby("donor", observed=True):
                h3_rows.append({
                    "donor": donor,
                    "n_cells": len(part),
                    "P2_naive_ADT_spearman_rho": rho(
                        part["P2"], part["naive_ADT_composite"]),
                })
            h3_donor = pd.DataFrame(h3_rows)
            h3_effect, h3_p, h3_permutations = sign_permutation_test(
                h3_donor["P2_naive_ADT_spearman_rho"], "less")
            save_table(h3_donor, "GSE268991_primary_K8_paired_ADT_P2_donor_correlations")
            hypothesis_tests = pd.DataFrame([
                {"hypothesis": "H1", "contrast": "P2_TRGC1_minus_TRGC2",
                 "direction": "greater_than_zero", "n_donors": len(h1_values),
                 "effect_mean_donor_paired_difference": h1_effect,
                 "one_sided_exact_sign_permutation_p_value": h1_p,
                 "exact_permutations": h1_permutations,
                 "evidence": "independent gamma-delta TCR constant-gene assignment"},
                {"hypothesis": "H2", "contrast": "P3_TRGC2_minus_TRGC1",
                 "direction": "greater_than_zero", "n_donors": len(h2_values),
                 "effect_mean_donor_paired_difference": h2_effect,
                 "one_sided_exact_sign_permutation_p_value": h2_p,
                 "exact_permutations": h2_permutations,
                 "evidence": "independent gamma-delta TCR constant-gene assignment"},
                {"hypothesis": "H3", "contrast": "P2_vs_CD45RA_CCR7_ADT_naive_composite",
                 "direction": "less_than_zero", "n_donors": len(h3_donor),
                 "effect_mean_donor_paired_difference": h3_effect,
                 "one_sided_exact_sign_permutation_p_value": h3_p,
                 "exact_permutations": h3_permutations,
                 "evidence": "same-cell CITE-seq surface protein"},
            ])
            hypothesis_tests["q_value_BH_locked_primary_family"] = bh_adjust_gse268991(
                hypothesis_tests["one_sided_exact_sign_permutation_p_value"])
            save_table(hypothesis_tests, "GSE268991_primary_K8_locked_hypothesis_tests")

            adt_clr_columns = [column for column in adt.columns if column.startswith("ADT_CLR__")]
            adt_association_rows = []
            for program in programs:
                for adt_column in adt_clr_columns:
                    donor_correlations = []
                    for donor, ids in metadata.loc[accepted].groupby("donor", observed=True).groups.items():
                        donor_correlations.append(rho(usage.loc[ids, program], adt.loc[ids, adt_column]))
                    effect, p_value, permutations = sign_permutation_test(
                        donor_correlations, "two-sided")
                    adt_association_rows.append({
                        "program": program, "ADT_feature": adt_column.removeprefix("ADT_CLR__"),
                        "n_donors": int(np.isfinite(donor_correlations).sum()),
                        "mean_within_donor_spearman_rho": effect,
                        "two_sided_exact_sign_permutation_p_value": p_value,
                        "exact_permutations": permutations,
                    })
            adt_associations = pd.DataFrame(adt_association_rows)
            adt_associations["q_value_BH_all_program_ADT_pairs"] = bh_adjust_gse268991(
                adt_associations["two_sided_exact_sign_permutation_p_value"])
            save_table(adt_associations, "GSE268991_primary_K8_program_ADT_associations")

            benchmark_ids = joined.index[tcr_eligible]
            benchmark = pd.concat([
                usage.loc[benchmark_ids].add_prefix("cNMF_"),
                signature.loc[benchmark_ids].add_prefix("rank_signature_"),
                metadata.loc[benchmark_ids, ["condition"]],
            ], axis=1)
            benchmark["log1p_GEX_library_size"] = np.log1p(
                quality.loc[benchmark_ids, "GEX_library_size_over_frozen_universe"].astype(float))
            benchmark["log1p_ADT_total_counts"] = np.log1p(
                adt.loc[benchmark_ids, "ADT_total_counts"].astype(float))
            outcome = joined.loc[benchmark_ids, "TRGC_assignment"].eq("TRGC1").astype(int)
            donors = metadata.loc[benchmark_ids, "donor"].astype(str)
            cNMF_columns = [f"cNMF_{program}" for program in programs]
            signature_columns = [f"rank_signature_{program}" for program in programs]
            qc_columns = ["log1p_GEX_library_size", "log1p_ADT_total_counts"]
            model_columns = {
                "covariates_only": {"numeric": qc_columns, "categorical": ["condition"]},
                "cNMF": {"numeric": cNMF_columns, "categorical": []},
                "rank_signature": {"numeric": signature_columns, "categorical": []},
                "covariates_plus_cNMF": {
                    "numeric": qc_columns + cNMF_columns, "categorical": ["condition"]},
                "covariates_plus_rank_signature": {
                    "numeric": qc_columns + signature_columns, "categorical": ["condition"]},
            }
            benchmark_metrics, benchmark_predictions = donor_heldout_trgc_benchmark(
                benchmark, outcome, donors, model_columns)
            if benchmark_metrics["outer_donor"].nunique() < 5:
                raise RuntimeError("Fewer than five GSE268991 donors support donor-held-out benchmarking")
            save_table(benchmark_metrics, "GSE268991_primary_K8_donor_heldout_TRGC_benchmark")
            benchmark_prediction_path = atomic_parquet(
                benchmark_predictions,
                GSE268991_PROJECTION_DIR / "GSE268991_donor_heldout_TRGC_predictions.parquet")
            model_a = "covariates_plus_cNMF"
            model_b = "covariates_plus_rank_signature"
            delta, permutation_p, donor_deltas, n_benchmark_donors = (
                fixed_prediction_permutation_delta(
                    benchmark_predictions, model_a, model_b, draws=19999))
            rng = np.random.default_rng(20260919 + 1)
            bootstrap = np.mean(
                rng.choice(donor_deltas, size=(10000, len(donor_deltas)), replace=True), axis=1)
            ci_low, ci_high = np.quantile(bootstrap, [0.025, 0.975])
            metric_means = benchmark_metrics.groupby("model", observed=True)[
                ["ROC_AUC", "balanced_accuracy", "log_loss", "Brier_score"]].mean()
            calibration_noninferior = bool(
                metric_means.loc[model_a, "log_loss"] <= metric_means.loc[model_b, "log_loss"] and
                metric_means.loc[model_a, "Brier_score"] <= metric_means.loc[model_b, "Brier_score"])
            incremental_value = bool(ci_low > 0 and calibration_noninferior)
            incremental_summary = pd.DataFrame([{
                "endpoint": "productive_unambiguous_TRGC1_vs_TRGC2",
                "n_outer_donors": n_benchmark_donors,
                "mean_ROC_AUC_covariates_plus_cNMF": metric_means.loc[model_a, "ROC_AUC"],
                "mean_ROC_AUC_covariates_plus_rank_signature": metric_means.loc[model_b, "ROC_AUC"],
                "mean_donor_ROC_AUC_delta_cNMF_minus_signature": delta,
                "donor_bootstrap_CI95_low": ci_low, "donor_bootstrap_CI95_high": ci_high,
                "bootstrap_draws": 10000,
                "fixed_prediction_within_donor_permutation_p_value": permutation_p,
                "permutation_draws": 19999,
                "cNMF_mean_log_loss": metric_means.loc[model_a, "log_loss"],
                "signature_mean_log_loss": metric_means.loc[model_b, "log_loss"],
                "cNMF_mean_Brier_score": metric_means.loc[model_a, "Brier_score"],
                "signature_mean_Brier_score": metric_means.loc[model_b, "Brier_score"],
                "calibration_noninferior": calibration_noninferior,
                "incremental_value_established": incremental_value,
                "split": "leave_one_donor_out_with_inner_donor_C_selection",
            }])
            save_table(incremental_summary, "GSE268991_primary_K8_incremental_value_summary")

            condition_scores = (usage.loc[accepted]
                                .join(metadata[["donor", "condition"]])
                                .groupby(["donor", "condition"], observed=True)[programs]
                                .median().reset_index())
            paired_condition = condition_scores.pivot(index="donor", columns="condition", values=programs)
            condition_rows = []
            for program in programs:
                differences = (paired_condition[(program, "Tumor")] -
                               paired_condition[(program, "Normal")]).dropna().to_numpy(float)
                effect, p_value, permutations = sign_permutation_test(differences, "two-sided")
                condition_rows.append({
                    "program": program, "contrast": "Tumor_minus_Normal",
                    "n_paired_donors": len(differences),
                    "mean_paired_difference": effect,
                    "two_sided_exact_sign_permutation_p_value": p_value,
                    "exact_permutations": permutations,
                })
            condition_tests = pd.DataFrame(condition_rows)
            condition_tests["q_value_BH_eight_programs"] = bh_adjust_gse268991(
                condition_tests["two_sided_exact_sign_permutation_p_value"])
            save_table(condition_tests, "GSE268991_primary_K8_paired_Tumor_Normal_program_tests")

            trd_accepted = accepted & joined["n_productive_TRD"].fillna(0).gt(0)
            trd_condition_scores = (usage.loc[trd_accepted]
                                    .join(metadata[["donor", "condition"]])
                                    .groupby(["donor", "condition"], observed=True)[programs]
                                    .median().reset_index())
            trd_paired_condition = trd_condition_scores.pivot(
                index="donor", columns="condition", values=programs)
            trd_condition_rows = []
            for program in programs:
                differences = (trd_paired_condition[(program, "Tumor")] -
                               trd_paired_condition[(program, "Normal")]).dropna().to_numpy(float)
                effect, p_value, permutations = sign_permutation_test(differences, "two-sided")
                trd_condition_rows.append({
                    "program": program, "contrast": "Tumor_minus_Normal",
                    "population": "posthoc_productive_TRD_confirmed_gdT_within_Gray_q95",
                    "analysis_status": "posthoc_target_population_sensitivity",
                    "n_paired_donors": len(differences),
                    "mean_paired_difference": effect,
                    "two_sided_exact_sign_permutation_p_value": p_value,
                    "exact_permutations": permutations,
                })
            trd_condition_tests = pd.DataFrame(trd_condition_rows)
            trd_condition_tests["q_value_BH_eight_programs"] = bh_adjust_gse268991(
                trd_condition_tests["two_sided_exact_sign_permutation_p_value"])
            save_table(
                trd_condition_tests,
                "GSE268991_primary_K8_paired_Tumor_Normal_program_tests_TCR_confirmed_sensitivity")

            figure_data = benchmark_metrics.pivot(
                index="outer_donor", columns="model", values="ROC_AUC").dropna()
            fig, axes = plt.subplots(2, 2, figsize=(8.5, 7.5))
            for donor in p2.index:
                axes[0, 0].plot([0, 1], p2.loc[donor, ["TRGC2", "TRGC1"]],
                                color="#BDBDBD", linewidth=0.8, zorder=1)
                axes[0, 0].scatter([0, 1], p2.loc[donor, ["TRGC2", "TRGC1"]],
                                   color=["#D55E00", "#0072B2"], s=24, zorder=2)
            axes[0, 0].set(xticks=[0, 1], xticklabels=["TRGC2", "TRGC1"], ylabel="P2 usage",
                           title=f"Locked H1: mean paired Δ={h1_effect:.3f}, q="
                                 f"{hypothesis_tests.loc[hypothesis_tests.hypothesis.eq('H1'), 'q_value_BH_locked_primary_family'].iloc[0]:.3g}")
            for donor in p3.index:
                axes[0, 1].plot([0, 1], p3.loc[donor, ["TRGC1", "TRGC2"]],
                                color="#BDBDBD", linewidth=0.8, zorder=1)
                axes[0, 1].scatter([0, 1], p3.loc[donor, ["TRGC1", "TRGC2"]],
                                   color=["#0072B2", "#D55E00"], s=24, zorder=2)
            axes[0, 1].set(xticks=[0, 1], xticklabels=["TRGC1", "TRGC2"], ylabel="P3 usage",
                           title=f"Locked H2: mean paired Δ={h2_effect:.3f}, q="
                                 f"{hypothesis_tests.loc[hypothesis_tests.hypothesis.eq('H2'), 'q_value_BH_locked_primary_family'].iloc[0]:.3g}")
            axes[1, 0].axhline(0, color="black", linewidth=0.8)
            axes[1, 0].scatter(np.arange(len(h3_donor)), h3_donor["P2_naive_ADT_spearman_rho"],
                               color="#0072B2", s=28)
            axes[1, 0].set(xticks=np.arange(len(h3_donor)),
                           xticklabels=h3_donor["donor"], ylabel="Within-donor Spearman ρ",
                           title=("Locked H3: P2 vs CD45RA/CCR7 ADT\n"
                                  f"mean within-donor ρ={h3_effect:.2f}, q="
                                  f"{hypothesis_tests.loc[hypothesis_tests.hypothesis.eq('H3'), 'q_value_BH_locked_primary_family'].iloc[0]:.3g}"))
            axes[1, 0].tick_params(axis="x", rotation=45)
            axes[1, 1].plot([0, 1], [0, 1], color="black", linewidth=0.8, linestyle="--")
            axes[1, 1].scatter(figure_data[model_b], figure_data[model_a],
                               color="#0072B2", s=32)
            axes[1, 1].set(xlim=(0, 1), ylim=(0, 1),
                           xlabel="Covariates + rank signatures ROC AUC",
                           ylabel="Covariates + cNMF ROC AUC",
                           title=("Donor-held-out TRGC benchmark\n"
                                  f"n={len(figure_data)} donors, mean AUC Δ={delta:.3f}"))
            fig.suptitle("GSE268991 locked paired RNA–protein–TCR validation")
            save_figure(fig, "GSE268991_primary_K8_locked_multimodal_validation")

            model_card_path = PRIMARY_FREEZE_DIR / "primary_K8_model_card.json"
            model_card = json.loads(model_card_path.read_text())
            model_card["updated_at"] = datetime.now(timezone.utc).isoformat()
            readiness = model_card.setdefault("readiness", {})
            readiness["untouched_external_validation_completed"] = True
            readiness["paired_same_cell_RNA_protein_TCR_validation_completed"] = True
            readiness["incremental_value_over_simple_signature_established"] = incremental_value
            readiness["prospective_killing_or_degranulation_validation_completed"] = False
            readiness["NC_claim_package_complete"] = False
            model_card.setdefault("validation_readout", {})[
                "external_GSE268991_evidence_tier"] = (
                    "locked untouched final cohort; seven paired colorectal donors with raw-UMI "
                    "GEX, same-cell CITE-seq ADT and independent gamma-delta TCR")
            model_card["validation_readout"]["external_GSE268991_primary_hypotheses"] = (
                "; ".join(
                    f"{row.hypothesis} effect={row.effect_mean_donor_paired_difference:.4g}, "
                    f"q={row.q_value_BH_locked_primary_family:.4g}"
                    for row in hypothesis_tests.itertuples(index=False)))
            model_card["validation_readout"]["external_GSE268991_incremental_value"] = (
                f"mean donor ROC-AUC delta cNMF-minus-signature={delta:.4g}, "
                f"95% donor-bootstrap CI=[{ci_low:.4g}, {ci_high:.4g}], "
                f"incremental value established={incremental_value}")
            gd_target_quality = target_quality.loc[
                target_quality["population"].eq("posthoc_unambiguous_TRGC1_or_TRGC2")].iloc[0]
            model_card["validation_readout"]["external_GSE268991_reconstruction_scope"] = (
                f"locked all-cell Gray-q95 acceptance={accepted.mean():.3f}; post-hoc "
                f"TCR-confirmed unambiguous gamma-delta target-cell acceptance="
                f"{gd_target_quality.fraction_within_Gray_q95:.3f}. The latter diagnoses "
                "intended-population transportability but does not replace the locked all-cell result.")
            limitations = [
                "GSE268991 has seven paired colorectal donors; cells are nested subsamples and donor is the inference unit.",
                "GSE268991 supplies same-cell surface protein and independent TCR evidence but no GZMB, PRF1, CD107a, TNF, killing or degranulation endpoint.",
                "The untouched GSE268991 evaluation is one-shot: failed hypotheses or transportability cannot trigger threshold or model tuning.",
                "The locked GSE268991 all-cell population has a large reconstruction-error tail; high acceptance in TCR-confirmed gamma-delta cells is a post-hoc intended-population diagnostic and must be labelled as such.",
            ]
            model_card.setdefault("limitations", [])
            for limitation in limitations:
                if limitation not in model_card["limitations"]:
                    model_card["limitations"].append(limitation)
            model_card["status"] = (
                "research-grade frozen scorer; untouched paired multimodal external validation "
                f"completed; incremental value established={incremental_value}; "
                "prospective cytotoxicity validation still required")
            atomic_json(model_card_path, model_card)
            atomic_json(TABLE_DIR / "primary_K8_model_card.json", model_card)

            output_paths = [
                TABLE_DIR / "GSE268991_primary_K8_input_audit.csv",
                TABLE_DIR / "GSE268991_primary_K8_reconstruction_quality_gate.csv",
                TABLE_DIR / "GSE268991_primary_K8_target_population_quality_sensitivity.csv",
                TABLE_DIR / "GSE268991_primary_K8_TRGC_donor_scores.csv",
                TABLE_DIR / "GSE268991_primary_K8_paired_ADT_P2_donor_correlations.csv",
                TABLE_DIR / "GSE268991_primary_K8_locked_hypothesis_tests.csv",
                TABLE_DIR / "GSE268991_primary_K8_program_ADT_associations.csv",
                TABLE_DIR / "GSE268991_primary_K8_donor_heldout_TRGC_benchmark.csv",
                TABLE_DIR / "GSE268991_primary_K8_incremental_value_summary.csv",
                TABLE_DIR / "GSE268991_primary_K8_paired_Tumor_Normal_program_tests.csv",
                TABLE_DIR / "GSE268991_primary_K8_paired_Tumor_Normal_program_tests_TCR_confirmed_sensitivity.csv",
                FIGURE_DIR / "GSE268991_primary_K8_locked_multimodal_validation.pdf",
                usage_path, signature_path, quality_path, adt_path, tcr_path, metadata_path,
                benchmark_prediction_path, model_card_path,
            ]
            projection_manifest = TABLE_DIR / "GSE268991_primary_K8_final_validation_manifest.json"
            atomic_json(projection_manifest, {
                "schema_version": 1, "analysis": ANALYSIS_NAME,
                "stage": "gse268991_final_validation", "dataset": "GSE268991",
                "preanalysis_manifest_path": str(GSE268991_PREANALYSIS),
                "preanalysis_manifest_sha256": preanalysis_hash,
                "untouched_evaluation": True,
                "n_matrix_cells_released": int(input_audit["n_cells"].sum()),
                "n_scored_cells_nonzero_frozen_universe": len(expected_cells),
                "n_donors": metadata["donor"].nunique(),
                "n_paired_normal_tumor_donors": metadata.groupby("donor")["condition"].nunique().eq(2).sum(),
                "n_q95_accepted_cells": int(accepted.sum()),
                "fraction_q95_accepted": float(accepted.mean()),
                "n_unambiguous_productive_TRGC_cells_before_quality_gate": int(
                    joined["TRGC_assignment"].isin(["TRGC1", "TRGC2"]).sum()),
                "n_unambiguous_productive_TRGC_cells_after_Gray_q95_gate": int(tcr_eligible.sum()),
                "target_population_quality_sensitivity": target_quality.to_dict(orient="records"),
                "locked_primary_hypotheses": hypothesis_tests.to_dict(orient="records"),
                "incremental_value_established": incremental_value,
                "external_data_used_for_model_selection_or_tuning": False,
                "receptor_genes_in_frozen_HVGs": [],
                "evidence_boundary": (
                    "paired same-cell RNA-surface-protein and independent TCR validation; "
                    "not a killing or degranulation assay"),
                "outputs": {str(path): sha256sum(path) for path in output_paths},
            })
            note("GSE268991 final external validation", "completed_locked_one_shot",
                 f"accepted={accepted.mean():.3f}; incremental value={incremental_value}")
            display(hypothesis_tests)
            display(incremental_summary)

        if EXECUTION_STAGE == "gse268991_final_validation":
            run_gse268991_final_validation()
        else:
            print("GSE268991 final validation not launched in stage", EXECUTION_STAGE)
        """),
        md(r"""
        ## 7 · External-validation decision boundary

        Read the external outputs in this order:

        1. `GSE294268_primary_K8_reconstruction_quality_gate.csv`: compare the
           external error distribution with donor-held-out Gray cells. The Gray
           q95 is the acceptance gate; q90 is a stricter sensitivity analysis.
        2. `GSE294268_allergic_experiment_confounding_audit.csv`: determine whether
           a disease contrast is identifiable independently of Experiment before
           interpreting any pooled effect.
        3. `GSE294268_primary_K8_quality_sensitivity_donor_tests.csv`: require the
           paired NonA stimulation effects to persist across all, q95 and q90 gates.
        4. `GSE294268_primary_K8_cNMF_vs_rank_signature_effects.csv`: test whether
           cNMF adds information beyond the same genes used as simple rank scores.
        5. `GSE294268_primary_K8_program_prevalence.csv`: identify components that
           receive little or no external support rather than silently treating all
           eight programs as equally validated.
        6. `primary_K8_model_card.json`: the authoritative completion and prohibited-
           use record for this frozen scorer.

        For PANGEA, read `PANGEA_primary_K8_input_audit.csv` first, followed by
        `PANGEA_primary_K8_reconstruction_quality_gate.csv`,
        `PANGEA_primary_K8_series_quality.csv`, and
        `PANGEA_primary_K8_program_prevalence.csv`. The TRDV1/TRDV2 table is a
        Series-level association with expression-derived silver labels; it is not
        receptor-truth subtype validation. PANGEA has no orthogonal functional
        endpoint in this object, so it cannot establish functional validity or
        incremental value over a marker signature.

        For Gray Data S12, read `Gray_scRNA_Data_S12_donor_overlap.csv` before the
        association results. Zero overlap means that the flow assay is orthogonal
        and independent-donor, but not paired. The primary P2–GzmB result is reported
        together with the train-only rank-signature baseline and donor-bootstrap
        difference. A positive functional association does not establish cNMF's
        incremental value when the simple signature performs similarly.

        A significant pooled allergic-versus-NonA result is not disease validation
        when Experiment and allergy status are confounded. Agreement with a simple
        rank-signature score is not orthogonal validation. The scorer is complete as
        a frozen research artifact only when its model card, quality gate and hashes
        validate; an NC-level biological claim additionally needs incremental-value
        evidence and orthogonal protein or functional validation.
        """),
        md(r"""
        ## 6 · Assemble donor-level evidence and cross-K recurrence

        The thresholds below are predeclared operational recurrence screens, not
        biological truth: cosine ≥ 0.80, top-50 Jaccard ≥ 0.20 and positive matching
        margin. Continuous metrics remain in the exported tables. K selection must
        consider stability and held-out reconstruction together; external GSE294268
        results are not part of K selection.
        """),
        code(r"""
        # 9. Verify job checkpoints, aggregate evidence and produce vector figures
        if EXECUTION_STAGE in {"assemble_screen", "assemble_full"}:
            all_metrics, all_matches, inventory = [], [], []
            for fold in range(N_FOLDS):
                for k in K_GRID:
                    label = f"fold{fold}_K{k}"
                    job = CHECKPOINT_DIR / RUN_TIER / label
                    spec_path = job / "job_spec.json"
                    marker_path = job / "heldout_scoring.complete.json"
                    failure_path = job / "consensus_density_filter.failure.json"
                    state = "missing"
                    if spec_path.is_file() and marker_path.is_file():
                        signature = stable_token(json.loads(spec_path.read_text()))
                        state = "complete" if validate_marker(marker_path, signature) else "invalid"
                    elif validated_density_failure(job):
                        state = "failed_consensus_density_filter"
                    inventory.append({"tier": RUN_TIER, "fold": fold, "K": k, "job": label, "status": state})
                    if state == "complete":
                        all_metrics.append(pd.read_csv(job / "heldout_donor_metrics.csv"))
                        all_matches.append(pd.read_csv(job / "reference_matching.csv"))
            inventory = pd.DataFrame(inventory)
            save_table(inventory, f"Gray_{RUN_TIER}_checkpoint_inventory")
            unresolved = inventory.status.isin(["missing", "invalid"])
            if unresolved.any():
                missing_jobs = inventory.loc[unresolved, "job"].tolist()
                raise RuntimeError(f"Unresolved {RUN_TIER} jobs prevent assembly: {missing_jobs}")
            if not all_metrics:
                raise RuntimeError(f"No completed {RUN_TIER} jobs are available to assemble")
            metrics = pd.concat(all_metrics, ignore_index=True)
            matches = pd.concat(all_matches, ignore_index=True)
            matches["reference_program"] = matches["reference_program"].map(
                lambda x: str(int(float(x))))
            matches["fold_program"] = matches["fold_program"].map(lambda x: str(int(float(x))))
            save_table(metrics, f"Gray_{RUN_TIER}_heldout_donor_metrics")
            save_table(matches, f"Gray_{RUN_TIER}_reference_matching_all_pairs")

            # K10-reference recovery is an asymmetric diagnostic. Every completed
            # fold contributes all ten reference programs; a reference program that
            # cannot be assigned when K < 10 is explicitly counted as not recovered.
            assigned = matches.loc[matches.assigned_one_to_one].copy()
            assigned["operational_recurrence_pass"] = (
                assigned.cosine.ge(0.80) & assigned.top50_jaccard.ge(0.20) &
                assigned.margin_to_best_alternative.gt(0))
            completed_folds = inventory.loc[inventory.status.eq("complete"), ["K", "fold"]].copy()
            completed_folds["_join"] = 1
            reference_grid = pd.DataFrame({"reference_program": reference_spectra.index.astype(str), "_join": 1})
            recovery_observations = (completed_folds.merge(reference_grid, on="_join").drop(columns="_join")
                                     .merge(assigned, on=["K", "fold", "reference_program"], how="left"))
            recovery_observations["assigned_one_to_one"] = (
                recovery_observations["assigned_one_to_one"].fillna(False).astype(bool))
            recovery_observations["operational_recurrence_pass"] = (
                recovery_observations["operational_recurrence_pass"].fillna(False).astype(bool))
            recurrence = (recovery_observations.groupby(["K", "reference_program"], observed=True)
                          .agg(n_completed_folds=("fold", "nunique"),
                               n_assigned_folds=("assigned_one_to_one", "sum"),
                               n_passing_folds=("operational_recurrence_pass", "sum"),
                               recurrence_fraction=("operational_recurrence_pass", "mean"),
                               median_cosine=("cosine", "median"), min_cosine=("cosine", "min"),
                               median_top50_jaccard=("top50_jaccard", "median"),
                               min_margin=("margin_to_best_alternative", "min"))
                          .reset_index())
            conditional = (assigned.groupby(["K", "reference_program"], observed=True)
                           .agg(conditional_recurrence_fraction=("operational_recurrence_pass", "mean"))
                           .reset_index())
            recurrence = recurrence.merge(conditional, on=["K", "reference_program"], how="left")
            recurrence["n_folds"] = recurrence["n_completed_folds"]
            recurrence["reference_basis"] = "full_Gray_K10"
            save_table(recurrence, f"Gray_{RUN_TIER}_cross_K_program_recurrence")

            # K-neutral evidence: compare consensus spectra between every pair of
            # completed donor folds at the same K. This does not privilege K=10.
            density = str(DENSITY_THRESHOLD).replace(".", "_")
            spectra_by_fold = {}
            for row in completed_folds.itertuples(index=False):
                job = CHECKPOINT_DIR / RUN_TIER / f"fold{row.fold}_K{row.K}"
                path = one_file(job, f"*gene_spectra_tpm.k_{row.K}.dt_{density}.txt")
                frame = pd.read_csv(path, sep="\t", index_col=0)
                frame.index = frame.index.map(lambda x: str(int(float(x))))
                spectra_by_fold[(int(row.K), int(row.fold))] = frame

            within_rows = []
            for k in K_GRID:
                folds = sorted(fold for kk, fold in spectra_by_fold if kk == k)
                for fold_a, fold_b in itertools.combinations(folds, 2):
                    spectra_a = spectra_by_fold[(k, fold_a)]
                    spectra_b = spectra_by_fold[(k, fold_b)]
                    common = sorted(set(spectra_a.columns) & set(spectra_b.columns))
                    a = spectra_a[common].to_numpy(float)
                    b = spectra_b[common].to_numpy(float)
                    cosine = a @ b.T / np.maximum(
                        np.outer(np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)), 1e-15)
                    row_index, column_index = linear_sum_assignment(-cosine)
                    for i, j in zip(row_index, column_index):
                        top_a = set(spectra_a.loc[spectra_a.index[i], common].nlargest(50).index)
                        top_b = set(spectra_b.loc[spectra_b.index[j], common].nlargest(50).index)
                        jaccard = len(top_a & top_b) / max(1, len(top_a | top_b))
                        within_rows.append({
                            "tier": RUN_TIER, "K": k, "fold_a": fold_a, "fold_b": fold_b,
                            "program_a": spectra_a.index[i], "program_b": spectra_b.index[j],
                            "cosine": cosine[i, j], "top50_jaccard": jaccard,
                            "operational_stability_pass": bool(cosine[i, j] >= 0.80 and jaccard >= 0.20),
                            "n_common_genes": len(common),
                        })
            within_matches = pd.DataFrame(within_rows)
            save_table(within_matches, f"Gray_{RUN_TIER}_within_K_fold_pair_matching")
            within_stability = (within_matches.groupby("K", observed=True)
                                .agg(n_fold_pairs=("fold_a", lambda x: len(set(zip(
                                         x, within_matches.loc[x.index, "fold_b"])))),
                                     n_component_matches=("cosine", "size"),
                                     median_cosine=("cosine", "median"),
                                     q1_cosine=("cosine", lambda x: x.quantile(0.25)),
                                     min_cosine=("cosine", "min"),
                                     median_top50_jaccard=("top50_jaccard", "median"),
                                     q1_top50_jaccard=("top50_jaccard", lambda x: x.quantile(0.25)),
                                     min_top50_jaccard=("top50_jaccard", "min"),
                                     cosine_ge_0_80_fraction=("cosine", lambda x: x.ge(0.80).mean()),
                                     joint_pass_fraction=("operational_stability_pass", "mean"))
                                .reset_index())
            complete_counts = (completed_folds.groupby("K", observed=True).size()
                               .rename("n_complete_folds").reset_index())
            within_stability = complete_counts.merge(within_stability, on="K", how="left")
            save_table(within_stability, f"Gray_{RUN_TIER}_within_K_stability")

            # P9/P10 are evaluated jointly so a stable cycling subspace is not
            # penalized merely because one K splits it and another K merges it.
            p9_p10_rows = []
            pair_labels = [label for label in ("9", "10") if label in reference_spectra.index]
            if len(pair_labels) != 2:
                raise ValueError("The frozen K10 reference must contain P9 and P10")
            for (k, fold), fold_spectra in sorted(spectra_by_fold.items()):
                common = sorted(set(reference_spectra.columns) & set(fold_spectra.columns))
                ref_pair = reference_spectra.loc[pair_labels, common].to_numpy(float)
                fitted = fold_spectra.loc[:, common].to_numpy(float)
                pair_cosine = ref_pair @ fitted.T / np.maximum(
                    np.outer(np.linalg.norm(ref_pair, axis=1), np.linalg.norm(fitted, axis=1)), 1e-15)
                pair_rows, pair_columns = linear_sum_assignment(-pair_cosine)
                selected_indices = sorted(set(pair_columns.tolist()))
                selected_labels = fold_spectra.index[selected_indices].astype(str).tolist()
                ref_unit = ref_pair / np.maximum(ref_pair.sum(axis=1, keepdims=True), 1e-15)
                fitted_selected = fitted[selected_indices]
                fitted_unit = fitted_selected / np.maximum(
                    fitted_selected.sum(axis=1, keepdims=True), 1e-15)
                ref_composite = ref_unit.sum(axis=0)
                fitted_composite = fitted_unit.sum(axis=0)
                composite_cosine = float(ref_composite @ fitted_composite / max(
                    np.linalg.norm(ref_composite) * np.linalg.norm(fitted_composite), 1e-15))
                ref_top = set().union(*[
                    set(reference_spectra.loc[label, common].nlargest(50).index) for label in pair_labels])
                fitted_top = set().union(*[
                    set(fold_spectra.loc[label, common].nlargest(50).index) for label in selected_labels])
                q_ref = np.linalg.qr(ref_pair.T, mode="reduced")[0]
                q_fitted = np.linalg.qr(fitted_selected.T, mode="reduced")[0]
                principal_cosines = np.linalg.svd(q_ref.T @ q_fitted, compute_uv=False)
                p9_p10_rows.append({
                    "tier": RUN_TIER, "K": k, "fold": fold,
                    "n_selected_fold_programs": len(selected_labels),
                    "selected_fold_programs": ";".join(selected_labels),
                    "composite_cosine": composite_cosine,
                    "composite_top50_union_jaccard": len(ref_top & fitted_top) / max(1, len(ref_top | fitted_top)),
                    "subspace_min_cosine": float(principal_cosines.min()),
                    "subspace_mean_cosine": float(principal_cosines.mean()),
                    "n_common_genes": len(common),
                })
            p9_p10 = pd.DataFrame(p9_p10_rows)
            save_table(p9_p10, f"Gray_{RUN_TIER}_P9_P10_composite_recovery")
            p9_p10_summary = (p9_p10.groupby("K", observed=True)
                              .agg(n_completed_folds=("fold", "nunique"),
                                   median_composite_cosine=("composite_cosine", "median"),
                                   min_composite_cosine=("composite_cosine", "min"),
                                   median_composite_top50_union_jaccard=(
                                       "composite_top50_union_jaccard", "median"),
                                   median_subspace_min_cosine=("subspace_min_cosine", "median"),
                                   min_subspace_min_cosine=("subspace_min_cosine", "min"))
                              .reset_index())
            save_table(p9_p10_summary, f"Gray_{RUN_TIER}_P9_P10_composite_summary")

            donor_error = (metrics.groupby(["K", "donor"], observed=True)
                           .agg(median_error=("median_scaled_HVG_relative_error", "median"),
                                median_signature_rho=("rho_training_signature", "median"))
                           .reset_index())
            k_evidence = (donor_error.groupby("K", observed=True)
                          .agg(n_donors=("donor", "nunique"),
                               median_heldout_error=("median_error", "median"),
                               median_signature_rho=("median_signature_rho", "median"))
                          .reset_index())
            reference_recovery = (recurrence.groupby("K", observed=True)
                         .agg(n_reference_programs_matched=("reference_program", "nunique"),
                              median_K10_reference_recovery=("recurrence_fraction", "median"),
                              median_K10_reference_program_cosine=("median_cosine", "median"),
                              median_K10_reference_program_jaccard=("median_top50_jaccard", "median"))
                         .reset_index())
            k_evidence = k_evidence.merge(reference_recovery, on="K", how="outer").sort_values("K")
            k_evidence = k_evidence.merge(
                within_stability[["K", "n_complete_folds", "median_cosine", "median_top50_jaccard",
                                  "cosine_ge_0_80_fraction", "joint_pass_fraction"]].rename(columns={
                    "median_cosine": "within_K_median_cosine",
                    "median_top50_jaccard": "within_K_median_top50_jaccard",
                    "cosine_ge_0_80_fraction": "within_K_cosine_ge_0_80_fraction",
                    "joint_pass_fraction": "within_K_joint_pass_fraction",
                }), on="K", how="outer")
            k_evidence = k_evidence.merge(
                p9_p10_summary[["K", "median_composite_cosine", "median_subspace_min_cosine"]],
                on="K", how="outer").sort_values("K")
            job_counts = (inventory.assign(
                n_jobs_complete=inventory.status.eq("complete").astype(int),
                n_jobs_failed_consensus=inventory.status.eq("failed_consensus_density_filter").astype(int))
                .groupby("K", observed=True)
                .agg(n_jobs_expected=("job", "size"),
                     n_jobs_complete=("n_jobs_complete", "sum"),
                     n_jobs_failed_consensus=("n_jobs_failed_consensus", "sum"))
                .reset_index())
            k_evidence = k_evidence.merge(job_counts, on="K", how="outer").sort_values("K")
            save_table(k_evidence, f"Gray_{RUN_TIER}_K_selection_evidence")

            minimum_error = float(k_evidence["median_heldout_error"].min())
            qualified = k_evidence.loc[
                k_evidence["median_heldout_error"].le(minimum_error + 0.002) &
                k_evidence["within_K_joint_pass_fraction"].ge(0.90) &
                k_evidence["n_jobs_complete"].eq(k_evidence["n_jobs_expected"])
            ].sort_values("K")
            primary_k = int(qualified.iloc[0].K) if len(qualified) else None
            sensitivity_k = 10 if primary_k != 10 and 10 in set(k_evidence.K) else None
            decision_record = {
                "tier": RUN_TIER, "primary_K_recommendation": primary_k,
                "sensitivity_K_recommendation": sensitivity_k,
                "status": "computational recommendation; external biological validation pending",
                "rule": {
                    "heldout_error_tolerance_above_best": 0.002,
                    "minimum_within_K_joint_pass_fraction": 0.90,
                    "require_all_registered_jobs_complete": True,
                    "select_smallest_qualified_K": True,
                },
                "caveat": (
                    "This transparent operational rule was added after inspecting the initial validation results; "
                    "it is not a preregistered confirmatory decision rule. K10-reference recovery is diagnostic "
                    "only and is excluded from selection."
                ),
            }
            atomic_json(TABLE_DIR / f"Gray_{RUN_TIER}_K_decision_record.json", decision_record)

            fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.8))
            axes[0].plot(k_evidence.K, k_evidence.median_heldout_error, marker="o", color="#0072B2")
            axes[0].set(xlabel="K", ylabel="Median held-out relative error", title="Reconstruction")
            axes[1].plot(k_evidence.K, k_evidence.median_signature_rho, marker="o", color="#009E73")
            axes[1].set(xlabel="K", ylabel="Median donor Spearman ρ", title="Training-signature agreement")
            axes[2].plot(k_evidence.K, k_evidence.within_K_joint_pass_fraction, marker="o", color="#D55E00")
            axes[2].set(xlabel="K", ylabel="Matched components passing screen",
                        title="Within-K fold stability", ylim=(-0.02, 1.02))
            for ax in axes: ax.set_xticks(K_GRID)
            save_figure(fig, f"Gray_{RUN_TIER}_K_evidence")

            pivot = recurrence.pivot(index="reference_program", columns="K", values="recurrence_fraction")
            pivot = pivot.reindex(index=sorted(pivot.index, key=int), columns=K_GRID)
            fig, ax = plt.subplots(figsize=(6.8, 5.2))
            image = ax.imshow(pivot.to_numpy(float), vmin=0, vmax=1, cmap="viridis", aspect="auto")
            ax.set(xticks=np.arange(len(pivot.columns)), xticklabels=pivot.columns,
                   yticks=np.arange(len(pivot.index)), yticklabels=[f"P{x}" for x in pivot.index],
                   xlabel="K", ylabel="Full-Gray K=10 reference program",
                   title="K=10-reference recovery (diagnostic)")
            colorbar = fig.colorbar(image, ax=ax); colorbar.set_label("Fraction of completed folds passing screen")
            save_figure(fig, f"Gray_{RUN_TIER}_program_recurrence")

            fig, ax = plt.subplots(figsize=(6.4, 4.2))
            for k, part in p9_p10.groupby("K", observed=True, sort=True):
                ax.scatter(np.full(len(part), k), part["composite_cosine"], color="#56B4E9", s=28)
                ax.plot([k - 0.22, k + 0.22], [part["composite_cosine"].median()] * 2,
                        color="black", linewidth=1.2)
            ax.set(xticks=K_GRID, xlabel="K", ylabel="P9/P10 composite cosine",
                   title="P9/P10 split–merge recovery", ylim=(-0.02, 1.02))
            save_figure(fig, f"Gray_{RUN_TIER}_P9_P10_composite_recovery")
            print("K decision:", decision_record)
            display(k_evidence)
        """),
        md(r"""
        ## 7 · Cross-K interpretation and decision boundary

        Read outputs in this order:

        1. `GSE294268_eligibility.csv`: feature coverage and whether raw UMI counts
           permit exact scoring. SCT-only data must remain blocked.
        2. `Gray_*_checkpoint_inventory.csv`: every expected `fold × K` unit must be
           complete before a definitive K claim.
        3. `Gray_*_within_K_stability.csv`: K-neutral fold-to-fold consensus stability.
        4. `Gray_*_K_selection_evidence.csv`: donor-level reconstruction,
           training-signature agreement and within-K stability; no cell-level
           pseudoreplication.
        5. `Gray_*_cross_K_program_recurrence.csv`: recovery of each full-data K=10
           reference program. Unassigned programs count as failures. Because the
           reference is K=10, this table is diagnostic and must not select K.
        6. `Gray_*_P9_P10_composite_summary.csv`: split–merge-aware evidence for the
           combined P9/P10 cycling signal.

        Do not select K from reconstruction error alone. Do not select K using an
        external dataset and then reuse the same dataset as validation. The written
        K decision is a transparent post-hoc computational recommendation, not a
        preregistered confirmatory rule. If K=10 adds no stable program beyond a
        smaller K, report the stable programs and keep K=10 as sensitivity rather
        than asserting ten definitive states.
        """),
        md(r"""
        ## Additional validation: implementation status and evidence boundary

        The helpers below run only when `GDT_19B_STAGE=robustness_validation`.
        Successful implementation tests alone are **not completed biological
        validation**. They do not alter the frozen K8 basis, q90/q95
        thresholds, original preanalysis manifest, or historical projection outputs.

        - Audit gene IDs against the Gray training annotation before calling a gene
          absent. Use one destination per raw feature, retain all symbol/ID conflicts,
          and report per-program top-gene and weighted-spectrum coverage.
        - Evaluate raw-UMI thinning and missing-feature sensitivity within held-out
          Gray donors. Report compositional usage changes, raw coefficients,
          reconstruction and numerical convergence. Fold-model spread describes
          sensitivity; it is not a confidence interval from independent models.
        - Compare the existing rank baseline and AUCell (decoupler 2.1.2) with cNMF
          on identical donor-held-out splits. Train normalization and any endpoint
          predictor only on training donors. AUCell is a reimplementation by
          decoupler, not the original R AUCell package or UCell.
        - Apply explicit single-productive-TRG/TRD eligibility for the corresponding
          TCR endpoint and exclude zero-total-ADT cells from protein endpoints.
          Report these as implementation corrections/sensitivity analyses after
          the original external evaluation, not a second untouched validation.

        ### Paired functional study: draft, not an executed experiment

        Pair each donor's RNA measurement with a matched aliquot for CD107a and
        target-cell killing. Preserve donor, Vdelta subset, stimulation, culture
        time, target identity, effector-to-target ratio and assay batch in the join
        key. Keep unstimulated, stimulated and target co-culture conditions paired;
        do not infer a donor link from matching group names or sample counts.

        A proposed primary endpoint is donor-level baseline frozen P2 usage versus
        target-specific killing under one predeclared condition and time point.
        CD107a and cytokines are secondary functional endpoints. Before collection,
        agree on the target, assay conditions, viable-effector normalization, sample
        exclusions, minimum cell counts and clinically/biologically meaningful
        effect with the experimental collaborator. Sample size requires a pilot
        variance estimate and prospective power simulation; no donor count is
        claimed sufficient here. Randomize processing across batches and blind
        endpoint analysis where feasible. Technical wells are not independent donors.

        For incremental value, compare covariates, covariates + rank signature,
        covariates + AUCell, covariates + cNMF, and covariates + signature + cNMF
        with identical outer-donor splits. Use donor-level prediction error and
        paired donor-resampled uncertainty, with any tuning nested inside training
        donors. Small public cohorts support feasibility, not reliable calibration
        or a definitive generalization claim.

        To investigate mechanism, select a regulator only after independent evidence
        links it to the program. A perturbation should alter the prespecified state
        and functional endpoint while accounting for viability and proliferation;
        association alone does not identify a causal regulator. No gene perturbation
        has been performed by this notebook.

        Public follow-up source: DOI `10.3389/fimmu.2026.1722590`, Zenodo record
        `17590095`, source filenames `seurat_obj.rds` and `code_final.Rmd`.
        A Seurat object and analysis code do not themselves establish donor-linked
        functional measurements. Inspect the object and obtain a donor/condition
        crosswalk plus underlying assay values before paired functional inference.

        **Object audit, 2026-09-19:** The checksum-verified released object contains
        7,363 cells from donorA/donorB/donorC, a 36,485-feature RNA counts assay and
        a 139-feature `citeseq` counts assay. Both assays contain nonnegative integer
        values and identical ordered cell IDs; no cell has zero total `citeseq`
        counts. Features include `CD69-CITESEQ`, `CD137-CITESEQ`, `CD107a-CITESEQ`
        and `TCRVd2-CITESEQ`. This is a promising paired-protein candidate, but the
        published methods explicitly describe RNA/TCR and hashtag multiplexing,
        not the provenance of this 139-antibody panel. Do not equate an assay name
        with verified experimental measurement. Confirm measured-versus-imputed
        origin, panel, staining conditions and sample linkage with the authors.
        The article reports 8,249 cells in Fig. 3B, whereas the released object has
        7,363; reconcile this cohort difference before confirmatory interpretation.
        CD107a counts are not automatically the separate target-coculture
        degranulation endpoint. Only three independent donors are available.

        Verified source MD5: `seurat_obj.rds` =
        `9a9bdf00eb10595f20d7eacfbd9ce27d`; `code_final.Rmd` =
        `c71b315fdb056b91796b978022fef461`. R 4.6.0 read the object successfully;
        Python rdata 0.11.2 could not parse its R serialization (RObjectType 249).

        **Author-request draft (not sent):** We are evaluating frozen gamma-delta
        T-cell transcriptional programs against functional endpoints. Could you
        share de-identified donor-linked killing and CD107a measurements, with the
        mapping to scRNA-seq donors A/B/C where applicable? We would need culture
        condition/time, Vdelta subset, target line, effector-to-target ratio, batch,
        technical replicate and viability information, and clarification of which
        donors overlap across assays. Raw RNA counts/feature identifiers and flow
        gating/FCS metadata would help ensure a reproducible comparison. We would
        respect the applicable consent, access conditions and attribution requirements.
        Please also clarify how the released `citeseq` assay was generated, whether
        its values are experimental ADT counts, which antibody panel and stimulation
        conditions apply, and why the released object has 7,363 rather than the
        8,249 cells reported in Fig. 3B.
        """),
        code(VALIDATION_HELPERS),
        md(r"""
        ### Run and interpret `robustness_validation`

        The run uses the existing five donor-held-out Gray K8 models. Each donor,
        perturbation and repeat is saved independently. External projection uses
        every released cell in 14 GSE268991 samples and 13 GSE294268 libraries,
        with gene-ID alignment, identical cNMF/rank/AUCell features and saved
        original-versus-corrected reconstruction/gate comparisons.

        All results are post-evaluation sensitivity analyses. The full-data
        reference names fold programs but does not fit their held-out cells.
        Observed-feature fits are experimental; their errors cannot use the
        original q95 gate. Fold-model ranges are not confidence intervals.

        The GSE268991 benchmark predicts the two released CD45RA/CCR7 surface
        protein readouts in cells with exactly one productive TRG and TRD and
        nonzero ADT counts. Target centering/scaling is learned only in training
        donors; the mean squared error is averaged over endpoints, then within
        each held-out donor. This is phenotype validation, not a killing assay.

        The GSE294268 benchmark first aggregates NonA cells to donor × Experiment
        × stimulus. It requires at least 20 cells per group and all three stimuli
        for each included donor, reported separately before/after the frozen gate.
        It predicts CR/MO/HDMAPP, NOT allergy or stimulation versus unstimulated.
        Both leave-one-donor-out and leave-one-Experiment-out results are exported.
        Covariates comprise log library size and condition (protein benchmark) or
        Experiment (stimulation benchmark). Fixed regularization is used without
        tuning on the external results. Bootstrap intervals resample paired
        held-out donor errors conditional on the fitted models; they do not
        capture all training uncertainty or shared-Experiment dependence.

        Outputs use the approved `Gray_primary_K8_robustness_validation_…`,
        `GSE268991_primary_K8_robustness_validation_…` and
        `GSE294268_primary_K8_robustness_validation_…` prefixes. Checkpoints are
        stored under `results/models/19b_cNMF_valdiation/checkpoints/robustness_validation/`.
        A checkpoint is reused only after its configuration, input hashes, expected
        files, checksums, required columns and unique row index are verified.
        The progress manifest remains `running` until all required loops finish.

        Resume from the same project and environment:

        ```bash
        cd /data/users/kaelynwen/gdT_project/gdt_lr_12model
        GDT_19B_STAGE=robustness_validation GDT_19B_THREADS=2 NUMBA_NUM_THREADS=2 \
          OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
          /data/users/kaelynwen/conda_envs/cnmf_env/bin/python -m jupyter nbconvert \
          --execute --to notebook --inplace --ExecutePreprocessor.timeout=-1 \
          --ExecutePreprocessor.kernel_name=python3 notebooks/19b_cNMF_valdiation.ipynb
        ```

        A code/configuration change creates a different run signature; do not
        delete old checkpoints to force reuse. No cNMF training is repeated.
        """),
        code(ROBUSTNESS_CODE),
        md(r"""
        ## Depth calibration: technical reliability, not functional prediction

        Run with `GDT_19B_STAGE=depth_calibration`. This reuses the completed
        robustness checkpoints; it does not refit K8. Each existing basis has
        4–5 unseen donors. One is the reliability-test donor; the remaining 3–4
        learn a fixed-regularization 0.95 quantile function of log depth. Donor,
        depth and repeated-thinning weights prevent technical repeats from
        dominating the fit. Test-donor data do not set features, weights or bounds.

        Compare the unchanged original q95, an equally weighted calibration-donor
        constant q95 and the depth-dependent q95 on identical supported observations.
        Original q95 is a historical full-Gray threshold, not an independent
        held-out threshold; the calibration-donor constant is the fair comparator
        for isolating the additional value of depth conditioning.
        Report quantile loss as well as rejection: accepting everything is not
        success. Unsupported/invalid observations are reported separately, never
        silently accepted. A bound near 1 is uninformative and its frequency is saved.

        Program targets are absolute changes from the same cell's full-depth
        reference, not true biological error. Their models use thinned data only.
        These empirical bounds do not have a distribution-free donor-level 95%
        guarantee. Fold-specific calibration is NOT automatically transferable to
        the full-Gray K8 scorer or an external cohort. Existing external outcomes
        are not used to select or tune these models.

        Checkpoints are atomic donor × target units (22 × 9 = 198), with source,
        input/configuration hashes, donor identities, model parameters and predictions.
        A stopped run retains all validated completed units. A smoke run with
        `GDT_19B_DEPTH_SMOKE=1` processes only the first donor and is explicitly
        marked incomplete. Resume the full run using the command below:

        ```bash
        cd /data/users/kaelynwen/gdT_project/gdt_lr_12model
        GDT_19B_STAGE=depth_calibration GDT_19B_THREADS=2 \
          OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
          /data/users/kaelynwen/conda_envs/cnmf_env/bin/python -m jupyter nbconvert \
          --execute --to notebook --inplace --ExecutePreprocessor.timeout=-1 \
          --ExecutePreprocessor.kernel_name=python3 notebooks/19b_cNMF_valdiation.ipynb
        ```

        The two figures address different questions: does a depth-conditioned
        reconstruction reference improve technical rejection behavior, and do
        individual programs have adequately calibrated technical-error bounds?
        Lines/points summarize donors; shaded bands/segments show donor IQR, not
        confidence intervals. Full numerical results remain in the donor tables.
        Functional-assay design and the unsent author-request draft above remain
        separate from this technical calibration experiment.
        """),
        code(DEPTH_CALIBRATION_HELPERS),
        code(DEPTH_CALIBRATION_CODE),
        md(r"""
        ## P7 failure diagnosis and frozen RNA time course

        These two approved stages extend, rather than repeat, the existing spectra
        similarity, ablation, fold stability and P9/P10 composite analyses.
        `p7_diagnostics` reuses verified checkpoints for all 22 Gray donors. It
        distinguishes absolute usage distortion from exceedance of the previously
        fitted technical-error bounds. Calibration usage summaries give donors
        equal weight. Full-depth usage strata are retrospective diagnostics, not
        prospective confidence estimates. No threshold or model is refitted.

        `serum_time_course` uses the released Zenodo 17590095 RNA counts only,
        projected with the frozen full-data K8 basis. Exact source donor/condition
        names are retained. D0 is a single baseline, not two serum observations.
        DonorA D0 was processed separately; the release contains 7,363 cells rather
        than the paper's 8,249. Results describe this released object.

        Each of 21 donor-condition samples has one native and six perturbed runs:
        50% UMI thinning (three seeds), and expected 1,000-UMI depth matching (three
        seeds; restricted to native cells with >=1,000 training-universe UMIs).
        The latter is binomial thinning, not exact-depth rarefaction. Compare to
        native scores on those same cells. Export all valid/converged scores as
        primary and the original Gray q95 as sensitivity, never as externally
        calibrated reliability. Gray fold calibrators are not transferred.

        No `citeseq` measurements or protein-derived cluster labels are used.
        `RNA_snn_res.0.8` strata are expression-derived exploratory strata: their
        composition decomposition is arithmetic, not causal or orthogonal
        evidence. Restrict decomposition to strata with >=10 valid cells per
        condition, renormalize their proportions and disclose retained fractions.
        Three donors support descriptive paired effects, not cell-level p-values.
        Technical replicates do not create additional biological replication.

        ### Reproducible input conversion and execution

        On a host with base R and Matrix (the local Mac is available), run:
        `python3 build_19b_cNMF_valdiation.py --export-serum-rna`.
        The command verifies the published source MD5 and exports RNA only into
        `results/models/19b_cNMF_valdiation/Zenodo17590095_primary_K8_time_course_input/`.
        Copy that directory unchanged to the corresponding Woody project path.
        No R package installation or Seurat re-analysis is needed on Woody.

        Generate this notebook on Woody and run it with `GDT_19B_STAGE=p7_diagnostics`
        or `GDT_19B_STAGE=serum_time_course`, `GDT_19B_THREADS=2`, and the existing
        cnmf_env Python/kernel. The standard nbconvert execute command applies.
        Restart only after checking no duplicate process is running. Each stage
        resumes only matching, checksum-verified source/config/input units.
        Completion requires 22 or 147 units, respectively. Final tables include
        donor metrics, original/gated denominators and technical sensitivity.
        Figures are single-panel vector PDFs with donor/sample-specific values,
        not inferential confidence intervals. See the source paper:
        https://doi.org/10.3389/fimmu.2026.1722590.

        Kaelyn has contacted the authors. Functional donor linkage and the
        provenance of the released `citeseq` assay remain unresolved; no function
        measurement is inferred from the RNA time course.
        """),
        code(FOLLOWUP_HELPERS),
        code(P7_DIAGNOSTICS_CODE),
        code(SERUM_TIME_COURSE_CODE),
        md(r"""
        ## 10 · Incremental value beyond learned and independent signatures

        **Question.** Does frozen K8 add predictive information beyond experimental
        covariates, its own top-50 gene signatures, and all 50 independent MSigDB
        Hallmark 2025.1 gene sets? This is a development comparison on previously
        examined cohorts, not another untouched final validation.

        The original rank score is custom, not UCell. AUCell uses decoupler 2.1.2.
        Independent Hallmark gene membership is never selected from the test
        outcomes. A fixed full-measured gene intersection uses only released
        feature annotations across the 27 libraries. Compare Hallmark and top-50
        AUCell on this universe, and retain the frozen-universe versions as
        sensitivity analyses. Coverage is reported for every library and set.

        GSE268991 uses measured CD45RA/CCR7 ADT CLR with seven held-out donors.
        GSE294268 uses donor x Experiment x stimulus medians, at least 20 cells
        per condition, 15 NonA donors with CR/MO/HDMAPP, and four Experiments.
        Both all_cells and Gray_q95_accepted are reported; q95 is not retuned.
        These endpoints do not measure killing ability.

        Every decoder receives the same inner group splits and candidate budget:
        C or alpha in (0.01, 0.1, 1, 10, 100). Inner fits never contain outer-test
        donors. Scaling is fitted on training cells only; model fitting and loss
        summaries balance donors. Inner Experiment losses balance Experiments.
        Test predictions, scaling parameters, all inner losses and chosen
        parameters are retained. No K8 spectra or score thresholds are refitted.

        **Restartability.** The new stage expects 27 scoring units plus 624
        outer-group x method units. Each completed unit has an atomic marker,
        exact source/config/input signatures and output checksums. Interruptions
        preserve finished units. A changed configuration creates a separate run;
        corrupt checkpoints stop rather than being silently accepted.

        Run from the Woody project directory after checking for duplicate jobs:
        ```bash
        GDT_19B_STAGE=incremental_value_validation GDT_19B_THREADS=2 \
        NUMBA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
        /data/users/kaelynwen/conda_envs/cnmf_env/bin/python -m jupyter nbconvert \
          --execute --to notebook --inplace --ExecutePreprocessor.timeout=-1 \
          --ExecutePreprocessor.kernel_name=gdt_cnmf notebooks/19b_cNMF_valdiation.ipynb
        ```
        The same command resumes only validated units. The authoritative source
        remains `build_19b_cNMF_valdiation.py`; this notebook is generated from it.

        **Interpretation.** Positive paired error reduction favours the named
        model. Report all donor effects and, for Experiment transportability,
        all four batch effects. Conditional bootstrap intervals do not include
        refitting uncertainty and are exploratory, not multiplicity-adjusted
        superiority tests. Failure to establish superiority is not equivalence.
        Favourable stimulation discrimination cannot establish functional or
        mechanistic claims. Full gene-universe and q95 sensitivity results must
        not be selected after inspection to manufacture a winning comparison.
        """),
        code(INCREMENTAL_HELPERS),
        code(INCREMENTAL_CODE),
        md(r"""
        ## 11 · Raw versus normalized usage: development diagnostic

        Raw usages were already computed and used in UMI-thinning diagnostics.
        The previous downstream benchmarks used normalized usages. This stage
        tests the missing representation comparison, without refitting K8,
        changing q95, recomputing AUCell, or modifying historical results.

        Read both representations from validated checkpoints. Reproduce all 104
        relevant historical outer predictions at their saved penalties before
        fitting raw predictors. Use the same seven protein donors, fifteen NonA
        stimulation donors, four Experiments, exact observations and inner grids.
        Raw coefficients receive the existing donor-weighted, training-only
        StandardScaler. There is no search over log transforms. The two method
        families are covariates + cNMF and covariates + full-measured Hallmark
        AUCell + cNMF. `representation` distinguishes `usage_raw` from
        `usage_normalized`; neither is an absolute RNA measurement.

        Each of the 104 outer-group x method fits has an atomic checkpoint with
        predictions, donor losses, inner tuning losses and training scalers.
        Verified completed units are reused; interrupted units are recomputed.
        Source/configuration changes create a new run signature. No aggregate
        is declared complete before every required unit exists.

        The same stage diagnoses the previous GD005 result by stimulus and donor,
        distinguishing log-loss improvement from corrected class labels. q95
        retention and accepted/rejected profiles are descriptive: exclusion does
        not establish that a cell is biologically invalid. No new biological
        mechanism, killing prediction or untouched-validation claim is made.

        **Run or resume** from `/data/users/kaelynwen/gdT_project/gdt_lr_12model`
        after checking that another instance is not running:
        ```bash
        GDT_19B_STAGE=usage_representation_validation GDT_19B_THREADS=2 \
        NUMBA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
        /data/users/kaelynwen/conda_envs/cnmf_env/bin/python -m jupyter nbconvert \
          --execute --to notebook --inplace --ExecutePreprocessor.timeout=-1 \
          --ExecutePreprocessor.kernel_name=gdt_cnmf notebooks/19b_cNMF_valdiation.ipynb
        ```
        Positive paired loss reductions favour raw usage. Show all donors and
        all Experiments. Conditional bootstrap intervals resample fixed-model
        losses and are exploratory, without refitting or multiplicity adjustment.
        Better reconstruction or one favourable subgroup is not sufficient to
        select a deployment method. Both gates must remain visible.
        """),
        code(USAGE_REPRESENTATION_HELPERS),
        code(USAGE_REPRESENTATION_CODE),
        md(r"""
        ## Evidence synthesis: program reliability, pipeline audit and serum application

        **Question:** What does each frozen K8 program support, under which conditions,
        and has the complete lineage/subtype/state pipeline actually been evaluated?
        This stage reads existing evidence only. It does not retrain, tune thresholds,
        replace deployment methods or convert historical K10 findings into K8 validation.

        The eight program cards retain the established K10-reference labels P1, P2,
        P3, P4, P5, P7, P9 and P10. Their K8 component IDs are separate. Fold-to-full
        matches share discovery donors; cross-K recurrence uses a K10 reference and
        is not an independent K-selection criterion. Nearest-spectrum similarity is
        descriptive, not proof that usages are non-identifiable. Technical Gray
        error bounds are not calibrated external biological confidence intervals.
        A q95 reconstruction pass does not establish 95% reliability of each program.

        The end-to-end audit inspects saved prediction schemas and exact cell overlap.
        Silver annotations, missing receptor calls and a file named "state" do not
        establish independent truth or complete state coverage. Subtype uncertainty
        must remain distinct from state QC. No first-layer model is silently selected.

        The serum application retains all three donors, both existing gate choices,
        technical depth sensitivities and shared-RNA-stratum decomposition. Paired
        median contrasts and shared-stratum mean decompositions are different
        estimands. Three technical thinnings are not three independent donors.
        RNA and functional donors were confirmed different in the author email;
        citeseq provenance remains unresolved. No killing, causal-factor or
        population-significance claim follows from this application.

        **Run:** use `GDT_19B_STAGE=evidence_synthesis` with the existing gdt_cnmf
        kernel. Each program card and each synthesis section is checkpointed with
        source/input/configuration hashes. The final manifest requires all 11 units.
        Existing evidence and model artifacts remain unchanged. The final cells show
        the program summaries, pipeline gaps and donor-based serum summaries.
        """),
        code(EVIDENCE_SYNTHESIS_HELPERS),
        code(EVIDENCE_SYNTHESIS_CODE),
        md(r"""
        ## End-to-end validation: frozen lineage selection followed by K8 state scoring

        **Question:** How do the saved 22_LR_hierarchical_CE lineage calls change
        program distributions relative to reference populations, and can state
        output remain available when subtype predictions are unavailable?

        This stage freezes the existing lineage release, K8 basis, normalization
        and q95. It scores the union of predicted gamma-delta cells, silver gamma-delta
        annotations and barcode-linked positive/negative receptor controls. Numerical
        failures remain missing; no confident subtype is fabricated from atlas labels.
        Series checkpoints validate input/model/cohort hashes and are safely resumable.

        Primary PANGEA excludes GSE149356, as before. GSE149356 is retained only as
        previously examined sensitivity with processed TRG+TRD/CDR3 support lacking
        productive/full-length/QC flags. Strict paired alpha-beta controls and unknown
        receptor evidence are never conflated. Comparison units are study/sample proxies,
        not automatically biological donors. Reference cells missed by layer 1 remain
        in the scoring population. Empty comparison arms are missing, not zero effects.

        **Interpretation:** prediction-versus-silver differences describe selection,
        not proven contamination. Removal of known strict alpha-beta controls is only
        a limited diagnostic; other predictions have unknown truth. This stage tests
        subtype-independent output routing. The follow-up below links the approved
        frozen second-layer predictions for adult GSE149356 cells only; it does not
        establish subtype accuracy across all PANGEA. Neither part establishes
        killing/function or causal biology.
        Existing serum/time-course results and their limitations remain unchanged.

        **Run:** `GDT_19B_STAGE=end_to_end_validation` using the gdt_cnmf kernel.
        Per-Series checkpoints preserve complete scoring units across interruptions.
        The final manifest lists input hashes, software versions, exact population,
        completed units, outputs and remaining evidence gaps. Tables and figures below
        distinguish primary silver-label results from previously used receptor sensitivity.
        """),
        code(END_TO_END_HELPERS),
        code(END_TO_END_CODE),
        md(r"""
        ### GSE149356 follow-up: missed reference cells and real subtype routing

        **Questions:** Are receptor-supported cells missed by layer 1 concentrated
        in particular program states? Does selection alter the adult peripheral-blood
        versus neonatal cord-blood contrast? Does a genuinely uncertain layer-2
        prediction incorrectly remove an otherwise available state score?

        Reuse the completed base run only after verifying every recorded input,
        output and immutable source snapshot. An incomplete or changed base run
        fails closed and requires inspection; it is not silently overwritten.
        Join the saved `14_TCRfree_Vd1_Vd2_subtype_scVI_LR` predictions (primary
        seed `20260829`) using sample and exact barcode identity. The user-approved
        revised probability policy is: at most 0.30 gives Vd2,
        at least 0.70 gives Vd1, and intermediate values remain
        `other_or_unresolved`. CB1/CB2 have no predictions in this approved
        release and remain unavailable. These probabilities are conditional on
        gamma-delta identity, not an additional lineage classifier.

        This policy was chosen after inspection of GSE149356. Its evaluation
        here is developmental, not an untouched external threshold test.
        Before replacing published follow-up outputs, preserve the completed
        previous policy's outputs and manifest under its existing run-id
        checkpoint directory. The original subtype model release, calibrated
        probabilities, first-layer selection and K8 scores remain unchanged.

        Expression-only and processed-sequence-assisted subtype assignments are
        separate columns. Sequence evidence can aid interpretation but cannot
        count as success of the expression classifier. A missed reference cell
        remains in the diagnostic tables, not the deployed positive population.
        State availability depends on layer-1 selection and state-score validity,
        not a confident subtype call. q95 is a reconstruction gate, not proof of
        95% per-program accuracy, biological validity or lineage correctness.

        **Statistics:** Primary summaries use all valid cells; original Gray q95
        is a sensitivity analysis. Report every program, sample, TRDV stratum and
        arm size. Empty arms remain missing. Group contrasts use equal-donor means
        of sample medians: HC1/HC2 minus CB1/CB2. SOFT metadata verify the groups;
        with two donors per group and age confounded with blood source, these are
        descriptive contrasts, not tests of a pure age effect or equivalence.

        **Restart and outputs:** The same `end_to_end_validation` command also
        runs this follow-up. Each of four samples has a hash-validated atomic
        checkpoint. The separate `followup_manifest` records input provenance,
        completed units, source CSVs, figure legends and vector-PDF hashes.
        Existing model parameters, base outputs and serum/time-course are retained.
        This audit evaluates observed-state scoring and pipeline selection;
        it does not predict an unobserved post-perturbation state.
        """),
        code(END_TO_END_FOLLOWUP_HELPERS),
        code(END_TO_END_FOLLOWUP_CODE),
        md("""
        ### Full input denominator and three-layer output accounting

        Reconcile every cell in the saved first-layer PANGEA population, including
        cells not selected or state-scored. Separate standalone subtype agreement
        on sequence-supported binary references from correct subtype output per
        **all original binary reference cells**. Missing models, lineage misses,
        uncertain calls and incorrect calls remain visible in mutually exclusive
        counts. An available state score is not a verified functional prediction.

        `cell_flow` records study/sample counts and pooled totals;
        `performance_summary` records numerator, denominator, value and evidence
        restrictions; `model_contract` records frozen versions, provenance,
        input/output requirements, completion and software-release limitations.
        Receptor-positive recall and receptor-negative false-positive rates are
        separate. Do not pool classes from different studies into an independent
        accuracy claim. Annotation-only silver comparisons remain separate.
        Empty denominators produce missing values, never 100% or artificial zero.

        Each study is an atomic, hash-validated restart unit. All units must be
        complete before final aggregation. Sample IDs are not inferred donors.
        This is a reproducible analysis workflow, not a released inference package;
        packaging and independent new-donor evaluation remain separate tasks.
        """),
        code(END_TO_END_SUMMARY_HELPERS),
        code(END_TO_END_SUMMARY_CODE),
        md(r"""
        ## Perturbation-response prediction: a three-donor development pilot

        **Question:** Can a donor's D0 profile and a known culture condition predict
        its later mean frozen-K8 usages without seeing that donor's future RNA?
        This is a new prediction task, not a refit of cNMF and not functional truth.

        **Design:** The existing Zenodo 17590095 time course contains donorA,
        donorB and donorC, each at D0 and D7/D10/D14 WITH_SERUM/WO_SERUM.
        For each held-out donor, fit only on the other two donors. The prediction
        helper receives training outcomes, held-out D0 and the six-condition
        design; a separate evaluator receives held-out future outcomes.
        Cells, time points, technical thinning replicates and the three overlapping
        training folds do not increase the number of independent donors beyond 3.

        **Target and methods:** Use mean normalized usage for all eight frozen
        programs, so the target profile sums to one. This differs from the existing
        descriptive median analyses, which remain unchanged. Compare (1) no change
        from D0, (2) the other donors' mean change at the same day/serum setting,
        and (3) ridge regression of change on the same program's D0 mean, day/14,
        serum indicator and their time-by-serum interaction. Ridge alpha is fixed
        at 1 with an intercept and SVD solver; no tuning, selection, test-fitted
        scaling or deep-learning model is used. With only two training donors,
        personalized baseline-response coefficients are weakly supported.

        All three methods apply the same Euclidean simplex projection to enforce
        nonnegative predicted usages summing to one. Save pre-projection predictions,
        projection magnitudes and fitted coefficients. A decrease in normalized
        usage is relative, not proof of absolute pathway suppression or improvement.

        **Primary evaluation:** All-valid native RNA scores. Each donor contributes
        equally; within donor, each of six conditions and eight programs contributes
        equally to MAE. Also report RMSE, program-specific errors and descriptive
        exact direction agreement; near-zero changes are not meaningful evidence
        of biological directional accuracy. Historical q95 is a separate, fixed
        sensitivity pipeline, not a rule chosen using the future test responses.
        Its evaluation population depends on observed post-treatment quality and
        cannot be promised as a prospectively known subset. Preserve counts and
        low-retention samples, including donorA D14. Existing depth/composition
        results remain unchanged and do not establish response causality.

        **Reproducibility:** Verify the historical source, all parent outputs,
        exported RNA inputs and frozen-model hashes. Eighteen checkpoints represent
        donor x method x gate; each is written atomically and reused only after
        signature, hash and schema checks. Only after all units are present are
        final CSVs, two vector PDFs, PNG previews and a manifest assembled.
        Run with `GDT_19B_STAGE=perturbation_response_prediction` and gdt_cnmf.

        **Limits:** Previously examined development data; no untouched final test,
        unseen perturbation, independent killing/protein endpoint, same-cell
        longitudinal tracking, causal serum factor, calibrated prediction interval
        or automatic model-selection claim. Predicting cNMF-derived targets better
        does not itself establish superiority over signatures on independent biology.
        See scGen (doi:10.1038/s41592-019-0494-8), CPA
        (doi:10.15252/msb.202211517) and PerturBench (NeurIPS 2025) for distinct
        response-prediction approaches and the importance of baseline comparisons.
        """),
        code(PERTURBATION_PREDICTION_HELPERS),
        code(PERTURBATION_PREDICTION_CODE),
        code(r"""
        # 10. Reproducibility manifest and compact handoff
        package_names = [
            "anndata", "matplotlib", "numpy", "openpyxl", "pandas", "scipy", "scikit-learn"
        ]
        versions = {}
        for package in package_names:
            try: versions[package] = im.version(package)
            except im.PackageNotFoundError: versions[package] = None
        try: versions["cnmf"] = im.version("cnmf")
        except im.PackageNotFoundError: versions["cnmf"] = None
        manifest = {
            "schema_version": 1, "implementation_revision": IMPLEMENTATION_REVISION,
            "orchestration_revision": ORCHESTRATION_REVISION,
            "analysis": ANALYSIS_NAME, "timestamp": datetime.now(timezone.utc).isoformat(),
            "stage": EXECUTION_STAGE, "tier": RUN_TIER, "configuration": {
                "K_grid": K_GRID, "n_folds": N_FOLDS, "screen_starts": SCREEN_STARTS,
                "full_starts": FULL_STARTS, "density_threshold": DENSITY_THRESHOLD,
                "n_hvg": N_HVG, "seed": SEED, "minimum_HVG_coverage": MIN_HVG_COVERAGE,
            }, "sources": SOURCES, "status": STATUS, "versions": versions,
            "platform": platform.platform(), "python": sys.version,
            "evidence_boundary": "cNMF cell-state usage only; not lineage or Vd subtype prediction",
        }
        atomic_json(TABLE_DIR / "run_manifest.json", manifest)
        handoff = TABLE_DIR / f"{ANALYSIS_NAME}_{EXECUTION_STAGE}_handoff.zip"
        with zipfile.ZipFile(handoff, "w", zipfile.ZIP_DEFLATED) as archive:
            for root in (TABLE_DIR, FIGURE_DIR, LOG_DIR):
                for path in sorted(root.rglob("*")):
                    if path.is_file() and not path.name.endswith("_handoff.zip"):
                        archive.write(path, path.relative_to(PROJECT / "results"))
        print("Tables:", TABLE_DIR)
        print("Figures:", FIGURE_DIR)
        print("Models/checkpoints:", MODEL_DIR)
        print("Logs:", LOG_DIR)
        print("Handoff:", handoff)
        """),
        md(r"""
        ## Woody commands

        Generate the notebook from the authoritative source:

        ```bash
        cd /data/users/kaelynwen/gdT_project/gdt_lr_12model
        python build_19b_cNMF_valdiation.py
        ```

        Run the safe eligibility audit:

        ```bash
        GDT_19B_STAGE=input_qc jupyter nbconvert --execute --to notebook --inplace \
          --ExecutePreprocessor.timeout=-1 notebooks/19b_cNMF_valdiation.ipynb
        ```

        Run or resume one registered full job; replace only the fold/K selector:

        ```bash
        GDT_19B_STAGE=gray_full GDT_19B_ONLY_JOB=fold0_K5 \
          jupyter nbconvert --execute --to notebook --inplace \
          --ExecutePreprocessor.timeout=-1 notebooks/19b_cNMF_valdiation.ipynb
        ```

        After all 25 full jobs complete:

        ```bash
        GDT_19B_STAGE=assemble_full jupyter nbconvert --execute --to notebook --inplace \
          --ExecutePreprocessor.timeout=-1 notebooks/19b_cNMF_valdiation.ipynb
        ```

        After `assemble_full` records K=8 as primary, run or resume the full-data
        200-start K=8 freeze:

        ```bash
        GDT_19B_STAGE=gray_primary_k8 jupyter nbconvert --execute --to notebook --inplace \
          --ExecutePreprocessor.timeout=-1 notebooks/19b_cNMF_valdiation.ipynb
        ```

        After the primary freeze validates, run or resume the 13-library raw-UMI
        projection and donor-level external analysis:

        ```bash
        GDT_19B_STAGE=gse294268_score jupyter nbconvert --execute --to notebook --inplace \
          --ExecutePreprocessor.timeout=-1 notebooks/19b_cNMF_valdiation.ipynb
        ```

        Run or resume the PANGEA raw-UMI atlas projection; each Series is an
        independently validated restart unit:

        ```bash
        GDT_19B_STAGE=pangea_score jupyter nbconvert --execute --to notebook --inplace \
          --ExecutePreprocessor.timeout=-1 notebooks/19b_cNMF_valdiation.ipynb
        ```

        Run the Gray Data S12 independent-donor orthogonal functional validation;
        this reuses the five completed donor-held-out K8 checkpoints and does not
        refit cNMF:

        ```bash
        GDT_19B_STAGE=gray_functional_validation \
          jupyter nbconvert --execute --to notebook --inplace \
          --ExecutePreprocessor.timeout=-1 notebooks/19b_cNMF_valdiation.ipynb
        ```

        Run or resume the locked one-shot GSE268991 paired GEX/ADT/gamma-delta-TCR
        final external validation; each of the 14 tissue samples is an independently
        validated restart unit:

        ```bash
        GDT_19B_STAGE=gse268991_final_validation \
          jupyter nbconvert --execute --to notebook --inplace \
          --ExecutePreprocessor.timeout=-1 notebooks/19b_cNMF_valdiation.ipynb
        ```

        Scheduler partition, account, memory and walltime are intentionally not
        invented here; wrap the single-job command in the Woody-approved scheduler
        template used by your group.
        """),
    ]

    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.update({
        "kernelspec": {
            "display_name": "Python (gdT cNMF)", "language": "python", "name": "gdt_cnmf"
        },
        "language_info": {"name": "python", "version": "3"},
    })
    return notebook


def _self_test():
    assert normalize_gse294268_cell_id("AAAC.1_13") == "AAAC-1_13"
    assert normalize_gse294268_cell_id("AAAC-1_13") == "AAAC-1_13"
    notebook = build_notebook()
    assert len(notebook.cells) >= 10
    joined = "\n".join("".join(cell.source) for cell in notebook.cells)
    assert "raw counts" in joined and "GSE294268" in joined
    assert "lineage classifier" in joined and "not" in joined
    assert "GSE294268_RAW" in joined and "within_K_fold_pair_matching" in joined
    assert "K=10-reference recovery" in joined and "P9_P10_composite_recovery" in joined
    assert "pangea_score" in joined and "PANGEA_primary_K8_projection" in joined
    assert "sample proxy" in joined and "GSE149356" in joined
    assert "gray_functional_validation" in joined and "Gray Data S12" in joined
    assert "P2–GzmB" in joined and "independent-donor" in joined
    assert "gse268991_final_validation" in joined and "GSE268991" in joined
    assert "preanalysis_evaluation_manifest" in joined and "TRGC_assignment" in joined
    for cell in notebook.cells:
        if cell.cell_type == "code":
            compile(cell.source, "<19b generated cell>", "exec")


def _validation_self_test(include_aucell=False):
    """Small deterministic tests, callable without loading any project data."""
    import numpy as np
    import pandas as pd
    from scipy import sparse
    namespace = {}
    exec(VALIDATION_HELPERS, namespace)
    universe = pd.Index(["A", "B", "ENSG003", "C"])
    training_var = pd.DataFrame({"feature_name": ["A", "B", "ENSG003", "D"]},
                                index=["ENSG001", "ENSG002", "ENSG003", "ENSG004"])
    features = pd.DataFrame({"feature_id": ["ENSG001.2", "ENSG002", "ENSG003", "other", "ENSG004"],
                             "feature_name": ["old_A", "A", "old_C", "A", "B"]})
    mapping, audit = namespace["_align_training_features"](features, training_var, universe)
    assert mapping.toarray().tolist() == [[1, 0, 0, 0], [0, 1, 0, 0],
                                         [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 0]]
    assert audit.symbol_ID_disagreement.sum() == 4
    counts = sparse.csr_matrix([[10, 5, 0, 0], [0, 0, 0, 0], [2, 8, 0, 0]])
    thin = namespace["_thin_raw_counts"]
    assert (thin(counts, 1, 3) != counts).nnz == 0
    assert (thin(counts, .5, 3) != thin(counts, .5, 3)).nnz == 0
    assert np.all(thin(counts, .5, 3).toarray() <= counts.toarray())
    try:
        thin(counts.astype(float) * .01, .5, 3)
    except ValueError:
        pass
    else:
        raise AssertionError("Noninteger counts accepted for thinning")
    scorer = {"universe": universe, "hvg_positions": np.arange(4), "sd": np.ones(4),
              "basis": np.asarray([[1., 0, 0, 0], [0, 1., 0, 0]]),
              "solver_params": {"solver": "cd", "max_iter": 1000, "tol": 1e-7}}
    raw, usage, quality = namespace["_project_validation_counts"](counts, scorer, 3)
    assert np.allclose(usage[[0, 2]].sum(axis=1), 1)
    assert np.isnan(usage[1]).all() and np.isnan(quality.scaled_HVG_relative_error.iloc[1])
    assert np.allclose(usage[0], [2 / 3, 1 / 3])
    assert quality.scaled_HVG_relative_error.iloc[0] < 1e-7
    tcr = pd.DataFrame({"n_productive_TRG": [1, 2, 1], "n_productive_TRD": [1, 1, 0],
                        "TRGC_assignment": ["TRGC1", "TRGC1", "TRGC2"]}, index=["a", "b", "c"])
    adt = pd.DataFrame({"ADT_total_counts": [0, 2, 3]}, index=tcr.index)
    eligible = namespace["_strict_multimodal_eligibility"](tcr, adt)
    assert eligible.single_productive_TRG_TRD.tolist() == [True, False, False]
    assert eligible.has_ADT_counts.tolist() == [False, True, True]
    if include_aucell:
        genes = pd.Index([f"gene{i}" for i in range(100)])
        matrix = sparse.csr_matrix(np.vstack([np.arange(100), np.arange(100)[::-1], np.zeros(100)]))
        scores = namespace["_aucell_validation_scores"](
            matrix, genes, {"P1": ["gene99", "gene98"], "P2": ["gene0", "gene1"]})
        assert scores.loc[0, "P1"] > scores.loc[1, "P1"]
        assert scores.loc[1, "P2"] > scores.loc[0, "P2"]
        assert scores.iloc[2].isna().all()
    return {"feature_mapping": "passed", "UMI_thinning": "passed",
            "projection_and_zero_counts": "passed", "multimodal_eligibility": "passed",
            "AUCell": "passed" if include_aucell else "not run"}


def _robustness_self_test():
    """Test both benchmark paths with synthetic data, without creating outputs."""
    import ast
    import hashlib
    import json
    import numpy as np
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.metrics import log_loss, balanced_accuracy_score
    parsed = ast.parse(ROBUSTNESS_CODE)
    runner = next(n for n in parsed.body if isinstance(n, ast.FunctionDef))
    definitions = [n for n in runner.body if isinstance(n, ast.FunctionDef)
                   and n.name in {"methods", "benchmark"}]
    namespace = dict(locals())
    calls = []
    def fake_unit(label, specification, schema, produce):
        frames = produce()
        assert set(frames) == set(schema)
        for name, columns in schema.items():
            assert set(columns) <= set(frames[name].columns)
            assert frames[name].index.is_unique
        calls.append(label)
        return frames
    namespace["unit"] = fake_unit
    namespace["stable_token"] = lambda value: hashlib.sha256(
        json.dumps(value, sort_keys=True).encode()).hexdigest()
    exec(compile(ast.Module(body=definitions, type_ignores=[]), "<benchmark test>", "exec"), namespace)
    rng = np.random.default_rng(20260919)
    metadata = pd.DataFrame({"donor": np.repeat(["a", "b", "c"], 6),
                             "Experiment": np.repeat(["x", "y", "z"], 6)})
    features = pd.DataFrame(rng.normal(size=(18, 4)), columns=[
        "covariate_depth", "cNMF_P2", "rank_signature_P2", "AUCell_P2"])
    features["covariate_Experiment"] = metadata.Experiment
    labels = pd.DataFrame({"Stim": np.tile(["CR", "MO", "HDMAPP"], 6)})
    classification = namespace["benchmark"](features, labels, metadata, True,
                                             "synthetic test only", "all_cells", "donor")
    assert len(classification) == 27
    assert np.isfinite(classification.log_loss).all()
    outcomes = pd.DataFrame(rng.normal(size=(18, 2)), columns=["CD45RA", "CCR7"])
    regression = namespace["benchmark"](features, outcomes, metadata, False,
                                         "synthetic test only", "all_cells", "donor")
    assert len(regression) == 27
    assert np.isfinite(regression.standardized_MSE).all()
    assert len(calls) == 54
    return {"donor_classification": "passed", "donor_protein_regression": "passed",
            "checkpoint_units_tested": len(calls)}


def _depth_self_test():
    import json
    import numpy as np
    import pandas as pd
    namespace = {"np": np, "pd": pd}
    exec(DEPTH_CALIBRATION_HELPERS, namespace)
    check = namespace["_depth_assert_split"]
    check(["train"], ["a", "b", "c"], "test")
    for train, calibrate, test in [(["a"], ["a", "b", "c"], "test"),
                                   (["train"], ["a", "b", "test"], "test"),
                                   (["train"], ["a", "b"], "test")]:
        try: check(train, calibrate, test)
        except ValueError: pass
        else: raise AssertionError("Invalid donor partition accepted")
    frame = pd.DataFrame({"donor": np.repeat(["a", "b", "c"], 40),
        "fraction": np.tile(np.repeat([.25, 1.], 20), 3),
        "cell_id": [str(i) for i in range(120)],
        "library_over_training_universe": np.tile(np.arange(1, 41) * 100., 3),
        "valid_objective": True, "error": np.tile(np.linspace(.95, .3, 40), 3)})
    weights = namespace["_depth_weights"](frame)
    assert np.allclose(pd.Series(weights).groupby(frame.donor).sum(), 1 / 3)
    assert np.allclose(pd.Series(weights).groupby([frame.donor, frame.fraction]).sum(), 1 / 6)
    model = namespace["_depth_fit"](frame, "error")
    restored = json.loads(json.dumps(model))
    predict = namespace["_depth_predict"]
    depths = [0, np.nan, -1, 99, 100, 4000, 4001]
    prediction, raw, support = predict(restored, depths)
    assert support.tolist() == [False, False, False, False, True, True, False]
    assert np.isnan(prediction[~support]).all()
    assert ((prediction[support] >= 0) & (prediction[support] <= 1)).all()
    assert np.allclose(predict(model, [100, 500, 1000])[0], predict(restored, [100, 500, 1000])[0])
    constant = frame.copy(); constant["error"] = .2; constant["library_over_training_universe"] = 100
    flat = namespace["_depth_fit"](constant, "error")
    assert np.allclose(predict(flat, [100])[0], .2)
    invalid = frame.copy(); invalid["library_over_training_universe"] = 0
    try: namespace["_depth_fit"](invalid, "error")
    except ValueError: pass
    else: raise AssertionError("All-zero depths accepted")
    return {"donor_isolation": "passed", "donor_depth_weights": "passed",
            "support_and_zero_depth": "passed", "model_roundtrip": "passed", "constant_inputs": "passed"}


def export_serum_rna():
    """Deterministic, atomic RNA-only conversion; requires Rscript and Matrix."""
    import os, json, gzip, hashlib, shutil, subprocess, tempfile
    project = Path(__file__).resolve().parent
    source = project.parent / "seurat_obj.rds"
    source_md5 = hashlib.md5(source.read_bytes()).hexdigest()
    if source_md5 != "9a9bdf00eb10595f20d7eacfbd9ce27d":
        raise ValueError("Seurat object differs from the verified published release")
    destination = project / "results/models/19b_cNMF_valdiation/Zenodo17590095_primary_K8_time_course_input"
    code_hash = hashlib.sha256(SERUM_RNA_EXPORT.encode()).hexdigest()
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    if destination.exists():
        manifest = json.loads((destination / "export_manifest.json").read_text())
        if manifest["source_md5"] != source_md5 or manifest["export_code_sha256"] != code_hash:
            raise ValueError("Existing RNA export has a different configuration; do not overwrite")
        for name, value in manifest["outputs"].items():
            if digest(destination / name) != value: raise ValueError("Existing RNA export is damaged")
        print("Validated existing export:", destination); return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".Zenodo17590095_primary_K8_time_course_", dir=destination.parent))
    rscript = os.environ.get("GDT_RSCRIPT", shutil.which("Rscript") or "Rscript")
    subprocess.run([rscript, "-", str(source), str(temporary)], input=SERUM_RNA_EXPORT,
                   text=True, check=True)
    matrix = temporary / "RNA_counts.mtx"
    with matrix.open("rb") as src, (temporary / "RNA_counts.mtx.gz").open("wb") as target:
        with gzip.GzipFile(fileobj=target, mode="wb", mtime=0) as zipped:
            shutil.copyfileobj(src, zipped)
    matrix.unlink()  # Only this newly generated, reproducible temporary matrix.
    outputs = {x.name: digest(x) for x in temporary.iterdir() if x.is_file()}
    manifest = {"source_md5": source_md5, "source_sha256": digest(source), "source_name": source.name,
                "export_code_sha256": code_hash, "outputs": outputs, "RNA_only": True,
                "gene_by_cell_dimensions": [36485, 7363]}
    (temporary / "export_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)
    print(destination)


def _followup_self_test():
    import numpy as np, pandas as pd, json, hashlib, tempfile
    namespace = {"np": np, "pd": pd, "Path": Path, "json": json,
                 "sha256sum": lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()}
    exec(FOLLOWUP_HELPERS, namespace)
    frame = pd.DataFrame([dict(donor=d, day=day, program="P7", scenario="native", replicate=0,
                              gate="all_valid", serum=s, median_usage=v)
        for d in ("A", "B") for day in (0, 7) for s, v in (("+", .3), ("-", .1))])
    paired = namespace["_serum_paired"](frame)
    assert len(paired) == 2 and np.allclose(paired.WITH_SERUM_minus_WO_SERUM, .2)
    missing = frame.loc[~((frame.donor == "B") & (frame.serum == "-"))]
    assert not namespace["_serum_paired"](missing).query("donor == 'B'").paired.iloc[0]
    duplicate = pd.concat([frame, frame.iloc[[2]]])
    try: namespace["_serum_paired"](duplicate)
    except ValueError: pass
    else: raise AssertionError("Duplicate pair accepted")
    cells = pd.DataFrame([{"serum": s, "RNA_snn_res.0.8": c, "P7": v}
        for s, c, n, v in (("+", "0", 20, .8), ("+", "1", 10, .4), ("-", "0", 10, .6), ("-", "1", 20, .2))
        for _ in range(n)])
    result = namespace["_serum_decompose"](cells, ["P7"])
    assert np.allclose(result.within_RNA_stratum + result.composition, .3333333333333333)
    assert namespace["_serum_decompose"](cells, ["P7"], 30).empty
    directory = Path(__file__).resolve().parent / "results/models/19b_cNMF_valdiation"
    with tempfile.TemporaryDirectory(prefix=".p7_diagnostics_test_", dir=directory) as tmp:
        folder = Path(tmp); (folder / "data").write_bytes(b"synthetic test fixture")
        (folder / "scoring.complete.json").write_text(json.dumps({"signature": "test",
            "outputs": {"data": namespace["sha256sum"](folder / "data")}}))
        namespace["_followup_read_unit"](folder, "test", {"data"})
        (folder / "data").write_bytes(b"deliberate corruption")
        try: namespace["_followup_read_unit"](folder, "test", {"data"})
        except ValueError: pass
        else: raise AssertionError("Corrupted checkpoint accepted")
    return {"paired_sign_and_donor_separation": "passed", "D0_excluded": "passed",
            "missing_and_duplicate_conditions": "passed", "decomposition_identity": "passed",
            "checkpoint_mutation": "passed"}


def _incremental_self_test():
    """Synthetic boundary tests only; never used as biological evidence."""
    import numpy as np
    import pandas as pd
    import tempfile
    namespace = {}
    assert "INCREMENTAL_HELPERS" in globals(), "Incremental validation helpers not implemented"
    exec(INCREMENTAL_HELPERS, namespace)
    assert "_iv_same_rows" in namespace, "Exact historical cohort guard absent"
    same_rows = namespace["_iv_same_rows"]
    same_rows(pd.Index(["a", "b"]), pd.Index(["b", "a"]))
    for wrong in [pd.Index(["a", "c"]), pd.Index(["a", "a"])]:
        try: same_rows(pd.Index(["a", "b"]), wrong)
        except ValueError: pass
        else: raise AssertionError("Same-size but different/duplicated cohort accepted")
    split = namespace["_iv_splits"]
    meta = pd.DataFrame({"donor": list("aabbccdd"), "Experiment": list("xxyyzzww")})
    folds = list(split(meta, "donor", "d"))
    assert len(folds) == 3
    for _, train, test in folds:
        assert not set(train) & set(test)
        assert not {6, 7} & (set(train) | set(test))
        assert not set(meta.loc[train, "donor"]) & set(meta.loc[test, "donor"])
    bad = meta.copy(); bad.index = [0] * len(bad)
    try: list(split(bad, "donor", "d"))
    except ValueError: pass
    else: raise AssertionError("Duplicate row IDs accepted")
    X = pd.DataFrame({"x": [0., 2., 0., 2., 0., 2., 100., 200.]})
    Y = pd.DataFrame({"endpoint": [0., 2., 0., 2., 0., 2., 100., 200.]})
    fit = namespace["_iv_fit"]
    pred, metrics, audit = fit(X, Y, meta, [0, 1, 2, 3], [6, 7], False, 1.)
    assert np.isclose(audit.loc[audit.feature.eq("x"), "center"].iloc[0], 1.)
    assert np.isclose(audit.loc[audit.feature.eq("endpoint"), "center"].iloc[0], 1.)
    assert np.isclose(metrics.loss.iloc[0], np.square(pred.predicted_standardized__endpoint -
        pred.observed_standardized__endpoint).mean())
    select = namespace["_iv_select"]
    chosen, losses = select(X, Y, meta, "donor", "d", False, [0.1, 1.])
    changed_x, changed_y = X.copy(), Y.copy()
    changed_x.loc[[6, 7], "x"] = [-1e6, 1e6]
    changed_y.loc[[6, 7], "endpoint"] = [-1e6, 1e6]
    chosen2, losses2 = select(changed_x, changed_y, meta, "donor", "d", False, [0.1, 1.])
    assert chosen == chosen2 and losses.equals(losses2), "Outer test affected tuning"
    unbalanced_x = pd.DataFrame({"x": [0., 2., 9., 50.]})
    unbalanced_y = pd.DataFrame({"endpoint": [0., 2., 9., 50.]})
    unbalanced_meta = pd.DataFrame({"donor": ["a", "a", "b", "c"]})
    _, _, unbalanced_audit = fit(unbalanced_x, unbalanced_y, unbalanced_meta, [0, 1, 2], [3], False, 1.)
    assert np.isclose(unbalanced_audit.loc[unbalanced_audit.feature.eq("x"), "center"].iloc[0], 5.), "Feature scaling is not donor-balanced"
    class_meta = pd.DataFrame({"donor": np.repeat(list("abcd"), 3), "Experiment": np.repeat(list("wxyz"), 3)})
    class_x = pd.DataFrame({"x": np.tile([-2., 0., 2.], 4)})
    class_y = pd.DataFrame({"Stim": np.tile(["CR", "MO", "HDMAPP"], 4)})
    class_pred, class_metrics, _ = fit(class_x, class_y, class_meta, list(range(9)), [9, 10, 11], True, 1.)
    probability_columns = [c for c in class_pred if c.startswith("probability_")]
    assert np.allclose(class_pred[probability_columns].sum(axis=1), 1.)
    manual_loss = -np.mean([np.log(row["probability_" + row["observed"]]) for _, row in class_pred.iterrows()])
    assert np.isclose(class_metrics.loss.iloc[0], manual_loss)
    checkpoint = namespace["_iv_checkpoint"]
    with tempfile.TemporaryDirectory(prefix="19b_incremental_test_") as temporary:
        folder = Path(temporary)
        result, reused = checkpoint(folder, "test-signature", {"scores": ["value"]},
                                   lambda: {"scores": pd.DataFrame({"value": [1.]})})
        assert not reused and result["scores"].value.iloc[0] == 1.
        def must_not_run(): raise AssertionError("Valid checkpoint was recomputed")
        _, reused = checkpoint(folder, "test-signature", {"scores": ["value"]}, must_not_run)
        assert reused
        (folder / "scores.parquet").write_bytes(b"corrupt")
        try: checkpoint(folder, "test-signature", {"scores": ["value"]}, must_not_run)
        except ValueError: pass
        else: raise AssertionError("Corrupt checkpoint accepted")
    return {"grouped_splits": "passed", "train_only_scaling": "passed",
            "outer_test_tuning_isolation": "passed", "loss_recalculation": "passed",
            "checkpoint_reuse_and_corruption": "passed"}


def _incremental_notebook_test():
    namespace = {"EXECUTION_STAGE": "input_qc"}
    assert "INCREMENTAL_CODE" in globals(), "Incremental stage not implemented"
    exec(INCREMENTAL_CODE, namespace)
    assert callable(namespace["run_incremental_value_validation"])
    assert "_run_incremental_value_validation" in namespace, "Stage failure-state guard is absent"
    recorded = {}
    def fail_during_assembly(context):
        context["progress"] = recorded
        context["flush"] = lambda: recorded.update(flushed=True)
        raise ValueError("test assembly failure")
    namespace["_run_incremental_value_validation"] = fail_during_assembly
    try: namespace["run_incremental_value_validation"]()
    except ValueError: pass
    else: raise AssertionError("Assembly failure swallowed")
    assert recorded.get("status") == "failed" and recorded.get("flushed")
    notebook = build_notebook()
    for cell in notebook.cells:
        if cell.cell_type == "code": compile(cell.source, "<19b cell>", "exec")
    return {"stage_isolation_and_notebook_compile": "passed"}


def _incremental_plot_test():
    import pandas as pd
    import matplotlib.pyplot as plt
    namespace = {}
    exec(INCREMENTAL_HELPERS, namespace)
    assert "_iv_plot_added_value" in namespace, "Paired-loss plot not implemented"
    records = []
    for baseline in ["covariates_only", "covariates_plus_rank_signature", "covariates_plus_AUCell",
                     "covariates_plus_full_measured_AUCell", "covariates_plus_Hallmark_AUCell",
                     "covariates_plus_full_measured_Hallmark_AUCell"]:
        model = "covariates_plus_cNMF" if baseline == "covariates_only" else baseline + "_plus_cNMF"
        for donor, experiment, delta in [("a", "x", 1.), ("b", "x", 3.), ("c", "y", -1.)]:
            records.append(dict(donor=donor, Experiment=experiment, error_reduction=delta,
                                baseline=baseline, model=model, dataset="test", quality_gate="all_cells", split="Experiment"))
    fig, source = namespace["_iv_plot_added_value"](pd.DataFrame(records), "test", "all_cells", "Experiment", "Liberation Sans")
    assert len(source) == 12 and set(source.error_reduction) == {2., -1.}
    assert len(fig.axes) == 1 and abs(fig.get_figwidth() * 25.4 - 180) < 1e-8
    plt.close(fig)
    return {"batch_balanced_figure_source": "passed"}


def _perturbation_prediction_self_test():
    import numpy as np
    import pandas as pd
    assert 'PERTURBATION_PREDICTION_HELPERS' in globals(), 'Perturbation prediction helpers absent'
    ns={'np':np,'pd':pd};exec(PERTURBATION_PREDICTION_HELPERS,ns)
    programs=['P1','P2','P3','P4','P5','P7','P9','P10'];rows=[]
    for donor,base,change in [('donorA',.2,.1),('donorB',.3,.2),('donorC',.4,-.1)]:
        for gate in ['all_valid','historical_q95']:
            for condition,day,serum in [('D0',0,'Day 0')]+[(f'D{d}_{tag}',d,s) for d in [7,10,14] for tag,s in [('WITH_SERUM','+'),('WO_SERUM','-')]]:
                v=base+(change if day else 0)
                for program,value in zip(programs,[v,1-v,0,0,0,0,0,0]):
                    rows.append(dict(donor=donor,gate=gate,condition=condition,day=day,serum=serum,program=program,
                        n_cells=100,mean_usage=value,scenario='native',replicate=0))
    frame=pd.DataFrame(rows);native=ns['_pr_prepare'](frame,programs)
    # Program-specific parent summaries must not duplicate the sample inventory.
    decorated=native.assign(median_usage=np.linspace(0,1,len(native)),matched_native_median=.5,paired_cell_MAE=.01)
    quality=native[['donor','condition','scenario','replicate']].drop_duplicates().assign(n_valid_converged=100,n_historical_q95_accepted=100)
    assert '_pr_inventory' in ns, 'Explicit sample-level inventory helper absent'
    inventory=ns['_pr_inventory'](decorated,quality)
    assert len(inventory)==42 and not inventory.duplicated(['donor','gate','condition']).any()
    assert not {'program','mean_usage','median_usage','matched_native_median','paired_cell_MAE'} & set(inventory)
    train=native[(native.donor!='donorC')&(native.gate=='all_valid')]
    baseline=native[(native.donor=='donorC')&(native.gate=='all_valid')&(native.day==0)]
    for method,expected in [('no change',.4),('training-donor mean change',.55)]:
        pred,params=ns['_pr_forecast'](train,baseline,programs,method)
        assert len(pred)==48 and np.allclose(pred.loc[pred.program=='P1','predicted_mean_usage'],expected)
        assert not any(c in pred for c in ['observed_mean_usage','absolute_error'])
        assert np.allclose(pred.groupby('condition').predicted_mean_usage.sum(),1)
    pred,params=ns['_pr_forecast'](train,baseline,programs,'ridge regression')
    assert len(params)==40 and np.isfinite(pred.predicted_mean_usage).all()
    future=native[(native.donor=='donorC')&(native.gate=='all_valid')&(native.day>0)]
    before=ns['_pr_evaluate'](pred,future);changed=future.copy()
    changed.loc[changed.program=='P1','mean_usage']+=.1;changed.loc[changed.program=='P2','mean_usage']-=.1
    after=ns['_pr_evaluate'](pred,changed)
    pd.testing.assert_series_equal(before.predicted_mean_usage,after.predicted_mean_usage)
    assert not np.allclose(before.absolute_error,after.absolute_error)
    p,_=ns['_pr_forecast'](train,baseline,programs,'no change');e=ns['_pr_evaluate'](p,future)
    metric=ns['_pr_metrics'](e);overall=metric[metric.program.isna()].iloc[0]
    assert np.isclose(overall.MAE,.025) and overall.n_evaluations==48
    assert np.allclose(ns['_pr_simplex'](np.array([[-.2,1.2],[.6,.6]])),[[0,1],[.5,.5]])
    for bad in [pd.concat([frame,frame.iloc[:1]]),frame.iloc[1:],frame.assign(mean_usage=-1),frame.assign(n_cells=0)]:
        try:ns['_pr_prepare'](bad,programs)
        except ValueError:pass
        else:raise AssertionError('Invalid or incomplete summary accepted')
    try:ns['_pr_forecast'](native[native.gate=='all_valid'],baseline,programs,'no change')
    except ValueError:pass
    else:raise AssertionError('Held-out donor admitted to training')
    unequal=train.copy();unequal.loc[unequal.donor=='donorA','n_cells']=100000
    p,_=ns['_pr_forecast'](unequal,baseline,programs,'training-donor mean change')
    assert np.allclose(p.loc[p.program=='P1','predicted_mean_usage'],.55)
    exec(PERTURBATION_PREDICTION_CODE,{'EXECUTION_STAGE':'input_qc'})
    for c in build_notebook().cells:
        if c.cell_type=='code':compile(c.source,'<perturbation notebook>','exec')
    return 'prediction contract, held-out isolation, literal forecasts, simplex, errors and donor weighting passed'


def _end_to_end_summary_self_test():
    import numpy as np
    import pandas as pd
    ns={'np':np,'pd':pd};exec(END_TO_END_HELPERS,ns);exec(END_TO_END_SUMMARY_HELPERS,ns)
    p=pd.DataFrame({'cell_id':list('abcdef'),'Series':['GSE149356']*3+['GSE165045','other','GSE165045'],
        'Sample':['HC1']*3+['BL1','sample','BL1'],'hard_call':['predicted gdT','predicted abT','predicted gdT','predicted gdT','predicted gdT','predicted abT'],
        'annotation_Level2':['T_gdT_TRDV1','T_gdT_TRDV2','T_gdT_TRDV2','T_CD4_N&CM','NK_CD16','T_CD4_N&CM'],
        'silver_evaluation_eligible':[True,True,True,True,False,True],'silver_y':[1,1,1,0,0,0]})
    e=p[['cell_id','Series','Sample','hard_call']].copy();e['receptor_evidence']=['TRD + TRG with CDR3; QC flags unavailable']*3+['strict_paired_abTCR','not linked; unknown','strict_paired_abTCR']
    s=p.copy();s['P1']=1.;s['valid_objective']=True;s['solver_converged']=True;s['scaled_HVG_relative_error']=[.9,.1,.1,.1,.1,.1]
    s=ns['_e2e_route'](s,['P1'],.8)
    f=s.iloc[:3].copy();f['p_calibrated']=[.8,.1,.5];f['expression_subtype']=['Vd1','Vd2','other_or_unresolved'];f['sequence_subtype']=['Vd1','Vd2','Vd2']
    full=ns['_e2s_prepare'](p,e,s,f,lower=.3,upper=.7)
    flow,metrics=ns['_e2s_tables'](full,'pooled')
    row=flow.iloc[0];assert row.n_input==6 and row.n_selected==4 and row.n_receptor_unknown==1
    assert row.n_binary_reference==3 and row.n_binary_reference_missed==1 and row.n_binary_reference_unresolved==1 and row.n_binary_reference_correct==1
    def metric(name,ref):return metrics.loc[metrics.metric.eq(name)&metrics.reference_set.eq(ref)].iloc[0]
    assert metric('assigned_accuracy','paired_TRG_TRD_primary').value==1.
    assert np.isclose(metric('end_to_end_correct_subtype_with_state','paired_TRG_TRD_primary').value,1/3)
    assert metric('end_to_end_correct_subtype_with_q95','paired_TRG_TRD_primary').value==0.
    assert np.isnan(metric('lineage_precision','receptor_supported').value), 'Cross-study one-class evidence must not become pooled precision'
    assert metric('lineage_false_positive_rate','strict_paired_abTCR').value==.5
    z=full[full.Series.eq('other')];_,empty=ns['_e2s_tables'](z,'Series')
    assert empty.loc[empty.reference_set.eq('paired_TRG_TRD_primary'),'value'].isna().all()
    _,a=ns['_e2s_tables'](full[full.Series.eq('GSE149356')],'Series');assert a.loc[a.metric.eq('lineage_precision')&a.reference_set.eq('receptor_supported'),'value'].isna().all()
    for bad_s,bad_f in [(s.iloc[1:],f),(s,pd.concat([f,f.iloc[:1]])),(s,f.assign(p_calibrated=[.2,.1,.5]))]:
        try:ns['_e2s_prepare'](p,e,bad_s,bad_f,lower=.3,upper=.7)
        except ValueError:pass
        else:raise AssertionError('Broken population or subtype provenance accepted')
    shuffled=ns['_e2s_prepare'](p.sample(frac=1,random_state=1),e,s.sample(frac=1,random_state=2),f,lower=.3,upper=.7)
    pd.testing.assert_frame_equal(full.sort_index(),shuffled.sort_index())
    exec(END_TO_END_SUMMARY_CODE,{'EXECUTION_STAGE':'input_qc'})
    for cell in build_notebook().cells:
        if cell.cell_type=='code':compile(cell.source,'<complete denominator notebook>','exec')
    return 'complete denominator, reference scope, exact joins, uncertainty and stage isolation passed'


def _end_to_end_followup_self_test():
    import numpy as np
    import pandas as pd
    import re
    assert 'END_TO_END_FOLLOWUP_HELPERS' in globals(), 'Approved follow-up helpers absent'
    ns={'np':np,'pd':pd,'re':re};exec(END_TO_END_FOLLOWUP_HELPERS,ns)
    samples=['CB1','CB2','HC1','HC2'];groups={s:('neonate; cord blood' if s.startswith('CB') else 'adult; peripheral blood') for s in samples}
    x=pd.DataFrame({'cell_id':['GSE149356_'+s+'@ACGT-1' for s in samples],
        'Series':'GSE149356','Sample':samples,'is_receptor_positive':True,
        'hard_call':'predicted gdT','state_score_valid':True,'within_Gray_q95':[False,False,True,True],
        'state_output_available':True,'state_output_q95_available':[False,False,True,True],
        'P1':[.1,.3,.4,.8],'total_counts':[10.,20.,30.,40.],
        'n_genes_by_counts':[5.,10.,15.,20.],'pct_counts_mt':1.,'scaled_HVG_relative_error':.5})
    t=pd.DataFrame({'donor':samples,'barcode':[s+'_ACGT' for s in samples],
        'chain_TRD':'TRD','chain_TRG':'TRG','cdr3_TRD':'A','cdr3_nt_TRD':'AAA',
        'cdr3_TRG':'G','cdr3_nt_TRG':'GGG','v_gene_TRD':['TRDV3','TRDV1,TRDV2','TRDV1','TRDV2']})
    p=pd.DataFrame({'donor':['HC1','HC2'],'rna_barcode':['ACGT-1','ACGT-1'],'p_calibrated':[.3,.7]})
    j=ns['_e2f_join'](x,t,p,groups)
    assert j.expression_subtype.tolist()==['not_available','not_available','Vd2','Vd1']
    assert j.sequence_assisted_subtype.tolist()==['Vd3','ambiguous','Vd1','Vd2']
    assert j.state_output_available.all() and j.p_calibrated.iloc[:2].isna().all()
    p2=p.copy();p2.p_calibrated=[.300001,.699999];z=ns['_e2f_join'](x,t,p2,groups)
    assert z.expression_subtype.iloc[2:].eq('other_or_unresolved').all() and z.state_output_available.all()
    old=ns['_e2f_join'](x,t,p,groups,lower=.2,upper=.8)
    assert old.expression_subtype.iloc[2:].eq('other_or_unresolved').all()
    for col in ['p_calibrated','P1','hard_call','selected','state_output_available','state_output_q95_available','sequence_assisted_subtype']:
        pd.testing.assert_series_equal(old[col],j[col])
    pd.testing.assert_frame_equal(ns['_e2f_group_contrasts'](old,['P1']),ns['_e2f_group_contrasts'](j,['P1']))
    for lower,upper in [(.7,.3),(.5,.5),(-.1,.7),(.3,1.1),(np.nan,.7)]:
        try:ns['_e2f_join'](x,t,p,groups,lower=lower,upper=upper)
        except ValueError:pass
        else:raise AssertionError('Invalid or overlapping subtype policy accepted')
    for bad in [pd.concat([p,p.iloc[:1]]),p.iloc[:1],p.assign(p_calibrated=[-0.1,.5])]:
        try:ns['_e2f_join'](x,t,bad,groups)
        except ValueError:pass
        else:raise AssertionError('Invalid adult predictions accepted')
    try:ns['_e2f_join'](x,pd.concat([t,t.iloc[:1]]),p,groups)
    except ValueError:pass
    else:raise AssertionError('Duplicate receptor keys accepted')
    summary=ns['_e2f_sample_summary'](j,['P1'])
    assert summary.missed_median.isna().all() and summary.n_missed.eq(0).all()
    contrast=ns['_e2f_group_contrasts'](j,['P1'])
    a=contrast[(contrast.quality_gate=='all_valid')&(contrast.population=='receptor_supported')].iloc[0]
    assert np.isclose(a.difference,.4) and a.n_adult_donors==2 and a.n_neonate_donors==2
    q=contrast[(contrast.quality_gate=='Gray_q95_accepted')&(contrast.population=='receptor_supported')].iloc[0]
    assert np.isnan(q.difference) and q.n_neonate_donors==0
    many=pd.concat([j,j.iloc[[2]*8]],ignore_index=True)
    a=ns['_e2f_group_contrasts'](many,['P1']);a=a[(a.quality_gate=='all_valid')&(a.population=='receptor_supported')].iloc[0]
    assert np.isclose(a.difference,.4), 'Cell counts must not weight donor contrast'
    exec(END_TO_END_FOLLOWUP_CODE,{'EXECUTION_STAGE':'input_qc'})
    for c in build_notebook().cells:
        if c.cell_type=='code':compile(c.source,'<followup notebook>','exec')
    return 'follow-up joins, uncertainty, sequence separation, missingness and donor weighting passed'


def _end_to_end_self_test():
    import numpy as np
    import pandas as pd
    assert 'END_TO_END_HELPERS' in globals(), 'End-to-end validation helpers absent'
    ns = {'np':np, 'pd':pd}; exec(END_TO_END_HELPERS, ns)
    p = pd.DataFrame({'cell_id':['a','b','c','d'], 'Series':['S','S','S','GSE149356'],
        'Sample':['x','x','y','HC1'], 'hard_call':['predicted gdT','predicted abT','predicted gdT','predicted gdT'],
        'annotation_Level2':['T_CD8_EM','T_gdT_TRDV1','T_CD8_EM','T_gdT_TRDV2']})
    e=p[['cell_id','Series','Sample','hard_call']].copy()
    e['receptor_evidence']=['not linked; unknown','not linked; unknown','strict_paired_abTCR','TRD + TRG with CDR3; QC flags unavailable']
    z=ns['_e2e_prepare'](p,e)
    assert z.is_receptor_negative.tolist()==[False,False,True,False]
    assert z.is_receptor_positive.tolist()==[False,False,False,True]
    assert z.in_primary_analysis.tolist()==[True,True,True,False]
    assert z.include_for_state.all()
    for bad in [pd.concat([e,e.iloc[:1]]),e.iloc[:-1]]:
        try: ns['_e2e_prepare'](p,bad)
        except ValueError: pass
        else: raise AssertionError('Invalid evidence join accepted')
    z['P1']=[.2,.8,.6,.3];z['P2']=1-z.P1
    z['valid_objective']=True;z['solver_converged']=[True,True,False,True]
    z['scaled_HVG_relative_error']=[.8,.8,.8,.96]
    z['subtype_prediction']=pd.NA
    out=ns['_e2e_route'](z,['P1','P2'],.9)
    assert out.state_output_available.tolist()==[True,False,False,True]
    assert out.state_output_q95_available.tolist()==[True,False,False,False]
    z['subtype_prediction']='Vd1'
    assert ns['_e2e_route'](z,['P1','P2'],.9).state_output_available.equals(out.state_output_available)
    z.loc['a','P1']=.9
    try:ns['_e2e_route'](z,['P1','P2'],.9)
    except ValueError:pass
    else:raise AssertionError('Non-normalized usage accepted')
    # Literal sample effects: x=-0.6, y=+0.2; equal-sample study effect=-0.2.
    rows=[]
    for sample, pred_value, ref_value, n in [('x',.2,.8,10),('y',.6,.4,1)]:
        for side,value in [('predicted',pred_value),('reference',ref_value)]:
            for j in range(n):rows.append({'Series':'S','Sample':sample,'P1':value,
                'hard_call':'predicted gdT' if side=='predicted' else 'predicted abT',
                'is_silver_gdT':side=='reference','is_receptor_positive':False,
                'is_receptor_negative':False,'in_primary_analysis':True,
                'state_score_valid':True,'within_Gray_q95':True})
    t=pd.DataFrame(rows);samples,studies=ns['_e2e_contrasts'](t,['P1'])
    s=studies[(studies.comparison=='predicted gdT minus PANGEA gdT annotation')&(studies.quality_gate=='all_valid')].iloc[0]
    assert s.n_paired_samples==2 and np.isclose(s.mean_paired_sample_difference,-.2)
    empty=samples[samples.comparison=='predicted gdT minus receptor-supported gdT']
    assert empty.difference.isna().all() and empty.n_reference.eq(0).all()
    # Both categorical keys retain >128 categories when an arm is empty.
    # pandas 2.3 / numpy 2 can overflow its int8 MultiIndex lookup codes.
    many=pd.concat([t.iloc[:1].assign(Series=f'S{i:03d}',Sample=f'x{i:03d}') for i in range(140)],ignore_index=True)
    for key in ['Series','Sample']:many[key]=pd.Categorical(many[key])
    categorical_samples,_=ns['_e2e_contrasts'](many,['P1'])
    assert categorical_samples.loc[categorical_samples.comparison.eq('predicted gdT minus receptor-supported gdT'),'difference'].isna().all()
    assert '_e2e_verify_evidence' in ns, 'Original receptor-audit provenance validation absent'
    original={'complete':True,'cell_output_sha256':'evidence-hash',
              'contract':{'inputs':{'pred':'pred-hash','receptor':'receptor-hash'}},
              'raw_inputs':{'rna':'rna-hash'}}
    current={'pred':'pred-hash','receptor':'receptor-hash','rna':'rna-hash'}
    ns['_e2e_verify_evidence'](original,'evidence-hash','pred',['receptor'],'rna',current)
    import copy
    for kind in ['incomplete','edited_evidence','changed_prediction','changed_receptor','changed_rna','absent_source']:
        test=copy.deepcopy(original);hashes=current.copy();eh='evidence-hash'
        if kind=='incomplete':test['complete']=False
        elif kind=='edited_evidence':eh='edited'
        elif kind=='changed_prediction':hashes['pred']='changed'
        elif kind=='changed_receptor':hashes['receptor']='changed'
        elif kind=='changed_rna':hashes['rna']='changed'
        else:del test['contract']['inputs']['receptor']
        try:ns['_e2e_verify_evidence'](test,eh,'pred',['receptor'],'rna',hashes)
        except ValueError:pass
        else:raise AssertionError('Stale receptor audit accepted: '+kind)
    ns2={'EXECUTION_STAGE':'input_qc'};exec(END_TO_END_CODE,ns2)
    for cell in build_notebook().cells:
        if cell.cell_type=='code':compile(cell.source,'<end_to_end notebook>','exec')
    return {'exact_join_unknown_labels':'passed','exclusion_and_routing':'passed',
            'sample_not_cell_weighting':'passed','absent_reference_preserved':'passed','stage_isolation':'passed'}


def _evidence_synthesis_self_test():
    import pandas as pd
    import numpy as np
    scope = {}
    assert 'EVIDENCE_SYNTHESIS_HELPERS' in globals(), 'Evidence synthesis helpers absent'
    exec(EVIDENCE_SYNTHESIS_HELPERS, scope)
    # Prevent technical replicates from being counted as independent donors.
    d = pd.DataFrame({'program': ['P7'] * 6, 'donor': ['A'] * 3 + ['B'] * 3,
                      'replicate': [0, 1, 2] * 2, 'value': [0, 0, 3, 2, 2, 2]})
    result = scope['_es_donor_summary'](d, ['program'], 'value').iloc[0]
    assert result.n_donors == 2 and result.donor_median == 1.5
    # Preserve missing contrasts and do not count missing as zero.
    d.loc[d.donor.eq('B'), 'value'] = np.nan
    result = scope['_es_donor_summary'](d, ['program'], 'value').iloc[0]
    assert result.n_donors == 1 and result.n_donors_missing == 1
    # A subset join cannot certify false-positive contamination of a full pipeline.
    ref = pd.DataFrame({'cell_id': ['a', 'b'], 'P7': [.2, .4]})
    pred = pd.DataFrame({'cell_id': ['a', 'b', 'c'], 'hard_call': ['gdT', 'abT', 'gdT']})
    joined = scope['_es_overlap'](pred, ref)
    assert joined['n_overlap'] == 2 and joined['n_lineage_only'] == 1
    deployed = pred.assign(hard_call=['predicted gdT','predicted abT','predicted gdT'])
    joined = scope['_es_overlap'](deployed, ref)
    assert joined['n_predicted_gdT'] == 2 and joined['n_predicted_gdT_with_state'] == 1
    assert joined['n_predicted_gdT_without_state'] == 1 and joined['n_state_not_predicted_gdT'] == 1
    try: scope['_es_overlap'](pd.concat([pred, pred.iloc[:1]]), ref)
    except ValueError: pass
    else: raise AssertionError('Duplicate cell IDs accepted')
    matrix = pd.DataFrame([[1., 0., 0.], [0., 1., 0.], [1., 1., 0.]], index=['P1','P2','P7'])
    pairs = scope['_es_spectrum_pairs'](matrix)
    row = pairs.loc[(pairs.program_a == 'P1') & (pairs.program_b == 'P7')].iloc[0]
    assert np.isclose(row.cosine, 2 ** -.5) and len(pairs) == 3
    # Source serum labels are +/-; condition names must not replace them.
    scores = pd.DataFrame({'donor':['A','A'], 'day':[7,7], 'program':['P7','P7'],
        'gate':['all_valid','all_valid'], 'replicate':[0,0], 'scenario':['native','native'],
        'serum':['+','-'], 'median_usage':[.1,.3]})
    native = scope['_es_native_serum_contrasts'](scores)
    assert len(native) == 1 and np.isclose(native.recomputed.iloc[0], -.2)
    scores.loc[0, 'median_usage'] = np.nan
    assert scope['_es_native_serum_contrasts'](scores).recomputed.isna().all()
    ns = {'EXECUTION_STAGE': 'input_qc'}
    exec(EVIDENCE_SYNTHESIS_CODE, ns)
    for cell in build_notebook().cells:
        if cell.cell_type == 'code': compile(cell.source, '<19b evidence cell>', 'exec')
    return {'donor_not_replicate_unit': 'passed', 'missing_preserved': 'passed',
            'subset_and_duplicate_join_guard': 'passed', 'spectrum_pairs': 'passed',
            'native_serum_label_and_missing_checks': 'passed', 'stage_isolation_and_compile': 'passed'}


def _usage_representation_self_test():
    import numpy as np
    import pandas as pd
    scope = {}
    exec(USAGE_REPRESENTATION_HELPERS, scope)
    import tempfile
    import os
    from pathlib import Path
    from unittest.mock import patch
    with tempfile.TemporaryDirectory() as temporary:
        source = Path(temporary) / 'source.py'; target = Path(temporary) / 'snapshot.py'
        source.write_bytes(b'validated source')
        with patch('os.replace', side_effect=OSError('simulated interruption')):
            try: scope['_ur_snapshot'](source, target)
            except OSError: pass
            else: raise AssertionError('Interrupted snapshot did not report failure')
        assert not target.exists(), 'Interrupted source snapshot looked complete'
        scope['_ur_snapshot'](source, target)
        assert target.read_bytes() == b'validated source'
        scope['_ur_snapshot'](source, target)
        source.write_bytes(b'changed source')
        try: scope['_ur_snapshot'](source, target)
        except ValueError: pass
        else: raise AssertionError('Immutable source snapshot overwritten')
    raw = pd.DataFrame([[1., 3.], [4., 4.]], index=['a', 'b'], columns=['P1', 'P2'])
    norm = pd.DataFrame([[.25, .75], [.5, .5]], index=raw.index, columns=raw.columns)
    scope['_ur_validate_usage'](raw, norm)
    for bad in [norm.iloc[::-1], norm * .9, norm.rename(columns={'P2': 'P3'})]:
        try: scope['_ur_validate_usage'](raw, bad)
        except ValueError: pass
        else: raise AssertionError('Invalid representation was accepted')
    try: scope['_ur_validate_usage'](raw * 0, norm)
    except ValueError: pass
    else: raise AssertionError('Zero coefficient sum was accepted')
    p = pd.DataFrame({'donor': ['a','b'], 'Experiment': ['GD005']*2,
        'observed': ['MO','CR'], 'probability_CR': [.1,.5],
        'probability_HDMAPP': [.1,.25], 'probability_MO': [.8,.25]})
    losses = scope['_ur_prediction_losses'](p)
    assert np.allclose(losses.loss, [.2231435513142097, .6931471805599453])
    assert losses.predicted.tolist() == ['MO','CR']
    bad = p.copy(); bad.loc[0, 'probability_CR'] = .9
    try: scope['_ur_prediction_losses'](bad)
    except ValueError: pass
    else: raise AssertionError('Invalid probability simplex was accepted')
    reg = pd.DataFrame({'donor': ['x'], 'predicted_standardized__A': [2.],
                        'observed_standardized__A': [0.], 'predicted_standardized__B': [4.],
                        'observed_standardized__B': [3.]})
    assert scope['_ur_prediction_losses'](reg).loss.iloc[0] == 2.5
    assert scope['_ur_verify_predictions'](p, p.copy()) is None
    try: scope['_ur_verify_predictions'](p, p.iloc[::-1])
    except ValueError: pass
    else: raise AssertionError('Changed historical prediction order accepted')
    # Stage must be inert during unrelated notebook execution.
    exec(USAGE_REPRESENTATION_CODE, {'EXECUTION_STAGE': 'input_qc'})
    print('usage representation tests passed: integrity, loss, identity, stage isolation')


def main():
    import sys
    if "--export-serum-rna" in sys.argv:
        export_serum_rna(); return
    if nbf is None:
        raise RuntimeError("Notebook generation requires nbformat; use the Woody cnmf_env interpreter")
    _self_test()
    project = Path(__file__).resolve().parent
    destination = project / "notebooks" / NOTEBOOK_NAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    notebook = build_notebook()
    # Preserve executed evidence only for byte-identical code cells. Newly added
    # or changed cells remain unexecuted; never attach stale outputs to new code.
    if destination.is_file():
        previous = nbf.read(destination, as_version=4)
        previous_code = {cell.source: cell for cell in previous.cells if cell.cell_type == "code"}
        for cell in notebook.cells:
            if cell.cell_type == "code" and cell.source in previous_code:
                old = previous_code[cell.source]
                cell.outputs = old.outputs
                cell.execution_count = old.execution_count
    nbf.write(notebook, temporary)
    nbf.read(temporary, as_version=4)
    temporary.replace(destination)
    print(destination)


if __name__ == "__main__":
    main()
