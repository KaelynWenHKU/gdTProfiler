"""Projection onto the eight γδ T-cell expression programmes (Gray K=8 cNMF reference).

Each cell is normalized to 10^6 counts over the reference gene set (12,927 genes),
restricted to the 2,000 reference genes, divided by their training standard
deviations (no centring or log transformation), and fitted by non-negative least
squares with the programme spectra held fixed. Usage is reported as the fraction of
each programme (rows sum to one). The relative reconstruction error indicates how well
the reference describes a cell; values above the Gray 95th percentile (0.924) flag
poorly described cells.
"""
from importlib import resources
import json
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import non_negative_factorization
from sklearn.exceptions import ConvergenceWarning

from .model import SoftmaxModel


def _path(name):
    return resources.files("gdtprofiler").joinpath("data", "programmes_K8", name)


class ProgrammeReference:
    """The fixed eight-programme reference (P1, P2, P3, P4, P5, P7, P9, P10)."""

    def __init__(self):
        self.ref = json.loads(_path("reference.json").read_text())
        with _path("spectra.tsv.gz").open("rb") as f:
            spectra = pd.read_csv(f, sep="\t", index_col=0, compression="gzip")
        with _path("training_sd.csv.gz").open("rb") as f:
            sd = pd.read_csv(f, index_col=0, compression="gzip")["sd"]
        if list(spectra.columns) != self.ref["hvgs"] or list(sd.index) != self.ref["hvgs"]:
            raise ValueError("Programme reference files are inconsistent")
        self.programmes = list(spectra.index)
        self.sd = sd.to_numpy(np.float64)
        self.basis = spectra.to_numpy(np.float64) / self.sd
        self.hvg_index = pd.Index(self.ref["universe"]).get_indexer(self.ref["hvgs"])

    def project(self, adata, layer=None, min_coverage=0.8):
        """Return programme usage and reconstruction error for every cell."""
        x = SoftmaxModel._counts(adata, layer)
        aligned, found = SoftmaxModel._align(x, adata.var_names, self.ref["universe"])
        aligned = aligned.astype(np.float64)
        coverage = float(found[self.hvg_index].mean())
        out = pd.DataFrame(np.nan, index=adata.obs_names, columns=self.programmes)
        out["relative_error"] = np.nan
        out["within_reference_q95"] = False
        out.attrs["reference_gene_coverage"] = coverage
        if coverage < min_coverage:
            warnings.warn(f"Only {coverage:.0%} of the 2,000 reference genes were found; "
                          "programme usage was not computed.")
            return out
        library = np.asarray(aligned.sum(1)).ravel()
        tpm = sparse.diags(np.divide(1e6, library, out=np.zeros_like(library), where=library > 0)) @ aligned
        y = tpm[:, self.hvg_index].multiply(1 / self.sd).tocsr()
        norm_sq = np.asarray(y.multiply(y).sum(1)).ravel()
        valid = (library > 0) & (norm_sq > 0)
        if valid.any():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                w, _, _ = non_negative_factorization(
                    y[valid], H=self.basis, update_H=False, n_components=len(self.programmes),
                    random_state=self.ref["seed"], **self.ref["solver"])
            total = w.sum(1)
            usage = np.divide(w, total[:, None], out=np.full_like(w, np.nan), where=total[:, None] > 0)
            resid = np.maximum(norm_sq[valid] - 2 * np.sum(w * (y[valid] @ self.basis.T), 1)
                               + np.sum((w @ (self.basis @ self.basis.T)) * w, 1), 0)
            out.loc[valid, self.programmes] = usage
            out.loc[valid, "relative_error"] = np.sqrt(resid / norm_sq[valid])
        out["within_reference_q95"] = out["relative_error"].le(self.ref["q95_relative_error"])
        return out


_REF = None


def score_programmes(adata, layer=None, copy=False):
    """Add programme usage (gdT_P1 … gdT_P10), gdT_relative_error and gdT_within_q95 to `adata.obs`."""
    global _REF
    if _REF is None:
        _REF = ProgrammeReference()
    res = _REF.project(adata, layer=layer)
    adata = adata.copy() if copy else adata
    for p in _REF.programmes:
        adata.obs["gdT_" + p] = res[p].values
    adata.obs["gdT_relative_error"] = res["relative_error"].values
    adata.obs["gdT_within_q95"] = res["within_reference_q95"].values
    adata.uns.setdefault("gdtprofiler", {})["reference_gene_coverage"] = res.attrs["reference_gene_coverage"]
    return adata if copy else None
