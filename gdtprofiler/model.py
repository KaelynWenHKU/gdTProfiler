"""Joint softmax model: αβ T / γδ T / NK identity and Vδ1/Vδ2 subtype.

Inputs are raw UMI counts. The model uses

* a fixed scVI encoder (posterior mean, 20 dimensions) over 2,000 non-TCR genes;
* nine TCR genes, normalized to 10^4 counts within a 2,009-gene panel and
  log-transformed;
* a 29-feature standardization, a shared 20→20 ReLU layer, an identity head
  (softmax over αβ T, γδ T and NK) and a subtype head (sigmoid, Vδ1 vs Vδ2).
"""
from importlib import resources
import json

import numpy as np
import pandas as pd
from scipy import sparse

CLASSES = ("alpha_beta", "gamma_delta", "NK")
TCR_GENES = ("TRAC", "TRBC1", "TRBC2", "TRDC", "TRGC1", "TRGC2", "TRDV1", "TRDV2", "TRDV3")


def _resource(name):
    return resources.files("gdtprofiler").joinpath("data", "softmax_v1", name)


class SoftmaxModel:
    """The trained gdTProfiler model (version softmax_v1)."""

    def __init__(self):
        with _resource("encoder.npz").open("rb") as f:
            enc = np.load(f, allow_pickle=False)
            self.encoder = {k: enc[k] for k in enc.files}
        self.genes = [str(g) for g in self.encoder.pop("genes")]
        self.panel_genes = [str(g) for g in self.encoder.pop("panel_genes")]
        weights = json.loads(_resource("weights.json").read_text())
        p = weights["parameters"]
        self.W_shared = np.asarray(p["shared.weight"], np.float32)
        self.b_shared = np.asarray(p["shared.bias"], np.float32)
        self.W_root = np.asarray(p["root.weight"], np.float32)
        self.b_root = np.asarray(p["root.bias"], np.float32)
        self.W_child = np.asarray(p["child.weight"], np.float32)
        self.b_child = np.asarray(p["child.bias"], np.float32)
        scaler = json.loads(_resource("input_scaler.json").read_text())
        self.mean = np.asarray(scaler["mean"], np.float64)
        self.scale = np.asarray(scaler["scale"], np.float64)
        self.tcr_index = [self.panel_genes.index(g) for g in TCR_GENES]

    # ------------------------------------------------------------------ inputs
    @staticmethod
    def _counts(adata, layer):
        x = adata.X if layer is None else adata.layers[layer]
        x = sparse.csr_matrix(x, dtype=np.float32)
        if x.nnz and ((x.data < 0).any() or not np.allclose(x.data, np.rint(x.data))):
            raise ValueError("gdTProfiler expects raw, non-negative integer UMI counts; "
                             "pass the raw-count layer with `layer=`.")
        return x

    @staticmethod
    def _align(x, var_names, genes):
        """Map input columns onto `genes` by symbol; duplicated symbols are summed."""
        position = {}
        for i, g in enumerate(map(str, var_names)):
            position.setdefault(g, []).append(i)
        rows, cols = [], []
        for j, g in enumerate(genes):
            for i in position.get(g, []):
                rows.append(i)
                cols.append(j)
        m = sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)),
                              shape=(x.shape[1], len(genes)))
        found = np.zeros(len(genes), bool)
        found[cols] = True
        return (x @ m).tocsr(), found

    def _encode(self, counts, batch_size=2048):
        e = self.encoder
        out = []
        for start in range(0, counts.shape[0], batch_size):
            h = np.log1p(counts[start:start + batch_size].toarray().astype(np.float32))
            for layer in (0, 1):
                h = h @ e[f"l{layer}_W"].T + e[f"l{layer}_b"]
                h = (h - e[f"l{layer}_rm"]) / np.sqrt(e[f"l{layer}_rv"] + 1e-3) * e[f"l{layer}_g"] + e[f"l{layer}_beta"]
                h = np.maximum(h, 0)
            out.append(h @ e["mean_W"].T + e["mean_b"])
        return np.vstack(out)

    def _tcr(self, panel_counts):
        total = np.asarray(panel_counts.sum(1)).ravel()
        x = panel_counts.multiply(1.0 / np.where(total > 0, total / 1e4, 1.0)[:, None]).tocsr()
        x.data = np.log1p(x.data)
        return x[:, self.tcr_index].toarray(), total

    # ------------------------------------------------------------- prediction
    def predict(self, adata, layer=None, threshold=0.5):
        """Score every cell in `adata`.

        Parameters
        ----------
        adata : anndata.AnnData
            Cells × genes with gene symbols as `var_names` and raw UMI counts.
        layer : str, optional
            Layer holding raw counts; defaults to `adata.X`.
        threshold : float
            Cut-off on p(γδ) used for the `gd_call` column (default 0.5, as in the manuscript).

        Returns
        -------
        pandas.DataFrame indexed like `adata.obs_names` with p_alpha_beta, p_gamma_delta,
        p_NK, predicted_class, gd_call and q_Vdelta1 (subtype score; meaningful for γδ T cells).
        """
        x = self._counts(adata, layer)
        enc_counts, enc_found = self._align(x, adata.var_names, self.genes)
        panel_counts, panel_found = self._align(x, adata.var_names, self.panel_genes)
        z = self._encode(enc_counts)
        t, _ = self._tcr(panel_counts)
        u = ((np.column_stack([z, t]) - self.mean) / self.scale).astype(np.float32)
        h = np.maximum(u[:, :20] @ self.W_shared.T + self.b_shared, 0)
        logits = np.column_stack([h, u[:, 20:]]) @ self.W_root.T + self.b_root
        logits -= logits.max(1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(1, keepdims=True)
        q = 1 / (1 + np.exp(-(h @ self.W_child.T + self.b_child).ravel()))
        res = pd.DataFrame(p, index=adata.obs_names, columns=["p_" + c for c in CLASSES])
        res["predicted_class"] = np.asarray(CLASSES)[p.argmax(1)]
        res["gd_call"] = res["p_gamma_delta"] >= threshold
        res["q_Vdelta1"] = q
        res.attrs["encoder_gene_coverage"] = float(enc_found.mean())
        res.attrs["missing_TCR_genes"] = [g for g, i in zip(TCR_GENES, self.tcr_index) if not panel_found[i]]
        return res


_MODEL = None


def predict(adata, layer=None, threshold=0.5, copy=False):
    """Predict identity and subtype and store the results in `adata.obs`.

    Adds the columns gdT_p_alpha_beta, gdT_p_gamma_delta, gdT_p_NK, gdT_class,
    gdT_call and gdT_q_Vdelta1. Returns the AnnData if ``copy=True``.
    """
    global _MODEL
    if _MODEL is None:
        _MODEL = SoftmaxModel()
    res = _MODEL.predict(adata, layer=layer, threshold=threshold)
    adata = adata.copy() if copy else adata
    for col, new in [("p_alpha_beta", "gdT_p_alpha_beta"), ("p_gamma_delta", "gdT_p_gamma_delta"),
                     ("p_NK", "gdT_p_NK"), ("predicted_class", "gdT_class"),
                     ("gd_call", "gdT_call"), ("q_Vdelta1", "gdT_q_Vdelta1")]:
        adata.obs[new] = res[col].values
    adata.uns["gdtprofiler"] = {"model": "softmax_v1", "threshold": threshold,
                                "encoder_gene_coverage": res.attrs["encoder_gene_coverage"],
                                "missing_TCR_genes": res.attrs["missing_TCR_genes"]}
    if res.attrs["missing_TCR_genes"]:
        import warnings
        warnings.warn("TCR genes absent from the input: " + ", ".join(res.attrs["missing_TCR_genes"])
                      + ". Identity predictions are unreliable without TRDV/TRGC genes.")
    return adata if copy else None
