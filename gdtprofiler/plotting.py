"""Plots of gdTProfiler results (requires matplotlib)."""
import numpy as np
import pandas as pd

PROGRAMMES = ["P1", "P2", "P3", "P4", "P5", "P7", "P9", "P10"]
CLASS_COLOURS = {"alpha_beta": "#3A86FF", "gamma_delta": "#FF7F11", "NK": "#8338EC"}
CLASS_LABELS = {"alpha_beta": "αβ T", "gamma_delta": "γδ T", "NK": "NK"}


def _plt():
    import matplotlib.pyplot as plt
    return plt


def identity_scores(adata, groupby, ax=None, color="#FF7F11"):
    """Distribution of p(γδ) in each group of `groupby` (for example a reference label)."""
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(4.2, 3))
    groups = list(pd.unique(adata.obs[groupby]))
    rng = np.random.default_rng(0)
    for i, g in enumerate(groups):
        y = adata.obs.loc[adata.obs[groupby] == g, "gdT_p_gamma_delta"].to_numpy()
        ax.scatter(i + rng.uniform(-0.3, 0.3, len(y)), y, s=4, alpha=0.8, color=color, lw=0)
    ax.axhline(adata.uns.get("gdtprofiler", {}).get("threshold", 0.5), color="black", lw=0.8, ls="--")
    ax.set_xticks(range(len(groups)), groups, rotation=30, ha="right")
    ax.set_ylabel("p(γδ T)")
    ax.set_ylim(-0.03, 1.03)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    return ax


def class_composition(adata, groupby, ax=None):
    """Fraction of cells assigned to αβ T, γδ T and NK within each group."""
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(4.2, 3))
    tab = pd.crosstab(adata.obs[groupby], adata.obs["gdT_class"], normalize="index")
    tab = tab.reindex(columns=[c for c in CLASS_COLOURS if c in tab.columns])
    left = np.zeros(len(tab))
    for c in tab.columns:
        ax.barh(range(len(tab)), tab[c], left=left, color=CLASS_COLOURS[c], label=CLASS_LABELS[c], height=0.7)
        left += tab[c].to_numpy()
    ax.set_yticks(range(len(tab)), tab.index)
    ax.invert_yaxis()
    ax.set_xlabel("fraction of cells")
    ax.legend(frameon=False, bbox_to_anchor=(1, 1), loc="upper left")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    return ax


def programme_usage(adata, groupby, cells=None, ax=None, cmap="YlGnBu"):
    """Mean programme usage per group, for cells within the reference error range."""
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(4.8, 0.45 * adata.obs[groupby].nunique() + 0.9))
    keep = adata.obs["gdT_within_q95"].astype(bool)
    if cells is not None:
        keep &= np.asarray(cells, bool)
    cols = ["gdT_" + p for p in PROGRAMMES]
    mean = adata.obs.loc[keep].groupby(groupby, observed=True)[cols].mean()
    im = ax.imshow(mean.to_numpy(), cmap=cmap, aspect="auto", vmin=0)
    ax.set_xticks(range(len(PROGRAMMES)), PROGRAMMES)
    ax.set_yticks(range(len(mean)), mean.index)
    ax.figure.colorbar(im, ax=ax, label="mean usage", fraction=0.04)
    return ax
