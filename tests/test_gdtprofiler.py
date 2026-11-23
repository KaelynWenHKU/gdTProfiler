import numpy as np
import pandas as pd
import pytest
import anndata as ad
from scipy import sparse

import gdtprofiler as g


@pytest.fixture(scope="module")
def adata():
    model = g.SoftmaxModel()
    ref = g.ProgrammeReference()
    genes = sorted(set(model.genes) | set(model.panel_genes) | set(ref.ref["universe"]))
    rng = np.random.default_rng(0)
    x = sparse.random(30, len(genes), density=0.1, random_state=1, data_rvs=lambda n: rng.integers(1, 20, n)).tocsr()
    a = ad.AnnData(x.astype(np.float32), var=pd.DataFrame(index=genes))
    a.obs_names = [f"cell{i}" for i in range(a.n_obs)]
    return a


def test_predict_outputs(adata):
    res = g.SoftmaxModel().predict(adata)
    p = res[["p_alpha_beta", "p_gamma_delta", "p_NK"]].to_numpy()
    assert res.shape[0] == adata.n_obs
    assert np.allclose(p.sum(1), 1, atol=1e-5)
    assert ((res.q_Vdelta1 >= 0) & (res.q_Vdelta1 <= 1)).all()
    assert res.attrs["encoder_gene_coverage"] == 1.0
    assert res.attrs["missing_TCR_genes"] == []


def test_predict_writes_obs(adata):
    b = g.predict(adata, copy=True)
    for col in ["gdT_p_alpha_beta", "gdT_p_gamma_delta", "gdT_p_NK", "gdT_class", "gdT_call", "gdT_q_Vdelta1"]:
        assert col in b.obs
    assert "gdT_p_gamma_delta" not in adata.obs


def test_programmes(adata):
    res = g.ProgrammeReference().project(adata)
    usage = res[["P1", "P2", "P3", "P4", "P5", "P7", "P9", "P10"]]
    ok = usage.notna().all(axis=1)
    assert ok.any()
    assert np.allclose(usage[ok].sum(axis=1), 1)


def test_missing_tcr_warns(adata):
    b = adata[:, ~adata.var_names.isin(["TRDV1", "TRDV2", "TRDV3"])].copy()
    with pytest.warns(UserWarning, match="TRDV"):
        g.predict(b)


def test_rejects_normalized_input(adata):
    b = adata.copy()
    b.X = b.X * 0.5
    with pytest.raises(ValueError):
        g.predict(b)


def test_example_dataset_and_plots():
    import pytest
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import gdtprofiler.plotting as pl
    a = g.datasets.spleen_example()
    assert a.n_obs == 1060 and "receptor_identity" in a.obs
    g.predict(a)
    g.score_programmes(a)
    ref = a.obs.receptor_identity.isin(["alpha_beta", "gamma_delta"])
    from sklearn.metrics import average_precision_score
    ap = average_precision_score(a.obs.receptor_identity[ref] == "gamma_delta", a.obs.gdT_p_gamma_delta[ref])
    assert ap > 0.98
    pl.identity_scores(a, "receptor_identity")
    pl.class_composition(a, "author_annotation")
    pl.programme_usage(a, "gdT_class")
