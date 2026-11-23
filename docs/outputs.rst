Outputs
=======

:func:`gdtprofiler.predict` adds to ``adata.obs``:

* ``gdT_p_alpha_beta``, ``gdT_p_gamma_delta``, ``gdT_p_NK``: class probabilities.
* ``gdT_class``: most probable class.
* ``gdT_call``: ``gdT_p_gamma_delta`` ≥ 0.5, the threshold used in the paper.
* ``gdT_q_Vdelta1``: Vδ1 score (values near 0 indicate Vδ2). Interpret it only for γδ T cells.

It also records ``encoder_gene_coverage`` and ``missing_TCR_genes`` in
``adata.uns["gdtprofiler"]``.

:func:`gdtprofiler.score_programmes` adds:

* ``gdT_P1``, ``gdT_P2``, ``gdT_P3``, ``gdT_P4``, ``gdT_P5``, ``gdT_P7``,
  ``gdT_P9``, ``gdT_P10``: relative programme usage (rows sum to 1).
* ``gdT_relative_error``: relative reconstruction error of the programme fit.
* ``gdT_within_q95``: whether the error is within the 95th percentile of the
  reference (0.924). Usage of cells outside this range should be interpreted
  with caution.

The programmes are labelled as in the paper. Their leading genes include
NKG7, FGFBP2, GZMH and PRF1 (P2, cytotoxic); NFKBIA, CD69 and FOS (P5,
immediate-early); HSPA1A, DNAJB1 and HSP90AA1 (P7, heat shock); and CDC20,
CCNB2 and PLK1 (P9, cell cycle).
