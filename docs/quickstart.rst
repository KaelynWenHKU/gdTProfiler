Quick start
===========

The input is an ``AnnData`` object with raw UMI counts and gene symbols as
``var_names``.

.. code-block:: python

    import anndata as ad
    import gdtprofiler as gdt

    adata = ad.read_h5ad("my_data.h5ad")
    gdt.predict(adata, layer="counts")
    gdt.score_programmes(adata, layer="counts")

``layer`` names the layer holding raw counts; omit it if the counts are in
``adata.X``. gdTProfiler raises an error for normalized or log-transformed
values.

A complete example with receptor-confirmed cells is in the
`tutorial notebook <https://github.com/KaelynWenHKU/gdTProfiler/blob/main/tutorial/gdTProfiler_tutorial.ipynb>`_.

Input requirements
------------------

* Gene symbols that match those used by 10x Genomics references (for example
  ``TRDC``, ``TRDV2``). Duplicated symbols are summed.
* The nine TCR genes TRAC, TRBC1, TRBC2, TRDC, TRGC1, TRGC2, TRDV1, TRDV2 and
  TRDV3. gdTProfiler warns when any are missing. Without TRDV and TRGC genes,
  identity predictions are unreliable.
* At least 80% of the 2,000 programme reference genes for programme scoring.
