"""Example data shipped with gdTProfiler."""
from importlib import resources


def spleen_example():
    """1,060 spleen T and NK cells from donor A31 of the Cross-tissue Immune Cell Atlas.

    Raw counts are in ``.X``. ``obs`` holds the authors' annotation
    (``author_annotation``) and the receptor-defined identity and Vδ1/Vδ2 subtype
    from paired TCR sequencing (``receptor_identity``, ``receptor_subtype``).
    Source: Domínguez Conde et al., Science 376, eabl5197 (2022), CC BY 4.0.
    """
    import anndata as ad
    with resources.as_file(resources.files("gdtprofiler").joinpath("data", "example", "spleen_A31.h5ad")) as path:
        return ad.read_h5ad(path)
