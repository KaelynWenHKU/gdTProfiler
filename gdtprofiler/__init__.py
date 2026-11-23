"""gdTProfiler: identification, Vδ1/Vδ2 subtyping and cell-state profiling of human γδ T cells."""
__version__ = "1.0.0"
from .model import SoftmaxModel, predict
from .programmes import ProgrammeReference, score_programmes
from . import datasets, plotting

__all__ = ["SoftmaxModel", "predict", "ProgrammeReference", "score_programmes",
           "datasets", "plotting", "__version__"]
