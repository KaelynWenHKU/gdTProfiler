import os, shutil, sys
sys.path.insert(0, os.path.abspath(".."))
project = "gdTProfiler"
author = "Ka Ning Wen, Yuanhua Huang, Sung Hee Park"
release = "1.0.0"
extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.autosummary", "nbsphinx"]

# Render the tutorial notebook (with its saved outputs) as a web page.
_here = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(_here, "tutorials"), exist_ok=True)
shutil.copy(os.path.join(_here, "..", "tutorial", "gdTProfiler_tutorial.ipynb"),
            os.path.join(_here, "tutorials", "gdTProfiler_tutorial.ipynb"))
nbsphinx_execute = "never"
exclude_patterns = ["_build", "**.ipynb_checkpoints"]
autodoc_mock_imports = ["anndata", "sklearn"]
html_theme = "sphinx_rtd_theme"
