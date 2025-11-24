from src.proposed.explainers.kernelshap._explanation import Cohorts, Explanation

# explainers
from src.proposed.explainers.kernelshap._kernel import KernelExplainer

try:
    # Version from setuptools-scm
    from ._version import version as __version__
except ImportError:
    # Expected when running locally without build
    __version__ = "0.0.0-not-built"

_no_matplotlib_warning = (
    "matplotlib is not installed so plotting is not available! Run `pip install matplotlib` to fix this."
)


# plotting (only loaded if matplotlib is present)
def unsupported(*args, **kwargs):
    raise ImportError(_no_matplotlib_warning)


class UnsupportedModule:
    def __getattribute__(self, item):
        raise ImportError(_no_matplotlib_warning)

'''
try:
    import matplotlib  # noqa: F401

    have_matplotlib = True
except ImportError:
    have_matplotlib = False
if have_matplotlib:
    from src.proposed.explainers.kernelshap import plots
    from src.proposed.explainers.kernelshap.plots._bar import bar_legacy as bar_plot
    from src.proposed.explainers.kernelshap.plots._beeswarm import summary_legacy as summary_plot
    from src.proposed.explainers.kernelshap.plots._decision import decision as decision_plot
    from src.proposed.explainers.kernelshap.plots._decision import multioutput_decision as multioutput_decision_plot
    from src.proposed.explainers.kernelshap.plots._embedding import embedding as embedding_plot
    from src.proposed.explainers.kernelshap.plots._force import force as force_plot
    from src.proposed.explainers.kernelshap.plots._force import getjs, initjs, save_html
    from src.proposed.explainers.kernelshap.plots._group_difference import group_difference as group_difference_plot
    from src.proposed.explainers.kernelshap.plots._heatmap import heatmap as heatmap_plot
    from src.proposed.explainers.kernelshap.plots._image import image as image_plot
    from src.proposed.explainers.kernelshap.plots._monitoring import monitoring as monitoring_plot
    from src.proposed.explainers.kernelshap.plots._partial_dependence import partial_dependence as partial_dependence_plot
    from src.proposed.explainers.kernelshap.plots._scatter import dependence_legacy as dependence_plot
    from src.proposed.explainers.kernelshap.plots._text import text as text_plot
    from src.proposed.explainers.kernelshap.plots._violin import violin as violin_plot
    from src.proposed.explainers.kernelshap.plots._waterfall import waterfall as waterfall_plot
else:
    bar_plot = unsupported
    summary_plot = unsupported
    decision_plot = unsupported
    multioutput_decision_plot = unsupported
    embedding_plot = unsupported
    force_plot = unsupported
    getjs = unsupported
    initjs = unsupported
    save_html = unsupported
    group_difference_plot = unsupported
    heatmap_plot = unsupported
    image_plot = unsupported
    monitoring_plot = unsupported
    partial_dependence_plot = unsupported
    dependence_plot = unsupported
    text_plot = unsupported
    violin_plot = unsupported
    waterfall_plot = unsupported
    # If matplotlib is available, then the plots submodule will be directly available.
    # If not, we need to define something that will issue a meaningful warning message
    # (rather than ModuleNotFound).
    plots = UnsupportedModule()  # type: ignore


# other stuff :)
#from src.proposed.utils import datasets, links, utils  # noqa: E402
#from .actions._optimizer import ActionOptimizer  # noqa: E402
'''
# from . import benchmark
from .utils._legacy import kmeans  # noqa: E402

# Use __all__ to let type checkers know what is part of the public API.
__all__ = [
    "Cohorts",
    "Explanation",
    # Explainers
    "KernelExplainer",
    # Plots
    "plots",
    "bar_plot",
    "summary_plot",
    "decision_plot",
    "multioutput_decision_plot",
    "embedding_plot",
    "force_plot",
    "getjs",
    "initjs",
    "save_html",
    "group_difference_plot",
    "heatmap_plot",
    "image_plot",
    "monitoring_plot",
    "partial_dependence_plot",
    "dependence_plot",
    "text_plot",
    "violin_plot",
    "waterfall_plot",
    # Other stuff
    "links",
    "utils",
    "kmeans",
]
"""
    "ActionOptimizer",
    "approximate_interactions",
    "sample",
        "req.py",
"""