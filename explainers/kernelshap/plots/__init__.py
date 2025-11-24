try:
    import matplotlib  # noqa: F401
except ImportError:
    raise ImportError(
        "matplotlib is not installed so plotting is not available! Run `pip install matplotlib` to fix this."
    )

from src.proposed.explainers.kernelshap.plots._bar import bar
from src.proposed.explainers.kernelshap.plots._beeswarm import beeswarm
from src.proposed.explainers.kernelshap.plots._benchmark import benchmark
from src.proposed.explainers.kernelshap.plots._decision import decision
from src.proposed.explainers.kernelshap.plots._embedding import embedding
from src.proposed.explainers.kernelshap.plots._force import force, initjs
from src.proposed.explainers.kernelshap.plots._group_difference import group_difference
from src.proposed.explainers.kernelshap.plots._heatmap import heatmap
from src.proposed.explainers.kernelshap.plots._image import image, image_to_text
from src.proposed.explainers.kernelshap.plots._monitoring import monitoring
from src.proposed.explainers.kernelshap.plots._partial_dependence import partial_dependence
from src.proposed.explainers.kernelshap.plots._scatter import scatter
from src.proposed.explainers.kernelshap.plots._text import text
from src.proposed.explainers.kernelshap.plots._violin import violin
from src.proposed.explainers.kernelshap.plots._waterfall import waterfall

__all__ = [
    "bar",
    "beeswarm",
    "benchmark",
    "decision",
    "embedding",
    "force",
    "initjs",
    "group_difference",
    "heatmap",
    "image",
    "image_to_text",
    "monitoring",
    "partial_dependence",
    "scatter",
    "text",
    "violin",
    "waterfall",
]