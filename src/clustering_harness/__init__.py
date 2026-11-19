"""clustering_harness — LLM-verified clustering of support tickets and other short texts.

Core objects:

* :class:`Dataset` — your records (id, optional title, text, optional gold label).
* :class:`PipelineConfig` — every algorithm threshold; presets ``reference`` (~15k records) and ``small`` (~1k).
* :class:`Pipeline` — embeds, builds a consensus bank, cascades, classifies the residual with LLMs, scores and
  rolls clusters up into report lines. Models and embedders are pluggable.
* :mod:`clustering_harness.evaluation` — external/internal metrics, stability, LLM-judge evaluation.
* :mod:`clustering_harness.baselines` — standard reference clusterers.
"""
from .config import PipelineConfig
from .data import Dataset, JunkRule, TextTemplate
from .pipeline import ClusteringResult, Pipeline
from .prompts import PromptSet

__version__ = "0.1.0"
__all__ = ["Dataset", "JunkRule", "TextTemplate", "PipelineConfig", "Pipeline", "ClusteringResult", "PromptSet", "__version__"]
