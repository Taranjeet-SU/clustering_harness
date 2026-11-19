"""Pipeline configuration.

Every threshold of the algorithm is a named field. Two calibrated presets ship:

* :meth:`PipelineConfig.reference` — the settings of the reference run on ~15k records (cluster minimum
  10, base HDBSCAN min_cluster_size 50 / min_samples 15, consensus grid sized for 60–400 clusters).
* :meth:`PipelineConfig.small` — the same algorithm rescaled for ~1k records (cluster minimum 5, base
  min_cluster_size 8 / min_samples 5, grid sized for 15–80 clusters, 4 refile streams).

:meth:`PipelineConfig.auto` picks one by data size. Configs round-trip through YAML/JSON.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field

__all__ = ["PipelineConfig", "JunkConfig", "TextConfig", "BankConfig", "BaseSelection", "GateConfig", "CascadeConfig",
           "ResidualConfig", "ScoringConfig", "RollupConfig", "RoleMap", "MaskingConfig"]


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


class JunkConfig(_M):
    placeholder_title: str = (r"^\s*(a new case|an existing case|new case|existing case)\s*\d*\s*$|^\s*test(ing)?\s*\d*\s*$|"
                              r"chat initiated but no information was shared")
    min_alnum: int = 12
    min_desc_alnum: int = 3


class TextConfig(_M):
    embed_with_title: str = "Title: {title}\nDescription: {text}"
    embed_without_title: str = "{text}"
    title_chars: int = 120


class MaskingConfig(_M):
    enabled: bool = True
    names: list[str] = Field(default_factory=list)
    requester_prefix: Optional[str] = None


class BankConfig(_M):
    """Consensus bank of alternative partitions (step 3) and their hard constraints."""
    knn: int = 51
    umap_components: int = 10
    umap_neighbors: list[int] = [15, 30]
    seeds: list[int] = [0, 1, 2]
    hdbscan_min_samples: list[int] = [5, 15]
    tree_min_cluster_size: int = 25
    selections: list[str] = ["eom", "leaf"]
    min_cluster_sizes: list[int] = [25, 50, 100, 200]
    kmeans_k: list[int] = [100, 150, 200, 300, 400]
    kmeans_batch_size: int = 4096
    kmeans_n_init: int = 3
    pca_components: int = 20
    leiden_k: list[int] = [15, 30]
    leiden_resolutions: list[float] = [0.25, 0.5, 1.0, 2.0]
    min_cluster: int = 10
    n_clusters_range: tuple[int, int] = (60, 400)
    largest_frac_max: float = 0.12
    noise_frac_max: float = 0.40
    workers: int = 4


class BaseSelection(_M):
    """The fixed base partition (step 4) is the bank member with these settings."""
    umap_neighbors: int = 30
    seed: int = 0
    min_samples: int = 15
    selection: Literal["eom", "leaf"] = "eom"
    min_cluster_size: int = 50


class GateConfig(_M):
    """Two-judge calibrated gate (steps 5+)."""
    sample_size: int = 8
    reps: int = 3
    desc_chars: int = 260
    accept_mean: float = 3.6
    accept_min_judge: float = 3.2
    reject_mean: float = 3.0
    escalate_accept: float = 3.4
    max_tokens: int = 300
    max_tokens_reasoning: int = 900
    retry_tokens_reasoning: int = 2400


class CascadeConfig(_M):
    large_parent: float = 0.01          # < 1: fraction of eligible records; >= 1: absolute size
    split_min_size: int = 30
    split_umap_components: int = 5
    split_neighbors: int = 15
    split_mcs_divisor: int = 8
    split_min_child: int = 10
    split_cap: int = 6
    replace_min_coverage: float = 0.6
    cores_min_parent: int = 20
    coassociation: float = 0.8
    core_min_size: int = 10
    core_max_jaccard: float = 0.6
    cores_per_parent: int = 4
    deep_split_min: int = 100
    deep_split_cap: int = 4
    candidate_min_size: int = 10
    leiden_scan: int = 30
    leiden_k: int = 15
    leiden_resolution: float = 1.0
    leiden_min_size: int = 10
    leiden_max_gated: int = 40
    prune_sample: int = 10
    prune_seed: int = 11
    prune_min_precision: float = 0.5
    reassign_hi_percentile: float = 25
    reassign_lo_percentile: float = 5
    reassign_margin: float = 0.01
    band_sample: int = 8
    band_seed: int = 7
    band_admit_min: int = 4
    band_admit_fraction: float = 0.75
    growth_regate: float = 0.25
    merge_min_cosine: float = 0.85
    merge_top_pairs: int = 15
    run_cascade_a: bool = True
    seed: int = 0


class ResidualConfig(_M):
    """LLM residual classification (C_full) and the refile pass (C_full_v2)."""
    streams: int = 4
    refile_streams: int = 24
    refile: bool = True
    batch_size: int = 15
    repair_batch_size: int = 5
    near_lines: int = 30
    near_provisional: int = 30
    desc_chars: int = 220
    canon_distance: float = 0.2
    canon_chunk: int = 30
    min_label_size: int = 4
    confirm_batch: int = 10
    seed: int = 0
    generic_label: str = r"\b(general|misc|miscellaneous|other|various|multiple|unrelated|assorted|mixed)\b"


class ScoringConfig(_M):
    panel_size: int = 12
    max_panels: int = 3
    escalate_size: int = 100
    evidence_chars: int = 650
    naming_chars: int = 600
    competitor_chars: int = 300
    concurrency: int = 4
    name_with: Literal["a", "b"] = "b"


class RollupConfig(_M):
    enabled: bool = True
    thresholds: list[Optional[float]] = [None, 0.22, 0.28, 0.34, 0.40, 0.46]
    max_group: int = 30
    target_max: int = 150
    guard_min: float = 3.5
    guard_drop: float = 0.5
    census_arm: str = "clusters"      # judge-sample seeds are derived from these names
    guard_arm: str = "rollup_lines"


class RoleMap(_M):
    """Which model slot ("a" or "b") performs each LLM role. Both slots always judge."""
    classifier: Literal["a", "b"] = "a"
    canonicalizer: Literal["a", "b"] = "b"
    labeler: Literal["a", "b"] = "b"
    verifier: Literal["a", "b"] = "a"
    rechecker: Literal["a", "b"] = "b"
    rollup: Literal["a", "b"] = "a"


class PipelineConfig(_M):
    prompts: str = "default"
    domain: dict = Field(default_factory=dict)
    junk: JunkConfig = JunkConfig()
    text: TextConfig = TextConfig()
    masking: MaskingConfig = MaskingConfig()
    bank: BankConfig = BankConfig()
    base: BaseSelection = BaseSelection()
    gate: GateConfig = GateConfig()
    cascade: CascadeConfig = CascadeConfig()
    residual: ResidualConfig = ResidualConfig()
    scoring: ScoringConfig = ScoringConfig()
    rollup: RollupConfig = RollupConfig()
    roles: RoleMap = RoleMap()

    # ------------------------------------------------------------------ presets
    @classmethod
    def reference(cls, **kw) -> "PipelineConfig":
        return cls(**kw)

    @classmethod
    def small(cls, **kw) -> "PipelineConfig":
        c = cls(**kw)
        c.bank = c.bank.model_copy(update=dict(hdbscan_min_samples=[5, 10], tree_min_cluster_size=6, min_cluster_sizes=[6, 8, 10, 20],
                                               kmeans_k=[35, 40, 45, 50, 55], leiden_k=[10, 20], leiden_resolutions=[0.5, 1.0, 1.5, 2.0],
                                               min_cluster=5, n_clusters_range=(15, 80)))
        c.base = c.base.model_copy(update=dict(min_samples=5, min_cluster_size=8))
        c.cascade = c.cascade.model_copy(update=dict(large_parent=30, split_min_size=20, split_min_child=5, cores_min_parent=10,
                                                     core_min_size=5, deep_split_min=40, candidate_min_size=5, leiden_min_size=5))
        c.residual = c.residual.model_copy(update=dict(refile_streams=4))
        return c

    @classmethod
    def auto(cls, n_records: int, **kw) -> "PipelineConfig":
        """``small`` below 5,000 records, ``reference`` otherwise."""
        return cls.small(**kw) if n_records < 5000 else cls.reference(**kw)

    # ------------------------------------------------------------------ io
    @classmethod
    def from_file(cls, path: str | Path) -> "PipelineConfig":
        p = Path(path)
        data = yaml.safe_load(p.read_text()) if p.suffix in (".yaml", ".yml") else json.loads(p.read_text())
        preset = data.pop("preset", None)
        base = {"small": cls.small, "reference": cls.reference}.get(preset, cls)()
        return cls.model_validate(_deep_merge(base.model_dump(), data))

    def to_file(self, path: str | Path) -> None:
        p = Path(path)
        d = self.model_dump(mode="json")
        p.write_text(yaml.safe_dump(d, sort_keys=False) if p.suffix in (".yaml", ".yml") else json.dumps(d, indent=1))

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.model_dump(mode="json"), sort_keys=True).encode()).hexdigest()[:16]


def _deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out
