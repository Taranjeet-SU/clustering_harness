"""Evaluation: external / internal clustering metrics and run-to-run stability."""
from .metrics import (
    NOISE,
    apply_noise_policy,
    bcubed_scores,
    cluster_label_report,
    contingency_table,
    external_metrics,
    internal_metrics,
    normalize_labels,
    purity_scores,
)
from .stability import agreement_summary, coassignment_matrix, consensus_stability, pairwise_agreement

__all__ = [
    "NOISE",
    "agreement_summary",
    "apply_noise_policy",
    "bcubed_scores",
    "cluster_label_report",
    "coassignment_matrix",
    "consensus_stability",
    "contingency_table",
    "external_metrics",
    "internal_metrics",
    "normalize_labels",
    "pairwise_agreement",
    "purity_scores",
]
