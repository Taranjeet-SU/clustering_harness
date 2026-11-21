# Validation

## 1. Reference parity (exact)

**Question.** Does the package implement *exactly* the reference pipeline it was extracted from?

**Method.** The reference implementation (a set of research scripts) had been run end to end on a private
dataset of 1,231 support tickets with two local models (a 27B Qwen model as slot a and `gpt-oss:20b` as slot
b). We re-executed those scripts with every LLM call served from their own cache and instrumented each call
site to record the exact request (messages, budget, seed, JSON mode) and answer — 1,692 calls in total. The
re-execution reproduced all of the reference outputs byte for byte, which validates the trace.

The trace was then loaded into the package's response cache, and the package was run on the same records
with an `OfflineBackend` for both slots: any request that was not byte-identical to a reference request would
be a cache miss and abort the run with a diff. The original run cached answers that broke the JSON contract and
used cache keys without model settings, so the replay sets `cache_contract_failures=True` and
`legacy_cache_keys=True` (since 0.2 both are off by default). The embedding vectors of the reference run were supplied as
precomputed vectors; everything else (bank, UMAP, HDBSCAN, Leiden, k-means, splits, label embeddings,
geometry) was recomputed by the package.

**Result.**

| stage | compared | result |
|---|---|---|
| consensus bank | 151 partition label arrays | 151 / 151 identical |
| base partition | assignments (29 clusters, 734 records) | identical |
| cascade A / cascade B | cluster id, name, status, stage per record | identical |
| residual pass (v1) | cluster id, name, status, placement, reason per record | identical |
| refile pass (v2) | cluster id, name, status, placement, reason per record | identical |
| scoring | 38 columns per cluster: geometry, names, criteria, all semantic scores, panels, flags | identical (judge rationale prefixes now use the configured model names) |
| rollup | report lines (ids, names, criteria, scores, membership) | identical |
| final assignments | cluster id, cluster name, line id per record | identical |
| LLM requests | 1,692 | 0 cache misses (every prompt byte-identical) |

Headline numbers therefore match exactly: 34 clusters, 919 of 1,231 records assigned (74.7%), mean
two-model semantic score 3.909, size-weighted 3.890.

## 2. Live rerun (same data, everything recomputed)

**Question.** With nothing reused (embeddings recomputed by the package, every model call made live), is the
result as relevant as the reference run?

**Method.** Same 1,231 records, same two local models, `reference` preset, SearchUnify prompt preset. The only
differences from the reference are numerical: embeddings computed on a different device (cosine to the
reference vectors ≥ 0.9999, 99.8% identical nearest neighbours) and non-cached model answers. UMAP/HDBSCAN is
sensitive to such tiny changes, so the base partition differs and the run is a relevance test, not an identity test.

| metric | reference | package, live |
|---|---|---|
| clusters | 34 | 33 |
| records assigned (of 1,231) | 919 | 888 |
| coverage of eligible records | 0.775 | 0.749 |
| junk | 45 | 45 |
| mean two-model semantic score | 3.909 | 4.109 |
| size-weighted semantic score | 3.890 | 4.022 |
| share of clusters scoring ≥ 4 | 0.588 | 0.758 |
| label fit (mean) | 3.865 | 4.078 |
| distinction (mean) | 4.034 | 4.184 |
| silhouette (size-weighted) | 0.021 | 0.025 |

Agreement between the two partitions: status agreement (cluster / unassigned / junk) 0.81; on the 785 records
assigned in both, ARI 0.49 and NMI 0.71 (all records, noise as singletons: ARI 0.44, AMI 0.58).

Relevance is equal or better on every quality measure, at 2.6 points lower coverage. The partitions agree
on structure (NMI 0.71) but not cluster by cluster, as expected when the base partition comes from a
different (equally valid) UMAP/HDBSCAN solution. One cluster's review needed the ID-sanitising fallback
(a model kept returning ticket ids outside the cluster); this is recorded in its `limitation` field.
Wall-clock: about 2.5 h on CPU-offloaded local models.

## 3. Public benchmark: BANKING77

[BANKING77](https://huggingface.co/datasets/PolyAI/banking77) test split (CC-BY-4.0): 3,080 customer-support
queries, 77 gold intents, no titles. Pipeline: `small` preset, default (domain-neutral) prompts with the domain
set to a digital bank, gte-large-en-v1.5 embeddings, bge-large-en-v1.5 label embeddings, `gpt-4o-mini` in both
model slots. The pipeline is **not** told the number of intents; baselines that need *k* get the true k = 77.
External metrics use the conservative `singleton` noise policy (every unassigned record is its own cluster).

| partition | clusters | coverage | ARI | AMI | NMI | purity | BCubed F1 | silhouette |
|---|---|---|---|---|---|---|---|---|
| **pipeline** (clusters = report lines here) | 135 | 0.981 | **0.615** | 0.797 | **0.857** | 0.844 | 0.645 | 0.238 |
| k-means, k = 77 (oracle k) | 77 | 1.000 | 0.608 | 0.802 | 0.845 | 0.751 | 0.655 | 0.227 |
| agglomerative (cosine, average), k = 77 | 77 | 1.000 | 0.507 | 0.801 | 0.835 | 0.575 | 0.616 | 0.197 |
| UMAP + HDBSCAN (mcs 10) | 103 | 0.894 | 0.634 | 0.783 | 0.855 | 0.879 | 0.662 | 0.260 |
| HDBSCAN (mcs 10) | 31 | 0.296 | 0.217 | 0.334 | 0.732 | 0.904 | 0.285 | 0.392 |
| Leiden, mutual-kNN (res 1.0) | 49 | 0.991 | 0.482 | 0.793 | 0.825 | 0.529 | 0.589 | 0.179 |

Without knowing *k*, the pipeline matches the best baselines on ARI/NMI, assigns 98% of the queries, and has
higher purity than every full-coverage baseline. It splits more finely than the gold intents (135 vs 77
clusters), the expected direction for a method built to find report lines that a support lead would not merge.

**Stability (3 seeds, `clustering-harness multirun`):** NMI 0.856 ± 0.001, ARI 0.602 ± 0.009, purity
0.857 ± 0.012, coverage 0.975 ± 0.003, 138 ± 5 clusters; pairwise agreement between runs ARI 0.760 ± 0.013,
AMI 0.876 ± 0.003. `gpt-4o-mini` is not fully deterministic at temperature 0, so part of the variation comes
from the model rather than the seeds.

**Independent reproduction.** A separate agent with only the public repository reproduced this benchmark:
pipeline ARI 0.622, NMI 0.860, purity 0.856, coverage 0.977, 133 clusters; every baseline row identical;
LLM judge gold 4.98 / pipeline 4.99; three seeds NMI 0.855 ± 0.003, ARI 0.609 ± 0.013, pairwise ARI 0.755.
Single run 12.4 minutes and $0.49. Two of its findings changed the published setup: the example config now
matches this table (it previously put a different model in slot a), and the embedder now runs on CPU unless
a GPU is requested.

**One model in both slots.** Both slots ran `gpt-4o-mini`, so each two-judge decision was one model asked
twice. The gate and scores are still applied, but they are not independent opinions; see the LLM-judge
caveat below.

**LLM-judge caveat.** With `gpt-4o-mini` as both judges, the two-judge coherence evaluation is near ceiling for
every partition (gold labels 4.98, pipeline 4.98, k-means 4.93, Leiden 4.82 on 40 sampled clusters each), and
the gate's frozen calibration had a mean of 4.93 (sd 0.29): a lenient judge cannot separate good from mediocre
clusters on short, clean queries. Use a stricter judge pair (below) when the LLM evaluation is meant to
discriminate.

**Two different local models.** The same benchmark with a 27B Qwen model in slot a and `gpt-oss:20b`
(reasoning) in slot b, both served locally, everything else unchanged:

| partition | clusters | coverage | ARI | NMI | purity | LLM judge (40 clusters) |
|---|---|---|---|---|---|---|
| **pipeline** | 139 | 0.975 | 0.614 | 0.855 | 0.860 | 4.93 |
| k-means, k = 77 | 77 | 1.000 | 0.608 | 0.845 | 0.751 | 4.78 |
| UMAP + HDBSCAN | 103 | 0.894 | 0.634 | 0.855 | 0.879 | 4.96 |
| agglomerative, k = 77 | 77 | 1.000 | 0.507 | 0.835 | 0.575 | 4.68 |
| Leiden, mutual-kNN | 49 | 0.991 | 0.482 | 0.825 | 0.529 | 4.74 |
| gold labels | 77 | 1.000 | 1.000 | 1.000 | 1.000 | 4.92 |

The clustering is as good as with `gpt-4o-mini` (ARI 0.614 vs 0.615, NMI 0.855 vs 0.857), so the result
does not depend on one model judging twice. The two-model judge is somewhat stricter: its spread across
partitions is 0.28 against 0.16 with `gpt-4o-mini`, and it ranks the full-coverage baselines in the order the
gold metrics do. It is still near ceiling on these short, clean queries.

Cost of the single run: 3,613 gpt-4o-mini requests, 2.4 M input and 0.22 M output tokens (≈ $0.50 at list
price), 13 minutes wall-clock.

**Claude Haiku in both slots.** The same benchmark with `claude-haiku-5-5` (Anthropic API) in both slots,
everything else unchanged (package 0.2.0):

| partition | clusters | coverage | ARI | NMI | purity | LLM judge (40 clusters) |
|---|---|---|---|---|---|---|
| **pipeline clusters** | 156 | 0.986 | 0.552 | 0.845 | 0.842 | 4.75 |
| **pipeline report lines** | 110 | 0.986 | 0.605 | 0.859 | 0.808 | 4.73 |
| k-means, k = 77 | 77 | 1.000 | 0.608 | 0.845 | 0.751 | 4.39 |
| UMAP + HDBSCAN | 103 | 0.894 | 0.634 | 0.855 | 0.879 | 4.74 |
| agglomerative, k = 77 | 77 | 1.000 | 0.507 | 0.835 | 0.575 | 4.22 |
| Leiden, mutual-kNN | 49 | 0.991 | 0.482 | 0.825 | 0.529 | 4.33 |
| gold labels | 77 | 1.000 | 1.000 | 1.000 | 1.000 | 4.63 |

Haiku splits more finely than `gpt-4o-mini` (156 clusters against 135), which lowers the cluster-level ARI
(0.552 against 0.615) at a similar NMI and purity. Because there are more than 150 clusters, the rollup merges
them into 110 report lines, and at that level ARI and NMI (0.605, 0.859) match the `gpt-4o-mini` clusters. As a
judge of its own clusters Haiku is near ceiling (it scores the pipeline above the gold labels), so its LLM-judge
column does not discriminate. Cost: 5,990 requests, 5.4 M input and 0.82 M output tokens (≈ $0.95 at the
built-in list price, including the judge column), 12.4 minutes wall-clock.

## 4. Stage conformance tests

`tests/unit/test_*_conformance.py` test each stage against the written specification
([algorithm.md](algorithm.md)) with deterministic fake models: gate calibration and decision bands, bank
grid and post-processing, split / core / Leiden / pruning / reassignment / merge rules, residual classifier
parsing and passes, scoring geometry against brute-force definitions, rollup guard. Metric implementations
are tested against scikit-learn and textbook examples (`tests/unit/test_metrics.py`).

### Intentional deviations from the reference

Robustness fixes made after the conformance review change three bookkeeping fields of the reference run and
nothing else (memberships, names, scores and report lines stay identical, still with zero cache misses):

| field | change | records / clusters affected on the reference data |
|---|---|---|
| `placement` (first residual pass) | records that were filed under a line but not confirmed, then added by the centroid reassignment, are labelled `reassigned` (the reference said `filed_existing_confirmed`) | 13 records |
| `cases_with_exact_text_also_in_other_clusters` | an exact duplicate whose copy is unassigned or junk no longer counts as "in another cluster" | 2 clusters |
| model answers | `"belongs": "false"` / `"same_line": "no"` strings are read as *no* (the reference read any non-empty string as yes); `null` names are ignored; malformed rollup answers are skipped; a record answered twice in one classifier batch keeps its first answer; `NEW`/`JUNK` require a colon | none on the reference data |

Known behaviours kept for fidelity (documented, tested): gate scores use Python rounding while the census
judge truncates to an integer; a reasoning judge with an empty answer is retried by both the LLM wrapper and
the gate; the generic-label filter matches whole words; rollup round 0 (identical names) always runs.
