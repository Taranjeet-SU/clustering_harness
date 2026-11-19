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
be a cache miss and abort the run with a diff. The embedding vectors of the reference run were supplied as
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

## 2. Live rerun

See the results section appended below (live models, embeddings recomputed by the package).

## 3. Stage conformance tests

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
