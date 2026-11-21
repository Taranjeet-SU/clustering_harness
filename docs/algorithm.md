# Algorithm specification

This document specifies every stage of the pipeline as implemented in `clustering_harness`. Each rule names
the configuration field that controls it (`PipelineConfig.<section>.<field>`); defaults are the
`reference` preset (calibrated on ~15k support tickets). The `small` preset (calibrated on ~1.2k tickets)
changes only the size-dependent values listed in [the presets table](#presets).

Terminology: *eligible* records are the records not quarantined by the junk rule. *Slot a* and *slot b* are
the two LLMs; both act as judges, and each additionally performs fixed roles (`config.roles`). A *reasoning*
model (`LLM(reasoning=True)`) spends hidden reasoning tokens from its generation budget; the pipeline gives it
larger budgets and text-mode judging.

---

## 1. Records and junk quarantine — `data.py`, `config.junk`, `config.text`

* Each record has a unique string id, an optional title and a text (description). Ids are never used as
  clustering content.
* A description is **empty** if it has fewer than `junk.min_desc_alnum` (3) letters/digits (any script; a CJK character counts as 3) or equals the
  title (case- and whitespace-insensitive).
* A record is **junk** if its title matches `junk.placeholder_title` *and* its description is empty, or if
  title + description contain fewer than `junk.min_alnum` (12) letters/digits. Junk records are kept in every
  output with status `junk` and never clustered.
* Exact duplicates are kept as separate records; scoring reports duplicate shares.
* Text shown to LLMs is **masked** (`masking.py`: URLs, e-mails, IPv4, phone numbers, host names, configured
  person names) and whitespace-collapsed; titles are cut to `text.title_chars` (120). Embeddings use the
  unmasked text rendered with `text.embed_with_title` / `text.embed_without_title`.

## 2. Embeddings — `embeddings.py`

* Any `Embedder` returning one L2-normalised vector per record (row order preserved). Reference:
  `Alibaba-NLP/gte-large-en-v1.5`, CLS pooling, max 8,192 tokens, length-sorted token-budget batching,
  float16 storage, re-normalised in float32.
* A *label embedder* (reference `BAAI/bge-large-en-v1.5`, CPU) embeds short labels and titles for
  canonicalisation pre-grouping, provisional-label retrieval and rollup neighbourhoods.

## 3. Consensus bank — `algorithms/partitions.py`, `config.bank`

Exact cosine neighbours (`faiss.IndexFlatIP`, top `bank.knn` = 51 including self) of the eligible vectors are
computed once. Partitions:

| family | grid | count (reference) |
|---|---|---|
| UMAP + HDBSCAN | UMAP(`umap_components`=10, cosine, min_dist 0, precomputed neighbours) for each `umap_neighbors` × `seeds`; for each `hdbscan_min_samples` one HDBSCAN tree at `tree_min_cluster_size`; labels for each `selections` × `min_cluster_sizes` | 2×3×2×2×4 = 96 |
| MiniBatchKMeans (raw vectors) | `kmeans_k` × `seeds`, batch `kmeans_batch_size`, `kmeans_n_init` | 5×3 = 15 |
| PCA + HDBSCAN | PCA(`pca_components`) ; `hdbscan_min_samples` × `selections` × `min_cluster_sizes` | 16 |
| Leiden, mutual-kNN graph | `leiden_k` × `leiden_resolutions` × `seeds`; edge weight max(1e-8, cosine); RBConfiguration | 24 |

Every partition: clusters with fewer than `bank.min_cluster` (10) members become noise; labels are re-indexed
by descending size. A partition is **eligible for consensus** when its cluster count lies in
`bank.n_clusters_range` (60–400), its largest cluster is at most `bank.largest_frac_max` (0.12) of the eligible
records, and its noise share is at most `bank.noise_frac_max` (0.40).

## 4. Base partition — `config.base`

The base is the UMAP + HDBSCAN bank member with (`umap_neighbors`=30, `seed`=0, `min_samples`=15,
`selection`=eom, `min_cluster_size`=50). Base clusters are named `P<label>`.

## 5. Two-judge calibrated gate — `algorithms/gate.py`, `config.gate`

For candidate cluster *c*, each judge scores `gate.reps` (3) seeded samples (seed from md5 of
`G|c|rep|0`) of up to `gate.sample_size` (8) members, showing title + first `gate.desc_chars` (260) description
characters, on the 1–5 "one report line?" scale defined by the prompt (granularity contract + three anchor
examples). Budgets: `gate.max_tokens` (300), reasoning models `gate.max_tokens_reasoning` (900) with one retry
at `gate.retry_tokens_reasoning` (2400) on a contract failure. Scores are rounded and clamped to 1–5.

Calibration is frozen from the first gated batch (the base parents): per-judge mean μⱼ and population sd σⱼ
(σ = 0 → 1), pooled mean μ and sd σ. Adjusted score: `adj = μ + σ · (score − μⱼ) / σⱼ`.

* **ACCEPT** if mean(adj) ≥ `accept_mean` (3.6) and each judge's mean(adj) ≥ `accept_min_judge` (3.2);
* **REJECT** if mean(adj) ≤ `reject_mean` (3.0);
* otherwise (or if all calls failed) **ESCALATE**: one more sample from both judges, re-decide; if still in the
  band, ACCEPT iff mean(adj) ≥ `escalate_accept` (3.4). `CONTRACT_FAIL` if no valid score exists.

The cluster name is the first non-empty name among valid calls in (judge a, judge b) × replicate order.

## 6. Cascade S1 and S2 — `algorithms/cascade.py`, `config.cascade`

* **S1** gates every base parent and freezes the calibration.
* **S2** splits each parent that was not accepted, and each parent with at least `large_parent` members
  (fraction of eligible records if < 1, else an absolute size). `local_split`: no children below
  `split_min_size` (30) members; otherwise UMAP(`split_umap_components`=5, cosine, neighbours
  min(`split_neighbors`, n−1), min_dist 0, seed) + HDBSCAN leaf with min_cluster_size
  max(`split_min_child`, n // `split_mcs_divisor`); keep children with ≥ `split_min_child` (10) members; need at
  least two; return the `split_cap` (6) largest as `P<k>.<i>`. Children are gated.
* A rejected parent is replaced by its accepted children. An accepted large parent is replaced only if its
  accepted children cover ≥ `replace_min_coverage` (60%) of it **and** their size-weighted adjusted score
  exceeds the parent's.

## 7. Cascade A (side output)

A copy of the S2 result is reassigned with the rule of step 11 and written as `cascade_A`. Cascade B restarts
from the S2 result; cascade A never feeds later stages (`cascade.run_cascade_a` toggles it).

## 8. S2b — consensus cores and depth-two splits

* For each S1-rejected parent with ≥ `cores_min_parent` (20) members: co-association = share of eligible bank
  partitions putting two members in the same non-noise cluster; build the graph of pairs with
  co-association ≥ `coassociation` (0.8); its **connected components** with ≥ `core_min_size` (10) members are
  candidate cores, skipping any with Jaccard ≥ `core_max_jaccard` (0.6) against an accepted S2 child, at most
  `cores_per_parent` (4) per parent (`P<k>.core<j>`).
* Rejected S2 children with ≥ `deep_split_min` (100) members are split again (`local_split`, cap
  `deep_split_cap` = 4).
* All candidates are gated. In descending adjusted-score order, already-taken members are removed and a
  candidate is added only if ≥ `candidate_min_size` (10) members remain.

## 9. S3B — Leiden communities in the residual

For the eligible records still unassigned: scan cached neighbour columns 1…`leiden_scan` (30) and keep the
first `leiden_k` (15) that are also residual; keep mutual edges, weight = cosine; Leiden RBConfiguration at
`leiden_resolution` (1.0), seed. Communities with ≥ `leiden_min_size` (10) members, the `leiden_max_gated` (40)
largest, are gated (`L<i>`); accepted ones are added. The yield rule (accepted ≥ 3% of residual and ≥ 10% of
gated) is logged only.

## 10. P1–P2 — sampled pruning

For each accepted cluster the labeller (slot b) writes a report line + inclusion criterion from the gate
sample. `prune_sample` (10) members are sampled (`random.Random(prune_seed=11)`); the verifier (slot a)
checks them against the line, criterion and three central exemplars (avoiding sampled members); the rechecker
(slot b) re-checks only explicit "no" answers. A sampled member is ejected only if **both** say no (missing
answers are not "no"). The cluster is dissolved if 1 − ejected/sampled < `prune_min_precision` (0.5).
Unsampled members are not individually verified.

## 11. P3–P4 — centroid reassignment and merges

**Reassignment.** With unit centroids of the accepted clusters and all members' own-centroid cosines:
τ_hi = `reassign_hi_percentile` (25th) and τ_lo = `reassign_lo_percentile` (5th) percentile. For each residual
record with best / second-best centroid cosine s₁ / s₂:

* **auto-add** if s₁ ≥ τ_hi and s₁ − s₂ ≥ `reassign_margin` (0.01);
* **band** if not auto and s₁ ≥ τ_lo (this includes high-similarity records that fail the margin).

Per destination, up to `band_sample` (8) band records (`random.Random(band_seed=7)`) are verified; if at least
`band_admit_min` (4) were sampled and ≥ `band_admit_fraction` (75%) say yes, the destination's whole band is
admitted, otherwise only the sampled yes records. Clusters that grew by more than `growth_regate` (25%) are
re-gated with their additions; additions are reverted unless ACCEPT.

**Merges.** Centroid pairs with cosine ≥ `merge_min_cosine` (0.85), the `merge_top_pairs` (15) most similar,
are shown to both judges ("same report line?"); disjoint pairs where both say yes are unioned, re-gated and
merged only on ACCEPT. Output: `results/cascade_B`.

## 12–15. Residual classification (first pass) — `algorithms/residual.py` (`variant="v1"`), `config.residual`

All cascade-B residual records are shuffled (`random.Random(residual.seed)`) and sharded into
`residual.streams` (4) streams with KMeans on their vectors. Each stream is processed sequentially in batches
of `batch_size` (15): the classifier (slot a) sees the `near_lines` (30) accepted lines nearest the batch
centroid, the `near_provisional` (30) provisional labels of this stream nearest the batch titles (label
embedder), and title + `desc_chars` (220) description characters per record, and answers per record: an
accepted id `L<n>`, a provisional id `N<n>`, `NEW: <label>` (labels containing one of the whole words of
`generic_label` — general, misc, other, various, … — are dropped) or `JUNK: <reason>`. A record answered twice in
one batch keeps its first answer. New labels join the stream's provisional list immediately.

1. Provisional labels are pre-grouped (label-embedding cosine, average linkage, distance `canon_distance`
   0.2) and canonicalised by slot b in chunks of `canon_chunk` (30).
2. Canonical labels with ≥ `min_label_size` (4) distinct records are gated (`N<i>`).
3. Filings under existing lines are confirmed by the verifier in batches of `confirm_batch` (10) with three
   central exemplars.
4. Accepted labels still holding ≥ 4 unconfirmed records become clusters `CN<i>`, pruned as in step 10
   (exemplars: the first non-sampled members).
5. Remaining records are reassigned as in step 11 (exemplars: the first three members) and grown clusters
   re-gated.

## 16. Refile pass — `variant="v2"`

Every record the first pass left unassigned (including its LLM-junk records) is re-classified against all
clusters with the corrected rules (`classify_rules_v2`: recurring service requests are real report lines;
JUNK only for records without usable content), using `residual.refile_streams` streams, a repair pass that
re-classifies records of failed batches in batches of `repair_batch_size` (5), id-sorted orderings, label ids
`RN<i>` / clusters `CRN<i>` and placements prefixed `refiled_`. Steps 1–5 above are repeated.

## 17. Naming and scoring — `algorithms/scoring.py`, `config.scoring`

* **Geometry** per cluster (exact): mean cosine silhouette of members (scikit-learn `silhouette_samples`, cosine,
  assigned records only), mean pairwise cosine (self-pairs excluded), mean and 10th-percentile centroid cosine,
  mean share of each member's 10 nearest eligible neighbours in the same cluster (unassigned = disagreement),
  centroid margin (own − best other centroid cosine) and the share of negative margins, exact-duplicate counts.
* **Naming** (slot `scoring.name_with`, default b): 12 member excerpts (2 central, 2 boundary, 2 diverse,
  seeded random; unique texts) and 2 excerpts from each of the two nearest clusters → name, inclusion and
  exclusion criteria, rationale (schema-validated, 3 attempts in total; each failed attempt is fed back as a correction).
* **Review**: both models score a 12-case panel (naming cases excluded where possible) for coherence, label
  fit and distinction (1–5) and a decision. A second panel (follow-up model only) is drawn if the cluster has
  ≥ `escalate_size` (100) members, any score < 4, a non-ACCEPT decision, a suspicious member or a judge
  disagreement ≥ 1; a third only if the disagreement persists. Sampling stops when all unique texts were seen.
  The semantic score is the mean of the two models' means (equal weight). `quality_status` is
  `REVIEW_RECOMMENDED` if any member was flagged, either model's mean < 4 or any decision ≠ ACCEPT.

## 18. Cluster export

`output/assignments.csv` (one row per input record), `output/clusters.csv` (scores).

## 19. Guarded report-line rollup — `algorithms/rollup.py`, `config.rollup`

Each cluster is judged with the unified protocol (see `docs/evaluation.md`: 3 seeded samples of 8 × 2
judges; judge-balanced score). Round 0 (always run) proposes merging clusters whose normalised names are identical. While
there are more than `target_max` (150) lines, rounds 1…5 group lines by average linkage on
1 − (0.5 · name cosine + 0.5 · centroid cosine) at `thresholds` (0.22 … 0.46; groups larger than `max_group`
are split tighter) and the rollup model partitions each group into lines. **Guard**: a proposed line is kept
only if its judge-balanced score is ≥ `guard_min` (3.5) and ≥ the size-weighted mean of its parts −
`guard_drop` (0.5). Lines are named (name + inclusion criterion) and numbered `LN001…` by size.

## 20. Final export

`output/assignments.csv` gains `line_id`/`line_name`, `output/report_lines.csv`, `output/summary.json`.

---

## Presets

| field | `reference` | `small` |
|---|---|---|
| `bank.hdbscan_min_samples` | 5, 15 | 5, 10 |
| `bank.tree_min_cluster_size` | 25 | 6 |
| `bank.min_cluster_sizes` | 25, 50, 100, 200 | 6, 8, 10, 20 |
| `bank.kmeans_k` | 100…400 | 35…55 |
| `bank.leiden_k` / `leiden_resolutions` | 15, 30 / 0.25…2 | 10, 20 / 0.5…2 |
| `bank.min_cluster` | 10 | 5 |
| `bank.n_clusters_range` | 60–400 | 15–80 |
| `base.min_samples` / `min_cluster_size` | 15 / 50 | 5 / 8 |
| `cascade.large_parent` | 1% of eligible | 30 |
| `cascade.split_min_size` / `split_min_child` | 30 / 10 | 20 / 5 |
| `cascade.cores_min_parent` / `core_min_size` | 20 / 10 | 10 / 5 |
| `cascade.deep_split_min` | 100 | 40 |
| `cascade.candidate_min_size` / `leiden_min_size` | 10 / 10 | 5 / 5 |
| `residual.refile_streams` | 24 | 4 |

## Determinism

All sampling is seeded; partitions are deterministic for fixed library versions. LLM answers are cached by
exact request (model, messages, budget, seed, JSON mode, temperature), so a rerun or resumed run reproduces
earlier decisions exactly; providers that ignore seeds (e.g. Claude) are reproducible only through the cache.
