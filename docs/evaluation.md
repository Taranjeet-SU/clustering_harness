# Evaluation

`clustering_harness.evaluation` gives three kinds of measurement:

- **External metrics** compare a predicted partition with gold labels.
- **Internal metrics** need only the vectors.
- **Stability** measures how well several runs agree with each other.

Every metric below follows a published definition. Wherever scikit-learn implements a metric, the harness
calls scikit-learn. It implements purity and BCubed itself, using the sparse contingency matrix, so they cost
O(n + nnz) instead of O(n²).

## Label convention

A partition is a 1-D sequence with one label per record, in dataset order. Labels can be any hashable value.
These values count as **noise** (unassigned):

- a negative integer (normally `-1`);
- `None` or `NaN`;
- `""` or a string made only of whitespace.

`normalize_labels(labels)` maps any partition to integers `0..k-1`, numbered in order of first appearance, with
noise set to `-1`.

## Notation

- n is the number of records evaluated.
- The gold classes are L_1..L_m. The predicted clusters are C_1..C_k.
- n_ij = |L_i ∩ C_j| is the contingency count. a_i = |L_i| and b_j = |C_j| are the class and cluster sizes.
- H(·) is entropy and I(·;·) is mutual information, both computed from the contingency table.

## Noise handling for external metrics

Records whose **gold** label is missing are always dropped and counted in `n_gold_missing`. For example,
clinc150 out-of-scope queries are dropped if you leave their label empty.

**Predicted** noise is handled by the `noise=` policy:

| policy | meaning | when to use |
|---|---|---|
| `"singleton"` (default) | Each noise record becomes its own cluster. | Conservative standard. Leaving records unassigned is penalised in recall-type scores (inverse purity, BCubed recall, ARI). Purity itself *rises*, because every singleton is perfectly pure, so compare purity only between partitions with similar coverage (on BANKING77, UMAP+HDBSCAN scores 0.879 with singletons, 0.864 with noise excluded, 0.778 with noise as one cluster). |
| `"cluster"` | All noise records form one extra cluster. | Matches treating "noise" as a catch-all bucket. Penalises precision-type scores. |
| `"exclude"` | Noise records are removed from both partitions before scoring. | Quality of the clustered part only. **Always report it together with `coverage`.** |

Every result also contains these fields:

- `coverage`: the fraction of gold-labelled records with a non-noise prediction;
- `n_clusters`: the number of non-noise predicted clusters;
- `n_noise`, `n`, `n_evaluated`, `n_classes` and `noise_policy`.

## External metrics (`external_metrics`)

| key | definition | range | reference |
|---|---|---|---|
| `ari` | Adjusted Rand Index: (RI − E[RI]) / (max RI − E[RI]) under the permutation (hypergeometric) model. Computed from Σ C(n_ij,2), Σ C(a_i,2), Σ C(b_j,2). | [−0.5, 1]; 0 means chance level | Hubert & Arabie (1985). sklearn `adjusted_rand_score` |
| `ami` | Adjusted Mutual Information: (I − E[I]) / (mean(H(L),H(C)) − E[I]), using the **arithmetic** mean (sklearn default). | ≤ 1; 0 means chance level | Vinh, Epps & Bailey (2010). sklearn `adjusted_mutual_info_score` |
| `nmi` | Normalised Mutual Information: I(L;C) / ((H(L)+H(C))/2) | [0, 1] | Strehl & Ghosh (2002); Manning et al. (2008) §16.3. sklearn `normalized_mutual_info_score` |
| `homogeneity` | 1 − H(L\|C)/H(L). It equals 1 when every cluster contains a single class. | [0, 1] | Rosenberg & Hirschberg (2007). sklearn `homogeneity_score` |
| `completeness` | 1 − H(C\|L)/H(C). It equals 1 when every class sits in a single cluster. | [0, 1] | Rosenberg & Hirschberg (2007). sklearn `completeness_score` |
| `v_measure` | Harmonic mean of homogeneity and completeness. With β = 1 it equals NMI (arithmetic). | [0, 1] | Rosenberg & Hirschberg (2007). sklearn `v_measure_score` |
| `fowlkes_mallows` | TP / √((TP+FP)(TP+FN)) over record pairs | [0, 1] | Fowlkes & Mallows (1983). sklearn `fowlkes_mallows_score` |
| `purity` | (1/n) Σ_j max_i n_ij | (0, 1] | Zhao & Karypis (2001); Manning et al. (2008) §16.3 |
| `inverse_purity` | (1/n) Σ_i max_j n_ij | (0, 1] | Amigó et al. (2009) |
| `purity_f` | Harmonic mean of purity and inverse purity | (0, 1] | Harmonic mean of the two scores above |
| `bcubed_precision` | (1/n) Σ_records \|C(r)∩L(r)\| / \|C(r)\| = (1/n) Σ_ij n_ij² / b_j | (0, 1] | Bagga & Baldwin (1998); Amigó et al. (2009) |
| `bcubed_recall` | (1/n) Σ_records \|C(r)∩L(r)\| / \|L(r)\| = (1/n) Σ_ij n_ij² / a_i | (0, 1] | Bagga & Baldwin (1998); Amigó et al. (2009) |
| `bcubed_f1` | Harmonic mean of BCubed precision and recall. This is Van Rijsbergen's F with α = 0.5, as recommended by Amigó et al. | (0, 1] | Amigó et al. (2009) |

These metrics behave differently in a few ways:

- **Splitting and merging.** Purity and BCubed precision are trivially 1 when every record is its own cluster.
  Inverse purity and BCubed recall are trivially 1 when there is only one cluster. Always read the pairs
  together, or use the F scores.
- **BCubed constraints.** BCubed is the only family here that satisfies all four formal constraints of Amigó et
  al. (2009): homogeneity, completeness, rag bag and cluster size versus quantity.
- **Chance correction.** ARI and AMI are corrected for chance. Prefer them when you compare runs with very
  different numbers of clusters. NMI and V-measure grow with k under random labelling.

The helpers `purity_scores(t, p)` and `bcubed_scores(t, p)` expose the purity and BCubed metrics on their own.
They expect noise to be resolved already. Any remaining noise is treated as one cluster.

## Internal metrics (`internal_metrics`)

All internal metrics are computed **on non-noise records only**. Noise rows are removed first. The result
reports `coverage`, `n_clusters` and `n_evaluated` so that you can see how much data was scored. A metric is
`NaN` when it is undefined, which happens when there are fewer than 2 clusters or as many clusters as records.

| key | definition | range | reference |
|---|---|---|---|
| `silhouette` | Mean over records of (b − a) / max(a, b). Here a is the mean distance to the record's own cluster and b is the mean distance to the nearest other cluster. It uses `metric` (default `"cosine"`). The score is exact unless `sample_size` is given. In that case it is the mean over a random subsample drawn with `random_state`. | [−1, 1]; higher is better | Rousseeuw (1987). sklearn `silhouette_score` |
| `davies_bouldin` | Mean over clusters of max_{j≠i} (s_i + s_j) / d(c_i, c_j). Here s is the mean distance to the centroid and d is the distance between centroids. | ≥ 0; lower is better | Davies & Bouldin (1979). sklearn `davies_bouldin_score` |
| `calinski_harabasz` | [tr(B)/(k−1)] / [tr(W)/(n−k)], the ratio of between-cluster to within-cluster dispersion | ≥ 0; higher is better | Caliński & Harabasz (1974). sklearn `calinski_harabasz_score` |

Davies–Bouldin and Calinski–Harabasz are Euclidean by definition. When `metric="cosine"`, the harness computes
them on L2-normalised vectors so that they match cosine geometry.

## Tables

- `contingency_table(t, p)` returns a gold × predicted count `DataFrame`. Noise and missing labels appear as
  `"<noise>"`.
- `cluster_label_report(t, p)` returns one row per predicted cluster, sorted by size, with these columns:
  - `size`;
  - `majority_label` and `majority_count`;
  - `purity`, equal to majority_count / size;
  - `n_labels`, the number of distinct gold labels;
  - `is_noise`. The noise group appears as a single row unless you pass `include_noise=False`.

## Stability (`evaluation.stability`)

Use these functions on several runs over the same records, such as different seeds, samples or settings.

- **`pairwise_agreement(partitions, names, metric="ari"|"ami"|"nmi", noise="singleton")`** returns a symmetric
  R × R `DataFrame` with an agreement score for every pair of runs. The noise policy is applied to **both**
  partitions in each pair. With `"exclude"`, only records that are non-noise in both runs are compared.
- **`agreement_summary(matrix)`** returns the mean, std, min and max of the off-diagonal entries, plus `n_runs`
  and `n_pairs`.
- **`coassignment_matrix(partitions, dense=False)`** returns the consensus matrix of Monti et al. (2003). Its
  entry C[i, j] is the fraction of runs in which records i and j share a cluster.
  - Noise never co-clusters. This includes the diagonal, so C[i, i] is the fraction of runs in which record i
    was assigned to a cluster.
  - The result is sparse (CSR) by default. Its size grows with Σ cluster size².
- **`consensus_stability(partitions)`** returns a stability score for each record.
  - In run r, the co-members S_r(i) of record i are the other records in its cluster.
  - The agreement of record i in run r is a_r(i) = mean_{j∈S_r(i)} C[i, j]. This is Monti et al.'s *item
    consensus*, evaluated against each run's own clustering.
  - The record's stability is the mean of a_r(i) over the runs where it is non-noise and has at least one
    co-member. It is `NaN` for records that are never in a cluster of size ≥ 2.
  - A score of 1.0 means the record always lands with exactly the same co-members. Low values mark records
    whose grouping depends on the run.

## References

- Amigó, E., Gonzalo, J., Artiles, J., Verdejo, F. (2009). A comparison of extrinsic clustering evaluation metrics based on formal constraints. *Information Retrieval* 12(4), 461–486.
- Bagga, A., Baldwin, B. (1998). Entity-based cross-document coreferencing using the vector space model. *COLING-ACL 1998*.
- Caliński, T., Harabasz, J. (1974). A dendrite method for cluster analysis. *Communications in Statistics* 3(1), 1–27.
- Davies, D. L., Bouldin, D. W. (1979). A cluster separation measure. *IEEE TPAMI* 1(2), 224–227.
- Fowlkes, E. B., Mallows, C. L. (1983). A method for comparing two hierarchical clusterings. *JASA* 78(383), 553–569.
- Hubert, L., Arabie, P. (1985). Comparing partitions. *Journal of Classification* 2(1), 193–218.
- Manning, C. D., Raghavan, P., Schütze, H. (2008). *Introduction to Information Retrieval*, ch. 16. Cambridge University Press.
- Monti, S., Tamayo, P., Mesirov, J., Golub, T. (2003). Consensus clustering. *Machine Learning* 52, 91–118.
- Rosenberg, A., Hirschberg, J. (2007). V-Measure: a conditional entropy-based external cluster evaluation measure. *EMNLP-CoNLL 2007*.
- Rousseeuw, P. J. (1987). Silhouettes. *Journal of Computational and Applied Mathematics* 20, 53–65.
- Strehl, A., Ghosh, J. (2002). Cluster ensembles. *JMLR* 3, 583–617.
- Vinh, N. X., Epps, J., Bailey, J. (2010). Information theoretic measures for clusterings comparison. *JMLR* 11, 2837–2854.
- Zhao, Y., Karypis, G. (2001). Criterion functions for document clustering. Technical Report 01-40, University of Minnesota.
- scikit-learn user guide, "Clustering performance evaluation" (`sklearn.metrics.cluster`).
