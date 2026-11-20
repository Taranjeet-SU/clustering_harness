# clustering_harness

LLM-verified clustering of support tickets (and other short texts) into **recurring issue types** and
**report lines**, with built-in evaluation.

Plain embedding clustering finds groups that *look* similar; a support lead needs groups that are *one
recurring problem or request*. `clustering_harness` combines a consensus bank of density, centroid and graph
partitions with a calibrated two-LLM gate, an LLM classifier for the records clustering leaves behind,
sampled verification, evidence-grounded naming and a guarded roll-up into report lines — and it measures the
result with standard metrics, multi-run stability and LLM-as-judge evaluation.

* **Pluggable models**: Ollama, OpenAI and any OpenAI-compatible server (vLLM, LM Studio, TGI), Anthropic
  Claude, or your own function. Two model slots; each can be a different provider.
* **Pluggable embeddings**: any Hugging Face encoder, sentence-transformers, OpenAI, Ollama or precomputed vectors.
* **Not tied to one domain**: prompts are YAML prompt sets with domain variables and replaceable anchor
  examples; every algorithm threshold is a named config field with presets for ~1k and ~15k records.
* **Reproducible and cost-controlled**: every LLM answer is cached by exact request (reruns and resumed runs
  replay decisions), spend is recorded per model and capped (`budget_usd`).
* **Evaluation included**: ARI / AMI / NMI / V-measure / purity / BCubed against gold labels, silhouette and
  cluster geometry, pairwise stability across seeds, a two-judge LLM coherence evaluation that scores any
  partition (including baselines), and standard baseline clusterers.

## Install

```bash
pip install "clustering-harness[all]"        # or: pip install -e ".[all]" from a clone
```

Core dependencies: numpy, pandas, scikit-learn, umap-learn, hdbscan, faiss-cpu, igraph, leidenalg, pydantic.
Extras: `embeddings` (torch, transformers, sentence-transformers), `openai`, `anthropic`, `datasets`.

## Quick start

### Python

```python
from clustering_harness import Dataset, Pipeline, PipelineConfig
from clustering_harness.llm import LLM, OllamaBackend, OpenAICompatibleBackend, Ledger
from clustering_harness.embeddings import HFTransformerEmbedder, SentenceTransformerEmbedder

ds = Dataset.from_csv("tickets.csv", id_col="id", text_col="body", title_col="subject")
config = PipelineConfig.auto(len(ds), prompts="default",
                             domain={"product": "Acme Cloud", "product_description": "a SaaS platform with integrations"})
ledger = Ledger("~/.cache/clustering_harness/spend.sqlite")
llms = {"a": LLM(OllamaBackend("qwen3:8b"), name="qwen3"),
        "b": LLM(OpenAICompatibleBackend("gpt-4o-mini"), name="gpt-4o-mini", max_concurrency=8, budget_usd=5, ledger=ledger)}
pipe = Pipeline(config, llms=llms, embedder=HFTransformerEmbedder("Alibaba-NLP/gte-large-en-v1.5"),
                label_embedder=SentenceTransformerEmbedder("BAAI/bge-large-en-v1.5"), workdir="runs/acme")
result = pipe.run(ds)
result.clusters        # one row per cluster: name, criteria, size, geometry, semantic scores, review status
result.assignments     # one row per input record: cluster, report line, status, placement
result.summary         # coverage, mean scores, LLM usage
```

### Command line

```bash
clustering-harness init --preset small > run.yaml      # edit data / models / domain
clustering-harness run run.yaml
clustering-harness compare run.yaml --baselines kmeans:k=50 umap_hdbscan:min_cluster_size=10 --judge --judge-max-clusters 40
clustering-harness multirun run.yaml --seeds 0 1 2
clustering-harness evaluate --assignments runs/x/output/assignments.csv --gold gold.csv --id-col id --label-col label
```

## How it works

```
records ─ junk rule ─ embeddings ─┬─ consensus bank (151 partitions: UMAP+HDBSCAN, PCA+HDBSCAN, k-means, Leiden)
                                  └─ base partition ─ S1 two-judge gate ─ S2 local splits ─ S2b consensus cores
                                     ─ S3B Leiden on residual ─ sampled pruning ─ centroid reassignment ─ merges
                                     ─ LLM residual classification ─ refile pass ─ naming + scoring ─ report-line rollup
```

Every stage, threshold and its config field is specified in [docs/algorithm.md](docs/algorithm.md).
Model roles, backends and budgets: [docs/backends.md](docs/backends.md). Metrics and LLM-judge protocol:
[docs/evaluation.md](docs/evaluation.md). Validation: [docs/validation.md](docs/validation.md).

## Outputs (`<workdir>/output/`)

| file | content |
|---|---|
| `assignments.csv` | every input record: `cluster_id`, `cluster_name`, `status` (`cluster` / `unassigned` / `junk`), `placement` (how it got there), `reason`, `line_id`, `line_name` |
| `clusters.csv` | per cluster: name, inclusion/exclusion criteria, size, exact geometry (silhouette, cohesion, margin, 10-NN agreement, duplicates), two-model semantic, label-fit and distinction scores, suspicious sample ids, review status |
| `report_lines.csv` | report lines with member clusters, criterion and judge score |
| `summary.json` | coverage, score means, LLM usage |

Intermediate results (bank partitions, cascade logs, residual streams, gate calibration, LLM cache, evidence
shown to models) stay under `<workdir>/results` and `<workdir>/cache`; a rerun resumes from them.

## Evaluation

```python
from clustering_harness.evaluation import external_metrics, internal_metrics, pairwise_agreement
from clustering_harness.baselines import run_baseline
from clustering_harness.multirun import compare_partitions, eval_context

parts = {"pipeline": result.labels(), "kmeans": run_baseline("kmeans", X, k=50)}
table = compare_partitions(parts, X=X, gold=ds.labels,
                           judge_ctx=eval_context(ds, X, llms, "runs/eval"), judge_max_clusters=40)
```

## Validation

Full method and tables: [docs/validation.md](docs/validation.md).

### 1. Exact parity with the reference implementation

The package was extracted from a set of research scripts that had been run on a private dataset of 1,231
support tickets with two local models (a 27B Qwen model and `gpt-oss:20b`). Every LLM request of that run was
traced and replayed through the package with an offline backend, so any prompt that was not byte-identical
would have aborted the run.

| stage | result |
|---|---|
| consensus bank (151 partitions, recomputed) | 151 / 151 identical |
| base partition, cascades A and B, residual and refile passes | identical |
| scoring (38 columns per cluster), rollup, final assignments | identical |
| LLM requests | 1,692, zero cache misses |

Headline: 34 clusters, 919 / 1,231 records assigned, mean two-model semantic score 3.909. The replay is
rerun after every change to the algorithm modules.

### 2. Live rerun on the same data (nothing reused)

Embeddings recomputed by the package on a different device and every model call made live. The
embeddings are numerically almost identical (cosine ≥ 0.9999, 99.8% identical nearest neighbours), but
UMAP + HDBSCAN is sensitive to such differences, so this run tests relevance, not identity.

| metric | reference | package, live |
|---|---|---|
| clusters | 34 | 33 |
| records assigned | 919 (77.5% of eligible) | 888 (74.9%) |
| mean semantic score | 3.909 | **4.109** |
| size-weighted semantic score | 3.890 | **4.022** |
| clusters scoring ≥ 4 | 59% | **76%** |
| label fit / distinction | 3.87 / 4.03 | 4.08 / 4.18 |

Relevance is equal or better on every quality measure, at 2.6 points lower coverage. The two partitions
agree on 81% of record statuses (clustered / unassigned / junk) and have NMI 0.71 on records clustered in
both: the same structure, though individual clusters differ.

### 3. Stage conformance tests

`tests/unit/test_*_conformance.py` check every stage against the specification in
[docs/algorithm.md](docs/algorithm.md) with deterministic fake models: gate calibration and decision
bands, bank grid and post-processing, split / core / Leiden / pruning / reassignment / merge rules,
residual classifier parsing, scoring geometry against brute-force definitions, and the rollup guard. The
metrics are tested against scikit-learn and textbook examples. 233 tests pass in CI.

The review found and fixed these robustness issues in the reference logic:
- model booleans such as `"false"`/`"no"` were read as *yes*;
- `null` names were accepted;
- malformed rollup answers crashed the run;
- a record answered twice in one batch took the last answer instead of the first;
- `NEW`/`JUNK` were matched without a colon;
- duplicate texts that were unassigned elsewhere counted as cross-cluster duplicates;
- residual placements were mislabelled;
- a zero calibration spread was not handled;
- an empty eligible bank failed;
- model reviews returning ticket ids outside the cluster failed the run instead of being sanitised.

None of these fixes changes memberships, names or scores on the reference data. Behaviours kept for
fidelity are listed in the validation doc.

### 4. Public benchmark: BANKING77

Test split (3,080 queries, 77 gold intents), `small` preset, domain-neutral prompts, `gpt-4o-mini` in both
model slots. The pipeline is **not** told the number of intents; baselines that need *k* get the true 77.
Unassigned records count as singletons.

| partition | clusters | coverage | ARI | NMI | purity |
|---|---|---|---|---|---|
| **pipeline** | 135 | 0.981 | 0.615 | **0.857** | 0.844 |
| k-means, oracle k = 77 | 77 | 1.000 | 0.608 | 0.845 | 0.751 |
| UMAP + HDBSCAN | 103 | 0.894 | 0.634 | 0.855 | 0.879 |
| agglomerative, k = 77 | 77 | 1.000 | 0.507 | 0.835 | 0.575 |
| Leiden, mutual-kNN | 49 | 0.991 | 0.482 | 0.825 | 0.529 |

The pipeline matches the best baselines without knowing *k*, assigns 98% of queries, and splits more
finely than the gold intents. Over three seeds: NMI 0.856 ± 0.001, ARI 0.602 ± 0.009, pairwise agreement
between runs ARI 0.760 ± 0.013.

### Caveats

- **Lenient judges.** With `gpt-4o-mini` as both judges, the LLM coherence evaluation is near ceiling for
  every partition (gold 4.98, pipeline 4.98, k-means 4.93), so it cannot separate good from mediocre
  clusters on short, clean queries. Use a stricter judge pair when the LLM evaluation has to
  discriminate. A run with local open-weight judges is in progress.
- **Run-to-run variation.** Results depend on embedding numerics and model non-determinism. Compare
  runs with `clustering-harness multirun` rather than expecting identical partitions.
- **Not yet validated:** Anthropic models (the backend is implemented and unit-tested, with no
  end-to-end benchmark yet).

**Cost.** The BANKING77 run made about 3,600 `gpt-4o-mini` requests (≈ $0.50, 13 minutes). All
benchmarking so far cost $2.11.

## License

Apache-2.0. Dataset loaders download third-party data under its own license (see `clustering_harness.datasets.DATASET_INFO`).
