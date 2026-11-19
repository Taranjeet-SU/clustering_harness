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

* **Reference parity.** On a private 1,231-ticket support dataset the package reproduces the reference
  implementation exactly: replaying the reference run's LLM answers, every prompt the package sent was
  byte-identical (zero cache misses over 1,692 LLM calls) and every stage output — 151 bank partitions, base
  partition, cascades A and B, both residual passes, cluster geometry, names, scores and report lines — was
  identical. Details in [docs/validation.md](docs/validation.md).
* **Unit and conformance tests** check each stage against the written specification (`tests/`).
* **Public benchmark** results on BANKING77 are in [docs/validation.md](docs/validation.md).

## License

Apache-2.0. Dataset loaders download third-party data under its own license (see `clustering_harness.datasets.DATASET_INFO`).
