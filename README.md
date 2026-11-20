# clustering_harness

**Turn a pile of support tickets (or any short texts) into a short list of named, recurring issues, with
every assignment explained and the quality measured.**

You give it a few thousand records, for example tickets, chat messages, search queries or survey answers.
It gives back:

- **clusters**: groups that are *one recurring problem or request*, each with a name, an inclusion and
  exclusion rule, and quality scores;
- **report lines**: clusters merged further where a support lead would put them on one line of a "top issues"
  report;
- **a row per record**: which cluster it went to, how it got there and why, or why it was left out.

Example from the public BANKING77 dataset (3,080 banking app queries):

| report line | records | rule written by the model |
|---|---|---|
| ATM Cash Withdrawal Discrepancies | 68 | Tickets report issues related to receiving incorrect cash amounts from ATMs. |
| Transfer Issues and Failures | 59 | Tickets that report errors or failures during money transfers. |
| Card Freezing Due to Fraud Suspicions | 53 | Requests to freeze cards or accounts after unauthorised transactions. |

**Why not just k-means?** Embedding clustering groups texts that *look* similar. A support lead needs
groups that are *the same problem*: "card declined at checkout" and "card arrived damaged" look alike but
are different report lines. This package uses classic clustering to propose groups. It then lets two LLMs
check, split, name and fill them, and keeps a record of every decision so you can audit it.

---

## Contents

1. [Install](#1-install)
2. [Your first run (10 minutes)](#2-your-first-run-10-minutes)
3. [Key ideas in plain words](#3-key-ideas-in-plain-words)
4. [Loading your data (CSV, pandas, MongoDB, ...)](#4-loading-your-data)
5. [Choosing models (Ollama, OpenAI, Azure, Gemini, Anthropic, Bedrock, your own)](#5-choosing-models)
6. [Embeddings](#6-embeddings)
7. [Describing your domain](#7-describing-your-domain)
8. [Cost control: budgets and the ledger](#8-cost-control-budgets-and-the-ledger)
9. [Reading the outputs](#9-reading-the-outputs)
10. [How it works](#10-how-it-works)
11. [Evaluating and comparing](#11-evaluating-and-comparing)
12. [Troubleshooting](#12-troubleshooting)
13. [Validation](#validation)

---

## 1. Install

The package is not on PyPI yet; install it from GitHub. You need Python 3.10 or newer.

```bash
git clone https://github.com/Taranjeet-SU/clustering_harness.git
cd clustering_harness
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate

# No NVIDIA GPU? Install the CPU build of PyTorch first (saves about 3 GB of CUDA libraries):
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -e ".[all]"                                # everything
clustering-harness --help                              # check that it works
```

To install less, pick extras: `pip install -e ".[embeddings,openai,datasets]"`.

| extra | what it adds |
|---|---|
| `embeddings` | local embedding models (PyTorch, transformers, sentence-transformers) |
| `openai` | OpenAI, Azure OpenAI, Gemini and any OpenAI-compatible server |
| `anthropic` | Claude through the Anthropic API |
| `bedrock` | Amazon Bedrock (botocore) |
| `mongo` | loading records from MongoDB (pymongo) |
| `datasets` | public benchmark datasets from Hugging Face |
| `dev` | everything plus the test tools (`pytest -m "not network"`) |

Without cloning: `pip install "clustering-harness[all] @ git+https://github.com/Taranjeet-SU/clustering_harness.git"`.

## 2. Your first run (10 minutes)

The repository ships a ready-made run on a public dataset. All you need is an OpenAI API key; the run costs
about $0.50 and takes about 13 minutes on a laptop CPU.

```bash
export OPENAI_API_KEY=sk-...
clustering-harness -v run examples/banking77.yaml
```

`-v` prints each stage as it starts and finishes. When the run is done, open `runs/banking77/output/`:
`report_lines.csv` is the short list, `clusters.csv` has the details and `assignments.csv` has one row per
query. Then compare the result with classic clustering and the true labels:

```bash
clustering-harness compare examples/banking77.yaml --baselines kmeans:k=77 umap_hdbscan:min_cluster_size=10
```

This reruns nothing: every model answer is cached, so it only adds the baselines.

**Now with your own data.** Create a starter spec, edit the marked lines and run it:

```bash
clustering-harness init > my_run.yaml
clustering-harness -v run my_run.yaml
```

A run spec is a short YAML file:

```yaml
workdir: runs/my_first_run            # everything this run writes goes here; rerunning resumes from it
data:
  path: tickets.csv                   # .csv / .tsv / .json / .jsonl / .parquet (or a MongoDB source, section 4)
  id_col: id
  text_col: description
  title_col: subject                  # optional
pipeline:
  preset: auto                        # sensible settings for your data size (section 3)
  prompts: default
  domain:                             # who the tickets are about (section 7)
    product: Acme Cloud
    product_description: a SaaS platform with integrations, reporting and user management
models:                               # two LLMs, "a" and "b" (sections 3 and 5)
  a: {backend: openai, model: gpt-4o-mini, budget_usd: 5}
  b: {backend: ollama, model: "qwen3:8b"}
embedder: {backend: hf, model: Alibaba-NLP/gte-large-en-v1.5}
label_embedder: {backend: st, model: BAAI/bge-large-en-v1.5}
```

The same in Python:

```python
from clustering_harness import Dataset, Pipeline, PipelineConfig
from clustering_harness.llm import LLM, OllamaBackend, OpenAICompatibleBackend, Ledger
from clustering_harness.embeddings import HFTransformerEmbedder, SentenceTransformerEmbedder

ds = Dataset.from_csv("tickets.csv", id_col="id", text_col="description", title_col="subject")
config = PipelineConfig.auto(len(ds), domain={"product": "Acme Cloud",
                                              "product_description": "a SaaS platform with integrations"})
ledger = Ledger("runs/my_first_run/spend.sqlite")
llms = {"a": LLM(OpenAICompatibleBackend("gpt-4o-mini"), max_concurrency=8, budget_usd=5, ledger=ledger),
        "b": LLM(OllamaBackend("qwen3:8b"))}
pipe = Pipeline(config, llms=llms, embedder=HFTransformerEmbedder("Alibaba-NLP/gte-large-en-v1.5"),
                label_embedder=SentenceTransformerEmbedder("BAAI/bge-large-en-v1.5"), workdir="runs/my_first_run")
result = pipe.run(ds)
result.clusters      # one row per cluster
result.assignments   # one row per record
result.summary       # coverage, scores, model usage
```

## 3. Key ideas in plain words

| term | meaning |
|---|---|
| **record** | one input text, e.g. a ticket (title + description). It needs a unique id. |
| **embedding** | a list of numbers that represents a text's meaning; similar texts get similar vectors. The *embedder* computes them. |
| **cluster** | a group of records the models agree is *one recurring problem or request*. It has a name, an inclusion rule ("cases where...") and an exclusion rule. |
| **report line** | one or more clusters merged when they belong on one line of a "top issues" report. |
| **unassigned / junk** | *unassigned*: a real record that fits no cluster (rare, one-off issues). *junk*: empty or test records, removed up front. |
| **model slots `a` and `b`** | the pipeline uses two LLMs. Each has jobs (`a` sorts leftover records, `b` names clusters, ...) and **both act as judges**, so a decision needs two opinions. Using two *different* models makes the judging stricter. |
| **gate / judge** | a judge scores 1–5 whether a group is one report line. The gate combines both judges' scores to accept, split or reject a group. |
| **domain** | a few words about whose tickets these are (product name and description). They are inserted into every prompt. See section 7. |
| **prompt preset** | the full set of instructions sent to the models. `default` is domain-neutral; you can copy it and add your own examples. |
| **preset (`small` / `reference` / `auto`)** | algorithm settings tuned for about 1,000 (`small`) or about 15,000 (`reference`) records; `auto` picks by size. |
| **workdir** | the folder a run writes to. It holds the caches, so rerunning the same spec continues where it stopped and costs nothing for work already done. |
| **cache** | every model answer is stored by its exact request. A repeated request is answered from the cache, which makes reruns cheap and reproducible. |
| **ledger** | a small database that records how much each model has cost, one row per request. |
| **budget (`budget_usd`)** | a spending cap per model. When the ledger says the cap is reached, the run stops instead of overspending. |
| **seed** | a number that fixes the random choices. Running several seeds (`multirun`) shows how stable the result is. |

## 4. Loading your data

Every source becomes a `Dataset` with `ids`, `texts` and optional `titles`, gold `labels` and extra `meta` columns.

**Files and pandas**

```python
Dataset.from_csv("tickets.csv", id_col="id", text_col="body", title_col="subject")
Dataset.from_file("tickets.jsonl", id_col="id", text_col="body")        # .csv .tsv .json .jsonl .parquet .xlsx
Dataset.from_frame(df, id_col="ticket_id", text_col="text")              # any pandas DataFrame
```

**MongoDB** (`pip install -e ".[mongo]"`): one collection, several collections, a filter query or an aggregation pipeline.

```python
ds = Dataset.from_mongo(
    ["tickets_2025", "tickets_2026"],          # one name or a list; ids become "collection:id" when there are several
    uri_env="MONGODB_URI",                     # read the connection string from this environment variable
    database="support",
    query={"status": "closed"},                # or queries={"tickets_2025": {...}, "tickets_2026": {...}}
    text_field=["subject", "body.text"],       # several fields are joined; dotted paths reach nested fields
    title_field="subject",
    label_field="category",                    # optional gold labels for evaluation
    meta_fields=["product", "priority"],       # kept as extra columns
    limit=5000,
)
ds = Dataset.from_mongo("tickets", database="support", text_field="text",
                        pipeline=[{"$match": {"lang": "en"}}, {"$sample": {"size": 3000}}])   # aggregation pipeline
```

In a run spec:

```yaml
data:
  mongo:
    uri_env: MONGODB_URI                # keep the password out of the file
    database: support
    collections: [tickets_2025, tickets_2026]
    query: '{"created": {"$gte": {"$date": "2026-01-01T00:00:00Z"}}}'   # MongoDB Extended JSON for dates and ObjectIds
    text_field: [subject, description]
    title_field: subject
```

**Public benchmarks**: `data: {dataset: banking77, split: test}`. List them with `clustering-harness datasets`;
export one with its gold labels using `clustering-harness datasets --export banking77 --split test`.

## 5. Choosing models

Each slot (`a`, `b`) takes any backend. Options common to all: `concurrency` (parallel requests),
`budget_usd`, `price_in` / `price_out` (USD per million tokens, for models the package has no price for) and
`reasoning: true` (for models that think before answering, such as gpt-oss).

```yaml
models:
  # Local models with Ollama: free; speed depends on your GPU
  a: {backend: ollama, model: "qwen3:8b", url: "http://localhost:11434", concurrency: 2}

  # OpenAI (key from OPENAI_API_KEY)
  b: {backend: openai, model: gpt-4o-mini, concurrency: 8, budget_usd: 5}

  # Any OpenAI-compatible server (vLLM, LM Studio, TGI, ...)
  # a: {backend: openai_compatible, model: my-model, base_url: "http://gpu-box:8000/v1", api_key_env: VLLM_KEY}

  # Azure OpenAI: model is your deployment name (key from AZURE_OPENAI_API_KEY)
  # a: {backend: azure_openai, deployment: my-gpt-4o-mini, azure_endpoint: "https://NAME.openai.azure.com", api_version: "2024-10-21"}

  # Google Gemini through its OpenAI-compatible endpoint (key from GEMINI_API_KEY)
  # a: {backend: gemini, model: gemini-2.5-flash}

  # Anthropic Claude (key from ANTHROPIC_API_KEY)
  # a: {backend: anthropic, model: claude-haiku-5-5}
```

**Amazon Bedrock** supports every common way of signing in to AWS:

```yaml
  # 1. Your normal AWS setup: environment variables, ~/.aws profile, SSO, or the role of the machine
  a: {backend: bedrock, model: us.anthropic.claude-haiku-5-5-v1:0, region: us-east-1}
  # 2. A named profile
  a: {backend: bedrock, model: anthropic.claude-haiku-5-5-v1:0, region: us-east-1, profile_name: analytics}
  # 3. Access keys, optionally temporary ones with a session token
  a: {backend: bedrock, model: ..., region: us-east-1,
      credentials: {aws_access_key_id: AKIA..., aws_secret_access_key: ..., aws_session_token: ...}}
  # 4. Assume an IAM role (cross-account, with an external id); the temporary keys are renewed automatically
  a: {backend: bedrock, model: ..., region: us-east-1, role_arn: "arn:aws:iam::123456789012:role/bedrock-invoke", external_id: abc123}
  # 5. A Bedrock API key (bearer token), from the AWS_BEARER_TOKEN_BEDROCK environment variable
  a: {backend: bedrock, model: ..., region: us-east-1}
  # An inference profile ARN works as the model:  model: "arn:aws:bedrock:us-east-1:123456789012:inference-profile/..."
```

**Keeping keys out of config files.** For keys stored in a secrets manager, a database or an encrypted
store, point the backend at a function of yours that returns them. It is called when the client is created.
For Bedrock it is called again shortly before the keys it returned expire, and after an expired-key error:

```yaml
  a: {backend: bedrock, model: ..., credentials_provider: "my_company.secrets:bedrock_credentials"}
  b: {backend: openai, model: gpt-4o-mini, api_key_provider: "my_company.secrets:openai_key"}
```

```python
# my_company/secrets.py
def bedrock_credentials() -> dict:
    c = my_vault.read("bedrock")      # anything: a vault, a database, STS, ...
    return {"aws_access_key_id": c.key, "aws_secret_access_key": c.secret, "aws_session_token": c.token,
            "expiration": c.expires_at, "region": "us-east-1"}
```

The dict may also use the STS spellings (`AccessKeyId`, `SecretAccessKey`, `SessionToken`, `Expiration`),
so the `Credentials` block of an `assume_role` answer can be returned as it is.

**Your own model loader.** Any object with a `model_id` and a `complete(request)` method is a backend. Use it
from a spec by its import path, register it under a short name, or publish it from your own package:

```yaml
  a: {backend: "my_company.llm:GatewayBackend", model: house-model-v2, timeout: 60}    # import path + its arguments
```

```python
from clustering_harness.llm import ChatResult
from clustering_harness.factory import register_backend

class GatewayBackend:
    def __init__(self, model, timeout=60):
        self.model, self.model_id = model, f"gateway:{model}"
    def complete(self, request):              # request.messages, request.max_tokens, request.json_mode, ...
        text = my_gateway.chat(self.model, request.messages, max_tokens=request.max_tokens)
        return ChatResult(text=text, input_tokens=0, output_tokens=0)

register_backend("gateway", lambda opts: GatewayBackend(opts.pop("model"), opts.pop("timeout", 60)))
# or in your package's pyproject.toml:  [project.entry-points."clustering_harness.backends"]  gateway = "my_company.llm:make"
```

**Which models?** Small and mid-size models work: the validation runs used a 27B Qwen model with
`gpt-oss:20b` locally, and `gpt-4o-mini`. Two *different* models in `a` and `b` make the two-judge checks
meaningful. With the same model in both slots the two judges usually agree, and quality scores come out
inflated (section 13).

## 6. Embeddings

| backend | example | notes |
|---|---|---|
| `hf` | `{backend: hf, model: Alibaba-NLP/gte-large-en-v1.5}` | any Hugging Face encoder; runs on CPU by default |
| `st` | `{backend: st, model: BAAI/bge-large-en-v1.5}` | sentence-transformers models |
| `openai` | `{backend: openai, model: text-embedding-3-large}` | hosted |
| `ollama` | `{backend: ollama, model: bge-m3}` | local server |
| precomputed | `Pipeline(..., vectors=my_numpy_array)` | bring your own vectors |

The `embedder` clusters the records; the `label_embedder` compares cluster names and is used in the final
merge. Add `device: cuda` (or `device: auto`) to use a GPU. Embeddings are cached in
`~/.cache/clustering_harness/embeddings` (change it with `embedding_cache:` in the spec). The cache key
includes the model, pooling, length and precision.

## 7. Describing your domain

The `domain` block tells the models whose records these are:

```yaml
domain:
  product: Acme Cloud
  product_description: a SaaS platform with integrations, reporting and user management
```

It goes into every prompt ("You are a senior support lead at Acme Cloud (a SaaS platform with ...)").
Keep it short and factual: name the product and the main areas customers write about. For non-support
data, describe the source instead, for example `product: the city's 311 service`.

For the best results, copy the `default` prompt preset (`src/clustering_harness/presets/default.yaml`) and
replace its generic anchor examples with paraphrased examples from your own data. Then point
`prompts:` at your file: `prompts: my_prompts.yaml`.

## 8. Cost control: budgets and the ledger

- Every hosted request is priced (prices for common models are built in; set `price_in` / `price_out` for
  others) and written to the **ledger**. By default the ledger is `<workdir>/spend.sqlite`, so
  `budget_usd` caps the spend **of that run**.
- Set `ledger: ~/.cache/clustering_harness/spend.sqlite` in several specs to cap their **combined** spend
  instead, for example for a team-wide monthly limit.
- When a cap is reached, the run stops with `BudgetExceeded`. Raise the cap and rerun: cached answers are
  free, so you only pay for the rest.
- If a hosted model's price is unknown, the budget cannot be enforced and a warning says so.
- Rough cost: about 1.2 model requests per record with the `small` preset. 3,000 short records with
  `gpt-4o-mini` cost about $0.50.

## 9. Reading the outputs

Everything lands in `<workdir>/output/`:

| file | what's inside |
|---|---|
| `report_lines.csv` | the short list: line name, rule, number of records, member clusters, judge score |
| `clusters.csv` | per cluster: name, inclusion and exclusion rule, size, geometry (how tight and separate it is), two-model quality scores (1–5), example and suspicious record ids, `quality_status` (`MODEL_REVIEW_PASSED` or `REVIEW_RECOMMENDED`) |
| `assignments.csv` | per record: `status` (`cluster` / `unassigned` / `junk`), cluster and line, `placement` (which stage placed it) and `reason` (the model's reason) |
| `summary.json` | coverage, mean scores, model usage |

`<workdir>/manifest.json` records the dataset, the configuration fingerprint, the models, the embedder
settings (device, precision) and the library versions. Intermediate results (bank partitions, stage logs,
gate calibration, the evidence each model saw) are under `<workdir>/results/`.

## 10. How it works

```
records ─ junk rule ─ embeddings ─┬─ consensus bank (151 partitions: UMAP+HDBSCAN, PCA+HDBSCAN, k-means, Leiden)
                                  └─ base partition ─ S1 two-judge gate ─ S2 local splits ─ S2b consensus cores
                                     ─ S3B Leiden on residual ─ sampled pruning ─ centroid reassignment ─ merges
                                     ─ LLM residual classification ─ refile pass ─ naming + scoring ─ report-line rollup
```

In plain words:

1. **Clean.** Empty and test records are set aside as junk.
2. **Propose.** Records are embedded and clustered many ways (151 alternative partitions). One of them is
   the starting point, and the others show which records reliably belong together.
3. **Check.** Both models judge sampled records of each cluster. Clusters that are one issue are
   accepted. Clusters that mix issues are split and checked again, and groups nobody can name are dropped.
4. **Fill.** Records still outside any cluster are offered to a model with the list of clusters, which
   either files them under one, proposes a new cluster, or leaves them unassigned. A second pass reconsiders
   the leftovers.
5. **Name and score.** Each cluster gets a name and rules written from its records, and both models score it.
6. **Roll up.** Clusters a support lead would report together are merged into report lines, guarded by a
   judge so that quality does not drop.

Every stage, threshold and config field is specified in [docs/algorithm.md](docs/algorithm.md). Backends and
budgets: [docs/backends.md](docs/backends.md). Metrics: [docs/evaluation.md](docs/evaluation.md).

## 11. Evaluating and comparing

```bash
clustering-harness compare run.yaml --baselines kmeans:k=50 umap_hdbscan:min_cluster_size=10 --judge --judge-max-clusters 40
clustering-harness multirun run.yaml --seeds 0 1 2          # stability across seeds
clustering-harness evaluate --assignments runs/x/output/assignments.csv --gold gold.csv
```

- With gold labels you get ARI, AMI, NMI, V-measure, purity and BCubed. Without them you still get
  silhouette and cluster geometry.
- `--judge` adds a two-model coherence score for every partition, including the baselines.
- Available baselines: `kmeans`, `minibatch_kmeans`, `agglomerative_ward`, `agglomerative_cosine_average`,
  `hdbscan`, `umap_hdbscan`, `leiden_knn`, `gmm`.

```python
from clustering_harness.baselines import run_baseline
from clustering_harness.multirun import compare_partitions, eval_context
table = compare_partitions({"pipeline": result.labels(), "kmeans": run_baseline("kmeans", X, k=50)}, X=X, gold=ds.labels)
```

## 12. Troubleshooting

| symptom | fix |
|---|---|
| `CUDA out of memory` while embedding | the embedder uses the CPU unless you set `device: cuda`; remove it, or free the GPU |
| `... holds a run with a different config or dataset` | a workdir belongs to one configuration; use a new `workdir` after changing settings or data |
| `BudgetExceeded` | raise `budget_usd` and rerun; finished work is cached |
| warning `budget_usd ... cannot be enforced` | set `price_in` / `price_out` for that model |
| local models are slow | lower `concurrency` to what your GPU serves, or use a smaller model for one slot |
| `dataset ... has no records` | check the file path, the Mongo query and the collection names |
| Bedrock `no region` | set `region:` (or `AWS_REGION`) |

## Validation

How we know the package does what it claims. Full method and tables: [docs/validation.md](docs/validation.md).

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
metrics are tested against scikit-learn and textbook examples. All tests run in CI on every push.

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

Backends and loaders are tested the same way: Bedrock against a local server that speaks the Converse wire
format (signatures, session tokens, bearer tokens, role assumption, key refresh), MongoDB against a real
server, and the default embedder against the installed transformers version.

None of these fixes changes memberships, names or scores on the reference data. Behaviours kept for
fidelity are listed in the validation doc.

### 4. Public benchmark: BANKING77

Test split (3,080 queries, 77 gold intents), `small` preset, domain-neutral prompts, `gpt-4o-mini` in both
model slots ([examples/banking77.yaml](examples/banking77.yaml)). The pipeline is **not** told the number of intents; baselines that need *k* get the true 77.
Unassigned records count as singletons.

| partition | clusters | coverage | ARI | NMI | purity |
|---|---|---|---|---|---|
| **pipeline** | 135 | 0.981 | 0.615 | **0.857** | 0.844 |
| k-means, oracle k = 77 | 77 | 1.000 | 0.608 | 0.845 | 0.751 |
| UMAP + HDBSCAN | 103 | 0.894 | 0.634 | 0.855 | 0.879 |
| agglomerative, k = 77 | 77 | 1.000 | 0.507 | 0.835 | 0.575 |
| Leiden, mutual-kNN | 49 | 0.991 | 0.482 | 0.825 | 0.529 |

Without being told *k*, the pipeline is on par with the best baselines on NMI and ARI. Plain UMAP + HDBSCAN is
slightly higher on ARI and purity, but leaves 11% of the queries unassigned; the pipeline assigns 98% and
names every cluster. It splits more finely than the gold intents (135 clusters for 77 intents). Over three seeds: NMI 0.856 ± 0.001, ARI 0.602 ± 0.009, pairwise agreement
between runs ARI 0.760 ± 0.013.

**Independent reproduction.** A separate agent, starting from the README with no other help, reproduced
the table: pipeline ARI 0.622, NMI 0.860, coverage 0.977, 133 clusters, and identical baseline rows
(12.4 minutes, $0.49). Its review led to the fixes in this version: the CPU default for embeddings,
compatibility with recent transformers, this example config, and these docs.

### Caveats

- **One model in both slots.** The BANKING77 run used `gpt-4o-mini` for both slots, so the "two-judge"
  checks were one model judging twice. Use two different models when the checks should be independent.
- **Lenient judges.** With `gpt-4o-mini` as both judges, the LLM coherence evaluation is near ceiling for
  every partition (gold 4.98, pipeline 4.98, k-means 4.93), so it cannot separate good from mediocre
  clusters on short, clean queries. Use a stricter judge pair when the LLM evaluation has to
  discriminate. A run with local open-weight judges is in progress.
- **Run-to-run variation.** Results depend on embedding numerics and model non-determinism. Compare
  runs with `clustering-harness multirun` rather than expecting identical partitions.
- **Not yet benchmarked end to end:** Anthropic, Bedrock, Azure and Gemini backends. They are implemented and
  tested against their wire protocols, but no full benchmark run has used them yet.

**Cost.** The BANKING77 run made about 3,600 `gpt-4o-mini` requests (≈ $0.50, 13 minutes). All
benchmarking, including the independent reproduction, cost about $4.20.

## License

Apache-2.0. Dataset loaders download third-party data under its own license (`clustering-harness datasets` lists them).
