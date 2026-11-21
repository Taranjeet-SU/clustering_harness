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
2. [Your first run (about 15 minutes)](#2-your-first-run-about-15-minutes)
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

The package is not on PyPI yet; install it from GitHub. You need Python 3.9 or newer. The
oldest stack the test suite runs on (in CI) is Python 3.9 with numpy 1.22, pandas 1.1.5, scikit-learn 0.24.2 and
pydantic 2.13, so the package can be installed into an older service without upgrading its scientific libraries.

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

## 2. Your first run (about 15 minutes)

The repository ships a ready-made run on a public dataset. All you need is an OpenAI API key; the run costs
about $0.50 and takes 10–15 minutes on a laptop CPU (embedding takes 1–3 minutes, the model calls most of
the rest). Before the slow part starts, the run
sends one tiny test request to each model, so a wrong key, model name or server address fails within seconds.

```bash
export OPENAI_API_KEY=sk-...
clustering-harness -v run examples/banking77.yaml
```

`-v` prints each stage as it starts and finishes. When the run is done, open `runs/banking77/output/`:
`report_lines.csv` is the short list, `clusters.csv` has the details and `assignments.csv` has one row per
query. Then compare the result with classic clustering and the true labels:

```bash
clustering-harness compare examples/banking77.yaml --baselines kmeans:k=77 agglomerative_cosine_average:k=77 \
    umap_hdbscan:min_cluster_size=10 hdbscan:min_cluster_size=10 leiden_knn:resolution=1.0
```

This reruns nothing: every model answer is cached, so it only adds the baselines. It writes
`runs/banking77/output/comparison.csv`, with one row per partition. `pipeline_clusters` is the pipeline row of
the Validation table. Add `--judge --judge-max-clusters 40` for the LLM-judge column (about $0.15 with `gpt-4o-mini`). Baseline names
and options are checked before anything runs.

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

Paths: input files (`data.path`, a prompt YAML) are looked up in the current folder first, then next to the
spec. Outputs (`workdir`, `ledger`) are relative to the folder you run the command from. Unknown keys and
misspelled presets are errors, with the closest valid name suggested.

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
| **unassigned / junk** | *unassigned*: a real record that fits no cluster (rare, one-off issues). *junk*: records with no usable content, removed up front: a placeholder title (`test`, `new case`, ...) with an empty description, or fewer than 12 letters/digits in title and description together (any script; a Chinese/Japanese/Korean character counts as 3). Short queries such as "I lost my card" (11) count as junk; for one-line queries set `pipeline.junk.min_alnum: 5` or lower. |
| **model slots `a` and `b`** | the pipeline uses two LLMs. Each has jobs (`a` sorts leftover records, `b` names clusters, ...) and **both act as judges**, so a decision needs two opinions. Using two *different* models makes the judging stricter. |
| **gate / judge** | a judge scores 1–5 whether a group is one report line. The gate combines both judges' scores to accept, split or reject a group. |
| **domain** | a few words about whose tickets these are (product name and description). They are inserted into every prompt. See section 7. |
| **prompt preset** | the full set of instructions sent to the models. `default` is domain-neutral; you can copy it and add your own examples. |
| **preset (`small` / `reference` / `auto`)** | algorithm settings tuned for about 1,000 (`small`) or about 15,000 (`reference`) records; `auto` picks by size. |
| **workdir** | the folder a run writes to. It holds the caches, so rerunning the same spec continues where it stopped and costs nothing for work already done. `clustering-harness reset run.yaml` clears its results but keeps the model answers. |
| **cache** | every good model answer is stored by its exact request (failed requests and unparseable answers are not). A repeated request is answered from the cache, which makes reruns cheap and reproducible. It lives in `<workdir>/cache/llm.sqlite` (embeddings: `~/.cache/clustering_harness/embeddings`). |
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
    text_field=["body.text", "resolution"],    # the description: several fields are joined; dotted paths reach nested fields
    title_field="subject",                     # the title is added to the text automatically; don't repeat it above
    label_field="category",                    # optional gold labels for evaluation
    meta_fields=["product", "priority"],       # kept as extra columns
    limit=5000,                                # read at most 5,000 documents, collections in the order given
)
ds = Dataset.from_mongo("tickets", database="support", text_field="text",
                        pipeline=[{"$match": {"lang": "en"}}, {"$sample": {"size": 3000}}])   # aggregation pipeline
```

`limit` caps how many documents are read, in collection order. For a reproducible *random* sample across
all collections, set `limit:` (and `seed:`) at the `data:` level of a run spec instead. The pipeline then
samples after loading.

In a run spec:

```yaml
data:
  mongo:
    uri_env: MONGODB_URI                # keep the password out of the file
    database: support
    collections: [tickets_2025, tickets_2026]
    query: '{"created": {"$gte": {"$date": "2026-01-01T00:00:00Z"}}}'   # MongoDB Extended JSON for dates and ObjectIds
    text_field: description             # the body; the title is combined with it automatically
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

  # Any OpenAI-compatible server (vLLM, LM Studio, TGI, ...): no key is sent unless you name one with api_key_env
  # a: {backend: openai_compatible, model: my-model, base_url: "http://gpu-box:8000/v1", api_key_env: VLLM_KEY}

  # Azure OpenAI: model is your deployment name (key from AZURE_OPENAI_API_KEY)
  # a: {backend: azure_openai, deployment: my-gpt-4o-mini, azure_endpoint: "https://NAME.openai.azure.com", api_version: "2024-10-21"}

  # Google Gemini through its OpenAI-compatible endpoint (key from GEMINI_API_KEY)
  # a: {backend: gemini, model: gemini-2.5-flash}

  # Anthropic Claude (key from ANTHROPIC_API_KEY)
  # a: {backend: anthropic, model: claude-haiku-5-5}
```

**Where keys go.** Each key is read from the variable named for that model and sent only to that model's
server. `OPENAI_API_KEY` is used by default only for OpenAI itself; an `openai_compatible` server gets no key
unless you set `api_key_env`. If a named variable (`api_key_env`, `GEMINI_API_KEY`, `AZURE_OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`) is not set, the model is refused before any request, instead of silently using another key.

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

**Your own model loader.** Any object with a `model_id` and a `complete(request)` method is a backend. To
control failure handling, raise `BackendConfigError` for problems retrying cannot fix (bad key, unknown model),
`RequestRejectedError` for a request the server refuses as invalid, and `BackendError` for temporary problems
(all from `clustering_harness.llm`); any other exception is treated as temporary and retried once. Use it
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
        # report real token counts when the gateway returns them; with 0/0 the spend is estimated from the
        # text length (about 3 characters per token) so that budgets still hold
        return ChatResult(text=text, input_tokens=0, output_tokens=0)

# from Python (before you build models in the same process):
register_backend("gateway", lambda opts: GatewayBackend(opts.pop("model"), opts.pop("timeout", 60)))
# for YAML specs and the command line, use the import path above, or publish the short name from your package:
#   [project.entry-points."clustering_harness.backends"]   gateway = "my_company.llm:make"
```

`register_backend` only affects the Python process that calls it, so `backend: gateway` in a YAML spec works
only through the entry point (or when you build the pipeline from Python after registering).

An import path only works if Python can import the module. The command line adds the current folder and the
folder of the run spec, so a `my_backend.py` next to `run.yaml` just works. Anything else must be installed
(`pip install -e path/to/your/package`) or on `PYTHONPATH`.

**When a model fails.** Every failed request is logged. Requests the provider refuses as invalid (HTTP
400/413/422: input too long, content filter) are not retried and do not stop the run: that one item is treated
as "no answer", counted in `summary.json` (`model_problems`) and reported as a warning; many in a row stop the
model like other failures. If any other request of a stage still failed after its
retries, the stage is **not saved**: its files are removed and the run stops with `IncompleteStageError`. Rerun
the same command when the server is healthy. Answers that did arrive are cached, so only the failed requests
are sent again. A run that finishes therefore never contains results built from failed requests.

Only good answers are cached: a request that failed, or an answer that was still not valid JSON after one
repair request, is asked again on a rerun. (Set
`cache_contract_failures: true` on a model to cache those answers too; the exact-reproduction replay in
docs/validation.md does.) A configuration problem (invalid key, no permission, unknown model, missing region)
stops the run with the provider's message, and so do 8 failed requests in a row (`max_consecutive_failures`).
Once a model is marked unusable, requests already queued for it fail immediately instead of being sent, and
unstarted jobs are dropped. Requests already in flight finish first, so a few may still go out.
`timeout` and `max_retries` can be set per model (`openai`/`anthropic`/`bedrock`) to decide how long a dead
server is retried; those SDKs retry with backoff themselves, so the package adds no retries of its own on top.

**What the cache key covers.** The model id, the messages, `max_tokens`, seed, JSON mode, temperature, and
model settings that change answers without changing the request (`reasoning`, Ollama `think`/`num_ctx`,
Anthropic `thinking`/`effort`, Bedrock extra request fields). Changing any of them asks the model again.

**Which models?** Small and mid-size models work: the validation runs used a 27B Qwen model with
`gpt-oss:20b` locally, and `gpt-4o-mini`. Two *different* models in `a` and `b` make the two-judge checks
meaningful. With the same model in both slots the two judges send identical requests and share cached
answers, so every two-judge check is effectively one judge and quality scores come out inflated (section 13);
the run warns about it.

## 6. Embeddings

| backend | example | notes |
|---|---|---|
| `hf` | `{backend: hf, model: Alibaba-NLP/gte-large-en-v1.5}` | any Hugging Face encoder; runs on CPU by default. `trust_remote_code` is on by default because the GTE models ship their own model code; it runs code from the model's Hugging Face repository, so set `trust_remote_code: false` for models that don't need it |
| `st` | `{backend: st, model: BAAI/bge-large-en-v1.5}` | sentence-transformers models |
| `openai` | `{backend: openai, model: text-embedding-3-large}` | hosted |
| `ollama` | `{backend: ollama, model: bge-m3}` | local server |
| precomputed | `Pipeline(..., vectors=my_numpy_array)` | bring your own vectors |

The `embedder` clusters the records; the `label_embedder` compares cluster names and is used in the final
merge. Add `device: cuda` (or `device: auto`) to use a GPU. Embeddings are cached in
`~/.cache/clustering_harness/embeddings`. Change the folder with `embedding_cache:` in the spec, or for every
run with the `CLUSTERING_HARNESS_CACHE` environment variable (embeddings then go to
`$CLUSTERING_HARNESS_CACHE/embeddings`). The cache key includes the model, pooling, length, precision and the
exact texts.

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
- A cap applies **per model id** within a ledger (`openai:gpt-4o-mini`, `azure-openai:...:gpt-4o-mini` and
  `bedrock:...` are different ids). If slots `a` and `b` both use `gpt-4o-mini` through the same backend with
  `budget_usd: 5`, they share one $5 cap. To share a cap across backends, give each model the same
  `budget_scope: <name>`. `multirun` seeds share the main run's ledger, so the cap covers
  all seeds together.
- Set `ledger: ~/.cache/clustering_harness/spend.sqlite` in several specs to cap their **combined** spend
  instead, for example for a team-wide monthly limit.
- Requests in flight are counted against the cap before they are sent (a conservative estimate from the
  prompt length and `max_tokens`), so parallel requests cannot overshoot it. The run may stop slightly
  below the cap.
- When a cap is reached, the run stops with `BudgetExceeded`. Raise the cap and rerun: cached answers are
  free, so you only pay for the rest.
- A cap always holds. If the model's price is unknown (a new model name, a Bedrock ARN, a custom backend),
  setting `budget_usd` is an error until you add `price_in` / `price_out` (0 and 0 for a free model). In
  Python, an `LLM` with `budget_usd` but no `ledger` counts its own calls in memory.
- A backend that reports no token usage is charged an estimate (about 3 characters per token, which errs
  high), so caps hold for custom backends too.
- Rough cost: about 1.2 model requests per record with the `small` preset. 3,000 short records with
  `gpt-4o-mini` cost about $0.50.

## 9. Reading the outputs

Everything lands in `<workdir>/output/`:

| file | what's inside |
|---|---|
| `report_lines.csv` | the short list: `line_id`, `line_name`, `criterion` (the rule), `tickets`, `n_clusters`, `judge_score`, `round_formed` (the merge round that formed the line; -1 = a single cluster kept as is), `cluster_ids`, `cluster_names` |
| `clusters.csv` | per cluster: `cluster_name`, `inclusion_criterion` / `exclusion_criterion`, `case_count`, duplicate-text counts, geometry (`cosine_silhouette_mean`, centroid cosines, `nearest_cluster_id`, ...: how tight and separate it is), quality scores 1–5 (`semantic_score_1_to_5` and per judge, `label_fit`, `distinction`, `judge_disagreement`), `example_case_ids`, `suspicious_sample_ids`, `semantic_reasons`, and `quality_status`: `MODEL_REVIEW_PASSED` when both judges accepted every sampled panel with scores of at least 4 and no suspicious records were found, otherwise `REVIEW_RECOMMENDED` (a person should look at it) |
| `assignments.csv` | per record: `ticket_id`, `title`, `cluster_id` / `cluster_name`, `line_id` / `line_name`, `status` (`cluster` / `unassigned` / `junk`), `placement` (how it got its cluster, below), `reason` (why it was *not* clustered: `junk` = removed up front by the junk rule; `LLM junk: ...` = a model judged the record unusable while filing left-overs, with its reason; `not placed` = no cluster fitted; empty for clustered records) and `previous_reason` (the reason before the refile pass) |
| `summary.json` | coverage, mean scores, model usage, and `model_problems` (answers that stayed unparseable and requests the provider rejected, per slot; also printed as a warning) |

`placement` values: `cascade_B` (placed by the clustering cascade), `filed_existing_confirmed` (filed into an
existing cluster by a model and confirmed), `reassigned` (moved to a better-fitting cluster), `new_cluster`
(member of a cluster proposed by a model for left-over records), `refiled_...` (the same, during the second,
refile pass); empty for unassigned and junk records.

**What the models see.** Titles and description snippets placed in prompts are masked first: URLs, e-mail
addresses, IP addresses, phone numbers and host names are replaced (plus `pipeline.masking.names`). Embeddings
use the original text. Turn masking off with `pipeline.masking.enabled: false`.

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

1. **Clean.** Records with no usable content (placeholder titles, fewer than 12 letters/digits) are set aside
   as junk (see "unassigned / junk" in the glossary).
2. **Propose.** Records are embedded and clustered many ways (151 alternative partitions). One of them is
   the starting point, and the others show which records reliably belong together.
3. **Check.** Both models judge sampled records of each cluster. Clusters that are one issue are
   accepted. Clusters that mix issues are split and checked again, and groups nobody can name are dropped.
4. **Fill.** Records still outside any cluster are offered to a model with the list of clusters, which
   either files them under one, proposes a new cluster, or leaves them unassigned. A second pass reconsiders
   the leftovers.
5. **Name and score.** Each cluster gets a name and rules written from its records, and both models score it.
6. **Roll up.** Clusters a support lead would report together are merged into report lines, guarded by a
   judge so that quality does not drop. Merging only happens while there are more than `rollup.target_max`
   (default 150) clusters; at or below that, each cluster is its own line (as on BANKING77, about 135
   clusters). Lowering `pipeline.rollup.target_max` lets the rollup try to merge further, but every merge
   still needs the judge's approval, so distinct clusters stay separate lines.

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
| `... holds a run made with different settings` | a workdir belongs to one dataset, configuration, prompt set (file contents and domain), model pair with its settings, and embedder with its options (all checked; settings that cannot change results, like `bank.workers`, are not). The message names each changed setting and the first stage that depends on it. `clustering-harness reset run.yaml --from-stage <that stage>` keeps the earlier stages (a change that a kept stage depends on is still refused); `clustering-harness reset run.yaml` starts over. Both keep the paid model answers, so unchanged requests cost nothing |
| `IncompleteStageError: stage ... request(s) failed` | the model server failed some requests; nothing from that stage was saved. Rerun when it is healthy: only the failed requests are sent again |
| `ModelUnavailableError: ... cannot be used` | the provider rejected the settings (key, permission, model name, region); fix them and rerun. Nothing failed was cached |
| `ModelUnavailableError: ... requests in a row failed` | the server is down, unreachable or rate-limiting; the log shows each error. Rerun when it's back |
| `ModuleNotFoundError` for a `module:Class` backend | put the module next to the run spec, install your package, or add it to `PYTHONPATH` |
| `judge 'a' ... gave all N calibration answers the same score` | that model is not really judging (a stub, a broken prompt, or a model too weak for the task); use another model. The run stops here unless `pipeline.gate.allow_constant_judge: true` (which can be added to the same workdir) |
| `... did not answer a test request` | the test request sent before embedding failed; check the model name, key, URL and network |
| `unknown run spec key(s)` / `unknown data key(s)` / `pipeline.preset ...` | a typo in the spec; the message suggests the closest valid name |
| `field(s) not found in any of the N documents` | a MongoDB field name is wrong; the message lists the fields it saw |
| `no MongoDB connection string` | set `MONGODB_URI` (or the variable named by `uri_env`), or give `uri:` |
| `BudgetExceeded` | raise `budget_usd` and rerun; finished work is cached |
| `budget_usd=... cannot be enforced because the price ... is unknown` | set `price_in` / `price_out` for that model (0 and 0 if it is free) |
| `cannot reach MongoDB (...)` | check the URI, the network and that the server runs (fails after 10 s; change with `client_kwargs: {serverSelectionTimeoutMS: ...}`) |
| local models are slow | lower `concurrency` to what your GPU serves, or use a smaller model for one slot |
| `dataset ... has no records` | check the file path and the Mongo query (missing collections are reported by name) |
| a plain `error: ...` line | the command line prints one line for expected errors; add `-v` to see the full traceback |
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
(12.4 minutes, $0.49). Later independent reproductions on 0.1 and 0.2 gave ARI 0.615 / 0.604 / 0.616 and NMI
0.857 / 0.853 / 0.858, with identical baseline rows each time. Their reviews led to the robustness fixes in
this version.

**Claude Haiku in both slots** (`claude-haiku-5-5`): 156 clusters, ARI 0.552, NMI 0.845, purity 0.842,
coverage 0.986. Haiku splits more finely; its 110 report lines reach ARI 0.605 and NMI 0.859, on par with
the `gpt-4o-mini` clusters. 12.4 minutes, about $0.95 including the judge column. Details in
[docs/validation.md](docs/validation.md).

### Caveats

- **One model in both slots.** The BANKING77 run used `gpt-4o-mini` for both slots, so the "two-judge"
  checks were one model judging twice. Use two different models when the checks should be independent.
- **Lenient judges.** With `gpt-4o-mini` as both judges, the LLM coherence evaluation is near ceiling for
  every partition (gold 4.98, pipeline 4.98, k-means 4.93), so it cannot separate good from mediocre
  clusters on short, clean queries. Use a stricter judge pair when the LLM evaluation has to
  discriminate. With two different local models as judges (a 27B Qwen model and `gpt-oss:20b`), the
  clustering is just as good (ARI 0.614, NMI 0.855). The judge scores spread a little more (k-means 4.78,
  Leiden 4.74, pipeline 4.93) but are still near ceiling.
- **Run-to-run variation.** Results depend on embedding numerics and model non-determinism. Compare
  runs with `clustering-harness multirun` rather than expecting identical partitions.
- **Not yet benchmarked end to end:** Bedrock, Azure and Gemini backends. They are implemented and tested
  against their wire protocols, but no full benchmark run has used them yet. (OpenAI, Anthropic and local
  Ollama models have been benchmarked.)

**Cost.** The BANKING77 run made about 3,600 `gpt-4o-mini` requests (≈ $0.50, 13 minutes). All
benchmarking, including the independent reproduction, cost about $4.20.

## License

Apache-2.0. Dataset loaders download third-party data under its own license (`clustering-harness datasets` lists them).
