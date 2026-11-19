# Models, embedders and cost controls

## LLM slots

The pipeline uses two LLM *slots*, `a` and `b`. Both act as independent judges in the gate, the merge vote and
the scoring review; in addition each slot performs fixed roles (override with `config.roles`):

| role | default slot | used for |
|---|---|---|
| `classifier` | a | residual classification (batches of 15 records) |
| `verifier` | a | membership checks (pruning, confirmation, reassignment band) |
| `rollup` | a | report-line partition decisions and line names |
| `canonicalizer` | b | merging synonymous provisional labels |
| `labeler` | b | report line + inclusion criterion for pruning |
| `rechecker` | b | second opinion on "no" answers during pruning |
| scoring namer / follow-up panels | b (`scoring.name_with`) | evidence-grounded names, extra review panels |

Using two **different** models gives two independent judgments; with the same model in both slots identical
requests hit the same cache entry and the gate degenerates to a single judge (still valid, less robust).

Mark models that spend hidden reasoning tokens from the generation budget with `reasoning=True`
(e.g. `gpt-oss`): they get larger budgets, a budget retry on empty answers, and text-mode judging.

## Backends

```python
from clustering_harness.llm import LLM, OllamaBackend, OpenAICompatibleBackend, AnthropicBackend, CallableBackend

local = LLM(OllamaBackend("qwen3:8b", url="http://localhost:11434", num_ctx=8192), name="qwen", max_concurrency=2)
oss   = LLM(OllamaBackend("gpt-oss:20b", num_ctx=16384, think="low"), name="gpt-oss", reasoning=True)
oai   = LLM(OpenAICompatibleBackend("gpt-4o-mini"), max_concurrency=8, budget_usd=20, ledger=ledger)
vllm  = LLM(OpenAICompatibleBackend("Qwen/Qwen2.5-32B-Instruct", base_url="http://gpu-box:8000/v1",
                                    api_key="none", token_param="max_tokens"))
claude = LLM(AnthropicBackend("claude-haiku-5-5", effort="low"), max_concurrency=8, budget_usd=20, ledger=ledger)
custom = LLM(CallableBackend(lambda req: my_client.generate(req.messages, req.max_tokens), model_id="my-model"))
```

| backend | notes |
|---|---|
| `OllamaBackend` | `/api/chat`, JSON mode via `format: json`, pinned `num_ctx`, `think` passed through, seeds honoured |
| `OpenAICompatibleBackend` | OpenAI Chat Completions; `response_format=json_object`; works with vLLM / LM Studio / TGI via `base_url` (set `token_param="max_tokens"` for servers that do not know `max_completion_tokens`) |
| `AnthropicBackend` | official SDK; system messages → `system`; no temperature/seed (current Claude models reject non-default sampling), so reproducibility relies on the cache; thinking disabled + low effort by default |
| `CallableBackend` | any function `ChatRequest -> str` |
| `OfflineBackend` | never calls a model: replay strictly from a cache (misses raise) |

All backends are wrapped by `LLM`, which provides the uniform contract: JSON extraction with one corrective
retry, `{"_error": ...}` on contract failure, a persistent SQLite response cache keyed by the full request,
bounded concurrency, and optional spend caps.

## Cost controls

```python
from clustering_harness.llm import Ledger
ledger = Ledger("~/.cache/clustering_harness/spend.sqlite")      # shared across runs
LLM(OpenAICompatibleBackend("gpt-4o-mini"), budget_usd=20, ledger=ledger)
```

Every non-cached call records input/output tokens and USD (price table `PRICES_PER_MTOK`, override with
`price=Price(in, out)`). Once the ledger's spend for the model reaches `budget_usd`, further calls raise
`BudgetExceeded`; the run stops instead of overspending. Point several runs at the same ledger for a global
cap. `ledger.summary()` reports spend per model. Cached answers cost nothing, so resumed runs are free up to
the point of interruption.

## Embedders

```python
from clustering_harness.embeddings import HFTransformerEmbedder, SentenceTransformerEmbedder, OpenAIEmbedder, OllamaEmbedder, PrecomputedEmbedder
HFTransformerEmbedder("Alibaba-NLP/gte-large-en-v1.5", pooling="cls", max_length=8192)   # reference
SentenceTransformerEmbedder("BAAI/bge-large-en-v1.5", device="cpu")                    # reference label embedder
OpenAIEmbedder("text-embedding-3-small")
PrecomputedEmbedder(my_vectors)
```

Any object with `name` and `encode(texts) -> np.ndarray` works. Changing the embedder changes the geometry
that the consensus bank and the size presets were calibrated on; re-check `config.bank` / `config.base` when
you switch to a very different model.

## Prompts and domains

Prompts live in YAML prompt sets (`clustering_harness/presets/*.yaml`). The `default` set is domain-neutral;
fill its domain variables:

```python
PipelineConfig(prompts="default", domain={"product": "Acme Cloud", "product_description": "a SaaS platform ..."})
```

The gate's three anchor examples calibrate what "one report line" means — replace them with paraphrased
examples from your own corpus for best results (`PromptSet.load("default").to_yaml("my_prompts.yaml")`, edit,
then `prompts="my_prompts.yaml"`). `searchunify.yaml` is a complete example of an adapted prompt set.
