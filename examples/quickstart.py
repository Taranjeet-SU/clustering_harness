"""Cluster your own CSV of support tickets with local Ollama models.

    pip install -e ".[embeddings]"          (from a clone of the repository; see the README)
    ollama pull qwen3:8b && ollama pull gpt-oss:20b
    python examples/quickstart.py tickets.csv

The CSV needs an id column and a text column (a title/subject column is optional but helps).
"""
import json
import logging
import sys

from clustering_harness import Dataset, Pipeline, PipelineConfig
from clustering_harness.embeddings import HFTransformerEmbedder, SentenceTransformerEmbedder
from clustering_harness.llm import LLM, OllamaBackend

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

ds = Dataset.from_csv(sys.argv[1], id_col="id", text_col="text", title_col="title")
config = PipelineConfig.auto(len(ds), prompts="default",
                             domain={"product": "Acme Cloud", "product_description": "a SaaS platform with integrations and reporting"})
llms = {
    "a": LLM(OllamaBackend("qwen3:8b", num_ctx=8192), name="qwen3", max_concurrency=2),
    "b": LLM(OllamaBackend("gpt-oss:20b", num_ctx=16384, think="low"), name="gpt-oss", reasoning=True, max_concurrency=2),
}
pipe = Pipeline(config, llms=llms, embedder=HFTransformerEmbedder("Alibaba-NLP/gte-large-en-v1.5"),
                label_embedder=SentenceTransformerEmbedder("BAAI/bge-large-en-v1.5", device="cpu"), workdir="runs/quickstart")
result = pipe.run(ds)
print(json.dumps(result.summary, indent=1, default=str))
print(result.clusters[["cluster_id", "cluster_name", "case_count", "semantic_score_1_to_5"]].head(20).to_string(index=False))
