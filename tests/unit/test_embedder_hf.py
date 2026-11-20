"""Real Hugging Face embedder against the installed transformers version (downloads ~1.7 GB; network)."""
import numpy as np
import pytest

from clustering_harness.embeddings import HFTransformerEmbedder

pytest.importorskip("torch")
pytest.importorskip("transformers")


def test_default_device_is_cpu_and_name_has_no_precision_tag():
    e = HFTransformerEmbedder()
    assert e.device == "cpu" and not e.name.endswith(":fp16")


def test_fp16_gpu_vectors_get_a_distinct_cache_name():
    e = HFTransformerEmbedder(device="cuda", half=True)
    assert e.name.endswith(":fp16")


@pytest.mark.network
def test_gte_large_encodes_on_installed_transformers():
    texts = ["reset my password", "card payment declined", "my card was charged twice for one purchase"]
    V = HFTransformerEmbedder(device="cpu", half=False).encode(texts)
    assert V.shape == (3, 1024)
    np.testing.assert_allclose(np.linalg.norm(V, axis=1), 1.0, atol=1e-3)
    S = V @ V.T
    assert S[1, 2] > S[0, 1]          # the two card-payment texts are closer to each other than to the password one
