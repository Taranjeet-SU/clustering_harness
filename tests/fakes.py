"""The test doubles ship with the package as :mod:`clustering_harness.testing` (so services that embed the pipeline
can test their integration offline); this module keeps the old import path for the suite."""
from clustering_harness.testing import (KIND_MARKERS, TOPICS, FakeGate, FakeLLM, HashEmbedder, detect_kind,  # noqa: F401
                                        fake_llms, make_ctx, make_topic_dataset, rid_of, topic_of)
