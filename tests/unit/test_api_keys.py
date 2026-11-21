"""API keys must only go where they were meant to: never the OpenAI key to another server by accident."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

pytest.importorskip("openai")

from clustering_harness.llm import BackendConfigError, ChatRequest, OpenAICompatibleBackend  # noqa: E402
from clustering_harness.factory import build_backend  # noqa: E402

REQ = ChatRequest(messages=[{"role": "user", "content": "hi"}], max_tokens=5)


@pytest.fixture
def server():
    seen = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            seen.append(self.headers.get("Authorization"))
            out = json.dumps({"id": "x", "object": "chat.completion", "created": 0, "model": "m",
                              "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                              "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    srv.url, srv.seen = f"http://127.0.0.1:{srv.server_port}/v1", seen
    yield srv
    srv.shutdown()


def test_openai_key_is_not_sent_to_another_server_by_default(server, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-openai-key-123456")
    be = build_backend({"backend": "openai_compatible", "model": "m", "base_url": server.url})
    assert be.complete(REQ).text == "ok"
    assert server.seen == ["Bearer not-needed"]


def test_unset_named_key_variable_is_an_error_not_a_fallback(server, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-openai-key-123456")
    monkeypatch.delenv("VLLM_KEY", raising=False)
    be = build_backend({"backend": "openai_compatible", "model": "m", "base_url": server.url, "api_key_env": "VLLM_KEY"})
    with pytest.raises(BackendConfigError, match="VLLM_KEY .* is not set"):
        be.complete(REQ)
    assert server.seen == []                                     # nothing was sent


def test_named_key_is_used(server, monkeypatch):
    monkeypatch.setenv("VLLM_KEY", "local-secret")
    be = build_backend({"backend": "openai_compatible", "model": "m", "base_url": server.url, "api_key_env": "VLLM_KEY"})
    be.complete(REQ)
    assert server.seen == ["Bearer local-secret"]


def test_gemini_without_its_key_is_an_error(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-openai-key-123456")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(BackendConfigError, match="GEMINI_API_KEY"):
        build_backend({"backend": "gemini", "model": "gemini-2.0-flash"}).complete(REQ)


def test_openai_itself_still_uses_openai_api_key(monkeypatch):
    from clustering_harness.llm.backends import resolve_api_key
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real")
    assert resolve_api_key("auto", None) == "sk-real"
    assert resolve_api_key("auto", "https://api.openai.com/v1") == "sk-real"
    assert resolve_api_key("auto", "https://my-proxy.example/v1") == "not-needed"


def test_anthropic_named_key_unset_is_an_error(monkeypatch):
    pytest.importorskip("anthropic")
    from clustering_harness.llm import AnthropicBackend
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-real")
    monkeypatch.delenv("PROXY_KEY", raising=False)
    with pytest.raises(BackendConfigError, match="PROXY_KEY"):
        AnthropicBackend("claude-haiku-5-5", api_key_env="PROXY_KEY", base_url="http://127.0.0.1:9").complete(REQ)
