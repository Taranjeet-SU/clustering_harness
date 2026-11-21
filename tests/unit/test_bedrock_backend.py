"""BedrockBackend against a local HTTP server that speaks the Converse wire format (no AWS account needed).

Checks what actually goes over the wire for each authentication mode: SigV4 with the expected access key,
session tokens, the bearer-token header on unsigned requests; plus message mapping, credential refresh,
assume-role, retry after an expired token, and the run-spec / plugin plumbing."""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

pytest.importorskip("botocore")

from clustering_harness.factory import backend_kinds, build_backend, build_llm, register_backend  # noqa: E402
from clustering_harness.llm import BackendError, ChatRequest, price_for  # noqa: E402
from clustering_harness.llm.bedrock import BedrockBackend  # noqa: E402


class _Server:
    def __init__(self):
        self.requests: list[dict] = []
        self.fail_next: list[tuple[int, str]] = []
        srv = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
                srv.requests.append({"path": self.path, "headers": dict(self.headers), "body": body})
                if srv.fail_next:
                    code, err, *msg = srv.fail_next.pop(0)          # (status, error type[, message])
                    out = json.dumps({"message": msg[0] if msg else err}).encode()
                    self.send_response(code)
                    self.send_header("x-amzn-ErrorType", err)
                else:
                    text = "".join(c["text"] for m in body["messages"] for c in m["content"])
                    out = json.dumps({"output": {"message": {"role": "assistant", "content": [{"text": f"echo:{text}"}]}},
                                      "stopReason": "end_turn", "usage": {"inputTokens": 11, "outputTokens": 3, "totalTokens": 14},
                                      "metrics": {"latencyMs": 1}}).encode()
                    self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

        self.httpd = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()


@pytest.fixture
def server():
    s = _Server()
    yield s
    s.httpd.shutdown()


@pytest.fixture(autouse=True)
def _no_ambient_aws(monkeypatch, tmp_path):
    for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_PROFILE", "AWS_BEARER_TOKEN_BEDROCK",
              "AWS_REGION", "AWS_DEFAULT_REGION"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "none"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "none"))
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")


REQ = ChatRequest(messages=[{"role": "system", "content": "be brief"}, {"role": "user", "content": "hello"},
                            {"role": "user", "content": "again"}], max_tokens=50, temperature=0.0)


def test_access_keys_sign_requests_and_map_messages(server):
    be = BedrockBackend("anthropic.claude-haiku-5-5-v1:0", region="us-east-1", endpoint_url=server.url,
                        credentials={"aws_access_key_id": "AKIATESTKEY", "aws_secret_access_key": "secret"})
    r = be.complete(REQ)
    assert r.text == "echo:helloagain" and (r.input_tokens, r.output_tokens) == (11, 3) and be.auth_mode == "keys"
    req = server.requests[-1]
    assert req["path"].startswith("/model/anthropic.claude-haiku-5-5-v1%3A0/converse") or "converse" in req["path"]
    assert req["headers"]["Authorization"].startswith("AWS4-HMAC-SHA256 Credential=AKIATESTKEY/")
    body = req["body"]
    assert body["system"] == [{"text": "be brief"}]
    assert body["messages"] == [{"role": "user", "content": [{"text": "hello"}, {"text": "again"}]}]   # same-role turns merged
    assert body["inferenceConfig"] == {"maxTokens": 50}      # no temperature for Claude models (they reject it)


def test_temperature_is_sent_to_non_claude_models_and_can_be_forced(server):
    creds = {"aws_access_key_id": "AKIA1", "aws_secret_access_key": "s"}
    BedrockBackend("meta.llama3-70b-instruct-v1:0", region="us-east-1", endpoint_url=server.url, credentials=creds).complete(REQ)
    assert server.requests[-1]["body"]["inferenceConfig"] == {"maxTokens": 50, "temperature": 0.0}
    BedrockBackend("anthropic.claude-x", region="us-east-1", endpoint_url=server.url, credentials=creds,
                   send_temperature=True).complete(REQ)
    assert server.requests[-1]["body"]["inferenceConfig"]["temperature"] == 0.0


def test_sts_style_keys_with_session_token(server):
    be = BedrockBackend("m", region="eu-west-1", endpoint_url=server.url,
                        credentials={"AccessKeyId": "ASIATEMP", "SecretAccessKey": "s", "SessionToken": "tok123",
                                     "Expiration": "2099-01-01T00:00:00Z"})
    be.complete(REQ)
    h = server.requests[-1]["headers"]
    assert "Credential=ASIATEMP/" in h["Authorization"] and h["X-Amz-Security-Token"] == "tok123"


def test_bearer_token_is_sent_unsigned(server, monkeypatch):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "bedrock-api-key-xyz")
    be = BedrockBackend("m", region="us-west-2", endpoint_url=server.url)
    be.complete(REQ)
    assert be.auth_mode == "bearer"
    assert server.requests[-1]["headers"]["Authorization"] == "Bearer bedrock-api-key-xyz"


def test_inference_profile_overrides_model(server):
    arn = "arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.anthropic.claude-haiku-5-5-v1:0"
    be = BedrockBackend("base-model", region="us-east-1", endpoint_url=server.url,
                        credentials={"aws_access_key_id": "AKIA1", "aws_secret_access_key": "s", "inferenceProfileArn": arn})
    assert be.model == arn and be.model_id == f"bedrock:{arn}"
    be.complete(REQ)
    assert "inference-profile" in server.requests[-1]["path"]


def test_provider_is_called_again_before_expiry(server):
    calls = []

    def provider():
        calls.append(1)
        # first credentials are about to expire (inside the refresh margin), the second ones are fresh
        exp = time.time() + (10 if len(calls) == 1 else 3600)
        return {"aws_access_key_id": f"AKIA{len(calls)}", "aws_secret_access_key": "s", "expiration": exp, "region": "us-east-1"}

    be = BedrockBackend("m", credentials_provider=provider, endpoint_url=server.url)
    be.complete(REQ)
    be.complete(REQ)
    assert len(calls) == 2
    assert "Credential=AKIA1/" in server.requests[0]["headers"]["Authorization"]
    assert "Credential=AKIA2/" in server.requests[1]["headers"]["Authorization"]


def test_expired_token_error_refreshes_and_retries_once(server):
    n = []

    def provider():
        n.append(1)
        return {"aws_access_key_id": f"AKIA{len(n)}", "aws_secret_access_key": "s", "region": "us-east-1"}

    server.fail_next = [(403, "ExpiredTokenException")]
    be = BedrockBackend("m", credentials_provider=provider, endpoint_url=server.url, max_retries=1)
    assert be.complete(REQ).text.startswith("echo:")
    assert len(n) == 2


def test_client_errors_become_backend_errors(server):
    server.fail_next = [(400, "ValidationException")]
    be = BedrockBackend("m", region="us-east-1", endpoint_url=server.url, max_retries=1,
                        credentials={"aws_access_key_id": "AKIA1", "aws_secret_access_key": "s"})
    with pytest.raises(BackendError, match="ValidationException"):
        be.complete(REQ)


def test_assume_role_uses_sts_and_temporary_credentials(server, monkeypatch):
    import botocore.session
    from botocore.stub import Stubber

    real = botocore.session.Session.create_client
    seen = {}

    def create_client(self, service, *a, **kw):
        c = real(self, service, *a, **kw)
        if service == "sts":
            st = Stubber(c)
            st.add_response("assume_role", {"Credentials": {"AccessKeyId": "ASIAROLE00000000", "SecretAccessKey": "rs", "SessionToken": "rt",
                                                            "Expiration": "2099-01-01T00:00:00Z"},
                                            "AssumedRoleUser": {"AssumedRoleId": "AROA:x", "Arn": "arn:aws:sts::1:assumed-role/r/x"}},
                            {"RoleArn": "arn:aws:iam::1:role/r", "RoleSessionName": "clustering-harness", "DurationSeconds": 3600,
                             "ExternalId": "ext-1"})
            st.activate()
            seen["sts_key"] = self.get_credentials().access_key
        return c

    monkeypatch.setattr(botocore.session.Session, "create_client", create_client)
    be = BedrockBackend("m", region="us-east-1", endpoint_url=server.url, role_arn="arn:aws:iam::1:role/r", external_id="ext-1",
                        credentials={"aws_access_key_id": "AKIABASE", "aws_secret_access_key": "s"})
    be.complete(REQ)
    assert be.auth_mode == "assume_role" and seen["sts_key"] == "AKIABASE"
    h = server.requests[-1]["headers"]
    assert "Credential=ASIAROLE00000000/" in h["Authorization"] and h["X-Amz-Security-Token"] == "rt"


def test_missing_region_and_unknown_keys_fail_clearly():
    with pytest.raises(ValueError, match="unknown Bedrock credential key"):
        BedrockBackend("m", credentials={"password": "x"})
    with pytest.raises(BackendError, match="no region"):
        BedrockBackend("m", credentials={"aws_access_key_id": "A", "aws_secret_access_key": "s"}).complete(REQ)


def test_bedrock_ids_are_priced_by_contained_model_name():
    assert price_for("us.anthropic.claude-haiku-5-5-v1:0") == price_for("claude-haiku-5-5")
    assert price_for("gpt-4o-mini-2024-07-18") == price_for("gpt-4o-mini")   # longest name wins over gpt-4o


# ---------------------------------------------------------------- run-spec plumbing / plugins

def provider_for_tests():
    return {"aws_access_key_id": "AKIAPLUG", "aws_secret_access_key": "s", "region": "us-east-1"}


def test_run_spec_bedrock_with_provider_import_string(server):
    llm = build_llm({"backend": "bedrock", "model": "anthropic.claude-haiku-5-5-v1:0", "endpoint_url": server.url,
                     "credentials_provider": f"{__name__}:provider_for_tests", "concurrency": 1})
    assert llm.backend.complete(REQ).text.startswith("echo:")
    assert "Credential=AKIAPLUG/" in server.requests[-1]["headers"]["Authorization"]
    assert llm.price == price_for("claude-haiku-5-5")


class EchoBackend:
    def __init__(self, model="echo", prefix=">"):
        self.model, self.model_id, self.prefix = model, f"echo:{model}", prefix

    def complete(self, request):
        from clustering_harness.llm import ChatResult
        return ChatResult(text=self.prefix + request.messages[-1]["content"])


def test_import_path_backend_and_registry():
    be = build_backend({"backend": f"{__name__}:EchoBackend", "model": "x", "prefix": "#"})
    assert be.complete(REQ).text == "#again"
    register_backend("echo_test", lambda s: EchoBackend(s.pop("model"), s.pop("prefix", ">")))
    assert "echo_test" in backend_kinds()
    assert build_backend({"backend": "echo_test", "model": "y"}).model_id == "echo:y"
    with pytest.raises(ValueError, match="unknown options"):
        build_backend({"backend": "echo_test", "model": "y", "bogus": 1})
    with pytest.raises(ValueError, match="unknown backend"):
        build_backend({"backend": "nope"})


def test_azure_and_gemini_specs_build_openai_clients():
    pytest.importorskip("openai")
    az = build_backend({"backend": "azure_openai", "deployment": "my-gpt4o-mini", "azure_endpoint": "https://x.openai.azure.com",
                        "api_version": "2024-10-21", "api_key_provider": lambda: "k"})
    assert az.model == "my-gpt4o-mini" and az.model_id.startswith("azure-openai:")
    assert type(az._get()).__name__ == "AzureOpenAI"
    ge = build_backend({"backend": "gemini", "model": "gemini-2.5-flash", "api_key_provider": lambda: "k"})
    assert ge.base_url.startswith("https://generativelanguage.googleapis.com/") and ge.supports_seed is False


def test_invalid_model_id_is_a_config_error(server):
    from clustering_harness.llm import BackendConfigError
    server.fail_next = [(400, "ValidationException", "The provided model identifier is invalid.")]
    be = BedrockBackend("no-such-model", region="us-east-1", endpoint_url=server.url, max_retries=1,
                        credentials={"aws_access_key_id": "AKIA1", "aws_secret_access_key": "s"})
    with pytest.raises(BackendConfigError, match="model identifier"):
        be.complete(REQ)
