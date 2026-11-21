"""Amazon Bedrock backend (Converse API) with every common way of authenticating.

Credentials are resolved in this order (the first that applies wins):

1. ``credentials_provider``: a callable (or ``"package.module:function"`` import string) returning a credentials
   dict (keys below). It is called again shortly before an ``expiration`` it returned, and after an
   expired-token error. Use it to fetch keys from your own secrets store at run time, so they never
   appear in a config file.
2. ``credentials``: a dict with any of
   ``aws_access_key_id``, ``aws_secret_access_key``, ``aws_session_token``, ``expiration``,
   ``aws_bearer_token_bedrock``, ``region``, ``role_arn``, ``external_id``, ``role_session_name``,
   ``profile_name``, ``model_id`` / ``inference_profile_arn`` (overrides ``model``).
   The STS response spellings ``AccessKeyId``, ``SecretAccessKey``, ``SessionToken`` and ``Expiration``
   (e.g. the ``Credentials`` block of an ``assume_role`` call) are accepted too.
3. Constructor arguments of the same names.
4. A Bedrock API key in the environment variable ``bearer_token_env`` (default ``AWS_BEARER_TOKEN_BEDROCK``).
5. The standard AWS chain (environment variables, ``~/.aws`` profiles, SSO, instance/task roles).

If a ``role_arn`` is given, the resolved base credentials are used to call STS ``AssumeRole`` (with the
optional ``external_id``); the temporary credentials are refreshed automatically before they expire.

A Bedrock API key (bearer token) is sent as an ``Authorization: Bearer`` header on unsigned requests, which
works with any botocore version.

``model`` may be a model id (``anthropic.claude-...``), a cross-region inference profile id
(``us.anthropic.claude-...``) or an inference profile / provisioned throughput ARN.
"""
from __future__ import annotations

import datetime as _dt
import importlib
import logging
import os
import threading
import time
from typing import Callable

from .base import BackendConfigError, BackendError, ChatRequest, ChatResult

__all__ = ["BedrockBackend", "resolve_callable"]

log = logging.getLogger("clustering_harness")

_ALIASES = {
    "AccessKeyId": "aws_access_key_id", "access_key_id": "aws_access_key_id",
    "SecretAccessKey": "aws_secret_access_key", "secret_access_key": "aws_secret_access_key",
    "SessionToken": "aws_session_token", "session_token": "aws_session_token",
    "Expiration": "expiration",
    "bearer_token": "aws_bearer_token_bedrock", "api_key": "aws_bearer_token_bedrock",
    "region_name": "region",
    "roleArn": "role_arn", "externalId": "external_id", "roleSessionName": "role_session_name",
    "profile": "profile_name",
    "modelId": "model_id", "inferenceProfileArn": "inference_profile_arn",
}
_KEYS = {"aws_access_key_id", "aws_secret_access_key", "aws_session_token", "expiration", "aws_bearer_token_bedrock",
         "region", "role_arn", "external_id", "role_session_name", "profile_name", "model_id", "inference_profile_arn"}
_EXPIRED = {"ExpiredTokenException", "ExpiredToken", "UnrecognizedClientException", "InvalidSignatureException"}
_FATAL = _EXPIRED | {"AccessDeniedException", "ResourceNotFoundException", "UnauthorizedException", "AccessDenied",
                     "InvalidClientTokenId", "SignatureDoesNotMatch"}


def resolve_callable(ref):
    """``"package.module:attr"`` -> the attribute; callables are returned unchanged."""
    if ref is None or callable(ref):
        return ref
    mod, _, attr = str(ref).partition(":")
    if not attr:
        raise ValueError(f"expected 'package.module:attribute', got {ref!r}")
    obj = importlib.import_module(mod)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def _normalise(creds: dict | None) -> dict:
    out = {}
    for k, v in (creds or {}).items():
        k = _ALIASES.get(k, k)
        if k not in _KEYS:
            raise ValueError(f"unknown Bedrock credential key {k!r}; accepted: {sorted(_KEYS)}")
        if v not in (None, ""):
            out[k] = v.strip() if isinstance(v, str) else v
    return out


def _to_epoch(x) -> float | None:
    if x in (None, ""):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, _dt.datetime):
        return (x if x.tzinfo else x.replace(tzinfo=_dt.timezone.utc)).timestamp()
    return _dt.datetime.fromisoformat(str(x).replace("Z", "+00:00")).timestamp()


class BedrockBackend:
    """Chat through ``bedrock-runtime`` ``Converse``. Thread-safe; clients are rebuilt when credentials expire."""

    def __init__(self, model: str, region: str | None = None, credentials: dict | None = None,
                 credentials_provider: Callable[[], dict] | str | None = None,
                 aws_access_key_id: str | None = None, aws_secret_access_key: str | None = None,
                 aws_session_token: str | None = None, profile_name: str | None = None,
                 role_arn: str | None = None, external_id: str | None = None, role_session_name: str = "clustering-harness",
                 role_duration_seconds: int = 3600, bearer_token: str | None = None,
                 bearer_token_env: str = "AWS_BEARER_TOKEN_BEDROCK", max_tokens_cap: int = 8192,
                 send_temperature: bool | None = None, additional_model_request_fields: dict | None = None,
                 timeout: float = 120, max_retries: int = 6, refresh_margin_seconds: int = 300, endpoint_url: str | None = None):
        self._provider = resolve_callable(credentials_provider)
        args = {"aws_access_key_id": aws_access_key_id, "aws_secret_access_key": aws_secret_access_key,
                "aws_session_token": aws_session_token, "profile_name": profile_name, "role_arn": role_arn,
                "external_id": external_id, "aws_bearer_token_bedrock": bearer_token, "region": region}
        self._static = {**_normalise({k: v for k, v in args.items() if v}), **_normalise(credentials)}
        self.model = self._static.pop("inference_profile_arn", None) or self._static.pop("model_id", None) or model
        self.region = self._static.get("region") or region
        self.role_session_name = role_session_name
        self.role_duration_seconds = role_duration_seconds
        self.bearer_token_env = bearer_token_env
        self.max_tokens_cap = max_tokens_cap
        # Current Anthropic models reject non-default sampling parameters, so temperature is not sent to Claude
        # models unless asked for; other model families get it (reproducibility then relies on the cache).
        self.send_temperature = send_temperature if send_temperature is not None else not any(
            k in self.model.lower() for k in ("anthropic", "claude"))
        self.additional_model_request_fields = additional_model_request_fields
        self.timeout = timeout
        self.max_retries = max_retries
        self.refresh_margin_seconds = refresh_margin_seconds
        self.endpoint_url = endpoint_url
        self.model_id = f"bedrock:{self.model}"
        self._client = None
        self._expires_at: float | None = None
        self._lock = threading.Lock()
        self.auth_mode = None   # set when the client is built: provider | keys | bearer | assume_role | default_chain

    def cache_settings(self) -> dict:
        out = {}
        if self.additional_model_request_fields:
            out["additional_model_request_fields"] = self.additional_model_request_fields
        if self.max_tokens_cap != 8192:
            out["max_tokens_cap"] = self.max_tokens_cap
        return out

    # ------------------------------------------------------------------ credentials
    def _resolve(self) -> dict:
        creds = dict(self._static)
        if self._provider is not None:
            got = self._provider()
            if not isinstance(got, dict):
                raise BackendConfigError("bedrock credentials_provider must return a dict")
            creds.update(_normalise(got))
            if "inference_profile_arn" in creds or "model_id" in creds:
                self.model = creds.pop("inference_profile_arn", None) or creds.pop("model_id")
        if "aws_bearer_token_bedrock" not in creds and "aws_access_key_id" not in creds and os.environ.get(self.bearer_token_env):
            creds["aws_bearer_token_bedrock"] = os.environ[self.bearer_token_env]
        creds.setdefault("region", self.region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION"))
        if not creds["region"]:
            raise BackendConfigError("bedrock: no region (set `region`, a credentials `region`, or AWS_REGION)")
        return creds

    def _config(self, unsigned: bool = False):
        from botocore.config import Config
        kw = {"retries": {"max_attempts": self.max_retries, "mode": "adaptive"}, "connect_timeout": 30, "read_timeout": self.timeout}
        if unsigned:
            from botocore import UNSIGNED
            kw["signature_version"] = UNSIGNED
        return Config(**kw)

    def _build(self):
        import botocore.session
        creds = self._resolve()
        region = creds["region"]
        self.region = region
        self._expires_at = _to_epoch(creds.get("expiration"))
        if creds.get("aws_bearer_token_bedrock") and not creds.get("role_arn"):
            token = creds["aws_bearer_token_bedrock"]
            client = botocore.session.get_session().create_client("bedrock-runtime", region_name=region,
                                                                  config=self._config(unsigned=True), endpoint_url=self.endpoint_url)

            def add_bearer(request, **_):
                request.headers["Authorization"] = f"Bearer {token}"
            client.meta.events.register("before-send.bedrock-runtime.*", add_bearer)
            self.auth_mode = "bearer"
            return client
        session = botocore.session.Session(profile=creds.get("profile_name"))
        if creds.get("aws_access_key_id"):
            session.set_credentials(creds["aws_access_key_id"], creds.get("aws_secret_access_key"), creds.get("aws_session_token"))
            mode = "provider" if self._provider is not None else "keys"
        else:
            mode = "profile" if creds.get("profile_name") else "default_chain"
        if creds.get("role_arn"):
            sts = session.create_client("sts", region_name=region)
            kw = {"RoleArn": creds["role_arn"], "RoleSessionName": creds.get("role_session_name") or self.role_session_name,
                  "DurationSeconds": self.role_duration_seconds}
            if creds.get("external_id"):
                kw["ExternalId"] = creds["external_id"]
            c = sts.assume_role(**kw)["Credentials"]
            session = botocore.session.Session()
            session.set_credentials(c["AccessKeyId"], c["SecretAccessKey"], c["SessionToken"])
            self._expires_at = _to_epoch(c.get("Expiration"))
            mode = "assume_role"
        self.auth_mode = mode
        return session.create_client("bedrock-runtime", region_name=region, config=self._config(), endpoint_url=self.endpoint_url)

    def _get(self, force: bool = False):
        with self._lock:
            stale = self._expires_at is not None and time.time() >= self._expires_at - self.refresh_margin_seconds
            if self._client is None or force or stale:
                if self._client is not None:
                    log.info("bedrock: refreshing credentials (%s)", "expired" if stale else "forced")
                self._client = self._build()
            return self._client

    # ------------------------------------------------------------------ chat
    @staticmethod
    def _messages(request: ChatRequest):
        system = [{"text": m["content"]} for m in request.messages if m["role"] == "system" and m["content"]]
        msgs: list[dict] = []
        for m in request.messages:
            if m["role"] == "system":
                continue
            if msgs and msgs[-1]["role"] == m["role"]:          # Converse needs alternating roles
                msgs[-1]["content"].append({"text": m["content"]})
            else:
                msgs.append({"role": m["role"], "content": [{"text": m["content"]}]})
        return system, msgs

    def complete(self, request: ChatRequest) -> ChatResult:
        from botocore.exceptions import BotoCoreError, ClientError
        system, msgs = self._messages(request)
        inf = {"maxTokens": max(1, min(int(request.max_tokens), self.max_tokens_cap))}
        if self.send_temperature:
            inf["temperature"] = float(request.temperature)
        kw = {"modelId": self.model, "messages": msgs, "inferenceConfig": inf}
        if system:
            kw["system"] = system
        if self.additional_model_request_fields:
            kw["additionalModelRequestFields"] = self.additional_model_request_fields
        from botocore.exceptions import NoCredentialsError, NoRegionError, PartialCredentialsError
        for attempt in range(2):
            try:
                client = self._get(force=attempt > 0)
            except BackendError:
                raise
            except ClientError as e:          # e.g. STS AssumeRole refused
                code = e.response.get("Error", {}).get("Code", "")
                raise BackendConfigError(f"bedrock: could not obtain credentials: {code}: {str(e)[:300]}") from e
            except (NoCredentialsError, PartialCredentialsError, NoRegionError) as e:
                raise BackendConfigError(f"bedrock: {e}") from e
            except Exception as e:  # noqa: BLE001 - a failing credentials_provider, bad profile name, ...
                raise BackendConfigError(f"bedrock: could not build a client: {type(e).__name__}: {str(e)[:300]}") from e
            try:
                r = client.converse(**kw)
                break
            except ClientError as e:
                code = e.response.get("Error", {}).get("Code", "")
                if attempt == 0 and code in _EXPIRED and (self._provider is not None or "role_arn" in self._static):
                    continue
                bad_model = code == "ValidationException" and any(
                    t in str(e).lower() for t in ("model identifier", "model id", "on-demand throughput", "inference profile"))
                err = BackendConfigError if (code in _FATAL or bad_model) else BackendError
                raise err(f"bedrock {self.model}: {code}: {str(e)[:300]}") from e
            except (NoCredentialsError, PartialCredentialsError) as e:
                raise BackendConfigError(f"bedrock: {e} (no AWS credentials found: set credentials, a profile, a role or a "
                                         f"Bedrock API key)") from e
            except BotoCoreError as e:
                raise BackendError(f"bedrock {self.model}: {str(e)[:200]}") from e
        content = r.get("output", {}).get("message", {}).get("content", [])
        text = "".join(b["text"] for b in content if "text" in b)
        u = r.get("usage", {})
        return ChatResult(text=text, input_tokens=int(u.get("inputTokens") or 0), output_tokens=int(u.get("outputTokens") or 0),
                          raw={"stop_reason": r.get("stopReason")})
