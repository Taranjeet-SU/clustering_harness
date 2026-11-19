"""Masking of personal / environment data before text reaches an LLM prompt.

Embeddings are computed on the original text; only the snippets placed in prompts are masked. The default
:class:`RegexMasker` replaces URLs, e-mail addresses, IPv4 addresses, phone numbers and host names, plus any
configured person names. Supply your own callable ``str -> str`` for stricter policies."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Sequence

__all__ = ["RegexMasker", "NoMasker", "Masker"]

Masker = Callable[[str], str]

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[a-z]{2,}")
_PHONE = re.compile(r"\+?\d[\d\s().-]{8,}\d")
_URL = re.compile(r"(https?|ldaps?|jdbc:[a-z]+)://\S+", re.I)
_IP = re.compile(r"\b\d{1,3}(\.\d{1,3}){3}\b")
_HOST = re.compile(r"\b[\w-]+(\.[\w-]+)*\.(com|net|org|local|io|cloud|internal)\b", re.I)
_TEAMS = re.compile(r"(team|tam|csm|services|service|_|\.com|\.net|ops|support|group|admin)", re.I)


@dataclass
class RegexMasker:
    """Regex-based masker.

    ``requester_prefix`` (optional) is a regex with one group matching a requester name at the start of a
    title, e.g. ``r"^Request for (.+?) :"``; a matched name that does not look like a team is replaced by
    ``<person>``. ``names`` are additional person names masked everywhere (whole-word, case-insensitive).
    """

    names: Sequence[str] = field(default_factory=list)
    requester_prefix: str | None = None
    mask_urls: bool = True
    mask_emails: bool = True
    mask_ips: bool = True
    mask_phones: bool = True
    mask_hosts: bool = True

    def __post_init__(self):
        self._names = (re.compile(r"\b(" + "|".join(re.escape(n) for n in sorted(self.names, key=len, reverse=True)) + r")\b", re.I)
                       if self.names else None)
        self._req = re.compile(self.requester_prefix, re.I) if self.requester_prefix else None

    def __call__(self, s: str) -> str:
        s = s or ""
        if self.mask_urls:
            s = _URL.sub("<url>", s)
        if self.mask_emails:
            s = _EMAIL.sub("<email>", s)
        if self.mask_ips:
            s = _IP.sub("<ip>", s)
        if self.mask_phones:
            s = _PHONE.sub("<phone>", s)
        if self.mask_hosts:
            s = _HOST.sub("<host>", s)
        if self._req is not None:
            m = self._req.match(s)
            if m and not _TEAMS.search(m.group(1)):
                s = "Request for <person> :" + s[m.end():]
        if self._names is not None:
            s = self._names.sub("<person>", s)
        return s


class NoMasker:
    def __call__(self, s: str) -> str:
        return s or ""
