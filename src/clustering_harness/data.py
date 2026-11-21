"""Datasets: loading, record normalisation, rule-based junk detection, text views.

A :class:`Dataset` is an ordered collection of records. Every record has a unique string ``id``, an optional
``title`` (subject / short summary) and a ``text`` (description / body). Optional gold ``labels`` enable
external evaluation; any other columns are kept in ``meta`` and never influence clustering.

Row order is preserved everywhere: embeddings, assignments and exports use the dataset order. Algorithms
that need a deterministic member order sort ids with :meth:`Dataset.order_key` (numeric ids sort
numerically, as in the reference implementation; other ids sort by row position).
"""
from __future__ import annotations

import csv
import hashlib
import html
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

__all__ = ["Dataset", "JunkRule", "TextTemplate", "alnum_count", "one_line"]


_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff]")


def alnum_count(s: str | None) -> int:
    """Letters and digits in ``s``, in any script. A Chinese/Japanese/Korean character counts as 3, roughly one
    word, because those scripts write a word in one or two characters. For ASCII text this equals the number of
    ASCII letters and digits."""
    s = s or ""
    return sum(1 for c in s if c.isalnum()) + 2 * len(_CJK.findall(s))


def one_line(s: str | None, n: int) -> str:
    """Collapse whitespace and cut to ``n`` characters."""
    return re.sub(r"\s+", " ", s or "").strip()[:n]


@dataclass(frozen=True)
class JunkRule:
    """Rule-based quarantine of records with no usable content (evaluated before any clustering).

    A record is junk when its title matches ``placeholder_title`` and its description is empty, or when the
    title and description together have fewer than ``min_alnum`` letters/digits (any script; see
    :func:`alnum_count`). For very short texts such as one-line queries, lower ``min_alnum``. A description is *empty*
    when it has fewer than ``min_desc_alnum`` letters/digits or repeats the title.
    """

    placeholder_title: str = (r"^\s*(a new case|an existing case|new case|existing case)\s*\d*\s*$|^\s*test(ing)?\s*\d*\s*$|"
                              r"chat initiated but no information was shared")
    min_alnum: int = 12
    min_desc_alnum: int = 3

    def description_empty(self, title: str, desc: str) -> bool:
        return alnum_count(desc) < self.min_desc_alnum or desc.strip().lower() == title.strip().lower()

    def is_junk(self, title: str, desc: str) -> bool:
        empty = self.description_empty(title, desc)
        placeholder = bool(re.match(self.placeholder_title, title, re.I)) if self.placeholder_title else False
        return (placeholder and empty) or alnum_count(title + desc) < self.min_alnum


@dataclass(frozen=True)
class TextTemplate:
    """How a record is turned into the text that is embedded.

    ``with_title`` is used when the record has a title, ``without_title`` otherwise. Placeholders:
    ``{title}`` and ``{text}``.
    """

    with_title: str = "Title: {title}\nDescription: {text}"
    without_title: str = "{text}"

    def render(self, title: str, text: str) -> str:
        return (self.with_title if title else self.without_title).format(title=title, text=text)


@dataclass
class Dataset:
    """An ordered set of text records to cluster."""

    ids: list[str]
    texts: list[str]
    titles: list[str] | None = None
    labels: list | None = None
    meta: pd.DataFrame | None = None
    name: str = "dataset"
    _pos: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.ids = [str(i) for i in self.ids]
        n = len(self.ids)
        if len(self.texts) != n:
            raise ValueError("ids and texts must have the same length")
        if self.titles is None:
            self.titles = [""] * n
        if len(self.titles) != n:
            raise ValueError("titles must have the same length as ids")
        if self.labels is not None and len(self.labels) != n:
            raise ValueError("labels must have the same length as ids")
        self.texts = ["" if t is None or (isinstance(t, float) and np.isnan(t)) else str(t) for t in self.texts]
        self.titles = ["" if t is None or (isinstance(t, float) and np.isnan(t)) else str(t) for t in self.titles]
        if len(set(self.ids)) != n:
            dup = pd.Series(self.ids)[pd.Series(self.ids).duplicated()].iloc[:5].tolist()
            raise ValueError(f"duplicate ids, e.g. {dup}")
        if any(not i.strip() for i in self.ids):
            raise ValueError("empty id")
        self._pos = {t: i for i, t in enumerate(self.ids)}
        self._numeric = all(i.isdigit() for i in self.ids)

    # ------------------------------------------------------------------ basics
    def __len__(self) -> int:
        return len(self.ids)

    def pos(self, rid: str) -> int:
        """Row index of record ``rid``."""
        return self._pos[rid]

    @property
    def positions(self) -> dict:
        return self._pos

    def order_key(self, rid: str):
        """Deterministic sort key for member lists: numeric ids sort numerically, others by row position."""
        return int(rid) if self._numeric else self._pos[rid]

    def sort(self, ids: Iterable[str]) -> list[str]:
        return sorted(ids, key=self.order_key)

    def title(self, rid: str) -> str:
        return self.titles[self._pos[rid]]

    def text(self, rid: str) -> str:
        return self.texts[self._pos[rid]]

    def text_hashes(self, template: TextTemplate | None = None) -> list[str]:
        template = template or TextTemplate()
        return [hashlib.sha256(template.render(t, x).encode()).hexdigest() for t, x in zip(self.titles, self.texts)]

    def embedding_texts(self, template: TextTemplate | None = None) -> list[str]:
        template = template or TextTemplate()
        return [template.render(t, x) for t, x in zip(self.titles, self.texts)]

    def junk_mask(self, rule: JunkRule | None = None) -> np.ndarray:
        rule = rule or JunkRule()
        return np.array([rule.is_junk(t, x) for t, x in zip(self.titles, self.texts)], dtype=bool)

    def subset(self, idx: Sequence[int], name: str | None = None) -> "Dataset":
        idx = list(idx)
        return Dataset(ids=[self.ids[i] for i in idx], texts=[self.texts[i] for i in idx],
                       titles=[self.titles[i] for i in idx],
                       labels=None if self.labels is None else [self.labels[i] for i in idx],
                       meta=None if self.meta is None else self.meta.iloc[idx].reset_index(drop=True),
                       name=name or self.name)

    def sample(self, n: int, seed: int = 0, stratify: bool = False) -> "Dataset":
        """Uniform (or label-stratified) random subset of ``n`` records, in original order."""
        rng = np.random.default_rng(seed)
        if n >= len(self):
            return self.subset(range(len(self)))
        if stratify and self.labels is not None:
            lab = pd.Series(self.labels)
            frac = n / len(self)
            idx = []
            for _, g in lab.groupby(lab, sort=True):
                k = max(1, int(round(len(g) * frac)))
                idx += rng.choice(g.index.to_numpy(), size=min(k, len(g)), replace=False).tolist()
            if len(idx) > n:
                idx = rng.choice(np.array(idx), size=n, replace=False).tolist()
            idx = sorted(idx)
        else:
            idx = sorted(rng.choice(len(self), size=n, replace=False).tolist())
        return self.subset(idx)

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame({"id": self.ids, "title": self.titles, "text": self.texts})
        if self.labels is not None:
            df["label"] = self.labels
        if self.meta is not None:
            df = pd.concat([df, self.meta.reset_index(drop=True)], axis=1)
        return df

    # ------------------------------------------------------------------ loaders
    @classmethod
    def from_frame(cls, df: pd.DataFrame, id_col: str = "id", text_col: str = "text", title_col: str | None = None,
                   label_col: str | None = None, name: str = "dataset",
                   clean: Callable[[str], str] | None = None) -> "Dataset":
        """Build a dataset from a DataFrame. Columns other than id/text/title/label are kept as ``meta``."""
        for c in (id_col, text_col) + tuple(c for c in (title_col, label_col) if c):
            if c not in df.columns:
                raise KeyError(f"column {c!r} not found; available: {list(df.columns)}")
        clean = clean or (lambda s: s)
        ids = df[id_col].astype(str).tolist()
        texts = [clean("" if pd.isna(x) else str(x)) for x in df[text_col]]
        titles = [clean("" if pd.isna(x) else str(x)) for x in df[title_col]] if title_col else None
        labels = df[label_col].tolist() if label_col else None
        used = {id_col, text_col, title_col, label_col} - {None}
        meta = df[[c for c in df.columns if c not in used]].reset_index(drop=True)
        return cls(ids=ids, texts=texts, titles=titles, labels=labels, meta=meta if meta.shape[1] else None, name=name)

    @classmethod
    def from_mongo(cls, collection: str | list[str], **kw) -> "Dataset":
        """Load from MongoDB: one or several collections, with an optional filter ``query`` (or per-collection
        ``queries``) or an aggregation ``pipeline``. Connection from ``uri`` or the ``uri_env`` environment variable
        (default ``MONGODB_URI``). Field arguments: ``id_field`` (default ``_id``), ``text_field`` (one field or a
        list joined into one text), ``title_field``, ``label_field``, ``meta_fields``; dotted paths reach into
        nested documents. See :mod:`clustering_harness.mongo` for every option."""
        from .mongo import _from_mongo
        return _from_mongo(cls, collection, **kw)

    @classmethod
    def from_records(cls, records: Iterable[dict], id_key: str = "id", text_key: str = "text", title_key: str | None = None,
                     label_key: str | None = None, name: str = "dataset", clean: Callable[[str], str] | None = None) -> "Dataset":
        return cls.from_frame(pd.DataFrame(list(records)), id_key, text_key, title_key, label_key, name, clean)

    @classmethod
    def from_csv(cls, path: str | Path, id_col: str = "id", text_col: str = "text", title_col: str | None = None,
                 label_col: str | None = None, encoding: str = "utf-8-sig", **kw) -> "Dataset":
        sep = "\t" if str(path).lower().endswith(".tsv") else ","
        df = pd.read_csv(path, dtype={id_col: str}, encoding=encoding, keep_default_na=False, quoting=csv.QUOTE_MINIMAL, sep=sep)
        kw.setdefault("name", Path(path).stem)
        return cls.from_frame(df, id_col, text_col, title_col, label_col, **kw)

    @classmethod
    def from_json(cls, path: str | Path, id_col: str = "id", text_col: str = "text", title_col: str | None = None,
                  label_col: str | None = None, **kw) -> "Dataset":
        """JSON array of objects, or JSON Lines (``.jsonl``)."""
        path = Path(path)
        if path.suffix == ".jsonl":
            recs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        else:
            recs = json.loads(path.read_text())
            if isinstance(recs, dict):
                recs = next(v for v in recs.values() if isinstance(v, list))
        kw.setdefault("name", path.stem)
        return cls.from_frame(pd.DataFrame(recs), id_col, text_col, title_col, label_col, **kw)

    @classmethod
    def from_file(cls, path: str | Path, **kw) -> "Dataset":
        suffix = Path(path).suffix.lower()
        if suffix in (".csv", ".tsv"):
            return cls.from_csv(path, **kw)
        if suffix in (".json", ".jsonl"):
            return cls.from_json(path, **kw)
        kw.setdefault("name", Path(path).stem)
        if suffix == ".parquet":
            return cls.from_frame(pd.read_parquet(path), **kw)
        if suffix in (".xlsx", ".xlsm", ".xls"):
            sheet = kw.pop("sheet_name", 0)
            return cls.from_frame(pd.read_excel(path, sheet_name=sheet, dtype=str, keep_default_na=False), **kw)
        raise ValueError(f"unsupported file type {suffix!r}; use .csv .tsv .json .jsonl .parquet .xlsx, or Dataset.from_frame")


def parse_title_description(text: str, pattern: str = r"^\s*Title\s*:(.*?)\|kcs\|\s*Description\s*:(.*)$") -> tuple[str, str]:
    """Split ``"Title: ...|kcs| Description: ..."`` style text; returns ("", text) when it does not match."""
    m = re.match(pattern, text, re.S | re.I)
    if not m:
        return "", text
    return html.unescape(m[1]).strip(), html.unescape(m[2]).strip()
