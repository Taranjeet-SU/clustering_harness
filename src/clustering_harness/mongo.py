"""Load records from MongoDB: one collection, several collections, a filter query or an aggregation pipeline.

    from clustering_harness import Dataset
    ds = Dataset.from_mongo(
        uri_env="MONGODB_URI",                 # read the connection string from an environment variable
        database="support",
        collection=["tickets_2025", "tickets_2026"],
        query={"status": "closed", "created": {"$gte": {"$date": "2026-01-01T00:00:00Z"}}},
        text_field=["subject", "description"],  # several fields are joined into one text
        title_field="subject",
        meta_fields=["product", "priority"],
    )

Field names may be dotted paths into nested documents (``"fields.body"``). Lists are joined, nested
documents are rendered as JSON. Queries and pipelines can be given as Python objects or as MongoDB
Extended JSON strings (``{"$date": ...}``, ``{"$oid": ...}``), which is how they are written in YAML run specs.
Requires ``pymongo`` (``pip install clustering-harness[mongo]``).
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable

import pandas as pd

__all__ = ["load_mongo_frame", "get_path"]

log = logging.getLogger("clustering_harness")


def get_path(doc: Any, path: str):
    """Value at a dotted path (``a.b.c``); list elements are searched element-wise (like MongoDB does)."""
    cur = doc
    for part in path.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        elif isinstance(cur, list):
            if part.isdigit():
                i = int(part)
                cur = cur[i] if i < len(cur) else None
            else:
                vals = [get_path(x, part) for x in cur]
                cur = [v for v in vals if v is not None] or None
        else:
            return None
        if cur is None:
            return None
    return cur


def _as_text(v) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        return "\n".join(t for t in (_as_text(x) for x in v) if t)
    if isinstance(v, dict):
        from bson import json_util
        return json_util.dumps(v)
    return str(v)


def _ejson(x):
    """Extended-JSON strings (and dicts/lists written in Extended JSON) -> BSON-ready Python objects."""
    if x is None:
        return None
    from bson import json_util
    return json_util.loads(x if isinstance(x, str) else json.dumps(x))


def load_mongo_frame(collection: str | list[str], database: str | None = None, uri: str | None = None,
                     uri_env: str = "MONGODB_URI", query: dict | str | None = None, queries: dict | None = None,
                     pipeline: list | str | None = None, projection: dict | list | None = None, id_field: str = "_id",
                     text_field: str | list[str] = "text", title_field: str | None = None, label_field: str | None = None,
                     meta_fields: list[str] | None = None, sort: list | dict | None = None, limit: int | None = None,
                     text_joiner: str = "\n\n", id_with_collection: bool | None = None, client=None,
                     client_kwargs: dict | None = None) -> pd.DataFrame:
    """Read documents into a frame with columns ``id``, ``text`` and optional ``title``, ``label``,
    ``_collection`` and the requested ``meta_fields``. See the module docstring."""
    colls = [collection] if isinstance(collection, str) else list(collection)
    if not colls:
        raise ValueError("no collection given")
    if pipeline is not None and (query is not None or queries):
        raise ValueError("give either a query/queries or an aggregation pipeline, not both")
    text_fields = [text_field] if isinstance(text_field, str) else list(text_field)
    own_client = client is None
    if own_client:
        from pymongo import MongoClient
        uri = uri or os.environ.get(uri_env) or "mongodb://localhost:27017"
        client = MongoClient(uri, **(client_kwargs or {}))
    try:
        db = client[database] if database else client.get_default_database()
        if id_with_collection is None:
            id_with_collection = len(colls) > 1
        base_q = _ejson(query) or {}
        pipe = _ejson(pipeline) if pipeline is not None else None
        if isinstance(sort, dict):
            sort = list(sort.items())
        proj = projection
        if isinstance(proj, list):
            proj = {f: 1 for f in proj}
        rows = []
        for name in colls:
            left = None if limit is None else limit - len(rows)
            if left is not None and left <= 0:
                break
            coll = db[name]
            if pipe is not None:
                stages = list(pipe) + ([{"$limit": left}] if left is not None else [])
                cursor = coll.aggregate(stages, allowDiskUse=True)
            else:
                q = _ejson((queries or {}).get(name)) if queries and name in queries else base_q
                cursor = coll.find(q, proj)
                if sort:
                    cursor = cursor.sort(sort)
                if left is not None:
                    cursor = cursor.limit(left)
            n0 = len(rows)
            for d in cursor:
                rid = get_path(d, id_field)
                if rid is None:
                    raise ValueError(f"document in {name!r} has no {id_field!r} field")
                parts = [_as_text(get_path(d, f)).strip() for f in text_fields]
                row = {"id": f"{name}:{rid}" if id_with_collection else str(rid),
                       "text": text_joiner.join(p for p in parts if p)}
                if title_field:
                    row["title"] = _as_text(get_path(d, title_field))
                if label_field:
                    row["label"] = get_path(d, label_field)
                if len(colls) > 1:
                    row["_collection"] = name
                for f in meta_fields or []:
                    v = get_path(d, f)
                    row[f] = v if v is None or isinstance(v, (str, int, float, bool)) else _as_text(v)
                rows.append(row)
            log.info("mongo: %d documents from %s.%s", len(rows) - n0, db.name, name)
    finally:
        if own_client:
            client.close()
    if not rows:
        log.warning("mongo: no documents matched in %s (collections %s)", database or "default database", colls)
    df = pd.DataFrame(rows, columns=None if rows else ["id", "text"])
    if df["id"].duplicated().any():
        dup = df.loc[df["id"].duplicated(), "id"].head(5).tolist()
        raise ValueError(f"duplicate ids {dup}; use id_with_collection=True or a unique id_field")
    return df


def _from_mongo(cls, collection, *, name: str | None = None, clean: Callable[[str], str] | None = None, **kw):
    df = load_mongo_frame(collection, **kw)
    has = set(df.columns)
    return cls.from_frame(df, "id", "text", "title" if "title" in has else None, "label" if "label" in has else None,
                          name=name or (collection if isinstance(collection, str) else "+".join(collection)), clean=clean)
