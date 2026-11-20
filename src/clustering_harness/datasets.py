"""Open benchmark datasets with gold labels, returned as :class:`~clustering_harness.data.Dataset`.

Data are downloaded from the Hugging Face Hub at a pinned revision on first use and cached under
``~/.cache/clustering_harness/datasets`` (override with ``cache_dir`` or the ``CLUSTERING_HARNESS_CACHE``
environment variable). Nothing is ever written inside the repository. Licences and citations are recorded in
:data:`DATASET_INFO`; check them before redistributing derived data.

Record ids are stable strings ``"<name>-<split>-<row>"`` where ``<row>`` is the zero-padded row index in the
original split file (before any filtering or sub-sampling), e.g. ``"banking77-test-000123"``.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import Dataset

__all__ = ["DATASET_INFO", "available_datasets", "default_cache_dir", "load_dataset"]

DATASET_INFO: dict[str, dict[str, Any]] = {
    "banking77": {
        "hf_repo": "PolyAI/banking77",
        "hf_config": None,
        "revision": "1846984189f2bc0ac51c6fc3a600fa9e1714e0c6",
        "splits": ("train", "test"),
        "default_split": "test",
        "n_labels": 77,
        "fields": "text (single-sentence online-banking customer query); no title",
        "label": "intent",
        "license": "CC-BY-4.0",
        "url": "https://huggingface.co/datasets/PolyAI/banking77",
        "citation": ("Casanueva, I., Temcinas, T., Gerz, D., Henderson, M., Vulic, I. (2020). Efficient Intent "
                     "Detection with Dual Sentence Encoders. Proceedings of the 2nd Workshop on Natural Language "
                     "Processing for Conversational AI (NLP4ConvAI), ACL 2020. arXiv:2003.04807."),
    },
    "clinc150": {
        "hf_repo": "clinc/clinc_oos",
        "hf_config": "plus",
        "revision": "155b9c710419136e17307b80d0a13e68cd46b4ec",
        "splits": ("train", "validation", "test"),
        "default_split": "test",
        "n_labels": 150,
        "fields": "text (single-utterance assistant query, 10 domains); no title",
        "label": "intent (150 in-scope intents; the out-of-scope label 'oos' is dropped unless include_oos=True)",
        "license": "CC-BY-3.0",
        "url": "https://huggingface.co/datasets/clinc/clinc_oos",
        "citation": ("Larson, S., Mahendran, A., Peper, J. J., Clarke, C., Lee, A., Hill, P., Kummerfeld, J. K., "
                     "Leach, K., Laurenzano, M. A., Tang, L., Mars, J. (2019). An Evaluation Dataset for Intent "
                     "Classification and Out-of-Scope Prediction. EMNLP-IJCNLP 2019, pp. 1311-1316."),
    },
    "support_tickets": {
        "hf_repo": "Tobi-Bueck/customer-support-tickets",
        "hf_file": "aa_dataset-tickets-multi-lang-5-2-50-version.csv",
        "revision": "ddf1c81a5475992c4fa6752bf1e8b4e31f07bbeb",
        "splits": ("train",),
        "default_split": "train",
        "n_labels": 10,
        "fields": "title = email subject (often empty), text = email body; synthetic tickets (EN/DE)",
        "label": "queue (10 routing departments) by default; label_field='type' gives the ticket type",
        "license": "CC-BY-NC-4.0 (non-commercial use only)",
        "url": "https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets",
        "citation": ("Bueck, T. Customer Support Tickets (synthetic helpdesk emails with queue, type and "
                     "priority labels). Hugging Face dataset Tobi-Bueck/customer-support-tickets."),
    },
}


def available_datasets() -> list[str]:
    """Names accepted by :func:`load_dataset`."""
    return sorted(DATASET_INFO)


def default_cache_dir() -> Path:
    """``$CLUSTERING_HARNESS_CACHE/datasets`` if set, else ``~/.cache/clustering_harness/datasets``."""
    root = os.environ.get("CLUSTERING_HARNESS_CACHE")
    base = Path(root).expanduser() if root else Path.home() / ".cache" / "clustering_harness"
    return base / "datasets"


def _make_ids(name: str, split: str, rows: np.ndarray) -> list[str]:
    width = max(6, len(str(int(rows.max())))) if rows.size else 6
    return [f"{name}-{split}-{int(r):0{width}d}" for r in rows]


def _hf_split(info: dict[str, Any], split: str, cache_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    """Load a parquet-backed HF dataset split; returns the frame and the ClassLabel names of its label column."""
    import datasets as hf_datasets

    ds = hf_datasets.load_dataset(info["hf_repo"], info["hf_config"], split=split, revision=info["revision"],
                                  cache_dir=str(cache_dir / "hf"))
    label_col = "label" if "label" in ds.features else "intent"
    names = list(ds.features[label_col].names)
    df = ds.to_pandas().rename(columns={label_col: "label_id"})
    return df, names


def _load_banking77(split: str, cache_dir: Path) -> Dataset:
    df, names = _hf_split(DATASET_INFO["banking77"], split, cache_dir)
    rows = np.arange(len(df))
    return Dataset(ids=_make_ids("banking77", split, rows), texts=df["text"].tolist(),
                   labels=[names[i] for i in df["label_id"]],
                   meta=pd.DataFrame({"label_id": df["label_id"].astype(int).to_numpy()}),
                   name=f"banking77-{split}")


def _load_clinc150(split: str, cache_dir: Path, include_oos: bool = False) -> Dataset:
    df, names = _hf_split(DATASET_INFO["clinc150"], split, cache_dir)
    df["label"] = [names[i] for i in df["label_id"]]
    rows = np.arange(len(df))
    if not include_oos:
        keep = (df["label"] != "oos").to_numpy()
        df, rows = df[keep], rows[keep]
    return Dataset(ids=_make_ids("clinc150", split, rows), texts=df["text"].tolist(), labels=df["label"].tolist(),
                   meta=pd.DataFrame({"label_id": df["label_id"].astype(int).to_numpy()}),
                   name=f"clinc150-{split}")


def _load_support_tickets(split: str, cache_dir: Path, language: str | None = "en", label_field: str = "queue") -> Dataset:
    from huggingface_hub import hf_hub_download

    info = DATASET_INFO["support_tickets"]
    path = hf_hub_download(info["hf_repo"], info["hf_file"], repo_type="dataset", revision=info["revision"],
                           cache_dir=str(cache_dir / "hub"))
    df = pd.read_csv(path)
    if label_field not in ("queue", "type", "priority"):
        raise ValueError("label_field must be 'queue', 'type' or 'priority'")
    rows = np.arange(len(df))
    keep = df[label_field].notna().to_numpy()
    if language:
        keep = keep & (df["language"] == language).to_numpy()
    df, rows = df[keep], rows[keep]
    meta_cols = [c for c in ("queue", "type", "priority", "language") if c != label_field]
    # The CSV stores line breaks as literal "\\n" sequences; restore real newlines.
    unescape = lambda s: s.replace("\\n", "\n") if isinstance(s, str) else s  # noqa: E731
    return Dataset(ids=_make_ids("support_tickets", split, rows), texts=[unescape(x) for x in df["body"]],
                   titles=[unescape(x) for x in df["subject"]], labels=df[label_field].astype(str).tolist(),
                   meta=df[meta_cols].reset_index(drop=True), name=f"support_tickets-{split}")


_LOADERS = {"banking77": _load_banking77, "clinc150": _load_clinc150, "support_tickets": _load_support_tickets}


def load_dataset(name: str, split: str | None = None, limit: int | None = None, seed: int = 0,
                 cache_dir: str | Path | None = None, **options: Any) -> Dataset:
    """Load benchmark ``name`` (see :func:`available_datasets`) as a :class:`Dataset` with gold ``labels``
    (label names, not ids).

    ``split`` defaults to the dataset's ``default_split`` (``"test"`` for banking77 / clinc150, ``"train"``
    for support_tickets, which has a single split). ``limit`` draws a uniform random subset of that many
    records with ``seed`` (original order kept). Dataset-specific ``options``: clinc150 ``include_oos``
    (default False); support_tickets ``language`` (default ``"en"``, None for all) and ``label_field``
    (``"queue"`` default, ``"type"`` or ``"priority"``).
    """
    if name not in DATASET_INFO:
        raise ValueError(f"unknown dataset {name!r}; available: {available_datasets()}")
    info = DATASET_INFO[name]
    split = split or info["default_split"]
    if split not in info["splits"]:
        raise ValueError(f"dataset {name!r} has splits {list(info['splits'])}, got {split!r}")
    cache = Path(cache_dir).expanduser() if cache_dir else default_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    ds = _LOADERS[name](split, cache, **options)
    if limit is not None and limit < len(ds):
        ds = ds.sample(int(limit), seed=seed)
    return ds
