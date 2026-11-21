"""MongoDB loader. Pure helpers always run; the end-to-end tests need a server: set MONGODB_TEST_URI
(e.g. ``docker run --rm -d -p 27017:27017 mongo:7`` and ``MONGODB_TEST_URI=mongodb://localhost:27017``)."""
from __future__ import annotations

import datetime as dt
import os
import uuid

import pytest

pytest.importorskip("bson")

from clustering_harness import Dataset  # noqa: E402
from clustering_harness.cli import load_data  # noqa: E402
from clustering_harness.mongo import _ejson, get_path  # noqa: E402


def test_get_path_nested_and_lists():
    d = {"a": {"b": "x"}, "c": [{"t": "one"}, {"t": "two"}, {"u": 1}], "n": None}
    assert get_path(d, "a.b") == "x"
    assert get_path(d, "c.t") == ["one", "two"]
    assert get_path(d, "c.1.t") == "two"
    assert get_path(d, "missing.x") is None and get_path(d, "n.x") is None


def test_extended_json_queries():
    from bson import ObjectId
    q = _ejson('{"created": {"$gte": {"$date": "2026-01-01T00:00:00Z"}}, "_id": {"$oid": "65a000000000000000000001"}}')
    assert isinstance(q["created"]["$gte"], dt.datetime) and isinstance(q["_id"], ObjectId)
    assert _ejson({"status": "closed"}) == {"status": "closed"}      # YAML dicts pass through


URI = os.environ.get("MONGODB_TEST_URI")
needs_server = pytest.mark.skipif(not URI, reason="set MONGODB_TEST_URI to run against a MongoDB server")


@pytest.fixture
def db():
    from pymongo import MongoClient
    client = MongoClient(URI, serverSelectionTimeoutMS=3000)
    name = f"ch_test_{uuid.uuid4().hex[:8]}"
    d = client[name]
    d.tickets_a.insert_many([
        {"_id": 1, "subject": "Cannot log in", "body": {"text": "SSO loop after password reset"}, "status": "closed",
         "product": "portal", "intent": "login", "created": dt.datetime(2026, 2, 1)},
        {"_id": 2, "subject": "Card declined", "body": {"text": "payment fails at checkout"}, "status": "open",
         "product": "billing", "intent": "payment", "created": dt.datetime(2026, 3, 1)},
        {"_id": 3, "subject": "Export broken", "body": {"text": ["CSV export", "times out"]}, "status": "closed",
         "product": "reports", "intent": "export", "created": dt.datetime(2025, 12, 1)},
    ])
    d.tickets_b.insert_many([
        {"_id": 1, "subject": "Login again", "body": {"text": "MFA code not accepted"}, "status": "closed", "product": "portal",
         "intent": "login", "created": dt.datetime(2026, 4, 1)},
    ])
    yield name
    client.drop_database(name)
    client.close()


@needs_server
def test_single_collection_with_query_and_fields(db):
    ds = Dataset.from_mongo("tickets_a", uri=URI, database=db, query={"status": "closed"}, text_field=["subject", "body.text"],
                            title_field="subject", label_field="intent", meta_fields=["product"], sort=[("_id", 1)])
    assert ds.ids == ["1", "3"]
    assert ds.texts[0] == "Cannot log in\n\nSSO loop after password reset"
    assert ds.texts[1] == "Export broken\n\nCSV export\ntimes out"         # list field joined
    assert ds.titles == ["Cannot log in", "Export broken"] and ds.labels == ["login", "export"]
    assert ds.meta["product"].tolist() == ["portal", "reports"]


@needs_server
def test_multiple_collections_prefix_ids_and_record_source(db):
    ds = Dataset.from_mongo(["tickets_a", "tickets_b"], uri=URI, database=db, text_field="subject", sort={"_id": 1})
    assert ds.ids == ["tickets_a:1", "tickets_a:2", "tickets_a:3", "tickets_b:1"]
    assert ds.meta["_collection"].tolist() == ["tickets_a"] * 3 + ["tickets_b"]
    with pytest.raises(ValueError, match="duplicate ids"):
        Dataset.from_mongo(["tickets_a", "tickets_b"], uri=URI, database=db, text_field="subject", id_with_collection=False)


@needs_server
def test_per_collection_queries_extended_json_and_limit(db):
    ds = Dataset.from_mongo(["tickets_a", "tickets_b"], uri=URI, database=db, text_field="subject",
                            queries={"tickets_a": '{"created": {"$gte": {"$date": "2026-01-01T00:00:00Z"}}}',
                                     "tickets_b": {"status": "open"}})
    assert sorted(ds.ids) == ["tickets_a:1", "tickets_a:2"]
    assert len(Dataset.from_mongo(["tickets_a", "tickets_b"], uri=URI, database=db, text_field="subject", limit=2)) == 2


@needs_server
def test_aggregation_pipeline(db):
    ds = Dataset.from_mongo("tickets_a", uri=URI, database=db, id_field="_id", text_field="t",
                            pipeline=[{"$match": {"product": {"$ne": "billing"}}},
                                      {"$project": {"t": {"$concat": ["$subject", " / ", "$product"]}}}, {"$sort": {"_id": 1}}])
    assert ds.texts == ["Cannot log in / portal", "Export broken / reports"]
    with pytest.raises(ValueError, match="not both"):
        Dataset.from_mongo("tickets_a", uri=URI, database=db, query={"a": 1}, pipeline=[])


@needs_server
def test_run_spec_mongo_source_reads_uri_from_env(db, monkeypatch):
    monkeypatch.setenv("MY_MONGO", URI)
    ds = load_data({"mongo": {"uri_env": "MY_MONGO", "database": db, "collections": ["tickets_a"], "query": {"status": "closed"},
                              "text_field": ["subject", "body.text"]}})
    assert sorted(ds.ids) == ["1", "3"] and ds.name == "tickets_a"


def test_unset_uri_variable_is_an_error(monkeypatch):
    monkeypatch.delenv("NO_SUCH_MONGO_URI", raising=False)
    with pytest.raises(ValueError, match="set the NO_SUCH_MONGO_URI environment variable"):
        Dataset.from_mongo("tickets", uri_env="NO_SUCH_MONGO_URI", database="x")


@needs_server
def test_missing_collection_is_an_error(db):
    with pytest.raises(ValueError, match="'tickets_c'"):
        Dataset.from_mongo(["tickets_a", "tickets_c"], uri=URI, database=db, text_field="subject")


@needs_server
def test_misspelled_field_is_an_error(db):
    with pytest.raises(ValueError, match="'subjct' \\(did you mean 'subject'\\?\\)"):
        Dataset.from_mongo("tickets_a", uri=URI, database=db, text_field=["subjct"])
    with pytest.raises(ValueError, match="'titel'"):
        Dataset.from_mongo("tickets_a", uri=URI, database=db, text_field="subject", title_field="titel")


def test_unreachable_server_is_a_quick_one_line_error():
    pytest.importorskip("pymongo")
    with pytest.raises(ValueError, match=r"cannot reach MongoDB \(127.0.0.1:27011\)"):
        Dataset.from_mongo("t", uri="mongodb://u:secret@127.0.0.1:27011/", database="d",
                           client_kwargs={"serverSelectionTimeoutMS": 300})


@needs_server
def test_misspelled_meta_field_is_an_error(db):
    with pytest.raises(ValueError, match="'prodcut'"):
        Dataset.from_mongo("tickets_a", uri=URI, database=db, text_field="subject", meta_fields=["prodcut"])
