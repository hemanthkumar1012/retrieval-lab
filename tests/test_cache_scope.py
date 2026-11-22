"""Cache scoping and invalidation, without a database.

The ingest tests replace the database, embedding and cache modules with small stubs,
so they run in the light unit-test environment (no numpy, psycopg or onnxruntime).
"""
import importlib
import sys
import types

import pytest

from app.cache_scope import DEFAULT_TENANT, normalise_tenant, scope_for


def test_tenants_get_separate_scopes():
    assert scope_for("recursive_256", "hybrid", "acme") != scope_for("recursive_256", "hybrid", "globex")


def test_missing_tenant_means_public():
    assert normalise_tenant(None) == DEFAULT_TENANT
    assert normalise_tenant("") == DEFAULT_TENANT
    assert scope_for("fixed_128", "dense") == "public|fixed_128|dense"


def test_retrieval_config_is_still_part_of_the_scope():
    assert scope_for("fixed_128", "dense", "acme") != scope_for("fixed_128", "hybrid", "acme")


@pytest.mark.parametrize("bad", ["acme|public", "a b", "x" * 65, "../etc"])
def test_tenant_cannot_break_out_of_its_scope(bad):
    with pytest.raises(ValueError):
        normalise_tenant(bad)


def _stub(name, **attrs):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    return module


@pytest.fixture
def ingest_module(monkeypatch):
    """Import app.ingest with its heavy dependencies replaced; returns (module, clear calls)."""
    import app

    cleared = []
    cache_stub = _stub("app.cache", clear=lambda: cleared.append(True) or 0)
    monkeypatch.setitem(sys.modules, "app.cache", cache_stub)
    monkeypatch.setattr(app, "cache", cache_stub, raising=False)
    monkeypatch.setitem(sys.modules, "app.db", _stub("app.db", conn=None, open_pool=None, close_pool=None))
    monkeypatch.setitem(sys.modules, "app.embeddings", _stub("app.embeddings", embed_docs=None))
    monkeypatch.delitem(sys.modules, "app.ingest", raising=False)
    module = importlib.import_module("app.ingest")
    yield module, cleared
    sys.modules.pop("app.ingest", None)


def test_ingest_clears_cache_when_documents_change(ingest_module, monkeypatch):
    module, cleared = ingest_module
    monkeypatch.setattr(module, "_flush", lambda batch, strategy: 3)
    assert module.ingest([{"id": "d1", "text": "new text"}], "fixed_128") == 3
    assert cleared == [True]


def test_ingest_with_nothing_new_keeps_cache(ingest_module, monkeypatch):
    module, cleared = ingest_module
    monkeypatch.setattr(module, "_flush", lambda batch, strategy: 0)
    assert module.ingest([{"id": "d1", "text": ""}], "fixed_128") == 0
    assert cleared == []
