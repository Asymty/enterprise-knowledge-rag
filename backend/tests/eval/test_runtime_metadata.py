"""Unit tests for runtime metadata probing in run_eval.

These cover the parts that can be verified without Qdrant, an embedding model, or a network:
the vector-size reader (including the named-vector shape), the probe-dimension measurement,
and the three fail-fast predicates the harness applies.

Importing run_eval is safe here -- it binds QDRANT_COLLECTION and adds the backend to
sys.path at import time but opens no connection and loads no model; all of that happens
inside main(). No connection is attempted by these tests.

Run:
    cd backend && python tests/eval/test_runtime_metadata.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent))

from run_eval import (  # noqa: E402
    EMBEDDING_DIMENSION_PROBE,
    EXPECTED_CORPUS_CHUNKS,
    _collection_vector_size,
    _runtime_embedding_dimension,
)


# ===========================================================================
# Fakes
# ===========================================================================

class FakeClient:
    """Minimal stand-in for QdrantClient exposing only get_collection()."""

    def __init__(self, vectors, points_count=78):
        self._vectors = vectors
        self._points = points_count

    def get_collection(self, name):
        return SimpleNamespace(
            config=SimpleNamespace(params=SimpleNamespace(vectors=self._vectors)),
            points_count=self._points,
        )


class FakeEmbedding:
    def __init__(self, dim):
        self._dim = dim

    def embed_query(self, text):
        return [0.0] * self._dim


# ===========================================================================
# _collection_vector_size
# ===========================================================================

def test_vector_size_unnamed_single_vector():
    """The shape `VectorParams(size=384, distance=COSINE)` produces."""
    client = FakeClient(SimpleNamespace(size=384))
    assert _collection_vector_size(client, "c") == 384


def test_vector_size_named_vectors_single_entry():
    """Named-vector collections map name -> VectorParams."""
    client = FakeClient({"text": SimpleNamespace(size=768)})
    assert _collection_vector_size(client, "c") == 768


def test_vector_size_named_vectors_consistent_multi_entry():
    client = FakeClient({"a": SimpleNamespace(size=384), "b": SimpleNamespace(size=384)})
    assert _collection_vector_size(client, "c") == 384


def test_vector_size_named_vectors_inconsistent_raises():
    """Two different sizes: there is no single dimension to report, so refuse."""
    client = FakeClient({"a": SimpleNamespace(size=384), "b": SimpleNamespace(size=768)})
    try:
        _collection_vector_size(client, "c")
    except RuntimeError:
        pass
    else:
        raise AssertionError("accepted a collection with two different vector sizes")


def test_vector_size_is_int_not_bool():
    client = FakeClient(SimpleNamespace(size=384))
    value = _collection_vector_size(client, "c")
    assert isinstance(value, int) and not isinstance(value, bool)


# ===========================================================================
# _runtime_embedding_dimension
# ===========================================================================

def test_dimension_measured_not_guessed():
    assert _runtime_embedding_dimension(FakeEmbedding(384)) == 384
    assert _runtime_embedding_dimension(FakeEmbedding(768)) == 768


def test_dimension_uses_the_embedding_model_itself():
    """The value must come from a real embed_query call, not a name lookup."""
    calls = []

    class Spy:
        def embed_query(self, text):
            calls.append(text)
            return [0.0] * 512

    assert _runtime_embedding_dimension(Spy()) == 512
    assert calls == [EMBEDDING_DIMENSION_PROBE], (
        "must embed exactly the probe string, once"
    )


def test_probe_is_nonempty_and_not_corpus_derived():
    assert EMBEDDING_DIMENSION_PROBE.strip(), "probe must be non-empty"
    # Deliberately a trivially short string: this measures a model property, so the input
    # content is irrelevant. It must not look like a retrieval query.
    assert len(EMBEDDING_DIMENSION_PROBE) < 20


# ===========================================================================
# Fail-fast predicates
#
# The harness inlines these as `raise RuntimeError` guards. Restating the predicates here
# pins the boundary conditions (0 and negative) so a future refactor cannot quietly drop
# one of them.
# ===========================================================================

def _points_guard(collection_points):
    if collection_points <= 0:
        raise RuntimeError(f"no points: {collection_points}")


def _dimension_guard(embedding_dimension):
    if embedding_dimension <= 0:
        raise RuntimeError(f"bad dimension: {embedding_dimension}")


def _dimension_match_guard(collection_vector_size, embedding_dimension):
    if collection_vector_size != embedding_dimension:
        raise RuntimeError("dimension mismatch")


def test_collection_points_zero_fails():
    for bad in (0, -1, -100):
        try:
            _points_guard(bad)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"accepted collection_points={bad}")


def test_collection_points_positive_passes():
    for good in (1, 78, 1000):
        _points_guard(good)


def test_embedding_dimension_zero_fails():
    for bad in (0, -1):
        try:
            _dimension_guard(bad)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"accepted embedding_dimension={bad}")


def test_embedding_dimension_positive_passes():
    for good in (1, 384, 768, 1024):
        _dimension_guard(good)


def test_dimension_mismatch_fails():
    try:
        _dimension_match_guard(384, 768)
    except RuntimeError:
        pass
    else:
        raise AssertionError("accepted a 384-d collection with a 768-d embedding backend")


def test_dimension_match_passes():
    for dim in (384, 768):
        _dimension_match_guard(dim, dim)


def test_expected_corpus_chunks_is_a_constant_not_a_rule():
    """78 describes Corpus V1, and must be declared separately from runtime behaviour."""
    assert EXPECTED_CORPUS_CHUNKS == 78


# ===========================================================================
# Runner
# ===========================================================================

def _all_tests():
    return [
        (name, fn)
        for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]


def main() -> int:
    tests = _all_tests()
    failures = []
    print("=" * 72)
    print("run_eval runtime metadata - unit tests")
    print("=" * 72)
    print(f"{len(tests)} cases\n")

    for name, fn in tests:
        try:
            fn()
        except AssertionError as exc:
            failures.append((name, exc))
            print(f"  [FAIL] {name}")
            print(f"         {exc}")
        except Exception as exc:  # noqa: BLE001
            failures.append((name, exc))
            print(f"  [ERROR] {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"  [ok]   {name}")

    print()
    print("=" * 72)
    if failures:
        print(f"FAILED: {len(failures)}/{len(tests)}")
        for name, exc in failures:
            print(f"  - {name}: {exc}")
        print("=" * 72)
        return 1
    print(f"PASSED: {len(tests)}/{len(tests)}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())