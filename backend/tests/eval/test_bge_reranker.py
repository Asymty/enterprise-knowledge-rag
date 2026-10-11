"""M4 unit tests for the SiliconFlow BGE reranker adapter.

NO NETWORK. Every case drives ``SiliconFlowReranker`` through a fake transport, so these run
without an API key, without quota, and without the provider being up.

What is actually being pinned here is index mapping and response validation. The provider
scores documents by position and returns them out of order; a bug in mapping ``index`` back
onto the sent list pairs a document with another document's score, and the resulting ranking
is then indistinguishable from a real model output. That is the failure this suite exists to
catch, along with the retry/no-fallback policy and the pool-identity invariant.

Run:
    cd backend && python tests/eval/test_bge_reranker.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bge_reranker import (  # noqa: E402
    MAX_ATTEMPTS,
    SiliconFlowRerankError,
    SiliconFlowReranker,
    assert_pool_unchanged,
)


# ===========================================================================
# Fakes
# ===========================================================================

class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=None):
        self.status_code = status_code
        self._payload = payload
        self.text = text if text is not None else ""

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class FakeTransport:
    """Records every POST and replays a scripted list of responses."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        if not self._responses:
            raise AssertionError("FakeTransport ran out of scripted responses")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class RaisingTransport:
    def __init__(self, exc):
        self._exc = exc
        self.calls = 0

    def post(self, *a, **k):
        self.calls += 1
        raise self._exc


class Doc:
    """Stand-in for a langchain Document: only page_content and metadata are used."""

    def __init__(self, content, file_name="f.md", chunk_index=0):
        self.page_content = content
        self.metadata = {"file_name": file_name, "chunk_index": chunk_index}


def ok(pairs):
    """A 200 response given [(index, score), ...]."""
    return FakeResponse(200, {"results": [{"index": i, "relevance_score": s} for i, s in pairs]})


def make_reranker(transport, **kwargs):
    kwargs.setdefault("api_key", "test-key-not-a-real-secret")
    return SiliconFlowReranker(session=transport, **kwargs)


DOCS = [Doc("d0", "a.md"), Doc("d1", "b.md"), Doc("d2", "c.md")]


# ===========================================================================
# index mapping -- the core risk
# ===========================================================================

def test_index_mapping_pairs_each_document_with_its_own_score():
    """The provider returns results out of order; index is the ONLY thing that pairs them."""
    transport = FakeTransport([ok([(2, 0.9), (0, 0.1), (1, 0.5)])])
    reranker = make_reranker(transport)
    docs, scores = reranker.rerank_documents("q", DOCS)

    assert [d.page_content for d in docs] == ["d2", "d1", "d0"], "must sort by score desc"
    assert scores == [0.9, 0.5, 0.1]
    # The strongest check: the document that scored 0.1 must be the one at index 0.
    assert docs[-1].page_content == "d0"


def test_metadata_survives_reranking():
    transport = FakeTransport([ok([(1, 0.7), (0, 0.2), (2, 0.1)])])
    docs, _ = make_reranker(transport).rerank_documents("q", DOCS)
    assert [d.metadata["file_name"] for d in docs] == ["b.md", "a.md", "c.md"]


def test_original_document_objects_are_returned_not_copies():
    """Identity, not equality: the benchmark compares pools by object identity."""
    transport = FakeTransport([ok([(1, 0.9), (0, 0.5), (2, 0.1)])])
    docs, _ = make_reranker(transport).rerank_documents("q", DOCS)
    assert docs[0] is DOCS[1]


def test_top_k_truncates_after_sorting():
    transport = FakeTransport([ok([(2, 0.9), (0, 0.1), (1, 0.5)])])
    docs, scores = make_reranker(transport).rerank_documents("q", DOCS, top_k=2)
    assert [d.page_content for d in docs] == ["d2", "d1"]
    assert scores == [0.9, 0.5]


def test_empty_pool_short_circuits_without_a_request():
    transport = FakeTransport([])
    docs, scores = make_reranker(transport).rerank_documents("q", [])
    assert (docs, scores) == ([], [])
    assert transport.calls == [], "an empty pool must not spend an API call"


# ===========================================================================
# response validation
# ===========================================================================

def test_short_result_list_is_rejected():
    """Fewer scores than documents would silently narrow the pool."""
    transport = FakeTransport([FakeResponse(200, {"results": [{"index": 0, "relevance_score": 1.0}]})])
    try:
        make_reranker(transport).rerank_documents("q", DOCS)
    except SiliconFlowRerankError as exc:
        assert "returned 1 results for 3 documents" in str(exc)
    else:
        raise AssertionError("accepted a response with fewer results than documents")


def test_duplicate_index_is_rejected():
    transport = FakeTransport([ok([(0, 0.9), (0, 0.5), (2, 0.1)])])
    try:
        make_reranker(transport).rerank_documents("q", DOCS)
    except SiliconFlowRerankError as exc:
        assert "duplicate indices" in str(exc)
    else:
        raise AssertionError("accepted duplicate indices")


def test_out_of_range_index_is_rejected():
    for bad in (-1, 3, 99):
        transport = FakeTransport([ok([(0, 0.9), (1, 0.5), (bad, 0.1)])])
        try:
            make_reranker(transport).rerank_documents("q", DOCS)
        except SiliconFlowRerankError as exc:
            assert "outside the sent range" in str(exc), str(exc)
        else:
            raise AssertionError(f"accepted out-of-range index {bad}")


def test_non_integer_index_is_rejected():
    transport = FakeTransport([FakeResponse(200, {"results": [
        {"index": 0, "relevance_score": 0.9},
        {"index": "1", "relevance_score": 0.5},
        {"index": 2, "relevance_score": 0.1},
    ]})])
    try:
        make_reranker(transport).rerank_documents("q", DOCS)
    except SiliconFlowRerankError as exc:
        assert "non-integer index" in str(exc)
    else:
        raise AssertionError("accepted a string index")


def test_bool_index_is_rejected_even_though_bool_is_an_int():
    """True == 1 in Python, so a naive isinstance(index, int) check would accept it."""
    transport = FakeTransport([FakeResponse(200, {"results": [
        {"index": 0, "relevance_score": 0.9},
        {"index": True, "relevance_score": 0.5},
        {"index": 2, "relevance_score": 0.1},
    ]})])
    try:
        make_reranker(transport).rerank_documents("q", DOCS)
    except SiliconFlowRerankError as exc:
        assert "non-integer index" in str(exc)
    else:
        raise AssertionError("accepted a bool index")


def test_non_finite_score_is_rejected():
    """inf would sort to the top and win every comparison; NaN would sort unpredictably."""
    for bad in (float("inf"), float("-inf"), float("nan")):
        transport = FakeTransport([FakeResponse(200, {"results": [
            {"index": 0, "relevance_score": 0.9},
            {"index": 1, "relevance_score": bad},
            {"index": 2, "relevance_score": 0.1},
        ]})])
        try:
            make_reranker(transport).rerank_documents("q", DOCS)
        except SiliconFlowRerankError as exc:
            assert "non-finite score" in str(exc)
        else:
            raise AssertionError(f"accepted a non-finite score {bad}")


def test_non_numeric_score_is_rejected():
    transport = FakeTransport([FakeResponse(200, {"results": [
        {"index": 0, "relevance_score": 0.9},
        {"index": 1, "relevance_score": "high"},
        {"index": 2, "relevance_score": 0.1},
    ]})])
    try:
        make_reranker(transport).rerank_documents("q", DOCS)
    except SiliconFlowRerankError as exc:
        assert "non-numeric score" in str(exc)
    else:
        raise AssertionError("accepted a string score")


def test_missing_results_key_is_rejected():
    transport = FakeTransport([FakeResponse(200, {"data": []})])
    try:
        make_reranker(transport).rerank_documents("q", DOCS)
    except SiliconFlowRerankError as exc:
        assert "no 'results' list" in str(exc)
    else:
        raise AssertionError("accepted a response with no results key")


def test_non_json_body_is_rejected():
    transport = FakeTransport([FakeResponse(200, payload=None, text="<html>502</html>")])
    try:
        make_reranker(transport).rerank_documents("q", DOCS)
    except SiliconFlowRerankError as exc:
        assert "not valid JSON" in str(exc)
    else:
        raise AssertionError("accepted a non-JSON body")


def test_alternate_score_key_is_accepted():
    """`score` is the older field name; both are read so a schema rename is not fatal."""
    transport = FakeTransport([FakeResponse(200, {"results": [
        {"index": 0, "score": 0.2},
        {"index": 1, "score": 0.8},
        {"index": 2, "score": 0.4},
    ]})])
    docs, scores = make_reranker(transport).rerank_documents("q", DOCS)
    assert [d.page_content for d in docs] == ["d1", "d2", "d0"]


def test_pool_larger_than_provider_limit_is_rejected_before_any_request():
    transport = FakeTransport([])
    reranker = make_reranker(transport)
    try:
        reranker.rerank_documents("q", [Doc(f"d{i}") for i in range(101)])
    except SiliconFlowRerankError as exc:
        assert "exceeds the provider limit" in str(exc)
    else:
        raise AssertionError("accepted a pool beyond the provider limit")
    assert transport.calls == [], "the limit must be enforced before spending a request"


# ===========================================================================
# transport: auth, retries, timeouts, no fallback
# ===========================================================================

def test_missing_api_key_fails_fast():
    try:
        SiliconFlowReranker(api_key="", session=FakeTransport([]))
    except SiliconFlowRerankError as exc:
        assert "SILICONFLOW_API_KEY is not set" in str(exc)
    else:
        raise AssertionError("constructed with an empty API key")


def test_401_reports_authentication_failure_without_retrying():
    transport = FakeTransport([FakeResponse(401, text="invalid token")])
    reranker = make_reranker(transport)
    try:
        reranker.rerank_documents("q", DOCS)
    except SiliconFlowRerankError as exc:
        assert "authentication failed" in str(exc)
        assert "invalid token" in str(exc), "the provider detail must be surfaced"
    else:
        raise AssertionError("accepted a 401")
    assert len(transport.calls) == 1, "a 401 must not be retried"
    assert reranker.stats.retries == 0
    assert reranker.stats.failures == 1


def test_429_retries_then_reports_rate_limit():
    transport = FakeTransport([FakeResponse(429, text="rate limited")] * MAX_ATTEMPTS)
    reranker = make_reranker(transport)
    try:
        reranker.rerank_documents("q", DOCS)
    except SiliconFlowRerankError as exc:
        assert "rate limit or quota exhausted" in str(exc)
    else:
        raise AssertionError("accepted a persistent 429")
    assert len(transport.calls) == MAX_ATTEMPTS, "retries must be bounded"
    assert reranker.stats.retries == MAX_ATTEMPTS - 1


def test_500_retries_then_succeeds_are_counted():
    transport = FakeTransport([FakeResponse(503, text="upstream"), ok([(0, 0.3), (1, 0.9), (2, 0.1)])])
    reranker = make_reranker(transport)
    docs, _ = reranker.rerank_documents("q", DOCS)
    assert [d.page_content for d in docs] == ["d1", "d0", "d2"]
    assert reranker.stats.retries == 1
    assert reranker.stats.failures == 0
    assert reranker.stats.requests == 1


def test_400_is_not_retried():
    """A malformed request cannot be fixed by sending it again."""
    transport = FakeTransport([FakeResponse(400, text="bad request")])
    reranker = make_reranker(transport)
    try:
        reranker.rerank_documents("q", DOCS)
    except SiliconFlowRerankError as exc:
        assert "HTTP 400" in str(exc)
    else:
        raise AssertionError("accepted a 400")
    assert len(transport.calls) == 1


def test_timeout_raises_after_bounded_attempts():
    transport = RaisingTransport(TimeoutError("read timed out"))
    reranker = make_reranker(transport)
    try:
        reranker.rerank_documents("q", DOCS)
    except SiliconFlowRerankError as exc:
        assert "after 3 attempts" in str(exc)
        assert "TimeoutError" in str(exc)
    else:
        raise AssertionError("swallowed a timeout")
    assert transport.calls == MAX_ATTEMPTS
    assert reranker.stats.failures == 1


def test_connection_error_never_falls_back_to_the_local_model():
    """The failure must propagate. A fallback would fabricate the experiment's result."""
    transport = RaisingTransport(ConnectionError("no route to host"))
    try:
        make_reranker(transport).rerank_documents("q", DOCS)
    except SiliconFlowRerankError:
        pass
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(f"raised an unexpected {type(exc).__name__}: {exc}")
    else:
        raise AssertionError("a connection failure silently produced a ranking")


def test_no_retry_state_is_leaked_across_calls():
    transport = FakeTransport([
        FakeResponse(503, text="x"), ok([(0, 0.9), (1, 0.5), (2, 0.1)]),
        ok([(0, 0.1), (1, 0.5), (2, 0.9)]),
    ])
    reranker = make_reranker(transport)
    reranker.rerank_documents("q", DOCS)
    reranker.rerank_documents("q", DOCS)
    assert reranker.stats.retries == 1, "the retry count must not double-count"
    assert reranker.stats.requests == 2


# ===========================================================================
# request construction
# ===========================================================================

def test_request_carries_query_documents_and_model():
    transport = FakeTransport([ok([(0, 0.9), (1, 0.5), (2, 0.1)])])
    make_reranker(transport, model="BAAI/bge-reranker-v2-m3").rerank_documents("中文问题", DOCS)
    body = transport.calls[0]["json"]
    assert body["query"] == "中文问题"
    assert body["documents"] == ["d0", "d1", "d2"]
    assert body["model"] == "BAAI/bge-reranker-v2-m3"
    assert transport.calls[0]["url"].endswith("/rerank")


def test_authorization_header_is_sent_but_never_in_the_body():
    transport = FakeTransport([ok([(0, 0.9), (1, 0.5), (2, 0.1)])])
    make_reranker(transport, api_key="secret-abc").rerank_documents("q", DOCS)
    assert transport.calls[0]["headers"]["Authorization"] == "Bearer secret-abc"
    assert "secret-abc" not in str(transport.calls[0]["json"])


def test_latency_is_recorded():
    transport = FakeTransport([ok([(0, 0.9), (1, 0.5), (2, 0.1)])])
    reranker = make_reranker(transport)
    reranker.rerank_documents("q", DOCS)
    assert reranker.stats.latency_ms > 0.0
    assert len(reranker.stats.per_call_latency_ms) == 1


# ===========================================================================
# pool identity invariant
# ===========================================================================

def test_pool_unchanged_accepts_a_permutation():
    assert_pool_unchanged(DOCS[::-1], DOCS, [0.1, 0.5, 0.9])


def test_pool_unchanged_rejects_a_truncated_output():
    try:
        assert_pool_unchanged(DOCS[:2], DOCS, [0.9, 0.5])
    except SiliconFlowRerankError as exc:
        assert "pool of 3" in str(exc)
    else:
        raise AssertionError("accepted a truncated pool")


def test_pool_unchanged_rejects_a_substituted_document():
    """A same-file-name but different chunk must still count as a substitution."""
    substituted = [DOCS[0], DOCS[1], Doc("d2-SUBSTITUTED", "c.md")]
    try:
        assert_pool_unchanged(substituted, DOCS, [0.1, 0.5, 0.9])
    except SiliconFlowRerankError as exc:
        assert "dropped" in str(exc)
    else:
        raise AssertionError("accepted a substituted candidate")


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
    print("=" * 74)
    print("M4 SiliconFlow BGE reranker adapter - unit tests (no network)")
    print("=" * 74)
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
    print("=" * 74)
    if failures:
        print(f"FAILED: {len(failures)}/{len(tests)}")
        for name, exc in failures:
            print(f"  - {name}: {exc}")
        print("=" * 74)
        return 1
    print(f"PASSED: {len(tests)}/{len(tests)}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())