"""Unit tests for oracle pool semantics and the rerank invariance.

The bug these pin down: `_question_result` built its oracle `pool` from whatever `documents`
it was handed. Correct for vector and hybrid, wrong for hybrid_rerank, whose `documents` is
the reranked Top-5. The result was FileOracle@10 0.940 vs 0.800 and EvidenceOracle@10
0.820 vs 0.520 between two strategies that share one candidate pool -- a phantom reranker
regression that was purely a wrong-object bug.

No Qdrant, no embedding model, no network. `_question_result` is pure.

Run:
    cd backend && python tests/eval/test_oracle_pool_semantics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from run_eval import _question_result  # noqa: E402


def doc(file_name, content, chunk_index=0, total_chunks=1):
    class _D:
        pass

    d = _D()
    d.page_content = content
    d.metadata = {
        "file_name": file_name,
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
    }
    return d


QUESTION = {
    "id": "q001",
    "question": "上海住宿上限？",
    "domain": "finance",
    "category": "numeric",
    "difficulty": "medium",
    "relevant_files": ["travel_policy.md"],
    "relevant_anchors": ["不超过每晚 500 元"],
}

ANCHOR_TEXT = "## 上海住宿标准\n普通员工赴上海出差，住宿标准不超过每晚 500 元。"
OTHER_TEXT = "## 北京住宿标准\n普通员工赴北京出差，住宿标准不超过每晚 550 元。"

# A 10-item pool whose correct chunk sits at rank 7 -- outside the Top-5, so a rerank that
# truncates to 5 would lose it and report a different oracle.
POOL = (
    [doc("other_%d.md" % i, OTHER_TEXT) for i in range(6)]
    + [doc("travel_policy.md", ANCHOR_TEXT)]
    + [doc("tail_%d.md" % i, OTHER_TEXT) for i in range(3)]
)
POOL_SCORES = [0.9 - 0.02 * i for i in range(10)]


def _result(documents, scores, pool_docs=None, pool_scores=None):
    return _question_result(
        QUESTION,
        documents,
        scores,
        {"total_retrieval_ms": 1.0},
        {},  # context_by_file
        oracle_pool_documents=pool_docs,
        oracle_pool_scores=pool_scores,
    )


# ===========================================================================
# pool vs retrieved
# ===========================================================================

def test_default_pool_is_the_documents_argument():
    """vector/hybrid: output IS the pool, so the default must reproduce it."""
    r = _result(POOL, POOL_SCORES)
    assert len(r["pool"]) == 10
    assert len(r["retrieved"]) == 5
    assert [p["file_name"] for p in r["pool"]] == [d.metadata["file_name"] for d in POOL]


def test_explicit_pool_overrides_the_documents_argument():
    """rerank: retrieved is the reranked Top-5 while pool stays the shared Top-10."""
    reranked_docs = [doc("travel_policy.md", ANCHOR_TEXT)] + [
        doc("x_%d.md" % i, OTHER_TEXT) for i in range(4)
    ]
    reranked_scores = [0.99, 0.9, 0.8, 0.7, 0.6]
    r = _result(reranked_docs, reranked_scores, POOL, POOL_SCORES)
    assert len(r["retrieved"]) == 5
    assert len(r["pool"]) == 10, "pool must come from the shared candidate pool, not the rerank output"
    assert [p["file_name"] for p in r["pool"]] == [d.metadata["file_name"] for d in POOL]


def test_retrieved_reflects_the_strategy_output_not_the_pool():
    reranked_docs = [doc("x_%d.md" % i, OTHER_TEXT) for i in range(5)]
    r = _result(reranked_docs, [0.5] * 5, POOL, POOL_SCORES)
    assert [x["file_name"] for x in r["retrieved"]] == ["x_%d.md" % i for i in range(5)]
    assert r["metrics"]["recall@5"] == 0.0, "reranked output has no relevant file"
    assert [p["file_name"] for p in r["pool"]] == [d.metadata["file_name"] for d in POOL]


# ===========================================================================
# the invariant: oracle reads the pool, so it survives truncation
# ===========================================================================

def test_oracle_is_unchanged_by_truncating_the_output():
    """Ranking metrics move; oracles must not."""
    full_output = _result(POOL, POOL_SCORES)
    assert full_output["metrics"]["file_oracle@10"] == 1.0
    assert full_output["metrics"]["evidence_oracle@10"] == 1.0

    # A rerank that keeps only the (irrelevant) head of the pool.
    truncated_docs = POOL[:5]
    truncated = _result(truncated_docs, POOL_SCORES[:5], POOL, POOL_SCORES)
    assert truncated["metrics"]["recall@5"] == 0.0, "ranking metric should collapse"
    assert truncated["metrics"]["file_oracle@10"] == 1.0, "oracle must still see the pool"
    assert truncated["metrics"]["evidence_oracle@10"] == 1.0


def test_the_wrong_object_produces_the_phantom_difference():
    """Regression guard: reproduce the original bug and show it diverges."""
    correct = _result(POOL[:5], POOL_SCORES[:5], POOL, POOL_SCORES)
    buggy = _result(POOL[:5], POOL_SCORES[:5])  # no pool override -> pool = truncated output
    assert len(buggy["pool"]) == 5
    assert correct["metrics"]["file_oracle@10"] != buggy["metrics"]["file_oracle@10"], (
        "the bug must be visible: correct pool says 1.0, truncated pool says 0.0"
    )


def test_hybrid_and_rerank_agree_when_both_use_the_shared_pool():
    hybrid_result = _result(POOL, POOL_SCORES)
    reranked_docs = [POOL[0], POOL[1], POOL[2], POOL[3], POOL[4]]
    rerank_result = _result(reranked_docs, POOL_SCORES[:5], POOL, POOL_SCORES)
    assert (
        hybrid_result["metrics"]["file_oracle@10"]
        == rerank_result["metrics"]["file_oracle@10"]
    )
    assert (
        hybrid_result["metrics"]["evidence_oracle@10"]
        == rerank_result["metrics"]["evidence_oracle@10"]
    )


def test_oracle_still_responds_to_a_different_pool():
    """The invariant must not be achieved by making the oracle constant."""
    irrelevant_pool = [doc("n_%d.md" % i, OTHER_TEXT) for i in range(10)]
    a = _result(POOL, POOL_SCORES)
    b = _result(POOL, POOL_SCORES, irrelevant_pool, POOL_SCORES)
    assert a["metrics"]["file_oracle@10"] == 1.0
    assert b["metrics"]["file_oracle@10"] == 0.0
    assert a["metrics"]["evidence_oracle@10"] == 1.0
    assert b["metrics"]["evidence_oracle@10"] == 0.0


def test_pool_longer_than_oracle_k_is_capped():
    long_pool = POOL + [doc("extra_%d.md" % i, OTHER_TEXT) for i in range(4)]
    r = _result(POOL, POOL_SCORES, long_pool, POOL_SCORES + [0.1] * 4)
    assert len(r["pool"]) == 10, "pool is capped at ORACLE_K"


def test_pool_shorter_than_oracle_k_is_tolerated():
    short_pool = POOL[:3]
    r = _result(POOL, POOL_SCORES, short_pool, POOL_SCORES[:3])
    assert len(r["pool"]) == 3
    assert r["metrics"]["evidence_oracle@10"] == 0.0


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
    print("oracle pool semantics - unit tests")
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