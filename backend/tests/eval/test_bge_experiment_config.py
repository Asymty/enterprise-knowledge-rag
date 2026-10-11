"""M4 static audit: the experiment's binding and comparator must be what they claim.

NO network, NO Qdrant, NO embedding model. Everything here is either a module constant, a
pure function, or a read of M3's saved JSON.

Why this exists separately from run_eval_bge.py: the expensive run can only fail loudly at
the end. These checks fail in a second, and they cover the mistakes that would make the run
misleading rather than merely broken -- a wrong collection, a comparator that recomputes an
oracle it cannot recompute, or a binding silently overwritten by an import.

Run:
    cd backend && python tests/eval/test_bge_experiment_config.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))

import run_eval_bge as m4  # noqa: E402
import dataset_utils  # noqa: E402


# ===========================================================================
# Runtime binding
# ===========================================================================

def test_collection_binding_holds_after_all_imports():
    """The load-bearing one: importing run_eval re-sets QDRANT_COLLECTION, so the M4 module
    must re-assert it. If this fails, the run evaluates the WRONG collection."""
    from app.core.config import settings

    assert os.environ["QDRANT_COLLECTION"] == m4.COLLECTION, (
        "run_eval's import-time assignment clobbered the M4 collection binding"
    )
    assert m4.COLLECTION == "enterprise_rag_eval_multilingual_v1"
    assert settings.qdrant_collection == m4.COLLECTION


def test_embedding_binding_holds_after_all_imports():
    from app.core.config import settings

    assert os.environ["EMBEDDING_MODEL"] == m4.EMBEDDING_MODEL
    assert m4.EMBEDDING_MODEL == "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    assert settings.embedding_model == m4.EMBEDDING_MODEL


def test_vector_size_alone_cannot_prove_compatibility():
    """Both MiniLM models are 384-d, so dimension equality is necessary but NOT sufficient.

    This pins the reasoning behind comparing the model NAME as well: if a future edit drops
    the name check on the grounds that the dimension matches, this test documents why that
    would be wrong.
    """
    m3 = m4._load_m3_baseline()
    facts = m4._m3_experiment_facts(m3)
    # M3's own numbers show two DIFFERENT models sharing one dimension, which is precisely
    # why the name comparison is load-bearing and the dimension alone is not.
    assert facts["embedding_dimension"] == 384
    assert m4.base.EXPECTED_CORPUS_CHUNKS == 78
    assert m4.EMBEDDING_MODEL != "all-MiniLM-L6-v2"


# ===========================================================================
# Frozen constants
# ===========================================================================

def test_frozen_experiment_constants():
    assert m4.CANDIDATE_POOL_K == 10
    assert m4.EVALUATION_K_MAX == 5
    assert m4.ORACLE_K == 10
    assert m4.EXPECTED_CORPUS_CHUNKS == 78
    assert m4.EXPECTED_QUESTIONS == 50


def test_constants_are_not_silently_redefined_from_the_base_harness():
    """M4 must share the base harness's pool/eval/oracle geometry, not invent its own."""
    assert m4.CANDIDATE_POOL_K == m4.base.CANDIDATE_POOL_K
    assert m4.EVALUATION_K_MAX == m4.base.EVALUATION_K_MAX
    assert m4.ORACLE_K == m4.base.ORACLE_K
    assert m4.EXPECTED_CORPUS_CHUNKS == m4.base.EXPECTED_CORPUS_CHUNKS


def test_reranker_identity():
    assert m4.RERANKER_MODEL == "BAAI/bge-reranker-v2-m3"
    assert m4.RERANKER_BACKEND == "SiliconFlow API"


def test_frozen_dataset_loads_and_is_the_expected_size():
    dataset, hashes = dataset_utils.load_frozen_dataset()
    assert len(dataset["questions"]) == m4.EXPECTED_QUESTIONS
    assert all(hashes["canonical"]["match"].values())


# ===========================================================================
# M3 baseline: the numbers this run is judged against
# ===========================================================================

def test_m3_saved_json_matches_the_designed_constants():
    m3 = m4._load_m3_baseline()
    verified = m4._verify_m3_constants(m3)
    assert verified["m3_constants_match_saved_json"] is True
    # Spot-check the values the milestone brief quotes, read from the SAVED file.
    assert verified["values"]["hybrid_recall@1"] == 0.565
    assert verified["values"]["hybrid_recall@5"] == 0.805
    assert verified["values"]["hybrid_evidence_recall@5"] == 0.655
    assert verified["values"]["hybrid_file_oracle@10"] == 0.94
    assert verified["values"]["hybrid_evidence_oracle@10"] == 0.82
    assert verified["values"]["hybrid_rerank_recall@1"] == 0.365
    assert verified["values"]["hybrid_rerank_recall@5"] == 0.755
    assert verified["values"]["hybrid_rerank_evidence_recall@5"] == 0.475


def test_m3_runtime_facts_match_the_planned_binding():
    m3 = m4._load_m3_baseline()
    facts = m4._m3_experiment_facts(m3)
    assert facts["collection"] == m4.COLLECTION, "M3 ran on a different collection than M4 plans"
    assert facts["embedding_model"] == m4.EMBEDDING_MODEL
    assert facts["collection_points"] == 78
    assert facts["collection_vector_size"] == 384
    assert facts["embedding_dimension"] == 384
    assert facts["candidate_pool_k"] == 10


def test_a_tampered_m3_constant_is_rejected():
    """Guard: the constants must be checked against the file, not trusted."""
    saved = dict(m4.M3_EXPECTED)
    try:
        m4.M3_EXPECTED = {**saved, "hybrid_recall@1": 0.999}
        m3 = m4._load_m3_baseline()
        try:
            m4._verify_m3_constants(m3)
        except RuntimeError as exc:
            assert "do not match the constants" in str(exc)
        else:
            raise AssertionError("a wrong expectation was accepted over the saved file")
    finally:
        m4.M3_EXPECTED = saved


def test_m3_has_no_snippet_source_so_evidence_oracle_cannot_be_recomputed():
    """The brief's constraint, asserted against the actual saved file.

    If this ever becomes False (i.e. M3's JSON gained full text), the M4 comparator could
    legitimately recompute EvidenceOracle instead of comparing stored values -- so this test
    is the tripwire that tells a future editor the situation changed.
    """
    m3 = m4._load_m3_baseline()
    question = m3["hybrid"]["questions"][0]
    assert "snippet_source" not in question["retrieved"][0], (
        "M3's saved retrieved records now carry snippet_source; the M4 comparator's "
        "stored-value comparison could be upgraded to a recomputation."
    )
    assert "snippet_source" not in question["pool"][0], (
        "M3's saved pool records now carry snippet_source."
    )
    # The fields the comparator DOES rely on must be present.
    for key in ("file_name", "chunk_index"):
        assert key in question["pool"][0]


def test_m3_pool_identity_extraction_uses_only_saved_fields():
    m3 = m4._load_m3_baseline()
    pools = m4._m3_pool_identity(m3["hybrid"])
    assert len(pools) == 50
    first = pools["q001"] if "q001" in pools else next(iter(pools.values()))
    assert len(first) == 10
    assert set(first[0].keys()) == {"file_name", "chunk_index"}


# ===========================================================================
# Results directory isolation
# ===========================================================================

def test_results_dir_does_not_collide_with_existing_experiments():
    assert m4.RESULTS_DIR.name == "bge_reranker_v2_m3_v1"
    parent = m4.RESULTS_DIR.parent
    existing = {"bm25_zh_v1", "multilingual_embed_v1"}
    assert m4.RESULTS_DIR.name not in existing
    # The directories this run must never overwrite still exist and are untouched.
    for name in existing:
        assert (parent / name).is_dir(), f"pre-existing results dir {name} is missing"


def test_existing_experiment_files_are_not_the_write_target():
    targets = {m4.RESULTS_DIR / f"{name}.json" for name in m4.STRATEGY_NAMES}
    targets.add(m4.RESULTS_DIR / "summary.json")
    targets.add(m4.RESULTS_DIR / "comparison.json")
    protected = {
        m4.EVAL_DIR / "results" / "summary.json",
        m4.EVAL_DIR / "results" / "hybrid.json",
        m4.EVAL_DIR / "results" / "vector.json",
        m4.EVAL_DIR / "results" / "hybrid_rerank.json",
    }
    for name in ("bm25_zh_v1", "multilingual_embed_v1"):
        protected |= {
            m4.EVAL_DIR / "results" / name / f
            for f in ("summary.json", "vector.json", "hybrid.json", "hybrid_rerank.json", "comparison.json")
        }
    assert not (targets & protected), f"overlap: {targets & protected}"


def test_production_collection_is_never_a_write_target():
    assert m4.PRODUCTION_COLLECTION == "enterprise_rag"
    assert m4.PRODUCTION_COLLECTION != m4.COLLECTION


# ===========================================================================
# Comparison builder (pure)
# ===========================================================================

def _fake_summary():
    zero = {
        m: 0.0
        for m in (
            "Recall@1", "Recall@3", "Recall@5", "Hit@1", "MRR",
            "EvidenceRecall@1", "EvidenceRecall@3", "EvidenceRecall@5",
            "FileOracle@10", "EvidenceOracle@10",
        )
    }
    return {
        "overall": {"hybrid": dict(zero), "hybrid_bge_rerank": dict(zero), "vector": dict(zero)},
        "verified_fairness": {
            "live_hybrid_pool_equals_m3_pool": True,
            "live_hybrid_metrics_match_m3": True,
        },
        "latency_diagnostic": {"api_rerank_ms": {"mean": 1.0, "median": 1.0, "p95": 1.0}},
    }


def test_comparison_table_has_all_three_strategies_per_metric():
    m3 = m4._load_m3_baseline()
    comparison = m4._build_comparison(_fake_summary(), m3)
    for metric in ("Recall@1", "Recall@5", "MRR", "EvidenceRecall@5", "FileOracle@10", "EvidenceOracle@10"):
        row = comparison["table"][metric]
        assert set(row) >= {
            "multilingual_hybrid_no_rerank",
            "ms_marco_rerank_M3_saved",
            "bge_v2_m3_api_M4",
            "delta_bge_vs_ms_marco",
            "delta_bge_vs_hybrid",
        }
    # The MS MARCO column must be M3's SAVED value, not a placeholder.
    assert (
        comparison["table"]["Recall@5"]["ms_marco_rerank_M3_saved"]
        == m3["summary"]["overall"]["hybrid_rerank"]["Recall@5"]
        == 0.755
    )


def test_comparison_deltas_are_arithmetically_consistent():
    m3 = m4._load_m3_baseline()
    comparison = m4._build_comparison(_fake_summary(), m3)
    row = comparison["table"]["Recall@1"]
    assert abs(
        row["delta_bge_vs_ms_marco"]
        - (row["bge_v2_m3_api_M4"] - row["ms_marco_rerank_M3_saved"])
    ) < 1e-12


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
    print("M4 experiment config - static audit (no network)")
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
