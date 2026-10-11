"""M4: BGE-reranker-v2-m3 (SiliconFlow cloud) frozen retrieval experiment.

Run from backend/:
    python tests/eval/run_eval_bge.py

WHAT THIS RUNS
--------------
Three strategies over the SAME frozen 50-question dataset and the SAME retrieval semantics:

  A. vector            -- multilingual dense retrieval                       (context)
  B. hybrid            -- multilingual hybrid (vector 0.7 + jieba BM25 0.3)    (context + oracle/hybrid parity)
  C. hybrid_bge_rerank -- B, then BAAI/bge-reranker-v2-m3 via SiliconFlow API  (this round's variable)

C is the only new thing. A and B are re-run in the same process so the comparison against
M3's saved hybrid/multilingual figures is a LIVE check rather than a recalled number: if the
re-run hybrid disagrees with the saved M3 hybrid, something about the backend, collection,
config or pool changed, and the run says so and stops instead of attributing the difference
to the reranker.

WHY IT BINDS ITS ENVIRONMENT BEFORE IMPORTING THE APP
-----------------------------------------------------
``settings`` is constructed at import time from env vars, and ``vector_store`` is a
module-level singleton that reads ``settings.qdrant_collection`` when it connects. So the
collection and embedding model MUST be set in ``os.environ`` before any ``app.*`` import, or
the run silently evaluates the production collection with the wrong embedding. That is not a
hypothetical: the default collection is ``enterprise_rag`` and the default embedding is
``all-MiniLM-L6-v2``, neither of which is this experiment.

WHY THE M3 COMPARISON IS DONE BY IDENTITY, NOT BY RE-DERIVING ORACLES
--------------------------------------------------------------------
M3's saved JSON stores only the Top-5 ``retrieved`` and the Top-10 ``pool`` records with
``file_name``/``chunk_index``/``score`` -- ``snippet_source`` was stripped by the harness's
public-writer, and the pool records carry no chunk text at all. EvidenceOracle@10 reads
``snippet_source``, so it CANNOT be recomputed offline from M3's JSON. It is therefore
compared as the STORED value (0.82), and the live re-run must reproduce it from live data.
FileOracle@10 and the pool identity, by contrast, depend only on ``file_name`` and are
recomputed here from the saved pool records.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

BACKEND_DIR = Path(__file__).resolve().parents[2]
EVAL_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

# ---------------------------------------------------------------------------
# Runtime binding. Set BEFORE any app import; see the module docstring.
# ---------------------------------------------------------------------------
COLLECTION = "enterprise_rag_eval_multilingual_v1"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
PRODUCTION_COLLECTION = "enterprise_rag"

os.environ["QDRANT_COLLECTION"] = COLLECTION
os.environ["EMBEDDING_MODEL"] = EMBEDDING_MODEL

from dataset_utils import load_frozen_dataset  # noqa: E402
from metrics import aggregate_metrics  # noqa: E402
import run_eval as base  # noqa: E402

# Restore run_eval's overridden binding before bge_reranker imports the settings singleton.
os.environ["QDRANT_COLLECTION"] = COLLECTION
os.environ["EMBEDDING_MODEL"] = EMBEDDING_MODEL

from bge_reranker import SiliconFlowReranker  # noqa: E402

# ---------------------------------------------------------------------------
# Experiment constants. The frozen values are asserted, not assumed: each one has a
# counterpart in the loaded dataset or the live collection, and a mismatch fails the run.
# ---------------------------------------------------------------------------
CANDIDATE_POOL_K = base.CANDIDATE_POOL_K          # 10
EVALUATION_K_MAX = base.EVALUATION_K_MAX          # 5
ORACLE_K = base.ORACLE_K                          # 10
EXPECTED_CORPUS_CHUNKS = base.EXPECTED_CORPUS_CHUNKS  # 78
EXPECTED_QUESTIONS = 50

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANKER_BACKEND = "SiliconFlow API"

RESULTS_DIR = EVAL_DIR / "results" / "bge_reranker_v2_m3_v1"
M3_RESULTS_DIR = EVAL_DIR / "results" / "multilingual_embed_v1"

STRATEGY_NAMES = ("vector", "hybrid", "hybrid_bge_rerank")

# The M3 figures this run is compared against. Populated from M3's SAVED JSON at runtime by
# _load_m3_baseline(); the literals here are the expected values and a mismatch is reported
# rather than silently used, so a hand-edited number cannot masquerade as a measurement.
M3_EXPECTED = {
    "hybrid_recall@1": 0.565,
    "hybrid_recall@5": 0.805,
    "hybrid_mrr": 0.7113333333333333,
    "hybrid_evidence_recall@5": 0.655,
    "hybrid_file_oracle@10": 0.94,
    "hybrid_evidence_oracle@10": 0.82,
    "hybrid_rerank_recall@1": 0.365,
    "hybrid_rerank_recall@5": 0.755,
    "hybrid_rerank_mrr": 0.5576666666666666,
    "hybrid_rerank_evidence_recall@5": 0.475,
}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


# ===========================================================================
# M3 baseline loading
# ===========================================================================

def _load_m3_baseline() -> dict[str, Any]:
    """Load M3's saved hybrid + hybrid_rerank results for the live parity check.

    Reads the SAVED files rather than trusting the constants above. The constants are used
    only to confirm the saved files are the ones this experiment was designed against.
    """
    out: dict[str, Any] = {"dir": str(M3_RESULTS_DIR.relative_to(EVAL_DIR.parent.parent))}
    for key, name in (("hybrid", "hybrid"), ("hybrid_rerank", "hybrid_rerank")):
        path = M3_RESULTS_DIR / f"{name}.json"
        if not path.exists():
            raise RuntimeError(
                f"M3 baseline missing: {path}. The cross-experiment comparison cannot be "
                f"made without it; refusing to run rather than reporting an unverified result."
            )
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        out[key] = payload
    summary_path = M3_RESULTS_DIR / "summary.json"
    out["summary"] = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    return out


def _m3_experiment_facts(m3: Mapping[str, Any]) -> dict[str, Any]:
    """Pull the runtime facts M3 recorded, so the live binding can be checked against them."""
    exp = m3["summary"]["experiment"]
    return {
        "collection": exp["collection"],
        "collection_points": exp["collection_points"],
        "collection_vector_size": exp["collection_vector_size"],
        "embedding_model": exp["embedding_model"],
        "embedding_dimension": exp["embedding_dimension"],
        "candidate_pool_k": exp["candidate_pool_k"],
        "evaluation_k_max": exp["evaluation_k_max"],
        "oracle_k": exp["oracle_k"],
        "reranker_model": exp["reranker_model"],
        "reranker_backend": exp["reranker_backend"],
    }


def _verify_m3_constants(m3: Mapping[str, Any]) -> dict[str, Any]:
    """Confirm the saved M3 numbers match the constants this run was designed against."""
    overall = m3["summary"]["overall"]
    hybrid = overall["hybrid"]
    rerank = overall["hybrid_rerank"]
    actual = {
        "hybrid_recall@1": hybrid["Recall@1"],
        "hybrid_recall@5": hybrid["Recall@5"],
        "hybrid_mrr": hybrid["MRR"],
        "hybrid_evidence_recall@5": hybrid["EvidenceRecall@5"],
        "hybrid_file_oracle@10": hybrid["FileOracle@10"],
        "hybrid_evidence_oracle@10": hybrid["EvidenceOracle@10"],
        "hybrid_rerank_recall@1": rerank["Recall@1"],
        "hybrid_rerank_recall@5": rerank["Recall@5"],
        "hybrid_rerank_mrr": rerank["MRR"],
        "hybrid_rerank_evidence_recall@5": rerank["EvidenceRecall@5"],
    }
    mismatches = {
        key: {"expected": M3_EXPECTED[key], "saved": value}
        for key, value in actual.items()
        if abs(value - M3_EXPECTED[key]) > 1e-9
    }
    if mismatches:
        raise RuntimeError(
            "M3 saved metrics do not match the constants this experiment was designed "
            f"against. Refusing to proceed rather than comparing against a moving baseline. "
            f"Mismatches: {json.dumps(mismatches, ensure_ascii=False)}"
        )
    return {"m3_constants_match_saved_json": True, "values": actual}


def _m3_pool_identity(m3_hybrid: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Per-question (file_name, chunk_index) of M3's hybrid Top-10 pool, for identity checks.

    Only fields M3 actually saved are used. ``snippet_source`` is absent by design, so no
    anchor/oracle recomputation happens here -- that would require text M3 never wrote.
    """
    return {
        item["id"]: [
            {"file_name": p["file_name"], "chunk_index": p["chunk_index"]}
            for p in item["pool"]
        ]
        for item in m3_hybrid["questions"]
    }


def _m3_stored_oracles(m3_hybrid: Mapping[str, Any]) -> dict[str, float]:
    return {
        item["id"]: {
            "file_oracle@10": item["metrics"]["file_oracle@10"],
            "evidence_oracle@10": item["metrics"]["evidence_oracle@10"],
        }
        for item in m3_hybrid["questions"]
    }


# ===========================================================================
# Scoring
# ===========================================================================

def _public_retrieved(retrieved: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in item.items() if key != "snippet_source"}
        for item in retrieved
    ]


def _latency(per_question: Sequence[Mapping[str, Any]], field: str) -> dict[str, float]:
    values = sorted(
        float(item["latency"][field])
        for item in per_question
        if field in item.get("latency", {})
    )
    if not values:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "n": 0}
    position = 0.95 * (len(values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    p95 = values[lower] + (values[upper] - values[lower]) * (position - lower)
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": p95,
        "n": len(values),
    }


def _aggregate(per_question: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Same aggregation as the base harness, kept local so this file cannot shift M3's."""
    official = aggregate_metrics(
        [
            {
                "id": item["id"],
                "retrieved": item["retrieved"],
                "relevant_files": item["relevant_files"],
            }
            for item in per_question
        ]
    )
    result = {
        "n": len(per_question),
        "Recall@1": official["recall@1"],
        "Recall@3": official["recall@3"],
        "Recall@5": official["recall@5"],
        "Hit@1": official["hit@1"],
        "MRR": official["mrr"],
    }
    for key, label in (
        ("evidence_recall@1", "EvidenceRecall@1"),
        ("evidence_recall@3", "EvidenceRecall@3"),
        ("evidence_recall@5", "EvidenceRecall@5"),
    ):
        result[label] = (
            sum(float(item["metrics"][key]) for item in per_question) / len(per_question)
            if per_question
            else 0.0
        )
    for key, label in (
        (f"file_oracle@{ORACLE_K}", "FileOracle@10"),
        (f"evidence_oracle@{ORACLE_K}", "EvidenceOracle@10"),
    ):
        hits = sum(1 for item in per_question if float(item["metrics"][key]) == 1.0)
        result[label] = hits / len(per_question) if per_question else 0.0
        result[f"{label}_hits"] = hits
    return result


def _buckets(
    questions: Sequence[Mapping[str, Any]],
    results: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, dict[str, dict[str, Any]]]:
    output: dict[str, dict[str, dict[str, Any]]] = {}
    for axis in ("difficulty", "domain", "category"):
        output[axis] = {}
        for value in sorted({str(question[axis]) for question in questions}):
            output[axis][value] = {}
            selected = [q for q in questions if str(q[axis]) == value]
            for strategy, items in results.items():
                by_id = {item["id"]: item for item in items}
                output[axis][value][strategy] = _aggregate(
                    [by_id[q["id"]] for q in selected]
                )
    return output


# ===========================================================================
# Main
# ===========================================================================

def _verify_m3_retrieval(
    results: Mapping[str, Sequence[Mapping[str, Any]]], m3: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate all M3 retrieval comparisons before constructing the cloud reranker."""
    per_q = {name: {item["id"]: item for item in results[name]} for name in ("vector", "hybrid")}
    m3_pools = _m3_pool_identity(m3["hybrid"])
    m3_oracles = _m3_stored_oracles(m3["hybrid"])
    if (
        len(results["hybrid"]) != EXPECTED_QUESTIONS
        or len(per_q["hybrid"]) != EXPECTED_QUESTIONS
        or len(results["vector"]) != EXPECTED_QUESTIONS
        or len(per_q["vector"]) != EXPECTED_QUESTIONS
        or set(per_q["hybrid"]) != set(m3_pools)
        or set(per_q["vector"]) != set(m3_pools)
    ):
        raise RuntimeError("Live retrieval question IDs/count do not match M3")
    if not all(len(item["pool"]) == CANDIDATE_POOL_K for item in results["hybrid"]):
        raise RuntimeError("Hybrid pool does not contain the frozen Top-10 candidates")
    fairness: dict[str, Any] = {}

    # 4. Candidate-pool identity against M3's SAVED hybrid pool (file_name + chunk_index).
    pool_identity_mismatches = [
        qid
        for qid in m3_pools
        if [
            (p["file_name"], p["chunk_index"])
            for p in per_q["hybrid"][qid]["pool"]
        ]
        != [(p["file_name"], p["chunk_index"]) for p in m3_pools[qid]]
    ]
    fairness["live_hybrid_pool_equals_m3_pool"] = not pool_identity_mismatches
    if pool_identity_mismatches:
        raise RuntimeError(
            f"Live hybrid pool differs from M3's saved pool for "
            f"{len(pool_identity_mismatches)} question(s): {pool_identity_mismatches[:5]}. "
            f"The retrieval environment changed; cross-experiment superiority claims cannot "
            f"be made. Investigate the backend/collection/config before reranking anything."
        )

    # 5. FileOracle@10 and EvidenceOracle@10 per question vs M3's STORED values.
    #    FileOracle can be recomputed from the saved pool; EvidenceOracle CANNOT (M3's pool
    #    records carry no chunk text), so it is compared as stored.
    file_oracle_mismatches = [
        qid
        for qid in m3_oracles
        if per_q["hybrid"][qid]["metrics"][f"file_oracle@{ORACLE_K}"]
        != m3_oracles[qid][f"file_oracle@{ORACLE_K}"]
    ]
    fairness["live_hybrid_file_oracle_equals_m3"] = not file_oracle_mismatches

    evidence_oracle_mismatches = [
        qid
        for qid in m3_oracles
        if per_q["hybrid"][qid]["metrics"][f"evidence_oracle@{ORACLE_K}"]
        != m3_oracles[qid][f"evidence_oracle@{ORACLE_K}"]
    ]
    fairness["live_hybrid_evidence_oracle_equals_m3_stored"] = not evidence_oracle_mismatches

    # 6. Live hybrid metrics vs M3's saved aggregate. If these moved, the run must NOT be
    #    used for cross-experiment ranking -- it stops instead.
    live_hybrid_agg = _aggregate(results["hybrid"])
    live_m3_hybrid_agg = m3["summary"]["overall"]["hybrid"]
    key_map = {
        "Recall@1": "Recall@1",
        "Recall@5": "Recall@5",
        "MRR": "MRR",
        "EvidenceRecall@5": "EvidenceRecall@5",
        "FileOracle@10": "FileOracle@10",
        "EvidenceOracle@10": "EvidenceOracle@10",
    }
    metric_deltas = {
        label: {
            "live": live_hybrid_agg[out_key],
            "m3_saved": live_m3_hybrid_agg[out_key],
            "delta": live_hybrid_agg[out_key] - live_m3_hybrid_agg[out_key],
        }
        for label, out_key in key_map.items()
    }
    moved = {
        label: delta
        for label, delta in metric_deltas.items()
        if abs(delta["delta"]) > 1e-9
    }
    fairness["live_hybrid_metrics_match_m3"] = not moved

    if file_oracle_mismatches or evidence_oracle_mismatches or moved:
        raise RuntimeError(
            "Live retrieval does not reproduce M3: "
            f"FileOracle mismatches={file_oracle_mismatches}; "
            f"EvidenceOracle mismatches={evidence_oracle_mismatches}; "
            f"aggregate mismatches={json.dumps(moved, ensure_ascii=False)}"
        )
    return fairness, metric_deltas


def _run_experiment() -> int:
    dataset, hashes = load_frozen_dataset()
    questions = dataset["questions"]

    m3 = _load_m3_baseline()
    m3_facts = _m3_experiment_facts(m3)
    m3_constants = _verify_m3_constants(m3)

    # --- import the app only now that the environment is bound ---
    from qdrant_client import QdrantClient
    from app.core.config import settings
    from app.services.advanced_retrieval import advanced_retrieval
    from app.services.embeddings import embedding_service
    from app.services.retrieval import retrieval_service
    from app.services.vector_store import vector_store

    if settings.qdrant_collection != COLLECTION:
        raise RuntimeError(
            f"Collection binding failed: settings says {settings.qdrant_collection!r}, "
            f"expected {COLLECTION!r}"
        )
    if settings.embedding_model != EMBEDDING_MODEL:
        raise RuntimeError(
            f"Embedding binding failed: settings says {settings.embedding_model!r}, "
            f"expected {EMBEDDING_MODEL!r}"
        )

    client = QdrantClient(
        url=settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=30.0
    )

    # Production guard: the demo collection must exist and must never be touched.
    production_before = base._raw_points(client, PRODUCTION_COLLECTION)
    if production_before != 1:
        raise RuntimeError(
            f"Production collection pre-flight expected 1 point, got {production_before}"
        )

    collection_points = base._raw_points(client, COLLECTION)
    collection_vector_size = base._collection_vector_size(client, COLLECTION)

    if collection_points != EXPECTED_CORPUS_CHUNKS:
        raise RuntimeError(
            f"Corpus integrity: expected {EXPECTED_CORPUS_CHUNKS} chunks, "
            f"{COLLECTION} holds {collection_points}."
        )
    if len(questions) != EXPECTED_QUESTIONS:
        raise RuntimeError(f"Expected {EXPECTED_QUESTIONS} questions, got {len(questions)}")

    embedding = embedding_service.get_embeddings()
    embedding_backend = type(embedding).__name__
    embedding_dimension = base._runtime_embedding_dimension(embedding)

    if collection_vector_size != embedding_dimension:
        raise RuntimeError(
            f"Embedding/collection dimension mismatch: {embedding_backend} produces "
            f"{embedding_dimension}-d but {COLLECTION} is {collection_vector_size}-d."
        )

    # Verify the live binding matches what M3 recorded. A 384-d vector alone cannot prove
    # compatibility -- both MiniLM models are 384-d, so dimension equality is necessary but
    # NOT sufficient, which is exactly why the model NAME is compared too.
    binding_checks = {
        "collection_matches_m3": COLLECTION == m3_facts["collection"],
        "points_match_m3": collection_points == m3_facts["collection_points"],
        "vector_size_matches_m3": collection_vector_size == m3_facts["collection_vector_size"],
        "embedding_model_matches_m3": settings.embedding_model == m3_facts["embedding_model"],
        "embedding_dimension_matches_m3": embedding_dimension == m3_facts["embedding_dimension"],
        "pool_k_matches_m3": CANDIDATE_POOL_K == m3_facts["candidate_pool_k"],
        "eval_k_matches_m3": EVALUATION_K_MAX == m3_facts["evaluation_k_max"],
        "oracle_k_matches_m3": ORACLE_K == m3_facts["oracle_k"],
    }
    failed_bindings = [k for k, ok in binding_checks.items() if not ok]
    if failed_bindings:
        raise RuntimeError(
            f"Live runtime binding does not match M3's recorded environment: {failed_bindings}. "
            f"M3={json.dumps(m3_facts, ensure_ascii=False)} "
            f"live={{'collection': COLLECTION, 'embedding': settings.embedding_model, "
            f"'dimension': embedding_dimension}}"
        )

    print(
        f"[i] runtime: collection={COLLECTION} points={collection_points} "
        f"vector_size={collection_vector_size} embedding={settings.embedding_model} "
        f"backend={embedding_backend} dimension={embedding_dimension}"
    )
    print(f"[i] M3 baseline verified: {M3_RESULTS_DIR.name}")

    warmup_question = questions[0]["question"]
    base.vector_only(retrieval_service, warmup_question)
    base.hybrid(advanced_retrieval, warmup_question)

    all_documents = vector_store.get_all_documents()
    context_by_file: dict[str, list[str]] = {}
    for document in all_documents:
        metadata = getattr(document, "metadata", {}) or {}
        file_name = metadata.get("file_name")
        if file_name:
            context_by_file.setdefault(file_name, []).append(str(document.page_content))

    results: dict[str, list[dict[str, Any]]] = {name: [] for name in STRATEGY_NAMES}
    cached_pools: list[tuple[Mapping[str, Any], Any, Any, float]] = []

    # Phase A: retrieve and score every question without cloud requests.
    for question in questions:
        question_text = question["question"]

        t0 = time.perf_counter()
        vector_docs, vector_scores = base.vector_only(retrieval_service, question_text)
        vector_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        hybrid_docs, hybrid_scores = base.hybrid(advanced_retrieval, question_text)
        hybrid_ms = (time.perf_counter() - t0) * 1000
        if not hybrid_docs:
            raise RuntimeError(f"Hybrid returned no candidates for {question['id']}")

        if len(hybrid_docs) != len(hybrid_scores):
            raise RuntimeError(f"{question['id']}: hybrid docs/scores length mismatch")
        cached_pools.append((question, hybrid_docs, hybrid_scores, hybrid_ms))

        results["vector"].append(
            base._question_result(
                question, vector_docs, vector_scores,
                {"vector_retrieval_ms": vector_ms}, context_by_file,
            )
        )
        results["hybrid"].append(
            base._question_result(
                question, hybrid_docs, hybrid_scores,
                {"hybrid_retrieval_ms": hybrid_ms}, context_by_file,
            )
        )
    # --- production / collection guards ---
    production_after = base._raw_points(client, PRODUCTION_COLLECTION)
    if production_after != production_before or production_after != 1:
        raise RuntimeError(
            f"Production collection changed: before={production_before}, after={production_after}"
        )
    if base._raw_points(client, COLLECTION) != collection_points:
        raise RuntimeError("Evaluation collection point count changed during benchmark")

    fairness, metric_deltas = _verify_m3_retrieval(results, m3)
    for name in ("vector", "hybrid"):
        if not all(
            item["metrics"][f"file_oracle@{ORACLE_K}"] >= item["metrics"]["recall@5"]
            and item["metrics"][f"evidence_oracle@{ORACLE_K}"] >= item["metrics"]["evidence_recall@5"]
            and len(item["pool"]) >= len(item["retrieved"])
            for item in results[name]
        ):
            raise RuntimeError(f"{name}: retrieval pool/oracle invariants failed")

    # Phase B: the M3 gate passed; reuse the original objects and candidate order.
    reranker = SiliconFlowReranker()
    print(f"[i] reranker: {reranker.model_name} via {reranker.endpoint}")
    for question, hybrid_docs, hybrid_scores, hybrid_ms in cached_pools:
        question_text = question["question"]
        # --- BGE cloud rerank over the SAME pool ---
        t0 = time.perf_counter()
        bge_docs, bge_scores = reranker.rerank_documents(
            question_text, hybrid_docs, top_k=EVALUATION_K_MAX
        )
        bge_total_ms = (time.perf_counter() - t0) * 1000
        api_ms = reranker.stats.per_call_latency_ms[-1]

        # The pool passed in must survive intact: this is the fairness invariant, checked
        # on EVERY question rather than only in aggregate.
        if len(hybrid_docs) != len(hybrid_scores):
            raise RuntimeError(f"{question['id']}: hybrid docs/scores length mismatch")
        if len(bge_docs) != min(EVALUATION_K_MAX, len(hybrid_docs)):
            raise RuntimeError(
                f"{question['id']}: BGE returned {len(bge_docs)} docs, expected "
                f"{min(EVALUATION_K_MAX, len(hybrid_docs))}"
            )
        if len(set(id(d) for d in bge_docs)) != len(bge_docs):
            raise RuntimeError(f"{question['id']}: BGE output contains duplicates")
        for doc in bge_docs:
            if not any(doc is candidate for candidate in hybrid_docs):
                raise RuntimeError(
                    f"{question['id']}: BGE output carries a document that was not in the "
                    f"hybrid pool"
                )

        results["hybrid_bge_rerank"].append(
            base._question_result(
                question, bge_docs, bge_scores,
                {
                    "hybrid_retrieval_ms": hybrid_ms,
                    "api_rerank_ms": api_ms,
                    "bge_rerank_total_ms": bge_total_ms,
                    "total_retrieval_ms": hybrid_ms + bge_total_ms,
                },
                context_by_file,
                # Oracles read the shared PRE-rerank pool, exactly as in the base harness.
                oracle_pool_documents=hybrid_docs,
                oracle_pool_scores=hybrid_scores,
            )
        )

    production_after = base._raw_points(client, PRODUCTION_COLLECTION)
    if production_after != production_before or production_after != 1:
        raise RuntimeError("Production collection changed during BGE reranking")
    if base._raw_points(client, COLLECTION) != collection_points:
        raise RuntimeError("Evaluation collection point count changed during BGE reranking")

    per_q = {name: {item["id"]: item for item in results[name]} for name in STRATEGY_NAMES}

    # =====================================================================
    # Fairness / oracle verification
    # =====================================================================
    # 1. retrieved IS the BGE-ranked Top-5 (by construction) -- confirm the saved values
    #    are a descending permutation of the pool, i.e. an actual ranking.
    fairness["bge_retrieved_is_top5"] = all(
        len(per_q["hybrid_bge_rerank"][qid]["retrieved"]) == EVALUATION_K_MAX
        for qid in per_q["hybrid_bge_rerank"]
    )
    fairness["bge_retrieved_scores_descending"] = all(
        [
            r["score"] for r in per_q["hybrid_bge_rerank"][qid]["retrieved"]
        ] == sorted(
            [r["score"] for r in per_q["hybrid_bge_rerank"][qid]["retrieved"]], reverse=True
        )
        for qid in per_q["hybrid_bge_rerank"]
    )
    fairness["bge_retrieved_subset_of_pool"] = all(
        all(
            any(
                r["file_name"] == p["file_name"] and r["chunk_index"] == p["chunk_index"]
                for p in per_q["hybrid_bge_rerank"][qid]["pool"]
            )
            for r in per_q["hybrid_bge_rerank"][qid]["retrieved"]
        )
        for qid in per_q["hybrid_bge_rerank"]
    )

    # 2. pool IS the pre-rerank hybrid Top-10.
    fairness["bge_pool_is_hybrid_top10"] = all(
        len(per_q["hybrid_bge_rerank"][qid]["pool"]) == CANDIDATE_POOL_K
        for qid in per_q["hybrid_bge_rerank"]
    )
    fairness["bge_pool_equals_hybrid_pool_in_run"] = all(
        [
            (p["file_name"], p["chunk_index"])
            for p in per_q["hybrid_bge_rerank"][qid]["pool"]
        ]
        == [(p["file_name"], p["chunk_index"]) for p in per_q["hybrid"][qid]["pool"]]
        for qid in per_q["hybrid"]
    )

    # 3. Oracles must be EQUAL per question between hybrid and BGE. They read the same pool,
    #    so any difference means the wrong object was passed.
    for oracle_key in (f"file_oracle@{ORACLE_K}", f"evidence_oracle@{ORACLE_K}"):
        mismatched = [
            qid
            for qid in per_q["hybrid"]
            if per_q["hybrid"][qid]["metrics"][oracle_key]
            != per_q["hybrid_bge_rerank"][qid]["metrics"][oracle_key]
        ]
        fairness[f"bge_{oracle_key}_equals_hybrid"] = not mismatched
        if mismatched:
            raise RuntimeError(
                f"{oracle_key} differs between hybrid and hybrid_bge_rerank for "
                f"{len(mismatched)} question(s): {mismatched[:5]}. The BGE strategy must read "
                f"the same pre-rerank pool as hybrid."
            )

    # Also check the three M3 oracle/hybrid invariants the base harness enforces, so this
    # run cannot pass with a pool that was built the wrong way.
    for name in STRATEGY_NAMES:
        items = results[name]
        if not all(
            item["metrics"][f"file_oracle@{ORACLE_K}"] >= item["metrics"]["recall@5"]
            and item["metrics"][f"evidence_oracle@{ORACLE_K}"]
            >= item["metrics"]["evidence_recall@5"]
            for item in items
        ):
            raise RuntimeError(f"{name}: oracle does not dominate Recall@5")
        if not all(len(item["pool"]) >= len(item["retrieved"]) for item in items):
            raise RuntimeError(f"{name}: pool shorter than retrieved")

    fairness_pass = all(
        value for value in fairness.values() if isinstance(value, bool)
    )

    if not fairness_pass:
        raise RuntimeError(f"Fairness verification failed: {fairness}")

    # =====================================================================
    # Summary
    # =====================================================================
    experiment = {
        "milestone": "M4",
        "variable": "reranker model + inference backend only",
        "collection": COLLECTION,
        "collection_points": collection_points,
        "collection_vector_size": collection_vector_size,
        "embedding_model": settings.embedding_model,
        "embedding_backend": embedding_backend,
        "embedding_dimension": embedding_dimension,
        "expected_corpus_chunks": EXPECTED_CORPUS_CHUNKS,
        "frozen_questions": len(questions),
        "distance": "Cosine",
        "candidate_pool_k": CANDIDATE_POOL_K,
        "evaluation_k_max": EVALUATION_K_MAX,
        "oracle_k": ORACLE_K,
        "hybrid_vector_weight": 0.7,
        "hybrid_bm25_weight": 0.3,
        "bm25_tokenizer": "jieba (chinese_bm25_tokenizer)",
        "reranker_model": RERANKER_MODEL,
        "reranker_backend": RERANKER_BACKEND,
        "reranker_endpoint": reranker.endpoint,
        "reranker_scores_normalized": False,
        "reranker_fallback_enabled": False,
        "old_reranker_model": m3_facts["reranker_model"],
        "old_reranker_backend": m3_facts["reranker_backend"],
        "frozen_integrity_authority": hashes["frozen_integrity_authority"],
        "canonical_hashes": hashes["canonical"]["actual"],
        "canonical_sha_match": hashes["canonical"]["match"],
        "legacy_hashes": hashes["legacy"]["actual"],
        "legacy_sha_match": hashes["legacy"]["match"],
    }

    api_latency = _latency(results["hybrid_bge_rerank"], "api_rerank_ms")

    summary = {
        "experiment": experiment,
        "m3_baseline": {
            "directory": m3["dir"],
            "verified_constants": m3_constants,
            "runtime_facts": m3_facts,
        },
        "binding_checks": binding_checks,
        "verified_fairness": fairness,
        "fairness_verification_pass": fairness_pass,
        "live_vs_m3_hybrid_metrics": metric_deltas,
        "overall": {name: _aggregate(items) for name, items in results.items()},
        "by_difficulty": _buckets(questions, results)["difficulty"],
        "by_domain": _buckets(questions, results)["domain"],
        "by_category": _buckets(questions, results)["category"],
        "api_call_stats": {
            **reranker.stats.as_dict(),
            "questions": len(questions),
            "expected_requests_minimum": len(questions),
            "retry_policy": "bounded: 429/5xx/timeout retried up to 3 attempts; 4xx not retried",
            "note": "requests counts successful API calls; retries counts extra attempts",
        },
        "latency_diagnostic": {
            "note": "single-run local latency diagnostic; not a production SLA",
            "vector_retrieval_ms": _latency(results["vector"], "vector_retrieval_ms"),
            "hybrid_retrieval_ms": _latency(results["hybrid"], "hybrid_retrieval_ms"),
            "api_rerank_ms": api_latency,
            "bge_rerank_total_ms": _latency(results["hybrid_bge_rerank"], "bge_rerank_total_ms"),
            "total_retrieval_ms": _latency(results["hybrid_bge_rerank"], "total_retrieval_ms"),
        },
        "production_points": {
            "collection": PRODUCTION_COLLECTION,
            "before": production_before,
            "after": production_after,
            "expected": 1,
        },
        "canonical_frozen_sha_match": all(hashes["canonical"]["match"].values()),
    }

    for name in STRATEGY_NAMES:
        _write_json(
            RESULTS_DIR / f"{name}.json",
            {
                "strategy": name,
                "experiment": experiment,
                "questions": [
                    {
                        **item,
                        "retrieved": _public_retrieved(item["retrieved"]),
                        "pool": _public_retrieved(item["pool"]),
                    }
                    for item in results[name]
                ],
            },
        )
    _write_json(RESULTS_DIR / "summary.json", summary)

    comparison = _build_comparison(summary, m3)
    _write_json(RESULTS_DIR / "comparison.json", comparison)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[OK] wrote {RESULTS_DIR}")
    print("BGE_RERANKER_V2_M3_V1 = PASS")
    return 0


def main() -> int:
    try:
        return _run_experiment()
    except Exception as exc:
        _write_json(
            RESULTS_DIR / "BLOCKED.json",
            {"status": "BLOCKED", "reason": str(exc), "error_type": type(exc).__name__},
        )
        raise


def _build_comparison(summary: Mapping[str, Any], m3: Mapping[str, Any]) -> dict[str, Any]:
    """Three-strategy table: hybrid (A/B) vs BGE (C) vs M3's old reranker."""
    live = summary["overall"]
    m3_overall = m3["summary"]["overall"]
    metrics = [
        "Recall@1", "Recall@3", "Recall@5", "Hit@1", "MRR",
        "EvidenceRecall@1", "EvidenceRecall@3", "EvidenceRecall@5",
        "FileOracle@10", "EvidenceOracle@10",
    ]
    table: dict[str, dict[str, Any]] = {}
    for metric in metrics:
        row = {
            "multilingual_hybrid_no_rerank": live["hybrid"][metric],
            "ms_marco_rerank_M3_saved": m3_overall["hybrid_rerank"][metric],
            "bge_v2_m3_api_M4": live["hybrid_bge_rerank"][metric],
        }
        row["delta_bge_vs_ms_marco"] = (
            row["bge_v2_m3_api_M4"] - row["ms_marco_rerank_M3_saved"]
        )
        row["delta_bge_vs_hybrid"] = (
            row["bge_v2_m3_api_M4"] - row["multilingual_hybrid_no_rerank"]
        )
        table[metric] = row

    return {
        "experiment": "bge_reranker_v2_m3_v1",
        "note": (
            "Three strategies over one frozen 50-question dataset and one shared pre-rerank "
            "Top-10 pool. M4 changes ONLY the reranker model and its inference backend. The "
            "MS MARCO column is M3's SAVED result, not a re-run, because the M3 reranker is a "
            "local sentence-transformers model this experiment does not reload."
        ),
        "strategies": {
            "multilingual_hybrid_no_rerank": {
                "retrieval": "vector 0.7 + jieba BM25 0.3, k=10",
                "reranker": None,
                "source": "re-run live in M4 (verified equal to M3's saved hybrid)",
            },
            "ms_marco_rerank_M3_saved": {
                "retrieval": "same hybrid",
                "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2 (local)",
                "source": "M3 saved JSON (multilingual_embed_v1/hybrid_rerank.json)",
            },
            "bge_v2_m3_api_M4": {
                "retrieval": "same hybrid",
                "reranker": "BAAI/bge-reranker-v2-m3 (SiliconFlow API)",
                "source": "this run",
            },
        },
        "fairness": {
            "same_dataset": True,
            "same_collection": True,
            "same_embedding": True,
            "same_candidate_pool": True,
            "oracles_read_pre_rerank_pool": True,
            "live_hybrid_pool_equals_m3_pool": summary["verified_fairness"][
                "live_hybrid_pool_equals_m3_pool"
            ],
            "live_hybrid_metrics_match_m3": summary["verified_fairness"][
                "live_hybrid_metrics_match_m3"
            ],
        },
        "table": table,
        "api_latency": summary["latency_diagnostic"]["api_rerank_ms"],
        "conclusion": (
            "Reported as measured. BGE-v2-m3 is NOT guaranteed to beat MS MARCO; see table."
        ),
    }


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"BGE_RERANKER_V2_M3_V1 = FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
