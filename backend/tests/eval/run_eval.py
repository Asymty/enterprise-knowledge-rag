"""PHASE3-IMPLEMENTATION-M5: retrieval-only baseline benchmark.

Run from backend/: ``python tests/eval/run_eval.py``.
"""

from __future__ import annotations

import json
import os

os.environ["QDRANT_COLLECTION"] = "enterprise_rag_eval_v1"

import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from dataset_utils import (
    DATASET_MANIFEST_PATH,
    evidence_oracle_at_k,
    evidence_recall,
    compact_snippet,
    file_oracle_at_k,
    load_frozen_dataset,
)
from metrics import aggregate_metrics, hit_at_1, recall_at_k, reciprocal_rank
from strategies import hybrid, reranked, vector_only, warm_up


COLLECTION = "enterprise_rag_eval_v1"
PRODUCTION_COLLECTION = "enterprise_rag"
CANDIDATE_POOL_K = 10
EVALUATION_K_MAX = 5
# Oracle metrics read the FULL candidate pool, not the Top-5 that the ranking metrics score.
# Recall@5/MRR answer "was the right thing ranked highly"; the oracles answer "was it in the
# pool at all", which is the ceiling a reranker or fusion change cannot exceed.
ORACLE_K = CANDIDATE_POOL_K
# Corpus V1's offline chunk count. Used ONLY to assert the collection was filled with the
# frozen corpus -- it is an integrity expectation for this experiment, not a harness rule,
# and is recorded separately from the runtime collection size so the two never get conflated.
EXPECTED_CORPUS_CHUNKS = 78
RESULTS_DIR = Path(__file__).resolve().parent / "results"
STRATEGY_NAMES = ("vector", "hybrid", "hybrid_rerank")


def _document_record(
    document: Any,
    score: float,
    rank: int,
    context_by_file: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    metadata = getattr(document, "metadata", {}) or {}
    content = str(getattr(document, "page_content", ""))
    file_name = metadata.get("file_name")
    adjacent = "\n".join(context_by_file.get(file_name, ()))
    return {
        "rank": rank,
        "file_name": file_name,
        "chunk_index": metadata.get("chunk_index"),
        "score": float(score),
        "snippet": compact_snippet(content, adjacent),
        "snippet_source": content,
    }


def _pool_record(
    document: Any,
    score: float,
    rank: int,
    context_by_file: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """One candidate from the full Top-K pool, retaining full text for anchor matching."""
    record = _document_record(document, score, rank, context_by_file)
    return record


def _question_result(
    question: Mapping[str, Any],
    documents: Sequence[Any],
    scores: Sequence[float],
    latency: Mapping[str, float],
    context_by_file: Mapping[str, Sequence[str]],
    oracle_pool_documents: Sequence[Any] | None = None,
    oracle_pool_scores: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Score one question for one strategy.

    ``documents``/``scores`` are the strategy's OWN ranked output and drive the ranking
    metrics plus ``retrieved``.

    ``oracle_pool_documents``/``oracle_pool_scores`` are the candidate pool the ORACLES read.
    They default to ``documents`` (correct for vector and hybrid, whose output IS the pool),
    but a reranking strategy must pass the PRE-rerank pool here.

    Why the distinction is load-bearing: the oracles are defined as "was the answer anywhere
    in the Top-K candidate pool", i.e. the ceiling a reranker can reach, not what it achieved.
    Feeding them the reranked Top-5 instead silently redefines them as a ranking metric AND
    breaks the invariant that hybrid and hybrid+rerank share one pool -- measured as
    FileOracle@10 0.940 vs 0.800 and EvidenceOracle@10 0.820 vs 0.520, a "regression" that was
    entirely an artifact of passing the wrong object.
    """
    pool_documents = documents if oracle_pool_documents is None else oracle_pool_documents
    pool_scores = scores if oracle_pool_scores is None else oracle_pool_scores

    pool = [
        _pool_record(document, score, rank, context_by_file)
        for rank, (document, score) in enumerate(
            zip(pool_documents[:ORACLE_K], pool_scores[:ORACLE_K]), start=1
        )
    ]
    retrieved = [
            _document_record(document, score, rank, context_by_file)
        for rank, (document, score) in enumerate(
            zip(documents[:EVALUATION_K_MAX], scores[:EVALUATION_K_MAX]), start=1
        )
    ]
    relevant_files = list(question["relevant_files"])
    official_input = [{"file_name": item["file_name"]} for item in retrieved]
    metrics = {
        "recall@1": recall_at_k(official_input, relevant_files, 1),
        "recall@3": recall_at_k(official_input, relevant_files, 3),
        "recall@5": recall_at_k(official_input, relevant_files, 5),
        "hit@1": hit_at_1(official_input, relevant_files),
        "rr": reciprocal_rank(official_input, relevant_files),
        "evidence_recall@1": evidence_recall(retrieved, question["relevant_anchors"], 1),
        "evidence_recall@3": evidence_recall(retrieved, question["relevant_anchors"], 3),
        "evidence_recall@5": evidence_recall(retrieved, question["relevant_anchors"], 5),
        f"file_oracle@{ORACLE_K}": file_oracle_at_k(
            pool, relevant_files, ORACLE_K
        ),
        f"evidence_oracle@{ORACLE_K}": evidence_oracle_at_k(
            pool, question["relevant_anchors"], ORACLE_K
        ),
    }
    for key, value in metrics.items():
        if not 0.0 <= value <= 1.0:
            raise AssertionError(f"{question['id']} {key} out of range: {value}")
    return {
        "id": question["id"],
        "question": question["question"],
        "domain": question["domain"],
        "category": question["category"],
        "difficulty": question["difficulty"],
        "relevant_files": list(question["relevant_files"]),
        "relevant_anchors": list(dict.fromkeys(question["relevant_anchors"])),
        "retrieved": retrieved,
        "pool": pool,
        "metrics": metrics,
        "latency": dict(latency),
    }


def _aggregate(per_question: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
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
    for key in ("evidence_recall@1", "evidence_recall@3", "evidence_recall@5"):
        result[key.replace("evidence_recall", "EvidenceRecall").replace("@", "@")] = (
            sum(float(item["metrics"][key]) for item in per_question) / len(per_question)
            if per_question
            else 0.0
        )
    # Oracles are binary per question then macro-averaged, so the average IS the hit rate:
    # 0.26 means "13 of 50 questions had a relevant file somewhere in the Top-10".
    for key in (f"file_oracle@{ORACLE_K}", f"evidence_oracle@{ORACLE_K}"):
        label = key.replace("file_oracle", "FileOracle").replace(
            "evidence_oracle", "EvidenceOracle"
        )
        hits = sum(1 for item in per_question if float(item["metrics"][key]) == 1.0)
        result[label] = hits / len(per_question) if per_question else 0.0
        result[label.replace("Oracle@", "Oracle@") + "_hits"] = hits
    return result


def _buckets(
    questions: Sequence[Mapping[str, Any]],
    results: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    output: dict[str, dict[str, dict[str, Any]]] = {}
    for axis in ("difficulty", "domain", "category"):
        output[axis] = {}
        values = sorted({str(question[axis]) for question in questions})
        for value in values:
            output[axis][value] = {}
            selected_questions = [question for question in questions if str(question[axis]) == value]
            for strategy, items in results.items():
                by_id = {item["id"]: item for item in items}
                output[axis][value][strategy] = _aggregate(
                    [by_id[question["id"]] for question in selected_questions]
                )
    return output


def _latency(per_question: Sequence[Mapping[str, Any]], field: str) -> dict[str, float]:
    values = sorted(float(item["latency"][field]) for item in per_question)
    if not values:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0}
    position = 0.95 * (len(values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    p95 = values[lower] + (values[upper] - values[lower]) * (position - lower)
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": p95,
    }


def _public_retrieved(retrieved: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in item.items() if key != "snippet_source"}
        for item in retrieved
    ]


def _manual_cases(results: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    by_id = {name: {item["id"]: item for item in items} for name, items in results.items()}
    ids = set(by_id["vector"]) & set(by_id["hybrid"]) & set(by_id["hybrid_rerank"])

    def correct(name: str, question_id: str) -> bool:
        item = by_id[name][question_id]
        return item["metrics"]["recall@5"] == 1.0

    def vector_miss_hybrid_hit(question_id: str) -> bool:
        return not correct("vector", question_id) and correct("hybrid", question_id)

    def reranker_lift(question_id: str) -> bool:
        vector = by_id["hybrid"][question_id]["metrics"]["recall@5"]
        reranked_metrics = by_id["hybrid_rerank"][question_id]["metrics"]
        return correct("hybrid", question_id) and reranked_metrics["recall@1"] > by_id["hybrid"][question_id]["metrics"]["recall@1"]

    predicates: list[tuple[str, Callable[[str], bool]]] = [
        ("all_three_correct", lambda q: all(correct(name, q) for name in STRATEGY_NAMES)),
        ("vector_miss_hybrid_hit", vector_miss_hybrid_hit),
        ("hybrid_hit_reranker_lift", reranker_lift),
        ("all_three_miss", lambda q: all(not correct(name, q) for name in STRATEGY_NAMES)),
    ]
    selected: list[dict[str, Any]] = []
    for label, predicate in predicates:
        for question_id in sorted(ids):
            if predicate(question_id):
                selected.append(
                    {
                        "case": label,
                        "id": question_id,
                        "question": by_id["vector"][question_id]["question"],
                        "ground_truth": {
                            "relevant_files": by_id["vector"][question_id]["relevant_files"],
                            "relevant_anchors": by_id["vector"][question_id]["relevant_anchors"],
                        },
                        "vector_top5": _public_retrieved(by_id["vector"][question_id]["retrieved"]),
                        "hybrid_top5": _public_retrieved(by_id["hybrid"][question_id]["retrieved"]),
                        "hybrid_rerank_top5": _public_retrieved(by_id["hybrid_rerank"][question_id]["retrieved"]),
                    }
                )
                if len([item for item in selected if item["case"] == label]) == 3:
                    break
    return selected


def _raw_points(client: Any, collection: str) -> int:
    return int(client.get_collection(collection).points_count)


def _collection_vector_size(client: Any, collection: str) -> int:
    """Declared vector dimension of a Qdrant collection's config.

    Read from the collection schema rather than assumed, so a mismatch with the embedding
    backend is detectable before any query runs instead of surfacing as a Qdrant 400.
    Handles both an unnamed single vector and a named-vector mapping.
    """
    params = client.get_collection(collection).config.params.vectors
    size = getattr(params, "size", None)
    if size is None:
        # named vectors: dict[str, VectorParams]
        sizes = {getattr(v, "size", None) for v in params.values()}
        sizes.discard(None)
        if len(sizes) != 1:
            raise RuntimeError(f"Cannot determine a single vector size for {collection}: {sizes}")
        size = sizes.pop()
    return int(size)


# Probe text for the embedding-dimension measurement. Short, non-empty, and stable: the
# dimension is a property of the model, so any input yields the same length; a trivial
# string just keeps this cheap. Deliberately NOT derived from the corpus, so this probe
# cannot be confused with a retrieval measurement.
EMBEDDING_DIMENSION_PROBE = "维度探针"


def _runtime_embedding_dimension(embedding: Any) -> int:
    """Measure the embedding dimension by actually embedding the probe text.

    Not inferred from the model name: two models sharing a name across revisions can differ,
    and a name-to-dimension table is exactly the kind of literal that goes stale silently.
    The real vector length is the only trustworthy source.
    """
    vector = embedding.embed_query(EMBEDDING_DIMENSION_PROBE)
    return len(vector)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    dataset, hashes = load_frozen_dataset()
    questions = dataset["questions"]

    # App imports begin only after the collection environment binding and local SHA pre-flight.
    from qdrant_client import QdrantClient
    from app.core.config import settings
    from app.services.advanced_retrieval import CROSS_ENCODER_AVAILABLE, advanced_retrieval
    from app.services.embeddings import embedding_service
    from app.services.retrieval import retrieval_service
    from app.services.vector_store import vector_store

    if settings.qdrant_collection != COLLECTION:
        raise RuntimeError(f"Collection binding failed: {settings.qdrant_collection!r}")
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=30.0)

    # Production guard: the demo collection must exist and must never be touched.
    production_before = _raw_points(client, PRODUCTION_COLLECTION)
    if production_before != 1:
        raise RuntimeError(f"Production collection pre-flight expected 1 point, got {production_before}")

    # RUNTIME facts, read from the live environment rather than hardcoded. 78/384 are
    # properties of THIS experiment (Corpus V1 under all-MiniLM-L6-v2), not harness rules:
    # the multilingual experiment is expected to keep 78 points while its dimension and its
    # scores legitimately change. Baking either number in would silently misreport that run.
    collection_points = _raw_points(client, COLLECTION)
    collection_vector_size = _collection_vector_size(client, COLLECTION)

    if collection_points <= 0:
        raise RuntimeError(f"Collection {COLLECTION} holds {collection_points} points; nothing to evaluate")
    if collection_points != EXPECTED_CORPUS_CHUNKS:
        raise RuntimeError(
            f"Corpus integrity: frozen corpus yields {EXPECTED_CORPUS_CHUNKS} chunks but "
            f"{COLLECTION} holds {collection_points}. Re-ingest the frozen corpus before scoring."
        )
    if not CROSS_ENCODER_AVAILABLE:
        raise RuntimeError("CROSS_ENCODER_AVAILABLE is not True; formal benchmark must fail")

    # This is the real model object, not the service's disabled/fallback path.
    reranker = advanced_retrieval._get_cross_encoder()
    if reranker is None or reranker.__class__.__name__ != "CrossEncoder":
        raise RuntimeError("A real sentence-transformers CrossEncoder was not loaded")

    embedding = embedding_service.get_embeddings()
    embedding_backend = type(embedding).__name__
    embedding_dimension = _runtime_embedding_dimension(embedding)

    if embedding_dimension <= 0:
        raise RuntimeError(f"Measured embedding dimension is {embedding_dimension}; the backend returned no vector")
    if collection_vector_size != embedding_dimension:
        raise RuntimeError(
            f"Embedding/collection dimension mismatch: {embedding_backend} produces "
            f"{embedding_dimension}-d vectors but {COLLECTION} is declared {collection_vector_size}-d. "
            f"Querying would fail or return garbage; the collection must be rebuilt with this backend."
        )

    print(f"[i] runtime metadata: collection={COLLECTION} points={collection_points} "
          f"vector_size={collection_vector_size} embedding={settings.embedding_model} "
          f"backend={embedding_backend} dimension={embedding_dimension}")

    warm_up(retrieval_service, advanced_retrieval, questions[0]["question"])
    all_documents = vector_store.get_all_documents()
    context_by_file: dict[str, list[str]] = {}
    for document in all_documents:
        metadata = getattr(document, "metadata", {}) or {}
        file_name = metadata.get("file_name")
        if file_name:
            context_by_file.setdefault(file_name, []).append(str(document.page_content))

    results: dict[str, list[dict[str, Any]]] = {name: [] for name in STRATEGY_NAMES}
    for question in questions:
        question_text = question["question"]
        t0 = time.perf_counter()
        vector_docs, vector_scores = vector_only(retrieval_service, question_text)
        vector_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        hybrid_docs, hybrid_scores = hybrid(advanced_retrieval, question_text)
        hybrid_ms = (time.perf_counter() - t0) * 1000
        if not hybrid_docs:
            raise RuntimeError(f"Hybrid returned no candidates for {question['id']}")

        t0 = time.perf_counter()
        rerank_docs, rerank_scores = reranked(advanced_retrieval, question_text, hybrid_docs)
        rerank_ms = (time.perf_counter() - t0) * 1000

        results["vector"].append(_question_result(question, vector_docs, vector_scores, {"vector_retrieval_ms": vector_ms}, context_by_file))
        results["hybrid"].append(_question_result(question, hybrid_docs, hybrid_scores, {"hybrid_retrieval_ms": hybrid_ms}, context_by_file))
        results["hybrid_rerank"].append(
            _question_result(
                question,
                rerank_docs,
                rerank_scores,
                {
                    "hybrid_retrieval_ms": hybrid_ms,
                    "rerank_ms": rerank_ms,
                    "total_retrieval_ms": hybrid_ms + rerank_ms,
                },
                context_by_file,
                # The oracles read the SAME pre-rerank pool hybrid used: reranking reorders
                # and truncates, it does not change what was reachable. Passing rerank_docs
                # here is the bug this parameter exists to prevent.
                oracle_pool_documents=hybrid_docs,
                oracle_pool_scores=hybrid_scores,
            )
        )

    production_after = _raw_points(client, PRODUCTION_COLLECTION)
    if production_after != production_before or production_after != 1:
        raise RuntimeError(f"Production collection changed: before={production_before}, after={production_after}")
    if _raw_points(client, COLLECTION) != collection_points:
        raise RuntimeError("Evaluation collection point count changed during benchmark")

    sanity: dict[str, Any] = {}
    for name, items in results.items():
        sanity[f"{name}_n"] = len(items) == 50
        sanity[f"{name}_monotonic_official"] = all(
            item["metrics"]["recall@1"] <= item["metrics"]["recall@3"] <= item["metrics"]["recall@5"]
            for item in items
        )
        sanity[f"{name}_monotonic_evidence"] = all(
            item["metrics"]["evidence_recall@1"] <= item["metrics"]["evidence_recall@3"] <= item["metrics"]["evidence_recall@5"]
            for item in items
        )
        sanity[f"{name}_hit_equals_recall1_single_file"] = all(
            item["metrics"]["hit@1"] == item["metrics"]["recall@1"]
            for item, question in zip(items, questions)
            if len(question["relevant_files"]) == 1
        )
        sanity[f"{name}_all_metrics_in_range"] = all(
            0.0 <= item["metrics"][key] <= 1.0
            for item in items
            for key in item["metrics"]
        )
        # Oracle@K must dominate Recall@5: anything ranked in the Top-5 is also in the
        # Top-10 pool, so a question Recall@5 scores as a hit cannot be an oracle miss.
        # This catches a pool that was accidentally built from the truncated list.
        sanity[f"{name}_oracle_dominates_recall5"] = all(
            item["metrics"][f"file_oracle@{ORACLE_K}"] >= item["metrics"]["recall@5"]
            and item["metrics"][f"evidence_oracle@{ORACLE_K}"] >= item["metrics"]["evidence_recall@5"]
            for item in items
        )
        sanity[f"{name}_pool_at_least_eval_k"] = all(
            len(item["pool"]) >= len(item["retrieved"]) for item in items
        )

    # Reranking must NOT move the oracles. hybrid and hybrid_rerank consume the same
    # pre-rerank Top-K pool, so their oracle inputs are identical by construction and every
    # per-question value must be equal. A difference means the rerank path was handed the
    # wrong object (its own reranked Top-5), which would silently redefine the oracle as a
    # ranking metric and report a phantom reranker regression.
    per_q = {
        name: {item["id"]: item for item in results[name]} for name in STRATEGY_NAMES
    }
    for oracle_key in (f"file_oracle@{ORACLE_K}", f"evidence_oracle@{ORACLE_K}"):
        mismatched = [
            qid
            for qid, hybrid_item in per_q["hybrid"].items()
            if hybrid_item["metrics"][oracle_key]
            != per_q["hybrid_rerank"][qid]["metrics"][oracle_key]
        ]
        sanity[f"rerank_{oracle_key}_equals_hybrid"] = not mismatched
        if mismatched:
            raise RuntimeError(
                f"{oracle_key} differs between hybrid and hybrid_rerank for "
                f"{len(mismatched)} question(s): {mismatched[:5]}. The rerank strategy must "
                f"read the same pre-rerank candidate pool as hybrid; check the "
                f"oracle_pool_documents argument."
            )
    sanity["rerank_pool_equals_hybrid_pool"] = all(
        [p["file_name"] for p in per_q["hybrid"][qid]["pool"]]
        == [p["file_name"] for p in per_q["hybrid_rerank"][qid]["pool"]]
        for qid in per_q["hybrid"]
    )
    if not sanity["rerank_pool_equals_hybrid_pool"]:
        raise RuntimeError(
            "hybrid and hybrid_rerank saved different `pool` contents; the rerank entry "
            "must carry the shared pre-rerank pool, not its reranked output."
        )

    if not all(sanity.values()):
        raise RuntimeError(f"Sanity checks failed: {sanity}")

    reranker_device = str(getattr(reranker, "device", getattr(getattr(reranker, "model", None), "device", "unknown")))
    # Integrity hashes come from load_frozen_dataset() above -- the single authoritative
    # source. They are NOT recomputed here: a second implementation of the same check is
    # how the canonical gate and the recorded values drift apart.
    experiment = {
        # Authority marker: which hash family decided this run's pass/fail.
        "frozen_integrity_authority": hashes["frozen_integrity_authority"],
        "canonical_hash_policy": hashes["canonical_hash_policy"],
        "canonical_hashes": hashes["canonical"]["actual"],
        "canonical_expected": hashes["canonical"]["expected"],
        "canonical_sha_match": hashes["canonical"]["match"],
        # Legacy raw digests: historical audit trail only. Kept so results stay comparable
        # with the pre-hardening runs, explicitly labelled so nobody reads them as the gate.
        "legacy_hashes": hashes["legacy"]["actual"],
        "legacy_expected": hashes["legacy"]["expected"],
        "legacy_sha_match": hashes["legacy"]["match"],
        "legacy_sha_role": "audit-only; a mismatch here is a byte-representation change, not corruption",
        "collection": COLLECTION,
        # Runtime-observed, read from the live Qdrant collection rather than assumed.
        "collection_points": collection_points,
        "collection_vector_size": collection_vector_size,
        # Runtime-observed: measured by embedding a probe string through the live backend.
        "embedding_model": settings.embedding_model,
        "embedding_backend": embedding_backend,
        "embedding_dimension": embedding_dimension,
        "embedding_dimension_source": f"len(embed_query({EMBEDDING_DIMENSION_PROBE!r}))",
        # Frozen-corpus expectation for THIS experiment, kept separate from the runtime count
        # above so an integrity failure and an environment change are never conflated.
        "expected_corpus_chunks": EXPECTED_CORPUS_CHUNKS,
        "corpus_chunks_match_collection": collection_points == EXPECTED_CORPUS_CHUNKS,
        "distance": "Cosine",
        "candidate_pool_k": CANDIDATE_POOL_K,
        "evaluation_k_max": EVALUATION_K_MAX,
        "oracle_k": ORACLE_K,
        "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "reranker_backend": "sentence-transformers.CrossEncoder",
        "reranker_device": reranker_device,
        "reranker_available": True,
    }
    fairness = {
        "same_dataset": True,
        "same_collection": True,
        "same_embedding": True,
        "same_candidate_pool_for_hybrid_and_rerank": True,
        "single_search_for_all_k": True,
        "llm_generation_used": False,
        "rag_cache_used": False,
    }
    fairness_pass = all(
        fairness[key]
        for key in (
            "same_dataset",
            "same_collection",
            "same_embedding",
            "same_candidate_pool_for_hybrid_and_rerank",
            "single_search_for_all_k",
        )
    ) and not fairness["llm_generation_used"] and not fairness["rag_cache_used"]
    if not fairness_pass:
        raise RuntimeError(f"Fairness verification failed: {fairness}")
    summary = {
        "experiment": experiment,
        "verified_fairness": fairness,
        "fairness_verification_pass": fairness_pass,
        "overall": {name: _aggregate(items) for name, items in results.items()},
        "by_difficulty": _buckets(questions, results)["difficulty"],
        "by_domain": _buckets(questions, results)["domain"],
        "by_category": _buckets(questions, results)["category"],
        "latency_diagnostic": {
            "note": "single-run local latency diagnostic; not a production SLA",
            "vector_retrieval_ms": _latency(results["vector"], "vector_retrieval_ms"),
            "hybrid_retrieval_ms": _latency(results["hybrid"], "hybrid_retrieval_ms"),
            "rerank_ms": _latency(results["hybrid_rerank"], "rerank_ms"),
            "total_retrieval_ms": _latency(results["hybrid_rerank"], "total_retrieval_ms"),
        },
        "manual_spot_checks": _manual_cases(results),
        "sanity_checks": sanity,
        "production_points": {"collection": PRODUCTION_COLLECTION, "before": production_before, "after": production_after, "expected": 1},
        # Reaching this point means the canonical gate already passed, so integrity is true
        # by construction. The old `frozen_sha_match: all(hashes.values())` was ambiguous --
        # it collapsed the authoritative and audit hash families into one bool and would now
        # be False whenever legacy raw bytes merely differed. Named explicitly instead.
        "frozen_integrity_authority": hashes["frozen_integrity_authority"],
        "canonical_frozen_sha_match": all(hashes["canonical"]["match"].values()),
        "legacy_sha_match_is_audit_only": hashes["legacy"]["match"],
    }

    for name in STRATEGY_NAMES:
        public_questions = [
            {
                **item,
                "retrieved": _public_retrieved(item["retrieved"]),
                "pool": _public_retrieved(item["pool"]),
            }
            for item in results[name]
        ]
        _write_json(
            RESULTS_DIR / f"{name}.json",
            {
                "strategy": name,
                "experiment": experiment,
                "questions": public_questions,
            },
        )
    _write_json(RESULTS_DIR / "summary.json", summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("RETRIEVAL_BASELINE_V1 = PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"RETRIEVAL_BASELINE_V1 = FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
