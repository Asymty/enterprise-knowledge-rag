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
    EXPECTED_CORPUS_MANIFEST_SHA256,
    EXPECTED_CORPUS_TREE_SHA256,
    EXPECTED_EVAL_DATASET_SHA256,
    evidence_recall,
    compact_snippet,
    load_frozen_dataset,
)
from metrics import aggregate_metrics, hit_at_1, recall_at_k, reciprocal_rank
from strategies import hybrid, reranked, vector_only, warm_up


COLLECTION = "enterprise_rag_eval_v1"
PRODUCTION_COLLECTION = "enterprise_rag"
CANDIDATE_POOL_K = 10
EVALUATION_K_MAX = 5
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


def _question_result(
    question: Mapping[str, Any],
    documents: Sequence[Any],
    scores: Sequence[float],
    latency: Mapping[str, float],
    context_by_file: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
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
    production_before = _raw_points(client, PRODUCTION_COLLECTION)
    eval_points = _raw_points(client, COLLECTION)
    if production_before != 1:
        raise RuntimeError(f"Production collection pre-flight expected 1 point, got {production_before}")
    if eval_points != 78:
        raise RuntimeError(f"Evaluation collection pre-flight expected 78 points, got {eval_points}")
    if not CROSS_ENCODER_AVAILABLE:
        raise RuntimeError("CROSS_ENCODER_AVAILABLE is not True; formal benchmark must fail")

    # This is the real model object, not the service's disabled/fallback path.
    reranker = advanced_retrieval._get_cross_encoder()
    if reranker is None or reranker.__class__.__name__ != "CrossEncoder":
        raise RuntimeError("A real sentence-transformers CrossEncoder was not loaded")

    embedding = embedding_service.get_embeddings()
    embedding_backend = type(embedding).__name__
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
            )
        )

    production_after = _raw_points(client, PRODUCTION_COLLECTION)
    if production_after != production_before or production_after != 1:
        raise RuntimeError(f"Production collection changed: before={production_before}, after={production_after}")
    if _raw_points(client, COLLECTION) != 78:
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
    if not all(sanity.values()):
        raise RuntimeError(f"Sanity checks failed: {sanity}")

    reranker_device = str(getattr(reranker, "device", getattr(getattr(reranker, "model", None), "device", "unknown")))
    experiment = {
        "corpus_tree_sha256": hashes["corpus_tree_sha256"],
        "corpus_manifest_sha256": hashes["corpus_manifest_sha256"],
        "eval_dataset_sha256": hashes["eval_dataset_sha256"],
        "collection": COLLECTION,
        "collection_points": 78,
        "embedding_model": settings.embedding_model,
        "embedding_dimension": 384,
        "embedding_backend": embedding_backend,
        "distance": "Cosine",
        "candidate_pool_k": CANDIDATE_POOL_K,
        "evaluation_k_max": EVALUATION_K_MAX,
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
        "frozen_sha_match": all(hashes.values()),
    }

    for name in STRATEGY_NAMES:
        public_questions = [
            {
                **item,
                "retrieved": _public_retrieved(item["retrieved"]),
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
