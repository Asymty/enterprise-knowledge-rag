"""The three retrieval calls used by the M5 benchmark."""

from __future__ import annotations

from typing import Any


def warm_up(retrieval_service: Any, advanced_retrieval: Any, question: str) -> None:
    vector = retrieval_service.retrieve(question, k=10, with_scores=True)
    hybrid_docs, _ = advanced_retrieval.hybrid_search(question, k=10, with_scores=True)
    advanced_retrieval.rerank(question, hybrid_docs, top_k=5)
    if not vector.documents or not hybrid_docs:
        raise RuntimeError("Warm-up returned no retrieval results")


def vector_only(retrieval_service: Any, question: str) -> tuple[list[Any], list[float]]:
    result = retrieval_service.retrieve(question, k=10, with_scores=True)
    return list(result.documents), [float(score) for score in result.scores]


def hybrid(advanced_retrieval: Any, question: str) -> tuple[list[Any], list[float]]:
    documents, scores = advanced_retrieval.hybrid_search(
        question, k=10, with_scores=True
    )
    return list(documents), [float(score) for score in scores]


def reranked(
    advanced_retrieval: Any, question: str, candidate_pool: list[Any]
) -> tuple[list[Any], list[float]]:
    ranked = advanced_retrieval.rerank(question, candidate_pool, top_k=5)
    return [document for document, _ in ranked], [float(score) for _, score in ranked]
