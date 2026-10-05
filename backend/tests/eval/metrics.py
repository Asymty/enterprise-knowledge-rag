"""
Retrieval Evaluation Metrics
============================

Pure functions. No IO, no network, no Qdrant, no embedding model, no LLM.

Ground truth is FILE-LEVEL: a retrieved chunk is relevant when its `file_name` is in the
question's `relevant_files`. That is a deliberate choice for this milestone -- it needs no
anchor phrases and no corpus access, so the metrics can be unit-tested on hand-computed
values. Anchor-level (chunk-level) scoring is a later, stricter mode.

A retrieved item is any object carrying a `file_name`. Both a bare str and an object with a
`.file_name` attribute / `["file_name"]` key are accepted, so this module never has to import
langchain Document.

DEDUPLICATION, and why it is not optional:

    Recall@K = |relevant files found in Top-K| / |relevant_files|

The numerator is a SET of file names. Without deduplication, a single file contributing
several chunks to the Top-K would be counted several times and Recall could exceed 1.0 --
e.g. 3 chunks from one file against 2 relevant files gives 3/2 = 1.5, which is not a recall.
Every metric here deduplicates by file_name before counting.

What this milestone deliberately does NOT do:

  * It does not read the store. Callers pass in already-retrieved results.
  * It does not decide what "retrieved" means. Ordering is the caller's contract: the list
    must already be ranked, best first, because rank is the whole input to MRR.
  * It does not silently absorb bad ground truth. An empty `relevant_files` raises rather
    than returning 0 or 1, because every one of those choices would quietly corrupt the
    aggregate. See `_validate_relevant_files`.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

__all__ = [
    "EmptyRelevantFilesError",
    "InvalidKError",
    "file_name_of",
    "recall_at_k",
    "hit_at_1",
    "reciprocal_rank",
    "aggregate_metrics",
    "DEFAULT_K_VALUES",
]

# The k values the aggregate reports. 1/3/5 are the requested metrics; they are a module
# constant rather than a literal inside aggregate_metrics so the result keys and the
# reported table can never drift apart.
DEFAULT_K_VALUES: tuple = (1, 3, 5)


class EmptyRelevantFilesError(ValueError):
    """
    Raised when a question has no relevant_files.

    Deliberately fatal. The three alternatives all produce a plausible-looking number that
    is wrong:

      * return 0.0 -- silently counts an unlabelled question as a retrieval failure, which
        drags the aggregate down and is indistinguishable from a real miss.
      * return 1.0 -- silently counts it as a success, which inflates the aggregate.
      * skip the question -- changes the denominator without saying so, so a run that
        skipped 30% of the dataset reports the same shape as a complete one.

    None of those is visible in the output. A question with no ground truth is a broken
    eval set, not a measurement, so it stops the run.
    """


class InvalidKError(ValueError):
    """Raised when k is not a positive integer."""


# =============================================================================
# Input coercion
# =============================================================================

def file_name_of(item: Any) -> Optional[str]:
    """
    Extract a file name from a retrieved result.

    Accepts a plain str, an object with a `.file_name` attribute (what the eval harness
    builds), or a mapping with a "file_name" key (what a results.json round-trip produces).
    Returns None when the item carries no usable name, so the caller can count it as a
    non-match rather than raising halfway through a run.

    Returning None rather than raising is intentional: one malformed item in a 200-item
    Top-K should degrade that question's score, not abort the whole evaluation.
    """
    if isinstance(item, str):
        return item or None

    name = getattr(item, "file_name", None)
    if name is None and isinstance(item, dict):
        name = item.get("file_name")

    return name or None


def _validate_k(k: int) -> int:
    """k must be a positive int. bool is rejected because True is an int in Python."""
    if isinstance(k, bool) or not isinstance(k, int):
        raise InvalidKError(f"k must be an int, got {type(k).__name__}")
    if k < 1:
        raise InvalidKError(f"k must be >= 1, got {k}")
    return k


def _validate_relevant_files(relevant_files: Iterable[str]) -> Set[str]:
    """
    Normalise ground truth to a non-empty set of file names.

    Raises EmptyRelevantFilesError for the empty case -- see that class for why this is
    fatal rather than defaulted.
    """
    if relevant_files is None:
        raise EmptyRelevantFilesError(
            "relevant_files is None; every question needs at least one relevant file."
        )

    # A bare string would otherwise iterate as characters ("a.md" -> {a, ., m, d}), which
    # silently produces a set of garbage tokens that match nothing.
    if isinstance(relevant_files, str):
        raise EmptyRelevantFilesError(
            f"relevant_files must be a list of file names, got the bare string "
            f"{relevant_files!r}. Wrap it: [{relevant_files!r}]."
        )

    names = {name for name in relevant_files if name}
    if not names:
        raise EmptyRelevantFilesError(
            "relevant_files is empty; cannot score a question with no ground truth. "
            "Fix the eval set rather than defaulting this to 0 or 1."
        )
    return names


def _ranked_file_names(retrieved: Optional[Sequence[Any]]) -> List[Optional[str]]:
    """File names in rank order, 1-based by list position. Unnamed items become None."""
    if not retrieved:
        return []
    return [file_name_of(item) for item in retrieved]


# =============================================================================
# Metrics
# =============================================================================

def recall_at_k(
    retrieved: Sequence[Any],
    relevant_files: Iterable[str],
    k: int,
) -> float:
    """
    Fraction of the relevant files that appear anywhere in the Top-K.

        Recall@K = |{relevant files present in retrieved[:K]}| / |relevant_files|

    The numerator is a set intersection, so a file contributing several chunks to the
    Top-K still counts once. Duplicated chunks cannot push the score above 1.0.

    Args:
        retrieved: ranked results, best first. May be empty, or shorter than k.
        relevant_files: ground-truth file names. Must be non-empty.
        k: cutoff. A k larger than len(retrieved) scores everything available, which is
           correct: there is nothing beyond the end of the list to find.

    Returns:
        float in [0, 1].

    Raises:
        EmptyRelevantFilesError: relevant_files is empty/None/a bare string.
        InvalidKError: k is not a positive int.
    """
    _validate_k(k)
    relevant = _validate_relevant_files(relevant_files)

    found = {name for name in _ranked_file_names(retrieved)[:k] if name in relevant}
    return len(found) / len(relevant)


def hit_at_1(retrieved: Sequence[Any], relevant_files: Iterable[str]) -> float:
    """
    Whether the single top-ranked result is relevant.

        Hit@1 = 1 if file_name(retrieved[0]) in relevant_files else 0

    Binary, and blind to |relevant_files|: five relevant files with one at rank 1 scores
    the same as one relevant file at rank 1. That is the point -- it measures the thing the
    user actually sees, and it stays comparable across questions whose ground-truth sets
    differ in size, which Recall@1 does not.

    Returns 0.0 when nothing was retrieved.

    Raises:
        EmptyRelevantFilesError: relevant_files is empty/None/a bare string.
    """
    relevant = _validate_relevant_files(relevant_files)

    names = _ranked_file_names(retrieved)
    if not names:
        return 0.0

    return 1.0 if names[0] in relevant else 0.0


def reciprocal_rank(retrieved: Sequence[Any], relevant_files: Iterable[str]) -> float:
    """
    Reciprocal of the rank of the first relevant result.

        RR = 1 / rank, where rank = min{ i : file_name(retrieved[i-1]) in relevant_files }
             = 0     if no retrieved result is relevant

    rank is 1-based. A hit at position 1 gives 1.0, position 3 gives 1/3, and a complete
    miss gives 0.

    Unlike Hit@1 this keeps the gradient: 1/3 and 1/5 are different numbers, so a strategy
    that pulls a correct answer from rank 3 to rank 2 registers as an improvement even
    though Hit@1 stays 0 for both. That sensitivity is why MRR is the most discriminating
    aggregate over a small corpus.

    Returns 0.0 when nothing was retrieved.

    Raises:
        EmptyRelevantFilesError: relevant_files is empty/None/a bare string.
    """
    relevant = _validate_relevant_files(relevant_files)

    for position, name in enumerate(_ranked_file_names(retrieved), start=1):
        if name in relevant:
            return 1.0 / position

    return 0.0


# =============================================================================
# Aggregation
# =============================================================================

def aggregate_metrics(
    per_question: Sequence[Dict[str, Any]],
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> Dict[str, Any]:
    """
    Mean the per-question metrics over the whole dataset.

    Args:
        per_question: one dict per question, each with
            {
              "id":           optional, carried through to "per_question" in the output
              "retrieved":    ranked results, best first
              "relevant_files": ground truth, must be non-empty
            }
        k_values: cutoffs for Recall@K. Defaults to (1, 3, 5).

    Returns:
        {
          "n_questions": int,
          "recall@1": float, "recall@3": float, "recall@5": float,
          "hit@1": float,
          "mrr": float,
          # present only when per_question is empty:
          "warning": str,
        }
        or, when k_values contains numbers other than 1/3/5, "recall@<k>" for each of those
        instead. Keys are always "recall@<k>" so they cannot collide across k values.

    An empty dataset returns every metric as 0.0 plus a "warning" key. It does NOT raise:
    an empty run is a caller misconfiguration, not a broken eval set, and returning a
    fully-formed zero result lets the harness report "0 questions" instead of crashing.
    The warning key exists so a zero table can never be mistaken for a real measurement --
    every real result has n_questions >= 1 and no warning.

    Per-question errors are NOT swallowed here. A malformed question propagates
    EmptyRelevantFilesError from the metric functions, because silently dropping questions
    would shrink the denominator without recording it.

    Note on averaging: these are macro-averages (every question weighs the same). A
    question with 5 relevant files therefore contributes as much as one with 1, which keeps
    the number interpretable. Micro-averaging would let a single multi-file question
    dominate the total.
    """
    for k in k_values:
        _validate_k(k)

    n = len(per_question or [])
    if n == 0:
        result: Dict[str, Any] = {"n_questions": 0}
        for k in k_values:
            result[f"recall@{k}"] = 0.0
        result["hit@1"] = 0.0
        result["mrr"] = 0.0
        result["warning"] = (
            "empty dataset: every metric is 0.0 by construction, not by measurement. "
            "This result must not be reported as a score."
        )
        return result

    recall_totals = {k: 0.0 for k in k_values}
    hit_total = 0.0
    rr_total = 0.0

    for question in per_question:
        retrieved = question.get("retrieved")
        relevant = question.get("relevant_files")

        # Raises EmptyRelevantFilesError on a question with no ground truth -- the whole
        # run stops, which is what we want. One unlabelled question must not quietly
        # become a 0 in the mean.
        for k in k_values:
            recall_totals[k] += recall_at_k(retrieved, relevant, k)
        hit_total += hit_at_1(retrieved, relevant)
        rr_total += reciprocal_rank(retrieved, relevant)

    result = {"n_questions": n}
    for k in k_values:
        result[f"recall@{k}"] = recall_totals[k] / n
    result["hit@1"] = hit_total / n
    result["mrr"] = rr_total / n
    return result