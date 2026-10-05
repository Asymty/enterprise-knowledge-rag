"""
Unit tests for retrieval metrics.

Every case here is small enough to verify by hand, and the expected value is written out
as an arithmetic expression rather than a bare float wherever it is not obvious. That is
the point of this milestone: prove the FORMULA is implemented correctly before any of it
touches Qdrant, because a wrong formula discovered after a real run invalidates every
number already reported.

No Qdrant, no embedding model, no network, no LLM, no filesystem. The only import is the
module under test.

Run:
    cd backend && python tests/eval/test_metrics.py
    cd backend && python -m pytest tests/eval/test_metrics.py -v   # if pytest is available

Written as plain asserts with a __main__ runner because this repo has no pytest suite --
see CLAUDE.md, "there is no pytest suite". It still works under pytest if present.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from metrics import (  # noqa: E402
    EmptyRelevantFilesError,
    InvalidKError,
    aggregate_metrics,
    file_name_of,
    hit_at_1,
    recall_at_k,
    reciprocal_rank,
)


# =============================================================================
# Helpers
# =============================================================================

def doc(file_name: str):
    """
    Minimal stand-in for a retrieved chunk.

    Only `file_name` matters to these metrics, so the test does not import langchain. The
    real harness passes objects shaped like this one.
    """
    class _Item:
        def __init__(self, name):
            self.file_name = name

        def __repr__(self):
            return f"doc({self.name})"

    _Item.name = file_name
    return _Item(file_name)


def ranked(*names: str):
    """A ranked retrieval result, best first."""
    return [doc(n) for n in names]


def approx(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


# =============================================================================
# A. Single relevant file, hit at rank 1
# =============================================================================

def test_a_single_relevant_rank_1():
    """
    retrieved = [A, B, C],  relevant = {A}

    Recall@1 = |{A} ∩ {A}| / |{A}| = 1/1 = 1.0
    Hit@1    = retrieved[0] is A           -> 1.0
    RR       = first relevant at rank 1    -> 1/1 = 1.0
    """
    r = ranked("A.md", "B.md", "C.md")
    relevant = ["A.md"]

    assert recall_at_k(r, relevant, 1) == 1.0
    assert recall_at_k(r, relevant, 3) == 1.0
    assert hit_at_1(r, relevant) == 1.0
    assert reciprocal_rank(r, relevant) == 1.0


# =============================================================================
# B. Single relevant file, hit at rank 3
# =============================================================================

def test_b_single_relevant_rank_3():
    """
    retrieved = [X, Y, A],  relevant = {A}

    Recall@1 = 0/1 = 0.0   (A is not in Top-1)
    Recall@3 = 1/1 = 1.0   (A is in Top-3)
    Hit@1    = retrieved[0] is X, not relevant -> 0.0
    RR       = first relevant at rank 3        -> 1/3 = 0.333...
    """
    r = ranked("X.md", "Y.md", "A.md")
    relevant = ["A.md"]

    assert recall_at_k(r, relevant, 1) == 0.0
    assert recall_at_k(r, relevant, 3) == 1.0
    assert recall_at_k(r, relevant, 5) == 1.0  # k past the end finds what exists
    assert hit_at_1(r, relevant) == 0.0
    assert approx(reciprocal_rank(r, relevant), 1.0 / 3.0)


def test_b2_first_relevant_at_rank_2():
    """
    Pins the 1/rank gradient that Hit@1 throws away.

    retrieved = [X, A, Y],  relevant = {A}
    RR = 1/2 = 0.5   while Hit@1 = 0.0 -- the same Hit@1 as case B, a different MRR
    contribution. This is what makes MRR the discriminating metric on a small corpus.
    """
    r = ranked("X.md", "A.md", "Y.md")
    relevant = ["A.md"]

    assert hit_at_1(r, relevant) == 0.0
    assert reciprocal_rank(r, relevant) == 0.5


# =============================================================================
# C. Complete miss
# =============================================================================

def test_c_complete_miss():
    """
    retrieved = [X, Y, Z],  relevant = {A}   -- A appears nowhere

    Recall@1 = Recall@3 = Recall@5 = 0/1 = 0.0
    Hit@1    = 0.0
    RR       = no relevant found -> 0.0  (NOT 1/len, NOT undefined)
    """
    r = ranked("X.md", "Y.md", "Z.md")
    relevant = ["A.md"]

    assert recall_at_k(r, relevant, 1) == 0.0
    assert recall_at_k(r, relevant, 3) == 0.0
    assert recall_at_k(r, relevant, 5) == 0.0
    assert hit_at_1(r, relevant) == 0.0
    assert reciprocal_rank(r, relevant) == 0.0


def test_c2_miss_with_k_larger_than_results():
    """A miss stays a miss regardless of how deep k reaches past the list."""
    r = ranked("X.md", "Y.md")
    assert recall_at_k(r, ["A.md"], 10) == 0.0
    assert reciprocal_rank(r, ["A.md"]) == 0.0


# =============================================================================
# D. Multiple relevant files, partial hit
# =============================================================================

def test_d_multi_relevant_partial_hit():
    """
    retrieved = [A, X, B, Y, Z],  relevant = {A, B, C}   -> |relevant| = 3

    Recall@1 = |{A} ∩ {A,B,C}| / 3      = 1/3
    Recall@3 = |{A,B} ∩ {A,B,C}| / 3    = 2/3
    Recall@5 = |{A,B} ∩ {A,B,C}| / 3    = 2/3   (C never retrieved)
    Hit@1    = retrieved[0] is A, relevant -> 1.0
    RR       = first relevant at rank 1    -> 1.0
    """
    r = ranked("A.md", "X.md", "B.md", "Y.md", "Z.md")
    relevant = ["A.md", "B.md", "C.md"]

    assert approx(recall_at_k(r, relevant, 1), 1.0 / 3.0)
    assert approx(recall_at_k(r, relevant, 3), 2.0 / 3.0)
    assert approx(recall_at_k(r, relevant, 5), 2.0 / 3.0)
    assert hit_at_1(r, relevant) == 1.0
    assert reciprocal_rank(r, relevant) == 1.0


def test_d2_multi_relevant_full_hit():
    """
    retrieved = [B, A, C],  relevant = {A, B, C}

    Recall@3 = |{A,B,C}| / 3 = 3/3 = 1.0
    Hit@1    = retrieved[0] is B, which IS relevant -> 1.0
    RR       = rank 1 -> 1.0
    """
    r = ranked("B.md", "A.md", "C.md")
    relevant = ["A.md", "B.md", "C.md"]

    assert recall_at_k(r, relevant, 3) == 1.0
    assert hit_at_1(r, relevant) == 1.0
    assert reciprocal_rank(r, relevant) == 1.0


def test_d3_multi_relevant_zero_hit():
    """
    retrieved = [X, Y, Z],  relevant = {A, B}  -- none retrieved

    Recall@3 = 0/2 = 0.0    (the 0 is over the full ground-truth size, not over Top-K)
    """
    r = ranked("X.md", "Y.md", "Z.md")
    assert recall_at_k(r, ["A.md", "B.md"], 3) == 0.0


def test_d4_hit_at_1_does_not_scale_with_relevant_count():
    """
    Hit@1 must stay binary when |relevant_files| > 1.

    retrieved = [A, X, X, X, X],  relevant = {A, B, C, D, E}   -> |relevant| = 5
    Recall@1 = 1/5 = 0.2   (the rank-1 hit is only 1 of 5 answers)
    Hit@1    = 1.0         (the top result IS relevant, and that is all Hit@1 claims)
    """
    r = ranked("A.md", "X.md", "X.md", "X.md", "X.md")
    relevant = ["A.md", "B.md", "C.md", "D.md", "E.md"]

    assert approx(recall_at_k(r, relevant, 1), 1.0 / 5.0)
    assert hit_at_1(r, relevant) == 1.0
    assert reciprocal_rank(r, relevant) == 1.0


# =============================================================================
# E. Duplicate chunks from one relevant file must not double-count
# =============================================================================

def test_e_duplicate_chunks_from_same_file_counted_once():
    """
    THE deduplication case. `employee_policy.md` is chunked into several chunks, so it can
    legitimately occupy many of the Top-K slots.

    retrieved = [A1, A2, A3, X, Y],  relevant = {A}
    All three A chunks share file_name "A.md".

    Recall@1 = |{A}| / 1 = 1.0     (not 1 A chunk counted, the FILE counted)
    Recall@3 = |{A}| / 1 = 1.0     NOT 3/1 = 3.0 -- without dedup this would exceed 1.0
    Hit@1    = 1.0
    RR       = rank 1 -> 1.0
    """
    r = ranked("A.md", "A.md", "A.md", "X.md", "Y.md")

    assert recall_at_k(r, ["A.md"], 1) == 1.0
    assert recall_at_k(r, ["A.md"], 3) == 1.0
    assert recall_at_k(r, ["A.md"], 5) == 1.0
    assert hit_at_1(r, ["A.md"]) == 1.0
    assert reciprocal_rank(r, ["A.md"]) == 1.0


def test_e2_duplicates_cannot_exceed_one():
    """
    The overflow guard, stated as an invariant rather than a single number.

    retrieved = 5 chunks all from A.md,  relevant = {A, B}  -> |relevant| = 2
    Without dedup: 5/2 = 2.5, which is not a recall.
    With dedup:    |{A}| / 2 = 1/2 = 0.5
    """
    r = ranked("A.md", "A.md", "A.md", "A.md", "A.md")
    relevant = ["A.md", "B.md"]

    value = recall_at_k(r, relevant, 5)
    assert approx(value, 0.5)
    assert value <= 1.0


def test_e3_duplicates_mixed_with_real_hits():
    """
    retrieved = [A1, A2, B, B, X],  relevant = {A, B, C}

    Deduped Top-3 = {A, B}                   -> |{A,B} ∩ {A,B,C}| = 2
    Recall@1 = {A}          -> 1/3
    Recall@3 = {A, B}       -> 2/3
    Recall@5 = {A, B}       -> 2/3
    """
    r = ranked("A.md", "A.md", "B.md", "B.md", "X.md")
    relevant = ["A.md", "B.md", "C.md"]

    assert approx(recall_at_k(r, relevant, 1), 1.0 / 3.0)
    assert approx(recall_at_k(r, relevant, 3), 2.0 / 3.0)
    assert approx(recall_at_k(r, relevant, 5), 2.0 / 3.0)


def test_e4_duplicate_does_not_shift_first_relevant_rank():
    """
    Rank is the position of the first relevant item; a duplicate later in the list must not
    affect it.

    retrieved = [X, A1, A2], relevant = {A}
    RR = first relevant at rank 2 = 0.5  (A2 at rank 3 is irrelevant to MRR)
    """
    r = ranked("X.md", "A.md", "A.md")
    assert reciprocal_rank(r, ["A.md"]) == 0.5


# =============================================================================
# F. Empty relevant_files must fail loudly
# =============================================================================

def test_f_empty_relevant_files_raises():
    """
    An unlabelled question is a broken eval set, not a measurement. Every metric must raise
    rather than defaulting to 0 or 1 -- both of those are plausible-looking wrong answers.
    """
    r = ranked("A.md", "B.md")

    for fn, args in [
        (recall_at_k, (r, [], 3)),
        (hit_at_1, (r, [])),
        (reciprocal_rank, (r, [])),
    ]:
        try:
            fn(*args)
        except EmptyRelevantFilesError:
            pass
        else:
            raise AssertionError(f"{fn.__name__} accepted empty relevant_files")


def test_f2_none_relevant_files_raises():
    r = ranked("A.md")
    for fn in (recall_at_k, hit_at_1, reciprocal_rank):
        try:
            fn(r, None, 3) if fn is recall_at_k else fn(r, None)
        except EmptyRelevantFilesError:
            pass
        else:
            raise AssertionError(f"{fn.__name__} accepted relevant_files=None")


def test_f3_bare_string_relevant_files_raises():
    """
    A bare string iterates as characters, so "a.md" would become {"a", ".", "m", "d"} and
    match nothing -- a silent 0 that looks like a retrieval failure. Reject it explicitly.
    """
    r = ranked("a.md")
    try:
        recall_at_k(r, "a.md", 1)
    except EmptyRelevantFilesError:
        pass
    else:
        raise AssertionError("bare-string relevant_files was accepted")


def test_f4_empty_relevant_files_propagates_through_aggregate():
    """
    The aggregate must NOT swallow the error. Dropping an unlabelled question would shrink
    the denominator without recording it.
    """
    dataset = [
        {"id": "q1", "retrieved": ranked("A.md"), "relevant_files": ["A.md"]},
        {"id": "q2", "retrieved": ranked("B.md"), "relevant_files": []},
    ]
    try:
        aggregate_metrics(dataset)
    except EmptyRelevantFilesError:
        pass
    else:
        raise AssertionError("aggregate_metrics swallowed an empty relevant_files")


# =============================================================================
# G. Empty retrieved
# =============================================================================

def test_g_empty_retrieved():
    """
    Nothing retrieved against a valid ground truth is an ordinary miss, not an error.
    Every metric is 0.0 -- and crucially relevant_files is still validated, so this is a
    real 0 and not an artefact of an unlabelled question.
    """
    assert recall_at_k([], ["A.md"], 3) == 0.0
    assert hit_at_1([], ["A.md"]) == 0.0
    assert reciprocal_rank([], ["A.md"]) == 0.0


def test_g2_none_retrieved_is_treated_as_empty():
    assert recall_at_k(None, ["A.md"], 3) == 0.0
    assert hit_at_1(None, ["A.md"]) == 0.0
    assert reciprocal_rank(None, ["A.md"]) == 0.0


def test_g3_empty_retrieved_still_rejects_empty_relevant():
    """Empty retrieved must not mask a broken ground truth."""
    try:
        recall_at_k([], [], 3)
    except EmptyRelevantFilesError:
        pass
    else:
        raise AssertionError("empty retrieved short-circuited the ground-truth check")


# =============================================================================
# H. k larger than the number of retrieved results
# =============================================================================

def test_h_k_larger_than_retrieved():
    """
    k past the end of the list scores everything that exists. There is nothing beyond the
    end to find, so this is not an error and not a penalty.

    retrieved = [X, A], relevant = {A}
    Recall@5 = 1/1 = 1.0   same as Recall@3
    """
    r = ranked("X.md", "A.md")

    assert recall_at_k(r, ["A.md"], 3) == 1.0
    assert recall_at_k(r, ["A.md"], 5) == 1.0
    assert recall_at_k(r, ["A.md"], 100) == 1.0


def test_h2_k_larger_than_retrieved_with_multiple_relevant():
    """
    retrieved = [A], relevant = {A, B, C}
    Recall@5 = |{A}| / 3 = 1/3  -- a deep k cannot recover files never retrieved.
    """
    r = ranked("A.md")
    assert approx(recall_at_k(r, ["A.md", "B.md", "C.md"], 5), 1.0 / 3.0)


def test_h3_k_equal_to_len_is_the_full_list():
    r = ranked("A.md", "B.md", "C.md")
    assert recall_at_k(r, ["A.md", "C.md"], 3) == 1.0


# =============================================================================
# k validation
# =============================================================================

def test_invalid_k_raises():
    """k must be a positive int. bool is rejected because True == 1 in Python."""
    r = ranked("A.md")
    for bad_k in (0, -1, True, 1.5, "3", None):
        try:
            recall_at_k(r, ["A.md"], bad_k)
        except InvalidKError:
            pass
        else:
            raise AssertionError(f"recall_at_k accepted k={bad_k!r}")


# =============================================================================
# file_name_of coercion
# =============================================================================

def test_file_name_of_accepts_str_obj_and_dict():
    assert file_name_of("a.md") == "a.md"
    assert file_name_of(doc("b.md")) == "b.md"
    assert file_name_of({"file_name": "c.md"}) == "c.md"


def test_file_name_of_returns_none_for_unusable_items():
    """
    A malformed item degrades its own question's score rather than aborting the run.
    """
    assert file_name_of("") is None
    assert file_name_of({"file_name": ""}) is None
    assert file_name_of(object()) is None
    assert file_name_of(None) is None


def test_unnameable_items_do_not_match():
    r = [object(), {"file_name": ""}, "A.md"]
    assert recall_at_k(r, ["A.md"], 3) == 1.0
    assert hit_at_1(r, ["A.md"]) == 0.0          # first item carries no name
    assert reciprocal_rank(r, ["A.md"]) == 1.0 / 3.0


# =============================================================================
# aggregate_metrics
# =============================================================================

def test_aggregate_hand_computed():
    """
    Five questions, each verified above, averaged by hand.

    q1  retrieved=[A,B,C]      relevant={A}         R@1=1.0  R@3=1.0  R@5=1.0  Hit@1=1.0  RR=1.0
    q2  retrieved=[X,Y,A]      relevant={A}         R@1=0.0  R@3=1.0  R@5=1.0  Hit@1=0.0  RR=1/3
    q3  retrieved=[X,Y,Z]      relevant={A}         R@1=0.0  R@3=0.0  R@5=0.0  Hit@1=0.0  RR=0.0
    q4  retrieved=[A,X,B,Y,Z]  relevant={A,B,C}     R@1=1/3  R@3=2/3  R@5=2/3  Hit@1=1.0  RR=1.0
    q5  retrieved=[A,A,A,X,Y]  relevant={A,B,C}     R@1=1/3  R@3=1/3  R@5=1/3  Hit@1=1.0  RR=1.0

    Recall@1 = (1.0 + 0.0 + 0.0 + 1/3 + 1/3) / 5 = (1.666666...) / 5 = 1/3
    Recall@3 = (1.0 + 1.0 + 0.0 + 2/3 + 1/3) / 5 = (3.0) / 5 = 0.6
    Recall@5 = (1.0 + 1.0 + 0.0 + 2/3 + 1/3) / 5 = (3.0) / 5 = 0.6
    Hit@1    = (1 + 0 + 0 + 1 + 1) / 5 = 3/5 = 0.6
    MRR      = (1.0 + 1/3 + 0.0 + 1.0 + 1.0) / 5 = (3.333333...) / 5 = 2/3
    """
    dataset = [
        {"id": "q1", "retrieved": ranked("A.md", "B.md", "C.md"),        "relevant_files": ["A.md"]},
        {"id": "q2", "retrieved": ranked("X.md", "Y.md", "A.md"),        "relevant_files": ["A.md"]},
        {"id": "q3", "retrieved": ranked("X.md", "Y.md", "Z.md"),        "relevant_files": ["A.md"]},
        {"id": "q4", "retrieved": ranked("A.md", "X.md", "B.md", "Y.md", "Z.md"),
         "relevant_files": ["A.md", "B.md", "C.md"]},
        {"id": "q5", "retrieved": ranked("A.md", "A.md", "A.md", "X.md", "Y.md"),
         "relevant_files": ["A.md", "B.md", "C.md"]},
    ]

    m = aggregate_metrics(dataset)

    assert m["n_questions"] == 5
    assert approx(m["recall@1"], 1.0 / 3.0)
    assert approx(m["recall@3"], 0.6)
    assert approx(m["recall@5"], 0.6)
    assert approx(m["hit@1"], 0.6)
    assert approx(m["mrr"], 2.0 / 3.0)
    assert "warning" not in m


def test_aggregate_all_metrics_in_unit_range():
    """The published contract: every aggregate is a float in [0, 1]."""
    dataset = [
        {"id": "q1", "retrieved": ranked("A.md", "A.md"), "relevant_files": ["A.md"]},
        {"id": "q2", "retrieved": ranked("B.md"),         "relevant_files": ["A.md"]},
    ]
    m = aggregate_metrics(dataset)

    for key in ("recall@1", "recall@3", "recall@5", "hit@1", "mrr"):
        value = m[key]
        assert isinstance(value, float), f"{key} is {type(value).__name__}, expected float"
        assert 0.0 <= value <= 1.0, f"{key} = {value} is outside [0, 1]"


def test_aggregate_perfect_run():
    """Every question hits at rank 1 -> every metric is exactly 1.0."""
    dataset = [
        {"id": "q1", "retrieved": ranked("A.md", "X.md"), "relevant_files": ["A.md"]},
        {"id": "q2", "retrieved": ranked("B.md", "X.md"), "relevant_files": ["B.md"]},
        {"id": "q3", "retrieved": ranked("C.md", "X.md"), "relevant_files": ["C.md"]},
    ]
    m = aggregate_metrics(dataset)

    assert m["recall@1"] == 1.0
    assert m["recall@3"] == 1.0
    assert m["recall@5"] == 1.0
    assert m["hit@1"] == 1.0
    assert m["mrr"] == 1.0


def test_aggregate_total_failure():
    """Nothing relevant retrieved anywhere -> every metric is exactly 0.0."""
    dataset = [
        {"id": "q1", "retrieved": ranked("X.md", "Y.md"), "relevant_files": ["A.md"]},
        {"id": "q2", "retrieved": ranked("X.md"),         "relevant_files": ["B.md"]},
    ]
    m = aggregate_metrics(dataset)

    assert m["recall@1"] == 0.0
    assert m["recall@3"] == 0.0
    assert m["recall@5"] == 0.0
    assert m["hit@1"] == 0.0
    assert m["mrr"] == 0.0


def test_aggregate_empty_dataset_warns():
    """
    An empty run returns zeros WITH a warning key, so a zero table can never be mistaken
    for a measurement.
    """
    m = aggregate_metrics([])

    assert m["n_questions"] == 0
    assert m["recall@1"] == 0.0
    assert m["hit@1"] == 0.0
    assert m["mrr"] == 0.0
    assert "warning" in m


def test_aggregate_custom_k_values():
    """Result keys track k_values, so callers can request other cutoffs without collision."""
    dataset = [
        {"id": "q1", "retrieved": ranked("X.md", "Y.md", "Z.md", "A.md"), "relevant_files": ["A.md"]},
    ]
    m = aggregate_metrics(dataset, k_values=(2, 4))

    assert approx(m["recall@2"], 0.0)
    assert approx(m["recall@4"], 1.0)
    assert "recall@1" not in m
    assert "recall@3" not in m
    # Hit@1 and MRR are independent of k_values
    assert m["hit@1"] == 0.0
    assert m["mrr"] == 0.25


def test_aggregate_is_macro_average():
    """
    Every question weighs the same, whatever its ground-truth size.

    q1: |relevant|=1, hit            -> Recall@1 = 1.0
    q2: |relevant|=9, hit at rank 1  -> Recall@1 = 1/9
    Macro mean = (1.0 + 1/9) / 2 = 5/9
    (Micro-averaging would give 2/10 = 0.2 instead -- q2 would dominate.)
    """
    dataset = [
        {"id": "q1", "retrieved": ranked("A.md"), "relevant_files": ["A.md"]},
        {"id": "q2", "retrieved": ranked("A.md"),
         "relevant_files": ["A.md"] + [f"f{i}.md" for i in range(1, 9)]},
    ]
    m = aggregate_metrics(dataset)
    assert approx(m["recall@1"], 5.0 / 9.0)


def test_aggregate_reports_n_questions():
    dataset = [{"id": f"q{i}", "retrieved": ranked("A.md"), "relevant_files": ["A.md"]}
               for i in range(7)]
    assert aggregate_metrics(dataset)["n_questions"] == 7


# =============================================================================
# Runner
# =============================================================================

def _all_tests():
    return [
        (name, fn) for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]


def main() -> int:
    tests = _all_tests()
    failures = []

    print("=" * 72)
    print("Retrieval metrics - unit tests")
    print("=" * 72)
    print(f"{len(tests)} cases\n")

    for name, fn in tests:
        try:
            fn()
        except AssertionError as e:
            failures.append((name, e))
            print(f"  [FAIL] {name}")
            print(f"         {e}")
        except Exception as e:  # noqa: BLE001 - report, do not mask
            failures.append((name, e))
            print(f"  [ERROR] {name}: {type(e).__name__}: {e}")
        else:
            print(f"  [ok]   {name}")

    print()
    print("=" * 72)
    if failures:
        print(f"FAILED: {len(failures)}/{len(tests)}")
        for name, e in failures:
            print(f"  - {name}: {e}")
        print("=" * 72)
        return 1

    print(f"PASSED: {len(tests)}/{len(tests)}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())