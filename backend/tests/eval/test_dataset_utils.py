"""Unit tests for the reproducibility and oracle additions in dataset_utils.

Run:
    cd backend && python tests/eval/test_dataset_utils.py

Written as plain asserts with a __main__ runner: this repo has no pytest dependency
(CLAUDE.md, "there is no pytest suite"), matching test_metrics.py.

Covers the seven canonical-hash invariants and the oracle definitions. No Qdrant, no
network, no embedding model; the hash tests build their own temporary trees.
"""

from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dataset_utils import (  # noqa: E402
    CANONICAL_HASH_POLICY,
    canonical_sha256_bytes,
    canonical_sha256_file,
    canonical_sha256_tree,
    evidence_oracle_at_k,
    file_oracle_at_k,
)


def approx(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


def row(file_name, content):
    """A retrieved-candidate row as the harness builds it."""
    return {"file_name": file_name, "snippet_source": content}


# ===========================================================================
# A. LF and CRLF hash the same
# ===========================================================================

def test_a_lf_and_crlf_identical():
    lf = "第一行\n第二行\n第三行\n".encode("utf-8")
    crlf = "第一行\r\n第二行\r\n第三行\r\n".encode("utf-8")
    assert lf != crlf, "the two byte strings must actually differ"
    assert canonical_sha256_bytes(lf) == canonical_sha256_bytes(crlf)


def test_a2_lf_and_crlf_files_identical(tmp: Path):
    (tmp / "a.txt").write_bytes("x\ny\nz\n".encode("utf-8"))
    first = canonical_sha256_file(tmp / "a.txt")
    (tmp / "a.txt").write_bytes("x\r\ny\r\nz\r\n".encode("utf-8"))
    assert canonical_sha256_file(tmp / "a.txt") == first


# ===========================================================================
# B. bare CR hashes the same as LF
# ===========================================================================

def test_b_bare_cr_identical_to_lf():
    """Old-Mac line endings must normalise too, not just CRLF."""
    lf = "alpha\nbeta\n".encode("utf-8")
    cr = "alpha\rbeta\r".encode("utf-8")
    assert canonical_sha256_bytes(lf) == canonical_sha256_bytes(cr)


def test_b2_mixed_endings_identical():
    mixed = "a\r\nb\nc\rd".encode("utf-8")
    lf = "a\nb\nc\nd".encode("utf-8")
    assert canonical_sha256_bytes(mixed) == canonical_sha256_bytes(lf)


def test_b3_crlf_is_not_double_converted():
    """CRLF -> LF, then a separate CR pass. Wrong order gives '\\n\\n'."""
    crlf = "a\r\nb".encode("utf-8")
    assert canonical_sha256_bytes(crlf) == canonical_sha256_bytes("a\nb".encode("utf-8"))
    assert canonical_sha256_bytes("a\r\nb".encode("utf-8")) != canonical_sha256_bytes(
        "a\n\nb".encode("utf-8")
    )


# ===========================================================================
# C. UTF-8 BOM is stripped
# ===========================================================================

def test_c_bom_and_no_bom_identical():
    plain = "héllo wörld".encode("utf-8")
    with_bom = b"\xef\xbb\xbf" + plain
    assert plain != with_bom, "the two byte strings must actually differ"
    assert canonical_sha256_bytes(plain) == canonical_sha256_bytes(with_bom)


def test_c2_bom_plus_crlf_identical():
    assert canonical_sha256_bytes(b"\xef\xbb\xbfa\r\nb") == canonical_sha256_bytes(
        "a\nb".encode("utf-8")
    )


def test_c3_interior_bom_bytes_are_not_stripped():
    """Only a LEADING BOM is an encoding artefact; the same bytes mid-file are content."""
    interior = "a\xef\xbb\xbfb".encode("utf-8")
    assert canonical_sha256_bytes(interior) != canonical_sha256_bytes("ab".encode("utf-8"))


# ===========================================================================
# D. real character change MUST change the hash
# ===========================================================================

def test_d_content_change_changes_hash():
    assert canonical_sha256_bytes("500 元\n".encode("utf-8")) != canonical_sha256_bytes(
        "550 元\n".encode("utf-8")
    )


def test_d2_whitespace_change_changes_hash():
    """Canonicalisation normalises LINE ENDINGS, not all whitespace."""
    assert canonical_sha256_bytes("a b".encode("utf-8")) != canonical_sha256_bytes(
        "a  b".encode("utf-8")
    )


def test_d3_trailing_newline_matters():
    assert canonical_sha256_bytes("a".encode("utf-8")) != canonical_sha256_bytes(
        "a\n".encode("utf-8")
    )


# ===========================================================================
# E. tree hash is order-independent
# ===========================================================================

def _build(root: Path, files: dict[str, bytes]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for rel, data in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return root


def test_e_tree_creation_order_irrelevant(tmp: Path):
    a = _build(tmp / "one", {"z.md": b"zz\n", "a.md": b"aa\n", "m/n.md": b"nn\n"})
    b = _build(tmp / "two", {"m/n.md": b"nn\n", "a.md": b"aa\n", "z.md": b"zz\n"})
    assert canonical_sha256_tree(a) == canonical_sha256_tree(b)


def test_e2_tree_line_ending_irrelevant(tmp: Path):
    a = _build(tmp / "lf", {"x.md": b"a\nb\n"})
    b = _build(tmp / "crlf", {"x.md": b"a\r\nb\r\n"})
    assert canonical_sha256_tree(a) == canonical_sha256_tree(b)


def test_e3_single_file_subtree_matches_entry(tmp: Path):
    root = _build(tmp / "r", {"only.md": "内容\n".encode("utf-8")})
    assert canonical_sha256_tree(root) != canonical_sha256_file(root / "only.md"), (
        "tree binds the path too, so it must differ from a bare content hash"
    )


# ===========================================================================
# F. rename MUST change the tree hash
# ===========================================================================

def test_f_rename_changes_tree_hash(tmp: Path):
    """Hashing only contents would let a rename go unnoticed."""
    a = _build(tmp / "before", {"old_name.md": b"same bytes\n"})
    b = _build(tmp / "after", {"new_name.md": b"same bytes\n"})
    assert canonical_sha256_file(a / "old_name.md") == canonical_sha256_file(b / "new_name.md")
    assert canonical_sha256_tree(a) != canonical_sha256_tree(b)


def test_f2_move_between_directories_changes_tree_hash(tmp: Path):
    a = _build(tmp / "flat", {"n.md": b"x\n"})
    b = _build(tmp / "nested", {"sub/n.md": b"x\n"})
    assert canonical_sha256_tree(a) != canonical_sha256_tree(b)


def test_f3_nul_separator_disambiguates(tmp: Path):
    """Without separators 'a'/'bc' and 'ab'/'c' would concatenate identically."""
    a = _build(tmp / "p", {"a": b"bc"})
    b = _build(tmp / "q", {"ab": b"c"})
    assert canonical_sha256_tree(a) != canonical_sha256_tree(b)


# ===========================================================================
# G. content change MUST change the tree hash
# ===========================================================================

def test_g_tree_content_change_detected(tmp: Path):
    a = _build(tmp / "v1", {"x.md": b"500\n"})
    b = _build(tmp / "v2", {"x.md": b"550\n"})
    assert canonical_sha256_tree(a) != canonical_sha256_tree(b)


def test_g2_added_file_changes_tree_hash(tmp: Path):
    a = _build(tmp / "before", {"x.md": b"x\n"})
    b = _build(tmp / "after", {"x.md": b"x\n", "y.md": b"y\n"})
    assert canonical_sha256_tree(a) != canonical_sha256_tree(b)


def test_g3_removed_file_changes_tree_hash(tmp: Path):
    a = _build(tmp / "two", {"x.md": b"x\n", "y.md": b"y\n"})
    b = _build(tmp / "one", {"x.md": b"x\n"})
    assert canonical_sha256_tree(a) != canonical_sha256_tree(b)


def test_g4_empty_tree_is_stable(tmp: Path):
    (tmp / "empty").mkdir()
    assert canonical_sha256_tree(tmp / "empty") == hashlib.sha256(b"").hexdigest()


# ===========================================================================
# policy string
# ===========================================================================

def test_policy_string_is_present_and_specific():
    assert CANONICAL_HASH_POLICY == (
        "utf8-sig->utf8;crlf/cr->lf;sorted-relative-posix-paths-v1"
    )


# ===========================================================================
# Oracle: FileOracle@K
# ===========================================================================

def test_oracle_file_rank1():
    r = [row("A.md", "x"), row("B.md", "y")]
    assert file_oracle_at_k(r, ["A.md"], 10) == 1.0


def test_oracle_file_rank10():
    r = [row(f"f{i}.md", "x") for i in range(1, 10)] + [row("A.md", "x")]
    assert len(r) == 10
    assert file_oracle_at_k(r, ["A.md"], 10) == 1.0


def test_oracle_file_rank11_misses_at_k10():
    """The pool is exactly 10, so a hit at position 11 is outside it."""
    r = [row(f"f{i}.md", "x") for i in range(1, 11)] + [row("A.md", "x")]
    assert file_oracle_at_k(r, ["A.md"], 10) == 0.0


def test_oracle_file_complete_miss():
    r = [row("X.md", "x"), row("Y.md", "y")]
    assert file_oracle_at_k(r, ["A.md"], 10) == 0.0


def test_oracle_file_empty_candidates():
    assert file_oracle_at_k([], ["A.md"], 10) == 0.0


def test_oracle_file_multi_file_is_binary_ceiling():
    """Multi-file questions stay 0/1: any ONE relevant file satisfies the ceiling."""
    r = [row("X.md", "x"), row("B.md", "y")]
    assert file_oracle_at_k(r, ["A.md", "B.md", "C.md"], 10) == 1.0
    assert file_oracle_at_k(r, ["A.md", "C.md"], 10) == 0.0


def test_oracle_file_k_truncates():
    r = [row("X.md", "x"), row("A.md", "y")]
    assert file_oracle_at_k(r, ["A.md"], 1) == 0.0
    assert file_oracle_at_k(r, ["A.md"], 2) == 1.0


def test_oracle_file_rejects_bad_arguments():
    for bad_k in (0, -1):
        try:
            file_oracle_at_k([row("A.md", "x")], ["A.md"], bad_k)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted k={bad_k}")
    try:
        file_oracle_at_k([row("A.md", "x")], [], 10)
    except ValueError:
        pass
    else:
        raise AssertionError("accepted empty relevant_files")


# ===========================================================================
# Oracle: EvidenceOracle@K
# ===========================================================================

def test_oracle_evidence_rank1():
    r = [row("A.md", "普通员工赴上海出差，住宿标准不超过每晚 500 元。")]
    assert evidence_oracle_at_k(r, ["不超过每晚 500 元"], 10) == 1.0


def test_oracle_evidence_rank10():
    r = [row(f"f{i}.md", "无关内容") for i in range(1, 10)]
    r.append(row("A.md", "上海的住宿标准为每晚不超过五百元。"))
    assert evidence_oracle_at_k(r, ["不超过五百元"], 10) == 1.0


def test_oracle_evidence_rank11_misses():
    r = [row(f"f{i}.md", "无关") for i in range(1, 11)]
    r.append(row("A.md", "不超过五百元"))
    assert evidence_oracle_at_k(r, ["不超过五百元"], 10) == 0.0


def test_oracle_wrong_chunk_from_correct_file():
    """THE case these two metrics exist to separate.

    The right document is retrieved but the retrieved CHUNK is a different clause of it.
    FileOracle = 1 (the document was reachable), EvidenceOracle = 0 (the evidence was not).
    A ranking metric cannot express this distinction.
    """
    r = [row("travel_policy.md", "## 北京住宿标准\n普通员工赴北京出差，住宿标准不超过每晚 550 元。")]
    assert file_oracle_at_k(r, ["travel_policy.md"], 10) == 1.0
    assert evidence_oracle_at_k(r, ["不超过每晚 500 元"], 10) == 0.0


def test_oracle_evidence_any_anchor_counts():
    """Multiple anchors: one present is enough for the binary ceiling."""
    r = [row("A.md", "只包含第二条证据")]
    assert evidence_oracle_at_k(r, ["第一条证据", "第二条证据"], 10) == 1.0


def test_oracle_evidence_empty_candidates():
    assert evidence_oracle_at_k([], ["某证据"], 10) == 0.0


def test_oracle_evidence_ignores_snippet_truncation():
    """Scoring must read snippet_source, not the display snippet."""
    long_chunk = "x" * 400 + "关键证据"
    r = [{"file_name": "A.md", "snippet": long_chunk[:250], "snippet_source": long_chunk}]
    assert evidence_oracle_at_k(r, ["关键证据"], 10) == 1.0, (
        "must read the full chunk; the truncated snippet would miss this"
    )


def test_oracle_evidence_rejects_bad_arguments():
    for bad_k in (0, -1):
        try:
            evidence_oracle_at_k([row("A.md", "证据")], ["证据"], bad_k)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted k={bad_k}")
    try:
        evidence_oracle_at_k([row("A.md", "证据")], [], 10)
    except ValueError:
        pass
    else:
        raise AssertionError("accepted empty relevant_anchors")


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
    print("dataset_utils - canonical hashing + oracle tests")
    print("=" * 72)
    print(f"{len(tests)} cases\n")

    for name, fn in tests:
        try:
            if fn.__code__.co_argcount == 1:
                with tempfile.TemporaryDirectory() as tmp:
                    fn(Path(tmp))
            else:
                fn()
        except AssertionError as exc:
            failures.append((name, exc))
            print(f"  [FAIL] {name}")
            print(f"         {exc}")
        except Exception as exc:  # noqa: BLE001 - report, never mask
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