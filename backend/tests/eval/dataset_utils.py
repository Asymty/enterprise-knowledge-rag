"""Frozen-data helpers for the M5 retrieval benchmark.

Two hashing families live here, and the distinction matters:

LEGACY raw hashing (``sha256_bytes`` / ``sha256_file`` / ``sha256_tree``)
    Hashes the bytes exactly as they sit on disk. Reproducible only while the line endings
    on disk never change -- and they can. ``backend/tests/eval/**`` is declared
    ``text eol=lf`` in .gitattributes, so Git normalises on read, but a Windows tool that
    rewrites a file (PowerShell ``>``/``Out-File``, an editor saving CRLF) produces a
    different raw digest for identical text. ``sha256_tree`` additionally depended on
    ``rglob`` ordering, which is filesystem-dependent. These functions are kept because the
    frozen SHAs were computed with them; do not change them.

CANONICAL text hashing (``canonical_sha256_file`` / ``canonical_sha256_tree``)
    Normalises text before hashing so the digest describes CONTENT, not the byte encoding
    that happens to be on disk. Use this for anything new.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


EVAL_DIR = Path(__file__).resolve().parent
CORPUS_DIR = EVAL_DIR / "corpus"
CORPUS_MANIFEST_PATH = EVAL_DIR / "corpus_manifest.json"
DATASET_PATH = EVAL_DIR / "eval_dataset.json"
DATASET_MANIFEST_PATH = EVAL_DIR / "eval_dataset_manifest.json"

# ---------------------------------------------------------------------------
# LEGACY / FROZEN raw-byte hashes.  Historical record -- do not update in place.
#
# These still match the bytes on disk today, but they are byte-encoding sensitive: the two
# .json files are CRLF in the worktree while their Git blobs are LF, so a fresh
# `git checkout` of those paths would produce LF bytes and these values would stop matching
# for reasons that have nothing to do with the benchmark data. Keep them as the record of
# what the accepted results were computed against; add canonical values alongside rather
# than replacing these.
# ---------------------------------------------------------------------------
EXPECTED_CORPUS_TREE_SHA256 = "12e9f65afb1753c8b5489c4edfb7462ff51fe2a7ea144a1a835da7c3ac6b4815"
EXPECTED_CORPUS_MANIFEST_SHA256 = "8f769f4e19da025a05d782f5f56b3586c2549e62fcae6eb381af64c51a235462"
EXPECTED_EVAL_DATASET_SHA256 = "4bb1f59df0d504140ac0d5d55d2b90ab20de92f7a0f8ac430cca4fe512bb0dd9"
EXPECTED_QUESTION_COUNT = 50

# The canonical equivalents, recorded so a future run can assert content identity without
# depending on what line endings the checkout happened to produce.
EXPECTED_CANONICAL_CORPUS_TREE_SHA256 = "59338ee57180358b08251e67081c1cc944b70cb976f2e740619a410cffb1f6ac"
EXPECTED_CANONICAL_CORPUS_MANIFEST_SHA256 = "004d5c98c4d7cb01ce54741a2d9df2db8259f02ee95c28572694359beaf7f139"
EXPECTED_CANONICAL_EVAL_DATASET_SHA256 = "ef70367de540ceee5502602eb142cbd9122cc5c34a4fe3174852ac4919ba9b25"

# ---------------------------------------------------------------------------
# Canonical hash policy.
#
# Recorded in summary.json so a reader can tell exactly how a digest was computed instead
# of guessing. If the algorithm below changes, this string MUST change with it.
# ---------------------------------------------------------------------------
CANONICAL_HASH_POLICY = "utf8-sig->utf8;crlf/cr->lf;sorted-relative-posix-paths-v1"


# ===========================================================================
# LEGACY raw hashing (unchanged, kept for compatibility)
# ===========================================================================

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def sha256_tree(root: Path) -> str:
    """Hash sorted file bytes; this is the legacy frozen corpus-tree convention.

    NOTE: depends on traversal plus an explicit relative-POSIX sort. Prefer
    ``canonical_sha256_tree`` for new work; this is retained only because the frozen tree
    SHA was produced here.
    """
    digest = hashlib.sha256()
    files = sorted(
        (path for path in Path(root).rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    for path in files:
        digest.update(path.read_bytes())
    return digest.hexdigest()


# ===========================================================================
# Canonical text hashing
# ===========================================================================

def canonical_text_bytes(data: bytes) -> bytes:
    """Normalise text bytes so the result depends on content, not on byte encoding.

    Steps, in order:
      1. decode with ``utf-8-sig``, which transparently strips a leading UTF-8 BOM and is
         a no-op when there is none
      2. CRLF -> LF, then bare CR -> LF  (order matters: doing CR first would turn CRLF
         into LF LF)
      3. re-encode UTF-8 with no BOM

    The result is a deterministic byte string for a given logical text.
    """
    text = data.decode("utf-8-sig")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.encode("utf-8")


def canonical_sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(canonical_text_bytes(data)).hexdigest()


def canonical_sha256_file(path: Path) -> str:
    """Canonical digest of one text file's contents."""
    return canonical_sha256_bytes(Path(path).read_bytes())


def _canonical_tree_files(root: Path) -> list[Path]:
    """Every file under root, ordered by relative POSIX path.

    Sorted explicitly and never by ``rglob``'s own order: directory iteration order is
    filesystem-dependent, so relying on it would make the tree digest differ between
    machines for identical content.
    """
    root = Path(root)
    return sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )


def canonical_sha256_tree(root: Path) -> str:
    """Canonical digest of a tree, binding BOTH the relative path and the file content.

    Each entry contributes ``relative_posix_path + NUL + canonical_content + NUL``. The path
    is included deliberately: hashing only concatenated contents would make a rename
    invisible, so two different trees could share a digest. The NUL separators keep the
    encoding unambiguous -- without them a file named ``a`` containing ``bc`` and a file
    named ``ab`` containing ``c`` would hash identically.
    """
    root = Path(root)
    digest = hashlib.sha256()
    for path in _canonical_tree_files(root):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\x00")
        digest.update(canonical_text_bytes(path.read_bytes()))
        digest.update(b"\x00")
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    """Read JSON as text, tolerating a UTF-8 BOM.

    ``utf-8`` (not ``utf-8-sig``) leaves the BOM as a leading U+FEFF character, and
    ``json.loads`` then fails with "Unexpected UTF-8 BOM" on a file whose *content* is
    perfectly valid. The canonical hash deliberately treats a BOM as an encoding artefact,
    so the reader has to agree -- otherwise a BOM-only difference passes the integrity gate
    and then crashes the very next line.
    """
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_frozen_dataset() -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and validate the immutable local benchmark inputs before querying.

    INTEGRITY AUTHORITY IS CANONICAL, AND ONLY CANONICAL.

    The gate compares canonical digests (content-addressed: line endings and BOM are
    normalised away) against the canonical expected constants. A canonical mismatch means the
    benchmark DATA actually changed, and raises.

    The legacy raw-byte digests are computed and returned purely as an audit trail. A legacy
    mismatch does NOT raise, because a raw digest moves for reasons that have nothing to do
    with the data: `.gitattributes` declares `backend/tests/eval/** text eol=lf`, so a fresh
    checkout writes LF where the worktree currently holds CRLF, and the two `.json` files
    would take new raw digests on identical content. Treating that as corruption bricked the
    harness for a byte-representation difference -- see the return value, which now separates
    `authoritative` from `audit` so a caller cannot mistake one for the other.
    """
    # --- authoritative: canonical, content-addressed ---
    canonical_actual = {
        "canonical_corpus_tree_sha256": canonical_sha256_tree(CORPUS_DIR),
        "canonical_corpus_manifest_sha256": canonical_sha256_file(CORPUS_MANIFEST_PATH),
        "canonical_eval_dataset_sha256": canonical_sha256_file(DATASET_PATH),
    }
    canonical_expected = {
        "canonical_corpus_tree_sha256": EXPECTED_CANONICAL_CORPUS_TREE_SHA256,
        "canonical_corpus_manifest_sha256": EXPECTED_CANONICAL_CORPUS_MANIFEST_SHA256,
        "canonical_eval_dataset_sha256": EXPECTED_CANONICAL_EVAL_DATASET_SHA256,
    }
    canonical_mismatches = [
        f"  {name}:\n    expected {canonical_expected[name]}\n    actual   {canonical_actual[name]}"
        for name in canonical_expected
        if canonical_actual[name] != canonical_expected[name]
    ]
    if canonical_mismatches:
        raise RuntimeError(
            "Canonical frozen benchmark SHA pre-flight failed:\n"
            + "\n".join(canonical_mismatches)
        )
    canonical_match = {name: True for name in canonical_expected}

    # --- audit only: legacy raw bytes. Mismatch here is reported, never fatal. ---
    legacy_actual = {
        "corpus_tree_sha256": sha256_tree(CORPUS_DIR),
        "corpus_manifest_sha256": sha256_file(CORPUS_MANIFEST_PATH),
        "eval_dataset_sha256": sha256_file(DATASET_PATH),
    }
    legacy_expected = {
        "corpus_tree_sha256": EXPECTED_CORPUS_TREE_SHA256,
        "corpus_manifest_sha256": EXPECTED_CORPUS_MANIFEST_SHA256,
        "eval_dataset_sha256": EXPECTED_EVAL_DATASET_SHA256,
    }
    legacy_match = {
        name: legacy_actual[name] == legacy_expected[name] for name in legacy_expected
    }
    legacy_mismatched = sorted(name for name, ok in legacy_match.items() if not ok)
    if legacy_mismatched:
        print(
            "[i] Legacy raw SHA mismatch caused by byte-level representation; "
            "canonical frozen SHA matched. Not a data problem. Affected: "
            + ", ".join(legacy_mismatched)
        )

    dataset = load_json(DATASET_PATH)
    questions = dataset.get("questions")
    if not isinstance(questions, list) or len(questions) != EXPECTED_QUESTION_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_QUESTION_COUNT} questions, got "
            f"{len(questions) if isinstance(questions, list) else type(questions).__name__}"
        )

    # utf-8-sig, for the same reason as load_json: anchors are matched with `in`, so a
    # stray U+FEFF at the head of a file would not break matching, but it would put this
    # reader and the canonical hasher on different footings for no benefit.
    corpus_text = {
        path.name: path.read_text(encoding="utf-8-sig")
        for path in CORPUS_DIR.rglob("*.md")
    }
    ids: set[str] = set()
    for question in questions:
        _validate_question(question, ids, corpus_text)

    hashes: dict[str, Any] = {
        # The one thing that decides pass/fail.
        "frozen_integrity_authority": "canonical",
        "canonical_hash_policy": CANONICAL_HASH_POLICY,
        "canonical": {
            "actual": canonical_actual,
            "expected": canonical_expected,
            "match": canonical_match,
        },
        # Recorded for historical comparison with the pre-hardening results. Never a gate.
        "legacy": {
            "note": "raw-byte digests; line-ending sensitive; audit only, not a gate",
            "actual": legacy_actual,
            "expected": legacy_expected,
            "match": legacy_match,
        },
    }
    return dataset, hashes


def _validate_question(
    question: Mapping[str, Any], ids: set[str], corpus_text: Mapping[str, str]
) -> None:
    required = ("id", "question", "relevant_files", "relevant_anchors", "domain", "category", "difficulty")
    missing = [field for field in required if field not in question]
    if missing:
        raise RuntimeError(f"Question {question.get('id', '<unknown>')} missing {missing}")
    question_id = question["id"]
    if question_id in ids:
        raise RuntimeError(f"Duplicate question id: {question_id}")
    ids.add(question_id)

    relevant_files = question["relevant_files"]
    anchors = list(dict.fromkeys(question["relevant_anchors"]))
    if not relevant_files or not anchors:
        raise RuntimeError(f"Question {question_id} has empty ground truth")
    for file_name in relevant_files:
        if file_name not in corpus_text:
            raise RuntimeError(f"{question_id}: unknown relevant file {file_name}")
    for anchor in anchors:
        if not any(anchor in corpus_text[file_name] for file_name in relevant_files):
            raise RuntimeError(f"{question_id}: anchor absent from relevant files: {anchor}")


def evidence_recall(
    retrieved: Sequence[Mapping[str, Any]], relevant_anchors: Iterable[str], k: int
) -> float:
    """Unique anchor coverage in Top-K retrieved chunk contents."""
    anchors = set(relevant_anchors)
    if not anchors:
        raise ValueError("relevant_anchors must not be empty")
    contents = [str(item.get("snippet_source", "")) for item in retrieved[:k]]
    covered = {anchor for anchor in anchors if any(anchor in content for content in contents)}
    value = len(covered) / len(anchors)
    if not 0.0 <= value <= 1.0:
        raise AssertionError(f"EvidenceRecall out of range: {value}")
    return value


# ===========================================================================
# Oracle metrics
#
# Both are BINARY PER QUESTION and then macro-averaged, which makes them a retrieval
# CEILING rather than a ranking quality score: they answer "did the candidate pool contain
# the thing at all", not "was it near the top". They exist to separate two different
# failure modes that Recall@5/MRR conflate:
#
#   FileOracle@K     = 0  -> the retriever never even surfaced the right DOCUMENT.
#                            No reranking or fusion tuning can recover this; the dense
#                            embedding (or BM25) is the bottleneck.
#   FileOracle@K     = 1
#   EvidenceOracle@K = 0  -> the right document WAS in the pool, but none of the retrieved
#                            chunks held the actual evidence. This is a chunking/
#                            granularity problem, not a retrieval problem.
#
# Both must be computed on the FULL runtime candidate pool (pool_k=10). They cannot be
# recomputed from the saved result JSON, which stores only the Top-5 used for scoring.
# ===========================================================================

def file_oracle_at_k(
    retrieved: Sequence[Mapping[str, Any]],
    relevant_files: Iterable[str],
    k: int,
) -> float:
    """1.0 when at least one relevant FILE appears in the Top-K, else 0.0.

    For a multi-file question this stays binary ("at least one relevant file"), not a
    fraction: the question is whether the pool reached the right document at all, and a
    partial-coverage measure would blur that into the same thing Recall@K already reports.
    """
    files = {str(name) for name in relevant_files if name}
    if not files:
        raise ValueError("relevant_files must not be empty")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    for item in retrieved[:k]:
        if item.get("file_name") in files:
            return 1.0
    return 0.0


def evidence_oracle_at_k(
    retrieved: Sequence[Mapping[str, Any]],
    relevant_anchors: Iterable[str],
    k: int,
) -> float:
    """1.0 when at least one chunk in the Top-K contains any relevant anchor, else 0.0.

    Reads ``snippet_source`` (the full chunk text), never ``snippet`` (a 150-250 char
    display truncation). Scoring against the snippet would silently turn this into "is the
    anchor inside the first 250 characters", which is a different and much weaker question.
    """
    anchors = {str(a) for a in relevant_anchors if a}
    if not anchors:
        raise ValueError("relevant_anchors must not be empty")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    for item in retrieved[:k]:
        content = str(item.get("snippet_source", ""))
        if any(anchor in content for anchor in anchors):
            return 1.0
    return 0.0


def compact_snippet(
    text: str,
    context: str = "",
    minimum: int = 150,
    maximum: int = 250,
) -> str:
    """Keep enough source text for manual error analysis without bloating JSON."""
    material = text if len(text) >= minimum else text + ("\n[adjacent chunk context]\n" + context if context else "")
    return material[:maximum]
