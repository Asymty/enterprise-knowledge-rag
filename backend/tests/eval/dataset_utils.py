"""Frozen-data helpers for the M5 retrieval benchmark."""

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

EXPECTED_CORPUS_TREE_SHA256 = "12e9f65afb1753c8b5489c4edfb7462ff51fe2a7ea144a1a835da7c3ac6b4815"
EXPECTED_CORPUS_MANIFEST_SHA256 = "8f769f4e19da025a05d782f5f56b3586c2549e62fcae6eb381af64c51a235462"
EXPECTED_EVAL_DATASET_SHA256 = "4bb1f59df0d504140ac0d5d55d2b90ab20de92f7a0f8ac430cca4fe512bb0dd9"
EXPECTED_QUESTION_COUNT = 50


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def sha256_tree(root: Path) -> str:
    """Hash sorted file bytes; this is the frozen corpus-tree convention."""
    digest = hashlib.sha256()
    files = sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    for path in files:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_frozen_dataset() -> tuple[dict[str, Any], dict[str, str]]:
    """Load and validate the immutable local benchmark inputs before querying."""
    actual = {
        "corpus_tree_sha256": sha256_tree(CORPUS_DIR),
        "corpus_manifest_sha256": sha256_file(CORPUS_MANIFEST_PATH),
        "eval_dataset_sha256": sha256_file(DATASET_PATH),
    }
    expected = {
        "corpus_tree_sha256": EXPECTED_CORPUS_TREE_SHA256,
        "corpus_manifest_sha256": EXPECTED_CORPUS_MANIFEST_SHA256,
        "eval_dataset_sha256": EXPECTED_EVAL_DATASET_SHA256,
    }
    mismatches = [
        f"{name}: expected {expected[name]}, got {actual[name]}"
        for name in expected
        if actual[name] != expected[name]
    ]
    if mismatches:
        raise RuntimeError("Frozen SHA pre-flight failed:\n" + "\n".join(mismatches))

    dataset = load_json(DATASET_PATH)
    questions = dataset.get("questions")
    if not isinstance(questions, list) or len(questions) != EXPECTED_QUESTION_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_QUESTION_COUNT} questions, got "
            f"{len(questions) if isinstance(questions, list) else type(questions).__name__}"
        )

    corpus_text = {
        path.name: path.read_text(encoding="utf-8")
        for path in CORPUS_DIR.rglob("*.md")
    }
    ids: set[str] = set()
    for question in questions:
        _validate_question(question, ids, corpus_text)
    return dataset, actual


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


def compact_snippet(
    text: str,
    context: str = "",
    minimum: int = 150,
    maximum: int = 250,
) -> str:
    """Keep enough source text for manual error analysis without bloating JSON."""
    material = text if len(text) >= minimum else text + ("\n[adjacent chunk context]\n" + context if context else "")
    return material[:maximum]
