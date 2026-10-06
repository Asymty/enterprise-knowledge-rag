"""Frozen pre-flight tests: canonical SHA must be the only integrity gate.

These exercise load_frozen_dataset() against SIMULATED checkouts built in a temp directory.
The real frozen files are never written to -- a copy is made, mutated, and the module's
paths are pointed at it for the duration of each case.

Scenarios A-H from the M2.5-FINAL brief:

  A. current CRLF worktree                  legacy MATCH,    canonical MATCH   -> PASS
  B. fresh checkout (JSONs become LF)       legacy MISMATCH, canonical MATCH   -> PASS  <- the point
  C. a real edit to an eval question        legacy MISMATCH, canonical MISMATCH-> FAIL
  D. a real edit to a corpus file           canonical MISMATCH                 -> FAIL
  E. a real edit to corpus_manifest.json    canonical MISMATCH                 -> FAIL
  F. UTF-8 BOM added                        legacy mismatch, canonical MATCH   -> PASS
  G. corpus file creation order changed     canonical MATCH                    -> PASS
  H. legacy mismatch alone NEVER raises

Run:
    cd backend && python tests/eval/test_frozen_preflight.py
"""

from __future__ import annotations

import importlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import dataset_utils  # noqa: E402


REAL_EVAL_DIR = Path(__file__).resolve().parent


class SimulatedCheckout:
    """A temp copy of the frozen benchmark inputs, safe to mutate.

    Redirects the module-level paths in dataset_utils for the lifetime of the with-block, so
    load_frozen_dataset() reads the copy. Restores the originals afterwards.
    """

    def __init__(self):
        self._tmp = None
        self._saved = None

    def __enter__(self):
        self._tmp = Path(tempfile.mkdtemp())
        shutil.copytree(REAL_EVAL_DIR / "corpus", self._tmp / "corpus")
        for name in ("corpus_manifest.json", "eval_dataset.json"):
            shutil.copy2(REAL_EVAL_DIR / name, self._tmp / name)
        self._saved = (
            dataset_utils.CORPUS_DIR,
            dataset_utils.CORPUS_MANIFEST_PATH,
            dataset_utils.DATASET_PATH,
        )
        dataset_utils.CORPUS_DIR = self._tmp / "corpus"
        dataset_utils.CORPUS_MANIFEST_PATH = self._tmp / "corpus_manifest.json"
        dataset_utils.DATASET_PATH = self._tmp / "eval_dataset.json"
        return self

    def __exit__(self, *exc):
        (
            dataset_utils.CORPUS_DIR,
            dataset_utils.CORPUS_MANIFEST_PATH,
            dataset_utils.DATASET_PATH,
        ) = self._saved
        shutil.rmtree(self._tmp, ignore_errors=True)
        return False

    @property
    def root(self) -> Path:
        return self._tmp

    def to_lf(self, name: str) -> None:
        """Rewrite one JSON as a fresh checkout would: LF only, no CRLF."""
        path = self._tmp / name
        path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))

    def add_bom(self, name: str) -> None:
        path = self._tmp / name
        data = path.read_bytes()
        if not data.startswith(b"\xef\xbb\xbf"):
            path.write_bytes(b"\xef\xbb\xbf" + data)

    def edit_question(self, needle: str, replacement: str) -> None:
        """Change one character inside a real eval question string, keeping valid JSON."""
        path = self._tmp / "eval_dataset.json"
        raw = path.read_bytes()
        text = raw.decode("utf-8-sig")
        assert needle in text, f"fixture precondition failed: {needle!r} absent"
        text = text.replace(needle, replacement, 1)
        path.write_bytes(text.encode("utf-8"))

    def edit_corpus_file(self, name: str, needle: str, replacement: str) -> None:
        """Mutate corpus text while keeping every anchor intact.

        Appending is deliberately avoided: the anchors must still resolve, otherwise the
        failure would come from _validate_question rather than from the SHA gate, and the
        test would pass for the wrong reason.
        """
        path = self._tmp / "corpus" / name
        text = path.read_text(encoding="utf-8")
        assert needle in text, f"fixture precondition failed: {needle!r} absent in {name}"
        path.write_text(text.replace(needle, replacement, 1), encoding="utf-8")

    def edit_manifest_field(self) -> None:
        """Change a manifest value that nothing else depends on."""
        path = self._tmp / "corpus_manifest.json"
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        data["version"] = "tampered"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def recreate_corpus_in_reverse_order(self) -> None:
        """Rebuild the corpus tree with files written in the opposite order."""
        corpus = self._tmp / "corpus"
        files = sorted(corpus.rglob("*.md"), reverse=True)
        payload = [(p, p.read_bytes()) for p in files]
        for p in files:
            p.unlink()
        for p, data in reversed(payload):
            p.write_bytes(data)


def run_preflight() -> tuple[bool, str]:
    """Return (passed, message) for one load_frozen_dataset() attempt."""
    try:
        dataset, hashes = dataset_utils.load_frozen_dataset()
    except RuntimeError as exc:
        return False, str(exc)
    return True, hashes


# ===========================================================================
# A. current CRLF worktree
# ===========================================================================

def test_a_current_worktree_passes():
    with SimulatedCheckout() as sim:
        ok, payload = run_preflight()
        assert ok, f"current worktree should pass, got: {payload}"
        assert all(payload["canonical"]["match"].values())
        assert all(payload["legacy"]["match"].values())
        assert payload["frozen_integrity_authority"] == "canonical"


def test_a2_authority_marker_is_canonical():
    with SimulatedCheckout():
        ok, payload = run_preflight()
        assert ok
        assert payload["frozen_integrity_authority"] == "canonical"
        assert payload["canonical_hash_policy"] == dataset_utils.CANONICAL_HASH_POLICY


# ===========================================================================
# B. fresh checkout: JSONs become LF  -- THE case this milestone exists for
# ===========================================================================

def test_b_fresh_checkout_lf_still_passes():
    with SimulatedCheckout() as sim:
        sim.to_lf("eval_dataset.json")
        sim.to_lf("corpus_manifest.json")
        ok, payload = run_preflight()
        assert ok, (
            "a line-ending-only change must NOT fail the benchmark -- "
            f"this is the regression the milestone fixes; got: {payload}"
        )


def test_b2_fresh_checkout_reports_legacy_mismatch_but_passes():
    with SimulatedCheckout() as sim:
        sim.to_lf("eval_dataset.json")
        sim.to_lf("corpus_manifest.json")
        ok, payload = run_preflight()
        assert ok
        # canonical unmoved
        assert all(payload["canonical"]["match"].values())
        # legacy moved for exactly the two files that were CRLF
        assert payload["legacy"]["match"]["eval_dataset_sha256"] is False
        assert payload["legacy"]["match"]["corpus_manifest_sha256"] is False
        # ...and the corpus tree, which is already LF, did not move
        assert payload["legacy"]["match"]["corpus_tree_sha256"] is True


def test_b3_legacy_mismatch_alone_never_raises():
    """Scenario H, tested directly: a bogus legacy expectation must not fail the run."""
    saved = dataset_utils.EXPECTED_EVAL_DATASET_SHA256
    try:
        dataset_utils.EXPECTED_EVAL_DATASET_SHA256 = "0" * 64
        with SimulatedCheckout():
            ok, payload = run_preflight()
            assert ok, f"legacy mismatch must not raise, got: {payload}"
            assert payload["legacy"]["match"]["eval_dataset_sha256"] is False
            assert all(payload["canonical"]["match"].values())
    finally:
        dataset_utils.EXPECTED_EVAL_DATASET_SHA256 = saved


# ===========================================================================
# C. a real edit to an eval question
# ===========================================================================

def test_c_real_question_edit_fails():
    with SimulatedCheckout() as sim:
        sim.edit_question("上海", "广州市")
        ok, payload = run_preflight()
        assert not ok, "a real content change must fail"
        assert "Canonical frozen benchmark SHA pre-flight failed" in payload


def test_c2_real_question_edit_moves_both_hash_families():
    with SimulatedCheckout() as sim:
        sim.edit_question("上海", "广州市")
        ok, payload = run_preflight()
        assert not ok
        # canonical actually detects it, rather than the run failing on legacy alone
        assert payload.count("canonical_eval_dataset_sha256") >= 1


def test_c3_canonical_gate_reports_expected_and_actual():
    with SimulatedCheckout() as sim:
        sim.edit_question("上海", "广州市")
        ok, message = run_preflight()
        assert not ok
        assert "expected" in message and "actual" in message, (
            "the error must show both digests so a reader can see what changed"
        )


# ===========================================================================
# D. a real edit to a corpus file
# ===========================================================================

def test_d_real_corpus_edit_fails():
    with SimulatedCheckout() as sim:
        # swap a number that is not part of any anchor
        sim.edit_corpus_file("HR/attendance_policy.md", "八小时", "九小时")
        ok, payload = run_preflight()
        assert not ok, "a corpus content change must fail"
        assert "Canonical frozen benchmark SHA pre-flight failed" in payload


def test_d2_corpus_edit_moves_only_the_tree_hash():
    """Editing one file must move the tree digest, not the two JSON digests."""
    with SimulatedCheckout() as sim:
        sim.edit_corpus_file("HR/attendance_policy.md", "八小时", "九小时")
        # probe the canonical values directly, since the call itself raises
        actual_tree = dataset_utils.canonical_sha256_tree(dataset_utils.CORPUS_DIR)
        actual_ds = dataset_utils.canonical_sha256_file(dataset_utils.DATASET_PATH)
        assert actual_tree != dataset_utils.EXPECTED_CANONICAL_CORPUS_TREE_SHA256
        assert actual_ds == dataset_utils.EXPECTED_CANONICAL_EVAL_DATASET_SHA256


# ===========================================================================
# E. a real edit to corpus_manifest.json
# ===========================================================================

def test_e_manifest_edit_fails():
    with SimulatedCheckout() as sim:
        sim.edit_manifest_field()
        ok, payload = run_preflight()
        assert not ok, "a manifest content change must fail"
        assert "Canonical frozen benchmark SHA pre-flight failed" in payload


def test_e2_manifest_edit_moves_only_the_manifest_hash():
    with SimulatedCheckout() as sim:
        sim.edit_manifest_field()
        moved = dataset_utils.canonical_sha256_file(dataset_utils.CORPUS_MANIFEST_PATH)
        assert moved != dataset_utils.EXPECTED_CANONICAL_CORPUS_MANIFEST_SHA256
        assert (
            dataset_utils.canonical_sha256_file(dataset_utils.DATASET_PATH)
            == dataset_utils.EXPECTED_CANONICAL_EVAL_DATASET_SHA256
        )


# ===========================================================================
# F. BOM only
# ===========================================================================

def test_f_bom_only_change_passes():
    with SimulatedCheckout() as sim:
        sim.add_bom("eval_dataset.json")
        sim.add_bom("corpus_manifest.json")
        ok, payload = run_preflight()
        assert ok, f"a BOM-only difference must not fail the benchmark; got: {payload}"
        assert all(payload["canonical"]["match"].values())


def test_f2_bom_changes_legacy_digest():
    with SimulatedCheckout() as sim:
        before = dataset_utils.sha256_file(dataset_utils.DATASET_PATH)
        sim.add_bom("eval_dataset.json")
        after = dataset_utils.sha256_file(dataset_utils.DATASET_PATH)
        assert before != after, "the raw digest should see the BOM"
        assert (
            dataset_utils.canonical_sha256_file(dataset_utils.DATASET_PATH)
            == dataset_utils.EXPECTED_CANONICAL_EVAL_DATASET_SHA256
        )


# ===========================================================================
# G. creation / traversal order
# ===========================================================================

def test_g_recreation_order_does_not_change_tree_hash():
    with SimulatedCheckout() as sim:
        before = dataset_utils.canonical_sha256_tree(dataset_utils.CORPUS_DIR)
        sim.recreate_corpus_in_reverse_order()
        after = dataset_utils.canonical_sha256_tree(dataset_utils.CORPUS_DIR)
        assert before == after, "tree digest must not depend on write order"
        ok, payload = run_preflight()
        assert ok, f"still the same tree, so the run must pass; got: {payload}"


# ===========================================================================
# H. legacy is never a gate
# ===========================================================================

def test_h_all_three_legacy_expectations_can_be_wrong_without_raising():
    saved = (
        dataset_utils.EXPECTED_CORPUS_TREE_SHA256,
        dataset_utils.EXPECTED_CORPUS_MANIFEST_SHA256,
        dataset_utils.EXPECTED_EVAL_DATASET_SHA256,
    )
    try:
        dataset_utils.EXPECTED_CORPUS_TREE_SHA256 = "a" * 64
        dataset_utils.EXPECTED_CORPUS_MANIFEST_SHA256 = "b" * 64
        dataset_utils.EXPECTED_EVAL_DATASET_SHA256 = "c" * 64
        with SimulatedCheckout():
            ok, payload = run_preflight()
            assert ok, "no legacy expectation may gate the run"
            assert not any(payload["legacy"]["match"].values())
            assert all(payload["canonical"]["match"].values())
    finally:
        (
            dataset_utils.EXPECTED_CORPUS_TREE_SHA256,
            dataset_utils.EXPECTED_CORPUS_MANIFEST_SHA256,
            dataset_utils.EXPECTED_EVAL_DATASET_SHA256,
        ) = saved


def test_h2_canonical_expectations_still_gate():
    """Mirror of H: the canonical family must still be able to fail the run."""
    saved = dataset_utils.EXPECTED_CANONICAL_EVAL_DATASET_SHA256
    try:
        dataset_utils.EXPECTED_CANONICAL_EVAL_DATASET_SHA256 = "d" * 64
        with SimulatedCheckout():
            ok, payload = run_preflight()
            assert not ok
            assert "Canonical frozen benchmark SHA pre-flight failed" in payload
    finally:
        dataset_utils.EXPECTED_CANONICAL_EVAL_DATASET_SHA256 = saved


# ===========================================================================
# real files must be untouched by this suite
# ===========================================================================

def test_real_frozen_files_unchanged_after_suite():
    """Guard: everything above ran on copies, so the real digests must still match."""
    assert (
        dataset_utils.canonical_sha256_tree(dataset_utils.CORPUS_DIR)
        == dataset_utils.EXPECTED_CANONICAL_CORPUS_TREE_SHA256
    )
    assert (
        dataset_utils.canonical_sha256_file(dataset_utils.CORPUS_MANIFEST_PATH)
        == dataset_utils.EXPECTED_CANONICAL_CORPUS_MANIFEST_SHA256
    )
    assert (
        dataset_utils.canonical_sha256_file(dataset_utils.DATASET_PATH)
        == dataset_utils.EXPECTED_CANONICAL_EVAL_DATASET_SHA256
    )


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
    print("frozen pre-flight - canonical authority tests")
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