"""M4: SiliconFlow cloud `BAAI/bge-reranker-v2-m3` reranking adapter.

This is a SELF-CONTAINED adapter for the frozen retrieval benchmark. It deliberately does
NOT touch ``app/services/advanced_retrieval.py``: that module's ``rerank()`` loads a local
sentence-transformers CrossEncoder and is on the production request path. M4 changes only
which reranker model and which inference backend score the candidate pool, so the cloud
reranker lives beside the harness instead of inside the service.

WHAT IT REPLACES, AND WHAT IT MUST NOT CHANGE
---------------------------------------------
The comparison is (MS MARCO local cross-encoder) vs (BGE-v2-m3 cloud API) over an IDENTICAL
pre-rerank candidate pool. So this module must return exactly what ``strategies.reranked()``
returns -- ``(documents, scores)``, documents first, highest score first -- and nothing about
the pool, the retrieval, or the metrics may differ. Any change in candidate identity would
make the two rerankers incomparable, which is the whole experiment.

WHY THE RESPONSE IS VALIDATED SO HARD
-------------------------------------
The provider returns ``results`` as ``[{index, relevance_score}, ...]``. ``index`` is a
position into the list of documents we SENT. If we mapped that index wrong -- or accepted a
response whose indices are duplicated, out of range, or short -- we would silently pair
documents with the wrong scores and report a reranker's ranking that no model ever produced.
That failure mode looks exactly like a legitimate result, so every invariant is checked and
a violation raises instead of degrading to a plausible number.

There is deliberately NO fallback to the local MS MARCO model on failure. A silent fallback
would make a broken cloud call indistinguishable from a cloud call that simply ranked
differently -- i.e. it would fabricate the experiment's conclusion. Errors propagate.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import requests
from langchain_core.documents import Document

# The app package lives two levels up (backend/). Resolved here rather than relying on the
# caller's cwd, because this module is imported both by run_eval (which inserts the path
# itself) and by the unit tests, which import it directly from the eval directory.
_BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from app.core.config import settings  # noqa: E402


# The provider's per-request ceiling on documents. The benchmark sends 10 (the frozen
# candidate pool), so this is a guard against a future pool-size change silently exceeding
# the API limit, not a value the current experiment relies on.
MAX_DOCUMENTS_PER_REQUEST = 100

# Retry policy: a SMALL, BOUNDED number of attempts for purely transient failures. An
# unbounded retry loop on a rate-limited endpoint turns one slow experiment into an
# indefinite hang, and a retry on a 400/401 is pointless -- the request is wrong, and
# repeating it just burns quota. Only connection errors, timeouts, and 429/5xx retry.
MAX_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 2.0


class SiliconFlowRerankError(RuntimeError):
    """Any failure of the cloud rerank call. Never swallowed."""


class SilentFallbackError(SiliconFlowRerankError):
    """Raised if something ever tries to route a cloud-rerank failure to the local model.

    Exists so the prohibition is enforced in code rather than only in a comment: the
    benchmark asserts this is never raised, and a future edit that adds a fallback would
    have to delete this class to do it.
    """


@dataclass
class RerankCallStats:
    """Per-call accounting, so the run can report real request/latency numbers.

    ``latency_ms`` is measured around the HTTP request ONLY (including any retries), which
    is the API cost the report claims. It excludes the local index-mapping work, so it is
    not inflated by our own CPU.
    """

    requests: int = 0
    failures: int = 0
    retries: int = 0
    latency_ms: float = 0.0
    per_call_latency_ms: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "requests": self.requests,
            "failures": self.failures,
            "retries": self.retries,
            "latency_ms_total": self.latency_ms,
        }


class SiliconFlowReranker:
    """Cloud reranker returning the same ``(documents, scores)`` shape as the local one."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: int | None = None,
        session: Any | None = None,
    ) -> None:
        self._api_key = api_key if api_key is not None else settings.siliconflow_api_key
        self._base_url = (base_url or settings.siliconflow_base_url).rstrip("/")
        self._model = model or settings.siliconflow_rerank_model
        self._timeout = timeout or settings.siliconflow_rerank_timeout
        # Injectable so tests can drive the adapter with a fake transport and no network.
        self._session = session if session is not None else requests
        self.stats = RerankCallStats()

        if not self._api_key:
            raise SiliconFlowRerankError(
                "SILICONFLOW_API_KEY is not set. The M4 experiment cannot run without it; "
                "refusing to proceed rather than falling back to the local reranker."
            )

    # -- public surface ----------------------------------------------------

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def endpoint(self) -> str:
        return f"{self._base_url}/rerank"

    def rerank_documents(
        self, query: str, documents: Sequence[Document], top_k: int | None = None
    ) -> tuple[list[Document], list[float]]:
        """Score ``documents`` against ``query`` and return them ranked, best first.

        The candidate pool is passed through untouched -- the same Document objects, with
        their metadata, only reordered. Callers rely on that: the benchmark compares this
        strategy's pool against the hybrid pool by identity.
        """
        if not documents:
            return [], []
        if len(documents) > MAX_DOCUMENTS_PER_REQUEST:
            raise SiliconFlowRerankError(
                f"{len(documents)} documents exceeds the provider limit of "
                f"{MAX_DOCUMENTS_PER_REQUEST} per request."
            )

        contents = [str(getattr(doc, "page_content", "")) for doc in documents]
        indices, scores, latency_ms = self._call_api(query, contents)

        self.stats.requests += 1
        self.stats.latency_ms += latency_ms
        self.stats.per_call_latency_ms.append(latency_ms)

        # Map provider indices back onto the ORIGINAL Document objects. The index is a
        # position into the list we sent, and winnow_documents schema validation already
        # proved every index is in range and unique, so this pairing is exact.
        ranked = sorted(
            zip(indices, scores), key=lambda pair: pair[1], reverse=True
        )
        out_docs = [documents[index] for index, _ in ranked]
        out_scores = [float(score) for _, score in ranked]

        if top_k is not None:
            out_docs = out_docs[:top_k]
            out_scores = out_scores[:top_k]

        return out_docs, out_scores

    # -- transport ---------------------------------------------------------

    def _call_api(
        self, query: str, contents: Sequence[str]
    ) -> tuple[list[int], list[float], float]:
        """POST the rerank request, retrying only transient failures.

        Returns ``(indices, scores, latency_ms)`` with raw (un-normalised) provider scores.

        Scores are NOT min-max normalised the way the local cross-encoder's are. Two reasons:
        the provider already returns comparable relevance scores, and min-max rescaling is a
        rank-preserving transform that cannot change the ordering this experiment measures --
        so normalising would add a step that erases the model's own calibration for no
        benefit to any reported metric.
        """
        payload = {
            "model": self._model,
            "query": query,
            "documents": list(contents),
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        started = time.perf_counter()
        last_error: Exception | None = None

        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                response = self._session.post(
                    self.endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self._timeout,
                )
            except Exception as exc:  # noqa: BLE001 -- connection/timeout surface varies
                last_error = exc
                if attempt < MAX_ATTEMPTS:
                    self.stats.retries += 1
                    time.sleep(RETRY_BACKOFF_SECONDS * attempt)
                    continue
                self.stats.failures += 1
                raise SiliconFlowRerankError(
                    f"SiliconFlow rerank request failed after {MAX_ATTEMPTS} attempts: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

            status = int(getattr(response, "status_code", 0))

            if status == 200:
                latency_ms = (time.perf_counter() - started) * 1000.0
                return self._parse_response(response, len(contents), latency_ms)

            # Retry ONLY what can plausibly succeed on a second try.
            retryable = status == 429 or 500 <= status < 600
            detail = self._error_detail(response)
            if retryable and attempt < MAX_ATTEMPTS:
                self.stats.retries += 1
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
                continue

            self.stats.failures += 1
            if status in (401, 403):
                raise SiliconFlowRerankError(
                    f"SiliconFlow authentication failed (HTTP {status}). The API key was "
                    f"rejected; check SILICONFLOW_API_KEY. Detail: {detail}"
                )
            if status == 429:
                raise SiliconFlowRerankError(
                    f"SiliconFlow rate limit or quota exhausted (HTTP 429) after "
                    f"{attempt} attempt(s). Stopping rather than switching to a paid model. "
                    f"Detail: {detail}"
                )
            raise SiliconFlowRerankError(
                f"SiliconFlow rerank returned HTTP {status}. Detail: {detail}"
            )

        # Unreachable: the loop either returns or raises. Kept so the type checker and a
        # future edit that adds a `break` cannot fall through to a None return.
        raise SiliconFlowRerankError(f"SiliconFlow rerank exhausted retries: {last_error}")

    @staticmethod
    def _error_detail(response: Any) -> str:
        """Best-effort error body, truncated. Never includes the API key (it is a header)."""
        try:
            text = response.text
        except Exception:  # noqa: BLE001
            return "<no body>"
        text = (text or "").strip().replace("\n", " ")
        return text[:300] if text else "<empty body>"

    def _parse_response(
        self, response: Any, sent_count: int, latency_ms: float
    ) -> tuple[list[int], list[float], float]:
        """Validate the response shape and return raw ``(indices, scores, latency_ms)``.

        Every check here exists because its failure would be invisible downstream: a wrong
        index pairs a document with another document's score, and a short response would
        quietly narrow the pool the strategy reports.
        """
        try:
            body = response.json()
        except Exception as exc:  # noqa: BLE001
            raise SiliconFlowRerankError(
                f"SiliconFlow rerank response was not valid JSON: {exc}. "
                f"Body: {self._error_detail(response)}"
            ) from exc

        if not isinstance(body, Mapping):
            raise SiliconFlowRerankError(
                f"SiliconFlow rerank response was not a JSON object: {type(body).__name__}"
            )

        results = body.get("results")
        if not isinstance(results, list):
            raise SiliconFlowRerankError(
                f"SiliconFlow rerank response has no 'results' list. Body keys: "
                f"{sorted(body.keys())}"
            )
        if len(results) != sent_count:
            # The provider scores every document it was sent; a different count means we
            # cannot map indices onto the pool we passed, so the ranking would be partial.
            raise SiliconFlowRerankError(
                f"SiliconFlow rerank returned {len(results)} results for {sent_count} "
                f"documents; expected one score per document."
            )

        indices: list[int] = []
        scores: list[float] = []
        for position, item in enumerate(results):
            if not isinstance(item, Mapping):
                raise SiliconFlowRerankError(
                    f"SiliconFlow rerank result #{position} is not an object: "
                    f"{type(item).__name__}"
                )
            index = item.get("index")
            score = item.get("relevance_score", item.get("score"))

            if isinstance(index, bool) or not isinstance(index, int):
                raise SiliconFlowRerankError(
                    f"SiliconFlow rerank result #{position} has a non-integer index: "
                    f"{index!r}"
                )
            if not 0 <= index < sent_count:
                raise SiliconFlowRerankError(
                    f"SiliconFlow rerank index {index} is outside the sent range "
                    f"[0, {sent_count}). The response cannot be mapped onto the candidate pool."
                )
            try:
                numeric = float(score)
            except (TypeError, ValueError) as exc:
                raise SiliconFlowRerankError(
                    f"SiliconFlow rerank result #{position} has a non-numeric score: {score!r}"
                ) from exc
            # NaN compares false against everything, so `numeric != numeric` is the portable
            # finite check; inf would sort to the top and silently win every comparison.
            if numeric != numeric or numeric in (float("inf"), float("-inf")):
                raise SiliconFlowRerankError(
                    f"SiliconFlow rerank result #{position} has a non-finite score: {numeric}"
                )

            indices.append(index)
            scores.append(numeric)

        # Uniqueness: a duplicated index would drop a candidate and duplicate another, so
        # the strategy would report a pool that is not the one it was given.
        if len(set(indices)) != len(indices):
            duplicates = sorted({i for i in indices if indices.count(i) > 1})
            raise SiliconFlowRerankError(
                f"SiliconFlow rerank returned duplicate indices {duplicates}; "
                f"each document must be scored exactly once."
            )

        return indices, scores, latency_ms


def assert_pool_unchanged(
    produced: Sequence[Document],
    expected_pool: Sequence[Document],
    produced_scores: Sequence[float],
) -> None:
    """Assert the reranker's output is a REORDERING of the pool it was given.

    The M4 fairness requirement is that the BGE strategy consumed the same Top-10 pool the
    hybrid strategy produced -- reranking reorders and truncates, it never invents, drops, or
    substitutes a candidate. This checks that as a set equality on the Documents themselves
    (identity, not just file name), so a pool that was accidentally rebuilt or truncated
    fails here rather than surfacing later as an oracle mismatch that reads like a reranker
    regression.

    Deliberately compares the FULL produced list against the pool: this is meant to be called
    on the pre-truncation output, not the Top-5 the report scores.
    """
    if len(produced) != len(expected_pool):
        raise SiliconFlowRerankError(
            f"Reranker returned {len(produced)} documents for a pool of "
            f"{len(expected_pool)}; the candidate pool must survive reranking intact."
        )
    if len(produced_scores) != len(produced):
        raise SiliconFlowRerankError(
            f"Reranker returned {len(produced)} documents but {len(produced_scores)} scores."
        )
    missing = [doc for doc in expected_pool if not any(doc is other for other in produced)]
    if missing:
        raise SiliconFlowRerankError(
            f"Reranker dropped {len(missing)} candidate(s) from the pool; the output must "
            f"be a permutation of its input."
        )


def load_api_key_from_env_file(path: Any = None) -> str | None:
    """Read SILICONFLOW_API_KEY from backend/.env, falling back to the process environment.

    Used by the smoke-test script so it can run a single real request WITHOUT booting the
    app. The key is returned to the caller and never logged; callers must not print it, and
    this function deliberately returns the value rather than registering it anywhere.
    """
    import os
    from pathlib import Path

    env_path = Path(path) if path is not None else (
        Path(__file__).resolve().parents[2] / ".env"
    )
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, _, value = line.partition("=")
            if name.strip() == "SILICONFLOW_API_KEY":
                value = value.strip().strip('"').strip("'")
                if value:
                    return value
    return os.environ.get("SILICONFLOW_API_KEY") or None


__all__ = [
    "SiliconFlowReranker",
    "SiliconFlowRerankError",
    "SilentFallbackError",
    "RerankCallStats",
    "assert_pool_unchanged",
    "load_api_key_from_env_file",
    "MAX_ATTEMPTS",
    "MAX_DOCUMENTS_PER_REQUEST",
]