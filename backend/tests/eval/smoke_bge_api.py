"""M4 Phase 3: ONE real SiliconFlow rerank request, on Chinese documents.

This is the only place in M4 that talks to the live provider before the formal 50-question
run. It sends a single request with 3 documents and verifies that the response maps onto
them correctly -- the same invariants the unit tests pin against a fake, now checked against
the real schema.

Deliberately ONE request, not a sweep: the model's Chinese behaviour was already established
locally, so repeating exploratory calls would spend quota to re-learn a known answer. The
formal run does the remaining 50.

The API key is read from backend/.env and is NEVER printed; only its presence is reported.

Run:
    cd backend && python tests/eval/smoke_bge_api.py
"""

from __future__ import annotations

import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
BACKEND_DIR = EVAL_DIR.parents[1]
sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(0, str(BACKEND_DIR))

from langchain_core.documents import Document  # noqa: E402

from bge_reranker import (  # noqa: E402
    SiliconFlowRerankError,
    SiliconFlowReranker,
    load_api_key_from_env_file,
)

ENV_PATH = BACKEND_DIR / ".env"

QUERY = "员工请病假需要提供什么证明材料？"

# The relevant document is deliberately NOT first: a reranker that returns its input
# unchanged would then be caught, whereas a correct model must promote doc 2 to rank 1.
DOCUMENTS = [
    Document(
        page_content=(
            "## 年假申请流程\n员工申请年假应当提前三个工作日在考勤系统提交申请，"
            "经直属主管审批通过后方可休假。年假可以跨年度累计，但不得超过两个年度。"
        ),
        metadata={"file_name": "leave_policy.md", "chunk_index": 0},
    ),
    Document(
        page_content=(
            "## 病假证明要求\n员工请病假连续超过一个工作日的，应当提供二级以上医院出具的"
            "诊断证明和病假建议书。未能提供有效证明的，按照事假处理。"
        ),
        metadata={"file_name": "attendance_policy.md", "chunk_index": 1},
    ),
    Document(
        page_content=(
            "## 出差住宿标准\n普通员工赴北京出差，住宿标准不超过每晚 550 元；"
            "赴上海出差，住宿标准不超过每晚 500 元。超标部分由员工自行承担。"
        ),
        metadata={"file_name": "travel_policy.md", "chunk_index": 0},
    ),
]

EXPECTED_TOP_FILE = "attendance_policy.md"


def main() -> int:
    print("=" * 72)
    print("M4 smoke test: live SiliconFlow BGE-reranker-v2-m3 (1 request)")
    print("=" * 72)

    api_key = load_api_key_from_env_file(ENV_PATH)
    if not api_key:
        print(f"[FAIL] SILICONFLOW_API_KEY not found in {ENV_PATH} or the environment.")
        return 2
    print(f"[i] key present: yes (len={len(api_key)}, not printed)")

    try:
        reranker = SiliconFlowReranker(api_key=api_key)
    except SiliconFlowRerankError as exc:
        print(f"[FAIL] could not construct reranker: {exc}")
        return 2

    print(f"[i] model    : {reranker.model_name}")
    print(f"[i] endpoint : {reranker.endpoint}")
    print(f"[i] query    : {QUERY}")
    print(f"[i] documents: {len(DOCUMENTS)} (relevant one is at index 1, not index 0)")
    print()

    try:
        docs, scores = reranker.rerank_documents(QUERY, DOCUMENTS, top_k=len(DOCUMENTS))
    except SiliconFlowRerankError as exc:
        print(f"[FAIL] request failed: {exc}")
        return 1

    print("[i] ranked output:")
    for rank, (doc, score) in enumerate(zip(docs, scores), start=1):
        name = doc.metadata.get("file_name")
        print(f"    {rank}. {name}  score={score:.6f}")

    print()
    print(f"[i] API latency  : {reranker.stats.per_call_latency_ms[-1]:.1f} ms")
    print(f"[i] requests     : {reranker.stats.requests}")
    print(f"[i] retries      : {reranker.stats.retries}")
    print(f"[i] failures     : {reranker.stats.failures}")

    problems = []
    if len(docs) != len(DOCUMENTS):
        problems.append(f"expected {len(DOCUMENTS)} results, got {len(docs)}")
    if len({id(d) for d in docs}) != len(docs):
        problems.append("output contains duplicate document objects")
    for doc in docs:
        if not any(doc is original for original in DOCUMENTS):
            problems.append("output contains a document that was not sent")
            break
    if scores != sorted(scores, reverse=True):
        problems.append("scores are not in descending order")
    if docs[0].metadata.get("file_name") != EXPECTED_TOP_FILE:
        problems.append(
            f"expected the病假 evidence chunk ranked first, got "
            f"{docs[0].metadata.get('file_name')}"
        )

    print()
    print("=" * 72)
    if problems:
        print("[FAIL] smoke test found problems:")
        for problem in problems:
            print(f"  - {problem}")
        print("=" * 72)
        return 1
    print(f"[PASS] index mapping, ranking and metadata are correct "
          f"({EXPECTED_TOP_FILE} ranked first, as expected)")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())