"""Mock M4's full entry point: no network, model loading, or result-file writes.

Run from backend/: python tests/eval/test_bge_execution_order.py
"""

from __future__ import annotations

import copy
import io
import sys
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_eval_bge as m4
from bge_reranker import SiliconFlowReranker
from langchain_core.documents import Document


class ExecutionOrderTests(unittest.TestCase):
    def setUp(self):
        self.questions = [
            {
                "id": f"q{i:03d}", "question": f"question {i}",
                "domain": "hr", "category": "policy_lookup", "difficulty": "easy",
                "relevant_files": ["policy.md"], "relevant_anchors": ["evidence"],
            }
            for i in range(1, 51)
        ]
        self.pools = {
            q["question"]: [
                Document(
                    page_content=f"{q['id']} evidence chunk {i}",
                    metadata={"file_name": "policy.md", "chunk_index": i},
                )
                for i in range(10)
            ]
            for q in self.questions
        }
        self.scores = [1.0 - i * 0.01 for i in range(10)]
        records = [
            m4.base._question_result(q, self.pools[q["question"]], self.scores, {}, {})
            for q in self.questions
        ]
        aggregate = m4._aggregate(records)
        facts = {
            "collection": m4.COLLECTION, "collection_points": 78,
            "collection_vector_size": 384, "embedding_model": m4.EMBEDDING_MODEL,
            "embedding_dimension": 384, "candidate_pool_k": 10,
            "evaluation_k_max": 5, "oracle_k": 10,
            "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "reranker_backend": "sentence-transformers.CrossEncoder",
        }
        self.m3 = {
            "dir": "tests/eval/results/multilingual_embed_v1",
            "hybrid": {"questions": copy.deepcopy(records)},
            "summary": {
                "experiment": facts,
                "overall": {"hybrid": aggregate, "hybrid_rerank": dict(aggregate)},
            },
        }
        self.gate_passed = False
        self.vector = Mock(side_effect=self.retrieve)
        self.hybrid = Mock(side_effect=self.retrieve)
        self.transport = Mock()
        self.transport.post.side_effect = self.respond
        self.reranker = SiliconFlowReranker(
            api_key="mock-key", session=self.transport,
        )
        self.rerank = Mock(wraps=self.reranker.rerank_documents)
        self.reranker.rerank_documents = self.rerank
        self.writer = Mock()

    def retrieve(self, service, question):
        self.assertFalse(self.gate_passed, "Phase B must not retrieve again")
        self.assertEqual(self.transport.post.call_count, 0)
        return self.pools[question], self.scores

    def respond(self, *args, **kwargs):
        self.assertTrue(self.gate_passed)
        self.assertEqual(self.vector.call_count, 51)  # 50 questions plus warm-up
        self.assertEqual(self.hybrid.call_count, 51)
        body = kwargs["json"]
        self.assertEqual(
            body["documents"],
            [doc.page_content for doc in self.pools[body["query"]]],
        )
        return SimpleNamespace(
            status_code=200,
            json=lambda: {"results": [
                {"index": i, "relevance_score": score}
                for i, score in enumerate(self.scores)
            ]},
        )

    def run_mocked_main(self, fail_constants=False, fail_binding=False):
        settings = SimpleNamespace(
            qdrant_collection=m4.COLLECTION, embedding_model=m4.EMBEDDING_MODEL,
            qdrant_url="https://mock.invalid", qdrant_api_key="mock-key",
        )
        advanced = SimpleNamespace(
            rerank=Mock(side_effect=AssertionError("MS MARCO must not run")),
            _get_cross_encoder=Mock(side_effect=AssertionError("MS MARCO must not load")),
        )
        module_attributes = {
            "qdrant_client": {"QdrantClient": Mock()},
            "app.core.config": {"settings": settings},
            "app.services.advanced_retrieval": {"advanced_retrieval": advanced},
            "app.services.embeddings": {
                "embedding_service": SimpleNamespace(get_embeddings=Mock(return_value=object())),
            },
            "app.services.retrieval": {"retrieval_service": object()},
            "app.services.vector_store": {
                "vector_store": SimpleNamespace(get_all_documents=Mock(return_value=[])),
            },
        }
        modules = {}
        for name, attributes in module_attributes.items():
            module = ModuleType(name)
            module.__dict__.update(attributes)
            modules[name] = module
        hashes = {
            "frozen_integrity_authority": "canonical",
            "canonical": {"actual": {}, "match": {"all": True}},
            "legacy": {"actual": {}, "match": {"all": True}},
        }
        verify = m4._verify_m3_retrieval

        def gate(results, m3):
            self.assertEqual(self.transport.post.call_count, 0)
            self.assertEqual(self.vector.call_count, 51)
            self.assertEqual(self.hybrid.call_count, 51)
            result = verify(results, m3)
            self.gate_passed = True
            return result

        labels = {
            "recall@1": "Recall@1", "recall@5": "Recall@5", "mrr": "MRR",
            "evidence_recall@5": "EvidenceRecall@5",
            "file_oracle@10": "FileOracle@10", "evidence_oracle@10": "EvidenceOracle@10",
        }
        constants = {}
        for key in m4.M3_EXPECTED:
            strategy = "hybrid_rerank" if key.startswith("hybrid_rerank_") else "hybrid"
            metric = key[len(strategy) + 1:]
            constants[key] = self.m3["summary"]["overall"][strategy][labels[metric]]
        if fail_constants:
            constants["hybrid_recall@1"] = 0.5
        if fail_binding:
            self.m3["summary"]["experiment"]["embedding_model"] = "wrong-model"
        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, modules))
            for target, value in (
                ("load_frozen_dataset", Mock(return_value=({"questions": self.questions}, hashes))),
                ("_load_m3_baseline", Mock(return_value=self.m3)),
                ("M3_EXPECTED", constants),
                ("_verify_m3_retrieval", gate),
                ("SiliconFlowReranker", Mock(return_value=self.reranker)),
                ("_write_json", self.writer),
            ):
                stack.enter_context(patch.object(m4, target, value))
            for target, value in (
                ("vector_only", self.vector), ("hybrid", self.hybrid),
                ("_raw_points", Mock(side_effect=lambda client, name: 1 if name == m4.PRODUCTION_COLLECTION else 78)),
                ("_collection_vector_size", Mock(return_value=384)),
                ("_runtime_embedding_dimension", Mock(return_value=384)),
                ("warm_up", Mock(side_effect=AssertionError("Old reranker warm-up must not run"))),
            ):
                stack.enter_context(patch.object(m4.base, target, value))
            stack.enter_context(redirect_stdout(io.StringIO()))
            return m4.main()

    def assert_blocked_without_api(self, **kwargs):
        with self.assertRaises(RuntimeError):
            self.run_mocked_main(**kwargs)
        self.assertEqual(self.transport.post.call_count, 0)
        self.rerank.assert_not_called()
        self.writer.assert_called_once()
        path, report = self.writer.call_args.args
        self.assertEqual(path, m4.RESULTS_DIR / "BLOCKED.json")
        self.assertEqual(report["status"], "BLOCKED")
        if not kwargs.get("fail_constants") and not kwargs.get("fail_binding"):
            self.assertEqual(self.vector.call_count, 51)
            self.assertEqual(self.hybrid.call_count, 51)

    def test_pool_order_failure_blocks_before_api(self):
        self.m3["hybrid"]["questions"][-1]["pool"].reverse()
        self.assert_blocked_without_api()

    def test_file_oracle_failure_blocks_before_api(self):
        self.m3["hybrid"]["questions"][-1]["metrics"]["file_oracle@10"] = 0.0
        self.assert_blocked_without_api()

    def test_evidence_oracle_failure_blocks_before_api(self):
        self.m3["hybrid"]["questions"][-1]["metrics"]["evidence_oracle@10"] = 0.0
        self.assert_blocked_without_api()

    def test_aggregate_failure_blocks_before_api(self):
        self.m3["summary"]["overall"]["hybrid"]["MRR"] = 0.5
        self.assert_blocked_without_api()

    def test_saved_constants_failure_blocks_before_api(self):
        self.assert_blocked_without_api(fail_constants=True)

    def test_binding_failure_blocks_before_api(self):
        self.assert_blocked_without_api(fail_binding=True)

    def test_pass_reuses_original_pools_after_all_retrieval(self):
        self.assertEqual(self.run_mocked_main(), 0)
        self.assertEqual(self.transport.post.call_count, 50)
        self.assertEqual(self.rerank.call_count, 50)
        for call, question in zip(self.rerank.call_args_list, self.questions):
            text, documents = call.args
            self.assertEqual(text, question["question"])
            self.assertIs(documents, self.pools[text])
            self.assertEqual(len(documents), 10)
            self.assertEqual([doc.metadata["chunk_index"] for doc in documents], list(range(10)))
            self.assertEqual(call.kwargs["top_k"], 5)
        self.assertFalse(any(call.args[0].name == "BLOCKED.json" for call in self.writer.call_args_list))


if __name__ == "__main__":
    unittest.main()
