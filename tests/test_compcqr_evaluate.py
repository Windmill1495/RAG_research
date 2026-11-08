"""CompCQR evaluation must score the executed ranking without leaking labels."""
import contextlib
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from evaluate import evaluate
from evaluate_rewrite import evaluate_rewrites
from rag.retrieval import Chunk, Retriever


def sample(identifier="one"):
    return {"id": identifier, "split": "dev", "question": "When did it form?",
            "history": [{"role": "user", "content": "Who are Fleet Foxes?"}],
            "human_rewrite": "When did Fleet Foxes form?", "reference_answer": "gold answer secret",
            "positive_ids": ["gold"]}


def generated():
    return {"question": "When did it form?", "search_query": "When did Fleet Foxes form?",
            "search_queries": ["When did it form?", "When did Fleet Foxes form? hypothetical 2006"],
            "components": {"oq": "When did it form?", "mq": "When did Fleet Foxes form?"},
            "rewrite_source": "llm_compcqr", "rewrite_error": None,
            "rewrite_trace": {"query_count": 2, "llm_calls": 2},
            "latencies_ms": {"rewrite": 10.0, "total": 10.0}}


class CompCQREvaluationTests(unittest.TestCase):
    def engine(self, selection_error=None):
        engine = SimpleNamespace(rewrite_query=Mock(return_value=generated()), retriever=Mock())
        finished = {**generated(), "hits": [{"passage_id": "gold"}, {"passage_id": "noise"}],
                    "rewrite_trace": {"query_count": 2, "llm_calls": 3,
                                      "selection_error": selection_error,
                                      "selection_applied": selection_error is None},
                    "latencies_ms": {"rewrite": 10.0, "retrieval": 2.0, "fusion": 1.0,
                                     "selection": 8.0, "total": 21.0}}
        engine.retrieve_rewritten = Mock(return_value=finished)
        return engine

    def test_scores_final_fused_selection_and_reuses_components_without_gold(self):
        engine = self.engine()
        [summary], [row] = evaluate_rewrites(engine, [sample()], ["compcqr"],
                                             retrieval_ids={"one"}, progress=False)
        engine.rewrite_query.assert_called_once_with(question=sample()["question"],
                                                     history=sample()["history"], strategy="compcqr")
        engine.retrieve_rewritten.assert_called_once()
        plan = engine.retrieve_rewritten.call_args.args[0]
        self.assertNotIn("human_rewrite", plan)
        self.assertNotIn("positive_ids", plan)
        self.assertNotIn("reference_answer", plan)
        engine.retriever.search.assert_not_called()
        self.assertEqual(row["hits"][0]["passage_id"], "gold")
        self.assertEqual(summary["gold_recall_at_1"], 1.0)
        self.assertEqual(summary["ndcg_at_3"], 1.0)
        self.assertEqual(summary["rouge1_f1"], 1.0)
        self.assertEqual(summary["rouge_scope"], "base_rewrite_only")
        self.assertEqual(row["search_queries"], generated()["search_queries"])
        self.assertEqual(row["execution_scope"], "full_compcqr")
        self.assertEqual(summary["llm_calls"], 3)
        self.assertEqual(summary["mean_query_total_ms"], 21.0)
        self.assertEqual(summary["mean_latencies_ms"]["selection"], 8.0)

    def test_rewrite_only_marks_components_only_and_does_not_retrieve(self):
        engine = self.engine()
        [summary], [row] = evaluate_rewrites(engine, [sample()], ["compcqr"], progress=False)
        engine.retrieve_rewritten.assert_not_called()
        engine.retriever.search.assert_not_called()
        self.assertEqual(row["execution_scope"], "component_generation_only")
        self.assertEqual(summary["execution_scopes"], {"component_generation_only": 1})
        self.assertNotIn("gold_recall_at_3", summary)
        self.assertEqual(summary["llm_calls"], 2)

    def test_selection_error_preserves_ranking_and_is_counted(self):
        engine = self.engine("selector timeout; kept RRF")
        [summary], [row] = evaluate_rewrites(engine, [sample()], ["compcqr"],
                                             retrieval_ids={"one"}, progress=False)
        self.assertEqual(summary["mrr_at_3"], 1.0)
        self.assertEqual(summary["method_error_count"], 1)
        self.assertEqual(summary["selection_error_count"], 1)
        self.assertEqual(summary["pure_llm_success_subset_n"], 0)
        self.assertEqual(row["execution_scope"], "compcqr_degraded")
        self.assertIsNone(row["retrieval_error"])

    def test_failed_multisearch_keeps_fixed_retrieval_denominator(self):
        engine = self.engine()
        engine.retrieve_rewritten.side_effect = [engine.retrieve_rewritten.return_value,
                                                 RuntimeError("search unavailable")]
        [summary], rows = evaluate_rewrites(engine, [sample(), sample("two")], ["compcqr"],
                                             retrieval_ids={"one", "two"}, progress=False)
        self.assertEqual(summary["retrieval_sample_count"], 2)
        self.assertEqual(summary["retrieval_error_count"], 1)
        self.assertEqual(summary["gold_recall_at_3"], .5)
        self.assertEqual(summary["rouge1_f1"], 1.0)
        self.assertIn("search unavailable", rows[1]["retrieval_error"])
        self.assertEqual(rows[1]["execution_scope"], "compcqr_degraded")

    def test_rrf_only_ablation_and_demo_are_not_labelled_full_compcqr(self):
        for source, trace, expected in [
            ("compcqr_llm", {"selection_applied": False, "selection_skipped": "disabled_by_config"},
             "compcqr_rrf_only_ablation"),
            ("compcqr_demo", {"demo_note": "heuristic only", "selection_applied": False}, "compcqr_demo"),
        ]:
            with self.subTest(expected=expected):
                engine = self.engine()
                engine.rewrite_query.return_value["rewrite_source"] = source
                engine.retrieve_rewritten.return_value["rewrite_trace"].update(trace)
                [summary], [row] = evaluate_rewrites(engine, [sample()], ["compcqr"],
                                                     retrieval_ids={"one"}, progress=False)
                self.assertEqual(row["execution_scope"], expected)
                self.assertEqual(summary["execution_scopes"], {expected: 1})

    def test_retrieval_entrypoint_preserves_stage_trace_and_final_metrics(self):
        engine = Mock()
        engine.retrieve.return_value = self.engine("invalid passage choice").retrieve_rewritten.return_value
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            [summary], [row] = evaluate(engine, [sample()], ["compcqr"], 3, False)
        self.assertEqual(summary["gold_recall_at_1"], 1.0)
        self.assertEqual(summary["ndcg_at_3"], 1.0)
        self.assertEqual(summary["selection_error_count"], 1)
        self.assertEqual(row["search_queries"], generated()["search_queries"])
        self.assertEqual(summary["mean_latencies_ms"]["fusion"], 1.0)

    def test_hybrid_candidate_pool_is_same_for_baseline_and_compcqr_depth(self):
        # Disjoint sparse/dense top-12 lists expose a pool that incorrectly
        # changes from 12 to 400 when requested output K changes from 3 to 100.
        retriever = Retriever(Path("."), Path("unused-index.json"), backend="hybrid")
        retriever._built = True
        retriever.chunks = [Chunk(f"p{i:03d}", "doc", "title", "text", 0, 4)
                            for i in range(30)]
        retriever._bm25_candidates = list(range(30))
        retriever._bm25_scores = Mock(return_value=[30 - i for i in range(30)])
        dense_order = [*range(29, 17, -1), *range(18)]
        retriever.embeddings = [[30 - dense_order.index(i), 0.0] for i in range(30)]
        retriever._embed = Mock(return_value=[[1.0, 0.0]])
        scalar = retriever.search("query", 3)
        deep = retriever.search("query", 100)
        batch = retriever.search_many(["query"], 100)[0]
        self.assertEqual([row["id"] for row in scalar], [row["id"] for row in deep[:3]])
        self.assertEqual([row["id"] for row in scalar], [row["id"] for row in batch[:3]])


if __name__ == "__main__":
    unittest.main()
