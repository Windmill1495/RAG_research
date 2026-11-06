import contextlib
import io
import unittest
from unittest.mock import Mock

from evaluate import evaluate, retrieval_metrics


class EvaluationTests(unittest.TestCase):
    def test_generation_error_does_not_turn_successful_retrieval_into_a_miss(self):
        engine = Mock()
        engine.retrieve.return_value = {
            "hits": [{"document": "gold.md"}], "search_query": "q", "rewrite_source": "llm",
            "rewrite_error": None, "latencies_ms": {"retrieval": 1.0},
        }
        engine.answer_from_retrieval.side_effect = RuntimeError("generation timeout")
        examples = [{"id": "one", "split": "test", "question": "q", "history": [], "gold_document": "gold.md", "reference_answer": "a", "answer_keywords": ["a"]}]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            summaries, details = evaluate(engine, examples, ["rewrite"], 3, True)
        self.assertEqual(summaries[0]["recall_at_k"], 1.0)
        self.assertEqual(summaries[0]["error_count"], 1)
        self.assertEqual(details[0]["rewrite_source"], "llm")
        self.assertEqual(details[0]["error_stage"], "generation")

    def test_ranks_are_chunk_ranks_not_deduplicated_document_ranks(self):
        hits = [{"document": "noise.md"}, {"document": "noise.md"}, {"document": "gold.md"}]
        self.assertEqual(retrieval_metrics(hits, "gold.md", 3), (1.0, 1/3))
        self.assertEqual(retrieval_metrics(hits, "gold.md", 2), (0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
