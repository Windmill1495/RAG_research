"""Regression checks for rewrite inputs and optional passage scoring."""
from collections import Counter
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, call

from evaluate_rewrite import evaluate_rewrites, retrieval_coverage
from rag.data import public_input
from rag.rewrite_data import load_rewrite_examples


ROOT = Path(__file__).resolve().parents[1]


class RewriteDataTests(unittest.TestCase):
    def test_project_questions_include_both_splits(self):
        path = ROOT / "data/qrecc/questions.jsonl"
        rows = load_rewrite_examples(path, "all")
        self.assertEqual(Counter(row["split"] for row in rows), {"dev": 102, "test": 205})
        self.assertEqual(len({row["id"] for row in rows}), 307)
        self.assertEqual(load_rewrite_examples(path, "all", limit=1), rows[:1])

    def test_official_data_uses_only_prior_gold_and_accepts_missing_passages_or_answers(self):
        rows = [
            {"Conversation_no": 1, "Turn_no": 2, "Question": "His profession?", "Rewrite": "What is Alex's profession?",
             "Answer": "chemist", "Context": ["Who is Alex?"], "Passages": ["p1", "p1", "p2"]},
            {"Conversation_no": 1, "Turn_no": 1, "Question": "Who is Alex?", "Rewrite": "Who is Alex?", "Context": []},
            {"Conversation_no": 1, "Turn_no": 3, "Question": "His school?", "Rewrite": "Where did Alex study?",
             "Context": ["Who is Alex?", "What is Alex's profession?", "chemist"]},
        ]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "qrecc-test.json"
            path.write_text(json.dumps(rows), encoding="utf-8")
            examples = load_rewrite_examples(path, "all")
            with self.assertRaisesRegex(ValueError, "official test file"):
                load_rewrite_examples(path, "dev")
        self.assertEqual([example["id"] for example in examples], ["1-1", "1-2", "1-3"])
        self.assertEqual({example["split"] for example in examples}, {"test"})
        self.assertEqual(examples[0]["history"], [])
        self.assertEqual(public_input(examples[1]), {
            "question": "His profession?",
            "history": [{"role": "user", "content": "Who is Alex?", "turn_id": 1}],
        })
        self.assertEqual([message["role"] for message in examples[2]["history"]], ["user", "user", "assistant"])
        self.assertEqual(examples[2]["human_rewrite"], "Where did Alex study?")
        self.assertEqual(examples[1]["positive_ids"], ["p1", "p2"])
        self.assertNotIn("positive_ids", examples[2])
        self.assertNotIn("reference_answer", examples[1])

    def test_partial_corpus_requires_opt_in_and_reports_coverage(self):
        with TemporaryDirectory() as directory:
            corpus = Path(directory) / "passages.jsonl"
            corpus.write_text('{"id":"p1","text":"indexed"}\n', encoding="utf-8")
            config = {"corpus_format": "passages_jsonl", "corpus_path": str(corpus), "retriever_backend": "hybrid"}
            examples = [{"id": "a", "positive_ids": ["p1", "p2"]},
                        {"id": "b", "positive_ids": ["p3"]}, {"id": "c"}]
            with self.assertRaisesRegex(ValueError, "--allow-partial-corpus"):
                retrieval_coverage(config, examples)
            eligible, coverage = retrieval_coverage(config, examples, allow_partial_corpus=True)
        self.assertEqual(eligible, {"a"})
        self.assertEqual(coverage["samples_without_indexed_gold"], 1)
        self.assertEqual(coverage["samples_with_missing_gold"], 2)
        self.assertEqual(coverage["unlabeled_sample_count"], 1)

    def test_retrieval_uses_final_rewrite_and_keeps_separate_denominator(self):
        engine = Mock()
        examples = [{"id": key, "split": "test", "question": "What?", "history": [],
                     "human_rewrite": "What is it?", **labels}
                    for key, labels in [("ok", {"positive_ids": ["p1", "p2"]}),
                                        ("unlabeled", {}), ("failed_search", {"positive_ids": ["p1"]}),
                                        ("failed_rewrite", {"positive_ids": ["p2"]})]]
        output = {"search_query": "What is it?", "rewrite_source": "llm", "latencies_ms": {"rewrite": 10.0}}
        engine.rewrite_query.side_effect = [output, output, output, RuntimeError("rewrite unavailable")]
        retriever = Mock()
        retriever.search.side_effect = [[{"passage_id": "p1"}, {"passage_id": "p2"}], RuntimeError("embed unavailable")]
        [summary], details = evaluate_rewrites(engine, examples, ["rewrite"], progress=False,
                                               retrieval_ids={"ok", "failed_search", "failed_rewrite"}, retriever=retriever)
        self.assertEqual(engine.rewrite_query.call_count, 4)
        self.assertEqual(retriever.search.call_args_list, [call("What is it?", 3), call("What is it?", 3)])
        self.assertEqual(summary["sample_count"], 4)
        self.assertEqual(summary["retrieval_sample_count"], 3)
        self.assertEqual(summary["retrieval_error_count"], 1)
        self.assertAlmostEqual(summary["gold_recall_at_1"], 1 / 6)
        self.assertAlmostEqual(summary["gold_recall_at_3"], 1 / 3)
        self.assertAlmostEqual(summary["mrr_at_3"], 1 / 3)
        self.assertEqual(summary["rouge1_f1"], .75)
        self.assertIsNone(details[1]["gold_recall_at_3"])
        self.assertIn("embed unavailable", details[2]["retrieval_error"])

    def test_official_context_is_checked_even_after_limit(self):
        rows = [
            {"Conversation_no": 1, "Turn_no": 1, "Question": "Who is Alex?", "Rewrite": "Who is Alex?", "Context": []},
            {"Conversation_no": 1, "Turn_no": 2, "Question": "His profession?", "Rewrite": "What is Alex's profession?",
             "Context": ["wrong history"]},
        ]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "renamed.json"
            path.write_text(json.dumps(rows), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "renamed QReCC"):
                load_rewrite_examples(path, "all")
            with self.assertRaisesRegex(ValueError, "no alternating-role guess"):
                load_rewrite_examples(path, "test", limit=1)
