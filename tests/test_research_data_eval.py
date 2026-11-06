"""Research checks: role provenance, no label input, ranking and failure denominators."""
import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from evaluate import evaluate
from evaluate_rewrite import evaluate_rewrites, export_blind_review
from prepare_qrecc import prepare_dataset
from rag.data import load_examples, public_input
from rag.metrics import passage_metrics, rouge_scores
from rag.run_cache import RunCache


def qrecc_sample(identifier="1-3", split="dev"):
    return {"id": identifier, "conversation_id": 1, "turn": 3, "split": split,
            "question": "When did he retire?", "history": [{"role": "user", "content": "Who was Don Shula?", "turn_id": 1}],
            "human_rewrite": "When did Don Shula retire?", "reference_answer": "in 1996", "positive_ids": ["p1", "p2"]}


class ResearchDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def write_rows(self, path, rows):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return path

    def test_flat_history_is_rejected_instead_of_guessing_roles(self):
        row = qrecc_sample()
        row["history"] = ["an answer", "a question", "another answer"]
        path = self.write_rows(self.root / "q.jsonl", [row])
        with self.assertRaisesRegex(ValueError, "explicit"):
            load_examples(path)

    def test_split_leakage_is_checked_before_debug_limit(self):
        rows = [qrecc_sample(), {**qrecc_sample("1-4", "test"), "turn": 4}]
        with self.assertRaisesRegex(ValueError, "crosses"):
            load_examples(self.write_rows(self.root / "q.jsonl", rows), "dev", 1)

    def test_positive_ids_are_validated_against_fixed_corpus(self):
        with self.assertRaisesRegex(ValueError, "absent"):
            load_examples(self.write_rows(self.root / "q.jsonl", [qrecc_sample()]), corpus_ids={"p1"})

    def test_future_history_is_rejected(self):
        row = qrecc_sample()
        row["history"][0]["turn_id"] = 3
        with self.assertRaisesRegex(ValueError, "future"):
            load_examples(self.write_rows(self.root / "q.jsonl", [row]))

    def test_public_input_contains_no_reference_or_positive(self):
        self.assertEqual(set(public_input(qrecc_sample())), {"question", "history"})

    def official_fixture(self):
        return [
            {"Conversation_no": 1, "Turn_no": 1, "Question": "Who?", "Rewrite": "Who is Alex?", "Answer": "", "Context": [], "Passages": []},
            {"Conversation_no": 1, "Turn_no": 2, "Question": "His sport?", "Rewrite": "What sport does Alex play?", "Answer": "tennis",
             "Context": ["Who is Alex?"], "Passages": []},
            {"Conversation_no": 1, "Turn_no": 3, "Question": "When did he retire?", "Rewrite": "When did Alex retire?", "Answer": "1996",
             "Context": ["Who is Alex?", "What sport does Alex play?", "tennis"], "Passages": ["p1", "outside"]},
        ]

    def test_preparation_handles_missing_answers_and_filters_only_empty_passages(self):
        passages = self.write_rows(self.root / "source.jsonl", [{"id": "p1", "text": "Alex retired in 1996."}, {"id": "empty", "text": " "}])
        raw = self.root / "official.json"
        raw.write_text(json.dumps(self.official_fixture()), encoding="utf-8")
        selected = self.write_rows(self.root / "selected.jsonl", [{"id": "1-3", "split": "dev"}])
        manifest = prepare_dataset(passages, raw, self.root / "prepared", selected)
        samples = load_examples(self.root / "prepared/questions.jsonl")
        self.assertEqual([m["role"] for m in samples[0]["history"]], ["user", "user", "assistant"])
        self.assertEqual([m["content"] for m in samples[0]["history"]], self.official_fixture()[2]["Context"])
        self.assertNotIn("1996", [m["content"] for m in samples[0]["history"]])
        self.assertEqual(samples[0]["positive_ids"], ["p1"])
        self.assertEqual(manifest["source_passage_count"], 2)
        self.assertEqual(manifest["filtered_empty_ids"], ["empty"])
        self.assertEqual(manifest["n_passages"], 1)
        self.assertEqual(len(passages.read_text().splitlines()), 2, "original corpus must stay untouched")
        with self.assertRaises(FileExistsError):
            prepare_dataset(passages, raw, self.root / "prepared", selected)

    def test_official_context_mismatch_is_not_silently_repaired(self):
        passages = self.write_rows(self.root / "p.jsonl", [{"id": "p1", "text": "evidence"}])
        raw = self.root / "raw.json"
        rows = self.official_fixture()
        rows[2]["Context"] = ["wrong history"]
        raw.write_text(json.dumps(rows), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "no alternating-role guess"):
            prepare_dataset(passages, raw, self.root / "out")


class ResearchMetricTests(unittest.TestCase):
    def test_any_hit_and_gold_set_recall_differ(self):
        scores = passage_metrics([{"passage_id": "p1"}], ["p1", "p2", "p3"], 1)
        self.assertEqual(scores["hit_at_k"], 1)
        self.assertAlmostEqual(scores["gold_recall_at_k"], 1 / 3)
        self.assertEqual(scores["ndcg_at_k"], 1)

    def test_duplicates_do_not_inflate_recall_or_reorder_ranks(self):
        hits = [{"passage_id": value} for value in ["noise", "noise", "p1", "p1", "p2"]]
        scores = passage_metrics(hits, ["p1", "p2"], 5)
        self.assertEqual(scores["gold_recall_at_k"], 1)
        self.assertAlmostEqual(scores["mrr_at_k"], 1 / 3)
        self.assertAlmostEqual(scores["ndcg_at_k"], (1 / math.log2(4) + 1 / math.log2(6)) / (1 + 1 / math.log2(3)))

    def test_original_passage_id_is_required(self):
        with self.assertRaisesRegex(ValueError, "passage_id"):
            passage_metrics([{"id": "generated_chunk_hash"}], ["gold"], 1)

    def test_rouge_uses_multiset_overlap_and_lcs(self):
        scores = rouge_scores("cat cat dog", "cat dog", include_l=True)
        self.assertAlmostEqual(scores["rouge1_precision"], 2 / 3)
        self.assertEqual(scores["rouge1_recall"], 1)
        self.assertAlmostEqual(scores["rouge1_f1"], .8)
        self.assertAlmostEqual(scores["rougeL_f1"], .8)
        self.assertEqual(rouge_scores("", "gold")["rouge1_f1"], 0)


class ResearchEvaluatorTests(unittest.TestCase):
    def quiet(self):
        stack = contextlib.ExitStack()
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
        return stack

    def test_qrecc_generation_timeout_preserves_passage_metrics_and_labels_never_enter_api(self):
        engine = Mock()
        engine.retrieve.return_value = {"hits": [{"passage_id": "p1"}], "search_query": "query", "rewrite_source": "llm", "latencies_ms": {"retrieval": 1}}
        engine.answer_from_retrieval.side_effect = RuntimeError("generation timeout")
        with self.quiet():
            summaries, details = evaluate(engine, [qrecc_sample()], ["rewrite"], 5, True)
        self.assertEqual(summaries[0]["hit_at_k"], 1)
        self.assertEqual(summaries[0]["gold_recall_at_k"], .5)
        self.assertEqual(summaries[0]["generation_error_count"], 1)
        self.assertEqual(details[0]["error_stage"], "generation")
        self.assertEqual(set(engine.retrieve.call_args.kwargs), {"question", "history", "strategy", "top_k"})

    def test_retrieval_failure_remains_in_fixed_denominator(self):
        engine = Mock()
        engine.retrieve.side_effect = [{"hits": [{"passage_id": "p1"}], "search_query": "q"}, RuntimeError("failed")]
        examples = [qrecc_sample(), qrecc_sample("1-4")]
        with self.quiet():
            summary, _ = evaluate(engine, examples, ["current"], 3, False)
        self.assertEqual(summary[0]["sample_count"], 2)
        self.assertEqual(summary[0]["hit_at_k"], .5)
        self.assertEqual(summary[0]["retrieval_error_count"], 1)

    def test_generation_warnings_and_actual_evidence_are_preserved_for_review(self):
        engine = Mock()
        visible = [{"role": "user", "content": "Who was Don Shula?"}]
        engine.retrieve.return_value = {
            "hits": [{"passage_id": "p1"}], "search_query": "Don Shula retirement",
            "rewrite_source": "llm", "visible_history": visible,
            "latencies_ms": {"retrieval": 1}, "mode": "ollama", "dataset": "qrecc", "backend": "bm25",
        }
        sources = [{"passage_id": "p1", "label": "S1", "context_text": "Don Shula retired in 1996."}]
        engine.answer_from_retrieval.return_value = {
            "answer": "Don Shula retired in 1996.", "context": "[S1] Don Shula retired in 1996.",
            "sources": sources, "warnings": ["Model omitted citation labels; verify evidence manually."],
            "mode": "ollama", "model": "qwen2.5:3b", "backend": "bm25", "dataset": "qrecc",
            "latencies_ms": {"retrieval": 1, "generation": 10},
        }
        with self.quiet():
            summary, details = evaluate(engine, [qrecc_sample()], ["rewrite"], 5, True)
        row = details[0]
        self.assertEqual(summary[0]["error_count"], 0)
        self.assertIsNone(row["error"])
        self.assertEqual(row["warnings"], engine.answer_from_retrieval.return_value["warnings"])
        self.assertEqual(row["sources"], sources)
        self.assertIn("[S1]", row["context"])
        self.assertEqual(row["model"], "qwen2.5:3b")
        self.assertEqual(row["visible_history"], visible)
        self.assertEqual(row["hit_at_k"], 1)

    def test_signed_resume_does_not_repeat_already_completed_calls(self):
        with tempfile.TemporaryDirectory() as folder:
            engine = Mock()
            engine.retrieve.return_value = {"hits": [{"passage_id": "p1"}], "search_query": "q"}
            cache = RunCache(Path(folder) / "run", {"test": 1})
            with self.quiet():
                evaluate(engine, [qrecc_sample()], ["current"], 3, False, cache=cache)
                evaluate(engine, [qrecc_sample()], ["current"], 3, False,
                         cache=RunCache(Path(folder) / "run", {"test": 1}, resume=True))
            self.assertEqual(engine.retrieve.call_count, 1)

    def test_rewrite_failures_score_zero_and_fallbacks_are_separate(self):
        engine = Mock()
        engine.rewrite_query.side_effect = [
            {"search_query": "When did Don Shula retire?", "rewrite_source": "llm", "latencies_ms": {"rewrite": 10}},
            RuntimeError("offline"),
        ]
        with self.quiet():
            summary, details = evaluate_rewrites(engine, [qrecc_sample(), qrecc_sample("1-4")], ["rewrite"])
        self.assertEqual(summary[0]["sample_count"], 2)
        self.assertEqual(summary[0]["rouge1_f1"], .5)
        self.assertEqual(summary[0]["error_count"], 1)
        self.assertEqual(summary[0]["pure_llm_success_subset_n"], 1)
        self.assertEqual(details[1]["search_query"], "")
        self.assertEqual(set(engine.rewrite_query.call_args.kwargs), {"question", "history", "strategy"})

    def test_blind_review_uses_identical_samples_for_all_methods(self):
        engine = Mock()
        engine.rewrite_query.return_value = {"search_query": "When did Don Shula retire?", "rewrite_source": "heuristic_demo"}
        with self.quiet():
            _, details = evaluate_rewrites(engine, [qrecc_sample(), qrecc_sample("1-4")], ["rewrite", "structured"])
        with tempfile.TemporaryDirectory() as folder:
            export_blind_review(folder, details, count=1, seed=42)
            key = json.loads((Path(folder) / "blind_review_key.json").read_text(encoding="utf-8"))
            self.assertEqual(len(key["sample_ids"]), 1)
            self.assertEqual(len(key["mapping"]), 2)
            self.assertEqual({r["strategy"] for r in key["mapping"].values()}, {"rewrite", "structured"})
            csv_text = (Path(folder) / "blind_review_reviewer1.csv").read_text(encoding="utf-8-sig")
            self.assertNotIn("strategy", csv_text.splitlines()[0])
            self.assertNotIn("heuristic_demo", csv_text)


if __name__ == "__main__":
    unittest.main()
