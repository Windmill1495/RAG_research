"""Research integration: profiles, public IDs, fair inputs and answer prompts.

All model calls are explicit mocks. These checks establish software contracts,
not Qwen rewrite quality or retrieval benchmark performance.
"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from rag.config import load_config
from rag.pipeline import RAGEngine, clean_history
from rag.retrieval import Retriever
from rag.rewriters import STRATEGIES


class ResearchPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "configs").mkdir()
        (self.root / "manuals").mkdir()
        (self.root / "data" / "qrecc").mkdir(parents=True)
        self.passage_id = "https://example.invalid/wiki/Spacecraft_p0"
        self.long_text = "The spacecraft stores solar energy and supports orbital research. " * 20
        self.passages = [
            {"id": self.passage_id, "text": self.long_text},
            {"id": "https://example.invalid/wiki/Library_p0", "text": "The library lends books."},
        ]
        self.write_passages()
        (self.root / "manuals" / "scanner.md").write_text(
            "# Portable scanner\nThe portable scanner works for two hours per charge.", encoding="utf-8")
        (self.root / "config.json").write_text(json.dumps({
            "model": "qwen2.5:3b", "dataset": "qrecc", "chunk_size": 80,
            "overlap": 10, "history_turns": 2,
        }), encoding="utf-8")
        self.markdown_overrides = {
            "corpus_format": "markdown", "corpus_path": "manuals",
            "index_path": ".cache/manuals/index.json",
        }
        self.qrecc_profile = {
            "dataset": "qrecc", "corpus_format": "passages_jsonl",
            "corpus_path": "data/qrecc/passages.jsonl", "index_path": ".cache/qrecc/index.json",
            "eval_path": "data/qrecc/questions.jsonl", "answer_language": "en",
        }
        (self.root / "configs" / "qrecc.json").write_text(json.dumps(self.qrecc_profile), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def write_passages(self):
        (self.root / "data" / "qrecc" / "passages.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in self.passages), encoding="utf-8")

    def engine(self, profile="qrecc", mode="demo", **overrides):
        return RAGEngine(self.root, load_config(self.root, overrides, profile=profile), mode)

    def test_qrecc_profile_retrieves_full_original_passage_and_english_context(self):
        engine = self.engine()
        result = engine.answer("What stores solar energy?", [], "current", 1)
        hit = result["hits"][0]
        self.assertEqual(hit["passage_id"], self.passage_id)
        self.assertEqual(hit["text"], self.long_text)
        self.assertGreater(len(hit["text"]), engine.config["chunk_size"])
        self.assertEqual(result["dataset"], "qrecc")
        self.assertIn("Source:", result["context"])
        self.assertIn("Offline evidence extracts", result["answer"])
        self.assertIsNone(result["model"])
        self.assertEqual(engine.retriever.stats()["chunks"], 2)

    def test_query_only_does_not_read_or_build_missing_corpus(self):
        config = load_config(self.root, {"corpus_path": "missing/passages.jsonl"}, profile="qrecc")
        with patch.object(Retriever, "build", side_effect=AssertionError("must not build")) as build:
            with patch.object(Retriever, "_read_documents", side_effect=AssertionError("must not read corpus")) as read:
                engine = RAGEngine(self.root, config, "demo", build_index=False)
                result = engine.rewrite_query("What stores solar energy?", [], "current")
        build.assert_not_called()
        read.assert_not_called()
        self.assertEqual(result["search_query"], "What stores solar energy?")
        self.assertFalse(engine.retriever.stats()["built"])

    def test_history_selector_can_only_choose_from_the_bounded_window(self):
        engine = self.engine(mode="ollama", history_turns=2)
        history = [{"role": "user", "content": text} for text in
                   ("Older excluded turn", "What powers the spacecraft?", "Where is the library?")]
        engine.client.generate = Mock(return_value=json.dumps({
            "topic": "spacecraft", "referent": "the spacecraft",
            "standalone_query": "How long can the spacecraft operate?", "ambiguous": False,
        }))
        selector = Mock(return_value=[0])
        question = "How long can it operate?"
        result = engine.rewrite_query(question, history, "structured", history_selector=selector)
        selector.assert_called_once_with(question, tuple(history[-2:]))
        payload = json.loads(engine.client.generate.call_args.args[0].split("\n", 1)[1])
        self.assertEqual(payload["history"], history[1:2])
        self.assertEqual(result["history_window"], history[-2:])
        self.assertEqual(result["selected_history_positions"], [0])
        self.assertEqual(result["visible_history"], history[1:2])
        self.assertIsNone(result["rewrite_error"])

        selector.return_value = []
        kept = engine.rewrite_query(question, history, "structured", history_selector=selector)
        self.assertEqual(kept["visible_history"], history[-2:])
        for invalid in ([2], [1, 0], [0, 0], [True]):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "历史选择器"):
                engine.rewrite_query(question, history, "structured",
                                     history_selector=Mock(return_value=invalid))

    def test_all_strategies_receive_same_bounded_roles_with_missing_answer(self):
        history = [
            {"role": "system", "content": "not part of the conversation"},
            {"role": "user", "content": "Old topic"},
            {"role": "assistant", "content": "Old answer"},
            {"role": "user", "content": "What powers this spacecraft?"},
            # Deliberately no assistant answer here.
            {"role": "user", "content": "Does the spacecraft store solar energy?"},
            {"role": "assistant", "content": "The spacecraft has energy storage."},
        ]
        untouched = copy.deepcopy(history)
        engine = self.engine(history_turns=2)
        expected = history[-3:]
        self.assertEqual(clean_history(history, 2), expected)
        plans = [engine.rewrite_query("How long can it operate?", history, strategy) for strategy in STRATEGIES]
        self.assertEqual(len({plan["rewrite_trace"]["history_signature"] for plan in plans}), 1)
        self.assertEqual({plan["rewrite_trace"]["visible_history_messages"] for plan in plans}, {3})
        self.assertTrue(all(plan["rewrite_trace"]["llm_calls"] == 0 for plan in plans))
        history_plan = next(plan for plan in plans if plan["strategy"] == "history")
        self.assertNotIn("Old topic", history_plan["search_query"])
        self.assertIn("user：What powers this spacecraft?", history_plan["search_query"])
        self.assertEqual(history, untouched)

    def test_model_rewriters_see_identical_explicit_history(self):
        engine = self.engine(mode="ollama")
        history = [
            {"role": "user", "content": "Old question?"},
            {"role": "assistant", "content": "Old response."},
            {"role": "user", "content": "Tell me about the spacecraft."},
            {"role": "user", "content": "What powers the spacecraft?"},
            {"role": "assistant", "content": "The spacecraft stores solar energy."},
        ]
        visible = clean_history(history, 2)
        query = "How long can the spacecraft operate?"
        for strategy in ["rewrite", "structured"]:
            with self.subTest(strategy=strategy):
                output = ({"query": query} if strategy == "rewrite" else
                          {"topic": "spacecraft", "referent": "the spacecraft",
                           "standalone_query": query, "ambiguous": True})
                engine.client.generate = Mock(return_value=json.dumps(output))
                plan = engine.rewrite_query("How long can it operate?", history, strategy)
                self.assertIsNone(plan["rewrite_error"])
                payload = json.loads(engine.client.generate.call_args.args[0].split("\n", 1)[1])
                self.assertEqual(payload["history"], visible)
                self.assertEqual(payload["current_question"], "How long can it operate?")
                self.assertEqual(set(payload), {"history", "current_question"})

    def test_answer_prompt_is_identical_for_same_evidence_and_has_no_rewritten_query(self):
        engine = self.engine(mode="ollama")
        evidence = engine.retriever.search("spacecraft", 1)
        history = [{"role": "user", "content": "Tell me about the spacecraft."}]
        engine.client.generate = Mock(return_value="The spacecraft stores solar energy. [S1]")
        prompts = []
        for strategy, query in [("current", "UNIQUE_QUERY_SENTINEL_A"), ("structured", "UNIQUE_QUERY_SENTINEL_B")]:
            result = engine.answer_from_retrieval({
                "question": "What powers it?", "search_query": query, "strategy": strategy,
                "hits": evidence, "rewrite_error": None,
                "latencies_ms": {"rewrite": 1, "retrieval": 1, "total": 2},
            }, history)
            prompt = engine.client.generate.call_args.args[0]
            self.assertNotIn("UNIQUE_QUERY_SENTINEL", prompt)
            self.assertIn("What powers it?", prompt)
            self.assertIn(evidence[0]["text"][:50], prompt)
            self.assertEqual(result["warnings"], [])
            prompts.append(prompt)
        self.assertEqual(prompts[0], prompts[1])

    def test_configured_cache_rebuilds_never_mix_corpora(self):
        markdown = self.engine(**self.markdown_overrides)
        qrecc = self.engine()
        self.assertNotEqual(markdown.retriever.index_path, qrecc.retriever.index_path)
        self.assertEqual(markdown.retrieve("portable scanner", [], "current", 1)["hits"][0]["document"], "scanner.md")
        self.assertEqual(qrecc.retrieve("spacecraft", [], "current", 1)["hits"][0]["passage_id"], self.passage_id)
        self.assertTrue(self.engine(**self.markdown_overrides).retriever.stats()["cache_hit"])
        self.assertTrue(self.engine().retriever.stats()["cache_hit"])
        self.passages[0]["text"] = "The spacecraft operates on solar batteries."
        self.write_passages()
        changed = self.engine()
        self.assertFalse(changed.retriever.stats()["cache_hit"])
        self.assertTrue(self.engine(**self.markdown_overrides).retriever.stats()["cache_hit"])
        changed.retriever.index_path.unlink()
        self.assertFalse(self.engine().retriever.stats()["cache_hit"])
        self.assertTrue(self.engine(**self.markdown_overrides).retriever.stats()["cache_hit"])
        markdown.retriever.index_path.unlink()
        self.assertFalse(self.engine(**self.markdown_overrides).retriever.stats()["cache_hit"])
        self.assertTrue(self.engine().retriever.stats()["cache_hit"])

    def test_profile_overrides_survive_reloading_complete_config(self):
        config = load_config(self.root, {"history_turns": 1, "top_k": 1, "bm25_mode": "scan"}, profile="qrecc")
        engine = RAGEngine(self.root, config, "demo")
        self.assertEqual(engine.config["dataset"], "qrecc")
        self.assertEqual(engine.config["history_turns"], 1)
        self.assertEqual(engine.config["top_k"], 1)
        self.assertEqual(engine.retriever.bm25_mode, "scan")


if __name__ == "__main__":
    unittest.main()
