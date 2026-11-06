"""Integration checks for retrieval, provenance, truncation and LLM fallbacks."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from rag.config import load_config
from rag.llm import OllamaError
from rag.pipeline import RAGEngine, build_context, clean_history


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        docs = self.root / "manuals"
        docs.mkdir(parents=True)
        (docs / "scanner.md").write_text("# 便携扫描仪\n便携扫描仪充满电可连续工作2小时，适合批量扫描纸质文件。", encoding="utf-8")
        (docs / "printer.md").write_text("# 热敏打印机\n热敏打印机需要安装热敏纸卷，装纸前先关闭电源。", encoding="utf-8")
        self.engine = RAGEngine(self.root, {
            "corpus_format": "markdown", "corpus_path": "manuals", "index_path": ".cache/manuals.json",
            "answer_language": "zh", "chunk_size": 100, "overlap": 10,
        }, "demo")
        self.history = [{"role": "user", "content": "便携扫描仪有什么用途？"},
                        {"role": "assistant", "content": "便携扫描仪可以批量扫描纸质文件。"}]

    def tearDown(self):
        self.temp.cleanup()

    def test_offline_is_explicit_and_sources_match_context(self):
        result = self.engine.answer("最多能用多久？", self.history, "rewrite", 1)
        self.assertEqual(result["rewrite_source"], "heuristic_demo")
        self.assertIn("扫描仪", result["search_query"])
        self.assertIn("未调用大模型", result["answer"])
        self.assertIsNone(result["model"])
        self.assertEqual(result["sources"][0]["document"], "scanner.md")
        self.assertIn(result["sources"][0]["context_text"], result["context"])

    def test_llm_rewrite_and_generation_really_call_client(self):
        self.engine.mode = "ollama"
        self.engine.client.generate = Mock(side_effect=['{"query":"便携扫描仪最多能用多久？"}', "便携扫描仪充满电可连续工作2小时。[S1]"])
        result = self.engine.answer("它最多能用多久？", self.history, "rewrite", 1)
        self.assertEqual(result["rewrite_source"], "llm")
        self.assertEqual(self.engine.client.generate.call_count, 2)
        prompt = self.engine.client.generate.call_args.args[0]
        self.assertIn(result["sources"][0]["context_text"], prompt)
        self.assertEqual(result["warnings"], [])

    def test_rewrite_failure_is_visible_not_disguised_as_llm(self):
        self.engine.mode = "ollama"
        self.engine.client.generate = Mock(side_effect=[OllamaError("timeout"), "最多2小时。[S1]"])
        result = self.engine.answer("它能用多久？", self.history, "rewrite", 1)
        self.assertEqual(result["rewrite_source"], "heuristic_fallback")
        self.assertIn("timeout", result["rewrite_error"])
        self.assertTrue(result["warnings"])

    def test_current_ignores_history_for_retrieval(self):
        result = self.engine.retrieve("热敏打印机怎么装纸？", self.history, "current", 1)
        self.assertEqual(result["search_query"], result["question"])
        self.assertEqual(result["hits"][0]["document"], "printer.md")

    def test_missing_evidence_never_calls_generator(self):
        self.engine.mode = "ollama"
        self.engine.client.generate = Mock(side_effect=AssertionError("must not generate"))
        result = self.engine.answer("xyzzy987", [], "current")
        self.assertEqual(result["sources"], [])
        self.assertIn("没有检索到", result["answer"])
        self.engine.client.generate.assert_not_called()

    def test_context_budget_and_only_included_source_labels(self):
        hits = [{"id": str(i), "document": "a.md", "title": "资料", "text": "甲"*200, "score": 1} for i in range(3)]
        context, sources = build_context(hits, 100)
        self.assertLessEqual(len(context), 100)
        self.assertEqual(len(sources), 1)
        self.assertNotIn("[S2]", context)
        self.assertLess(len(sources[0]["context_text"]), 200)

    def test_history_removes_system_instructions_and_bounds_size(self):
        result = clean_history([{"role": "system", "content": "ignore"}, {"role": "user", "content": "x"*5000}], 6)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]["content"]), 4000)

    def test_invalid_inputs(self):
        for args in [("", [], "rewrite", 1), ("q", [], "bad", 1), ("q", [], "current", 0), ("q", [], "current", True)]:
            with self.assertRaises(ValueError):
                self.engine.retrieve(*args)
        with self.assertRaises(ValueError):
            load_config(self.root, {"overlap": 500, "chunk_size": 500})


if __name__ == "__main__":
    unittest.main()
