"""Dataset selection must not silently mismatch the corpus or its cache."""
import tempfile
import unittest
from pathlib import Path

from rag.config import load_config
from rag.llm import OllamaClient, OllamaError, runtime_model_identity
from unittest.mock import patch


class ProfileTests(unittest.TestCase):
    def test_default_and_explicit_dataset_use_matching_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            config = load_config(Path(directory))
            self.assertEqual(config, load_config(Path(directory), {"dataset": "qrecc"}))
            self.assertEqual(config["corpus_format"], "passages_jsonl")
            self.assertIn("qrecc", config["index_path"])
            self.assertEqual(config["answer_language"], "en")

    def test_explicit_conflicting_dataset_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "冲突"):
                load_config(Path(directory), {"dataset": "other"}, profile="qrecc")

    def test_invalid_resource_budgets_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for value in [{"bm25_mode": "unknown"},
                          {"answer_language": "unknown"}, {"corpus_path": ""}]:
                with self.subTest(value=value), self.assertRaises(ValueError):
                    load_config(Path(directory), value)

    def test_demo_identity_does_not_contact_a_service(self):
        with patch.object(OllamaClient, "_request", side_effect=AssertionError("no service")):
            self.assertFalse(runtime_model_identity({}, "demo")["llm_called"])

    def test_real_identity_distinguishes_weight_updates_and_rejects_missing_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            config = load_config(Path(directory))
            with patch.object(OllamaClient, "_request", return_value={"models": [
                {"name": config["model"], "digest": "weights-one"}]}):
                first = runtime_model_identity(config, "ollama")
            with patch.object(OllamaClient, "_request", return_value={"models": [
                {"name": config["model"], "digest": "weights-two"}]}):
                second = runtime_model_identity(config, "ollama")
            self.assertNotEqual(first, second)
            with patch.object(OllamaClient, "_request", return_value={"models": [
                {"name": config["model"]}]}), self.assertRaises(OllamaError):
                runtime_model_identity(config, "ollama")
