"""HTTP contract/error tests; these do not claim a real model was executed."""

import io
import json
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from rag.llm import OllamaClient, OllamaError


def response(value):
    return io.BytesIO(json.dumps(value, ensure_ascii=False).encode("utf-8"))


class OllamaClientTests(unittest.TestCase):
    @patch("rag.llm.urlopen")
    def test_generate_uses_nonstreaming_api(self, request):
        request.return_value = response({"response": " 答案 [1] ", "done": True})
        result = OllamaClient().generate("请回答", system="只用证据")
        self.assertEqual(result, "答案 [1]")
        req = request.call_args.args[0]
        payload = json.loads(req.data)
        self.assertEqual(req.full_url, "http://127.0.0.1:11434/api/generate")
        self.assertIs(payload["stream"], False)
        self.assertEqual(payload["model"], "qwen3:1.7b")
        self.assertIs(payload["think"], False)
        self.assertEqual(payload["system"], "只用证据")
        self.assertEqual(payload["options"]["temperature"], 0.0)

    @patch("rag.llm.urlopen")
    def test_embed_batch_contract(self, request):
        request.return_value = response({"embeddings": [[1, 0.5], [0, 1]]})
        self.assertEqual(OllamaClient().embed(["甲", "乙"]), [[1.0, 0.5], [0.0, 1.0]])
        payload = json.loads(request.call_args.args[0].data)
        self.assertEqual(payload["input"], ["甲", "乙"])
        self.assertIs(payload["truncate"], True)

    @patch("rag.llm.urlopen")
    def test_rejects_wrong_embedding_count_and_dimensions(self, request):
        request.return_value = response({"embeddings": [[1, 2]]})
        with self.assertRaises(OllamaError):
            OllamaClient().embed(["甲", "乙"])
        request.return_value = response({"embeddings": [[1], [1, 2]]})
        with self.assertRaises(OllamaError):
            OllamaClient().embed(["甲", "乙"])

    @patch("rag.llm.urlopen")
    def test_rejects_nonfinite_embedding(self, request):
        request.return_value = response({"embeddings": [[float("nan")]]})
        with self.assertRaises(OllamaError):
            OllamaClient().embed(["甲"])

    @patch("rag.llm.urlopen")
    def test_lists_local_models(self, request):
        request.return_value = response({"models": [{"name": "qwen2.5:1.5b"}]})
        self.assertEqual(OllamaClient().list_models(), ["qwen2.5:1.5b"])
        self.assertEqual(request.call_args.args[0].method, "GET")

    @patch("rag.llm.urlopen")
    def test_missing_model_has_actionable_message(self, request):
        request.side_effect = HTTPError("http://localhost", 404, "Not Found", {}, response({"error": "model not found"}))
        with self.assertRaisesRegex(OllamaError, "ollama pull qwen3:1.7b"):
            OllamaClient().generate("测试")

    @patch("rag.llm.urlopen", side_effect=URLError("refused"))
    def test_unavailable_service_does_not_return_fake_answer(self, request):
        with self.assertRaisesRegex(OllamaError, "启动 Ollama"):
            OllamaClient().generate("测试")

    @patch("rag.llm.urlopen")
    def test_rejects_empty_generation(self, request):
        request.return_value = response({"response": "", "done": True})
        with self.assertRaises(OllamaError):
            OllamaClient().generate("测试")

    def test_empty_embedding_batch_needs_no_server(self):
        self.assertEqual(OllamaClient().embed([]), [])

    def test_rejects_malformed_configuration(self):
        for url in ["file:///a", "https://", "http://user:secret@localhost", "http://localhost/api"]:
            with self.assertRaises(ValueError):
                OllamaClient(base_url=url)
        with self.assertRaises(ValueError):
            OllamaClient(timeout=0)


if __name__ == "__main__":
    unittest.main()
