"""Exercise the opt-in C API and its visible offline contract without a model."""
import json
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from unittest.mock import patch

from app import create_server
from rag.llm import OllamaError


class CompCQRWebTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch("app.OllamaClient.list_models", side_effect=OllamaError("offline")):
            cls.server = create_server(0, "demo", initial_page="rewrite")
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(2)

    def post(self, value):
        request = Request(self.url + "/api/rewrite/compare",
                          data=json.dumps(value).encode(),
                          headers={"Content-Type": "application/json"})
        with urlopen(request) as response:
            return json.load(response)

    def test_c_is_advertised_and_explicit_request_returns_three_rows(self):
        with urlopen(self.url + "/api/rewrite/samples") as response:
            catalogue = json.load(response)
        self.assertEqual(catalogue["strategies"], ["rewrite", "structured", "compcqr"])
        self.assertNotIn("human_rewrite", catalogue["samples"][0])
        with patch("rag.llm.OllamaClient.generate", side_effect=AssertionError("demo called LLM")):
            result = self.post({"sample_id": "1471-2", "strategies": catalogue["strategies"]})
        self.assertEqual([row["strategy"] for row in result["results"]], catalogue["strategies"])
        c = result["results"][2]
        self.assertEqual(c["rewrite_source"], "compcqr_demo")
        self.assertEqual(c["rewrite_trace"]["llm_calls"], 0)
        self.assertTrue(c["rewrite_trace"]["queries"])
        self.assertEqual(c["rewrite_trace"]["components"]["ma"], "")
        self.assertEqual(c["rouge_scope"], "base_rewrite_only")

    def test_rejects_unknown_duplicate_and_nonlist_strategies(self):
        for invalid in (["unknown"], ["compcqr", "compcqr"], "compcqr", [], [True]):
            with self.subTest(strategies=invalid), self.assertRaises(HTTPError) as caught:
                self.post({"sample_id": "1471-2", "strategies": invalid})
            self.assertEqual(caught.exception.code, 400)


if __name__ == "__main__":
    unittest.main()
