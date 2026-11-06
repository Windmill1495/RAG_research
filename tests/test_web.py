"""Exercise the real local HTTP server without downloading a model."""
import json
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from app import create_server
from rag.pipeline import RAGEngine
from rag.llm import OllamaError


class WebTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch("app.OllamaClient.list_models", side_effect=OllamaError("not installed")):
            cls.server = create_server(0, "auto")
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(2)

    def test_auto_mode_falls_back_visibly(self):
        with urlopen(self.url + "/api/status") as response:
            status = json.load(response)
        self.assertEqual(status["mode"], "demo")
        self.assertFalse(status["ollama_available"])
        self.assertEqual(status["dataset"], "qrecc")
        self.assertEqual(status["config"]["retriever_backend"], "bm25")

    def test_static_assets_and_question_roundtrip(self):
        for path in ("/", "/app.js", "/style.css"):
            with urlopen(self.url + path) as response:
                self.assertEqual(response.status, 200)
                self.assertTrue(response.read())
        body = {"question": "What is a fat tree topology?", "strategy": "current", "mode": "demo", "top_k": 2}
        request = Request(self.url + "/api/answer", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urlopen(request) as response:
            result = json.load(response)
        self.assertEqual(result["mode"], "demo")
        self.assertTrue(result["sources"])
        self.assertTrue(result["sources"][0]["passage_id"])

    def test_cross_origin_and_file_traversal_are_rejected(self):
        request = Request(self.url + "/api/status", headers={"Origin": "https://example.org"})
        with self.assertRaises(HTTPError) as caught:
            urlopen(request)
        self.assertEqual(caught.exception.code, 403)
        with self.assertRaises(HTTPError) as caught:
            urlopen(self.url + "/../config.json")
        self.assertEqual(caught.exception.code, 404)

    def test_slow_index_does_not_block_page_or_status(self):
        started, release, finished = threading.Event(), threading.Event(), threading.Event()

        def delayed_engine(*args, **kwargs):
            started.set()
            if not release.wait(5):
                raise AssertionError("HTTP startup waited for the index")
            try:
                return RAGEngine(*args, **kwargs)
            finally:
                finished.set()

        with (
            patch("app.OllamaClient.list_models", side_effect=OllamaError("not installed")),
            patch("app.RAGEngine", side_effect=delayed_engine),
        ):
            server = create_server(0, "auto")
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                self.assertTrue(started.wait(1))
                url = f"http://127.0.0.1:{server.server_port}"
                with urlopen(url + "/", timeout=2) as response:
                    self.assertEqual(response.status, 200)
                with urlopen(url + "/api/status", timeout=2) as response:
                    status = json.load(response)
                self.assertEqual(status["initialization"]["state"], "building")
                self.assertFalse(status["index"]["built"])
            finally:
                release.set()
                finished.wait(5)
                server.shutdown()
                server.server_close()
                thread.join(2)

    def test_rewrite_demo_is_visibly_offline(self):
        for path in ("/rewrite", "/rewrite.js"):
            with urlopen(self.url + path) as response:
                self.assertEqual(response.status, 200)
                self.assertTrue(response.read())
        with urlopen(self.url + "/api/rewrite/samples") as response:
            catalogue = json.load(response)
        self.assertEqual(catalogue["mode"], "demo")
        self.assertEqual(catalogue["history_turns"], 3)
        self.assertEqual(len(catalogue["samples"]), 307)
        self.assertEqual({row["split"] for row in catalogue["samples"]}, {"dev", "test"})
        self.assertNotIn("human_rewrite", catalogue["samples"][0])

        sample = next(row for row in catalogue["samples"] if row["split"] == "test")
        payload = json.dumps({"sample_id": sample["id"]}).encode()
        request = Request(self.url + "/api/rewrite/compare", data=payload,
                          headers={"Content-Type": "application/json"})
        with urlopen(request) as response:
            result = json.load(response)
        self.assertEqual(result["mode"], "demo")
        self.assertIsNone(result["model_digest"])
        self.assertEqual([row["strategy"] for row in result["results"]], ["rewrite", "structured"])
        for row in result["results"]:
            self.assertEqual(row["rewrite_source"], "heuristic_demo")
            self.assertEqual(row["rewrite_trace"]["llm_calls"], 0)
            self.assertEqual(row["id"], sample["id"])
            self.assertEqual(row["split"], "test")

    def test_rewrite_page_uses_a_shared_input_and_retrieves_each_final_rewrite(self):
        model = "qwen3:test"
        answer = "What year did Fleet Foxes form?"
        identity = {"name": model, "digest": "test-model-digest"}
        model_outputs = [
            json.dumps({"query": answer}),
            json.dumps({"topic": "band", "referent": "Unlisted Band",
                        "standalone_query": "What year did Unlisted Band form?", "ambiguous": False}),
            json.dumps({"standalone_query": answer}),
        ]
        with (
            patch("app.OllamaClient.list_models", return_value=[model, "bge-m3:latest"]),
            patch("app.runtime_model_identity", return_value=identity) as identify,
            patch("rag.llm.OllamaClient.generate", side_effect=model_outputs) as generate,
            patch("rag.pipeline.runtime_embedding_identity", return_value={"digest": "test-embedding-digest"}),
            patch("rag.retrieval.Retriever.build") as build,
            patch("rag.retrieval.Retriever.search", return_value=[]) as search,
        ):
            server = create_server(0, "auto", rewrite_model=model, initial_page="rewrite")
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                url = f"http://127.0.0.1:{server.server_port}"
                with urlopen(url + "/api/rewrite/samples") as response:
                    catalogue = json.load(response)
                self.assertEqual((catalogue["mode"], catalogue["model"]), ("ollama", model))
                build.assert_not_called()
                sample = next(row for row in catalogue["samples"] if row["id"] == "1471-2")
                request = Request(url + "/api/rewrite/compare",
                                  data=json.dumps({"sample_id": sample["id"]}).encode(),
                                  headers={"Content-Type": "application/json"})
                with urlopen(request) as response:
                    result = json.load(response)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(2)
        self.assertEqual(identify.call_args.args[0]["model"], model)
        self.assertEqual(result["reference_rewrite"], answer)
        self.assertEqual(result["model_digest"], identity["digest"])
        self.assertEqual(result["retriever_backend"], "hybrid")
        first, second = result["results"]
        self.assertEqual([first["rouge1_f1"], second["rouge1_f1"]], [1.0, 1.0])
        self.assertTrue(second["rewrite_trace"]["repaired"])
        self.assertEqual(generate.call_count, 3)
        build.assert_called_once()
        self.assertEqual([call.args for call in search.call_args_list], [(answer, 3), (answer, 3)])
        initial_inputs = [json.loads(call.args[0].rsplit("\n", 1)[1]) for call in generate.call_args_list[:2]]
        self.assertEqual(initial_inputs, [{"history": sample["history"], "current_question": sample["question"]}] * 2)


if __name__ == "__main__":
    unittest.main()
