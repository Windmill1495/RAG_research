"""Batch ranking equivalence and an end-to-end C evidence-selection boundary."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rag.config import load_config
from rag.pipeline import RAGEngine
from rag.retrieval import Retriever


class MultiQueryRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.corpus = self.root / "passages.jsonl"
        self.corpus.write_text("\n".join(json.dumps({"id": str(i), "text": text}) for i, text in enumerate([
            "The band Fleet Foxes formed in 2006.", "FAT tree topology is a network architecture.",
            "A band performs music together."])), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_batch_matches_scalar_all_backends_and_encodes_unique_queries_once(self):
        queries = ["band Fleet Foxes", "network architecture", "band Fleet Foxes"]
        for backend in ("bm25", "dense", "hybrid"):
            calls = []
            def embed(texts):
                calls.append(list(texts))
                return [[1 + t.lower().count("band"), 1 + t.lower().count("network"),
                         1 + t.lower().count("foxes")] for t in texts]
            r = Retriever(self.corpus, self.root / (backend + ".json"), backend=backend,
                          corpus_format="passages_jsonl", embedding_fn=embed)
            r.build()
            expected = [r.search(q, 3) for q in queries]
            calls.clear()
            actual = r.search_many(queries, 3)
            self.assertEqual(actual, expected)
            self.assertEqual(calls, [] if backend == "bm25" else [queries[:2]])
            with patch.dict("sys.modules", {"numpy": None}):
                self.assertEqual(r.search_many(queries, 3), expected)
            actual[0][0]["score"] = -999
            self.assertNotEqual(actual[2][0]["score"], -999)

    def test_c_plan_retrieved_once_and_hypothetical_answer_never_becomes_evidence(self):
        config = load_config(self.root, {"corpus_path": str(self.corpus),
            "index_path": str(self.root / "index.json"), "retriever_backend": "bm25"})
        engine = RAGEngine(self.root, config, mode="ollama", build_index=False)
        replies = [json.dumps({"query": "What year did Fleet Foxes form?", "answer": "UNVERIFIED_2006"}),
                   json.dumps({"followup_query": "Where did Fleet Foxes start?", "next_answer": "UNVERIFIED_Seattle"}),
                   json.dumps({"selected_passage_number": 2, "reason": "chosen excerpt"})]
        with patch.object(engine.client, "generate", side_effect=replies) as generate:
            plan = engine.rewrite_query("What year did the band form?", [
                {"role": "user", "content": "Tell me about Fleet Foxes."}], "compcqr")
            with patch.object(engine.retriever, "search_many", wraps=engine.retriever.search_many) as search:
                result = engine.retrieve_rewritten(plan, 3)
                self.assertEqual(search.call_count, 1)
            self.assertEqual(generate.call_count, 3)
            selection_prompt = generate.call_args.args[0]
            self.assertNotIn("UNVERIFIED_", selection_prompt)
        self.assertEqual(result["hits"][0]["passage_id"], result["rewrite_trace"]["selected_passage_id"])
        self.assertEqual(result["rewrite_trace"]["llm_calls"], 3)
        self.assertTrue(all("UNVERIFIED_" not in hit["text"] for hit in result["hits"]))
        self.assertGreaterEqual(result["latencies_ms"]["total"], result["latencies_ms"]["rewrite"])


if __name__ == "__main__":
    unittest.main()
