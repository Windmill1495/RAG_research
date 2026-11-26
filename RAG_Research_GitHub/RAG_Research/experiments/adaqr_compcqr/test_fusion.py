"""CPU checks of reward construction with the project's original ComPQR functions."""
import os
from pathlib import Path
import sys
import unittest

sys.path.insert(0, os.environ.get("RAG_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))

from train_fusion import candidates_for, mixed_pairs, score_turn


class MixedRewardTests(unittest.TestCase):
    def test_mixing_can_reverse_preference(self):
        record = {"id": "1-2", "prompt": "prompt", "candidates": [
            {"rewrite": "A", "single_reward": -1.0, "fusion_reward": -6.0},
            {"rewrite": "B", "single_reward": -2.0, "fusion_reward": -2.0},
        ]}
        self.assertEqual(mixed_pairs(record, 0, 0.1)[0]["chosen"], "A")
        self.assertEqual(mixed_pairs(record, 0.5, 0.1)[0]["chosen"], "B")
        self.assertEqual(mixed_pairs(record, 1, 0.1)[0]["chosen"], "B")
        self.assertEqual(mixed_pairs(record, 0.2, 0.1), [])

    def test_separate_rankings_and_reuse(self):
        row = {"id": "1-2", "split": "train", "history": [{"role": "user", "content": "earlier"}],
               "question": "original question", "answer": "gold", "candidates": ["candidate A", "candidate B", "candidate B"]}
        components = {"oq": "original question", "mq": "generated baseline",
                      "ma": "hypothetical", "mq2": "", "ma2": ""}
        calls = []

        class Search:
            def search_many(self, queries, top_k):
                self.queries = queries
                rankings = []
                for query in queries:
                    candidate = next((tag for tag in ("candidate A", "candidate B") if tag in query), "shared")
                    rankings.append([{"id": candidate, "text": candidate, "score": 5.0},
                                     {"id": "common", "text": "common", "score": 4.0}])
                return rankings

        def score(prompt, answer):
            self.assertEqual(answer, "gold")
            self.assertIn("Question: original question", prompt)
            calls.append(prompt)
            return -1.0

        search = Search()
        result = score_turn(row, retriever=search, components=components, answer_logp=score,
                            config={"retriever_backend": "bm25", "compcqr_max_components": 5,
                                    "compcqr_retrieval_depth": 100, "compcqr_rrf_k": 0})
        self.assertEqual(len(result["candidates"]), 2)
        self.assertEqual(len(search.queries), len(set(search.queries)))
        self.assertTrue(all("generated baseline" not in query and "gold" not in query
                            for query in search.queries))
        self.assertEqual(components["mq"], "generated baseline")
        self.assertIn("candidate A", result["candidates"][0]["fusion_passage_ids"])
        self.assertNotIn("candidate B", result["candidates"][0]["fusion_passage_ids"])
        self.assertIn("candidate B", result["candidates"][1]["fusion_passage_ids"])
        self.assertNotIn("candidate A", result["candidates"][1]["fusion_passage_ids"])
        self.assertEqual(len(calls), 4)

    def test_test_rows_cannot_be_scored_as_training(self):
        with self.assertRaises(ValueError):
            candidates_for({"id": "test:1-2", "split": "test", "candidates": ["A", "B", "C"]})


if __name__ == "__main__":
    unittest.main()
