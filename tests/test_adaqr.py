"""Lightweight AdaQR algorithm checks; no model or index is loaded."""
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rag.adaqr import AdaQRSettings, collect_preferences, marginal_reward, preference_pairs, rewrite_prompt
from rag.adaqr_model import encode_completion
from adaqr_train import training_rows
from rag.rewriters import build_query


class AdaQRTests(unittest.TestCase):
    def test_reward_is_softmax_weighted_log_probability(self):
        hits = [{"score": 0.0}, {"score": math.log(3)}]
        self.assertAlmostEqual(marginal_reward(hits, [-2.0, -0.4]), -0.8)
        self.assertNotAlmostEqual(marginal_reward(hits, [-2.0, -0.4]),
                                  math.log(.25 * math.exp(-2) + .75 * math.exp(-.4)))

    def test_all_pairs_with_gap_above_delta_and_no_duplicate_text(self):
        candidates = [{"rewrite": "a", "reward": -0.1}, {"rewrite": "b", "reward": -0.3},
                      {"rewrite": "c", "reward": -0.6}]
        self.assertEqual(len(preference_pairs(candidates)), 3)
        self.assertEqual(preference_pairs(candidates, 0.5), [])
        self.assertEqual(preference_pairs([candidates[0], {"rewrite": "a", "reward": -1}]), [])

    def test_collection_uses_original_question_and_each_passage(self):
        example = {"id": "x", "history": [{"role": "user", "content": "Who is Ada?"}],
                   "question": "When was she born?", "answer": "1815"}
        prompts = []
        def score(prompt, answer):
            prompts.append(prompt)
            return -1.0 if "good" in prompt else -2.0
        def search(rewrite, top_k):
            self.assertEqual(top_k, 5)
            return [{"id": rewrite, "score": 1.0, "text": "good" if rewrite == "a" else "bad"}]
        result = collect_preferences(example, lambda prompt, n, t: ["a", "b", "c"],
                                     search, score, AdaQRSettings())
        self.assertEqual(len(prompts), 3)
        self.assertTrue(all("Question: When was she born?" in p for p in prompts))
        self.assertEqual(len(result["pairs"]), 2)
        self.assertEqual(result["pairs"][0]["chosen"], "a")

    def test_answer_score_can_exclude_eos_while_sft_keeps_it(self):
        class Tokenizer:
            eos_token_id = 99
            def encode(self, text, add_special_tokens):
                return ([1] if add_special_tokens else []) + [ord(c) for c in text]
        tokenizer = Tokenizer()
        answer, start = encode_completion(tokenizer, "p", "a", 10, include_eos=False)
        rewrite, _ = encode_completion(tokenizer, "p", "a", 10)
        self.assertEqual(answer[start:], [ord("a")])
        self.assertEqual(rewrite[-2:], [ord("a"), 99])

    def test_training_reader_rejects_nontrain_rows(self):
        row = {"id": "x", "split": "train", "turn": 2,
               "history": [{"role": "user", "content": "Earlier question", "turn_id": 1}],
               "question": "Now?", "rewrite": "Now about X?", "answer": "X"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            self.assertEqual(len(training_rows(path)), 1)
            path.write_text(json.dumps({**row, "split": "test"}) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                training_rows(path)

    def test_strategy_requires_adapter_and_never_falls_back(self):
        history = [{"role": "user", "content": "Who is Ada?"}]
        with self.assertRaises(ValueError):
            build_query("When?", history, "adaqr", client=None, mode="demo")
        with patch("rag.adaqr_model.rewrite_with_adapter", return_value="When was Ada born?") as rewrite:
            result = build_query("When?", history, "adaqr", client=None, mode="ollama",
                                 config={"adaqr_model_path": "/tmp/fake"})
        self.assertEqual(result["query"], "When was Ada born?")
        self.assertEqual(result["source"], "adaqr_dpo")
        rewrite.assert_called_once()


if __name__ == "__main__":
    unittest.main()
