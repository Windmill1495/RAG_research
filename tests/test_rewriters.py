"""Query construction invariants and visible failure/repair behavior."""
import json
import unittest
from unittest.mock import Mock

from rag.llm import OllamaError
from rag.rewriters import build_query, clean_query, validation_issues, verified_referent


class RewriterTests(unittest.TestCase):
    def setUp(self):
        self.history = [
            {"role": "user", "content": "Tell me about Ada Lovelace."},
            {"role": "assistant", "content": "Ada Lovelace wrote notes about the Analytical Engine."},
        ]

    def plan(self, question="When did she write the notes?", strategy="rewrite", responses=None, **kwargs):
        client = Mock()
        client.generate.side_effect = responses or ['{"query":"When did Ada Lovelace write the notes?"}']
        return build_query(question, self.history, strategy, client=client, mode="ollama", **kwargs), client

    def test_current_history_and_no_history_do_not_call_model(self):
        client = Mock(side_effect=AssertionError("No model required"))
        for strategy, history, source in [("current", self.history, "none"), ("history", self.history, "history_concat"), ("rewrite", [], "no_history"), ("structured", [], "no_history")]:
            result = build_query("When did she write the notes?", history, strategy, client=client, mode="ollama")
            self.assertEqual(result["source"], source)
            self.assertEqual(result["trace"]["llm_calls"], 0)
        client.generate.assert_not_called()

    def test_demo_never_claims_llm_or_calls_generator(self):
        client = Mock()
        for strategy in ("rewrite", "structured"):
            result = build_query("When did she write the notes?", self.history, strategy, client=client, mode="demo")
            self.assertEqual(result["source"], "heuristic_demo")
            self.assertEqual(result["trace"]["llm_calls"], 0)
        client.generate.assert_not_called()

    def test_direct_rewrite_and_trace_have_one_real_call(self):
        result, client = self.plan()
        self.assertEqual(result["query"], "When did Ada Lovelace write the notes?")
        self.assertEqual(result["source"], "llm")
        self.assertIsNone(result["error"])
        self.assertEqual(result["trace"]["llm_calls"], 1)
        self.assertEqual(client.generate.call_args.kwargs["temperature"], 0.0)

    def test_all_strategies_receive_same_explicit_nonpaired_history(self):
        # Consecutive user messages represent a missing historical answer.
        history = [{"role": "user", "content": f"Earlier question {index}"} for index in range(9)]
        client = Mock()
        client.generate.side_effect = ['{"query":"What does Ada Lovelace do?"}', '{"standalone_query":"What does Ada Lovelace do?","referent":null}']
        for strategy in ("rewrite", "structured"):
            result = build_query("What does she do?", history, strategy, client=client, mode="ollama")
            self.assertIsNone(result["error"])
            payload = json.loads(client.generate.call_args.args[0].split("\n", 1)[1])
            self.assertEqual(payload["history"], history)
            self.assertEqual(result["trace"]["visible_history_messages"], 9)

    def test_api_failure_is_counted_and_visibly_falls_back(self):
        result, _ = self.plan(responses=[OllamaError("timeout")])
        self.assertEqual(result["source"], "heuristic_fallback")
        self.assertIn("timeout", result["error"])
        self.assertEqual(result["trace"]["llm_calls"], 1)
        self.assertTrue(result["trace"]["fallback"])

    def test_empty_and_invalid_json_can_repair_without_indexerror(self):
        for initial in ("", "not JSON", '{"standalone_query":""}'):
            with self.subTest(initial=initial):
                result, _ = self.plan(strategy="structured", responses=[initial, '{"standalone_query":"When did Ada Lovelace write the notes?"}'])
                self.assertEqual(result["source"], "llm_structured")
                self.assertTrue(result["trace"]["repaired"])
                self.assertEqual(result["trace"]["llm_calls"], 2)
                self.assertIsNone(result["error"])
        self.assertEqual(clean_query(""), "")
        self.assertEqual(clean_query(None), "")

    def test_invalid_repair_stops_after_two_calls_and_falls_back(self):
        result, client = self.plan(strategy="structured", responses=["not JSON", "not JSON"])
        self.assertEqual(result["source"], "heuristic_fallback")
        self.assertEqual(client.generate.call_count, 2)
        self.assertTrue(result["trace"]["fallback"])

    def test_unseen_entity_is_not_injected_and_substring_name_is_rejected(self):
        response = '{"standalone_query":"When were the notes written?","referent":"Albert Einstein"}'
        result, _ = self.plan(strategy="structured", responses=[response])
        self.assertNotIn("Einstein", result["query"])
        self.assertIsNone(result["trace"]["referent_verified"])
        self.assertIsNone(verified_referent("Ann", [{"role":"user", "content":"Joanne wrote notes."}]))

    def test_candidate_hallucinated_referent_must_repair_or_fall_back(self):
        bad = '{"standalone_query":"When did Albert Einstein write the notes?","referent":"Albert Einstein"}'
        result, _ = self.plan(strategy="structured", responses=[bad, '{"standalone_query":"When did Ada Lovelace write the notes?"}'])
        self.assertNotIn("Einstein", result["query"])
        self.assertTrue(result["trace"]["repaired"])
        self.assertIn("model_referent_not_supported_by_visible_history", result["trace"]["draft_validation_issues"])

    def test_different_current_answer_type_triggers_repair(self):
        bad = '{"standalone_query":"Where did Ada Lovelace write the notes?","referent":"Ada Lovelace"}'
        result, _ = self.plan(strategy="structured", responses=[bad, '{"standalone_query":"When did Ada Lovelace write the notes?"}'])
        self.assertTrue(result["trace"]["repaired"])
        self.assertIn("current_answer_type_changed", result["trace"]["draft_validation_issues"])

    def test_numeric_and_negation_constraints_survive(self):
        question = "Which two books did she not publish in 1843?"
        missing = '{"query":"Which books did Ada Lovelace publish?"}'
        result, _ = self.plan(question=question, responses=[missing])
        self.assertEqual(result["source"], "heuristic_fallback")
        self.assertIn("current_numeric_constraint_missing", result["trace"]["validation_issues"])
        self.assertIn("current_negation_missing", result["trace"]["validation_issues"])
        self.assertEqual(validation_issues("Which 2 books did Ada Lovelace not publish in 1843?", question, self.history), [])
        self.assertEqual(validation_issues("图书馆能借2小时吗？", "能借2小时吗？", []), [])

    def test_short_chinese_query_is_valid(self):
        client = Mock()
        client.generate.return_value = '{"query":"图书馆在哪？"}'
        result = build_query("在哪？", [{"role":"user","content":"我想去图书馆。"}], "rewrite", client=client, mode="ollama")
        self.assertEqual(result["source"], "llm")

    def test_label_like_config_values_never_enter_model_prompt(self):
        result, client = self.plan(config={"reference_answer":"GOLD_CANARY", "human_rewrite":"REWRITE_CANARY", "positive_ids":["GOLD_PID"]})
        self.assertIsNone(result["error"])
        prompt = client.generate.call_args.args[0]
        for canary in ("GOLD_CANARY", "REWRITE_CANARY", "GOLD_PID"):
            self.assertNotIn(canary, prompt)


if __name__ == "__main__":
    unittest.main()
