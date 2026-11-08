"""CompCQR invariants: query composition, passage fusion, and honest failures."""
import json
import unittest
from unittest.mock import Mock

from rag.compcqr import build_plan, compose_queries, fuse_rankings, retrieve_plan
from rag.llm import OllamaError
from rag.rewriters import AB_STRATEGIES, RESEARCH_STRATEGIES, build_query


def passage(pid):
    return {"id": "chunk_" + pid, "passage_id": pid, "text": "Evidence for " + pid,
            "document": pid, "title": "Example", "score": 99.0}


class CompCQRTests(unittest.TestCase):
    def setUp(self):
        self.question = "When did she write the notes?"
        self.history = [{"role": "user", "content": "Tell me about Ada Lovelace."},
                        {"role": "assistant", "content": "Ada Lovelace wrote Analytical Engine notes."}]
        self.answers = [
            json.dumps({"query": "When did Ada Lovelace write the notes?",
                        "answer": "HYPOTHETICAL_1843"}),
            json.dumps({"followup_query": "Which Analytical Engine notes did Ada Lovelace write?",
                        "next_answer": "HYPOTHETICAL_TRANSLATION"}),
        ]
        self.config = {"retriever_backend": "hybrid", "compcqr_retrieval_depth": 100,
                       "compcqr_rrf_k": 0, "compcqr_selection_top_k": 5,
                       "compcqr_passage_chars": 700, "compcqr_selection_char_budget": 6000,
                       "compcqr_select": True}

    def plan(self, responses=None, mode="ollama", config=None):
        client = Mock()
        client.generate.side_effect = self.answers if responses is None else responses
        result = build_query(self.question, self.history, "compcqr", client=client,
                             mode=mode, config=config or self.config)
        return result, client

    def test_325_ordered_queries_and_31_bm25_subsets(self):
        components = dict(zip(("oq", "mq", "ma", "mq2", "ma2"), ("a", "b", "c", "d", "e")))
        ordered = compose_queries(components)
        sparse = compose_queries(components, "bm25")
        self.assertEqual(len(ordered), 325)
        self.assertEqual(len(set(ordered)), 325)
        self.assertEqual(len(sparse), 31)
        self.assertIn("b\na", ordered)
        self.assertNotIn("b\na", sparse)
        self.assertEqual(ordered, compose_queries(components))

    def test_absent_and_duplicate_atoms_do_not_multiply_votes(self):
        components = {"oq": "a", "mq": "a", "ma": "b", "mq2": "", "ma2": ""}
        self.assertEqual(compose_queries(components), ["a", "b", "a\nb", "b\na"])
        self.assertEqual(compose_queries(components, max_components=1), ["a", "b"])

    def test_dispatch_preserves_ab_and_records_two_calls(self):
        result, client = self.plan()
        self.assertEqual(AB_STRATEGIES, ("rewrite", "structured"))
        self.assertIn("compcqr", RESEARCH_STRATEGIES)
        self.assertEqual(result["source"], "compcqr_llm")
        self.assertEqual(result["trace"]["llm_calls"], 2)
        self.assertEqual(result["trace"]["query_count"], 325)
        self.assertEqual(client.generate.call_count, 2)
        self.assertIsNone(result["error"])
        first_payload = json.loads(client.generate.call_args_list[0].args[0].split("\n", 1)[1])
        self.assertEqual(first_payload, {"history": self.history, "current_question": self.question})
        self.assertFalse(result["trace"]["hypothetical_answers_are_evidence"])

    def test_no_history_still_uses_component_method(self):
        client = Mock()
        client.generate.side_effect = [
            '{"query":"Who wrote the notes?","answer":"An author."}',
            '{"followup_query":"","next_answer":""}',
        ]
        result = build_query("Who wrote the notes?", [], "compcqr", client=client, mode="ollama")
        self.assertEqual(client.generate.call_count, 2)
        self.assertEqual(result["source"], "compcqr_llm")
        self.assertEqual(result["components"]["mq2"], "")

    def test_first_failure_is_visible_and_no_second_call(self):
        result, client = self.plan([OllamaError("timeout")])
        self.assertEqual(client.generate.call_count, 1)
        self.assertEqual(result["source"], "compcqr_fallback")
        self.assertIn("timeout", result["error"])
        self.assertTrue(result["trace"]["fallback"])
        self.assertEqual(result["components"]["ma"], "")
        self.assertEqual(result["trace"]["stage_errors"][0]["stage"], "rewrite_answer")

    def test_followup_failure_retains_first_pair_and_marks_partial(self):
        result, _ = self.plan([self.answers[0], "invalid JSON"])
        self.assertEqual(result["source"], "compcqr_partial")
        self.assertEqual(result["components"]["ma"], "HYPOTHETICAL_1843")
        self.assertEqual(result["trace"]["query_count"], 15)
        self.assertTrue(result["trace"]["component_errors"])

    def test_empty_followup_is_explicitly_allowed(self):
        for response in ('{"followup_query":"","next_answer":""}', '""'):
            with self.subTest(response=response):
                result, _ = self.plan([self.answers[0], response])
                self.assertEqual(result["source"], "compcqr_llm")
                self.assertFalse(result["trace"]["fallback"])
                self.assertEqual(result["trace"]["query_count"], 15)

    def test_demo_never_calls_llm_or_creates_hypothetical_answer(self):
        result, client = self.plan(mode="demo")
        client.generate.assert_not_called()
        self.assertEqual(result["source"], "compcqr_demo")
        self.assertEqual(result["trace"]["llm_calls"], 0)
        self.assertEqual(result["components"]["ma"], "")
        self.assertEqual(result["components"]["ma2"], "")

    def test_rrf_combines_ranks_not_incompatible_scores(self):
        a, b, c = passage("a"), passage("b"), passage("c")
        a["score"] = -999
        b["score"] = 999
        result = fuse_rankings([[a, b], [a, c]])
        self.assertEqual([hit["passage_id"] for hit in result], ["a", "b", "c"])
        self.assertEqual(result[0]["rrf_score"], 2)
        self.assertEqual(result[0]["query_frequency"], 2)
        self.assertEqual(result[1]["rrf_score"], 0.5)

    def test_duplicate_passage_in_one_ranking_counts_once_at_original_rank(self):
        a, b = passage("a"), passage("b")
        result = fuse_rankings([[a, a, b]])
        self.assertEqual(result[0]["rrf_score"], 1)
        self.assertEqual(result[0]["query_frequency"], 1)
        self.assertAlmostEqual(result[1]["rrf_score"], 1 / 3)

    def test_fusion_rejects_invalid_rrf_constant(self):
        for value in (-1, float("nan"), float("inf"), True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                fuse_rankings([[passage("a")]], value)

    def test_selector_promotes_one_preserving_other_relative_order(self):
        plan, client = self.plan()
        client.generate.side_effect = ['{"selected_passage_number":3,"reason":"contains answer"}']
        retriever = Mock()
        retriever.search_many.return_value = [[passage("a"), passage("b"), passage("c")]]
        result = retrieve_plan(self.question, self.history, plan["queries"], retriever=retriever,
                               client=client, mode="ollama", config=self.config, trace=plan["trace"])
        self.assertEqual([hit["passage_id"] for hit in result["hits"]], ["c", "a", "b"])
        self.assertEqual(result["trace"]["selected_passage_id"], "c")
        self.assertEqual(result["trace"]["llm_calls"], 3)
        selection_prompt = client.generate.call_args.args[0]
        self.assertNotIn("HYPOTHETICAL_1843", selection_prompt)
        self.assertNotIn("HYPOTHETICAL_TRANSLATION", selection_prompt)
        payload = json.loads(selection_prompt.split("\n", 1)[1])
        self.assertEqual(payload["history"], self.history)
        self.assertEqual(set(payload), {"current_question", "history", "passages"})

    def test_invalid_or_out_of_range_selection_keeps_rrf_with_error(self):
        for number in (True, 0, 3, "1.0", None):
            with self.subTest(number=number):
                client, retriever = Mock(), Mock()
                client.generate.return_value = json.dumps({"selected_passage_number": number})
                retriever.search_many.return_value = [[passage("a"), passage("b")]]
                result = retrieve_plan(self.question, self.history, ["query"], retriever=retriever,
                                       client=client, mode="ollama", config=self.config, trace={})
                self.assertEqual([hit["passage_id"] for hit in result["hits"]], ["a", "b"])
                self.assertTrue(result["trace"]["fallback"])
                self.assertTrue(result["trace"]["selection_error"])
                self.assertFalse(result["trace"]["selection_applied"])

    def test_budget_and_disabled_selection_are_visible(self):
        client, retriever = Mock(), Mock()
        retriever.search_many.return_value = [[passage("a"), passage("b")]]
        for options, reason in (({"compcqr_select": False}, "disabled_by_config"),
                                ({"compcqr_selection_char_budget": 2}, "no_candidates_within_budget")):
            result = retrieve_plan(self.question, self.history, ["query"], retriever=retriever,
                                   client=client, mode="ollama", config={**self.config, **options}, trace={})
            self.assertEqual(result["trace"]["selection_skipped"], reason)
        client.generate.assert_not_called()

    def test_full_prompt_budget_includes_original_history_without_dropping_it(self):
        client, retriever = Mock(), Mock()
        retriever.search_many.return_value = [[passage("a"), passage("b")]]
        long_history = [{"role": "user", "content": "x" * 7000}]
        result = retrieve_plan(self.question, long_history, ["query"], retriever=retriever,
                               client=client, mode="ollama", config=self.config, trace={"llm_calls": 2})
        client.generate.assert_not_called()
        self.assertEqual(result["trace"]["llm_calls"], 2)
        self.assertTrue(result["trace"]["selection_error"])
        self.assertTrue(result["trace"]["fallback"])
        client.generate.return_value = '{"selected_passage_number":1}'
        result = retrieve_plan(self.question, self.history, ["query"], retriever=retriever,
                               client=client, mode="ollama", config=self.config, trace={"llm_calls": 2})
        actual = client.generate.call_args.args[0]
        self.assertLessEqual(len(actual), self.config["compcqr_selection_char_budget"])
        payload = json.loads(actual.split("\n", 1)[1])
        self.assertEqual(payload["history"], self.history)

    def test_selection_cannot_promote_unsupplied_passage_even_if_it_is_gold(self):
        client, retriever = Mock(), Mock()
        retriever.search_many.return_value = [[passage("a"), passage("gold")]]
        client.generate.return_value = '{"selected_passage_number":2}'
        result = retrieve_plan(self.question, self.history, ["query"], retriever=retriever,
                               client=client, mode="ollama",
                               config={**self.config, "compcqr_selection_top_k": 1}, trace={})
        self.assertEqual(result["hits"][0]["passage_id"], "a")
        self.assertIsNone(result["trace"]["selected_passage_id"])
        self.assertTrue(result["trace"]["selection_error"])
        self.assertNotIn('"passage_id": "gold"', client.generate.call_args.args[0])

    def test_retrieval_failure_is_not_hidden_as_success(self):
        retriever = Mock()
        retriever.search_many.side_effect = OllamaError("embedding unavailable")
        with self.assertRaisesRegex(OllamaError, "embedding unavailable"):
            retrieve_plan(self.question, self.history, ["query"], retriever=retriever,
                          client=Mock(), mode="ollama", config=self.config, trace={})


if __name__ == "__main__":
    unittest.main()
