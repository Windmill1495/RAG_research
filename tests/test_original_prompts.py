"""Original-template parity and original output contracts."""
import json
from pathlib import Path
import unittest
from unittest.mock import Mock

from rag.compcqr_prompts import original_rewrite_prompt, followup_prompt, selection_prompt, parse_rar
from rag.compcqr import build_plan, retrieve_plan, parse_followup_response
from rag.run_cache import code_hashes


class OriginalPromptTests(unittest.TestCase):
    def test_real_model_missing_comma_preserves_both_values_exactly(self):
        raw = ('"followup_query": "What was the name of the band\'s first album?"\n'
               '"next_answer": "The band\'s first album was titled \'The Night We Met,\' released in 2008."')
        for text in (raw, '{' + raw + '}', '```json\n' + raw + '\n```'):
            obj, repaired = parse_followup_response(text)
            self.assertTrue(repaired)
            self.assertEqual(obj['followup_query'], "What was the name of the band's first album?")
            self.assertEqual(obj['next_answer'], "The band's first album was titled 'The Night We Met,' released in 2008.")

    def test_format_parser_never_fills_missing_or_truncated_text(self):
        for raw in ('"followup_query": "Who?"', '"followup_query": "Who?" "next_answer": "unfinished',
                    '"followup_query": "Who?" "unexpected": "X"',
                    '"followup_query": "Who?" "followup_query": "Where?" "next_answer": "X"'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_followup_response(raw)

    def test_rar_matches_independently_rendered_upstream_fixture(self):
        root = Path(__file__).resolve().parents[1]
        expected = (root / 'tests/fixtures/compcqr_rar_expected.txt').read_text(encoding='utf-8')
        self.assertEqual(original_rewrite_prompt('When did she write them?', [
            {'role':'user','content':'Tell me about Ada Lovelace.'},
            {'role':'assistant','content':'She wrote notes.'}]), expected)
        self.assertEqual(parse_rar('Rewrite: The context names Ada. So the question should be rewritten as: When did Ada write them?\nResponse: A hypothesis.'),
                         ['The context names Ada. ', 'When did Ada write them?', 'A hypothesis.'])

    def test_followup_accepts_original_unbraced_fields_and_has_no_added_word_cap(self):
        client = Mock()
        client.generate.side_effect = [
            'Rewrite: The question is complete. So the question should be rewritten as: Who wrote the notes?\nResponse: A hypothetical author.',
            '"followup_query": "Which notes did the author write?", "next_answer": "Analytical Engine notes."']
        result = build_plan('Who wrote the notes?', [], client=client, mode='ollama', config={})
        self.assertIsNone(result['error'])
        self.assertIn('Which notes', result['components']['mq2'])
        self.assertNotIn('80 words', followup_prompt('Who?', []))
        self.assertIn('If no follow-up is needed, return an empty string.', followup_prompt('Who?', []))

    def test_selector_keeps_all_original_analysis_fields_and_assets_are_signed(self):
        client, retriever = Mock(), Mock()
        hit={'id':'x','passage_id':'p1','text':'Actual evidence.'}
        retriever.search_many.return_value=[[hit]]
        client.generate.return_value=json.dumps({'query analysis':'Question analysis', 'passage analysis':'Evidence analysis',
            'reason for selection':'Supports it', 'selected_passage_number':'1'})
        value=retrieve_plan('Who?', [], ['Who?'], retriever=retriever, client=client, mode='ollama',config={},trace={})
        self.assertEqual(value['trace']['passage_analysis'],'Evidence analysis')
        self.assertEqual(value['trace']['selection_reason'],'Supports it')
        prompt=client.generate.call_args.args[0]
        self.assertIn('Step 1: List up',prompt)
        self.assertIn('Step 2: Analyze whether each passage',prompt)
        self.assertIn('Step 3: Choose the passage',prompt)
        self.assertEqual(client.generate.call_args.kwargs['system'],'')
        signed=code_hashes(Path(__file__).resolve().parents[1])
        self.assertIn('prompts/compcqr/llm4cs_demonstrations.json',signed)


if __name__ == '__main__':
    unittest.main()
