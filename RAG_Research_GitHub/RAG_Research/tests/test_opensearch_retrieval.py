"""Server retrieval contracts; no GPU or remote index is claimed by these mocks."""
import json
import unittest
from unittest.mock import Mock, patch

from rag.llm import EMBEDDING_INPUT_POLICY
from rag import official_index
from rag.opensearch_retrieval import OpenSearchRetriever, SearchServiceError, document_key, vector_mapping


def config():
    return {"retriever_backend": "hybrid", "embedding_model": "bge-m3", "embedding_digest": "digest",
            "opensearch_dimension": 2, "hybrid_candidate_depth": 400, "opensearch_msearch_batch": 2}


def hit(pid, text="Evidence", score=1):
    return {"_source": {"passage_id": pid, "text": text, "title": pid}, "_score": score}


def response(hits):
    return {"timed_out": False, "_shards": {"total": 2, "successful": 2, "failed": 0}, "hits": {"hits": hits}}


def mapping(c, **updates):
    meta = {"complete": True, "expected_documents": 3, "embedding_model": "bge-m3",
            "embedding_digest": "digest", "embedding_input_policy": EMBEDDING_INPUT_POLICY,
            "dimensions": 2, "corpus_fingerprint": "corpus", "source_identity": {"files": ["official.zip"]},
            "index_schema": "qrecc-opensearch-hybrid-v1", "corpus_scope": "diagnostic_source",
            "vector_config": vector_mapping(c)}
    meta.update(updates)
    return {"qrecc-full": {"mappings": {"_meta": {"qrecc": meta}, "properties": {"vector": vector_mapping(c)}}}}


class OpenSearchRetrieverTests(unittest.TestCase):
    def ready(self):
        r = OpenSearchRetriever({**config(), "evaluation_scope": "pilot", "allow_diagnostic_index": True},
                                Mock(return_value=[[1, 0], [0, 1]]))
        r._request = Mock(side_effect=[mapping(config()), {"count": 3},
            {"qrecc-full": {"settings": {"index": {"uuid": "index-uuid", "blocks": {"write": "true"}}}}}])
        r.build()
        return r

    def test_build_validates_complete_count_weights_uuid_and_readonly(self):
        r = self.ready()
        self.assertEqual(r.index_identity()["indexed_passage_count"], 3)
        self.assertEqual(r.index_identity()["uuid"], "index-uuid")
        self.assertEqual(r.stats()["hybrid_candidate_depth"], 400)
        with self.assertRaises(ValueError):
            r.build(force=True)
        for change in ({"complete": False}, {"embedding_digest": "wrong"}, {"expected_documents": 2},
                       {"corpus_scope": "official_qrecc_full"}):
            r = OpenSearchRetriever({**config(), "evaluation_scope": "pilot", "allow_diagnostic_index": True}, Mock())
            r._request = Mock(side_effect=[mapping(config(), **change), {"count": 3}])
            with self.subTest(change=change), self.assertRaises(SearchServiceError):
                r.build()

    def test_index_embedding_cpu_budget_must_match_current_configuration(self):
        c = {**config(), 'ollama_num_thread': 2, 'ollama_num_gpu': 0,
             'evaluation_scope': 'pilot', 'allow_diagnostic_index': True}
        r = OpenSearchRetriever(c, Mock())
        r._request = Mock(return_value=mapping(c, embedding_runtime_options={'num_thread': 4, 'num_gpu': 0}))
        with self.assertRaisesRegex(SearchServiceError, 'runtime options'):
            r.build()

    def test_batched_two_route_queries_keep_original_ids_and_fixed_depth(self):
        r = self.ready()
        r._request = Mock(return_value={"responses": [response([hit("a", score=100), hit("b")]),
            response([hit("b", score=-10), hit("a")]), response([hit("c")]), response([hit("c")])]})
        result = r.search_many(["first", "second", "first"], 3)
        self.assertEqual([x["passage_id"] for x in result[0]], ["a", "b"])
        self.assertEqual(result[2], result[0])
        self.assertEqual(result[1][0]["passage_id"], "c")
        self.assertEqual(r.embedding_fn.call_args.args[0], ["first", "second"])
        body = r._request.call_args.args[2]
        self.assertTrue(body.endswith("\n"))
        lines = [json.loads(s) for s in body.splitlines()]
        self.assertEqual(lines[1]["size"], 400)
        self.assertEqual(lines[3]["query"]["knn"]["vector"]["k"], 400)
        self.assertEqual(lines[3]["_source"], ["passage_id", "text", "title"])
        self.assertEqual(lines[0]["search_type"], "dfs_query_then_fetch")

    def test_timeout_failed_shard_and_missing_responses_are_errors(self):
        for value in ({"responses": []}, {"responses": [{"timed_out": True}, response([])]},
                      {"responses": [{"_shards": {"total": 2, "successful": 1, "failed": 1}, "hits": {"hits": []}}, response([])]}):
            r = self.ready()
            r.embedding_fn.return_value = [[1, 0]]
            r._request = Mock(return_value=value)
            with self.subTest(value=value), self.assertRaises(SearchServiceError):
                r.search("query")

    def test_coverage_queries_only_gold_ids_and_checks_original_id(self):
        r = self.ready()
        r._request = Mock(return_value={"docs": [{"found": True, "_source": {"passage_id": "gold"}}, {"found": False}]})
        self.assertEqual(r.contains_passage_ids(["gold", "missing"]), {"gold"})
        self.assertEqual(r._request.call_args.args[2], {"docs": [
            {"_id": document_key("gold"), "_source": ["passage_id"]},
            {"_id": document_key("missing"), "_source": ["passage_id"]}]})
        self.assertEqual(len(document_key("https://example.com/" + "x" * 5000)), 64)
        r._request.return_value = {"docs": [{"found": True, "_source": {"passage_id": "wrong"}}]}
        with self.assertRaises(SearchServiceError):
            r.contains_passage_ids(["gold"])

    def test_bm25_reads_lexical_scores_only_and_dedupes_variants(self):
        r = OpenSearchRetriever({**config(), "retriever_backend": "bm25"},
                                Mock(side_effect=AssertionError("BM25 called embeddings")))
        r._built = True
        r._meta = {"document_key_policy": official_index.VARIANT_POLICY}
        r._request = Mock(return_value={"responses": [
            response([hit("a", "first", 5), hit("a", "variant", 4), hit("b", score=3)])]})
        result = r.search_many(["orchids", "orchids"], 2)
        self.assertEqual(result[0], result[1])
        self.assertEqual([(item["passage_id"], item["score"]) for item in result[0]], [("a", 5), ("b", 3)])
        body = r._request.call_args.args[2].splitlines()
        self.assertEqual(len(body), 2)
        self.assertEqual(json.loads(body[1])["query"], {"match": {"text": {"query": "orchids"}}})
        r.embedding_fn.assert_not_called()

    def test_scalar_top3_and_top100_have_identical_prefix(self):
        r = self.ready()
        r.embedding_fn.return_value = [[1, 0]]
        r._request = Mock(return_value={"responses": [response([hit(str(i)) for i in range(30)]),
            response([hit(str(i)) for i in range(30, 0, -1)])]})
        self.assertEqual(r.search("q", 3), r.search("q", 100)[:3])

    def test_variant_coverage_and_fusion_use_original_id(self):
        r = self.ready()
        r._meta["document_key_policy"] = official_index.VARIANT_POLICY
        r._request = Mock(return_value={**response([]), "aggregations": {"covered": {
            "buckets": [{"key": document_key("gold"), "doc_count": 2}], "sum_other_doc_count": 0}}})
        self.assertEqual(r.contains_passage_ids(["gold", "missing"]), {"gold"})
        self.assertTrue(r._request.call_args.args[1].endswith("/_search"))
        hits = r._fuse([hit("gold", "first"), hit("gold", "second")], [hit("gold", "second")], 3, variants=True)
        self.assertEqual([(h["passage_id"], h["text"]) for h in hits], [("gold", "first")])
        self.assertAlmostEqual(hits[0]["score"], 2 / 61)

    def test_embedding_and_http_batches_are_bounded_for_many_queries(self):
        r = self.ready()
        r.embedding_fn.side_effect = lambda texts: [[1, 0] for _ in texts]
        r._request = Mock(side_effect=lambda method, path, payload, **kwargs: {
            "responses": [response([hit("evidence")]) for _ in range(len(payload.splitlines()) // 2)]})
        self.assertEqual(len(r.search_many(["q" + str(i) for i in range(33)], 3)), 33)
        self.assertEqual([len(call.args[0]) for call in r.embedding_fn.call_args_list], [16, 16, 1])
        self.assertEqual(r._request.call_count, 17)
        self.assertTrue(all(len(call.args[2].splitlines()) <= 8 for call in r._request.call_args_list))


if __name__ == "__main__":
    unittest.main()
