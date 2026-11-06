"""Retrieval invariants, cache invalidation, and semantic-backend boundaries."""

import json
import random
from pathlib import Path
import tempfile
import unittest

from rag.retrieval import INDEX_VERSION, Retriever, chunk_text, tokenize


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.docs = self.root / "docs"
        self.docs.mkdir()
        self.index = self.root / "cache" / "index.json"

    def tearDown(self):
        self.temporary.cleanup()

    def write(self, name, text):
        path = self.docs / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def retriever(self, **kwargs):
        return Retriever(self.docs, self.index, **kwargs)

    def test_tokenizer_handles_chinese_single_characters_and_english(self):
        tokens = tokenize("猫与狗 RAG Embedding_Model")
        self.assertIn("猫", tokens)
        self.assertIn("猫与", tokens)
        self.assertIn("rag", tokens)
        self.assertIn("embedding_model", tokens)
        self.assertNotIn("狗r", tokens)

    def test_chunking_preserves_offsets_and_covers_all_nonwhitespace(self):
        text = "  第一段介绍检索增强生成。\n\n" + "连续追问需要结合历史。" * 18 + "\n 最后一段。 "
        chunks = chunk_text(text, chunk_size=43, overlap=9)
        self.assertGreater(len(chunks), 2)
        covered = set()
        for content, start, end in chunks:
            self.assertEqual(content, text[start:end])
            self.assertLessEqual(len(content), 43)
            covered.update(range(start, end))
        self.assertTrue(all(index in covered for index, char in enumerate(text) if not char.isspace()))
        self.assertTrue(chunks[-1][0].endswith("最后一段。"))

    def test_chunking_strict_limit_without_sentence_boundaries(self):
        chunks = chunk_text("甲" * 101, 20, 5)
        self.assertEqual(chunks[0], ("甲" * 20, 0, 20))
        self.assertEqual(chunks[-1][2], 101)
        self.assertTrue(all(len(text) <= 20 for text, _, _ in chunks))

    def test_invalid_chunking_configuration(self):
        for chunk_size, overlap in [(0, 0), (10, -1), (10, 10), (True, 0), (10, 1.5)]:
            with self.subTest(chunk_size=chunk_size, overlap=overlap):
                with self.assertRaises(ValueError):
                    chunk_text("", chunk_size, overlap)

    def test_bm25_ranks_chinese_evidence_and_has_no_unknown_results(self):
        self.write("library.md", "# 图书馆\n图书馆借阅期限为三十天，每人最多借阅十本图书。")
        self.write("sports.txt", "运动场开放时间为早上七点至晚上十点。")
        retriever = self.retriever()
        result = retriever.search("图书馆借阅期限是多少？", top_k=1)
        self.assertEqual(result[0]["document"], "library.md")
        self.assertIn("三十天", result[0]["text"])
        self.assertGreater(result[0]["score"], 0)
        self.assertEqual(retriever.search("zxqvunmatched"), [])

    def test_new_instance_loads_cache_and_detects_content_change(self):
        self.write("rules.md", "旧规定：借阅期限为二十天。")
        first = self.retriever()
        self.assertFalse(first.build()["cache_hit"])
        second = self.retriever()
        self.assertTrue(second.build()["cache_hit"])
        first_ids = [chunk.id for chunk in second.chunks]
        self.write("rules.md", "新规定：借阅期限为三十天。")
        third = self.retriever()
        self.assertFalse(third.build()["cache_hit"])
        self.assertNotEqual(first_ids, [chunk.id for chunk in third.chunks])
        self.assertIn("三十天", third.search("借阅期限")[0]["text"])

    def test_cache_invalidates_on_configuration_and_removed_document(self):
        self.write("a.md", "检索增强生成。" * 20)
        self.write("b.md", "对话查询改写。" * 20)
        self.retriever(chunk_size=100, overlap=10).build()
        changed = self.retriever(chunk_size=80, overlap=10)
        self.assertFalse(changed.build()["cache_hit"])
        (self.docs / "b.md").unlink()
        deleted = self.retriever(chunk_size=80, overlap=10)
        self.assertFalse(deleted.build()["cache_hit"])
        self.assertEqual({chunk.document for chunk in deleted.chunks}, {"a.md"})

    def test_corrupt_index_rebuilds(self):
        self.write("a.txt", "文档索引支持缓存。")
        self.retriever().build()
        self.index.write_text("{incomplete", encoding="utf-8")
        retriever = self.retriever()
        self.assertFalse(retriever.build()["cache_hit"])
        self.assertEqual(json.loads(self.index.read_text(encoding="utf-8"))["version"], INDEX_VERSION)

    def test_relative_document_paths_distinguish_equal_filenames(self):
        self.write("a/rules.md", "图书馆借书。")
        self.write("b/rules.md", "体育馆预约。")
        retriever = self.retriever()
        retriever.build()
        self.assertEqual({chunk.document for chunk in retriever.chunks}, {"a/rules.md", "b/rules.md"})

    def test_utf8_bom_and_windows_newlines(self):
        (self.docs / "rules.txt").write_bytes("图书馆\r\n借阅规则。".encode("utf-8-sig"))
        retriever = self.retriever()
        retriever.build()
        self.assertNotIn("\ufeff", retriever.chunks[0].text)
        self.assertNotIn("\r", retriever.chunks[0].text)

    def test_empty_or_wrong_encoding_corpus_has_clear_error(self):
        with self.assertRaisesRegex(ValueError, "知识库为空"):
            self.retriever().build()
        (self.docs / "bad.txt").write_bytes(b"\xff\xfe\xff")
        with self.assertRaisesRegex(ValueError, "UTF-8"):
            self.retriever().build()

    def test_dense_retrieval_can_match_without_shared_terms_and_caches_vectors(self):
        self.write("cat.md", "猫喜欢晒太阳。")
        self.write("dog.md", "狗喜欢散步。")
        calls = []

        def embed(texts):
            calls.append(list(texts))
            return [[3.0, 0.0] if "猫" in text or "feline" in text else [0.0, 2.0] for text in texts]

        first = self.retriever(backend="dense", embedding_fn=embed)
        first.build()
        self.assertEqual(len(calls), 1)
        second = self.retriever(backend="dense", embedding_fn=embed)
        self.assertTrue(second.build()["cache_hit"])
        self.assertEqual(len(calls), 1, "Loading cached document vectors must not call the service")
        result = second.search("feline", top_k=1)
        self.assertEqual(result[0]["document"], "cat.md")
        self.assertAlmostEqual(result[0]["score"], 1.0)
        self.assertEqual(result[0]["bm25_score"], 0.0)

    def test_hybrid_fuses_lexical_and_dense_agreement(self):
        self.write("cat.md", "猫喜欢晒太阳。")
        self.write("dog.md", "狗喜欢散步。")
        self.write("fish.md", "鱼生活在水中。")

        def embed(texts):
            return [[1, 0] if "猫" in text else [0, 1] for text in texts]

        retriever = self.retriever(backend="hybrid", embedding_fn=embed)
        result = retriever.search("猫", top_k=2)
        self.assertEqual(result[0]["document"], "cat.md")
        self.assertGreater(result[0]["score"], result[1]["score"])
        self.assertIn("dense_score", result[0])

    def test_invalid_embedding_vectors_are_rejected(self):
        self.write("a.txt", "向量检索。")
        for response in [[], [[0, 0]], [[float("nan"), 1]], [[float("inf"), 1]]]:
            with self.subTest(response=response):
                with self.assertRaises(ValueError):
                    self.retriever(backend="dense", embedding_fn=lambda texts: response).build(force=True)

    def test_query_embedding_dimension_mismatch_is_rejected(self):
        self.write("a.txt", "向量检索。")
        retriever = self.retriever(backend="dense", embedding_fn=lambda texts: [[1, 0] for _ in texts])
        retriever.build()
        retriever.embedding_fn = lambda texts: [[1, 0, 0] for _ in texts]
        with self.assertRaisesRegex(ValueError, "维度不同"):
            retriever.search("查询")

    def test_missing_embedding_service_has_clear_error(self):
        self.write("a.txt", "向量检索。")
        with self.assertRaisesRegex(RuntimeError, "Embedding"):
            self.retriever(backend="dense").build()

    def test_search_validates_query_and_top_k(self):
        self.write("a.txt", "检索系统。")
        retriever = self.retriever()
        with self.assertRaises(ValueError):
            retriever.search(" ")
        for top_k in [0, -1, 1.5, True]:
            with self.subTest(top_k=top_k):
                with self.assertRaises(ValueError):
                    retriever.search("检索", top_k)

    def test_external_symlink_not_read(self):
        outside = self.root / "outside.txt"
        outside.write_text("不属于知识库。", encoding="utf-8")
        try:
            (self.docs / "link.txt").symlink_to(outside)
        except OSError:
            self.skipTest("This Windows account cannot create symlinks")
        with self.assertRaisesRegex(ValueError, "链接"):
            self.retriever().build()

    def test_inverted_scores_and_ranks_equal_scan_with_repetition_and_ties(self):
        rng = random.Random(20261102)
        words = ["orbit", "moon", "earth", "solar", "network", "data-center", "图书馆", "检索"]
        for index in range(30):
            self.write(f"{index:02d}.txt", " ".join(rng.choices(words, k=rng.randrange(5, 60))))
        self.write("tie_a.txt", "orbit moon network")
        self.write("tie_b.txt", "orbit moon network")
        inverted = self.retriever(bm25_mode="inverted", chunk_size=150, overlap=20)
        scan = self.retriever(bm25_mode="scan", chunk_size=150, overlap=20)
        inverted.build()
        self.assertTrue(scan.build()["cache_hit"], "The optimization need not rebuild the same corpus index")
        for query in ["orbit", "orbit " * 5, "moon solar data-center", "图书馆检索", "unseen", ""]:
            with self.subTest(query=query):
                self.assertEqual(inverted._bm25_scores(query), scan._bm25_scores(query))
                if query:
                    self.assertEqual(inverted.search(query, 100), scan.search(query, 100))
        hit = inverted.search("orbit moon network", 100)
        tied = [row for row in hit if row["document"] in {"tie_a.txt", "tie_b.txt"}]
        self.assertEqual([row["id"] for row in tied], sorted(row["id"] for row in tied))

    def test_inverted_only_visits_matching_postings_and_preserves_query_frequency(self):
        self.write("target.txt", "raretarget common")
        for index in range(20):
            self.write(f"noise{index}.txt", "common noise")
        retriever = self.retriever(bm25_mode="inverted")
        first = retriever.search("raretarget", 1)[0]["score"]
        repeated = retriever.search("raretarget " * 5, 1)[0]["score"]
        self.assertAlmostEqual(repeated, first * 5, places=7)
        retriever.search("raretarget unseen", 1)
        work = retriever.stats()["last_query_work"]
        self.assertEqual(work["posting_entries_visited"], 1)
        self.assertEqual(work["scored_chunks"], 1)
        retriever.bm25_mode = "scan"
        retriever.search("raretarget unseen", 1)
        self.assertEqual(retriever.stats()["last_query_work"]["term_chunk_pairs_examined"], 42)


class PassageRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.corpus = self.root / "passages.jsonl"
        self.index = self.root / "cache" / "passages.json"
        self.rows = [
            {"id": "http://example.invalid/wiki/Orbit_p0", "text": "orbit " * 120},
            {"id": "http://example.invalid/wiki/Library_p0", "text": "图书馆可预约研讨室。"},
        ]
        self.write_rows(self.rows)

    def tearDown(self):
        self.temp.cleanup()

    def write_rows(self, rows):
        self.corpus.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

    def retriever(self, **kwargs):
        return Retriever(self.corpus, self.index, corpus_format="passages_jsonl", **kwargs)

    def test_preserves_original_id_and_does_not_rechunk_long_passage(self):
        retriever = self.retriever(chunk_size=30, overlap=5)
        stats = retriever.build()
        self.assertEqual(stats["documents"], 2)
        self.assertEqual(stats["chunks"], 2)
        self.assertEqual(stats["corpus_format"], "passages_jsonl")
        result = retriever.search("orbit", 1)[0]
        self.assertEqual(result["passage_id"], self.rows[0]["id"])
        self.assertEqual(result["document"], self.rows[0]["id"])
        self.assertEqual(result["text"], self.rows[0]["text"])
        self.assertEqual((result["start"], result["end"]), (0, len(self.rows[0]["text"])))
        self.assertGreater(len(result["text"]), 500)
        cached = self.retriever(chunk_size=30, overlap=5)
        self.assertTrue(cached.build()["cache_hit"], "Long pre-segmented passages must pass cache validation")
        self.assertEqual(cached.search("orbit", 1), retriever.search("orbit", 1))

    def test_file_bytes_and_passage_content_invalidate_cache(self):
        first = self.retriever()
        first.build()
        initial_stats = first.stats()
        self.corpus.write_text(self.corpus.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        changed_bytes = self.retriever()
        self.assertFalse(changed_bytes.build()["cache_hit"])
        self.assertNotEqual(initial_stats["corpus_sha256"], changed_bytes.stats()["corpus_sha256"])
        self.assertNotEqual(initial_stats["signature"], changed_bytes.stats()["signature"])
        self.rows[0]["text"] = "New orbit policy"
        self.write_rows(self.rows)
        changed_content = self.retriever()
        self.assertFalse(changed_content.build()["cache_hit"])
        self.assertEqual(changed_content.search("orbit", 1)[0]["text"], "New orbit policy")

    def test_rejects_duplicate_ids_empty_and_invalid_rows(self):
        cases = [
            ([self.rows[0], dict(self.rows[0])], "重复 id"),
            ([{"id": " ", "text": "a"}], "id 必须"),
            ([{"id": " a", "text": "a"}], "id 必须"),
            ([{"id": 1, "text": "a"}], "id 必须"),
            ([{"id": "a", "text": " \t"}], "text 必须"),
            ([{"id": "a", "text": 2}], "text 必须"),
            ([{"id": "a", "contents": "a"}], "id、text"),
            ([["a", "text"]], "id、text"),
            ([], "知识库为空"),
        ]
        for rows, message in cases:
            with self.subTest(rows=rows):
                self.write_rows(rows)
                with self.assertRaisesRegex(ValueError, message):
                    self.retriever().build()
        self.corpus.write_text('{"id":"a", broken}\n', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "第 1 行不是有效 JSON"):
            self.retriever().build()
        self.corpus.write_bytes(b"\xff\xfe")
        with self.assertRaisesRegex(ValueError, "UTF-8"):
            self.retriever().build()

    def test_cannot_reuse_passage_cache_without_original_ids(self):
        self.retriever().build()
        saved = json.loads(self.index.read_text(encoding="utf-8"))
        saved["chunks"][0].pop("passage_id")
        self.index.write_text(json.dumps(saved), encoding="utf-8")
        rebuilt = self.retriever()
        self.assertFalse(rebuilt.build()["cache_hit"])
        self.assertEqual(rebuilt.search("orbit", 1)[0]["passage_id"], self.rows[0]["id"])

    def test_cache_cannot_attach_wrong_text_to_valid_passage_id(self):
        self.retriever().build()
        saved = json.loads(self.index.read_text(encoding="utf-8"))
        saved["chunks"][0]["text"] = "x" * len(saved["chunks"][0]["text"])
        self.index.write_text(json.dumps(saved), encoding="utf-8")
        rebuilt = self.retriever()
        self.assertFalse(rebuilt.build()["cache_hit"])
        self.assertEqual(rebuilt.search("orbit", 1)[0]["text"], self.rows[0]["text"])

    def test_passage_dense_reuses_cached_vectors(self):
        calls = []
        def embed(texts):
            calls.append(list(texts))
            return [[1, 0] if "orbit" in text else [0, 1] for text in texts]
        first = self.retriever(backend="dense", embedding_fn=embed)
        first.build()
        cached = self.retriever(backend="dense", embedding_fn=embed)
        self.assertTrue(cached.build()["cache_hit"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(cached.search("orbit", 1)[0]["passage_id"], self.rows[0]["id"])

    def test_utf8_bom_and_normalized_offsets_for_passages(self):
        row = {"id": "p1", "text": "a\r\nb\r c"}
        self.corpus.write_bytes((json.dumps(row) + "\r\n").encode("utf-8-sig"))
        retriever = self.retriever()
        retriever.build()
        chunk = retriever.chunks[0]
        self.assertEqual(chunk.text, "a\nb\n c")
        self.assertEqual(chunk.end, len(chunk.text))

    def test_bad_modes_and_missing_corpus_fail_clearly(self):
        for kwargs in [{"corpus_format": "csv"}, {"bm25_mode": "anything"}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Retriever(self.corpus, self.index, **kwargs)
        self.corpus.unlink()
        with self.assertRaisesRegex(ValueError, "JSONL 文件不存在"):
            self.retriever().build()


if __name__ == "__main__":
    unittest.main()
