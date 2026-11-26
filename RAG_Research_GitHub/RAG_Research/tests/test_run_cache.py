"""Signed resume preserves fixed samples and refuses cross-experiment mixing."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rag.run_cache import RunCache, code_hashes, make_signature


class RunCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.output = self.root / "run"
        self.identity = {"task":"retrieval", "model":"qwen2.5:3b", "dataset_sha":"aaa", "prompt_sha":"bbb", "history_turns":6}

    def tearDown(self):
        self.temporary.cleanup()

    def test_new_resume_retains_failed_samples_as_completed(self):
        cache = RunCache(self.output, self.identity)
        cache.append({"id":"1", "strategy":"rewrite", "error":"timeout", "hit_at_k":0.0}, "rewrite:1")
        restored = RunCache(self.output, self.identity, resume=True)
        self.assertEqual(restored.completed_keys, {"rewrite:1"})
        self.assertEqual(restored.rows[0]["error"], "timeout")
        with self.assertRaisesRegex(ValueError, "已有记录"):
            restored.append({"error":None}, "rewrite:1")

    def test_changed_model_data_prompt_or_history_refuses_resume(self):
        RunCache(self.output, self.identity)
        for key in ("model", "dataset_sha", "prompt_sha", "history_turns"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "签名不匹配"):
                RunCache(self.output, {**self.identity, key:"changed"}, resume=True)

    def test_partial_unterminated_last_write_is_recovered(self):
        cache = RunCache(self.output, self.identity)
        cache.append({"id":"1"}, "a:1")
        with cache.checkpoint_path.open("ab") as handle:
            handle.write(b'{"id":"interrupted')
        restored = RunCache(self.output, self.identity, resume=True)
        restored.append({"id":"2"}, "a:2")
        final = RunCache(self.output, self.identity, resume=True)
        self.assertEqual([row["id"] for row in final.rows], ["1", "2"])

    def test_terminated_bad_line_is_not_silently_discarded(self):
        cache = RunCache(self.output, self.identity)
        cache.checkpoint_path.write_bytes(b'{broken}\n')
        with self.assertRaisesRegex(ValueError, "损坏"):
            RunCache(self.output, self.identity, resume=True)

    def test_complete_last_line_without_newline_can_resume_append(self):
        cache = RunCache(self.output, self.identity)
        cache.checkpoint_path.write_text(json.dumps({"id":"1", "cache_key":"a:1"}), encoding="utf-8")
        restored = RunCache(self.output, self.identity, resume=True)
        restored.append({"id":"2"}, "a:2")
        self.assertEqual(len(RunCache(self.output, self.identity, resume=True).rows), 2)

    def test_large_checkpoint_restore_never_reads_the_whole_file(self):
        cache = RunCache(self.output, self.identity)
        rows = [{"cache_key": f"a:{index}", "history": "历史 " * 40} for index in range(2000)]
        cache.checkpoint_path.write_bytes(b"".join(
            (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8") for row in rows))
        original_open = Path.open

        class BoundedReader:
            def __init__(self, handle):
                self.handle = handle

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return self.handle.__exit__(*args)

            def readline(self, *args):
                return self.handle.readline(*args)

            def read(self, size=-1):
                if size < 0:
                    raise AssertionError("Checkpoint restore must not read all raw bytes")
                return self.handle.read(size)

            def readlines(self, *args):
                raise AssertionError("Checkpoint restore must not materialize all raw lines")

        def bounded_open(path, mode="r", *args, **kwargs):
            handle = original_open(path, mode, *args, **kwargs)
            return BoundedReader(handle) if path == cache.checkpoint_path and mode == "rb" else handle

        with patch.object(Path, "open", bounded_open):
            restored = RunCache(self.output, self.identity, resume=True)
        self.assertEqual(restored.rows, rows)

    def test_partial_utf8_tail_truncates_at_exact_byte_boundary(self):
        cache = RunCache(self.output, self.identity)
        cache.append({"id": "保留"}, "a:1")
        valid = cache.checkpoint_path.read_bytes()
        with cache.checkpoint_path.open("ab") as handle:
            handle.write(b'{"cache_key":"a:2","text":"\xe4\xbd')
        restored = RunCache(self.output, self.identity, resume=True)
        self.assertEqual(restored.completed_keys, {"a:1"})
        self.assertEqual(cache.checkpoint_path.read_bytes(), valid)

    def test_corrupt_middle_line_and_duplicate_final_key_are_not_repaired(self):
        cache = RunCache(self.output, self.identity)
        valid = json.dumps({"cache_key": "a:1"}).encode()
        for raw, error in ((valid + b'\n{broken}\n{"cache_key":"a:2"}', "第 2 行损坏"),
                           (valid + b"\n" + valid, "重复 cache_key"),
                           (valid + b"\n{}", "缺少有效 cache_key")):
            cache.checkpoint_path.write_bytes(raw)
            with self.subTest(raw=raw), self.assertRaisesRegex(ValueError, error):
                RunCache(self.output, self.identity, resume=True)
            self.assertEqual(cache.checkpoint_path.read_bytes(), raw)

    def test_crlf_records_and_empty_checkpoint_restore(self):
        cache = RunCache(self.output, self.identity)
        self.assertEqual(RunCache(self.output, self.identity, resume=True).rows, [])
        first = json.dumps({"cache_key": "a:1", "text": "中文"}, ensure_ascii=False).encode("utf-8")
        second = json.dumps({"cache_key": "a:2"}).encode()
        cache.checkpoint_path.write_bytes(first + b"\r\n" + second)
        restored = RunCache(self.output, self.identity, resume=True)
        self.assertEqual(restored.completed_keys, {"a:1", "a:2"})
        self.assertEqual(cache.checkpoint_path.read_bytes(), first + b"\r\n" + second + b"\n")

    def test_existing_run_requires_explicit_resume_and_valid_manifest(self):
        RunCache(self.output, self.identity)
        with self.assertRaises(FileExistsError):
            RunCache(self.output, self.identity)
        manifest = json.loads((self.output / "manifest.json").read_text(encoding="utf-8"))
        manifest["identity"]["model"] = "tampered"
        (self.output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "签名不匹配"):
            RunCache(self.output, self.identity, resume=True)

    def test_identity_order_is_stable_and_source_changes_affect_hash(self):
        self.assertEqual(make_signature({"a":1,"b":2}), make_signature({"b":2,"a":1}))
        (self.root / "rag").mkdir()
        code = self.root / "rag" / "rewriters.py"
        code.write_text("PROMPT='first'", encoding="utf-8")
        first = code_hashes(self.root)
        code.write_text("PROMPT='second'", encoding="utf-8")
        self.assertNotEqual(first, code_hashes(self.root))


if __name__ == "__main__":
    unittest.main()
