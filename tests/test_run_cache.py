"""Signed resume preserves fixed samples and refuses cross-experiment mixing."""
import json
import tempfile
import unittest
from pathlib import Path

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
