"""Bound CompCQR BM25 load and archive a stage before a clean restart."""
from collections import Counter
from copy import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import BoundedSemaphore, Event

from rag.opensearch_retrieval import SearchServiceError


class CompCQRSearch:
    """Keep the original rankings per query while bounding each multi-search."""

    def __init__(self, retriever, stop: Event, *, batch_size: int, max_inflight: int, timeout_seconds: int):
        if min(batch_size, max_inflight, timeout_seconds) < 1:
            raise ValueError("CompCQR search batch, concurrency and timeout must be positive.")
        self.retriever = copy(retriever)
        self.retriever.timeout = timeout_seconds
        self.stop = stop
        self.batch_size = batch_size
        self.gate = BoundedSemaphore(max_inflight)

    def search_many(self, queries, top_k=100):
        rankings = []
        for offset in range(0, len(queries), self.batch_size):
            group = queries[offset:offset + self.batch_size]
            with self.gate:
                rankings.extend(self.search_batch(group, top_k))
        return rankings

    def search_batch(self, group, top_k):
        for delay in (5, 15, None):
            if self.stop.is_set():
                raise InterruptedError("CompCQR search interrupted.")
            try:
                return self.retriever.search_many(group, top_k)
            except SearchServiceError as exc:
                message = str(exc).lower()
                retryable = (
                    getattr(exc, "status", None) in {429, 502, 503, 504}
                    or "timed out" in message or "timeout" in message
                )
                if not retryable or delay is None:
                    raise
                print(f"\nCompCQR BM25 batch of {len(group)} queries failed; retrying in {delay}s: {exc}",
                      flush=True)
                if self.stop.wait(delay):
                    raise InterruptedError("CompCQR search interrupted.") from exc


def restart_compcqr(output_dir, sample_ids):
    """Archive the old checkpoint and keep the two completed stages byte-for-byte."""
    output_dir = Path(output_dir)
    checkpoint = output_dir / "checkpoint.jsonl"
    expected = set(sample_ids)
    if not expected or len(expected) != len(sample_ids):
        raise ValueError("Invalid sample IDs in the signed experiment.")
    retained = {"rewrite": set(), "structured": set()}
    allowed = {"rewrite", "structured", "compcqr"}
    seen = set()
    removed = Counter()
    temp = None
    try:
        with NamedTemporaryFile(mode="wb", prefix=".checkpoint-restart-", dir=output_dir,
                                delete=False) as new, checkpoint.open("rb") as old:
            temp = Path(new.name)
            for line_number, line in enumerate(old, 1):
                row = json.loads(line)
                strategy = row["strategy"]
                sample = row["id"]
                key = row["cache_key"]
                if strategy not in allowed or sample not in expected or key != f"{strategy}:{sample}" or key in seen:
                    raise ValueError(f"Unexpected checkpoint key on line {line_number}: {key!r}")
                seen.add(key)
                if strategy == "compcqr":
                    removed["rows"] += 1
                    if row.get("retrieval_error"):
                        removed["retrieval_errors"] += 1
                else:
                    retained[strategy].add(sample)
                    new.write(line)
            if any(ids != expected for ids in retained.values()):
                raise ValueError("Both earlier strategies must be complete before restarting CompCQR.")
            new.flush()
            os.fsync(new.fileno())
        if not removed["rows"]:
            print("CompCQR checkpoint is already empty.", flush=True)
            return
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup = checkpoint.with_name(f"checkpoint.before-compcqr-restart.{stamp}.jsonl")
        os.link(checkpoint, backup)
        directory_fd = os.open(output_dir, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
            os.replace(temp, checkpoint)
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        print(json.dumps({
            "backup": str(backup),
            "retained": {key: len(ids) for key, ids in retained.items()},
            "removed_compcqr": dict(removed),
        }), flush=True)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)
