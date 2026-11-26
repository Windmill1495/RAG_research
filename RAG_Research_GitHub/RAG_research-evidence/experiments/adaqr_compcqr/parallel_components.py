"""Distribute component generation and baseline reward scoring across GPUs."""

import argparse
from functools import partial
import json
import os
import selectors
import subprocess
import sys
from pathlib import Path

import parallel_adaqr_half as base


GPUS = base.gpu_indices("ADAQR_COMPONENT_GPUS", "0,1")
BATCH_THREADS_ENV = "ADAQR_BM25_BATCH_THREADS"
DEFAULT_BATCH_THREADS = 4


class BatchedPaperBM25(base.PaperBM25):
    """Use native Lucene concurrency and return hits in the original query order."""

    def __init__(self, directory, threads=DEFAULT_BATCH_THREADS):
        super().__init__(directory)
        self.threads = threads
        self._searcher_ready = False

    def search_many(self, queries, top_k):
        self._documents.clear()
        first = []
        if queries and not self._searcher_ready:
            # Initialize Anserini's Java IndexSearcher before concurrent use.
            first = super().search_many(queries[:1], top_k)
            self._searcher_ready = True
            queries = queries[1:]
        qids = [str(index) for index in range(len(queries))]
        if not qids:
            return first
        results = self.searcher.batch_search(queries, qids, k=top_k, threads=self.threads)
        if results.keys() != set(qids) or any(results[qid] is None for qid in qids):
            # Keep the original behavior if a query was omitted by the batch API.
            return first + super().search_many(queries, top_k)
        return first + [
            [{"id": hit.docid, "passage_id": hit.docid, "score": float(hit.score)}
             for hit in results[qid]]
            for qid in qids
        ]


def worker(phase, gpu, result_fd, assigned_ids):
    rows, _ = base.dataset.load(base.RUN / "dataset")
    asked = set(assigned_ids)
    selected = [row for row in rows if row["id"] in asked]
    if len(selected) != len(assigned_ids):
        raise ValueError(f"Assigned {phase} IDs are incomplete or duplicated")
    if phase == "baseline":
        components = {
            row["id"]: row
            for slot in range(base.SHARDS)
            for row in base.read_jsonl(base.cache_path("components", slot) / "checkpoint.jsonl")
            if row["id"] in asked
        }
        if components.keys() != asked:
            raise ValueError("Missing component results for assigned reward turns")
        threads = int(os.environ.get(BATCH_THREADS_ENV, DEFAULT_BATCH_THREADS))
        if threads < 0:
            raise ValueError(f"{BATCH_THREADS_ENV} must be nonnegative")
        if threads:
            # This binding is private to the worker. The signed scoring code
            # and both checkpoint shards are shared with the serial launcher.
            base.PaperBM25 = partial(BatchedPaperBM25, threads=threads)
            print(f"baseline | GPU {gpu} | BM25 batch search ({threads} threads)", flush=True)
        operation = base.baseline_operation(components)
    else:
        operation = base.component_operation()
    with os.fdopen(result_fd, "w", encoding="utf-8", buffering=1) as stream:
        print(f"{phase} | GPU {gpu} | {len(selected)} pending turns", flush=True)
        for number, row in enumerate(selected, 1):
            result = operation(row)
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            if number == 1 or number % 25 == 0 or number == len(selected):
                print(f"{phase} | GPU {gpu} | {number}/{len(selected)} processed", flush=True)


def checkpoint_state(phase, rows, identity):
    caches = [base.cache_for(phase, slot, identity) for slot in range(base.SHARDS)]
    slots = {row["id"]: position % base.SHARDS for position, row in enumerate(rows)}
    complete = set()
    for slot, cache in enumerate(caches):
        for key in cache.completed_keys:
            if slots.get(key) != slot:
                raise ValueError(f"Unexpected turn {key} in existing shard {slot}")
            complete.add(key)
    return caches, slots, complete


def write_result(payload, caches, slots, pending):
    record = json.loads(payload)
    key = record["id"]
    if key not in pending:
        raise ValueError(f"Unexpected or duplicate result: {key}")
    if record.get("status") not in {"ok", "component_error"}:
        raise ValueError(f"Invalid result status for {key}")
    caches[slots[key]].append(record, key)
    pending.remove(key)


def start_worker(phase, gpu, device, ids, lock_fd, selector):
    reader, writer = os.pipe()
    command = [sys.executable, "-u", "-B", str(Path(__file__).resolve()), "worker",
               "--phase", phase, "--gpu", str(gpu), "--fd", str(writer),
               "--ids", json.dumps(ids)]
    try:
        process = subprocess.Popen(command, env={**os.environ, "CUDA_VISIBLE_DEVICES": device},
                                   pass_fds=(lock_fd, writer))
    except BaseException:
        os.close(reader)
        raise
    finally:
        os.close(writer)
    selector.register(reader, selectors.EVENT_READ, gpu)
    return process


def collect(phase, selector, processes, caches, slots, pending, total):
    buffers = {}
    while selector.get_map():
        for key, _ in selector.select(timeout=1):
            reader, gpu = key.fd, key.data
            chunk = os.read(reader, 65_536)
            if not chunk:
                if buffers.get(reader):
                    raise ValueError(f"GPU {gpu} closed the result stream mid-record")
                selector.unregister(reader)
                os.close(reader)
                continue
            parts = (buffers.get(reader, b"") + chunk).split(b"\n")
            buffers[reader] = parts.pop()
            for part in parts:
                write_result(part, caches, slots, pending)
                done = total - len(pending)
                if done % 25 == 0 or not pending:
                    print(f"{phase} | {done}/{total} persisted across both original shards",
                          flush=True)
        for gpu, process in processes.items():
            code = process.poll()
            if code not in (None, 0):
                raise RuntimeError(f"{phase} worker on GPU {gpu} exited {code}; rerun to resume")


def launch(phase):
    import fcntl

    if phase == "baseline":
        gpu_list = base.gpu_indices("ADAQR_REWARD_GPUS", ",".join(map(str, base.GPUS)))
        if len(gpu_list) < 2:
            raise ValueError("ADAQR_REWARD_GPUS requires two or more physical GPUs")
        base.RUN.mkdir(parents=True, exist_ok=True)
    else:
        gpu_list = GPUS
    with base.LOCK.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another fusion job still has the lock; wait for it to finish") from error
        if phase == "baseline":
            rows, manifest = base.dataset.prepare(base.SOURCE, base.RUN / "dataset")
        else:
            rows, manifest = base.dataset.load(base.RUN / "dataset")
        identity = base.component_identity(manifest)
        if phase == "baseline":
            base.completed_shards("components", rows, identity)
            identity = base.get_baseline_identity(identity)
        caches, slots, complete = checkpoint_state(phase, rows, identity)
        todo = [row["id"] for row in rows if row["id"] not in complete]
        total = len(rows)
        print(f"{phase} | Resuming {len(complete)}/{total} complete | {len(todo)} pending", flush=True)
        devices = base.gpu_map(gpu_list[:len(todo)]) if todo else {}
        pending = set(todo)
        processes = {}
        with selectors.DefaultSelector() as selector:
            try:
                for offset, (gpu, device) in enumerate(devices.items()):
                    assigned = todo[offset::len(devices)]
                    if assigned:
                        processes[gpu] = start_worker(phase, gpu, device, assigned, lock.fileno(), selector)
                collect(phase, selector, processes, caches, slots, pending, total)
                for gpu, process in processes.items():
                    if process.wait():
                        raise RuntimeError(f"{phase} worker on GPU {gpu} failed; rerun to resume")
                if pending:
                    raise RuntimeError(f"{len(pending)} {phase} turns remain; rerun to resume")
            finally:
                for conn in tuple(selector.get_map().values()):
                    selector.unregister(conn.fd)
                    os.close(conn.fd)
                for process in processes.values():
                    if process.poll() is None:
                        process.terminate()
                for process in processes.values():
                    process.wait()
        if phase == "baseline":
            del caches
            base.publish(phase, rows, identity)
        base.status()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("check", "run", "baseline", "worker"))
    parser.add_argument("--phase", choices=("components", "baseline"), default="components")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--fd", type=int)
    parser.add_argument("--ids")
    args = parser.parse_args()
    if args.stage == "check":
        try:
            print(base.gpu_map(GPUS))
        except RuntimeError as error:
            parser.exit(1, f"{error}\n")
    elif args.stage == "worker":
        worker(args.phase, args.gpu, args.fd, json.loads(args.ids))
    else:
        launch("baseline" if args.stage == "baseline" else "components")


if __name__ == "__main__":
    main()
