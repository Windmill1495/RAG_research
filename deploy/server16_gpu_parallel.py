"""Resume the signed Server16 BM25 run with one Ollama replica per GPU."""
import argparse
import fcntl
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from threading import Event, Lock
from time import monotonic

from server16_gpu_main import (
    BASE_URL, CONFIG, MODEL, OLLAMA, OUTPUT_ROOT, ROOT,
    gpu_status, start_model, stop_server,
)

sys.path.insert(0, str(ROOT))

import evaluate_rewrite as evaluator
from rag.pipeline import RAGEngine
from rag.run_cache import file_sha256, make_signature
from run_experiments import build_jobs, parse_args
from server16_compcqr_recovery import CompCQRSearch, restart_compcqr


class CheckpointShard:
    """Delegate durable writes to the one shared checkpoint writer."""

    def __init__(self, cache, stop, on_append, *, require_retrieval=False):
        self.rows = ()
        self.cache = cache
        self.stop = stop
        self.on_append = on_append
        self.require_retrieval = require_retrieval

    def append(self, row, key):
        if self.require_retrieval and row.get("retrieval_error"):
            self.stop.set()
            raise RuntimeError(f"CompCQR retrieval failed for {key}; left pending for resume: {row['retrieval_error']}")
        self.cache.append(row, key)
        self.on_append()
        if self.stop.is_set():
            raise InterruptedError("Parallel evaluation interrupted")


class StrategyProgress:
    def __init__(self, strategy, completed, total):
        self.strategy = strategy
        self.completed = completed
        self.total = total
        self.lock = Lock()
        self.started = monotonic()
        self.updated = self.started
        self.initial = completed
        self.render()

    def render(self):
        elapsed = max(monotonic() - self.started, 0.01)
        rate = (self.completed - self.initial) / elapsed
        print(f"\r{self.strategy}: {self.completed}/{self.total}  {rate:.2f} turns/s", end="", flush=True)

    def advance(self):
        with self.lock:
            self.completed += 1
            now = monotonic()
            if now - self.updated >= 0.5 or self.completed == self.total:
                self.render()
                self.updated = now


def parallel_evaluation(original, replicas, slots, stop, search_limits, engine, examples, strategies, *,
                        include_l=False, cache=None, progress=True, retrieval_ids=None, retriever=None):
    if cache is None:
        raise ValueError("Parallel evaluation requires the official run checkpoint.")
    if engine.config["retrieval_store"] != "opensearch" or engine.config["retriever_backend"] != "bm25":
        raise ValueError("This launcher requires the official OpenSearch BM25 configuration.")

    target = engine.retriever if retriever is None else retriever
    compcqr_target = CompCQRSearch(target, stop, **search_limits)
    lanes = [
        RAGEngine(ROOT, {**engine.config, "base_url": url}, mode=engine.mode, build_index=False)
        for _, url in replicas for _ in range(slots)
    ]
    audit_path = cache.output_dir / "parallel-execution.jsonl"
    with audit_path.open("a", encoding="utf-8") as log:
        log.write(json.dumps({
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint_signature": cache.signature,
            "existing_rows": len(cache.rows),
            "launcher_sha256": file_sha256(__file__),
            "replicas": [{"gpu": index, "base_url": url} for index, url in replicas],
            "workers_per_gpu": slots,
            "compcqr_search_limits": search_limits,
            "compcqr_recovery_sha256": file_sha256(Path(__file__).with_name("server16_compcqr_recovery.py")),
        }) + "\n")

    with ThreadPoolExecutor(max_workers=len(lanes)) as pool:
        try:
            for strategy in strategies:
                pending = [
                    example for example in examples
                    if f"{strategy}:{example['id']}" not in cache.completed_keys
                ]
                status = StrategyProgress(strategy, len(examples) - len(pending), len(examples))
                is_compcqr = strategy == "compcqr"
                futures = [
                    pool.submit(
                        original, lane, pending[offset::len(lanes)], [strategy],
                        include_l=include_l,
                        cache=CheckpointShard(cache, stop, status.advance, require_retrieval=is_compcqr),
                        progress=False, retrieval_ids=retrieval_ids,
                        retriever=compcqr_target if is_compcqr else target,
                    )
                    for offset, lane in enumerate(lanes)
                    if offset < len(pending)
                ]
                for future in as_completed(futures):
                    future.result()
                print(flush=True)
        except BaseException:
            stop.set()
            raise

    expected = len(strategies) * len(examples)
    if len(cache.completed_keys) != expected:
        raise RuntimeError(f"Expected {expected} checkpoint rows, found {len(cache.completed_keys)}.")
    # The original evaluator sorts the completed rows and calculates the final
    # metrics. All keys are committed, so this pass makes no retrieval or LLM calls.
    return original(
        engine, examples, strategies, include_l=include_l, cache=cache, progress=False,
        retrieval_ids=retrieval_ids, retriever=retriever,
    )


def existing_plan():
    runner = parse_args([
        "--config", str(CONFIG), "--experiment", "main", "--split", "test",
        "--main-model", MODEL, "--output-root", str(OUTPUT_ROOT),
    ])
    job, = build_jobs(runner)
    control = OUTPUT_ROOT / "main.launcher.json"
    saved = json.loads(control.read_text(encoding="utf-8"))
    identity = saved["identity"]
    command = job["command"]
    if saved.get("signature") != make_signature(identity) or identity["command"] != command:
        raise ValueError("The stored launcher does not match this official main experiment.")
    manifest = Path(job["output"]) / "manifest.json"
    signed = json.loads(manifest.read_text(encoding="utf-8"))
    if signed.get("signature") != make_signature(signed["identity"]):
        raise ValueError("The existing experiment manifest signature is invalid.")
    return command, signed["identity"]["runtime_model_identity"]["digest"]


def launch_replicas(indices, slots, config, digest, stack):
    targets = [(index, gpu_status(index), f"http://127.0.0.1:{11438 + offset}")
               for offset, index in enumerate(indices)]
    for offset in range(len(targets)):
        with socket.socket() as port:
            port.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            port.bind(("127.0.0.1", 11438 + offset))

    model_logs = OUTPUT_ROOT / "replicas"
    model_logs.mkdir(exist_ok=True)
    common = {
        **os.environ,
        "OLLAMA_BASE_URL": BASE_URL,
        "OPENSEARCH_URL": config["opensearch_url"],
        "OLLAMA_MODELS": str(ROOT.parent / "ollama_models"),
        "OLLAMA_NOPRUNE": "true",
        "OLLAMA_SCHED_SPREAD": "false",
        "OLLAMA_NUM_PARALLEL": str(slots),
        "OLLAMA_MAX_LOADED_MODELS": "1",
        "QRECC_DATA_DIR": str(ROOT.parent / "qrecc_full"),
    }
    no_proxy = ",".join(filter(None, (os.environ.get("NO_PROXY"), os.environ.get("no_proxy"), "127.0.0.1")))
    common.update(NO_PROXY=no_proxy, no_proxy=no_proxy)
    os.environ.update({key: common[key] for key in (
        "OLLAMA_BASE_URL", "OPENSEARCH_URL", "QRECC_DATA_DIR", "NO_PROXY", "no_proxy",
    )})
    servers = []
    for index, (uuid, name, memory), url in targets:
        temp = model_logs / f"gpu-{index}-tmp"
        temp.mkdir(exist_ok=True)
        log_path = model_logs / f"gpu-{index}.log"
        env = {**common, "CUDA_VISIBLE_DEVICES": uuid, "OLLAMA_HOST": url,
               "TMPDIR": str(temp), "OLLAMA_TMPDIR": str(temp)}
        print(f"GPU {index} | {name} | {memory} MiB free | {url}", flush=True)
        log = stack.enter_context(log_path.open("ab"))
        server = subprocess.Popen(
            [str(OLLAMA), "serve"], cwd=ROOT, env=env, start_new_session=True,
            stdout=log, stderr=subprocess.STDOUT,
        )
        stack.callback(stop_server, server)
        servers.append((server, url, log_path))
    with ThreadPoolExecutor(max_workers=len(servers)) as pool:
        checks = [pool.submit(start_model, server, config, base_url=url, service_log=log_path,
                              expected_digest=digest) for server, url, log_path in servers]
        for check in as_completed(checks):
            check.result()
    return [(index, url) for index, _, url in targets]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, nargs="+", default=[4, 5, 6, 7])
    parser.add_argument("--workers-per-gpu", type=int, choices=range(1, 5), default=1)
    parser.add_argument("--compcqr-search-batch", type=int, default=2,
                        help="Maximum BM25 subqueries in each CompCQR request")
    parser.add_argument("--compcqr-search-timeout", type=int, default=300,
                        help="Read timeout in seconds for each CompCQR BM25 request")
    parser.add_argument("--compcqr-search-concurrency", type=int, default=1,
                        help="Maximum simultaneous CompCQR BM25 requests")
    parser.add_argument("--reset-compcqr-checkpoint", action="store_true",
                        help="Archive the checkpoint and remove the CompCQR stage from the live copy")
    args = parser.parse_args()
    if min(args.compcqr_search_batch, args.compcqr_search_concurrency, args.compcqr_search_timeout) < 1:
        parser.error("CompCQR search batch, concurrency and timeout must be positive")
    if min(args.gpus) < 0 or len(set(args.gpus)) != len(args.gpus):
        parser.error("--gpus must contain distinct non-negative device IDs")
    command, digest = existing_plan()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    with ExitStack() as stack:
        lock = stack.enter_context((OUTPUT_ROOT / ".parallel.lock").open("a"))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.reset_compcqr_checkpoint:
            if subprocess.run(["tmux", "has-session", "-t", "rag16-bm25-main-gpu"], capture_output=True).returncode == 0:
                raise RuntimeError("Stop the active main experiment before resetting its checkpoint.")
            output = OUTPUT_ROOT / "main"
            identity = json.loads((output / "manifest.json").read_text(encoding="utf-8"))["identity"]
            restart_compcqr(output, identity["sample_ids"])
            return 0
        replicas = launch_replicas(args.gpus, args.workers_per_gpu, config, digest, stack)
        stop = Event()
        original = evaluator.evaluate_rewrites

        search_limits = {
            "batch_size": args.compcqr_search_batch,
            "max_inflight": args.compcqr_search_concurrency,
            "timeout_seconds": args.compcqr_search_timeout,
        }
        print(f"CompCQR BM25 search limits: {search_limits}", flush=True)

        def evaluate_batch(*parameters, **options):
            return parallel_evaluation(original, replicas, args.workers_per_gpu, stop, search_limits,
                                       *parameters, **options)

        # Replace only loop scheduling in this process. The official evaluator
        # still verifies every signed input and performs all per-turn scoring.
        old_argv = sys.argv
        evaluator.evaluate_rewrites = evaluate_batch
        sys.argv = [str(ROOT / "evaluate_rewrite.py"), *command[command.index(str(ROOT / "evaluate_rewrite.py")) + 1:], "--resume"]
        try:
            return evaluator.main()
        finally:
            sys.argv = old_argv
            evaluator.evaluate_rewrites = original


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Parallel main experiment: {error}", file=sys.stderr)
        raise SystemExit(1)
