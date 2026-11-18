"""Evaluate the provided official QReCC test subset with the main experiment settings."""
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
from pathlib import Path
import sys
from threading import Event
from unittest.mock import patch

from server16_gpu_parallel import (
    CONFIG,
    OUTPUT_ROOT,
    ROOT,
    evaluator,
    existing_plan,
    launch_replicas,
    parallel_evaluation,
)
from rag.run_cache import code_hashes, make_signature


BASELINE = OUTPUT_ROOT / "main"
OUTPUT = OUTPUT_ROOT / "main-test2000"
TEST_SOURCE = ROOT.parent / "qrecc_full/raw/qrecc-test.json"
TEST_MD5 = "7ad29296a1dfd2f75dafd653a8822269"
BASELINE_STRATEGIES = ("rewrite", "structured")
COVERAGE_COUNTS = {"annotated_sample_count", "unlabeled_sample_count", "scored_sample_count"}


def subset_selection(path):
    source = TEST_SOURCE.read_bytes()
    if hashlib.md5(source).hexdigest() != TEST_MD5:
        raise ValueError("The official QReCC test file no longer matches its published MD5.")
    def key(row):
        return f"test:{row['Conversation_no']}-{row['Turn_no']}"

    official = {key(row): row for row in json.loads(source)}
    raw = path.read_bytes()
    rows = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(rows, list) or len(rows) != 2000:
        raise ValueError("Expected exactly 2,000 QReCC test records.")
    ids = []
    for row in rows:
        identifier = key(row)
        if official.get(identifier) != row:
            raise ValueError(f"Subset record differs from official QReCC test: {identifier}")
        ids.append(identifier)
    if len(set(ids)) != len(ids):
        raise ValueError("The subset contains duplicate test IDs.")
    info = {"name": path.name, "sha256": hashlib.sha256(raw).hexdigest(),
            "official_split": "test", "selected_turns": len(ids),
            "official_test_turns": len(official)}
    return ids, info


def subset_examples(path, split, limit, *, data_format, ids):
    if split != "test" or limit is not None:
        raise ValueError("The saved subset already defines the evaluation sample IDs.")
    full = evaluator_source_loader(path, split, None, data_format=data_format)
    records = {row["id"]: row for row in full}
    return [records[identifier] for identifier in ids]


def read_baseline_rows(ids):
    selected = set(ids)
    records = {}
    with (BASELINE / "checkpoint.jsonl").open("rb") as stream:
        for line in stream:
            row = json.loads(line)
            strategy = row["strategy"]
            if strategy not in BASELINE_STRATEGIES or row["id"] not in selected:
                continue
            key = f"{strategy}:{row['id']}"
            if row.get("cache_key") != key or key in records or not row.get("retrieval_executed"):
                raise ValueError(f"Invalid or incomplete baseline record: {key}")
            records[key] = row
    if len(records) != len(ids) * len(BASELINE_STRATEGIES):
        raise ValueError("The full run does not contain both baseline stages for all selected IDs.")
    return records


def verify_baseline_identity(cache_identity, baseline_identity):
    comparable = dict(cache_identity)
    comparable["sample_ids"] = baseline_identity["sample_ids"]
    comparable["prepared_dataset"] = dict(comparable["prepared_dataset"])
    comparable["prepared_dataset"].pop("test_subset")
    comparable["retrieval"] = dict(comparable["retrieval"])
    for key in COVERAGE_COUNTS:
        comparable["retrieval"][key] = baseline_identity["retrieval"][key]
    differences = sorted(key for key in set(comparable) | set(baseline_identity)
                         if comparable.get(key) != baseline_identity.get(key))
    if differences:
        raise ValueError("The baseline and subset evaluations differ in: " + ", ".join(differences))


def seed_baselines(cache, records, ids):
    for row in cache.rows:
        if row["strategy"] in BASELINE_STRATEGIES and records.get(row["cache_key"]) != row:
            raise ValueError(f"Existing subset baseline record differs: {row['cache_key']}")
    added = 0
    for strategy in BASELINE_STRATEGIES:
        for identifier in ids:
            key = f"{strategy}:{identifier}"
            if key not in cache.completed_keys:
                cache.append(records[key], key)
                added += 1
    print(f"Matched baseline results | rewrite={len(ids)} | structured={len(ids)} | inserted={added}", flush=True)


evaluator_source_loader = evaluator.load_rewrite_examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-file", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--workers-per-gpu", type=int, choices=range(1, 5), default=1)
    parser.add_argument("--compcqr-search-batch", type=int, default=2)
    parser.add_argument("--compcqr-search-concurrency", type=int, default=2)
    parser.add_argument("--compcqr-search-timeout", type=int, default=300)
    args = parser.parse_args()
    if min(args.gpus) < 0 or len(set(args.gpus)) != len(args.gpus):
        parser.error("--gpus must contain distinct non-negative GPU IDs")
    search_limits = {"batch_size": args.compcqr_search_batch,
                     "max_inflight": args.compcqr_search_concurrency,
                     "timeout_seconds": args.compcqr_search_timeout}
    if min(search_limits.values()) < 1:
        parser.error("Search batch, concurrency and timeout must be positive")
    command, digest = existing_plan()
    signed = json.loads((BASELINE / "manifest.json").read_text(encoding="utf-8"))
    baseline_identity = signed["identity"]
    if baseline_identity.get("code") != code_hashes(ROOT):
        raise ValueError("Evaluation source hashes differ from the full baseline.")
    ids, subset = subset_selection(args.subset_file.resolve())
    subset["reused_baseline_signature"] = signed["signature"]
    subset["reused_baseline_stages"] = list(BASELINE_STRATEGIES)
    subset["baseline_timing"] = "Original recorded timings from the full test evaluation."
    if OUTPUT.exists():
        previous = json.loads((OUTPUT / "manifest.json").read_text(encoding="utf-8"))
        if (previous["signature"] != make_signature(previous["identity"])
                or previous["identity"]["prepared_dataset"].get("test_subset") != subset
                or previous["identity"]["sample_ids"] != ids):
            raise ValueError(f"Subset checkpoint belongs to a different run: {OUTPUT}")
        if evaluator.completed_report(OUTPUT, previous["signature"]) is not None:
            print(f"Evaluation already complete: {OUTPUT}", flush=True)
            return 0

    with ExitStack() as stack:
        lock = stack.enter_context((OUTPUT_ROOT / ".parallel.lock").open("a"))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        records = read_baseline_rows(ids)
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        replicas = launch_replicas(args.gpus, args.workers_per_gpu, config, digest, stack)
        old_evaluate = evaluator.evaluate_rewrites
        old_prepared = evaluator.prepared_dataset_identity
        stop = Event()

        def restrict_examples(path, split, limit=None, *, data_format="auto"):
            return subset_examples(path, split, limit, data_format=data_format, ids=ids)

        def subset_identity(config, path):
            return {**old_prepared(config, path), "test_subset": subset}

        def evaluate_batch(engine, examples, strategies, *, cache=None, **kwargs):
            verify_baseline_identity(cache.identity, baseline_identity)
            seed_baselines(cache, records, ids)
            return parallel_evaluation(old_evaluate, replicas, args.workers_per_gpu, stop, search_limits,
                                       engine, examples, strategies, cache=cache, **kwargs)

        argv = command[command.index(str(ROOT / "evaluate_rewrite.py")):]
        argv[argv.index("--output-dir") + 1] = str(OUTPUT)
        if OUTPUT.exists():
            argv.append("--resume")
        print(f"Official QReCC subset | {len(ids)} turns | {OUTPUT}", flush=True)
        stack.enter_context(patch.object(evaluator, "load_rewrite_examples", restrict_examples))
        stack.enter_context(patch.object(evaluator, "prepared_dataset_identity", subset_identity))
        stack.enter_context(patch.object(evaluator, "evaluate_rewrites", evaluate_batch))
        stack.enter_context(patch.object(sys, "argv", argv))
        return evaluator.main()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        raise SystemExit(f"Subset experiment: {error}") from error
