"""Run four resumable shards of the frozen Server16 single-query evaluation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import evaluate_fusion as fusion


SHARDS = 4
ROOT = Path("/data1/zixuan/RAG_Research")
QRECC = ROOT.parent / "qrecc_full"
HERE = Path(__file__).resolve().parent
METHOD = fusion.STRATEGY
COVERAGE_COUNTS = ("annotated_sample_count", "unlabeled_sample_count", "scored_sample_count",
                   "samples_with_missing_gold", "samples_without_indexed_gold")


def load_plan(args):
    root = args.project_root.resolve()
    baseline = (args.baseline_run or root / "outputs/bm25_gpu4_experiments/main-test2000").resolve()
    base, evaluator = fusion.import_project(root)
    from rag.run_cache import code_hashes, file_sha256, make_signature

    signed = fusion.read_json(baseline / "manifest.json")
    old = signed["identity"]
    signature = make_signature(old)
    if signed["signature"] != signature or base.completed_report(baseline, signature) is None:
        raise ValueError("Baseline must have a verified completed report")
    hashes = code_hashes(root)
    if old["code"] != hashes:
        raise ValueError("Baseline evaluator source hashes differ from --project-root")
    config = dict(old["config"])
    if (config.get("retriever_backend"), config.get("retrieval_store"), config.get("evaluation_scope")) != (
            "bm25", "opensearch", "full"):
        raise ValueError("Expected the frozen full-QReCC OpenSearch BM25 baseline")
    config["opensearch_timeout"] = args.timeout
    ids = fusion.select_ids(args.subset_file, args.official_test, old["sample_ids"], None)
    data = root / config["eval_path"]
    prepared = base.prepared_dataset_identity(config, data)
    if prepared != {key: value for key, value in old["prepared_dataset"].items() if key != "test_subset"}:
        raise ValueError("Prepared dataset identity differs from baseline")
    source = evaluator.load_rewrite_examples(data, "test", None, data_format=old["data_format"])
    by_id = {row["id"]: row for row in source}
    paths = (Path(__file__), Path(fusion.__file__), HERE / "adaqr_support/adaqr_model.py",
             HERE / "adaqr_support/adaqr.py")
    common = {
        "task": "server16_adaqr_compcqr_single_test2000_sharded",
        "strategy": METHOD, "inference": "single", "mode": "hf_adapter",
        "config": config, "prepared_dataset": prepared, "sample_ids": ids,
        "subset_sha256": file_sha256(args.subset_file), "baseline_signature": signature,
        "code": hashes, "standalone_code": {path.name: file_sha256(path) for path in paths},
        "normalization": evaluator.NORMALIZATION, "rouge_l": bool(old.get("rouge_l", False)),
        "review_count": old.get("review_count", 30), "review_seed": old.get("review_seed", 20261102),
        "adapter": str(args.adapter.resolve()), "shards": SHARDS,
    }
    return SimpleNamespace(root=root, base=base, evaluator=evaluator, common=common,
                           examples=[by_id[identifier] for identifier in ids],
                           corpus=old["retrieval"]["corpus_fingerprint"])


class SuccessfulSearchCache:
    """Leave failed service requests pending while retaining the other outcomes."""

    def __init__(self, cache):
        self.cache = cache

    @property
    def rows(self):
        return self.cache.rows

    def append(self, row, key):
        error = row.get("retrieval_error") or ""
        if error.startswith("SearchServiceError:"):
            raise RuntimeError(f"{key} left pending for resume: {error}")
        self.cache.append(row, key)


def shard_dir(args, rank):
    return args.output_dir / f"shard-{rank}"


def worker(args, plan):
    import fcntl

    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock = args.output_dir / f".shard-{args.rank}.lock"
    with lock.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_worker(args, plan)


def run_worker(args, plan):
    from adaqr_support import adaqr_model
    from rag.opensearch_retrieval import SearchServiceError
    from rag.pipeline import RAGEngine
    from rag.run_cache import RunCache

    config = plan.common["config"]
    path = args.adapter.resolve()
    examples = plan.examples[args.rank::SHARDS]
    model, tokenizer = adaqr_model.load_rewriter(str(path))
    runtime = fusion.adapter_model_identity(adaqr_model, path, model, tokenizer)
    engine = fusion.AdapterEngine(RAGEngine(plan.root, config, mode="ollama"), path, adaqr_model)
    eligible, coverage = plan.evaluator.retrieval_coverage(config, examples, retriever=engine.retriever)
    if coverage["corpus_fingerprint"] != plan.corpus:
        raise ValueError("Live frozen BM25 index identity differs from baseline")
    identity = {"common": plan.common, "rank": args.rank, "sample_ids": [r["id"] for r in examples],
                "runtime_model_identity": runtime, "retrieval": coverage}
    output = shard_dir(args, args.rank)
    cache = RunCache(output, identity, resume=(output / "manifest.json").is_file())
    search = engine.retriever.search

    def retry_search(query, top_k):
        for delay in (5, 15):
            try:
                return search(query, top_k)
            except SearchServiceError as exc:
                print(f"Rank {args.rank}: retrying OpenSearch in {delay}s: {exc}", flush=True)
                time.sleep(delay)
        return search(query, top_k)

    engine.retriever.search = retry_search
    print(f"Shard {args.rank}: {len(cache.rows)}/{len(examples)} already checkpointed", flush=True)
    _, rows = plan.evaluator.evaluate_rewrites(engine, examples, [METHOD],
        include_l=plan.common["rouge_l"], cache=SuccessfulSearchCache(cache), retrieval_ids=eligible)
    print(f"Shard {args.rank}: {len(rows)}/{len(examples)} complete", flush=True)


def merge(args, plan):
    from rag.run_cache import RunCache, make_signature

    rows, signatures = [], []
    expected_model, retrieval = None, None
    for rank in range(SHARDS):
        directory = shard_dir(args, rank)
        signed = fusion.read_json(directory / "manifest.json")
        identity = signed["identity"]
        expected = plan.common["sample_ids"][rank::SHARDS]
        if identity["common"] != plan.common or identity["rank"] != rank or identity["sample_ids"] != expected:
            raise ValueError(f"Shard {rank} did not use the frozen configuration and sample IDs")
        cache = RunCache(directory, identity, resume=True)
        if sorted(record["id"] for record in cache.rows) != sorted(expected):
            raise ValueError(f"Shard {rank} is incomplete or contains unexpected/duplicate IDs")
        if any(record["strategy"] != METHOD or record["cache_key"] != f"{METHOD}:{record['id']}"
               or (record.get("retrieval_error") or "").startswith("SearchServiceError:")
               for record in cache.rows):
            raise ValueError(f"Shard {rank} contains invalid or retryable records")
        coverage = identity["retrieval"]
        if coverage["corpus_fingerprint"] != plan.corpus:
            raise ValueError(f"Shard {rank} used a different BM25 index")
        model = identity["runtime_model_identity"]
        if rank == 0:
            expected_model, retrieval = model, dict(coverage)
        else:
            policy = {key: value for key, value in coverage.items() if key not in COVERAGE_COUNTS}
            if model != expected_model or policy != {key: value for key, value in retrieval.items()
                                                     if key not in COVERAGE_COUNTS}:
                raise ValueError(f"Shard {rank} differs in model or retrieval policy")
            for key in COVERAGE_COUNTS:
                retrieval[key] += coverage[key]
        rows.extend(cache.rows)
        signatures.append(signed["signature"])
    expected_ids = plan.common["sample_ids"]
    if len(rows) != 2000 or len({row["id"] for row in rows}) != 2000:
        raise ValueError("Expected exactly 2,000 unique records")
    eligible = {row["id"] for row in plan.examples if row.get("positive_ids")}
    if {row["id"] for row in rows if row["retrieval_eligible"]} != eligible:
        raise ValueError("Merged retrieval eligibility differs from frozen labels")
    signature = make_signature({"common": plan.common, "workers": signatures})
    entries = plan.base.completed_report(args.output_dir, signature)
    if entries is not None:
        print(f"Retained verified report: {args.output_dir / 'report.json'}", flush=True)
        return
    # Every key is cached. The original evaluator only orders rows and computes metrics here.
    summaries, details = plan.evaluator.evaluate_rewrites(None, plan.examples, [METHOD],
        include_l=plan.common["rouge_l"], cache=SimpleNamespace(rows=rows), progress=False,
        retrieval_ids=eligible, retriever=object())
    metadata = {**plan.common, "sample_count": len(expected_ids), "retrieval": retrieval,
                "run_signature": signature, "shard_signatures": signatures,
                "runtime_model_identity": expected_model, "retrieval_top_k": plan.evaluator.RETRIEVAL_TOP_K,
                "evaluation_scope": "same_frozen_test2000", "created_at": datetime.now(timezone.utc).isoformat(),
                "comparison_scope": "Trained standalone rewrite followed by frozen BM25 retrieval.",
                "timing_caveat": "Four concurrent HF/PEFT workers share the search service."
                                 " Latencies are not directly comparable with a serial Ollama run."}
    plan.evaluator.export_blind_review(args.output_dir, details,
        count=plan.common["review_count"], seed=plan.common["review_seed"])
    plan.base.save_report(args.output_dir, metadata, summaries, details,
                          extra_artifacts=plan.evaluator.REVIEW_ARTIFACTS)
    if plan.base.completed_report(args.output_dir, signature) is None:
        raise RuntimeError("Merged report failed checksum verification")
    print(f"Published {len(details)} records: {args.output_dir / 'report.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("worker", "merge"))
    parser.add_argument("--rank", type=int, choices=range(SHARDS))
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--baseline-run", type=Path)
    parser.add_argument("--subset-file", type=Path, default=QRECC / "subsets/test_2000.json")
    parser.add_argument("--official-test", type=Path, default=QRECC / "raw/qrecc-test.json")
    parser.add_argument("--adapter", type=Path, default=HERE / "outputs/train_half_v1/dpo_baseline")
    parser.add_argument("--output-dir", type=Path, default=HERE / "outputs/train_half_v1/eval_test2000_4gpu")
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    if (args.phase == "worker") != (args.rank is not None) or args.timeout < 1:
        parser.error("worker requires --rank, merge forbids it, and --timeout must be positive")
    args.output_dir = args.output_dir.resolve()
    if not args.output_dir.is_relative_to(HERE / "outputs") or args.output_dir == HERE / "outputs":
        parser.error("--output-dir must be a dedicated directory below the experiment's outputs/")
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    try:
        plan = load_plan(args)
        (worker if args.phase == "worker" else merge)(args, plan)
    except (ValueError, OSError, RuntimeError, KeyError, ImportError) as exc:
        print(f"Sharded AdaQR evaluation: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
