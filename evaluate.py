"""Signed fixed-history experiments; generation failures preserve retrieval scores."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from rag.config import load_config, offline_config
from rag.data import load_examples, load_passage_ids, public_input
from rag.metrics import document_metrics, passage_metrics, rouge_scores
from rag.llm import runtime_embedding_identity, runtime_model_identity
from rag.pipeline import RAGEngine
from rag.rewriters import STRATEGIES
from rag.run_cache import RunCache, code_hashes, file_sha256

PROJECT_DIR = Path(__file__).resolve().parent


def retrieval_metrics(hits, gold_document, top_k):
    """Score a source-document hit at the original chunk ranks."""
    return document_metrics(hits, gold_document, top_k)


def keyword_coverage(answer, keywords):
    if not keywords:
        return None
    normalized = "".join(answer.casefold().split())
    return sum("".join(word.casefold().split()) in normalized for word in keywords) / len(keywords)


def mean(values):
    return statistics.fmean(values) if values else None


def is_fallback(row):
    trace = row.get("rewrite_trace") or {}
    return "fallback" in str(row.get("rewrite_source", "")) or isinstance(trace, dict) and bool(trace.get("fallback"))


def summarize(details, strategies):
    summaries = []
    for strategy in strategies:
        rows = [row for row in details if row["strategy"] == strategy]
        keys = sorted({key for row in rows for key in row.get("latencies_ms", {})})
        summaries.append({
            "strategy": strategy, "sample_count": len(rows),
            "metric_level": rows[0]["metric_level"] if rows else None,
            "error_count": sum(row.get("error") is not None for row in rows),
            "retrieval_error_count": sum(row.get("error_stage") in {"retrieval", "metrics"} for row in rows),
            "generation_error_count": sum(row.get("error_stage") == "generation" for row in rows),
            "rewrite_error_count": sum(bool(row.get("rewrite_error")) for row in rows),
            "fallback_count": sum(is_fallback(row) for row in rows),
            "rewrite_sources": dict(Counter(row.get("rewrite_source", "error") for row in rows)),
            **{key: mean([row[key] for row in rows if row.get(key) is not None])
               for key in ("hit_at_k", "recall_at_k", "gold_recall_at_k", "mrr_at_k", "ndcg_at_k",
                           "gold_recall_at_1", "gold_recall_at_3", "mrr_at_3", "keyword_coverage", "answer_rouge1_f1")},
            "mean_wall_time_ms": mean([row["wall_time_ms"] for row in rows]),
            "mean_latencies_ms": {key: mean([float(row["latencies_ms"][key]) for row in rows
                if isinstance(row.get("latencies_ms", {}).get(key), (int, float))]) for key in keys},
        })
    return summaries


def evaluate(engine, examples, strategies, top_k, generate, *, cache=None):
    strategies = list(dict.fromkeys(strategies))
    allowed = {f"{strategy}:{example['id']}" for strategy in strategies for example in examples}
    details = list(cache.rows) if cache else []
    if any(row.get("cache_key") not in allowed for row in details):
        raise ValueError("Checkpoint contains rows outside this fixed experiment")
    done = {row.get("cache_key") for row in details}
    for strategy in strategies:
        for index, example in enumerate(examples, 1):
            key = f"{strategy}:{example['id']}"
            if key in done:
                continue
            qrecc = "positive_ids" in example
            row = {
                "id": example["id"], "conversation_id": example.get("conversation_id"),
                "split": example["split"], "strategy": strategy,
                "question": example["question"], "history": example["history"],
                "gold_document": example.get("gold_document"), "positive_ids": example.get("positive_ids"),
                "human_rewrite": example.get("human_rewrite"),
                "reference_answer": example.get("reference_answer", ""), "answer_keywords": example.get("answer_keywords", []),
                "metric_level": "passage" if qrecc else "source_document_at_chunk_ranks",
                "hit_at_k": 0.0, "recall_at_k": 0.0, "mrr_at_k": 0.0,
                "gold_recall_at_k": 0.0 if qrecc else None, "ndcg_at_k": 0.0 if qrecc else None,
                "gold_recall_at_1": 0.0 if qrecc else None,
                "gold_recall_at_3": 0.0 if qrecc and top_k >= 3 else None,
                "mrr_at_3": 0.0 if qrecc and top_k >= 3 else None,
                "keyword_coverage": 0.0 if generate and example.get("answer_keywords") else None,
                "answer_rouge1_f1": 0.0 if generate and example.get("reference_answer") else None,
                "rewrite_source": "error", "rewrite_error": None, "rewrite_trace": {},
                "latencies_ms": {}, "error": None,
            }
            started, stage = time.perf_counter(), "retrieval"
            try:
                result = engine.retrieve(**public_input(example), strategy=strategy, top_k=top_k)
                hits = result.get("hits", result.get("sources", []))
                row.update({"hits": hits, "search_query": result.get("search_query", ""),
                            "rewrite_source": result.get("rewrite_source", "unknown"),
                            "rewrite_error": result.get("rewrite_error"), "rewrite_trace": result.get("rewrite_trace", {}),
                            "latencies_ms": dict(result.get("latencies_ms", {}))})
                # Keep actual inputs and runtime metadata for a later evidence
                # audit; the reference history above may exceed the model cap.
                row.update({field: result[field] for field in (
                    "visible_history", "mode", "dataset", "backend"
                ) if field in result})
                stage = "metrics"
                if qrecc:
                    row.update(passage_metrics(hits, example["positive_ids"], top_k))
                    row["recall_at_k"] = row["gold_recall_at_k"]
                    row["gold_recall_at_1"] = passage_metrics(hits, example["positive_ids"], 1)["gold_recall_at_k"]
                    if top_k >= 3:
                        at_3 = row if top_k == 3 else passage_metrics(hits, example["positive_ids"], 3)
                        row.update(gold_recall_at_3=at_3["gold_recall_at_k"], mrr_at_3=at_3["mrr_at_k"])
                else:
                    hit, mrr = retrieval_metrics(hits, example["gold_document"], top_k)
                    row.update(hit_at_k=hit, recall_at_k=hit, mrr_at_k=mrr)
                if generate:
                    stage = "generation"
                    result = engine.answer_from_retrieval(result, example["history"])
                    row["answer"] = result.get("answer", "")
                    row["latencies_ms"] = dict(result.get("latencies_ms", {}))
                    # A citation warning is a successful API response requiring
                    # review, not an API failure. Preserve what the LLM saw.
                    row.update({field: result[field] for field in (
                        "context", "sources", "warnings", "mode", "model", "backend", "dataset"
                    ) if field in result})
                    row["keyword_coverage"] = keyword_coverage(row["answer"], row["answer_keywords"])
                    row["answer_rouge1_f1"] = (rouge_scores(row["answer"], row["reference_answer"])["rouge1_f1"]
                                                    if row["reference_answer"] else None)
            except Exception as error:
                row["error"] = f"{type(error).__name__}: {error}"
                row["error_stage"] = stage
                print(f"\n[{strategy}/{example['id']}] {stage}: {row['error']}", file=sys.stderr)
            row["wall_time_ms"] = (time.perf_counter() - started) * 1000
            row["cache_key"] = key
            if cache:
                cache.append(row, key)
            details.append(row)
            done.add(key)
            print(f"\r{strategy}: {index}/{len(examples)}", end="", flush=True)
        print()
    order = {f"{strategy}:{example['id']}": i for i, (strategy, example) in enumerate(
        (strategy, example) for strategy in strategies for example in examples)}
    details.sort(key=lambda row: order[row["cache_key"]])
    return summarize(details, strategies), details


def save_report(output_dir, metadata, summaries, details):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "report.json").open("x", encoding="utf-8") as handle:
        json.dump({"metadata": metadata, "summary": summaries, "details": details}, handle, ensure_ascii=False, indent=2)
    for name, rows in (("summary", summaries), ("details", details)):
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with (output_dir / f"{name}.csv").open("x", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def corpus_fingerprint(path):
    path = Path(path)
    if path.is_file():
        return {"format": "file", "sha256": file_sha256(path)}
    if not path.is_dir():
        raise ValueError(f"Corpus does not exist: {path}")
    files = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in {".md", ".txt"})
    if any(p.is_symlink() or not p.resolve().is_relative_to(path.resolve()) for p in files):
        raise ValueError("External corpus links are not allowed")
    return {"format": "directory", "files": {p.relative_to(path).as_posix(): file_sha256(p) for p in files}}


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_DIR / path


def parse_args():
    parser = argparse.ArgumentParser(description="Controlled QReCC retrieval evaluation")
    parser.add_argument("--dataset", choices=["qrecc"], default="qrecc")
    parser.add_argument("--mode", choices=["demo", "ollama"], default="demo")
    parser.add_argument("--model", help="Installed Ollama model tag, overriding config.json for this run")
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=["current", "history", "rewrite"])
    parser.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--overlap", type=int)
    parser.add_argument("--backend", choices=["bm25", "dense", "hybrid"])
    parser.add_argument("--bm25-mode", choices=["inverted", "scan"])
    parser.add_argument("--data", type=Path)
    parser.add_argument("--limit", type=int, help="Debugging prefix only; not representative random sampling")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--generate", action="store_true")
    group.add_argument("--retrieval-only", action="store_true")
    args = parser.parse_args()
    if args.top_k is not None and not 1 <= args.top_k <= 20:
        parser.error("--top-k must be in [1,20]")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.resume and args.output_dir is None:
        parser.error("--resume requires --output-dir")
    return args


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = parse_args()
    try:
        overrides = {"model": args.model, "chunk_size": args.chunk_size, "overlap": args.overlap,
                     "retriever_backend": args.backend, "bm25_mode": args.bm25_mode, "top_k": args.top_k}
        config = load_config(PROJECT_DIR, {k: v for k, v in overrides.items() if v is not None}, profile=args.dataset)
        if args.mode == "demo" and args.backend in {None, "bm25"}:
            config = offline_config(config)
        data_path = args.data or resolve_path(config["eval_path"])
        corpus_path = resolve_path(config["corpus_path"])
        uses_passages = config["corpus_format"] == "passages_jsonl"
        ids = load_passage_ids(corpus_path) if uses_passages else None
        examples = load_examples(data_path, args.split, args.limit, corpus_ids=ids)
        dataset_type = examples[0]["dataset_type"]
        if (dataset_type == "qrecc") != uses_passages:
            raise ValueError("Evaluation schema and corpus format do not match")
        strategies = list(dict.fromkeys(args.strategies))
        output = args.output_dir or PROJECT_DIR / "outputs" / "runs" / datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
        # Resolve the actual installed weights before validating a checkpoint;
        # the same mutable Ollama model tag can later identify different bytes.
        model_identity = runtime_model_identity(config, args.mode)
        embedding_identity = runtime_embedding_identity(config)
        if embedding_identity:
            config["embedding_digest"] = embedding_identity["digest"]
        identity = {"task": "retrieval", "dataset": args.dataset, "data_sha256": file_sha256(data_path),
                    "corpus": corpus_fingerprint(corpus_path), "config": config, "mode": args.mode,
                    "runtime_model_identity": model_identity,
                    "runtime_embedding_identity": embedding_identity,
                    "split": args.split, "strategies": strategies, "generate": args.generate,
                    "sample_ids": [row["id"] for row in examples], "code": code_hashes(PROJECT_DIR)}
        cache = RunCache(output, identity, resume=args.resume)
        if (output / "report.json").exists():
            raise ValueError("Run already has a final report; use a new output directory")
        started = time.perf_counter()
        engine = RAGEngine(PROJECT_DIR, config, mode=args.mode)
        initialization_ms = (time.perf_counter() - started) * 1000
        summaries, details = evaluate(engine, examples, strategies, config["top_k"], args.generate, cache=cache)
        metadata = {
            "created_at": datetime.now().astimezone().isoformat(), "dataset": str(data_path.resolve()),
            "dataset_type": dataset_type, "sample_count_per_strategy": len(examples), "split": args.split,
            "mode": args.mode, "generate": args.generate, "limit": args.limit, "config": config,
            "run_signature": cache.signature, "initialization_ms": initialization_ms,
            "runtime_model_identity": model_identity,
            "runtime_embedding_identity": embedding_identity,
            "metric_definition": ("QReCC: original passage IDs; Hit@K is any positive, gold_recall@K is fraction of unique indexed positives, MRR uses original first-positive rank, nDCG uses binary unique positives."
                if uses_passages else "Single gold source-document hit in top K chunks; MRR uses original chunk ranks. recall_at_k is this single-gold hit."),
            "failure_policy": "All fixed samples remain in denominator. Retrieval errors score zero; later generation errors preserve retrieval scores. Fallbacks separately counted.",
            "history_definition": "Same bounded explicit-role reference history for all strategies; not model-generated rolling dialogue.",
            "latency_definition": "Stage times exclude initialization; resumed rows retain their original measured latency. Model cold starts and service variation affect timing.",
            "quality_warning": "Keyword coverage and answer ROUGE are lexical proxies, not factual accuracy or evidence support.",
            "demo_warning": "Demo rewrites are heuristics and answers are extracts, not LLM experiment results." if args.mode == "demo" else None,
            "qrecc_warning": "Fixed coverage-conditioned subset, not full-corpus QReCC. Supplied test cases were previously inspected and are diagnostic." if uses_passages else None,
        }
        save_report(output, metadata, summaries, details)
    except (ValueError, OSError, RuntimeError) as error:
        print(f"Evaluation failed: {error}", file=sys.stderr)
        return 1
    print(f"\nmode={args.mode}, backend={config['retriever_backend']}, dataset={args.dataset}, n={len(examples)}, K={config['top_k']}")
    for summary in summaries:
        print(f"{summary['strategy']:<12} Hit@K={summary['hit_at_k']:.4f} MRR@K={summary['mrr_at_k']:.4f} errors={summary['error_count']} fallbacks={summary['fallback_count']}")
        if summary["gold_recall_at_3"] is not None:
            print(f"             Gold Recall@1={summary['gold_recall_at_1']:.4f} Gold Recall@3={summary['gold_recall_at_3']:.4f} MRR@3={summary['mrr_at_3']:.4f}")
    print(f"Saved: {output.resolve()}")
    if args.mode == "demo":
        print("DEMO: heuristic query construction only; not LLM-rewrite results.")
    return int(any(row["error_count"] or row["rewrite_error_count"] for row in summaries))


if __name__ == "__main__":
    raise SystemExit(main())
