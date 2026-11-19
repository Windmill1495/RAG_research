"""Signed fixed-history experiments; generation failures preserve retrieval scores."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import sys
import time
import tempfile
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from rag.config import (add_compcqr_arguments, compcqr_overrides, load_config,
                        load_experiment_config, offline_config, validate_experiment_data)
from rag.data import load_examples, load_passage_ids, public_input
from rag.metrics import document_metrics, passage_metrics, rouge_scores
from rag.llm import runtime_embedding_identity, runtime_model_identity
from rag.pipeline import RAGEngine
from rag.rewriters import ALL_STRATEGIES
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


def method_errors(row):
    """Degraded multi-query stages remain visible even when a ranking survives."""
    trace = row.get("rewrite_trace") or {}
    errors = []
    for field in ("component_errors", "selection_error"):
        value = trace.get(field)
        if isinstance(value, (list, tuple)):
            errors.extend(str(item) for item in value if item)
        elif value:
            errors.append(str(value))
    return errors


def compcqr_execution_scope(row, *, retrieval_attempted):
    """Describe the stages actually completed, not merely the selected method."""
    trace = row.get("rewrite_trace") or {}
    if ((row.get("error") and row.get("error_stage") != "generation")
            or row.get("retrieval_error") or row.get("rewrite_error")
            or method_errors(row) or is_fallback(row)):
        return "compcqr_degraded"
    if row.get("rewrite_source") == "compcqr_demo" or trace.get("demo_note"):
        return "compcqr_demo"
    if not retrieval_attempted:
        return "component_generation_only"
    if trace.get("selection_applied"):
        return "full_compcqr"
    if trace.get("selection_skipped") == "disabled_by_config":
        return "compcqr_rrf_only_ablation"
    return "compcqr_degraded"


def compact_experiment_record(row, engine):
    """Avoid 325 repeated query strings per turn in full-dataset checkpoints."""
    config = getattr(engine, "config", {})
    if (row.get("strategy") != "compcqr" or not isinstance(config, dict)
            or config.get("trace_detail", "full") != "summary"):
        return
    trace = dict(row.get("rewrite_trace") or {})
    trace.pop("queries", None)
    row.pop("search_queries", None)
    row["query_storage"] = "reconstruct_from_components_and_code_signature"
    outputs = trace.get("raw_responses", [])
    if (not method_errors(row) and not row.get("rewrite_error") and not is_fallback(row)
            and sum(len(str(value)) for value in outputs) > 24000):
        trace["raw_response_sha256"] = [hashlib.sha256(str(value).encode("utf-8")).hexdigest()
                                        for value in outputs]
        trace.pop("raw_responses", None)
        trace["raw_response_storage"] = "large_success_outputs_omitted_in_summary_mode"
    row["rewrite_trace"] = trace


def summarize(details, strategies):
    summaries = []
    for strategy in strategies:
        rows = [row for row in details if row["strategy"] == strategy]
        keys = sorted({key for row in rows for key in row.get("latencies_ms", {})})
        summaries.append({
            "strategy": strategy, "sample_count": len(rows),
            "retrieval_sample_count": sum(row.get("retrieval_eligible", True) for row in rows),
            "unlabeled_sample_count": sum(not row.get("retrieval_eligible", True) for row in rows),
            "metric_level": rows[0]["metric_level"] if rows else None,
            "error_count": sum(row.get("error") is not None for row in rows),
            "retrieval_error_count": sum(row.get("error_stage") in {"retrieval", "metrics"} for row in rows),
            "generation_error_count": sum(row.get("error_stage") == "generation" for row in rows),
            "rewrite_error_count": sum(bool(row.get("rewrite_error")) for row in rows),
            "fallback_count": sum(is_fallback(row) for row in rows),
            "method_error_count": sum(bool(method_errors(row)) for row in rows),
            "selection_error_count": sum(bool(row.get("rewrite_trace", {}).get("selection_error")) for row in rows),
            "llm_calls": sum(row.get("rewrite_trace", {}).get("llm_calls", 0) for row in rows),
            "mean_query_count": mean([row.get("rewrite_trace", {}).get("query_count", 1) for row in rows]),
            "execution_scopes": dict(Counter(row.get("execution_scope", "single_query") for row in rows)),
            "rewrite_sources": dict(Counter(row.get("rewrite_source", "error") for row in rows)),
            **{key: mean([row[key] for row in rows if row.get(key) is not None])
               for key in ("hit_at_k", "recall_at_k", "gold_recall_at_k", "mrr_at_k", "ndcg_at_k",
                           "gold_recall_at_1", "gold_recall_at_3", "mrr_at_3", "ndcg_at_3", "keyword_coverage", "answer_rouge1_f1")},
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
            qrecc = example.get("dataset_type") == "qrecc" or "positive_ids" in example
            labeled = bool(example.get("positive_ids")) if qrecc else True
            row = {
                "id": example["id"], "conversation_id": example.get("conversation_id"), "turn": example.get("turn"),
                "split": example["split"], "strategy": strategy,
                "question": example["question"], "history": example["history"],
                "gold_document": example.get("gold_document"), "positive_ids": example.get("positive_ids"),
                "human_rewrite": example.get("human_rewrite"),
                "reference_answer": example.get("reference_answer", ""), "answer_keywords": example.get("answer_keywords", []),
                "metric_level": "passage" if qrecc else "source_document_at_chunk_ranks",
                "retrieval_eligible": labeled,
                "hit_at_k": 0.0 if labeled else None, "recall_at_k": 0.0 if labeled else None,
                "mrr_at_k": 0.0 if labeled else None,
                "gold_recall_at_k": 0.0 if qrecc and labeled else None,
                "ndcg_at_k": 0.0 if qrecc and labeled else None,
                "gold_recall_at_1": 0.0 if qrecc and labeled else None,
                "gold_recall_at_3": 0.0 if qrecc and labeled and top_k >= 3 else None,
                "mrr_at_3": 0.0 if qrecc and labeled and top_k >= 3 else None,
                "ndcg_at_3": 0.0 if qrecc and labeled and top_k >= 3 else None,
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
                    "visible_history", "mode", "dataset", "backend", "search_queries", "components"
                ) if field in result})
                stage = "metrics"
                if qrecc and labeled:
                    row.update(passage_metrics(hits, example["positive_ids"], top_k))
                    row["recall_at_k"] = row["gold_recall_at_k"]
                    row["gold_recall_at_1"] = passage_metrics(hits, example["positive_ids"], 1)["gold_recall_at_k"]
                    if top_k >= 3:
                        at_3 = row if top_k == 3 else passage_metrics(hits, example["positive_ids"], 3)
                        row.update(gold_recall_at_3=at_3["gold_recall_at_k"], mrr_at_3=at_3["mrr_at_k"],
                                   ndcg_at_3=at_3["ndcg_at_k"])
                elif not qrecc:
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
            row["execution_scope"] = (compcqr_execution_scope(row, retrieval_attempted=True)
                                      if strategy == "compcqr" else "single_query")
            row["wall_time_ms"] = (time.perf_counter() - started) * 1000
            row["cache_key"] = key
            compact_experiment_record(row, engine)
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


@contextmanager
def atomic_export(path, *, encoding="utf-8", newline=None):
    """Publish one derived export only after its complete write reaches disk."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding=encoding, newline=newline,
                                         dir=path.parent, prefix="." + path.name + ".",
                                         suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def completed_report(output_dir, run_signature=None):
    """Return only a final report whose declared derived exports are complete.

    A malformed/interrupted export is regenerable from an already verified
    RunCache. This function never changes the canonical checkpoint.
    """
    directory = Path(output_dir)
    try:
        report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
        if (not isinstance(report, dict) or report.get("export_version") != 1
                or report.get("export_complete") is not True):
            return None
        if run_signature is not None and report.get("metadata", {}).get("run_signature") != run_signature:
            return None
        artifacts = report.get("artifacts")
        if (not isinstance(artifacts, dict) or not {"summary.csv", "details.csv"} <= artifacts.keys()
                or not isinstance(report.get("summary"), list) or not isinstance(report.get("details"), list)):
            return None
        for name, digest in artifacts.items():
            if not isinstance(name, str) or Path(name).name != name or name in {".", ".."}:
                return None
            # Reviewer copies are intentionally edited after export. A valid
            # final marker must not turn those ratings into blank templates.
            mutable = name in {"blind_review_reviewer1.csv", "blind_review_reviewer2.csv"}
            if (not (directory / name).is_file()
                    or not mutable and file_sha256(directory / name) != digest):
                return None
        return report
    except (ValueError, OSError, UnicodeError, AttributeError, TypeError):
        return None


def save_report(output_dir, metadata, summaries, details, *, extra_artifacts=()):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("summary", summaries), ("details", details)):
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with atomic_export(output_dir / f"{name}.csv", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value for key, value in row.items()})
    names = ["summary.csv", "details.csv", *extra_artifacts]
    artifacts = {name: file_sha256(output_dir / name) for name in names}
    # report.json is the commit marker and must be published last. Every other
    # derived file can safely be regenerated after signed checkpoint recovery.
    with atomic_export(output_dir / "report.json") as handle:
        json.dump({"export_version": 1, "export_complete": True, "artifacts": artifacts,
                   "metadata": metadata, "summary": summaries, "details": details},
                  handle, ensure_ascii=False, indent=2)


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


def prepared_dataset_identity(config, data_path):
    """Prove a full run uses all prepared official questions, not the pilot file."""
    if config.get("evaluation_scope") != "full":
        return {"scope": "pilot", "data_sha256": file_sha256(data_path)}
    manifest_path = resolve_path(config["dataset_manifest_path"])
    if not manifest_path.is_file():
        raise ValueError("Full QReCC manifest missing. Run prepare_qrecc_full.py on the server first.")
    manifest = validate_experiment_data(config, data_path)
    actual = file_sha256(data_path)
    if not manifest.get("complete") or manifest.get("scope") != "official_full_questions":
        raise ValueError("Prepared manifest does not describe complete official QReCC questions")
    if actual != manifest.get("questions_sha256"):
        raise ValueError("Full QReCC question file does not match its prepared manifest")
    return {"scope": "full_official_questions", "questions_sha256": actual,
            "manifest_sha256": file_sha256(manifest_path),
            "question_count": manifest.get("question_count"), "split_counts": manifest.get("split_counts")}


def retrieval_coverage_for_config(config, examples, *, retriever=None, allow_partial_corpus=False):
    """Check only annotated IDs against a remote index; never load its full ID set."""
    labels = {row["id"]: set(row["positive_ids"]) for row in examples if row.get("positive_ids")}
    remote = config.get("retrieval_store", "local") == "opensearch"
    if remote:
        if retriever is None:
            raise ValueError("Remote coverage requires a validated full-index retriever")
        requested = set().union(*labels.values()) if labels else set()
        indexed_ids = retriever.contains_passage_ids(requested) if requested else set()
        identity = retriever.index_identity()
        if config.get('evaluation_scope') == 'full':
            from rag.hf_corpus import validate_index_corpus
            validate_index_corpus(config, identity)
        eligible = set(labels)
        count = identity.get("indexed_passage_count", identity.get("count", identity.get("documents")))
        scope = "configured_full_index"
    else:
        corpus_path = resolve_path(config["corpus_path"])
        indexed_ids = load_passage_ids(corpus_path)
        identity = corpus_fingerprint(corpus_path)
        eligible = {identifier for identifier, positives in labels.items() if positives & indexed_ids}
        count = len(indexed_ids)
        scope = ("bundled_passage_subset" if corpus_path.resolve() ==
                 (PROJECT_DIR / "data/qrecc/passages.jsonl").resolve() else "configured_passage_corpus")
    incomplete = sum(not positives <= indexed_ids for positives in labels.values())
    if incomplete and not allow_partial_corpus:
        raise ValueError(f"{incomplete} samples have gold passages outside the configured index. "
                         "Use a complete matching index; --allow-partial-corpus is diagnostic only.")
    return eligible, {"scope": scope, "backend": config["retriever_backend"],
                      "corpus_fingerprint": identity, "indexed_passage_count": count,
                      "annotated_sample_count": len(labels), "unlabeled_sample_count": len(examples) - len(labels),
                      "scored_sample_count": len(eligible), "samples_with_missing_gold": incomplete,
                      "samples_without_indexed_gold": sum(not bool(p & indexed_ids) for p in labels.values()),
                      "allow_partial_corpus": allow_partial_corpus,
                      "coverage_policy": "all_annotated" if remote else "indexed_positive_required"}


def parse_args():
    parser = argparse.ArgumentParser(description="Controlled QReCC retrieval evaluation")
    parser.add_argument("--dataset", choices=["qrecc"], default="qrecc")
    parser.add_argument("--mode", choices=["demo", "ollama"], default="ollama")
    parser.add_argument("--scope", choices=["full", "pilot"], default="full")
    parser.add_argument("--config", type=Path, help="Machine-local configuration overrides")
    parser.add_argument("--trace-detail", choices=["summary", "full"], default=None)
    parser.add_argument("--model", help="Installed Ollama model tag, overriding config.json for this run")
    parser.add_argument("--adaqr-model-path", type=Path, help="Trained AdaQR DPO adapter directory")
    parser.add_argument("--strategies", nargs="+", choices=ALL_STRATEGIES, default=["current", "history", "rewrite"])
    parser.add_argument("--split", choices=["train", "dev", "test", "all"], default="test")
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--overlap", type=int)
    parser.add_argument("--backend", choices=["bm25"])
    parser.add_argument("--bm25-mode", choices=["inverted", "scan"])
    parser.add_argument("--data", type=Path)
    parser.add_argument("--limit", type=int, help="Debugging prefix only; not representative random sampling")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-partial-corpus", action="store_true",
                        help="Explicit diagnostic only; never claim official full-corpus results")
    add_compcqr_arguments(parser)
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
                     "retriever_backend": args.backend, "bm25_mode": args.bm25_mode, "top_k": args.top_k,
                     "adaqr_model_path": str(args.adaqr_model_path.expanduser().resolve()) if args.adaqr_model_path else None}
        overrides.update(compcqr_overrides(args))
        config = load_experiment_config(PROJECT_DIR, {k: v for k, v in overrides.items() if v is not None},
                                        scope=args.scope, config_path=args.config)
        config["trace_detail"] = args.trace_detail or ("summary" if args.scope == "full" else "full")
        if args.scope == "full" and args.mode != "ollama":
            raise ValueError("Full experiments require --mode ollama; use --scope pilot for offline checks")
        if args.mode == "demo" and args.backend in {None, "bm25"}:
            config = offline_config(config)
        data_path = args.data or resolve_path(config["eval_path"])
        if not data_path.is_file():
            raise ValueError("Question data missing. Run prepare_qrecc_full.py on the server first, or select --scope pilot.")
        prepared = prepared_dataset_identity(config, data_path)
        uses_passages = config["corpus_format"] == "passages_jsonl"
        examples = load_examples(data_path, args.split, args.limit)
        dataset_type = examples[0]["dataset_type"]
        if (dataset_type == "qrecc") != uses_passages:
            raise ValueError("Evaluation schema and corpus format do not match")
        strategies = list(dict.fromkeys(args.strategies))
        adaqr_identity = None
        if "adaqr" in strategies:
            if args.mode != "ollama" or not config.get("adaqr_model_path"):
                raise ValueError("AdaQR requires --mode ollama and --adaqr-model-path")
            from rag.adaqr_model import adapter_identity
            adaqr_identity = adapter_identity(config["adaqr_model_path"])
        output = args.output_dir or PROJECT_DIR / "outputs" / "runs" / datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
        model_identity = runtime_model_identity(config, args.mode)
        embedding_identity = runtime_embedding_identity(config)
        if embedding_identity:
            config["embedding_digest"] = embedding_identity["digest"]
        started = time.perf_counter()
        engine = RAGEngine(PROJECT_DIR, config, mode=args.mode)
        initialization_ms = (time.perf_counter() - started) * 1000
        coverage = None
        if uses_passages:
            _, coverage = retrieval_coverage_for_config(config, examples, retriever=engine.retriever,
                                                        allow_partial_corpus=args.allow_partial_corpus)
            corpus_identity = coverage["corpus_fingerprint"]
        else:
            corpus_identity = corpus_fingerprint(resolve_path(config["corpus_path"]))
        identity = {"task": "retrieval", "dataset": args.dataset, "data_sha256": file_sha256(data_path),
                    "prepared_dataset": prepared, "corpus": corpus_identity, "config": config, "mode": args.mode,
                    "runtime_model_identity": model_identity, "adaqr_adapter_identity": adaqr_identity,
                    "runtime_embedding_identity": embedding_identity,
                    "split": args.split, "strategies": strategies, "generate": args.generate,
                    "sample_ids": [row["id"] for row in examples], "code": code_hashes(PROJECT_DIR)}
        cache = RunCache(output, identity, resume=args.resume)
        if completed_report(output, cache.signature) is not None:
            raise ValueError("Run already has a final report; use a new output directory")
        summaries, details = evaluate(engine, examples, strategies, config["top_k"], args.generate, cache=cache)
        metadata = {
            "created_at": datetime.now().astimezone().isoformat(), "dataset": str(data_path.resolve()),
            "dataset_type": dataset_type, "sample_count_per_strategy": len(examples), "split": args.split,
            "mode": args.mode, "generate": args.generate, "limit": args.limit, "config": config,
            "run_signature": cache.signature, "initialization_ms": initialization_ms,
            "scope": args.scope, "prepared_dataset": prepared, "retrieval_coverage": coverage,
            "split_warning": "Includes training/development data; not independent test evaluation." if args.split == "all" else None,
            "first_turn_count": sum(not row["history"] for row in examples),
            "runtime_model_identity": model_identity, "adaqr_adapter_identity": adaqr_identity,
            "runtime_embedding_identity": embedding_identity,
            "metric_definition": ("QReCC: original passage IDs; Hit@K is any positive, gold_recall@K is fraction of unique annotated positives, MRR uses original first-positive rank, nDCG uses binary unique positives."
                if uses_passages else "Single gold source-document hit in top K chunks; MRR uses original chunk ranks. recall_at_k is this single-gold hit."),
            "failure_policy": "Every annotated sample remains in the retrieval denominator, including failures (zero). Unannotated turns have null retrieval metrics. Later generation errors preserve retrieval scores; fallbacks are separately counted.",
            "history_definition": "Same bounded explicit-role reference history for all strategies; not model-generated rolling dialogue.",
            "latency_definition": "Stage times exclude initialization; resumed rows retain their original measured latency. Model cold starts and service variation affect timing.",
            "quality_warning": "Keyword coverage and answer ROUGE are lexical proxies, not factual accuracy or evidence support.",
            "compcqr_policy": "CompCQR scores its final fused and selected passage ranking; query count and stage errors are reported. Retrieval uses the corpus and backend recorded in this report." if "compcqr" in strategies else None,
            "demo_warning": "Demo rewrites are heuristics and answers are extracts, not LLM experiment results." if args.mode == "demo" else None,
            "qrecc_warning": ("Pilot subset, not full-corpus QReCC." if args.scope == "pilot" else
                               "Full prepared official questions; retrieval metrics exclude unannotated turns. A debug --limit is not a full test result."),
        }
        save_report(output, metadata, summaries, details)
    except (ValueError, OSError, RuntimeError) as error:
        print(f"Evaluation failed: {error}", file=sys.stderr)
        return 1
    print(f"\nmode={args.mode}, backend={config['retriever_backend']}, dataset={args.dataset}, n={len(examples)}, K={config['top_k']}")
    for summary in summaries:
        scores = (f"Hit@K={summary['hit_at_k']:.4f} MRR@K={summary['mrr_at_k']:.4f}"
                  if summary["hit_at_k"] is not None else "No annotated retrieval samples (metrics null)")
        print(f"{summary['strategy']:<12} {scores} errors={summary['error_count']} fallbacks={summary['fallback_count']}")
        if summary["gold_recall_at_3"] is not None:
            print(f"             Gold Recall@1={summary['gold_recall_at_1']:.4f} Gold Recall@3={summary['gold_recall_at_3']:.4f} MRR@3={summary['mrr_at_3']:.4f}")
    print(f"Saved: {output.resolve()}")
    if args.mode == "demo":
        print("DEMO: heuristic query construction only; not LLM-rewrite results.")
    return int(any(row["error_count"] or row["rewrite_error_count"] or row["method_error_count"] for row in summaries))


if __name__ == "__main__":
    raise SystemExit(main())
