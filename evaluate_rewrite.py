"""Evaluate standalone rewrites and the passages retrieved using those rewrites.

ROUGE is token overlap, not semantic accuracy. Same fixed samples and histories
are used for all strategies. Blind-review CSVs accompany automatic scores.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from evaluate import PROJECT_DIR, compcqr_execution_scope, corpus_fingerprint, is_fallback, mean, method_errors, resolve_path, save_report
from rag.config import add_compcqr_arguments, compcqr_overrides, load_config, offline_config
from rag.data import load_passage_ids, public_input
from rag.metrics import NORMALIZATION, passage_metrics, rouge_scores
from rag.llm import runtime_embedding_identity, runtime_model_identity
from rag.pipeline import HistorySelector, RAGEngine
from rag.rewrite_data import REWRITE_DATA_FORMATS, load_rewrite_examples
from rag.rewriters import RESEARCH_STRATEGIES, STRATEGIES
from rag.run_cache import RunCache, code_hashes, file_sha256


# A future group can register a selector here and reuse B's complete rewrite flow.
HISTORY_SELECTORS: dict[str, HistorySelector] = {}
RETRIEVAL_TOP_K = 3
RETRIEVAL_METRICS = ("gold_recall_at_1", "gold_recall_at_3", "mrr_at_3", "ndcg_at_3")


def evaluate_rewrites(engine, examples, strategies, *, include_l=False, cache=None, progress=True,
                      retrieval_ids: set[str] | None = None, retriever=None):
    if any(not isinstance(row.get("human_rewrite"), str) or not row["human_rewrite"].strip() for row in examples):
        raise ValueError("Every selected sample must have a human_rewrite reference")
    strategies = list(dict.fromkeys(strategies))
    allowed = {f"{strategy}:{example['id']}" for strategy in strategies for example in examples}
    details = list(cache.rows) if cache else []
    if any(row.get("cache_key") not in allowed for row in details):
        raise ValueError("Checkpoint contains records outside this rewrite experiment")
    done = {row.get("cache_key") for row in details}
    if retrieval_ids and retriever is None:
        retriever = engine.retriever
    for strategy in strategies:
        selector = HISTORY_SELECTORS.get(strategy)
        options = ({"strategy": "structured", "history_selector": selector}
                   if selector is not None else {"strategy": strategy})
        for index, example in enumerate(examples, 1):
            key = f"{strategy}:{example['id']}"
            if key in done:
                continue
            started = time.perf_counter()
            row = {
                "id": example["id"], "conversation_id": example.get("conversation_id"),
                "split": example["split"], "strategy": strategy,
                "question": example["question"], "history": example["history"],
                "human_rewrite": example["human_rewrite"], "search_query": "",
                "rewrite_source": "error", "rewrite_error": None, "rewrite_trace": {},
                "latencies_ms": {}, "error": None,
                "rouge_scope": "base_rewrite_only" if strategy == "compcqr" else "standalone_query",
                "execution_scope": "component_generation_only" if strategy == "compcqr" else "single_query",
                **rouge_scores("", example["human_rewrite"], include_l=include_l),
            }
            retrieve = retrieval_ids is not None and example["id"] in retrieval_ids
            if retrieval_ids is not None:
                row.update(retrieval_eligible=retrieve, positive_ids=example.get("positive_ids", []),
                           hits=[], retrieval_error=None,
                           **{field: 0.0 if retrieve else None for field in RETRIEVAL_METRICS})
            try:
                # No answer, gold rewrite, or positive ID is sent to the engine.
                result = engine.rewrite_query(**public_input(example), **options)
                prediction = result.get("search_query", "")
                if not isinstance(prediction, str) or not prediction.strip():
                    raise ValueError("Rewriter returned an empty query")
                row.update(search_query=prediction, rewrite_source=result.get("rewrite_source", "unknown"),
                           rewrite_error=result.get("rewrite_error"), rewrite_trace=result.get("rewrite_trace", {}),
                           visible_history=result.get("visible_history", example["history"]),
                           latencies_ms=dict(result.get("latencies_ms", {})))
                row.update({field: result[field] for field in ("search_queries", "components") if field in result})
                if selector is not None:
                    row["history_window"] = result["history_window"]
                    row["selected_history_positions"] = result["selected_history_positions"]
                row.update(rouge_scores(prediction, example["human_rewrite"], include_l=include_l))
            except Exception as error:
                row["error"] = f"{type(error).__name__}: {error}"
                if progress:
                    print(f"\n[{strategy}/{example['id']}] {row['error']}", file=sys.stderr)
            if retrieve and not row["error"]:
                retrieval_started = time.perf_counter()
                try:
                    if strategy == "compcqr":
                        # Reuse the generated components. Searching only the base
                        # rewrite would discard CompCQR's fusion and selection.
                        result = engine.retrieve_rewritten(result, RETRIEVAL_TOP_K, retriever=retriever)
                        hits = result["hits"]
                        row.update(rewrite_trace=result.get("rewrite_trace", {}),
                                   latencies_ms=dict(result.get("latencies_ms", {})))
                    else:
                        hits = retriever.search(row["search_query"], RETRIEVAL_TOP_K)
                    retrieval_finished = time.perf_counter()
                    at_1 = passage_metrics(hits, row["positive_ids"], 1)
                    at_3 = passage_metrics(hits, row["positive_ids"], RETRIEVAL_TOP_K)
                    row.update(hits=hits, gold_recall_at_1=at_1["gold_recall_at_k"],
                               gold_recall_at_3=at_3["gold_recall_at_k"], mrr_at_3=at_3["mrr_at_k"],
                               ndcg_at_3=at_3["ndcg_at_k"])
                except Exception as error:
                    row["retrieval_error"] = f"{type(error).__name__}: {error}"
                    if progress:
                        print(f"\n[{strategy}/{example['id']}] retrieval: {row['retrieval_error']}", file=sys.stderr)
                else:
                    elapsed = (retrieval_finished - retrieval_started) * 1000
                    if strategy != "compcqr":
                        row["latencies_ms"]["retrieval"] = round(elapsed, 2)
                        if "rewrite" in row["latencies_ms"]:
                            row["latencies_ms"]["total"] = round(row["latencies_ms"]["rewrite"] + elapsed, 2)
            if strategy == "compcqr":
                row["execution_scope"] = compcqr_execution_scope(row, retrieval_attempted=retrieve)
            row["wall_time_ms"] = (time.perf_counter() - started) * 1000
            row["cache_key"] = key
            if cache:
                cache.append(row, key)
            details.append(row)
            done.add(key)
            if progress:
                print(f"\r{strategy}: {index}/{len(examples)}", end="", flush=True)
        if progress:
            print()
    order = {f"{strategy}:{example['id']}": i for i, (strategy, example) in enumerate(
        (strategy, example) for strategy in strategies for example in examples)}
    details.sort(key=lambda row: order[row["cache_key"]])
    score_fields = ["rouge1_precision", "rouge1_recall", "rouge1_f1"] + (["rougeL_f1"] if include_l else [])
    summaries = []
    for strategy in strategies:
        group = [row for row in details if row["strategy"] == strategy]
        pure = [row for row in group if not row["error"] and not is_fallback(row)
                and not row.get("rewrite_error") and not method_errors(row)
                and (str(row["rewrite_source"]).startswith("llm") or row["rewrite_source"] == "compcqr_llm")]
        phase_keys = sorted({key for row in group for key in row.get("latencies_ms", {})})
        summaries.append({
            "strategy": strategy, "sample_count": len(group),
            "error_count": sum(bool(row["error"]) for row in group),
            "rewrite_error_count": sum(bool(row.get("rewrite_error")) for row in group),
            "fallback_count": sum(is_fallback(row) for row in group),
            "repair_count": sum(bool(row["rewrite_trace"].get("repaired")) for row in group),
            "llm_calls": sum(row["rewrite_trace"].get("llm_calls", 0) for row in group),
            "method_error_count": sum(bool(method_errors(row)) for row in group),
            "selection_error_count": sum(bool(row["rewrite_trace"].get("selection_error")) for row in group),
            "mean_query_count": mean([row["rewrite_trace"].get("query_count", 1) for row in group]),
            "rouge_scope": "base_rewrite_only" if strategy == "compcqr" else "standalone_query",
            "execution_scopes": dict(Counter(row.get("execution_scope", "single_query") for row in group)),
            "rewrite_sources": dict(Counter(row["rewrite_source"] for row in group)),
            **{field: mean([row[field] for row in group]) for field in score_fields},
            "pure_llm_success_subset_n": len(pure),
            "pure_llm_success_subset_rouge1_f1": mean([row["rouge1_f1"] for row in pure]),
            "mean_wall_time_ms": mean([row["wall_time_ms"] for row in group]),
            "mean_rewrite_ms": mean([row["latencies_ms"]["rewrite"] for row in group
                                      if isinstance(row.get("latencies_ms", {}).get("rewrite"), (int, float))]),
            "mean_latencies_ms": {key: mean([row["latencies_ms"][key] for row in group
                if isinstance(row.get("latencies_ms", {}).get(key), (int, float))]) for key in phase_keys},
            "score_policy": "Main ROUGE averages all fixed samples (hard failures zero); strategy fallbacks retained and separately counted. Pure LLM success subset is diagnostic only.",
        })
        if retrieval_ids is not None:
            eligible = [row for row in group if row["retrieval_eligible"]]
            measured = [row["latencies_ms"] for row in eligible if "retrieval" in row["latencies_ms"]]
            summaries[-1].update(
                retrieval_sample_count=len(eligible),
                retrieval_error_count=sum(bool(row["retrieval_error"]) for row in eligible),
                mean_retrieval_ms=mean([times["retrieval"] for times in measured]),
                mean_query_total_ms=mean([times["total"] for times in measured if "total" in times]),
                **{field: mean([row[field] for row in eligible]) for field in RETRIEVAL_METRICS},
            )
    return summaries, details


def export_blind_review(output_dir, details, *, count=30, seed=20261102):
    """Same sample IDs for every method; method identity is in a separate key."""
    if count < 0:
        raise ValueError("Review count cannot be negative")
    identifiers = sorted({row["id"] for row in details})
    rng = random.Random(seed)
    selected = sorted(rng.sample(identifiers, min(count, len(identifiers))))
    rows = [row for row in details if row["id"] in selected]
    rng.shuffle(rows)
    fields = ["review_id", "sample_id", "question", "visible_history", "reference_rewrite", "prediction",
              "coreference_ellipsis", "intent_entities_constraints", "no_unsupported_addition", "standalone",
              "overall_pass_fail_na", "notes"]
    key = {}
    for reviewer in (1, 2):
        with (Path(output_dir) / f"blind_review_reviewer{reviewer}.csv").open("x", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for number, row in enumerate(rows, 1):
                review_id = f"R{number:04d}"
                key[review_id] = {"id": row["id"], "strategy": row["strategy"], "error": row["error"],
                                  "prediction_scope": row.get("rouge_scope", "standalone_query")}
                review_history = row.get("history_window", row.get("visible_history", row["history"]))
                writer.writerow({"review_id": review_id, "sample_id": row["id"], "question": row["question"],
                                 "visible_history": json.dumps(review_history, ensure_ascii=False),
                                 "reference_rewrite": row["human_rewrite"], "prediction": row["search_query"]})
    (Path(output_dir) / "blind_review_key.json").write_text(json.dumps({"seed": seed, "sample_ids": selected, "mapping": key}, ensure_ascii=False, indent=2), encoding="utf-8")
    (Path(output_dir) / "人工核查说明.md").write_text(
        "# 主实验改写人工核查\n\n分别把 reviewer1/reviewer2 CSV 交给两名评审，策略映射 key 只由分析者保管。\n"
        "同一批固定样本用于所有方法；等价措辞可以合格，不要求逐字等于参考。\n\n"
        "每项填写 pass/fail/na：指代与省略是否补全；意图、实体及时间/数量/否定约束是否保留；是否添加无历史依据的事实；能否独立理解。\n"
        "所有适用项通过才整体合格；空输出、调用失败也留在固定样本分母中。先独立评分，再讨论分歧并记录一致率。\n"
        "人工 Rewrite 不一定是唯一正确表达。ROUGE 和参考词面相似不等于语义正确，也不保证检索改善。\n"
        "包含 CompCQR 时，评审文本仅为其基础独立重写；不拼接所有候选 Query，也不代表完整多查询检索效果。\n",
        encoding="utf-8")


def retrieval_coverage(config, examples, *, allow_partial_corpus=False):
    """Freeze which samples can be scored and keep original gold denominators."""
    if config["corpus_format"] != "passages_jsonl":
        raise ValueError("QReCC passage retrieval requires a passages_jsonl corpus")
    corpus_path = resolve_path(config["corpus_path"])
    indexed_ids = load_passage_ids(corpus_path)
    labels = {row["id"]: set(row["positive_ids"]) for row in examples if row.get("positive_ids")}
    eligible = {identifier for identifier, positives in labels.items() if positives & indexed_ids}
    incomplete = sum(not positives <= indexed_ids for positives in labels.values())
    if incomplete and not allow_partial_corpus:
        raise ValueError(f"{incomplete} samples have gold passages outside the configured corpus. "
                         "Use --rewrite-only, a corpus with matching passage IDs, or "
                         "--allow-partial-corpus for a diagnostic run.")
    bundled = corpus_path.resolve() == (PROJECT_DIR / "data/qrecc/passages.jsonl").resolve()
    return eligible, {
        "backend": config["retriever_backend"], "top_k": RETRIEVAL_TOP_K,
        "corpus_path": str(corpus_path.resolve()), "corpus_fingerprint": corpus_fingerprint(corpus_path),
        "scope": "bundled_passage_subset" if bundled else "configured_passage_corpus",
        "indexed_passage_count": len(indexed_ids), "annotated_sample_count": len(labels),
        "unlabeled_sample_count": len(examples) - len(labels), "scored_sample_count": len(eligible),
        "samples_with_missing_gold": incomplete, "samples_without_indexed_gold": len(labels) - len(eligible),
        "allow_partial_corpus": allow_partial_corpus,
    }


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["qrecc"], default="qrecc")
    parser.add_argument("--mode", choices=["demo", "ollama"], default="demo")
    parser.add_argument("--strategies", nargs="+", choices=(*STRATEGIES, *HISTORY_SELECTORS),
                        default=RESEARCH_STRATEGIES, help="Defaults to A: rewrite, B: structured, C: compcqr")
    parser.add_argument("--model", help="Installed Ollama model tag, overriding config.json for this run")
    parser.add_argument("--split", choices=["train", "dev", "test", "all"],
                        help="Defaults to all for the configured data or official JSON, dev for custom JSONL")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--data-format", choices=REWRITE_DATA_FORMATS, default="auto")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--rouge-l", action="store_true")
    parser.add_argument("--review-count", type=int, default=30)
    parser.add_argument("--review-seed", type=int, default=20261102)
    parser.add_argument("--rewrite-only", action="store_true", help="Skip retrieval and score rewrite text only")
    parser.add_argument("--allow-partial-corpus", action="store_true",
                        help="Allow diagnostic scoring when the corpus lacks annotated passage IDs")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    add_compcqr_arguments(parser)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1 or args.review_count < 0:
        parser.error("limit must be positive and review-count nonnegative")
    if args.resume and not args.output_dir:
        parser.error("--resume requires --output-dir")
    if args.rewrite_only and args.allow_partial_corpus:
        parser.error("--allow-partial-corpus requires retrieval")
    try:
        overrides = {"model": args.model.strip()} if args.model is not None else {}
        overrides.update(compcqr_overrides(args))
        config = load_config(PROJECT_DIR, overrides, profile="qrecc")
        if args.mode == "demo":
            config = offline_config(config)
        data_path = args.data or resolve_path(config["eval_path"])
        data_format = args.data_format
        if data_format == "auto":
            data_format = "qrecc-json" if data_path.suffix.lower() == ".json" else "jsonl"
        split = args.split or ("all" if args.data is None or data_format == "qrecc-json" else "dev")
        examples = load_rewrite_examples(data_path, split, args.limit, data_format=data_format)
        retrieval_ids, retrieval = None, None
        if not args.rewrite_only:
            retrieval_ids, retrieval = retrieval_coverage(config, examples, allow_partial_corpus=args.allow_partial_corpus)
            embedding_identity = runtime_embedding_identity(config) if retrieval_ids else None
            if embedding_identity is not None:
                config = {**config, "embedding_digest": embedding_identity["digest"]}
                retrieval["embedding_model_identity"] = embedding_identity
        strategies = list(dict.fromkeys(args.strategies))
        output = args.output_dir or PROJECT_DIR / "outputs" / "rewrite_runs" / datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
        model_identity = runtime_model_identity(config, args.mode)
        identity = {"task": "rewrite_quality", "dataset": "qrecc", "data_sha256": file_sha256(data_path),
                    "config": config, "mode": args.mode, "split": split, "strategies": strategies,
                    "data_format": data_format, "retrieval": retrieval,
                    "runtime_model_identity": model_identity,
                    "sample_ids": [row["id"] for row in examples], "code": code_hashes(PROJECT_DIR),
                    "normalization": NORMALIZATION, "rouge_l": args.rouge_l,
                    "review_count": args.review_count, "review_seed": args.review_seed}
        cache = RunCache(output, identity, resume=args.resume)
        if (output / "report.json").exists():
            raise ValueError("Run already has a final report; choose a new output directory")
        initialization_started = time.perf_counter()
        engine = RAGEngine(PROJECT_DIR, config, mode=args.mode, build_index=bool(retrieval_ids))
        initialization_ms = (time.perf_counter() - initialization_started) * 1000
        summaries, details = evaluate_rewrites(engine, examples, strategies, include_l=args.rouge_l,
                                               cache=cache, retrieval_ids=retrieval_ids)
        metadata = {"task": "rewrite_quality", "dataset": str(data_path.resolve()),
                    "dataset_type": "qrecc", "data_format": data_format,
                    "mode": args.mode, "split": split,
                    "sample_splits": sorted({example["split"] for example in examples}),
                    "limit": args.limit,
                    "config": config, "run_signature": cache.signature, "normalization": NORMALIZATION,
                    "retrieval": retrieval, "initialization_ms": initialization_ms,
                    "runtime_model_identity": model_identity,
                    "created_at": datetime.now().astimezone().isoformat(),
                    "metric_warning": "Macro lexical ROUGE with no stemming/stop-word removal; not semantic accuracy. Equivalent wording may be valid.",
                    "compcqr_policy": "CompCQR ROUGE and blind review score its base independent rewrite only. With --rewrite-only, no retrieval/fusion/selection is executed. Otherwise eligible samples score the final fused/selected passage ranking; component and selection failures are separately counted." if "compcqr" in strategies else None,
                    "failure_policy": "All fixed samples are scored; hard failures have empty prediction and zero scores. Fallbacks separately counted; successful-only LLM subset is auxiliary.",
                    "retrieval_policy": "Score annotated samples with at least one indexed positive. Partial-coverage runs retain all annotated IDs in each scored sample's recall denominator. Failed rewrites or retrieval calls receive zero retrieval scores on eligible samples. Unlabeled and uncovered samples contribute only to rewrite quality." if retrieval else None,
                    "latency_warning": "Rewrite time includes component generation or history selection, model calls, validation and repair. CompCQR query total also includes multi-query retrieval, fusion and document selection. Times exclude initialization and scoring. Retrieval timing summaries use successful searches. Cold model loads and resumed rows affect timing.",
                    "demo_warning": "Heuristic demo only, not LLM rewrite-quality results." if args.mode == "demo" else None}
        export_blind_review(output, details, count=args.review_count, seed=args.review_seed)
        save_report(output, metadata, summaries, details)
    except (ValueError, OSError, RuntimeError) as error:
        print(f"Rewrite evaluation failed: {error}", file=sys.stderr)
        return 1
    for row in summaries:
        summary = f"{row['strategy']:<12} ROUGE-1 F1={row['rouge1_f1']:.4f} n={row['sample_count']} errors={row['error_count']} fallbacks={row['fallback_count']}"
        if row["strategy"] == "compcqr":
            summary += f" scope=base_rewrite_only method_errors={row['method_error_count']}"
        if row.get("retrieval_sample_count"):
            summary += (f" | Gold Recall@1={row['gold_recall_at_1']:.4f} Gold Recall@3={row['gold_recall_at_3']:.4f} "
                        f"MRR@3={row['mrr_at_3']:.4f} nDCG@3={row['ndcg_at_3']:.4f} retrieval_n={row['retrieval_sample_count']} "
                        f"retrieval_errors={row['retrieval_error_count']}")
        print(summary)
    if retrieval is not None:
        print(f"Passage scoring: {retrieval['scope']}, backend={retrieval['backend']}, "
              f"{retrieval['scored_sample_count']}/{len(examples)} samples per strategy.")
    print(f"Saved rewrite report and two blind-review templates: {output.resolve()}")
    if args.mode == "demo":
        print("DEMO: heuristic rewrites and optional BM25 retrieval, not a hybrid LLM experiment.")
    return int(any(row["error_count"] or row["rewrite_error_count"] or row.get("retrieval_error_count") or row["method_error_count"]
                   for row in summaries))


if __name__ == "__main__":
    raise SystemExit(main())
