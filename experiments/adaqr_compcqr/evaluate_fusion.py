"""Evaluate an AdaQR-trained rewrite with the frozen Server16 QReCC metrics.

This standalone experiment reads the existing project and baseline artifacts.
Its primary inference is ONE trained rewrite followed by the SAME BM25 search.
It does not silently execute or claim CompCQR's multi-query fusion/selection.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time

from paper_runtime import LLAMA as ORIGINAL_COMPCQR_MODEL, OriginalCompCQRClient


HERE = Path(__file__).resolve().parent
STRATEGY = "adaqr_compcqr_single"
FULL_STRATEGY = "adaqr_compcqr_full"
OFFICIAL_TEST_MD5 = "7ad29296a1dfd2f75dafd653a8822269"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def import_project(root: Path):
    """Import installed baseline code without patching it or its strategy registry."""
    for name in ("evaluate", "evaluate_rewrite", "rag"):
        existing = sys.modules.get(name)
        if existing and not Path(existing.__file__).resolve().is_relative_to(root):
            raise ValueError(f"{name} already imported from another project")
    sys.path.insert(0, str(root))
    base = importlib.import_module("evaluate")
    evaluator = importlib.import_module("evaluate_rewrite")
    if Path(base.__file__).resolve().parent != root or Path(evaluator.__file__).resolve().parent != root:
        raise ValueError("Evaluation modules do not belong to --project-root")
    return base, evaluator


def select_ids(subset: Path, official_test: Path, baseline_ids: list[str], limit: int | None):
    """Validate raw selected rows against the official source, preserving order."""
    source = official_test.read_bytes()
    if hashlib.md5(source).hexdigest() != OFFICIAL_TEST_MD5:
        raise ValueError("Official qrecc-test.json differs from its published MD5")
    key = lambda row: f"test:{row['Conversation_no']}-{row['Turn_no']}"
    official = {key(row): row for row in json.loads(source)}
    rows = read_json(subset)
    if not isinstance(rows, list) or len(rows) != 2000:
        raise ValueError("--subset-file must contain exactly 2,000 raw official test records")
    selected = []
    for row in rows:
        identifier = key(row)
        if official.get(identifier) != row:
            raise ValueError(f"Subset record differs from official source: {identifier}")
        selected.append(identifier)
    if len(set(selected)) != len(selected):
        raise ValueError("Subset has duplicate IDs")
    if selected != baseline_ids:
        raise ValueError("Subset IDs/order differ from baseline manifest")
    return selected[:limit] if limit else selected


def adapter_model_identity(support_model, adapter: Path, model, tokenizer):
    """Bind the run to adapter bytes and the base/tokenizer actually loaded."""
    result = support_model.adapter_identity(str(adapter))
    adapter_config = read_json(adapter / "adapter_config.json")
    base_name = adapter_config["base_model_name_or_path"]
    base_path = Path(base_name).expanduser()
    if base_path.is_dir():
        base_path = base_path.resolve()
        files = sorted(path for path in base_path.iterdir() if path.is_file()
                       and (path.suffix in {".safetensors", ".json", ".model", ".txt"}
                            or path.name.startswith("pytorch_model")))
        if not any(path.suffix == ".safetensors" or path.name.startswith("pytorch_model") for path in files):
            raise ValueError("Local base model lacks identifiable weight files")
        result["base_model"] = {"path": str(base_path),
                                "files_sha256": {path.name: sha256(path) for path in files}}
    else:
        commit = getattr(model.config, "_commit_hash", None)
        if not commit and hasattr(model, "get_base_model"):
            commit = getattr(model.get_base_model().config, "_commit_hash", None)
        if not commit:
            raise ValueError("Remote base model has no resolved Hugging Face commit identity")
        result["base_model"] = {"repository": base_name, "resolved_commit": commit}
    result.update(tokenizer_class=type(tokenizer).__name__, tokenizer_name=tokenizer.name_or_path,
                  inference={"do_sample": False, "max_new_tokens": 128,
                             "prompt_wrapper": "adaqr_support.adaqr.rewrite_prompt"})
    return result


class ReplaceMQClient:
    """Retain original RAR answer, replacing only mq as in preference training."""

    def __init__(self, auxiliary, trained_query):
        self.auxiliary, self.trained_query = auxiliary, trained_query
        self.first, self.original_rar = True, None

    def generate(self, prompt, system="", temperature=0.0):
        raw = self.auxiliary.generate(prompt, system=system, temperature=temperature)
        if not self.first:
            return raw
        self.first = False
        from rag.compcqr_prompts import parse_rar
        reason, original_query, answer = parse_rar(raw)
        self.original_rar = {"raw": raw, "query": original_query}
        return ("Rewrite: " + reason + " So the question should be rewritten as: "
                + self.trained_query + "\nResponse: " + answer)


class AdapterEngine:
    """Minimal evaluator interface, sharing the baseline history/check policies."""

    def __init__(self, base_engine, adapter: Path, support_model):
        self.config = base_engine.config
        self.retriever = base_engine.retriever
        self.mode = "hf_adapter"
        self.adapter = str(adapter)
        self.support_model = support_model

    def rewrite_query(self, question, history=None, strategy=STRATEGY):
        from rag.pipeline import clean_history
        from rag.rewriters import clean_query, heuristic_rewrite, validation_issues
        if strategy != STRATEGY:
            raise ValueError("This entrypoint evaluates only the standalone AdaQR rewrite")
        if not isinstance(question, str) or not question.strip() or len(question) > 2000:
            raise ValueError("Question must be nonempty and at most 2,000 characters")
        question = question.strip()
        messages = clean_history(history, self.config["history_turns"])
        started = time.perf_counter()
        trace = {"strategy": strategy, "llm_calls": 0, "query_count": 1, "fallback": False,
                 "repaired": False, "raw_responses": [], "validation_policy": "same baseline heuristic checks",
                 "inference": "single_trained_rewrite_then_bm25", "prompt_wrapper": "AdaQR original rewrite_prompt"}
        error = None
        if not messages:
            query, source = question, "no_history"
        else:
            try:
                trace["llm_calls"] = 1
                raw = self.support_model.rewrite_with_adapter(question, messages, self.adapter)
                trace["raw_responses"].append(raw)
                query = clean_query(raw)
                issues = validation_issues(query, question, messages)
                trace["validation_issues"] = issues
                if issues:
                    raise ValueError("Invalid trained rewrite: " + ", ".join(issues))
                source = "llm_adaqr"
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                query, source = heuristic_rewrite(question, messages), "adaqr_fallback"
                trace.update(fallback=True, fallback_reason=error)
        elapsed = round((time.perf_counter() - started) * 1000, 2)
        return {"question": question, "search_query": query, "strategy": strategy,
                "rewrite_source": source, "rewrite_error": error, "rewrite_trace": trace,
                "visible_history": messages, "mode": self.mode, "dataset": "qrecc", "backend": "bm25",
                "latencies_ms": {"rewrite": elapsed, "total": elapsed}}


class TrainedCompCQREngine(AdapterEngine):
    """AdaQR mq plus original CompCQR ma/follow-up/RRF/selector, up to four calls."""

    def __init__(self, base_engine, adapter, support_model, auxiliary):
        super().__init__(base_engine, adapter, support_model)
        self.auxiliary = auxiliary

    def rewrite_query(self, question, history=None, strategy="compcqr"):
        if strategy != "compcqr":
            raise ValueError("Internal CompCQR evaluator strategy required")
        from rag.compcqr import build_plan, compose_queries
        started = time.perf_counter()
        trained = super().rewrite_query(question, history, STRATEGY)
        proxy = ReplaceMQClient(self.auxiliary, trained["search_query"])
        plan = build_plan(trained["question"], trained["visible_history"], client=proxy,
                          mode="ollama", config=self.config)
        # Auxiliary failures must not replace a successfully trained mq with
        # the baseline heuristic. Keep its degraded status, but retain learned mq.
        plan["components"]["mq"] = trained["search_query"]
        plan["query"] = trained["search_query"]
        plan["queries"] = compose_queries(plan["components"], self.config["retriever_backend"],
                                           self.config["compcqr_max_components"])
        trace = plan["trace"]
        trace.update(components=plan["components"], queries=plan["queries"], query_count=len(plan["queries"]))
        trace.update(learned_mq_source=trained["rewrite_source"],
                     learned_mq_raw=trained["rewrite_trace"].get("raw_responses", []),
                     auxiliary_original_rar=proxy.original_rar,
                     inference="adaqr_mq_plus_original_compcqr", trained_rewriter_calls=trained["rewrite_trace"]["llm_calls"])
        trace["llm_calls"] += trained["rewrite_trace"]["llm_calls"]
        error = plan["error"]
        if trained["rewrite_error"]:
            error = "; ".join(value for value in (trained["rewrite_error"], error) if value)
            trace["component_errors"].append("trained_mq: " + trained["rewrite_error"])
            trace["fallback"] = True
        elapsed = round((time.perf_counter() - started) * 1000, 2)
        return {**trained, "strategy": "compcqr", "search_query": plan["query"],
                "search_queries": plan["queries"], "components": plan["components"],
                "rewrite_source": plan["source"], "rewrite_error": error,
                "rewrite_trace": trace, "rouge_scope": "base_rewrite_only",
                "latencies_ms": {"rewrite": elapsed, "total": elapsed}}

    def retrieve_rewritten(self, result, top_k=3, *, retriever=None):
        from rag.compcqr import retrieve_plan
        if type(top_k) is not int or not 1 <= top_k <= 20:
            raise ValueError("top_k must be in [1,20]")
        started = time.perf_counter()
        retrieved = retrieve_plan(result["question"], result["visible_history"], result["search_queries"],
            retriever=self.retriever if retriever is None else retriever, client=self.auxiliary,
            mode="ollama", config=self.config, trace=result["rewrite_trace"])
        times = {**result["latencies_ms"], **retrieved["latencies_ms"]}
        times["total"] = round(times["rewrite"] + (time.perf_counter() - started) * 1000, 2)
        return {**result, "hits": retrieved["hits"][:top_k], "rewrite_trace": retrieved["trace"],
                "latencies_ms": times}


def relabel_record(row, external=True):
    """Keep standalone output names distinct while reusing the stock C evaluator."""
    row = copy.deepcopy(row)
    old, new = ("compcqr", FULL_STRATEGY) if external else (FULL_STRATEGY, "compcqr")
    if row.get("strategy") == old:
        row["strategy"] = new
        if "id" in row:
            row["cache_key"] = f"{new}:{row['id']}"
    return row


class NamedCompCQRCache:
    """Translate only this standalone checkpoint's label, without modifying baseline code."""

    def __init__(self, cache):
        self.cache = cache

    @property
    def rows(self):
        return [relabel_record(row, external=False) for row in self.cache.rows]

    def append(self, row, key):
        row = relabel_record(row)
        self.cache.append(row, row["cache_key"])


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path)
    parser.add_argument("--subset-file", type=Path)
    parser.add_argument("--official-test", type=Path)
    parser.add_argument("--config", type=Path, help="Optional compatibility check against machine-local settings")
    parser.add_argument("--inference", choices=["single", "compcqr"], default="single")
    parser.add_argument("--aux-model", default=ORIGINAL_COMPCQR_MODEL,
                        help="HF ID or local Llama-3.1-8B-Instruct path for optional complete CompCQR")
    parser.add_argument("--aux-device", default="cuda:1")
    parser.add_argument("--aux-seed", type=int, default=42)
    parser.add_argument("--compcqr-selection-top-k", type=int, default=10,
                        help="Paper's Llama selector candidate count; only affects --inference compcqr")
    parser.add_argument("--limit", type=int, help="Diagnostic prefix of the same frozen 2,000 IDs")
    parser.add_argument("--rewrite-only", action="store_true")
    parser.add_argument("--review-count", type=int)
    parser.add_argument("--review-seed", type=int)
    parser.add_argument("--warmup", type=int, default=0, help="Extra public-input generations excluded from query timing")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate files and frozen plan without ML or services")
    args = parser.parse_args(argv)
    if args.limit is not None and not 1 <= args.limit <= 2000:
        parser.error("--limit must be in [1,2000]")
    if args.review_count is not None and args.review_count < 0 or args.warmup < 0:
        parser.error("review-count and warmup must be nonnegative")
    if args.resume and args.output_dir is None:
        parser.error("--resume requires --output-dir")
    if not 1 <= args.compcqr_selection_top_k <= 50 or not 0 <= args.aux_seed < 2**31:
        parser.error("selector K must be in [1,50] and aux-seed in [0,2**31)")
    return args


def main(argv=None):
    args = parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    try:
        root, adapter = args.project_root.expanduser().resolve(), args.adapter.expanduser().resolve()
        baseline = (args.baseline_run or root / "outputs/bm25_gpu4_experiments/main-test2000").resolve()
        subset = (args.subset_file or root.parent / "qrecc_full/subsets/test_2000.json").resolve()
        official = (args.official_test or root.parent / "qrecc_full/raw/qrecc-test.json").resolve()
        output = args.output_dir or Path(datetime.now(timezone.utc).strftime("eval_%Y%m%d_%H%M%S_%f"))
        output = (output if output.is_absolute() else HERE / "outputs" / output).resolve()
        if not output.is_relative_to(HERE / "outputs") or output == HERE / "outputs":
            raise ValueError("--output-dir must be a new run directory below this standalone experiment's outputs/")
        if output == baseline or output.is_relative_to(baseline):
            raise ValueError("Baseline output is read-only")
        base, evaluator = import_project(root)
        from rag.run_cache import RunCache, code_hashes, make_signature
        from rag.pipeline import RAGEngine, clean_history
        from rag.config import load_experiment_config
        signed = read_json(baseline / "manifest.json")
        baseline_identity = signed["identity"]
        if signed.get("signature") != make_signature(baseline_identity):
            raise ValueError("Baseline run manifest signature is invalid")
        if base.completed_report(baseline, signed["signature"]) is None:
            raise ValueError("Baseline must have a verified completed report")
        if baseline_identity.get("code") != code_hashes(root):
            raise ValueError("Baseline evaluator source hashes differ from --project-root")
        config = copy.deepcopy(baseline_identity["config"])
        if (config.get("retriever_backend") != "bm25" or config.get("retrieval_store") != "opensearch"
                or config.get("evaluation_scope") != "full"):
            raise ValueError("Expected the frozen full-QReCC OpenSearch BM25 baseline")
        if args.config:
            configured = load_experiment_config(root, scope="full", config_path=args.config)
            keys = ("eval_path", "dataset_manifest_path", "history_turns", "retrieval_store",
                    "retriever_backend", "opensearch_url", "opensearch_index", "bm25_mode",
                    "hybrid_candidate_depth", "corpus_path", "top_k")
            differences = [key for key in keys if configured.get(key) != config.get(key)]
            if differences:
                raise ValueError("Config differs from baseline on: " + ", ".join(differences))
        strategy = FULL_STRATEGY if args.inference == "compcqr" else STRATEGY
        compcqr_changes = {}
        if args.inference == "compcqr":
            overrides = {"compcqr_max_components": 5, "compcqr_retrieval_depth": 100,
                         "compcqr_rrf_k": 0, "compcqr_select": True,
                         "compcqr_selection_top_k": args.compcqr_selection_top_k}
            compcqr_changes = {key: {"baseline": config.get(key), "actual": value}
                               for key, value in overrides.items() if config.get(key) != value}
            config.update(overrides)
        ids = select_ids(subset, official, baseline_identity["sample_ids"], args.limit)
        data_path = Path(config["eval_path"])
        data_path = data_path if data_path.is_absolute() else root / data_path
        prepared = base.prepared_dataset_identity(config, data_path)
        old_prepared = {key: value for key, value in baseline_identity["prepared_dataset"].items()
                        if key != "test_subset"}
        if prepared != old_prepared:
            raise ValueError("Prepared dataset identity differs from baseline")
        all_examples = evaluator.load_rewrite_examples(data_path, "test", None, data_format=baseline_identity["data_format"])
        by_id = {row["id"]: row for row in all_examples}
        examples = [by_id[identifier] for identifier in ids]
        review_count = args.review_count if args.review_count is not None else baseline_identity.get("review_count", 30)
        review_seed = args.review_seed if args.review_seed is not None else baseline_identity.get("review_seed", 20261102)
        include_l = bool(baseline_identity.get("rouge_l", False))
        plan = {"strategy": strategy, "inference": args.inference, "sample_count": len(ids),
                "subset_sha256": sha256(subset), "baseline_signature": signed["signature"],
                "adapter": str(adapter), "output": str(output), "retrieval_top_k": evaluator.RETRIEVAL_TOP_K,
                "retrieval": not args.rewrite_only, "rouge_l": include_l,
                "auxiliary_model": args.aux_model if args.inference == "compcqr" else None,
                "compcqr_configuration_changes": compcqr_changes,
                "retrieval_adaptation": "Frozen baseline OpenSearch BM25 retained for comparability; this is not the original Pyserini BM25 k1=0.82,b=0.68 benchmark.",
                "metrics": ["rouge1_precision", "rouge1_recall", "rouge1_f1", *evaluator.RETRIEVAL_METRICS],
                "timing_caveat": "HF/PEFT greedy inference vs baseline Ollama; timings are not backend-controlled comparisons."}
        if args.dry_run:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return 0
        sys.path.insert(0, str(HERE))
        support_model = importlib.import_module("adaqr_support.adaqr_model")
        support_prompt = importlib.import_module("adaqr_support.adaqr")
        support_files = [Path(support_model.__file__).resolve(), Path(support_prompt.__file__).resolve()]
        if any(not path.is_relative_to(HERE / "adaqr_support") for path in support_files):
            raise ValueError("AdaQR helpers must come from this standalone experiment")
        initialization_started = time.perf_counter()
        model, tokenizer = support_model.load_rewriter(str(adapter))
        model_identity = adapter_model_identity(support_model, adapter, model, tokenizer)
        auxiliary = None
        if args.inference == "compcqr":
            auxiliary = OriginalCompCQRClient(args.aux_model, seed=args.aux_seed, device=args.aux_device)
            model_identity["auxiliary_compcqr"] = auxiliary.identity()
        base_engine = RAGEngine(root, config, mode="ollama", build_index=not args.rewrite_only)
        engine = (TrainedCompCQREngine(base_engine, adapter, support_model, auxiliary)
                  if auxiliary is not None else AdapterEngine(base_engine, adapter, support_model))
        retrieval_ids, retrieval = None, None
        if not args.rewrite_only:
            retrieval_ids, retrieval = evaluator.retrieval_coverage(config, examples, retriever=engine.retriever)
            old_corpus = baseline_identity["retrieval"]["corpus_fingerprint"]
            if retrieval["corpus_fingerprint"] != old_corpus:
                raise ValueError("Live frozen BM25 index identity differs from baseline")
        initialization_ms = round((time.perf_counter() - initialization_started) * 1000, 2)
        identity = {"task": "standalone_adaqr_compcqr_rewrite_quality", "strategy": strategy,
                    "inference": args.inference, "mode": "hf_adapter", "config": config,
                    "prepared_dataset": prepared, "sample_ids": ids, "subset_sha256": sha256(subset),
                    "baseline_signature": signed["signature"], "runtime_model_identity": model_identity,
                    "retrieval": retrieval, "code": code_hashes(root), "wrapper_sha256": sha256(Path(__file__)),
                    "support_sha256": {path.name: sha256(path) for path in support_files},
                    "normalization": evaluator.NORMALIZATION, "rouge_l": include_l,
                    "review_count": review_count, "review_seed": review_seed, "warmup": args.warmup}
        cache = RunCache(output, identity, resume=args.resume)
        if base.completed_report(output, cache.signature) is not None:
            print("Retained completed standalone report: " + str(output), flush=True)
            return 0
        warmup_started = time.perf_counter()
        for index in range(args.warmup):
            example = examples[index % len(examples)]
            support_model.rewrite_with_adapter(example["question"],
                clean_history(example["history"], config["history_turns"]), str(adapter))
        warmup_ms = round((time.perf_counter() - warmup_started) * 1000, 2)
        summaries, details = evaluator.evaluate_rewrites(engine, examples,
            ["compcqr"] if auxiliary is not None else [STRATEGY], include_l=include_l,
            cache=NamedCompCQRCache(cache) if auxiliary is not None else cache, retrieval_ids=retrieval_ids)
        if auxiliary is not None:
            summaries = [relabel_record(row) for row in summaries]
            details = [relabel_record(row) for row in details]
        metadata = {**plan, "task": identity["task"], "mode": "hf_adapter", "config": config,
                    "run_signature": cache.signature, "runtime_model_identity": model_identity,
                    "created_at": datetime.now(timezone.utc).isoformat(), "initialization_ms": initialization_ms,
                    "warmup_ms": warmup_ms, "retrieval": retrieval, "prepared_dataset": prepared,
                    "normalization": evaluator.NORMALIZATION,
                    "evaluation_scope": "diagnostic_prefix" if args.limit else "same_frozen_test2000",
                    "first_turn_policy": "No visible history: trained mq preserves original question; full CompCQR still generates auxiliary components, as baseline C does.",
                    "failure_policy": "Same baseline heuristic validation/fallback; all fixed samples retained, failures and fallbacks separately counted.",
                    "comparison_scope": ("Trained mq plus original CompCQR auxiliary components, RRF and selector; up to four model calls (one trained plus three auxiliary), ROUGE scores trained mq only. BF16 and prompt-derived seeds are explicit runtime choices."
                        if auxiliary is not None else "Trained main rewrite + single-query BM25; not full CompCQR multi-query fusion/selection."),
                    "training_claim": "Adapter training provenance is supplied separately; evaluation never reads current gold rewrite/answer as model input."}
        evaluator.export_blind_review(output, details, count=review_count, seed=review_seed)
        base.save_report(output, metadata, summaries, details, extra_artifacts=evaluator.REVIEW_ARTIFACTS)
        print(json.dumps({"summary": summaries, "output": str(output)}, ensure_ascii=False, indent=2))
        return int(any(row["error_count"] or row["rewrite_error_count"] or row.get("retrieval_error_count")
                       or row.get("method_error_count") for row in summaries))
    except (ValueError, OSError, RuntimeError, KeyError, ImportError) as exc:
        print(f"Standalone AdaQR evaluation: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
