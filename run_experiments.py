"""Portable sequential launch plans for all three full-QReCC experiments."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from rag.config import load_experiment_config, offline_config
from rag.llm import runtime_embedding_identity, runtime_model_identity
from rag.pipeline import RAGEngine
from rag.run_cache import code_hashes, make_signature
from evaluate import completed_report, prepared_dataset_identity

PROJECT = Path(__file__).resolve().parent


def build_jobs(args):
    root = args.output_root if args.output_root.is_absolute() else PROJECT / args.output_root
    jobs = []
    if args.experiment in {"experiment1", "all"}:
        jobs.append(("experiment1", "evaluate.py", args.query_model, ["current", "history", "rewrite"]))
    if args.experiment in {"experiment2", "all"}:
        for model in dict.fromkeys(args.models):
            label = "".join(char if char.isalnum() or char in "._-" else "_" for char in model)
            jobs.append(("experiment2_" + label, "evaluate.py", model, ["rewrite"]))
    if args.experiment in {"main", "all"}:
        jobs.append(("main", "evaluate_rewrite.py", args.main_model, ["rewrite", "structured", "compcqr"]))
    plans = []
    for name, script, model, strategies in jobs:
        output = root / name
        command = [sys.executable, "-X", "utf8", "-u", "-B", str(PROJECT / script),
                   "--scope", args.scope, "--mode", args.mode, "--model", model,
                   "--split", args.split, "--strategies", *strategies,
                   "--trace-detail", args.trace_detail, "--output-dir", str(output)]
        if script == "evaluate.py":
            command.extend(["--backend", "bm25", "--top-k", "3", "--retrieval-only"])
        if args.config:
            command.extend(["--config", str(args.config)])
        if args.limit is not None:
            command.extend(["--limit", str(args.limit)])
        plans.append({"name": name, "script": script, "model": model, "strategies": strategies,
                      "output": str(output), "command": command})
    return plans


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=["experiment1", "experiment2", "main", "all"], default="all")
    parser.add_argument("--scope", choices=["full", "pilot"], default="full")
    parser.add_argument("--mode", choices=["ollama", "demo"], default="ollama")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--split", choices=["train", "dev", "test", "all"], default="test")
    parser.add_argument("--query-model", default="qwen3:1.7b")
    parser.add_argument("--main-model", default="qwen3:8b")
    # The mutable 4b alias now selects Thinking-2507; pin the original Qwen3 4B
    # quantization tag for the same non-thinking baseline as the other sizes.
    parser.add_argument("--models", nargs="+", default=["qwen3:1.7b", "qwen3:4b-q4_K_M", "qwen3:8b", "qwen3:14b", "qwen3:32b"])
    parser.add_argument("--output-root", type=Path, default=Path("outputs/bm25_experiments"))
    parser.add_argument("--limit", type=int, help="Explicit debugging prefix; omitted means every selected turn")
    parser.add_argument("--trace-detail", choices=["summary", "full"], default="summary")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print argument arrays without contacting services or running models")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.scope == "full" and args.mode != "ollama":
        parser.error("Full experiments require ollama; offline checks use --scope pilot --mode demo")
    return args


def main(argv=None):
    args = parse_args(argv)
    jobs = build_jobs(args)
    if args.dry_run:
        print(json.dumps({"scope": args.scope, "split": args.split, "limit": args.limit,
                          "split_warning": "Includes training/development data" if args.split == "all" else None,
                          "jobs": jobs}, ensure_ascii=False, indent=2))
        return 0
    try:
        for job in jobs:
            overrides = {"model": job["model"], "trace_detail": args.trace_detail, "retriever_backend": "bm25"}
            if job["script"] == "evaluate.py":
                overrides["top_k"] = 3
            config = load_experiment_config(PROJECT, overrides, scope=args.scope, config_path=args.config)
            if args.mode == "demo":
                config = offline_config(config)
            data = Path(config["eval_path"])
            data = data if data.is_absolute() else PROJECT / data
            if not data.is_file():
                raise ValueError("QReCC data missing. Prepare full dialogs on the server before running experiments.")
            prepared = prepared_dataset_identity(config, data)
            model_identity = runtime_model_identity(config, args.mode)
            embedding = runtime_embedding_identity(config) if args.mode == "ollama" else None
            if embedding:
                config["embedding_digest"] = embedding["digest"]
            index_identity = None
            if config.get("retrieval_store") == "opensearch":
                engine = RAGEngine(PROJECT, config, args.mode)
                index_identity = engine.retriever.index_identity()
            identity = {"command": job["command"], "config": config, "model": model_identity,
                        "embedding": embedding, "index": index_identity,
                        "prepared_dataset": prepared, "code": code_hashes(PROJECT)}
            output = Path(job["output"])
            control = output.parent / (output.name + ".launcher.json")
            signature = make_signature(identity)
            if control.exists():
                previous = json.loads(control.read_text(encoding="utf-8"))
                if not args.resume or previous.get("signature") != signature:
                    raise ValueError("Existing launch identity differs or --resume is missing: " + str(control))
            else:
                if output.exists():
                    raise ValueError("Existing output has no matching launcher identity; use a fresh output root")
                control.parent.mkdir(parents=True, exist_ok=True)
                control.write_text(json.dumps({"signature": signature, "identity": identity}, ensure_ascii=False, indent=2), encoding="utf-8")
            report = completed_report(output)
            if report is not None:
                manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
                if (manifest.get("signature") != make_signature(manifest.get("identity"))
                        or report.get("metadata", {}).get("run_signature") != manifest.get("signature")):
                    raise ValueError("Completed report does not match its signed run manifest: " + str(output))
                expected = {"config": config, "code": identity["code"], "prepared_dataset": prepared,
                            "runtime_model_identity": model_identity, "mode": args.mode,
                            "split": args.split, "strategies": job["strategies"]}
                if any(manifest["identity"].get(key) != value for key, value in expected.items()):
                    raise ValueError("Completed report is from a different experiment: " + str(output))
                if any(row.get("error_count") or row.get("rewrite_error_count") or row.get("retrieval_error_count")
                       or row.get("method_error_count") for row in report["summary"]):
                    raise ValueError("Completed report contains errors; inspect it and use a new output root for a new evaluation")
                print("Retained completed verified run: " + job["name"], flush=True)
                continue
            command = list(job["command"])
            if args.resume and (output / "manifest.json").exists():
                command.append("--resume")
            print("Running " + job["name"] + ": " + json.dumps(command, ensure_ascii=False), flush=True)
            code = subprocess.run(command, cwd=PROJECT).returncode
            if code:
                return code
    except (ValueError, OSError, RuntimeError) as error:
        print("Experiment launcher: " + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
