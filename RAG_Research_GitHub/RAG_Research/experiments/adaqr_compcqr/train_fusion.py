"""Train an AdaQR rewriter with single-query and CompCQR retrieval rewards."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
import sys

from adaqr_support.adaqr import AdaQRSettings, answer_prompt, marginal_reward, preference_pairs, rewrite_prompt
from adaqr_support.adaqr_model import completion_log_probability, require_ml
from paper_runtime import LLAMA, MISTRAL, RRF_CONFIG, OriginalCompCQRClient, PaperBM25


HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"
SETTINGS = AdaQRSettings()


def candidates_for(row):
    candidates = row.get("candidates")
    if (row.get("split") != "train" or not isinstance(row.get("answer"), str)
            or not row["answer"].strip() or not row.get("history")):
        raise ValueError(f"Preference training requires a later training turn with an answer: {row.get('id')}")
    if (not isinstance(candidates, list) or len(candidates) != SETTINGS.candidates
            or any(not isinstance(candidate, str) or not candidate.strip() for candidate in candidates)):
        raise ValueError(f"Expected three nonempty sampled rewrites: {row['id']}")
    return list(dict.fromkeys(candidate.strip() for candidate in candidates))


def score_turn(row, *, retriever, components, answer_logp, config):
    """Share query results and answer scoring within a turn, never RRF rankings."""
    from rag.compcqr import compose_queries, fuse_rankings

    candidates = candidates_for(row)
    plans = {
        candidate: compose_queries({**components, "mq": candidate}, config["retriever_backend"],
                                   config["compcqr_max_components"])
        for candidate in candidates
    }
    depth = config["compcqr_retrieval_depth"]
    # Every candidate also participates as its own single-component query. Include
    # it explicitly in case a duplicate component affected query construction.
    queries = list(dict.fromkeys([*candidates, *(q for plan in plans.values() for q in plan)]))
    rankings = retriever.search_many(queries, max(depth, SETTINGS.top_k))
    if len(rankings) != len(queries):
        raise ValueError("Search result count differs from the number of distinct queries")
    retrieved = dict(zip(queries, rankings))
    score_cache = {}

    def branch(hits):
        if not hits:
            raise ValueError(f"No passages retrieved for training turn {row['id']}")
        ids, logps = [], []
        for hit in hits:
            passage_id = hit.get("passage_id") or hit["id"]
            passage_text = hit.get("text") or retriever.passage_text(passage_id)
            key = (passage_id, passage_text)
            if key not in score_cache:
                prompt = answer_prompt(row["history"], row["question"], passage_text)
                score_cache[key] = answer_logp(prompt, row["answer"])
            ids.append(key[0])
            logps.append(score_cache[key])
        return marginal_reward(hits, logps), ids

    records = []
    for candidate, plan in plans.items():
        single, single_ids = branch(retrieved[candidate][:SETTINGS.top_k])
        fused_hits = fuse_rankings([retrieved[query][:depth] for query in plan], config["compcqr_rrf_k"])
        fused, fused_ids = branch(fused_hits[:SETTINGS.top_k])
        records.append({"rewrite": candidate, "single_reward": single, "fusion_reward": fused,
                        "single_passage_ids": single_ids, "fusion_passage_ids": fused_ids,
                        "query_count": len(plan)})
    return {"id": row["id"], "prompt": rewrite_prompt(row["history"], row["question"]),
            "status": "ok", "components": components, "candidates": records,
            "distinct_queries": len(queries), "scored_passages": len(score_cache)}


def mixed_pairs(record, alpha, delta):
    rows = record.get("candidates", [])
    scored = [
        {"rewrite": row["rewrite"], "reward": (1 - alpha) * row["single_reward"] + alpha * row["fusion_reward"]}
        for row in rows
    ]
    return [{"id": record["id"], "prompt": record["prompt"], "alpha": alpha, "delta": delta, **pair}
            for pair in preference_pairs(scored, delta)]


@contextmanager
def lock_run(path):
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"The reward run is already active: {path}") from exc
        yield


def reward(args):
    from rag.compcqr import PROMPT_VERSION, build_plan
    from rag.data import read_jsonl
    from rag.run_cache import RunCache, file_sha256

    rows = list(read_jsonl(args.samples))
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("The training sample file must contain unique turn IDs")
    for row in rows:
        candidates_for(row)
    retriever = PaperBM25(args.lucene_index)
    torch, AutoModel, AutoTokenizer = require_ml()
    if torch.cuda.device_count() < 2:
        raise RuntimeError("Reward collection needs two visible CUDA GPUs, one for Mistral and one for Llama")
    tokenizer = AutoTokenizer.from_pretrained(args.answer_model, revision=args.revision)
    model = AutoModel.from_pretrained(args.answer_model, revision=args.revision,
                                     torch_dtype=torch.bfloat16, device_map={"": 0})
    model.eval()
    llama_tokenizer = AutoTokenizer.from_pretrained(args.component_model, revision=args.component_revision)
    llama = AutoModel.from_pretrained(args.component_model, revision=args.component_revision,
                                     torch_dtype=torch.bfloat16, device_map={"": 1})
    llama.eval()
    generator = OriginalCompCQRClient(args.component_model, seed=args.seed, model=llama, tokenizer=llama_tokenizer)
    code = [
        HERE / "train_fusion.py", HERE / "adaqr_support/adaqr.py", HERE / "adaqr_support/adaqr_model.py",
        HERE / "paper_runtime.py", args.project_root / "rag/compcqr.py",
        args.project_root / "rag/compcqr_prompts.py",
    ]

    def model_identity(name, loaded, revision):
        directory = Path(name)
        files = {}
        if directory.is_dir():
            for path in sorted(directory.glob("*.safetensors")):
                stat = path.stat()
                files[path.name] = [stat.st_size, stat.st_mtime_ns]
        return {"name": name, "revision": getattr(loaded.config, "_commit_hash", None) or revision,
                "weight_files": files}

    identity = {
        "experiment": "adaqr_compcqr_reward_v1", "samples_sha256": file_sha256(args.samples),
        "sample_count": len(rows), "prompt_version": PROMPT_VERSION,
        "answer_model": model_identity(args.answer_model, model, args.revision),
        "component_model": generator.identity(),
        "max_length": args.max_length, "seed": args.seed,
        "top_k": SETTINGS.top_k, "retrieval": RRF_CONFIG, "index": retriever.identity(),
        "code": {str(path): file_sha256(path) for path in code},
    }

    def log_probability(prompt, answer):
        with torch.inference_mode():
            return float(completion_log_probability(model, tokenizer, prompt, answer,
                                                    args.max_length, include_eos=False).item())

    with lock_run(args.output.with_name(f".{args.output.name}.lock")):
        cache = RunCache(args.output, identity, resume=args.resume)
        errors = sum(record["status"] != "ok" for record in cache.rows)
        print(f"Reward collection | {len(cache.rows)}/{len(rows)} previously recorded",
              flush=True)
        for index, row in enumerate(rows, 1):
            key = row["id"]
            if key in cache.completed_keys:
                continue
            plan = build_plan(row["question"], row["history"], client=generator, mode="ollama", config=RRF_CONFIG)
            if plan["source"] != "compcqr_llm":
                # Exclude invalid generated components from training pairs. A
                # service or GPU exception stops the run and leaves the turn pending.
                if plan["error"] and any(
                        report["error"].split(": ")[1:2] != ["ValueError"]
                        for report in plan["trace"]["stage_errors"]):
                    raise RuntimeError(f"CompCQR generation failed for training turn {key}: {plan['error']}")
                record = {"id": key, "status": "component_error", "error": plan["error"]}
                errors += 1
            else:
                record = score_turn(row, retriever=retriever, components=plan["components"],
                                    answer_logp=log_probability, config=RRF_CONFIG)
            cache.append(record, key)
            print(f"Reward {index}/{len(rows)} | {key} | {record['status']} | component errors {errors}", flush=True)


def export_pairs(args):
    from rag.data import read_jsonl

    manifest = json.loads((args.rewards / "manifest.json").read_text(encoding="utf-8"))
    records = list(read_jsonl(args.rewards / "checkpoint.jsonl"))
    identity = manifest["identity"]
    if (identity.get("experiment") != "adaqr_compcqr_reward_v1"
            or len(records) != identity["sample_count"] or len({row["id"] for row in records}) != len(records)):
        raise ValueError("Complete reward collection before constructing preference pairs")
    pairs = [pair for record in records if record["status"] == "ok"
             for pair in mixed_pairs(record, args.alpha, args.delta)]
    print(f"Training turns: {len(records)} | valid rewards: {sum(row['status'] == 'ok' for row in records)}"
          f" | pairs: {len(pairs)} | alpha: {args.alpha} | delta: {args.delta}", flush=True)
    if not pairs:
        raise ValueError("No qualifying pairs. Inspect reward rankings on the training pilot.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        for pair in pairs:
            stream.write(json.dumps(pair, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=HERE.parents[1])
    stages = parser.add_subparsers(dest="stage", required=True)
    sft = stages.add_parser("sft", help="Train a QLoRA adapter on reference rewrites")
    sft.add_argument("--seed-data", type=Path, required=True)
    sft.add_argument("--seed-source", choices=["infocqr_initial_fsl", "user_qrecc"], default="infocqr_initial_fsl")
    sft.add_argument("--base-model", default=MISTRAL, help="Mistral-7B-v0.1 HF ID or local snapshot")
    sample = stages.add_parser("sample", help="Sample three candidates from the SFT rewriter")
    sample.add_argument("--target-data", type=Path, required=True)
    sample.add_argument("--target-ids", type=Path, required=True,
                        help="JSON array of the 39,677 answered later-turn training IDs from AdaQR")
    sample.add_argument("--sft-adapter", type=Path, required=True)
    for stage in (sft, sample):
        stage.add_argument("--max-samples", type=int)
    scoring = stages.add_parser("reward", help="Collect both rewards in a resumable run")
    scoring.add_argument("--samples", type=Path, required=True)
    scoring.add_argument("--answer-model", default=MISTRAL, help="Mistral-7B-v0.1 HF ID or local snapshot")
    scoring.add_argument("--component-model", default=LLAMA, help="Llama-3.1-8B-Instruct HF ID or local snapshot")
    scoring.add_argument("--revision")
    scoring.add_argument("--component-revision")
    scoring.add_argument("--lucene-index", type=Path, required=True)
    scoring.add_argument("--resume", action="store_true")
    pairs = stages.add_parser("pairs", help="Export preferences from the saved rewards")
    pairs.add_argument("--rewards", type=Path, required=True)
    pairs.add_argument("--alpha", type=float, default=0.5)
    pairs.add_argument("--delta", type=float, default=SETTINGS.delta)
    dpo = stages.add_parser("dpo", help="Optimize the rewriter on the preference pairs")
    dpo.add_argument("--preferences", type=Path, required=True)
    dpo.add_argument("--sft-adapter", type=Path, required=True)
    for stage in (sft, sample, dpo, scoring):
        stage.add_argument("--seed", type=int, default=42)
    for stage in (sft, dpo):
        stage.add_argument("--epochs", type=int, default=1)
    for stage in (sft, scoring, dpo):
        stage.add_argument("--max-length", type=int, default=2048)
    for stage in (sft, sample, scoring, pairs, dpo):
        stage.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    args.project_root = args.project_root.resolve()
    args.output = args.output.resolve()
    if OUTPUTS not in args.output.parents:
        parser.error(f"--output must be a new path inside {OUTPUTS}")
    if args.output.exists() and not getattr(args, "resume", False):
        parser.error("--output already exists")
    if getattr(args, "max_samples", None) is not None and args.max_samples < 1:
        parser.error("--max-samples must be positive")
    if getattr(args, "epochs", 1) < 1 or getattr(args, "max_length", 2048) < 128:
        parser.error("epochs must be positive and max-length must be at least 128")
    if args.stage == "pairs" and (not 0 <= args.alpha <= 1 or not math.isfinite(args.delta) or args.delta <= 0):
        parser.error("alpha must be in [0, 1] and delta must be finite and positive")
    sys.path.insert(0, str(args.project_root))
    if args.stage == "pairs":
        export_pairs(args)
        return
    if args.stage == "reward":
        reward(args)
    else:
        from adaqr_support import training

        {"sft": training.sft, "sample": training.sample, "dpo": training.dpo}[args.stage](args)


if __name__ == "__main__":
    main()
