"""Train an AdaQR rewriter with single-query and CompCQR retrieval rewards."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys


from adaqr_support.adaqr import AdaQRSettings, preference_pairs
from paper_runtime import MISTRAL

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


def mixed_pairs(record, alpha, delta):
    rows = record.get("candidates", [])
    scored = [
        {"rewrite": row["rewrite"], "reward": (1 - alpha) * row["single_reward"] + alpha * row["fusion_reward"]}
        for row in rows
    ]
    return [{"id": record["id"], "prompt": record["prompt"], "alpha": alpha, "delta": delta, **pair}
            for pair in preference_pairs(scored, delta)]


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
    pairs = stages.add_parser("pairs", help="Export preferences from the saved rewards")
    pairs.add_argument("--rewards", type=Path, required=True)
    pairs.add_argument("--alpha", type=float, default=0.5)
    pairs.add_argument("--delta", type=float, default=SETTINGS.delta)
    dpo = stages.add_parser("dpo", help="Optimize the rewriter on the preference pairs")
    dpo.add_argument("--preferences", type=Path, required=True)
    dpo.add_argument("--sft-adapter", type=Path, required=True)
    for stage in (sft, sample, dpo):
        stage.add_argument("--seed", type=int, default=42)
    for stage in (sft, dpo):
        stage.add_argument("--epochs", type=int, default=1)
    for stage in (sft, dpo):
        stage.add_argument("--max-length", type=int, default=2048)
    for stage in (sft, sample, pairs, dpo):
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
    from adaqr_support import training

    {"sft": training.sft, "sample": training.sample, "dpo": training.dpo}[args.stage](args)


if __name__ == "__main__":
    main()
