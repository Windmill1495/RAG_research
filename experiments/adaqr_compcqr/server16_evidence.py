"""Score fixed-turn evidence using completed Server16 foundation rewards."""

from __future__ import annotations

import argparse
from collections import Counter
from itertools import islice
import json
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory


EXPERIMENT = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT.parents[1]))

import adaqr_half_dataset as dataset
from adaqr_half_rewards import apply_evidence_reward
from evidence_verifier import EvidenceVerifier
from rag.data import read_jsonl
from rag.run_cache import RunCache, file_sha256, make_signature


EXPERIMENT_ID = "adaqr_compcqr_reward_v1"
REQUIRED_VERIFIER_FILES = (
    "config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
)


def load_manifest(directory):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    identity = manifest["identity"]
    if manifest.get("version") != 1 or manifest.get("signature") != make_signature(identity):
        raise ValueError(f"Invalid run signature in {directory}")
    return manifest


def load_foundation(run):
    rows, cohort = dataset.load(run / "dataset")
    samples = {row["id"]: row for row in rows}
    baseline_dir = run / "rewards_baseline"
    manifest = load_manifest(baseline_dir)
    identity = manifest["identity"]
    if (identity.get("experiment") != EXPERIMENT_ID or identity.get("arm") != "baseline"
            or identity.get("cohort") != "half_train_v1"
            or identity.get("sample_count") != len(samples)
            or identity.get("frozen_data") != cohort["files"]):
        raise ValueError("Published baseline and frozen training cohort do not match")
    return samples, baseline_dir / "checkpoint.jsonl", manifest


def validated_rows(path, expected):
    seen = set()
    for row in read_jsonl(path):
        key = row.get("id")
        if key not in expected or key in seen or row.get("cache_key") != key:
            raise ValueError(f"Unexpected or repeated turn ID in {path}: {key}")
        if row.get("status") not in {"ok", "component_error"}:
            raise ValueError(f"Unexpected status in {path}: {key}")
        seen.add(key)
        yield row
    if len(seen) != len(expected):
        raise ValueError(f"Incomplete records in {path}: {len(seen)}/{len(expected)}")


def verifier_identity(directory):
    files = [directory / name for name in REQUIRED_VERIFIER_FILES]
    if any(not p.is_file() or not p.stat().st_size for p in files):
        missing = [p.name for p in files if not p.is_file() or not p.stat().st_size]
        raise FileNotFoundError(f"Missing verifier files at {directory}: {', '.join(missing)}")
    return {p.name: file_sha256(p) for p in files}


def run_identity(args, samples, baseline, manifest, verifier):
    return {
        "experiment": EXPERIMENT_ID, "arm": "evidence", "sample_count": len(samples),
        "cohort": manifest["identity"]["cohort"],
        "foundation_signature": manifest["signature"],
        "foundation_checkpoint_sha256": file_sha256(baseline),
        "verifier": verifier,
        "implementation": {
            name: file_sha256(EXPERIMENT / name)
            for name in ("adaqr_half_rewards.py", "evidence_verifier.py", "server16_evidence.py")
        },
        "hypothesis": "shared_turn_compcqr_mq_plus_reference_answer",
        "evidence_weight": args.weight, "evidence_epsilon": args.epsilon,
        "batch_size": args.batch_size, "workers": args.workers, "device": args.device,
    }


def shard_dir(output, rank):
    return output / "work" / f"rank-{rank}"


def refresh_reward(args, model, row, sample):
    return apply_evidence_reward(row, sample["answer"], model.score_many,
                                 args.weight, args.epsilon)


def score(args, samples, path, identity, assignments):
    directory = shard_dir(args.output, args.rank)
    cache = RunCache(directory, {**identity, "rank": args.rank}, resume=directory.exists())
    assigned = assignments[args.rank]
    expected = set(assigned)
    if not cache.completed_keys.issubset(expected):
        raise ValueError(f"Unexpected saved reference on rank {args.rank}")
    pending = expected - cache.completed_keys
    print(f"Evidence | rank {args.rank} | restored {len(cache.rows)}/{len(assigned)}", flush=True)
    if not pending:
        return

    model = EvidenceVerifier(args.verifier, device=args.device, batch_size=args.batch_size)
    for row in validated_rows(path, samples):
        key = row["id"]
        if key in pending:
            result = refresh_reward(args, model, row, samples[key])
            cache.append(result, key)
            done = len(cache.rows)
            if done == 1 or done % 50 == 0 or done == len(assigned):
                print(f"Evidence | rank {args.rank} | {done}/{len(assigned)} saved", flush=True)
    if cache.completed_keys != expected:
        raise ValueError(f"Scoring did not finish for rank {args.rank}")


def merge(args, samples, identity, assignments):
    destination = args.output / "rewards_evidence"
    if destination.exists():
        if load_manifest(destination)["identity"] != identity:
            raise ValueError(f"Existing evidence run differs from the requested inputs: {destination}")
        count = sum(1 for _ in validated_rows(destination / "checkpoint.jsonl", samples))
        print(f"Evidence rewards already published | {count} turns | {destination}", flush=True)
        return

    collected = {}
    for rank, assigned in enumerate(assignments):
        directory = shard_dir(args.output, rank)
        if load_manifest(directory)["identity"] != {**identity, "rank": rank}:
            raise ValueError(f"Evidence shard signature differs: {directory}")
        collected.update((row["id"], row)
                         for row in validated_rows(directory / "checkpoint.jsonl", set(assigned)))

    with TemporaryDirectory(prefix=".evidence-merge-", dir=args.output) as temporary:
        staged = Path(temporary) / "published"
        output = RunCache(staged, identity)
        with output.checkpoint_path.open("w", encoding="utf-8") as stream:
            for key in samples:
                stream.write(json.dumps(collected[key], ensure_ascii=False, allow_nan=False) + "\n")
        staged.rename(destination)
    print(f"Published evidence rewards | {len(collected)} turns | {destination}", flush=True)


def status(output):
    for path in sorted((output / "work").glob("rank-*/checkpoint.jsonl")):
        with path.open("rb") as stream:
            lines = sum(line.endswith(b"\n") for line in stream)
        print(f"Evidence | {path.parent.name} | {lines} records")
    print("Merged evidence rewards:", (output / "rewards_evidence/checkpoint.jsonl").is_file())
    print("Evidence preferences:", (output / "preferences_evidence.jsonl").is_file())
    print("Evidence DPO adapter:", (output / "dpo_evidence/adaqr_metadata.json").is_file())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("check", "smoke", "score", "merge", "status"))
    parser.add_argument("--foundation-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verifier", type=Path, required=True)
    parser.add_argument("--weight", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--limit", type=int, default=2, help="Number of turns for the read-only smoke test")
    args = parser.parse_args()
    if (not math.isfinite(args.weight) or args.weight < 0 or not 0 < args.epsilon <= 1
            or args.batch_size < 1 or args.limit < 1 or not 0 <= args.rank < args.workers):
        parser.error("Invalid evidence weights, batch size, limit, or worker/rank count")
    args.output = args.output.resolve()
    args.foundation_run = args.foundation_run.resolve()
    if (args.output == args.foundation_run or args.foundation_run in args.output.parents
            or args.output in args.foundation_run.parents):
        parser.error("Use a separate output directory from the existing foundation run")
    if args.stage == "status":
        status(args.output)
        return

    samples, baseline, manifest = load_foundation(args.foundation_run)
    model_hashes = verifier_identity(args.verifier)
    if args.stage == "check":
        counts = Counter(row["status"] for row in validated_rows(baseline, samples))
        print(f"Foundation rewards verified | {len(samples)} turns | {dict(counts)}")
        print(f"Local verifier verified | {args.verifier}")
        return
    if args.stage == "smoke":
        accepted = (row for row in validated_rows(baseline, samples) if row["status"] == "ok")
        model = EvidenceVerifier(args.verifier, device=args.device, batch_size=args.batch_size)
        for row in islice(accepted, args.limit):
            scored = refresh_reward(args, model, row, samples[row["id"]])
            print(f"Evidence smoke | {row['id']} | {len(scored['candidates'])} candidate rewrites validated",
                  flush=True)
        return

    identity = run_identity(args, samples, baseline, manifest, model_hashes)
    ordered = list(samples)
    assignments = [ordered[rank::args.workers] for rank in range(args.workers)]
    if args.stage == "score":
        score(args, samples, baseline, identity, assignments)
    else:
        merge(args, samples, identity, assignments)


if __name__ == "__main__":
    main()
