"""Run the AdaQR DPO stage on four CUDA devices, one preference pair per rank."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
from datetime import timedelta
from pathlib import Path
import sys
from tempfile import TemporaryDirectory


EXPERIMENT = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT.parents[1]))

from adaqr_support.adaqr_model import adapter_identity, completion_log_probability
from adaqr_support.training import quantized_model
from rag.data import read_jsonl
from rag.run_cache import RunCache, file_sha256


BETA = 0.1
LEARNING_RATE = 1e-5
WORLD_SIZE = 4


def reference_checkpoint(args, rank, model_name, sft_identity, resume_only=False):
    directory = args.output.with_name(args.output.name + "_reference") / f"rank-{rank}"
    if resume_only and not directory.is_dir():
        raise ValueError(f"Reference scoring is incomplete on rank {rank}")
    source = Path(model_name)
    weights = {part.name: [part.stat().st_size, part.stat().st_mtime_ns]
               for part in sorted(source.glob("*.safetensors"))}
    identity = {
        "stage": "adaqr_dpo_reference_v2",
        "sft": sft_identity,
        "model": model_name,
        "model_weights": weights,
        "model_config": file_sha256(source / "config.json") if weights else None,
        "preferences_sha256": file_sha256(args.preferences),
        "reference_implementation": {
            name: file_sha256(EXPERIMENT / "adaqr_support" / name)
            for name in ("training.py", "adaqr_model.py")
        },
        "seed": args.seed,
        "max_length": args.max_length,
        "rank": rank,
        "world_size": WORLD_SIZE,
    }
    return RunCache(directory, identity, resume=directory.exists())


def score_reference(model, tokenizer, items, max_length, rank, checkpoint):
    import torch

    model.eval()
    cached = {}
    saved = {row["cache_key"]: row for row in checkpoint.rows}
    for pair_id, row in items:
        if pair_id in saved:
            for label in ("chosen", "rejected"):
                cached[(row["prompt"], row[label])] = saved[pair_id][f"reference_{label}"]
            continue
        scores = {}
        with torch.no_grad():
            for label in ("chosen", "rejected"):
                input_key = (row["prompt"], row[label])
                if input_key not in cached:
                    value = completion_log_probability(model, tokenizer, *input_key, max_length).item()
                    if not math.isfinite(value):
                        raise ValueError(f"Nonfinite reference score on rank {rank}")
                    cached[input_key] = value
                scores[f"reference_{label}"] = cached[input_key]
        checkpoint.append(scores, pair_id)
        complete = len(checkpoint.completed_keys)
        if complete == 1 or complete % 100 == 0 or complete == len(items):
            print(f"Reference | rank {rank} | {complete}/{len(items)} pairs saved", flush=True)


def preference_loss_module(policy, tokenizer, max_length):
    import torch

    class PreferenceLoss(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.policy = policy

        def forward(self, row):
            scores = [completion_log_probability(self.policy, tokenizer, row["prompt"],
                                                 row[label], max_length)
                      for label in ("chosen", "rejected")]
            margin = (scores[0] - row["reference_chosen"]) - (scores[1] - row["reference_rejected"])
            return -torch.nn.functional.logsigmoid(BETA * margin)

    return PreferenceLoss()


def optimize(model, rows, total, world_size, rank):
    import torch
    import torch.distributed as dist
    from transformers import get_linear_schedule_with_warmup

    steps = math.ceil(total / world_size)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(optimizer, math.ceil(steps * 0.1), steps)
    model.train()
    for step in range(steps):
        live_ranks = min(world_size, total - step * world_size)
        if step < len(rows):
            loss = model(rows[step])
        else:
            # All ranks participate in the final reduction. Padding contributes no gradient.
            loss = model(rows[0]) * 0
        if not torch.isfinite(loss):
            raise ValueError(f"Nonfinite DPO loss on rank {rank}, step {step + 1}")
        (loss * (world_size / live_ranks)).backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        if step == 0 or (step + 1) % 50 == 0 or step + 1 == steps:
            mean = loss.detach().clone()
            dist.all_reduce(mean)
            if rank == 0:
                print(f"Training {step+1}/{steps} | pairs {min(total, (step+1)*world_size)}/{total}"
                      f" | loss {(mean/live_ranks).item():.4f}", flush=True)
    return steps


def save_policy(policy, output, metadata):
    with TemporaryDirectory(prefix=".dpo-publish-", dir=output.parent) as temp:
        staged = Path(temp) / "adapter"
        policy.save_pretrained(staged)
        if not (staged / "adapter_model.safetensors").is_file():
            raise RuntimeError("DPO did not save adapter weights")
        (staged / "adaqr_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        staged.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("reference", "train"))
    parser.add_argument("--preferences", type=Path, required=True)
    parser.add_argument("--sft-adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=2048)
    args = parser.parse_args()

    import torch
    import torch.distributed as dist
    from peft import PeftConfig
    from torch.nn.parallel import DistributedDataParallel

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != WORLD_SIZE or rank != local_rank or torch.cuda.device_count() != WORLD_SIZE:
        raise RuntimeError("Launch on one host with four visible CUDA devices using torchrun")
    output = args.output.resolve()
    if (EXPERIMENT / "outputs") not in output.parents or output.exists():
        raise ValueError("DPO output must be a new directory within experiment outputs")
    if args.max_length < 128:
        raise ValueError("Maximum sequence length must be at least 128")
    args.output = output
    rows = [(f"pair-{index}", row) for index, row in enumerate(read_jsonl(args.preferences))]
    total = len(rows)
    if total < world_size:
        raise ValueError("Four-card DPO requires at least four preference pairs")
    random.Random(args.seed).shuffle(rows)
    local_items = rows[rank::world_size]
    del rows

    sft = str(args.sft_adapter.resolve())
    identity = adapter_identity(sft)
    model_name = PeftConfig.from_pretrained(sft).base_model_name_or_path
    checkpoint = reference_checkpoint(args, rank, model_name, identity, resume_only=args.phase == "train")
    expected = {key for key, _ in local_items}
    if not checkpoint.completed_keys.issubset(expected):
        raise ValueError(f"Unrecognized reference scores on rank {rank}")
    torch.cuda.set_device(local_rank)
    torch.manual_seed(args.seed + rank)

    if args.phase == "reference":
        print(f"Reference | rank {rank} | restored {len(checkpoint.completed_keys)}/{len(local_items)} pairs",
              flush=True)
        if checkpoint.completed_keys != expected:
            reference, tokenizer = quantized_model(model_name, sft)
            score_reference(reference, tokenizer, local_items, args.max_length, rank, checkpoint)
        if checkpoint.completed_keys != expected:
            raise RuntimeError(f"Reference scoring did not finish on rank {rank}")
        print(f"Reference | rank {rank} | {len(local_items)}/{len(local_items)} complete", flush=True)
        return

    if checkpoint.completed_keys != expected:
        raise ValueError(f"Reference scoring is incomplete on rank {rank}")
    saved = {row["cache_key"]: row for row in checkpoint.rows}
    local_rows = [{**row, **saved[key]} for key, row in local_items]
    del checkpoint, local_items, saved
    gc.collect()
    policy, tokenizer = quantized_model(model_name, sft)
    # Each loss runs both completions through the same checkpointed policy.
    policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    dist.init_process_group("nccl", timeout=timedelta(minutes=45))
    try:
        dist.barrier(device_ids=[local_rank])
        model = DistributedDataParallel(preference_loss_module(policy, tokenizer, args.max_length),
                                        device_ids=[local_rank], output_device=local_rank,
                                        broadcast_buffers=False)
        steps = optimize(model, local_rows, total, world_size, rank)
        if rank == 0:
            save_policy(policy, output, {
                "stage": "dpo", "pairs": total, "steps": steps, "epochs": 1, "beta": BETA,
                "learning_rate": LEARNING_RATE, "seed": args.seed, "max_length": args.max_length,
                "world_size": world_size, "global_batch_pairs": world_size,
                "sft_adapter": sft, "sft_identity": identity,
                "preferences": str(args.preferences.resolve()),
                "preferences_sha256": file_sha256(args.preferences),
            })
            print(f"DPO completed | {output}", flush=True)
        dist.barrier(device_ids=[local_rank])
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
