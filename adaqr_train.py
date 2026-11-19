"""AdaQR stages: SFT -> self sampling -> answer-likelihood rewards -> DPO.

Run the GPU stages on a CUDA host only. Importing this file is lightweight.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from rag.adaqr import AdaQRSettings, answer_prompt, marginal_reward, preference_pairs, rewrite_prompt
from rag.adaqr_model import completion_log_probability, sample_rewrites
from rag.data import read_jsonl, validate_history
from rag.rewrite_data import iter_official_turns


def training_rows(path: Path) -> list[dict]:
    """Only official QReCC training JSON or explicitly marked training JSONL."""
    if path.suffix == ".json":
        if path.name.lower() != "qrecc-training.json":
            raise ValueError("Official input must be named qrecc-training.json")
        rows = []
        for raw, history, valid in iter_official_turns(path, context_policy="exclude"):
            if not valid:
                continue
            row = {"id": f"{raw['Conversation_no']}-{raw['Turn_no']}",
                   "turn": raw["Turn_no"], "history": history,
                   "question": raw.get("Question"), "rewrite": raw.get("Rewrite"),
                   "answer": raw.get("Answer")}
            rows.append(row)
    else:
        rows = []
        for raw in read_jsonl(path):
            if raw.get("split") != "train":
                raise ValueError("AdaQR training JSONL may contain only split=train")
            rows.append({"id": raw["id"], "turn": raw["turn"],
                         "history": validate_history(raw["history"], raw["id"], raw["turn"]),
                         "question": raw["question"],
                         "rewrite": raw.get("rewrite", raw.get("human_rewrite")),
                         "answer": raw.get("answer", raw.get("reference_answer"))})
    ids = set()
    for row in rows:
        if row["id"] in ids:
            raise ValueError("Duplicate training id")
        ids.add(row["id"])
        if not isinstance(row["question"], str) or not row["question"].strip():
            raise ValueError("Training question is missing")
    return [row for row in rows if row["turn"] > 1 and row["history"]]


def require_cuda():
    from rag.adaqr_model import require_ml
    torch, AutoModelForCausalLM, AutoTokenizer = require_ml()
    if not torch.cuda.is_available():
        raise RuntimeError("AdaQR QLoRA training/reward stages require a CUDA GPU; do not run on this Mac")
    try:
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError("Install peft, bitsandbytes, accelerate and transformers for AdaQR") from exc
    return torch, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training


def quantized_model(base: str, adapter: str | None = None):
    (torch, Model, Tokenizer, Bits, Lora, PeftModel, get_peft_model, prepare) = require_cuda()
    tokenizer = Tokenizer.from_pretrained(base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quant = Bits(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                 bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=False)
    model = Model.from_pretrained(base, quantization_config=quant,
                                  device_map={"": torch.cuda.current_device()})
    model = prepare(model)
    model.config.use_cache = False
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=True)
    else:
        model = get_peft_model(model, Lora(r=8, lora_alpha=16, lora_dropout=0.05,
                                           target_modules=["q_proj", "v_proj"], bias="none", task_type="CAUSAL_LM"))
    return model, tokenizer


def run_optimization(model, rows: list[dict], loss_fn, learning_rate: float,
                     epochs: int, seed: int, output: Path):
    torch, *_ = require_cuda()
    from torch.optim import AdamW
    if not rows:
        raise ValueError("No training examples/pairs")
    rng = random.Random(seed)
    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=learning_rate)
    total_steps = len(rows) * epochs
    warmup = max(1, math.ceil(total_steps * 0.1))
    model.train()
    step = 0
    for _ in range(epochs):
        order = list(range(len(rows)))
        rng.shuffle(order)
        for index in order:
            loss = loss_fn(rows[index])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            scale = step / warmup if step < warmup else max(0.0, (total_steps - step) / max(1, total_steps - warmup))
            for group in optimizer.param_groups:
                group["lr"] = learning_rate * scale
    output.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(output)
    return step


def sft(args):
    rows = [row for row in training_rows(args.seed_data) if isinstance(row["rewrite"], str) and row["rewrite"].strip()]
    if not rows:
        raise ValueError("No later-turn rewrite labels in seed dataset")
    rng = random.Random(args.seed)
    selected = rng.sample(rows, max(1, round(len(rows) * args.seed_fraction)))
    model, tokenizer = quantized_model(args.base_model)
    def loss(row):
        logp = completion_log_probability(model, tokenizer, rewrite_prompt(row["history"], row["question"]),
                                          row["rewrite"], args.max_length)
        return -logp
    steps = run_optimization(model, selected, loss, 1e-4, args.epochs, args.seed, args.output)
    (args.output / "adaqr_metadata.json").write_text(json.dumps({"stage": "sft", "seed_data": str(args.seed_data),
        "seed_fraction": args.seed_fraction, "samples": len(selected), "steps": steps,
        "base_model": args.base_model}, indent=2), encoding="utf-8")


def sample(args):
    from peft import PeftConfig
    rows = [row for row in training_rows(args.target_data) if isinstance(row["answer"], str) and row["answer"].strip()]
    if not rows:
        raise ValueError("No later-turn target examples with answers")
    base = PeftConfig.from_pretrained(args.sft_adapter).base_model_name_or_path
    model, tokenizer = quantized_model(base, str(args.sft_adapter))
    model.eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        for row in rows:
            prompt = rewrite_prompt(row["history"], row["question"])
            candidates = sample_rewrites(model, tokenizer, prompt, 3, 1.0)
            stream.write(json.dumps({**row, "candidates": candidates}, ensure_ascii=False) + "\n")


def reward(args):
    (torch, Model, Tokenizer, _, _, _, _, _) = require_cuda()
    from rag.config import load_config
    from rag.pipeline import RAGEngine
    project = args.project.resolve()
    config = load_config(project, {"retriever_backend": args.backend})
    engine = RAGEngine(project, config, mode="ollama", build_index=False)
    tokenizer = Tokenizer.from_pretrained(args.answer_model)
    scorer = Model.from_pretrained(args.answer_model, torch_dtype=torch.bfloat16, device_map="auto")
    scorer.eval()
    settings = AdaQRSettings()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        for row in read_jsonl(args.samples):
            if len(row["candidates"]) != settings.candidates:
                raise ValueError("Expected exactly three sampled rewrites")
            scored = []
            for rewrite in row["candidates"]:
                hits = engine.retriever.search(rewrite, settings.top_k)
                if not hits:
                    raise ValueError(f"No passages for {row['id']}")
                with torch.no_grad():
                    logps = [float(completion_log_probability(scorer, tokenizer,
                        answer_prompt(row["history"], row["question"], hit["text"]),
                        row["answer"], args.max_length, include_eos=False).item()) for hit in hits]
                scored.append({"rewrite": rewrite, "reward": marginal_reward(hits, logps),
                               "passage_ids": [hit.get("passage_id", hit["id"]) for hit in hits]})
            prompt = rewrite_prompt(row["history"], row["question"])
            for pair in preference_pairs(scored, settings.delta):
                stream.write(json.dumps({"id": row["id"], "prompt": prompt, **pair}, ensure_ascii=False) + "\n")


def dpo(args):
    import gc
    from peft import PeftConfig
    torch, *_ = require_cuda()
    rows = list(read_jsonl(args.preferences))
    if not rows:
        raise ValueError("No qualifying preference pairs; check reward data and threshold")
    base = PeftConfig.from_pretrained(args.sft_adapter).base_model_name_or_path
    reference, tokenizer = quantized_model(base, str(args.sft_adapter))
    reference.eval()
    for row in rows:
        with torch.no_grad():
            row["reference_chosen"] = float(completion_log_probability(reference, tokenizer, row["prompt"],
                                                    row["chosen"], args.max_length).item())
            row["reference_rejected"] = float(completion_log_probability(reference, tokenizer, row["prompt"],
                                                    row["rejected"], args.max_length).item())
    del reference
    gc.collect()
    torch.cuda.empty_cache()
    policy, tokenizer = quantized_model(base, str(args.sft_adapter))
    def loss(row):
        chosen = completion_log_probability(policy, tokenizer, row["prompt"], row["chosen"], args.max_length)
        rejected = completion_log_probability(policy, tokenizer, row["prompt"], row["rejected"], args.max_length)
        margin = (chosen - row["reference_chosen"]) - (rejected - row["reference_rejected"])
        return -torch.nn.functional.logsigmoid(0.1 * margin)
    steps = run_optimization(policy, rows, loss, 1e-5, args.epochs, args.seed, args.output)
    (args.output / "adaqr_metadata.json").write_text(json.dumps({"stage": "dpo", "pairs": len(rows),
        "steps": steps, "beta": 0.1, "delta": 0.1, "sft_adapter": str(args.sft_adapter)}, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["sft", "sample", "reward", "dpo"])
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--seed-data", type=Path)
    parser.add_argument("--target-data", type=Path)
    parser.add_argument("--samples", type=Path)
    parser.add_argument("--preferences", type=Path)
    parser.add_argument("--base-model", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--answer-model", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--sft-adapter", type=Path)
    parser.add_argument("--backend", choices=["bm25", "dense", "hybrid"], default="bm25")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=2048)
    args = parser.parse_args()
    required = {"sft": ("seed_data",), "sample": ("target_data", "sft_adapter"),
                "reward": ("samples",), "dpo": ("preferences", "sft_adapter")}
    if args.output.exists():
        parser.error("--output already exists; choose a new stage output path")
    for field in required[args.stage]:
        if getattr(args, field) is None:
            parser.error(f"{args.stage} requires --{field.replace('_', '-')}")
    if args.epochs < 1 or args.max_length < 128 or not 0 < args.seed_fraction <= 1:
        parser.error("epochs, max-length or seed-fraction out of range")
    {"sft": sft, "sample": sample, "reward": reward, "dpo": dpo}[args.stage](args)


if __name__ == "__main__":
    main()
