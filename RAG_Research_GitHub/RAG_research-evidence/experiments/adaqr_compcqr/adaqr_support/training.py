"""AdaQR stages: SFT -> self sampling -> answer-likelihood rewards -> DPO.

Run the GPU stages on a CUDA host only. Importing this file is lightweight.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

from .adaqr import rewrite_prompt
from .adaqr_model import completion_log_probability, sample_rewrites
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
                         "answer": raw.get("answer", raw.get("reference_answer")),
                         "rewrite_source": raw.get("rewrite_source")})
    ids = set()
    for row in rows:
        if row["id"] in ids:
            raise ValueError("Duplicate training id")
        ids.add(row["id"])
        if not isinstance(row["question"], str) or not row["question"].strip():
            raise ValueError("Training question is missing")
        row["question"] = row["question"].strip()
    return [row for row in rows if row["turn"] > 1 and row["history"]]


def require_cuda():
    from .adaqr_model import require_ml
    torch, AutoModelForCausalLM, AutoTokenizer = require_ml()
    if not torch.cuda.is_available():
        raise RuntimeError("AdaQR QLoRA training requires a visible CUDA GPU")
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
    from transformers import get_linear_schedule_with_warmup
    if not rows:
        raise ValueError("No training examples/pairs")
    rng = random.Random(seed)
    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=learning_rate)
    total_steps = len(rows) * epochs
    warmup = math.ceil(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total_steps)
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
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step == 1 or step % 50 == 0 or step == total_steps:
                print(f"Training {step}/{total_steps} | loss {loss.item():.4f}", flush=True)
    output.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(output)
    return step


def sft(args):
    rows = training_rows(args.seed_data)
    if (not rows or any(row.get("rewrite_source") != args.seed_source or
            not isinstance(row["rewrite"], str) or not row["rewrite"].strip() for row in rows)):
        raise ValueError(f"SFT labels must be nonempty, marked split=train and rewrite_source={args.seed_source}")
    if args.seed_source == "infocqr_initial_fsl" and len(rows) != 3850:
        raise ValueError("Paper SFT requires 3,850 non-first-turn InfoCQR initial-FSL rewrite labels")
    from rag.run_cache import file_sha256

    rng = random.Random(args.seed)
    selected = rng.sample(rows, min(args.max_samples or len(rows), len(rows)))
    print(f"SFT labels: {len(selected)}/{len(rows)} {args.seed_source} examples", flush=True)
    torch, *_ = require_cuda()
    torch.manual_seed(args.seed)
    model, tokenizer = quantized_model(args.base_model)
    def loss(row):
        logp = completion_log_probability(model, tokenizer, rewrite_prompt(row["history"], row["question"]),
                                          row["rewrite"], args.max_length)
        return -logp
    steps = run_optimization(model, selected, loss, 1e-4, args.epochs, args.seed, args.output)
    (args.output / "adaqr_metadata.json").write_text(json.dumps({"stage": "sft", "seed_data": str(args.seed_data),
        "samples": len(selected), "steps": steps, "base_model": args.base_model,
        "seed_source": args.seed_source, "seed_sha256": file_sha256(args.seed_data),
        "history": "full", "seed": args.seed}, indent=2), encoding="utf-8")


def sample(args):
    rows = [row for row in training_rows(args.target_data)
            if isinstance(row["answer"], str) and row["answer"].strip()]
    target_ids = json.loads(args.target_ids.read_text(encoding="utf-8-sig"))
    if (not isinstance(target_ids, list) or len(target_ids) != 39677
            or any(not isinstance(identifier, str) for identifier in target_ids)
            or len(set(target_ids)) != len(target_ids)):
        raise ValueError("AdaQR Table 11 requires a list of 39,677 unique QReCC train IDs")
    by_id = {row["id"]: row for row in rows}
    missing = set(target_ids).difference(by_id)
    if missing:
        raise ValueError(f"{len(missing)} paper training IDs are absent from the answered later-turn input")
    rows = [by_id[identifier] for identifier in target_ids]
    if args.max_samples is not None:
        rows = random.Random(args.seed).sample(rows, min(args.max_samples, len(rows)))
    from peft import PeftConfig

    torch, *_ = require_cuda()
    torch.manual_seed(args.seed)
    base = PeftConfig.from_pretrained(str(args.sft_adapter)).base_model_name_or_path
    model, tokenizer = quantized_model(base, str(args.sft_adapter))
    model.eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_name(args.output.name + ".partial")
    # An interrupted run keeps the partial file separate from completed samples.
    with partial.open("x", encoding="utf-8") as stream:
        for index, row in enumerate(rows, 1):
            prompt = rewrite_prompt(row["history"], row["question"])
            candidates = sample_rewrites(model, tokenizer, prompt, 3, 1.0)
            if not all(candidates):
                raise ValueError(f"Empty sampled rewrite for training turn {row['id']}")
            stream.write(json.dumps({**row, "split": "train", "candidates": candidates},
                                    ensure_ascii=False) + "\n")
            if index == 1 or index % 50 == 0 or index == len(rows):
                print(f"Sampling {index}/{len(rows)}", flush=True)
    partial.rename(args.output)


def dpo(args):
    import gc
    from peft import PeftConfig
    torch, *_ = require_cuda()
    rows = list(read_jsonl(args.preferences))
    if not rows:
        raise ValueError("No qualifying preference pairs; check reward data and threshold")
    torch.manual_seed(args.seed)
    base = PeftConfig.from_pretrained(str(args.sft_adapter)).base_model_name_or_path
    reference, tokenizer = quantized_model(base, str(args.sft_adapter))
    reference.eval()
    for index, row in enumerate(rows, 1):
        with torch.no_grad():
            row["reference_chosen"] = float(completion_log_probability(reference, tokenizer, row["prompt"],
                                                    row["chosen"], args.max_length).item())
            row["reference_rejected"] = float(completion_log_probability(reference, tokenizer, row["prompt"],
                                                    row["rejected"], args.max_length).item())
        if index == 1 or index % 50 == 0 or index == len(rows):
            print(f"Scoring SFT reference {index}/{len(rows)}", flush=True)
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
        "steps": steps, "beta": 0.1, "sft_adapter": str(args.sft_adapter),
        "preferences": str(args.preferences)}, indent=2), encoding="utf-8")
