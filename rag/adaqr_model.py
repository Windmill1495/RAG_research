"""Optional Hugging Face model operations for AdaQR; never run on import."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path


def require_ml():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("AdaQR requires torch and transformers; install optional AdaQR dependencies") from exc
    return torch, AutoModelForCausalLM, AutoTokenizer


def encode_completion(tokenizer, prompt: str, completion: str, max_length: int,
                      include_eos: bool = True):
    """Keep the complete target and truncate only old prompt context if necessary."""
    prefix = tokenizer.encode(prompt, add_special_tokens=True)
    target = tokenizer.encode(completion, add_special_tokens=False)
    if include_eos and tokenizer.eos_token_id is not None:
        target.append(tokenizer.eos_token_id)
    if len(target) >= max_length:
        raise ValueError("Target answer/rewrite exceeds max_length")
    prefix = prefix[-(max_length - len(target)):]
    ids = prefix + target
    return ids, len(prefix)


def completion_log_probability(model, tokenizer, prompt: str, completion: str,
                               max_length: int = 2048, include_eos: bool = True):
    """Sum token log probabilities of the complete target under causal LM."""
    torch, _, _ = require_ml()
    ids, start = encode_completion(tokenizer, prompt, completion, max_length, include_eos)
    device = next(model.parameters()).device
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    logits = model(input_ids=input_ids).logits[0, start - 1:-1]
    targets = input_ids[0, start:]
    return torch.log_softmax(logits, dim=-1).gather(1, targets[:, None]).sum()


def sample_rewrites(model, tokenizer, prompt: str, count: int = 3,
                    temperature: float = 1.0, max_new_tokens: int = 128) -> list[str]:
    torch, _, _ = require_ml()
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    encoded = {key: value.to(next(model.parameters()).device) for key, value in encoded.items()}
    with torch.no_grad():
        generated = model.generate(**encoded, do_sample=True, temperature=temperature,
                                   num_return_sequences=count, max_new_tokens=max_new_tokens,
                                   pad_token_id=tokenizer.eos_token_id)
    return [tokenizer.decode(sequence[encoded["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            for sequence in generated]


@lru_cache(maxsize=1)
def load_rewriter(adapter_path: str):
    torch, AutoModelForCausalLM, AutoTokenizer = require_ml()
    try:
        from peft import PeftConfig, PeftModel
    except ImportError as exc:
        raise RuntimeError("AdaQR inference requires peft") from exc
    path = Path(adapter_path).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"AdaQR adapter directory not found: {path}")
    peft_config = PeftConfig.from_pretrained(str(path))
    tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path,
                                                 torch_dtype="auto", device_map="auto")
    model = PeftModel.from_pretrained(model, str(path))
    model.eval()
    return model, tokenizer


def rewrite_with_adapter(question: str, history: list[dict], adapter_path: str) -> str:
    from .adaqr import rewrite_prompt
    model, tokenizer = load_rewriter(adapter_path)
    torch, _, _ = require_ml()
    prompt = rewrite_prompt(history, question)
    encoded = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
    with torch.no_grad():
        result = model.generate(**encoded, do_sample=False, max_new_tokens=128,
                                pad_token_id=tokenizer.eos_token_id)
    rewrite = tokenizer.decode(result[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    if not rewrite:
        raise ValueError("AdaQR model generated an empty rewrite")
    return rewrite


def adapter_identity(adapter_path: str) -> dict:
    """Bind evaluation cache to adapter bytes, not only a mutable path."""
    import hashlib
    root = Path(adapter_path).expanduser().resolve()
    config = root / "adapter_config.json"
    weights = root / "adapter_model.safetensors"
    if not config.is_file() or not weights.is_file():
        raise ValueError("AdaQR adapter needs adapter_config.json and adapter_model.safetensors")
    result = {"path": str(root)}
    for file in (config, weights):
        digest = hashlib.sha256()
        with file.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        result[file.name] = digest.hexdigest()
    return result
