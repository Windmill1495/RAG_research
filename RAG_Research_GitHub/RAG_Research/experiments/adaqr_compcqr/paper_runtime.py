"""Paper model identifiers and Pyserini settings for the QReCC BM25 experiments."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


MISTRAL = "mistralai/Mistral-7B-v0.1"
LLAMA = "meta-llama/Llama-3.1-8B-Instruct"
BM25_K1 = 0.82
BM25_B = 0.68
RRF_CONFIG = {
    "retriever_backend": "bm25",
    "compcqr_max_components": 5,
    "compcqr_retrieval_depth": 100,
    "compcqr_rrf_k": 0,
}


class PaperBM25:
    """Read a prepared Lucene index with retrievable QReCC passage text."""

    def __init__(self, directory: Path):
        from pyserini.search.lucene import LuceneSearcher

        self.directory = directory.resolve()
        if not self.directory.is_dir() or not list(self.directory.glob("segments_*")):
            raise ValueError(f"Pass the extracted QReCC Pyserini Lucene index: {self.directory}")
        self.searcher = LuceneSearcher(str(self.directory))
        self.searcher.set_bm25(BM25_K1, BM25_B)
        self._documents = {}

    def identity(self):
        files = sorted(p for p in self.directory.iterdir() if p.is_file() and p.name != "write.lock")
        return {"path": str(self.directory), "k1": BM25_K1, "b": BM25_B,
                "files": {p.name: [p.stat().st_size, p.stat().st_mtime_ns] for p in files}}

    def passage_text(self, identifier):
        if identifier not in self._documents:
            document = self.searcher.doc(identifier)
            raw = document.raw() if document is not None else None
            stored = json.loads(raw) if raw else {}
            fields = (stored.get("contents"), stored.get("text"), stored.get("passage"))
            text = next((value for value in fields if isinstance(value, str) and value.strip()), None)
            if text is None:
                raise ValueError(f"Lucene index must store passage text for AdaQR scoring: {identifier}")
            self._documents[identifier] = text
        return self._documents[identifier]

    def search_many(self, queries, top_k):
        self._documents.clear()
        return [[{"id": hit.docid, "passage_id": hit.docid, "score": float(hit.score)}
                 for hit in self.searcher.search(query, k=top_k)] for query in queries]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class OriginalCompCQRClient:
    """Reusable official HF chat/decode wrapper, with explicit resume-stable seeds.

    Official source: github.com/YunahJang/CompCQR, compcqr/llm.py. The paper's
    open-model setting uses Llama-3.1-8B-Instruct. RAR uses 256 new tokens,
    follow-up and selection use 1,024, all with temperature 0.7 and top_p 1.
    Model operations are lazy: importing this standalone file never loads torch.
    """

    def __init__(self, model_name=LLAMA, *, seed=42, device="cuda:1",
                 model=None, tokenizer=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch, self.seed, self.model_name = torch, seed, str(model_name)
        self.device = str(device)
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("The original auxiliary model requires an available CUDA device")
        if (model is None) != (tokenizer is None):
            raise ValueError("Pass both a preloaded model and tokenizer, or neither")
        self.model = (model if model is not None else AutoModelForCausalLM.from_pretrained(
            self.model_name, device_map={"": self.device}, torch_dtype=torch.bfloat16)).eval()
        self.tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(
            self.model_name, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.context_length = 8192

    def generate(self, prompt, system="", temperature=0.7):
        # The project calls this interface with temperature=0 for its baseline;
        # this dedicated wrapper intentionally applies the original paper recipe.
        limit = 256 if prompt.startswith("For an information-seeking dialog") else 1024
        messages = [{"role": "system", "content": system or "You are a helpful assistant."},
                    {"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        if encoded["input_ids"].shape[1] + limit > self.context_length:
            raise ValueError("Original CompCQR prompt plus output budget exceeds 8,192 tokens; no silent truncation")
        encoded = {key: value.to(next(self.model.parameters()).device) for key, value in encoded.items()}
        prompt_seed = (self.seed + int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)) % (2**31)
        devices = list(range(self.torch.cuda.device_count())) if self.torch.cuda.is_available() else []
        with self.torch.random.fork_rng(devices=devices), self.torch.no_grad():
            self.torch.manual_seed(prompt_seed)
            output = self.model.generate(**encoded, do_sample=True, temperature=0.7, top_p=1.0, top_k=0,
                max_new_tokens=limit, num_return_sequences=1, pad_token_id=self.tokenizer.pad_token_id)
        answer = self.tokenizer.decode(output[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        if not answer:
            raise ValueError("Original auxiliary model generated empty output")
        return answer

    def identity(self):
        config = self.model.config
        result = {"model": self.model_name, "resolved_commit": getattr(config, "_commit_hash", None),
                  "recipe": {"system": "You are a helpful assistant.", "context_length": 8192,
                    "temperature": 0.7, "top_p": 1.0, "rar_max_new_tokens": 256,
                    "dtype": "bfloat16", "device": self.device,
                    "other_max_new_tokens": 1024, "seed": self.seed,
                    "seed_policy": "base_seed_plus_sha256_chat_prompt_mod_2**31"}}
        path = Path(self.model_name).expanduser()
        if path.is_dir():
            files = sorted(p for p in path.iterdir() if p.is_file() and
                           (p.suffix in {".safetensors", ".json", ".model", ".txt"}
                            or p.name.startswith("pytorch_model")))
            result["local_files_sha256"] = {p.name: sha256(p) for p in files}
            if not any(p.suffix == ".safetensors" or p.name.startswith("pytorch_model") for p in files):
                raise ValueError("Auxiliary model has no identifiable local weights")
        elif not result["resolved_commit"]:
            raise ValueError("Auxiliary Hugging Face model has no resolved commit identity")
        return result


