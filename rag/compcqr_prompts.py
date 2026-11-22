"""Render the pinned LLM4CS and CompCQR assets supplied at runtime."""

import argparse
import ast
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import urllib.request


ASSETS = Path(__file__).resolve().parents[1] / "prompts" / "compcqr"
REVISION = "a292fb88e5b9898af3e5b000ca9a2c3d2c2ea99e"
UPSTREAM = f"https://raw.githubusercontent.com/kyriemao/LLM4CS/{REVISION}"
HASHES = {
    "chat_promptor.py": "6732e1d81f261bcdbb4ab2fe25685940c5cecceccc5dc986d5732f9ea7083d47",
    "llm4cs_demonstrations.json": "accd47e2285eaf349fc07dbd142ae160f04eab32201b6fdef86b8d03603cae80",
    "followup.txt": "f4bd952c483a4fcf800e42c0923c85d7df355692ec4b9452279effcf710f43c6",
    "selection.txt": "3a7a3a51826520f3569a9b1977318fe3f996aaa60054d7fdf3ea41a2a99b8c1c",
}


def asset(name):
    path = ASSETS / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing prompt asset: {path}. See prompts/compcqr/README.md.")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != HASHES[name]:
        raise ValueError(f"Prompt asset checksum differs: {path}")
    return content


@lru_cache(maxsize=1)
def promptor_type():
    source = ast.parse(asset("chat_promptor.py"))
    node, = (node for node in source.body
             if isinstance(node, ast.ClassDef) and node.name == "RewriteAndResponsePromptor")
    scope = {"json": json, "__name__": "_pinned_llm4cs_promptor"}
    # Execute only the selected class from the byte-verified upstream file.
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(ASSETS / "chat_promptor.py"),
                 "exec"), scope)
    return scope[node.name]


def validate_training_assets():
    promptor_type()
    asset("llm4cs_demonstrations.json")
    asset("followup.txt")


def original_rewrite_prompt(question, history):
    turns = []
    for message in history:
        if message["role"] == "user":
            turns.append({"question": message["content"], "response": "N/A"})
        elif turns:
            turns[-1]["response"] = message["content"]
        else:
            raise ValueError("Original RAR context requires a preceding user question")
    asset("llm4cs_demonstrations.json")
    path = ASSETS / "llm4cs_demonstrations.json"
    promptor = promptor_type()(str(path), enable_cot=True)
    if not promptor.demo:
        raise ValueError("Original demonstrations could not be loaded")
    return promptor.build_turn_prompt(turns or None, {"question": question})


def history_text(history):
    return "\n".join(("Question: " if m["role"] == "user" else "Response: ") + m["content"]
                     for m in history) or "N/A"


def followup_prompt(question, history):
    return (asset("followup.txt").decode("utf-8").replace("{history}", history_text(history))
            .replace("{question}", question))


def selection_prompt(question, history, passages):
    text = "\n\n".join("Passage {}. {}".format(p["number"], p["text"]) for p in passages)
    return (asset("selection.txt").decode("utf-8").replace("{TOP_K}", str(len(passages)))
            .replace("{formatted_context}", history_text(history)).replace("{oq}", question)
            .replace("{question}", question).replace("{passage_text}", text))


def parse_rar(raw):
    cls = promptor_type()
    value = cls.__new__(cls)
    value.enable_cot = True
    result = value.parse_returned_text(raw)
    if result is None:
        raise ValueError("Expected original RAR Rewrite/Response output")
    return result


def fetch_llm4cs():
    ASSETS.mkdir(parents=True, exist_ok=True)
    names = {"chat_promptor.py": "chat_promptor.py", "llm4cs_demonstrations.json": "demonstrations.json"}
    for local, remote in names.items():
        path = ASSETS / local
        if path.exists():
            asset(local)
            continue
        with urllib.request.urlopen(f"{UPSTREAM}/{remote}", timeout=30) as response:
            data = response.read(1_048_577)
        if hashlib.sha256(data).hexdigest() != HASHES[local]:
            raise ValueError(f"Downloaded asset checksum differs: {remote}")
        with path.open("xb") as handle:
            handle.write(data)
        print(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch-llm4cs", action="store_true", required=True)
    parser.parse_args()
    fetch_llm4cs()
