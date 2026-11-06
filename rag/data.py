"""Explicit evaluation contracts; supervision stays outside the online pipeline.

QReCC history must already contain roles. Its flat Context cannot safely be
converted with alternating user/assistant labels because answers may be absent.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number}: expected a JSON object")
            yield value


def load_passage_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    for row in read_jsonl(path):
        identifier = row.get("id")
        text = row.get("text", row.get("contents"))
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError("Passage id must be a nonempty string")
        if identifier in ids:
            raise ValueError(f"Duplicate passage id: {identifier}")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Passage {identifier} has no text")
        ids.add(identifier)
    if not ids:
        raise ValueError("Passage corpus is empty")
    return ids


def validate_history(history: Any, identifier: str, current_turn: int | None = None) -> list[dict]:
    if not isinstance(history, list):
        raise ValueError(f"Sample {identifier}: history must be a list")
    result = []
    for message in history:
        if (not isinstance(message, dict) or message.get("role") not in {"user", "assistant"}
                or not isinstance(message.get("content"), str) or not message["content"].strip()):
            raise ValueError(f"Sample {identifier}: history needs explicit user/assistant roles and nonempty content")
        if "turn_id" in message:
            turn = message["turn_id"]
            if type(turn) is not int or turn < 1 or current_turn is not None and turn >= current_turn:
                raise ValueError(f"Sample {identifier}: current/future turn found in history")
        # Preserve turn provenance but never carry arbitrary supervision fields.
        clean = {"role": message["role"], "content": message["content"]}
        if "turn_id" in message:
            clean["turn_id"] = message["turn_id"]
        result.append(clean)
    return result


def load_examples(path: Path, split: str = "all", limit: int | None = None,
                  *, corpus_ids: set[str] | None = None) -> list[dict[str, Any]]:
    """Validate the whole file before selecting a split or debugging prefix."""
    if split not in {"dev", "test", "train", "all"}:
        raise ValueError("Unknown dataset split")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("limit must be a positive integer")
    examples, ids, groups = [], set(), {}
    schema = None
    for raw in read_jsonl(path):
        item = dict(raw)
        identifier = item.get("id")
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError("Every sample requires a nonempty string id")
        if identifier in ids:
            raise ValueError(f"Duplicate sample id: {identifier}")
        ids.add(identifier)
        if item.get("split") not in {"train", "dev", "test"}:
            raise ValueError(f"Sample {identifier}: invalid split")
        if not isinstance(item.get("question"), str) or not item["question"].strip():
            raise ValueError(f"Sample {identifier}: question must be nonempty")
        kind = "qrecc" if "positive_ids" in item else "document"
        if schema is not None and schema != kind:
            raise ValueError("Do not mix document and QReCC schemas in one evaluation file")
        schema = kind
        if kind == "qrecc":
            cid = item.get("conversation_id")
            if isinstance(cid, bool) or not isinstance(cid, (str, int)) or str(cid) == "":
                raise ValueError(f"Sample {identifier}: missing conversation_id")
            turn = item.get("turn")
            if type(turn) is not int or turn < 1:
                raise ValueError(f"Sample {identifier}: invalid turn")
            prior_split = groups.setdefault(str(cid), item["split"])
            if prior_split != item["split"]:
                raise ValueError(f"Conversation {cid} crosses data splits")
            positives = item["positive_ids"]
            if (not isinstance(positives, list) or not positives
                    or any(not isinstance(pid, str) or not pid for pid in positives)
                    or len(set(positives)) != len(positives)):
                raise ValueError(f"Sample {identifier}: positive_ids must be unique nonempty strings")
            if corpus_ids is not None and not set(positives) <= corpus_ids:
                raise ValueError(f"Sample {identifier}: a positive passage is absent from the fixed corpus")
            if not isinstance(item.get("human_rewrite"), str) or not item["human_rewrite"].strip():
                raise ValueError(f"Sample {identifier}: missing human_rewrite")
            item["reference_answer"] = item.get("reference_answer", item.get("answer", ""))
            item["answer_keywords"] = item.get("answer_keywords", [])
            item["history"] = validate_history(item.get("history"), identifier, turn)
        else:
            required = {"gold_document", "reference_answer", "answer_keywords"}
            if required - item.keys():
                raise ValueError(f"Sample {identifier}: missing {sorted(required - item.keys())}")
            document = item["gold_document"]
            if (not isinstance(document, str) or not document or "/" in document or "\\" in document
                    or Path(document).name != document):
                raise ValueError(f"Sample {identifier}: gold_document must be a filename")
            item["history"] = validate_history(item.get("history"), identifier)
        if not isinstance(item["reference_answer"], str):
            raise ValueError(f"Sample {identifier}: reference_answer must be text")
        keywords = item["answer_keywords"]
        if not isinstance(keywords, list) or any(not isinstance(word, str) or not word for word in keywords):
            raise ValueError(f"Sample {identifier}: invalid answer_keywords")
        item["dataset_type"] = kind
        if split == "all" or item["split"] == split:
            examples.append(item)
    if limit is not None:
        examples = examples[:limit]
    if not examples:
        raise ValueError("No samples in the selected split")
    return examples


def public_input(example: dict) -> dict:
    """The only fields forwarded to the RAG engine by an evaluator."""
    return {"question": example["question"], "history": example["history"]}
