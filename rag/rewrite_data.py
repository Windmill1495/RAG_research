"""Load prepared rewrite samples or official dialogues with optional passage labels."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
import json
from pathlib import Path

from .data import load_examples


REWRITE_DATA_FORMATS = ("auto", "jsonl", "qrecc-json")
OFFICIAL_SPLITS = {"qrecc-training.json": "train", "qrecc-test.json": "test"}


def iter_official_turns(path: Path, *, context_policy: str = "error") -> Iterator[tuple[dict, list[dict], bool]]:
    """Yield each turn with its verified reference history before the current turn."""
    if context_policy not in {"error", "exclude"}:
        raise ValueError("context_policy must be error or exclude")
    rows = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("Official QReCC file must be a nonempty JSON array")
    groups, raw_ids = defaultdict(list), set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Invalid official QReCC row")
        cid, turn = row.get("Conversation_no"), row.get("Turn_no")
        if (isinstance(cid, bool) or not isinstance(cid, (str, int)) or not str(cid).strip()
                or type(turn) is not int or turn < 1):
            raise ValueError("Invalid Conversation_no/Turn_no")
        key = (str(cid), turn)
        if key in raw_ids:
            raise ValueError(f"Duplicate official turn: {cid}-{turn}")
        raw_ids.add(key)
        groups[str(cid)].append(row)

    for group in groups.values():
        history: list[dict] = []
        for row in sorted(group, key=lambda item: item["Turn_no"]):
            turn = row["Turn_no"]
            identifier = f"{row['Conversation_no']}-{turn}"
            context = row.get("Context")
            expected = [message["content"] for message in history]
            valid_context = isinstance(context, list) and context == expected
            if not valid_context and context_policy == "error":
                raise ValueError(f"{identifier}: Context does not match previous human rewrites/nonempty answers; no alternating-role guess is permitted")
            rewrite, answer = row.get("Rewrite") or "", row.get("Answer") or ""
            if not all(isinstance(value, str) for value in (rewrite, answer)):
                raise ValueError(f"{identifier}: rewrite/answer must be strings")
            if not rewrite.strip() and context_policy == "error":
                raise ValueError(f"{identifier}: missing Rewrite prevents exact official-history reconstruction")
            yield row, [message.copy() for message in history], valid_context
            # Current supervision becomes history only for later turns.
            if rewrite.strip():
                history.append({"role": "user", "content": rewrite, "turn_id": turn})
            if answer.strip():
                history.append({"role": "assistant", "content": answer, "turn_id": turn})


def load_rewrite_examples(path: Path, split: str, limit: int | None = None,
                          *, data_format: str = "auto") -> list[dict]:
    """Use prepared QReCC JSONL or the raw official JSON array."""
    path = Path(path)
    if data_format not in REWRITE_DATA_FORMATS:
        raise ValueError("Unknown rewrite data format")
    if split not in {"train", "dev", "test", "all"}:
        raise ValueError("Unknown dataset split")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("limit must be a positive integer")
    if data_format == "auto":
        data_format = "qrecc-json" if path.suffix.lower() == ".json" else "jsonl"
    if data_format == "jsonl":
        examples = load_examples(path, split, limit)
        if any(row["dataset_type"] != "qrecc" for row in examples):
            raise ValueError("Rewrite evaluation requires QReCC samples")
        return examples

    official_split = OFFICIAL_SPLITS.get(path.name.lower())
    if official_split and split not in {"all", official_split}:
        raise ValueError(f"{path.name} is the official {official_split} file; use --split {official_split} or all")
    if split == "all":
        if official_split is None:
            raise ValueError("Specify --split train, dev or test for a renamed QReCC JSON file")
        split = official_split

    examples = []
    for row, history, _ in iter_official_turns(path):
        cid, turn = row["Conversation_no"], row["Turn_no"]
        identifier = f"{cid}-{turn}"
        question = row.get("Question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"{identifier}: Question must be nonempty text")
        if limit is not None and len(examples) >= limit:
            continue
        labels = row.get("Passages")
        if labels is not None and (not isinstance(labels, list) or any(not isinstance(pid, str) for pid in labels)):
            raise ValueError(f"{identifier}: Passages must contain passage ID strings")
        positives = list(dict.fromkeys(pid.strip() for pid in (labels or []) if pid.strip()))
        example = {
            "id": identifier, "conversation_id": cid, "turn": turn, "split": split,
            "source": row.get("Conversation_source", "unknown"),
            "question": question, "history": history, "human_rewrite": row["Rewrite"],
            "dataset_type": "qrecc",
        }
        if positives:
            example["positive_ids"] = positives
        examples.append(example)
    return examples


