"""Prepare every official QReCC turn, independently of passage coverage."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from rag.full_data import OFFICIAL_FILES, download_official, file_digest

PROJECT_ROOT = Path(__file__).resolve().parent


def iter_full_official_turns(path: Path, *, answer_overrides: list | None = None):
    """Recover roles using preceding Rewrite anchors; preserve current Context text.

    Official training conversation 8048 has numeric turns 1,2,3,5,4 in its actual
    Context chain. Preserve those labels separately and number chronological
    positions for the prepared role/turn provenance. Conversation 9435 contains
    a different prior-answer wording in later Contexts. Context is authoritative:
    user slots must match known preceding Rewrite labels exactly, assistant slots
    may use that row's actual Context wording, with every override recorded.
    No current Rewrite/Answer enters the yielded history. Unmatched roles or
    question anchors fail rather than filter or infer roles by alternating.
    """
    rows = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("Official QReCC input must be a nonempty JSON array")
    groups, seen = defaultdict(list), set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Invalid official QReCC row")
        cid, turn = row.get("Conversation_no"), row.get("Turn_no")
        if (isinstance(cid, bool) or not isinstance(cid, (str, int)) or not str(cid).strip()
                or type(turn) is not int or turn < 1):
            raise ValueError("Invalid official conversation/turn label")
        key = (str(cid), turn)
        if key in seen:
            raise ValueError(f"Duplicate official turn: {cid}-{turn}")
        seen.add(key)
        if not isinstance(row.get("Context"), list) or any(not isinstance(x, str) for x in row["Context"]):
            raise ValueError(f"{cid}-{turn}: Context must contain strings")
        groups[str(cid)].append(row)
    for group in groups.values():
        slots = []
        for position, row in enumerate(sorted(group, key=lambda value: len(value["Context"])), 1):
            context = row["Context"]
            if len(context) != len(slots):
                raise ValueError(f"{row['Conversation_no']}-{row['Turn_no']}: Context does not match verified preceding role slots")
            history = []
            for content, slot in zip(context, slots):
                if slot["role"] == "user" and content != slot["content"]:
                    raise ValueError(f"{row['Conversation_no']}-{row['Turn_no']}: Context does not match verified preceding Rewrite anchors")
                if slot["role"] == "assistant" and content != slot["content"] and answer_overrides is not None:
                    answer_overrides.append({"conversation": str(row["Conversation_no"]),
                                             "official_turn": row["Turn_no"], "chronological_turn": position,
                                             "prior_chronological_turn": slot["turn_id"],
                                             "original_answer": slot["content"], "context_answer": content})
                history.append({**slot, "content": content})
            rewrite, answer = row.get("Rewrite"), row.get("Answer") or ""
            if not isinstance(rewrite, str) or not rewrite.strip() or not isinstance(answer, str):
                raise ValueError("Every full-source row requires a nonempty Rewrite and a text Answer")
            yield row, history, position
            slots.append({"role": "user", "content": rewrite, "turn_id": position})
            if answer.strip():
                slots.append({"role": "assistant", "content": answer, "turn_id": position})


def prepare_full_dataset(training: Path, test: Path, output: Path, *, dev_ratio: float = .1,
                         split_seed: str = "qrecc-full-v1", verify_official: bool = True) -> dict:
    """Preserve all turns and original gold IDs; only TRAINING conversations split."""
    if not 0 <= dev_ratio < 1:
        raise ValueError("dev_ratio must be in [0,1)")
    training, test, output = Path(training), Path(test), Path(output)
    verified = (file_digest(training, "md5") == OFFICIAL_FILES["qrecc-training.json"]
                and file_digest(test, "md5") == OFFICIAL_FILES["qrecc-test.json"])
    if verify_official and not verified:
        raise ValueError("Training/test inputs do not match the pinned official QReCC MD5 values")
    sources = {"train": file_digest(training), "test": file_digest(test)}
    identity = {"format_version": 3, "source_sha256": sources,
                "dev_ratio": dev_ratio, "split_seed": split_seed, "official_sources_verified": verified}
    output.mkdir(parents=True, exist_ok=True)
    manifest_path, questions_path = output / "manifest.json", output / "questions.jsonl"
    if manifest_path.exists() or questions_path.exists():
        if manifest_path.exists() and questions_path.exists():
            old = json.loads(manifest_path.read_text(encoding="utf-8"))
            if old.get("identity") == identity and old.get("questions_sha256") == file_digest(questions_path):
                return old
        raise FileExistsError("Prepared output differs or is incomplete; select a fresh data directory")
    temporary = output / "questions.jsonl.part"
    if temporary.exists():
        raise FileExistsError("An unfinished .part file exists; inspect it or choose a new output directory")
    counts, conversations, annotations, source_counts = Counter(), {}, Counter(), Counter()
    total, order_anomalies, answer_overrides = 0, [], []
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as writer:
            for source_split, path in (("train", training), ("test", test)):
                source_overrides = []
                for raw, history, turn in iter_full_official_turns(path, answer_overrides=source_overrides):
                    cid, official_turn = str(raw["Conversation_no"]), raw["Turn_no"]
                    if turn != official_turn:
                        order_anomalies.append({"official_split": source_split, "conversation": cid,
                                                "official_turn": official_turn, "chronological_turn": turn})
                    question = raw.get("Question")
                    if not isinstance(question, str) or not question.strip():
                        raise ValueError(f"{source_split}:{cid}-{turn}: missing Question")
                    gold = raw.get("Passages") or []
                    if not isinstance(gold, list) or any(not isinstance(pid, str) or not pid.strip() for pid in gold):
                        raise ValueError(f"{source_split}:{cid}-{turn}: invalid original Passage IDs")
                    positives = list(dict.fromkeys(gold))
                    group_id = f"{source_split}:{cid}"
                    ratio = int.from_bytes(hashlib.sha256(f"{split_seed}:{cid}".encode()).digest()[:8], "big") / 2**64
                    split = "test" if source_split == "test" else ("dev" if ratio < dev_ratio else "train")
                    conversations[group_id] = split
                    record = {"id": f"{source_split}:{cid}-{official_turn}", "official_id": f"{cid}-{official_turn}",
                              "conversation_id": group_id, "official_conversation_id": raw["Conversation_no"],
                              "turn": turn, "official_turn": official_turn, "split": split, "official_split": source_split,
                              "source": raw.get("Conversation_source", "unknown"), "dataset_type": "qrecc",
                              "question": question, "history": history, "human_rewrite": raw["Rewrite"],
                              "reference_answer": raw.get("Answer") or "", "answer_keywords": [],
                              "positive_ids": positives,
                              "history_provenance": "actual current-row Context; roles anchored by previous official Rewrite and nonempty Answer slots; current labels excluded"}
                    writer.write(json.dumps(record, ensure_ascii=False) + "\n")
                    counts[split] += 1
                    source_counts[source_split] += 1
                    annotations[split] += bool(positives)
                    total += 1
                answer_overrides.extend({"official_split": source_split, **item} for item in source_overrides)
        questions_hash = file_digest(temporary)
        manifest = {"identity": identity, "scope": "official_full_questions" if verified else "custom_questions",
                    "dataset_scope": "full_qrecc" if verified else "custom", "official_sources_verified": verified,
                    "complete": True, "complete_dialogues": True, "dev_from_training_only": True,
                    "questions_file": "questions.jsonl", "question_count": total,
                    "questions": total, "n_questions": total, "splits": dict(counts),
                    "split_counts": dict(counts), "counts_by_split": dict(counts),
                    "conversations": len(conversations),
                    "conversations_by_split": dict(Counter(conversations.values())),
                    "annotated_questions_by_split": dict(annotations),
                    "questions_sha256": questions_hash,
                    "source_files": [{"split": split, "path": (path.resolve().relative_to(output.resolve()).as_posix()
                                      if path.resolve().is_relative_to(output.resolve()) else str(path.resolve())),
                                      "original_count": source_counts[split], "sha256": sources[split]}
                                     for split, path in (("train", training), ("test", test))],
                    "official_sources": {split: {"sha256": sources[split], "original_count": source_counts[split]}
                                         for split in ("train", "test")},
                    "turn_order_anomaly_count": len(order_anomalies), "turn_order_anomalies": order_anomalies,
                    "context_answer_override_count": len(answer_overrides), "context_answer_overrides": answer_overrides,
                    "question_policy": "All official turns retained, including first turns, missing answers and empty positive labels. No passage coverage filter or gold intersection.",
                    "split_policy": "Official test unchanged; deterministic conversation-level dev sampled only from official training. all includes train+dev+test and is not an independent test score.",
                    "history_policy": "Order rows by Context length. Preserve each current-row Context verbatim; role/turn slots derive only from preceding Rewrite and nonempty Answer records. User anchors must match previous Rewrite exactly; differing prior assistant wording is retained and reported. Missing/extra slots or unmatched user anchors fail. Canonical turn is chronological position; original IDs/official_turn retained. Current labels are evaluation-only.",
                    "retrieval_requirement": "Full questions do not establish full-corpus retrieval. Build and verify the complete original passage index separately.",
                    "source": "https://zenodo.org/records/5760304"}
        temporary.replace(questions_path)
        with manifest_path.open("x", encoding="utf-8", newline="\n") as writer:
            json.dump(manifest, writer, ensure_ascii=False, indent=2)
            writer.write("\n")
        return manifest
    except Exception:
        # Keep partial evidence for inspection; never present it as prepared data.
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data/qrecc_full")
    parser.add_argument("--training", type=Path)
    parser.add_argument("--test", type=Path)
    parser.add_argument("--download-dialogs", action="store_true", help="Download verified training/test JSON only")
    parser.add_argument("--download-passages", action="store_true", help="Explicitly download the 27.5GB official passage ZIP")
    parser.add_argument("--dev-ratio", type=float, default=.1)
    parser.add_argument("--split-seed", default="qrecc-full-v1")
    args = parser.parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    raw = data_dir / "raw"
    try:
        if args.download_dialogs:
            for name in ("qrecc-training.json", "qrecc-test.json"):
                print(f"Preparing official {name}...", flush=True)
                download_official(name, raw)
        if args.download_passages:
            print("Explicit full passage download requested (27.5GB compressed).", flush=True)
            download_official("passages.zip", data_dir)
        training = args.training or raw / "qrecc-training.json"
        test = args.test or raw / "qrecc-test.json"
        if not training.is_file() or not test.is_file():
            raise FileNotFoundError("Supply --training/--test or run with --download-dialogs; no small-subset fallback is permitted")
        manifest = prepare_full_dataset(training, test, data_dir, dev_ratio=args.dev_ratio, split_seed=args.split_seed)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Full QReCC preparation failed: {error}\n")
    print(json.dumps({key: manifest[key] for key in ("scope", "questions", "splits", "annotated_questions_by_split")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
