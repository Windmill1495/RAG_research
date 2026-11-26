"""Prepare an index-first QReCC subset with verified, explicitly role-tagged history.

The corpus must be fixed independently before this script sees questions. The
result is coverage-conditioned subset evaluation, not official full-corpus QReCC.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

from rag.data import load_examples, load_passage_ids, read_jsonl
from rag.rewrite_data import iter_official_turns
from rag.run_cache import file_sha256


def prepare_dataset(passages_path: Path, raw_path: Path, output_dir: Path,
                    selected_path: Path | None = None, context_policy: str = "error",
                    split_salt: str = "20261101") -> dict:
    passages_path, raw_path, output_dir = Path(passages_path), Path(raw_path), Path(output_dir)
    if context_policy not in {"error", "exclude"}:
        raise ValueError("context_policy must be error or exclude")
    if any((output_dir / name).exists() for name in ("questions.jsonl", "manifest.json")):
        raise FileExistsError("Refusing to overwrite prepared questions or manifest; choose a new output folder")
    original_ids, filtered_empty_ids, valid_passages = set(), [], []
    for passage in read_jsonl(passages_path):
        pid, text = passage.get("id"), passage.get("text", passage.get("contents"))
        if not isinstance(pid, str) or not pid.strip() or pid in original_ids:
            raise ValueError("Source passage ids must be nonempty and unique")
        original_ids.add(pid)
        if not isinstance(text, str):
            raise ValueError(f"Passage {pid} text must be a string")
        if not text.strip():
            filtered_empty_ids.append(pid)
            continue
        valid_passages.append({"id": pid, "text": text})
    corpus_ids = {row["id"] for row in valid_passages}
    if not corpus_ids:
        raise ValueError("Source corpus has no nonempty passage")
    if filtered_empty_ids and (output_dir / "passages.jsonl").resolve() == passages_path.resolve():
        raise ValueError("Filtering requires a separate source passage file; original input will not be overwritten")
    selected = None
    if selected_path is not None:
        selected = {}
        for row in read_jsonl(Path(selected_path)):
            if not isinstance(row.get("id"), str) or row.get("split") not in {"dev", "test", "train"}:
                raise ValueError("Selected file needs id and split")
            if row["id"] in selected:
                raise ValueError(f"Duplicate selected id: {row['id']}")
            selected[row["id"]] = row["split"]
    questions, excluded, eligible_ids = [], [], set()
    for row, history, valid_context in iter_official_turns(raw_path, context_policy=context_policy):
        identifier = f"{row['Conversation_no']}-{row['Turn_no']}"
        if not valid_context:
            excluded.append({"id": identifier, "reason": "official_context_mismatch"})
        positives = row.get("Passages") or []
        if not isinstance(positives, list) or any(not isinstance(pid, str) for pid in positives):
            raise ValueError(f"{identifier}: invalid Passages")
        positives = sorted(corpus_ids.intersection(positives))
        answer, rewrite, question = row.get("Answer") or "", row.get("Rewrite") or "", row.get("Question") or ""
        if not isinstance(question, str):
            raise ValueError(f"{identifier}: question must be a string")
        eligible = (row["Turn_no"] >= 2 and positives and answer.strip()
                    and rewrite.strip() and question.strip() and valid_context)
        if eligible:
            eligible_ids.add(identifier)
            if selected is None or identifier in selected:
                hashed = hashlib.sha256(f"{split_salt}:{row['Conversation_no']}".encode()).digest()
                split = selected[identifier] if selected is not None else ("dev" if int.from_bytes(hashed[:4], "big") % 3 == 0 else "test")
                questions.append({
                    "id": identifier, "conversation_id": row["Conversation_no"], "turn": row["Turn_no"],
                    "source": row.get("Conversation_source", "unknown"), "split": split,
                    "question": question, "history": history,
                    "human_rewrite": rewrite, "reference_answer": answer,
                    "positive_ids": positives, "answer_keywords": [],
                    "history_provenance": "official previous human Rewrite and nonempty reference Answer, verified against Context",
                })
    if selected is not None:
        missing = sorted(set(selected) - {question["id"] for question in questions})
        if missing and context_policy == "error":
            raise ValueError(f"Selected samples are missing or ineligible: {missing[:10]}")
    else:
        missing = []
    if not questions:
        raise ValueError("No eligible questions in this fixed corpus")
    # Validate all outputs before writing; in particular group-disjoint splits.
    split_by_conversation = {}
    for question in questions:
        cid = str(question["conversation_id"])
        if split_by_conversation.setdefault(cid, question["split"]) != question["split"]:
            raise ValueError(f"Selected file splits conversation {cid} across subsets")
    output_dir.mkdir(parents=True, exist_ok=True)
    target_corpus = output_dir / "passages.jsonl"
    if filtered_empty_ids:
        with target_corpus.open("x", encoding="utf-8", newline="\n") as handle:
            for passage in valid_passages:
                handle.write(json.dumps(passage, ensure_ascii=False) + "\n")
    elif target_corpus.resolve() != passages_path.resolve():
        if target_corpus.exists():
            if file_sha256(target_corpus) != file_sha256(passages_path):
                raise FileExistsError("Output corpus differs from the supplied fixed corpus")
        else:
            shutil.copyfile(passages_path, target_corpus)
    target_questions = output_dir / "questions.jsonl"
    with target_questions.open("x", encoding="utf-8", newline="\n") as handle:
        for question in questions:
            handle.write(json.dumps(question, ensure_ascii=False) + "\n")
    load_examples(target_questions, corpus_ids=corpus_ids)
    manifest = {
        "design": "independently supplied fixed passage corpus, then eligible covered QReCC turns",
        "corpus_input_sha256": file_sha256(passages_path), "official_input_sha256": file_sha256(raw_path),
        "selected_input_sha256": file_sha256(Path(selected_path)) if selected_path else None,
        "source_passage_count": len(original_ids), "filtered_empty_count": len(filtered_empty_ids),
        "filtered_empty_ids": filtered_empty_ids,
        "n_passages": len(corpus_ids), "n_questions": len(questions),
        "n_conversations": len(split_by_conversation), "splits": dict(Counter(row["split"] for row in questions)),
        "split_policy": "preserved selected ids/splits" if selected is not None else f"conversation SHA256, salt={split_salt}, modulo 3",
        "history_policy": "previous Rewrite as user, previous nonempty Answer as assistant; exact Context equality; turn_id retained",
        "context_policy": context_policy, "context_mismatch_rows": excluded,
        "missing_selected_ids": missing, "eligible_before_selected_filter": len(eligible_ids),
        "limitation": "Coverage-conditioned fixed-prefix subset; not official full-corpus QReCC. Reference history differs from model-generated rolling dialogue. Previously inspected test cases are diagnostic, not an untouched final test.",
        "frozen_files_sha256": {"passages": file_sha256(target_corpus), "questions": file_sha256(target_questions)},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--passages", required=True, type=Path)
    parser.add_argument("--qrecc-test", required=True, type=Path, help="Official original QReCC JSON array, not the v2 flattened subset")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--selected", type=Path, help="Optional v2 questions.jsonl; retain its exact ids and splits")
    parser.add_argument("--context-policy", choices=["error", "exclude"], default="error")
    parser.add_argument("--split-salt", default="20261101")
    args = parser.parse_args()
    try:
        manifest = prepare_dataset(args.passages, args.qrecc_test, args.out, args.selected, args.context_policy, args.split_salt)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Preparation failed: {error}\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
