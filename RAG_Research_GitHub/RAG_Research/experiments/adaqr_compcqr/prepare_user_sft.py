"""Prepare user-selected QReCC train rows for AdaQR SFT."""
from collections import defaultdict
import argparse
import hashlib
import json
from pathlib import Path


def read_json(path, md5=None):
    data = path.read_bytes()
    if md5 and hashlib.md5(data).hexdigest() != md5:
        raise ValueError(f"Unexpected official QReCC file hash: {path}")
    return json.loads(data)


def raw_id(row):
    return f"{row['Conversation_no']}-{row['Turn_no']}"


def test_key(row):
    normal = lambda text: " ".join(text.casefold().split())
    return normal(row["Question"]), tuple(normal(message) for message in row["Context"] if message.strip())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--official-train", type=Path, required=True)
    parser.add_argument("--official-test", type=Path, required=True)
    parser.add_argument("--verified-dpo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sample = read_json(args.sample)
    official = read_json(args.official_train, "72abc475c62b99546f4c61192cd721f3")
    test = read_json(args.official_test, "7ad29296a1dfd2f75dafd653a8822269")
    if not isinstance(sample, list) or not sample:
        raise ValueError("Expected a nonempty array of QReCC training samples")
    eligible = {raw_id(row): row for row in sample}
    if len(eligible) != len(sample):
        raise ValueError("Duplicate IDs in the supplied SFT sample")
    official_by_id = {raw_id(row): row for row in official}
    if any(row != official_by_id.get(identifier) for identifier, row in eligible.items()):
        raise ValueError("All supplied records must exactly match the official training split")
    held_out_inputs = {test_key(row) for row in test}
    with args.verified_dpo.open(encoding="utf-8") as stream:
        reference = {row["id"]: row["history"] for row in map(json.loads, stream)}
    history = defaultdict(list)
    prepared = {}
    for row in official:
        identifier = raw_id(row)
        messages = history[row["Conversation_no"]]
        if identifier in eligible and row["Turn_no"] > 1:
            if test_key(row) in held_out_inputs:
                raise ValueError(f"SFT input also occurs in the official test set: {identifier}")
            if [message["content"] for message in messages] != row["Context"]:
                messages_to_copy = reference.get(identifier, [])
                if [message["content"] for message in messages_to_copy] != row["Context"]:
                    raise ValueError(f"Could not verify conversation history roles: {identifier}")
            else:
                messages_to_copy = messages
            if not messages_to_copy or not row["Rewrite"].strip():
                raise ValueError(f"Missing history or rewrite: {identifier}")
            prepared[identifier] = {"id": identifier, "split": "train", "turn": row["Turn_no"],
                                    "history": messages_to_copy.copy(), "question": row["Question"],
                                    "rewrite": row["Rewrite"], "rewrite_source": "user_qrecc"}
        for role, field in (("user", "Rewrite"), ("assistant", "Answer")):
            if row[field].strip():
                messages.append({"role": role, "content": row[field]})
    if len(prepared) != sum(row["Turn_no"] > 1 for row in sample):
        raise ValueError("Some later training rows were lost")
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        for identifier in eligible:
            if identifier in prepared:
                stream.write(json.dumps(prepared[identifier], ensure_ascii=False) + "\n")
    print(json.dumps({"source_rows": len(sample), "sft_rows": len(prepared),
                      "excluded_first_turns": len(sample) - len(prepared),
                      "label_source": "official QReCC human Rewrite from the user's sample",
                      "source_sha256": hashlib.sha256(args.sample.read_bytes()).hexdigest(),
                      "sft_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}), flush=True)


if __name__ == "__main__":
    main()
