"""Reconstruct the 39,677 QReCC DPO inputs reported in AdaQR Table 11."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


SOURCE_MD5 = "72abc475c62b99546f4c61192cd721f3"
INFOCQR_DEV_SHA256 = "b3c1a2fd5c7aeccd3837c74aa3c6a017e09a04c2f69bf554353e98c9bf2e8607"
INFOCQR_TRAIN_SHA256 = "aaa0e437ea2cefc7b3356db56260b916eb5b09b377cb96bd1838488dc844e0a4"
IDS_SHA256 = "237e9611ee9ac9c791c26b2c8fb715d820b8e202fafe78e402ecb9db43669aaa"
PRIOR_ANSWER_HASH = "ab3046d22a50ff98f2c8f0b9159848daaa8af08d236c1b8d56e891b8498a57a9"
STORED_CONTEXT_HASH = "3b6923c06f8577cbf8e3a0b14f32e1556f5c0ed2a7ce6bb4a8102ab3ce00d12b"
CONTEXT_VARIANTS = {"9435-3", "9435-4", "9435-5"}


def read_json(path: Path, digest: str, algorithm: str = "sha256"):
    data = path.read_bytes()
    if hashlib.new(algorithm, data).hexdigest() != digest:
        raise ValueError(f"Unexpected {algorithm} checksum for {path}")
    return json.loads(data)


def canonical_content(rows, question_key, answer_key):
    return Counter((str(row["Conversation_no"]), row[question_key], row[answer_key]) for row in rows)


def compose_rows(official, dev, published_train, expected_ids):
    development_conversations = {str(r["Conversation_no"]) for r in dev}
    if len(official) != 63501 or len(dev) != 11573 or len(development_conversations) != 2000:
        raise ValueError("The QReCC corpus or published dev split has unexpected counts")

    training = [row for row in official if str(row["Conversation_no"]) not in development_conversations]
    if len(training) != 51928 or sum(bool(row["Answer"].strip()) for row in training) != 47463:
        raise ValueError("The reconstructed QReCC split does not match AdaQR Table 11")
    if canonical_content(training, "Rewrite", "Answer") != canonical_content(published_train, "Truth_rewrite", "Truth_answer"):
        raise ValueError("Original IDs cannot be aligned with the published InfoCQR training records")

    # The official record order, even where a conversation's IDs are not
    # monotonic, records the conversation order validated by Context.
    histories = defaultdict(list)
    output = []
    variants = set()
    for row in training:
        cid = str(row["Conversation_no"])
        turn = row["Turn_no"]
        identifier = f"{cid}-{turn}"
        history = histories[cid]
        context = row["Context"]
        known_content = [message["content"] for message in history]
        if context != known_content:
            different = [index for index, (stored, known) in enumerate(zip(context, known_content)) if stored != known]
            if (identifier not in CONTEXT_VARIANTS or len(context) != len(history) or different != [1]
                    or history[1]["role"] != "assistant"
                    or hashlib.sha256(known_content[1].encode()).hexdigest() != PRIOR_ANSWER_HASH
                    or hashlib.sha256(context[1].encode()).hexdigest() != STORED_CONTEXT_HASH):
                raise ValueError(f"Unexpected conversation-history mismatch at {identifier}")
            variants.add(identifier)

        if turn > 1 and row["Answer"].strip():
            if not history or not row["Question"].strip():
                raise ValueError(f"Missing previous history or question at {identifier}")
            output.append({"id": identifier, "split": "train", "turn": turn,
                           "history": [{"role": message["role"], "content": content}
                                       for message, content in zip(history, context)],
                           "question": row["Question"], "answer": row["Answer"]})

        for role, field in (("user", "Rewrite"), ("assistant", "Answer")):
            if row[field].strip():
                history.append({"role": role, "content": row[field]})

    if ([row["id"] for row in output] != expected_ids or len(output) != 39677
            or len(set(expected_ids)) != len(output) or variants != CONTEXT_VARIANTS):
        raise ValueError("Normalized output does not match the verified DPO IDs or history variants")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-train", type=Path, required=True)
    parser.add_argument("--infocqr-dev", type=Path, required=True)
    parser.add_argument("--infocqr-train", type=Path, required=True)
    parser.add_argument("--ids-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.ids_output.exists() or args.output.resolve() == args.ids_output.resolve():
        parser.error("Choose two distinct new output paths")

    official = read_json(args.official_train, SOURCE_MD5, "md5")
    dev = read_json(args.infocqr_dev, INFOCQR_DEV_SHA256)
    assigned = {str(row["Conversation_no"]) for row in dev}
    ids = [f"{row['Conversation_no']}-{row['Turn_no']}" for row in official
           if str(row["Conversation_no"]) not in assigned and row["Turn_no"] > 1
           and row["Answer"].strip()]
    encoded = (json.dumps(ids, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    if hashlib.sha256(encoded).hexdigest() != IDS_SHA256:
        raise ValueError("Derived IDs differ from the known 39,677-turn selection")
    rows = compose_rows(official, dev, read_json(args.infocqr_train, INFOCQR_TRAIN_SHA256), ids)
    args.ids_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.ids_output.open("xb") as ids_file:
        ids_file.write(encoded)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"{len(rows):,} validated rows in {args.output}")


if __name__ == "__main__":
    main()
