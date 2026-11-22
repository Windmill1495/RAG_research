"""Freeze a stratified half-size training cohort for the two AdaQR reward arms."""

import json
from pathlib import Path
import random
import uuid

from rag.data import read_jsonl
from rag.run_cache import file_sha256
from train_fusion import candidates_for


SOURCE_COUNT = 39_677
DRAW_COUNT = 19_839
SEED = 42
VERSION = 1


def stratum(row):
    turn = row["turn"]
    history_chars = sum(len(message["content"]) for message in row["history"])
    turn_group = "02-03" if turn <= 3 else "04-06" if turn <= 6 else "07+"
    history_group = "0000-0500" if history_chars <= 500 else "0501-1500" if history_chars <= 1500 else "1501+"
    return f"{turn_group}/{history_group}"


def stratified_draw(rows, count, seed):
    if not 0 < count <= len(rows):
        raise ValueError("Invalid number of training turns to draw")
    groups = {}
    for row in rows:
        groups.setdefault(stratum(row), []).append(row["id"])
    quotas = {name: divmod(count * len(ids), len(rows)) for name, ids in groups.items()}
    additional = count - sum(quota for quota, _ in quotas.values())
    bonuses = set(sorted(groups, key=lambda name: (-quotas[name][1], name))[:additional])
    rng = random.Random(seed)
    selected = set()
    by_stratum = {}
    for name, ids in sorted(groups.items()):
        target = quotas[name][0] + (name in bonuses)
        selected.update(rng.sample(sorted(ids), target))
        by_stratum[name] = {"source": len(ids), "selected": target, "eligible": 0}
    if len(selected) != count:
        raise ValueError("Training data must have unique turn IDs")
    return selected, by_stratum


def load(dataset_dir):
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    if (manifest["version"], manifest["seed"], manifest["source_count"], manifest["draw_count"]) != (
            VERSION, SEED, SOURCE_COUNT, DRAW_COUNT):
        raise ValueError("Half-training dataset configuration differs from the frozen design")
    paths = {name: dataset_dir / name for name in ("eligible.jsonl", "selected_ids.json")}
    for name, path in paths.items():
        if manifest["files"][name] != file_sha256(path):
            raise ValueError(f"Frozen training data changed: {path}")
    selected = json.loads(paths["selected_ids.json"].read_text(encoding="utf-8"))
    if len(selected) != DRAW_COUNT or len(set(selected)) != DRAW_COUNT:
        raise ValueError("Frozen draw IDs must be unique and complete")
    rows = list(read_jsonl(paths["eligible.jsonl"]))
    ids = {row["id"] for row in rows}
    if (len(ids) != len(rows) or len(rows) != manifest["eligible_count"]
            or not ids.issubset(set(selected)) or any(len(candidates_for(row)) < 2 for row in rows)):
        raise ValueError("Invalid eligible half-training data")
    return rows, manifest


def prepare(source, dataset_dir):
    if dataset_dir.exists():
        rows, manifest = load(dataset_dir)
        if source.is_file() and file_sha256(source) != manifest["source_sha256"]:
            raise ValueError("Full source sample file differs from the frozen draw")
        return rows, manifest
    rows = list(read_jsonl(source))
    if len(rows) != SOURCE_COUNT or len({row["id"] for row in rows}) != SOURCE_COUNT:
        raise ValueError("Expected the full 39,677-turn QReCC training sample file")
    draw, counts = stratified_draw(rows, DRAW_COUNT, SEED)
    selected = []
    eligible = []
    for row in rows:
        if row["id"] in draw:
            selected.append(row["id"])
            if len(candidates_for(row)) >= 2:
                eligible.append(row)
                counts[stratum(row)]["eligible"] += 1

    temporary = dataset_dir.with_name(f".{dataset_dir.name}-{uuid.uuid4().hex}")
    temporary.mkdir(parents=True)
    selection_file = temporary / "selected_ids.json"
    selection_file.write_text(json.dumps(selected, ensure_ascii=False) + "\n", encoding="utf-8")
    eligible_file = temporary / "eligible.jsonl"
    with eligible_file.open("x", encoding="utf-8") as stream:
        for row in eligible:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "version": VERSION, "seed": SEED,
        "source_count": SOURCE_COUNT, "source_sha256": file_sha256(source),
        "draw_count": len(selected), "eligible_count": len(eligible),
        "excluded_single_distinct_candidate": len(selected) - len(eligible),
        "split": "train", "strata": counts,
        "files": {path.name: file_sha256(path) for path in (selection_file, eligible_file)},
    }
    (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.rename(dataset_dir)
    return load(dataset_dir)
