"""Half-size AdaQR + CompCQR experiment on Server16 GPUs 4 and 5."""

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(PROJECT))

import adaqr_half_dataset as dataset
import adaqr_half_rewards as rewards
from adaqr_component_formats import recovered_plan
from adaqr_support.adaqr_model import completion_log_probability, require_ml
from paper_runtime import OriginalCompCQRClient, PaperBM25, RRF_CONFIG
from rag.compcqr import PROMPT_VERSION
from rag.data import read_jsonl
from rag.run_cache import RunCache, file_sha256, make_signature
from train_fusion import SETTINGS


RUN = ROOT / "outputs/train_half_v1"
LEGACY = ROOT / "outputs/train4000_v1"
SOURCE = LEGACY / "samples.jsonl"
LOCK = LEGACY / ".fusion.lock"
SFT = LEGACY / "sft"
INDEX = Path("/data1/zixuan/qrecc_full/lucene-papers/lucene")
MODELS = Path("/data1/zixuan/models")
MISTRAL = MODELS / "Mistral-7B-v0.1"
LLAMA = MODELS / "Llama-3.1-8B-Instruct"
DEFAULT_NLI = MODELS / "DeBERTa-v3-base-mnli-fever-anli"
GPUS = (4, 5)
SHARDS = len(GPUS)
SEED = 42
MAX_LENGTH = 2048
ARCHIVE = PROJECT.parent / "adaqr_compcqr_archive/half-transition-v1"
OLD_WORK = ("rewards_gpu01234_work", "rewards_gpu01234_work_v2", "rewards_gpu01234_work_v3")


def stats(path):
    st = path.stat()
    return [st.st_size, st.st_mtime_ns]


def model_identity(directory):
    weights = {path.name: stats(path) for path in sorted(directory.glob("*.safetensors"))}
    if not weights:
        raise FileNotFoundError(f"No safetensors at {directory}")
    return {"path": str(directory), "weights": weights,
            "config_sha256": file_sha256(directory / "config.json")}


def component_sources():
    return [
        ROOT / "paper_runtime.py", ROOT / "adaqr_component_formats.py",
        PROJECT / "rag/compcqr.py", PROJECT / "rag/compcqr_prompts.py",
        PROJECT / "prompts/compcqr/followup.txt",
        PROJECT / "prompts/compcqr/llm4cs_demonstrations.json",
        PROJECT / "rag/rewriters.py",
    ]


def file_hashes(paths):
    return {str(path): file_sha256(path) for path in paths}


def component_identity(manifest):
    return {
        "experiment": "adaqr_compcqr_reward_v1", "cohort": "half_train_v1",
        "source_sha256": manifest["source_sha256"], "sample_count": manifest["eligible_count"],
        "draw_count": manifest["draw_count"], "frozen_data": manifest["files"],
        "shards": SHARDS, "seed": SEED, "prompt_version": PROMPT_VERSION,
        "retrieval": RRF_CONFIG, "component_model": model_identity(LLAMA),
        "component_format_recovery": "strict_parser_then_explicit_fields_v1",
        "code": file_hashes([*component_sources(), Path(__file__), ROOT / "adaqr_half_dataset.py"]),
    }


def cache_path(phase, slot):
    return RUN / "work" / phase / f"shard-{slot}"


def cache_for(phase, slot, identity):
    path = cache_path(phase, slot)
    return RunCache(path, {**identity, "phase": phase, "slot": slot}, resume=path.exists())


def saved_records(path):
    with path.open("rb") as stream:
        for line in stream:
            if not line.endswith(b"\n"):
                break
            yield json.loads(line)


def source_compatible(previous, current, version):
    for field in ("seed", "prompt_version", "retrieval", "component_model"):
        if previous.get(field) != current[field]:
            return False
    if previous.get("samples_sha256") != current["source_sha256"]:
        return False
    parser = ROOT / "adaqr_component_formats.py"
    known_files = [path for path in component_sources() if path.name != "rewriters.py"]
    for path in known_files:
        if path == parser and version == 2:
            continue
        if previous.get("code", {}).get(str(path)) != current["code"][str(path)]:
            return False
    return version != 3 or previous.get("component_format_recovery") == current["component_format_recovery"]


def seed_old_components(rows, identity):
    destination = {row["id"]: index % SHARDS for index, row in enumerate(rows)}
    caches = [cache_for("components", slot, identity) for slot in range(SHARDS)]
    reused = Counter()
    for version in (3, 2):
        directory = LEGACY / f"rewards_gpu01234_work_v{version}" / "components"
        for manifest_file in sorted(directory.glob("shard-*/manifest.json")):
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            previous = manifest["identity"]
            if (make_signature(previous) != manifest["signature"]
                    or previous.get("phase") != "components"
                    or not source_compatible(previous, identity, version)):
                raise ValueError(f"Cannot safely reuse components from {manifest_file}")
            for row in saved_records(manifest_file.with_name("checkpoint.jsonl")):
                key = row["id"]
                if key not in destination or (version == 2 and row["status"] != "ok"):
                    continue
                cache = caches[destination[key]]
                if key not in cache.completed_keys:
                    cache.append({**row, "component_origin": f"legacy_v{version}"}, key)
                    reused[row["status"]] += 1
    if reused:
        print(f"Reused legacy components | {dict(reused)}", flush=True)
    print(f"Retained component checkpoint rows | {sum(len(cache.rows) for cache in caches)}",
          flush=True)


def archive_legacy():
    # The caller holds the original .fusion.lock while validating and moving files.
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    artifacts = ARCHIVE / "outputs"
    artifacts.mkdir(exist_ok=True)
    code = [
        "parallel_adaqr_sample.py", "parallel_adaqr_sample.py.before-gpu-threshold",
        "parallel_adaqr_rewards.py", "parallel_adaqr_rewards_v2.py",
        "parallel_adaqr_rewards_v3.py", "parallel_adaqr_rewards_v3.py.before-gpu45",
        "adaqr_component_probe.py", "run_train4000_fusion.sh",
    ]
    data = [
        "samples.jsonl", "samples.jsonl.partial", "parallel_samples", "parallel.log",
        "component_diagnostics_v2.jsonl", "component_diagnostics_v2.log",
        "reward.log", "reward_gpu01234.log", "reward_gpu01234_v2.log", "reward_gpu01234_v3.log",
        "reward_gpu45_v3.log", "rewards", "rewards_gpu01234", "preferences.jsonl",
        "preferences_gpu01234.jsonl", "dpo",
        *OLD_WORK,
    ]
    pairs = [(ROOT / name, ARCHIVE / name) for name in code]
    pairs.append((ROOT / "train4000-fusion.log", artifacts / "train4000-fusion.log"))
    pairs += [(LEGACY / name, artifacts / name) for name in data]
    for source, target in pairs:
        if source.exists() and target.exists():
            raise FileExistsError(f"Archive already contains {target}")
    moved = []
    for source, target in pairs:
        if source.exists():
            source.rename(target)
            moved.append(str(target))
    print(f"Moved {len(moved)} obsolete scripts and outputs out of the active run | {ARCHIVE}",
          flush=True)


def completed_shards(phase, rows, identity):
    merged = {}
    for slot in range(SHARDS):
        path = cache_path(phase, slot)
        if not path.is_dir():
            raise ValueError(f"Finish {phase} before starting the next stage")
        cache = cache_for(phase, slot, identity)
        if cache.completed_keys != {row["id"] for row in rows[slot::SHARDS]}:
            raise ValueError(f"Finish {phase} before starting the next stage ({path.name} pending)")
        merged.update((row["id"], row) for row in cache.rows)
    return merged


def get_baseline_identity(components):
    return {**components, "arm": "baseline", "answer_model": model_identity(MISTRAL),
            "max_length": MAX_LENGTH, "top_k": SETTINGS.top_k,
            "index": {path.name: stats(path) for path in sorted(INDEX.iterdir())
                      if path.is_file() and path.name != "write.lock"},
            "reward_code": file_hashes([ROOT / "adaqr_half_rewards.py",
                                        ROOT / "train_fusion.py",
                                        ROOT / "adaqr_support/adaqr.py",
                                        ROOT / "adaqr_support/adaqr_model.py"])}


def get_evidence_identity(baseline, args):
    rewards.validate_model(args.nli_model)
    checkpoint = RUN / "rewards_baseline/checkpoint.jsonl"
    return {**baseline, "arm": "evidence", "baseline_checkpoint_sha256": file_sha256(checkpoint),
            "nli_model": model_identity(args.nli_model), "nli_revision": rewards.NLI_REVISION,
            "hypothesis": rewards.HYPOTHESIS_VERSION, "lambda": args.evidence_weight,
            "epsilon": args.evidence_epsilon, "nli_batch_size": args.nli_batch_size}


def component_operation():
    torch, model_class, tokenizer_class = require_ml()
    tokenizer = tokenizer_class.from_pretrained(str(LLAMA), local_files_only=True)
    model = model_class.from_pretrained(str(LLAMA), local_files_only=True,
                                        torch_dtype=torch.bfloat16, device_map={"": 0}).eval()
    client = OriginalCompCQRClient(str(LLAMA), seed=SEED, device="cuda:0", model=model,
                                  tokenizer=tokenizer)

    def generate(row):
        plan, repairs = recovered_plan(row, client, RRF_CONFIG)
        if plan["source"] == "compcqr_llm":
            record = {"id": row["id"], "status": "ok", "components": plan["components"]}
        else:
            reports = plan["trace"]["stage_errors"]
            recoverable = {"ValueError", json.JSONDecodeError.__name__}
            error_types = [report["error"].partition(": ")[2].partition(": ")[0]
                           for report in reports]
            if not error_types or not set(error_types).issubset(recoverable):
                raise RuntimeError(f"CompCQR failed for {row['id']}: {plan['error']}")
            record = {"id": row["id"], "status": "component_error", "error": plan["error"]}
        if repairs:
            record["format_recoveries"] = repairs
        return record

    return generate


def baseline_operation(components):
    retriever = PaperBM25(INDEX)
    torch, model_class, tokenizer_class = require_ml()
    tokenizer = tokenizer_class.from_pretrained(str(MISTRAL), local_files_only=True)
    model = model_class.from_pretrained(str(MISTRAL), local_files_only=True,
                                        torch_dtype=torch.bfloat16, device_map={"": 0}).eval()

    def logp(prompt, answer):
        with torch.inference_mode():
            return float(completion_log_probability(model, tokenizer, prompt, answer,
                                                    MAX_LENGTH, include_eos=False).item())

    def score(row):
        record = components[row["id"]]
        if record["status"] != "ok":
            return record
        result = rewards.score_turn(row, retriever=retriever, components=record["components"],
                                    answer_logp=logp, config=RRF_CONFIG)
        if "format_recoveries" in record:
            result["format_recoveries"] = record["format_recoveries"]
        if "component_origin" in record:
            result["component_origin"] = record["component_origin"]
        return result

    return score


def phase_operation(phase, slot, identity):
    if phase == "components":
        return component_operation()
    if phase == "baseline":
        values = {row["id"]: row for row in read_jsonl(
            cache_path("components", slot) / "checkpoint.jsonl")}
        return baseline_operation(values)
    values = {row["id"]: row for row in read_jsonl(
        cache_path("baseline", slot) / "checkpoint.jsonl")}
    scorer = rewards.NliScorer(identity["nli_model"]["path"], identity["nli_batch_size"])
    return lambda row: rewards.evidence_adjusted(row, values[row["id"]], scorer,
                                                 identity["lambda"], identity["epsilon"])


def worker(phase, slot, identity):
    rows, _ = dataset.load(RUN / "dataset")
    rows = rows[slot::SHARDS]
    cache = cache_for(phase, slot, identity)
    operation = phase_operation(phase, slot, identity)
    errors = sum(row["status"] != "ok" for row in cache.rows)
    gpu = GPUS[slot]
    print(f"{phase} | GPU {gpu} | {len(cache.rows)}/{len(rows)} restored", flush=True)
    for row in rows:
        key = row["id"]
        if key in cache.completed_keys:
            continue
        result = operation(row)
        cache.append(result, key)
        errors += result["status"] != "ok"
        done = len(cache.rows)
        if done == 1 or done % 25 == 0 or done == len(rows):
            print(f"{phase} | GPU {gpu} | {done}/{len(rows)} | errors {errors}", flush=True)


def gpu_map(indices):
    devices = {}
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.free", "--format=csv,noheader,nounits"], text=True)
    minimum = int(os.environ.get("ADAQR_GPU_MIN_FREE_MIB", "22000"))
    for line in output.splitlines():
        index, identifier, free = (part.strip() for part in line.split(","))
        if int(index) in indices:
            if int(free) < minimum:
                raise RuntimeError(f"GPU {index} has {free} MiB free, need {minimum}")
            devices[int(index)] = identifier
    if devices.keys() != set(indices):
        raise RuntimeError("nvidia-smi did not return the requested physical GPUs")
    return devices


def launch_workers(phase, rows, identity, lock_fd):
    needs = []
    for slot in range(SHARDS):
        cache = cache_for(phase, slot, identity)
        expected = {row["id"] for row in rows[slot::SHARDS]}
        if not cache.completed_keys.issubset(expected):
            raise ValueError(f"Unexpected IDs in {cache.checkpoint_path}")
        if cache.completed_keys != expected:
            needs.append(slot)
    devices = gpu_map([GPUS[slot] for slot in needs]) if needs else {}
    processes = {}
    try:
        for slot in needs:
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": devices[GPUS[slot]]}
            command = [sys.executable, "-u", "-B", str(Path(__file__).resolve()), "worker",
                       "--worker-phase", phase, "--slot", str(slot),
                       "--identity", json.dumps(identity)]
            processes[slot] = subprocess.Popen(command, env=env, pass_fds=(lock_fd,))
        while processes:
            for slot, child in tuple(processes.items()):
                status = child.poll()
                if status == 0:
                    del processes[slot]
                elif status is not None:
                    raise RuntimeError(f"{phase} shard {slot} failed with code {status}; rerun to resume")
            if processes:
                time.sleep(1)
    finally:
        for child in processes.values():
            if child.poll() is None:
                child.terminate()
        for child in processes.values():
            child.wait()


def publish(phase, rows, identity):
    destination = RUN / f"rewards_{phase}"
    if destination.exists():
        final = RunCache(destination, identity, resume=True)
        if final.completed_keys != {row["id"] for row in rows}:
            raise ValueError(f"Incomplete merged rewards: {destination}")
        print(f"Already published | {destination}", flush=True)
        return
    records = completed_shards(phase, rows, identity)
    temporary = RUN / f".rewards_{phase}-{uuid.uuid4().hex}"
    cache = RunCache(temporary, identity)
    with cache.checkpoint_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(records[row["id"]], ensure_ascii=False) + "\n")
    temporary.rename(destination)
    print(f"Published {len(records)} reward records | {destination}", flush=True)


def status():
    _, manifest = dataset.load(RUN / "dataset")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    for phase in ("components", "baseline", "evidence"):
        counts = Counter()
        for path in (RUN / "work" / phase).glob("shard-*/checkpoint.jsonl"):
            counts.update(row["status"] for row in saved_records(path))
        print(f"{phase} | {sum(counts.values())}/{manifest['eligible_count']} | {dict(counts)}")


def main():
    import fcntl

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "prepare-and-archive", "components",
                                         "baseline", "evidence", "fetch-nli", "status", "worker"))
    parser.add_argument("--nli-model", type=Path, default=DEFAULT_NLI)
    parser.add_argument("--evidence-weight", type=float, default=1.0)
    parser.add_argument("--evidence-epsilon", type=float, default=0.1)
    parser.add_argument("--nli-batch-size", type=int, default=16)
    parser.add_argument("--worker-phase", choices=("components", "baseline", "evidence"),
                        help=argparse.SUPPRESS)
    parser.add_argument("--slot", type=int, choices=range(SHARDS), help=argparse.SUPPRESS)
    parser.add_argument("--identity", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (not math.isfinite(args.evidence_weight) or args.evidence_weight < 0
            or not 0 < args.evidence_epsilon <= 1 or args.nli_batch_size < 1):
        parser.error("Evidence weight must be finite and nonnegative, epsilon in (0,1], batch positive")
    if args.stage == "worker":
        worker(args.worker_phase, args.slot, json.loads(args.identity))
        return
    if args.stage == "fetch-nli":
        rewards.fetch_model(args.nli_model)
        return
    if args.stage == "status":
        status()
        return

    LEGACY.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another fusion job is active. Stop it before changing the cohort.") from error
        if not (SFT / "adapter_model.safetensors").is_file():
            raise FileNotFoundError(f"Missing shared SFT adapter: {SFT}")
        rows, manifest = dataset.prepare(SOURCE, RUN / "dataset")
        components = component_identity(manifest)
        if args.stage in {"prepare", "prepare-and-archive", "components"}:
            seed_old_components(rows, components)
        if args.stage in {"prepare", "prepare-and-archive"}:
            status()
            if args.stage == "prepare-and-archive":
                archive_legacy()
            return
        identity = components
        if args.stage in {"baseline", "evidence"}:
            completed_shards("components", rows, components)
            identity = get_baseline_identity(components)
        if args.stage == "evidence":
            completed_shards("baseline", rows, identity)
            published = RunCache(RUN / "rewards_baseline", identity, resume=True)
            if published.completed_keys != {row["id"] for row in rows}:
                raise ValueError("Published baseline rewards are incomplete")
            identity = get_evidence_identity(identity, args)
        launch_workers(args.stage, rows, identity, lock.fileno())
        if args.stage != "components":
            publish(args.stage, rows, identity)
        status()


if __name__ == "__main__":
    main()
