"""Half-size AdaQR + CompCQR experiment with configurable physical CUDA devices."""

import argparse
from collections import Counter
import json
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
from rag.run_cache import RunCache, file_sha256
from train_fusion import SETTINGS


def input_path(name, default):
    return Path(os.environ.get(name, str(default))).expanduser().resolve()


def gpu_indices(name, default, count=None):
    indices = tuple(int(value) for value in os.environ.get(name, default).split(","))
    if (not indices or len(set(indices)) != len(indices) or min(indices) < 0
            or (count is not None and len(indices) != count)):
        raise ValueError(f"{name} requires distinct physical GPU indices")
    return indices


RUN = ROOT / "outputs/train_half_v1"
SOURCE = input_path("ADAQR_INPUT_SAMPLES", ROOT / "inputs/samples.jsonl")
LOCK = RUN / ".fusion.lock"
SFT = input_path("ADAQR_SFT_ADAPTER", ROOT / "inputs/sft")
INDEX = input_path("ADAQR_LUCENE_INDEX", ROOT / "inputs/lucene")
QWEN3 = input_path("ADAQR_QWEN3_DIR", ROOT / "models/Qwen3-8B")
LLAMA = input_path("ADAQR_LLAMA_DIR", ROOT / "models/Llama-3.1-8B-Instruct")
VERIFIER = input_path("ADAQR_VERIFIER_DIR", ROOT / "models/DeBERTa-v3-base-mnli-fever-anli")
EVIDENCE_LAMBDA = float(os.environ.get("ADAQR_EVIDENCE_LAMBDA", "1.0"))
EVIDENCE_EPSILON = 1e-4
GPUS = gpu_indices("ADAQR_SCORE_GPUS", "0,1", count=2)
SHARDS = 2
SEED = 42
MAX_LENGTH = 2048


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
        PROJECT / "prompts/compcqr/chat_promptor.py",
        PROJECT / "prompts/compcqr/llm4cs_demonstrations.json",
        PROJECT / "rag/rewriters.py",
    ]


def file_hashes(paths):
    return {str(path): file_sha256(path) for path in paths}


def component_identity(manifest):
    from rag.compcqr_prompts import validate_training_assets

    validate_training_assets()
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
    return {**components, "arm": "baseline", "answer_model": model_identity(QWEN3),
            "evidence_verifier": model_identity(VERIFIER),
            "evidence_lambda": EVIDENCE_LAMBDA, "evidence_epsilon": EVIDENCE_EPSILON,
            "max_length": MAX_LENGTH, "top_k": SETTINGS.top_k,
            "index": {path.name: stats(path) for path in sorted(INDEX.iterdir())
                      if path.is_file() and path.name != "write.lock"},
            "reward_code": file_hashes([ROOT / "adaqr_half_rewards.py",
                                        ROOT / "evidence_verifier.py",
                                        ROOT / "train_fusion.py",
                                        ROOT / "adaqr_support/adaqr.py",
                                        ROOT / "adaqr_support/adaqr_model.py"])}


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
    from evidence_verifier import EvidenceVerifier

    retriever = PaperBM25(INDEX)
    torch, model_class, tokenizer_class = require_ml()
    tokenizer = tokenizer_class.from_pretrained(str(QWEN3), local_files_only=True)
    model = model_class.from_pretrained(str(QWEN3), local_files_only=True,
                                        torch_dtype=torch.bfloat16, device_map={"": 0}).eval()
    verifier = EvidenceVerifier(VERIFIER)

    def logp(prompt, answer):
        with torch.inference_mode():
            return float(completion_log_probability(model, tokenizer, prompt, answer,
                                                    MAX_LENGTH, include_eos=False).item())

    def score(row):
        record = components[row["id"]]
        if record["status"] != "ok":
            return record
        result = rewards.score_turn(row, retriever=retriever, components=record["components"],
                                    answer_logp=logp, evidence_scores=verifier.score_many,
                                    evidence_lambda=EVIDENCE_LAMBDA,
                                    evidence_epsilon=EVIDENCE_EPSILON, config=RRF_CONFIG)
        if "format_recoveries" in record:
            result["format_recoveries"] = record["format_recoveries"]
        if "component_origin" in record:
            result["component_origin"] = record["component_origin"]
        return result

    return score


def worker(phase, slot, identity):
    rows, _ = dataset.load(RUN / "dataset")
    rows = rows[slot::SHARDS]
    cache = cache_for(phase, slot, identity)
    if phase == "components":
        operation = component_operation()
    else:
        values = {row["id"]: row for row in read_jsonl(
            cache_path("components", slot) / "checkpoint.jsonl")}
        operation = baseline_operation(values)
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
    for phase in ("components", "baseline"):
        counts = Counter()
        for path in (RUN / "work" / phase).glob("shard-*/checkpoint.jsonl"):
            counts.update(row["status"] for row in saved_records(path))
        print(f"{phase} | {sum(counts.values())}/{manifest['eligible_count']} | {dict(counts)}")


def main():
    import fcntl

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "components", "baseline", "status", "worker"))
    parser.add_argument("--worker-phase", choices=("components", "baseline"), help=argparse.SUPPRESS)
    parser.add_argument("--slot", type=int, choices=range(SHARDS), help=argparse.SUPPRESS)
    parser.add_argument("--identity", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.stage == "worker":
        worker(args.worker_phase, args.slot, json.loads(args.identity))
        return
    if args.stage == "status":
        status()
        return

    RUN.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another fusion process is writing this run") from error
        rows, manifest = dataset.prepare(SOURCE, RUN / "dataset")
        if args.stage == "prepare":
            status()
            return
        identity = component_identity(manifest)
        if args.stage == "baseline":
            completed_shards("components", rows, identity)
            identity = get_baseline_identity(identity)
        launch_workers(args.stage, rows, identity, lock.fileno())
        if args.stage == "baseline":
            publish(args.stage, rows, identity)
        status()


if __name__ == "__main__":
    main()
