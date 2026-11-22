"""Distribute pending components while retaining one writer for both caches."""

import argparse
import json
import os
import selectors
import subprocess
import sys
from pathlib import Path

import parallel_adaqr_half as base


GPUS = base.gpu_indices("ADAQR_COMPONENT_GPUS", "0,1")


def worker(gpu, result_fd, assigned_ids):
    rows, _ = base.dataset.load(base.RUN / "dataset")
    asked = set(assigned_ids)
    selected = [row for row in rows if row["id"] in asked]
    if len(selected) != len(assigned_ids):
        raise ValueError("Assigned component IDs are incomplete or duplicated")
    generate = base.component_operation()
    with os.fdopen(result_fd, "w", encoding="utf-8", buffering=1) as stream:
        print(f"GPU {gpu} | generating {len(selected)} pending component turns", flush=True)
        for number, row in enumerate(selected, 1):
            result = generate(row)
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            if number % 25 == 0 or number == len(selected):
                print(f"GPU {gpu} | {number}/{len(selected)} newly generated", flush=True)


def checkpoint_state(rows, identity):
    caches = [base.cache_for("components", slot, identity) for slot in range(base.SHARDS)]
    slots = {row["id"]: position % base.SHARDS for position, row in enumerate(rows)}
    complete = set()
    for slot, cache in enumerate(caches):
        for key in cache.completed_keys:
            if slots.get(key) != slot:
                raise ValueError(f"Unexpected turn {key} in existing shard {slot}")
            complete.add(key)
    return caches, slots, complete


def write_result(payload, caches, slots, pending):
    record = json.loads(payload)
    key = record["id"]
    if key not in pending:
        raise ValueError(f"Unexpected or duplicate component result: {key}")
    if record.get("status") not in {"ok", "component_error"}:
        raise ValueError(f"Invalid component status for {key}")
    caches[slots[key]].append(record, key)
    pending.remove(key)


def start_worker(gpu, device, ids, lock_fd, selector):
    reader, writer = os.pipe()
    command = [sys.executable, "-u", "-B", str(Path(__file__).resolve()), "worker",
               "--gpu", str(gpu), "--fd", str(writer), "--ids", json.dumps(ids)]
    try:
        process = subprocess.Popen(command, env={**os.environ, "CUDA_VISIBLE_DEVICES": device},
                                   pass_fds=(lock_fd, writer))
    except BaseException:
        os.close(reader)
        raise
    finally:
        os.close(writer)
    selector.register(reader, selectors.EVENT_READ, gpu)
    return process


def collect(selector, processes, caches, slots, pending, total):
    buffers = {}
    while selector.get_map():
        for key, _ in selector.select(timeout=1):
            reader, gpu = key.fd, key.data
            chunk = os.read(reader, 65_536)
            if not chunk:
                if buffers.get(reader):
                    raise ValueError(f"GPU {gpu} closed the result stream mid-record")
                selector.unregister(reader)
                os.close(reader)
                continue
            parts = (buffers.get(reader, b"") + chunk).split(b"\n")
            buffers[reader] = parts.pop()
            for part in parts:
                write_result(part, caches, slots, pending)
                done = total - len(pending)
                if done % 25 == 0 or not pending:
                    print(f"components | {done}/{total} persisted across both original shards",
                          flush=True)
        for gpu, process in processes.items():
            code = process.poll()
            if code not in (None, 0):
                raise RuntimeError(f"Component worker on GPU {gpu} exited {code}; rerun to resume")


def launch():
    import fcntl

    with base.LOCK.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("The existing component job still has the fusion lock") from error
        rows, manifest = base.dataset.load(base.RUN / "dataset")
        identity = base.component_identity(manifest)
        caches, slots, complete = checkpoint_state(rows, identity)
        todo = [row["id"] for row in rows if row["id"] not in complete]
        total = len(rows)
        print(f"Resuming {len(complete)}/{total} complete | {len(todo)} pending", flush=True)
        if not todo:
            return
        devices = base.gpu_map(GPUS)
        pending = set(todo)
        processes = {}
        with selectors.DefaultSelector() as selector:
            try:
                for offset, (gpu, device) in enumerate(devices.items()):
                    assigned = todo[offset::len(devices)]
                    if assigned:
                        processes[gpu] = start_worker(gpu, device, assigned, lock.fileno(), selector)
                collect(selector, processes, caches, slots, pending, total)
                for gpu, process in processes.items():
                    if process.wait():
                        raise RuntimeError(f"Component worker on GPU {gpu} failed; rerun to resume")
                if pending:
                    raise RuntimeError(f"{len(pending)} component turns remain; rerun to resume")
            finally:
                for conn in tuple(selector.get_map().values()):
                    selector.unregister(conn.fd)
                    os.close(conn.fd)
                for process in processes.values():
                    if process.poll() is None:
                        process.terminate()
                for process in processes.values():
                    process.wait()
        base.status()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("check", "run", "worker"))
    parser.add_argument("--gpu", type=int, choices=GPUS)
    parser.add_argument("--fd", type=int)
    parser.add_argument("--ids")
    args = parser.parse_args()
    if args.stage == "check":
        try:
            print(base.gpu_map(GPUS))
        except RuntimeError as error:
            parser.exit(1, f"{error}\n")
    elif args.stage == "worker":
        worker(args.gpu, args.fd, json.loads(args.ids))
    else:
        launch()


if __name__ == "__main__":
    main()
