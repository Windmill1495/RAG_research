"""Run the official BM25 main experiment with a dedicated multi-GPU Ollama server."""

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config.server16.gpu.local.json"
OLLAMA = ROOT / ".tools" / "ollama-0.9.6" / "bin" / "ollama"
OUTPUT_ROOT = ROOT / "outputs" / "bm25_gpu4_experiments"
SERVICE_LOG = OUTPUT_ROOT / "ollama.log"
MODEL = "qwen3:8b"
BASE_URL = "http://127.0.0.1:11438"


def ollama_request(path, payload=None, timeout=5, *, base_url=BASE_URL):
    data = json.dumps(payload).encode() if payload is not None else None
    request = Request(base_url + path, data=data, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def gpu_status(index):
    query = subprocess.check_output(
        ["nvidia-smi", "-i", str(index), "--query-gpu=uuid,name,memory.free",
         "--format=csv,noheader,nounits"],
        text=True,
    )
    uuid, name, memory = [field.strip() for field in query.strip().split(",")]
    processes = subprocess.check_output(
        ["nvidia-smi", "-i", uuid, "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    if processes:
        raise RuntimeError(f"GPU {index} has active compute processes ({processes}); select an available GPU.")
    return uuid, name, memory


def start_model(server, config, *, base_url=BASE_URL, service_log=SERVICE_LOG, expected_digest=None):
    deadline = time.monotonic() + 30
    while True:
        if server.poll() is not None:
            raise RuntimeError(f"Ollama exited with status {server.returncode}. See {service_log}")
        try:
            tags = ollama_request("/api/tags", base_url=base_url)
            break
        except URLError:
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Ollama did not start. See {service_log}")
            time.sleep(0.5)
    model = next((entry for entry in tags["models"] if entry["name"] == MODEL), None)
    if model is None or (expected_digest is not None and model.get("digest") != expected_digest):
        raise RuntimeError(f"{MODEL} is missing or differs from the signed model digest at {base_url}.")
    ollama_request("/api/generate", {
        "model": MODEL,
        "prompt": "",
        "stream": False,
        "keep_alive": "10m",
        "options": {
            "num_ctx": config["llm_num_ctx"],
            "num_thread": config["ollama_num_thread"],
        },
    }, timeout=180, base_url=base_url)
    loaded = next((entry for entry in ollama_request("/api/ps", base_url=base_url)["models"] if entry["name"] == MODEL), {})
    vram = loaded.get("size_vram", 0)
    size = loaded.get("size", 0)
    if not vram or vram < size:
        raise RuntimeError(f"{MODEL} did not fully load onto the selected GPU ({vram}/{size} bytes). See {service_log}")
    print(f"{MODEL} at {base_url} loaded on GPU using {vram / 1024**2:,.0f} MiB", flush=True)


def stop_server(server):
    for sig, timeout in ((signal.SIGTERM, 10), (signal.SIGKILL, 5)):
        try:
            os.killpg(server.pid, sig)
        except ProcessLookupError:
            break
        try:
            server.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            continue
        break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, nargs="+", required=True, help="Physical nvidia-smi GPU indices")
    parser.add_argument("--split", choices=["test", "all"], default="test")
    parser.add_argument("--resume", action="store_true", help="Resume an existing GPU run of the same split")
    parser.add_argument("--dry-run", action="store_true", help="Verify the experiment plan without starting Ollama")
    args = parser.parse_args()
    if min(args.gpus) < 0 or len(set(args.gpus)) != len(args.gpus):
        parser.error("--gpus must contain distinct non-negative GPU indices")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    output_root = OUTPUT_ROOT if args.split == "test" else OUTPUT_ROOT / "all"
    command = [sys.executable, "-u", "-B", str(ROOT / "run_experiments.py"),
               "--config", str(CONFIG), "--experiment", "main", "--split", args.split,
               "--main-model", MODEL, "--output-root", str(output_root)]
    if args.dry_run:
        return subprocess.call([*command, "--dry-run"], cwd=ROOT)
    if args.resume:
        command.append("--resume")
    elif (output_root / "main.launcher.json").exists():
        parser.error("An earlier GPU run exists. Use --resume with the original split.")
    if not OLLAMA.is_file():
        raise RuntimeError(f"Ollama executable is missing: {OLLAMA}")
    devices = [gpu_status(index) for index in args.gpus]
    with socket.socket() as port:
        port.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        port.bind(("127.0.0.1", 11438))
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    temp = OUTPUT_ROOT / "tmp"
    temp.mkdir(exist_ok=True)
    no_proxy = ",".join(filter(None, (os.environ.get("NO_PROXY"), os.environ.get("no_proxy"), "127.0.0.1")))
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": ",".join(uuid for uuid, _, _ in devices),
        "OLLAMA_SCHED_SPREAD": "true",
        "OLLAMA_HOST": BASE_URL,
        "OLLAMA_BASE_URL": BASE_URL,
        "OPENSEARCH_URL": config["opensearch_url"],
        "OLLAMA_MODELS": str(ROOT.parent / "ollama_models"),
        "OLLAMA_NOPRUNE": "true",
        "OLLAMA_NUM_PARALLEL": "1",
        "OLLAMA_MAX_LOADED_MODELS": "1",
        "OLLAMA_TMPDIR": str(temp),
        "TMPDIR": str(temp),
        "NO_PROXY": no_proxy,
        "no_proxy": no_proxy,
        "QRECC_DATA_DIR": str(ROOT.parent / "qrecc_full"),
    })
    for index, (_, name, memory) in zip(args.gpus, devices):
        print(f"GPU {index}: {name}, {memory} MiB free", flush=True)
    print(f"Experiment outputs: {output_root / 'main'}", flush=True)
    with SERVICE_LOG.open("ab") as log:
        server = subprocess.Popen(
            [str(OLLAMA), "serve"], cwd=ROOT, env=env, start_new_session=True,
            stdout=log, stderr=subprocess.STDOUT,
        )
        try:
            start_model(server, config)
            return subprocess.call(command, cwd=ROOT, env=env)
        finally:
            stop_server(server)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
