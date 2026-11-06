"""Start the installed local Ollama service if it is not already responding."""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from rag.config import load_config


def main():
    config = load_config(ROOT)
    base = config["base_url"].rstrip("/")
    def ready():
        try:
            with urlopen(base + "/api/version", timeout=2) as response:
                return response.status == 200
        except OSError:
            return False
    if ready():
        print("Ollama service is ready.")
        return 0
    address = urlsplit(base)
    if address.hostname not in {"127.0.0.1", "localhost", "::1"}:
        print("Remote Ollama service is unavailable; local startup skipped.")
        return 1
    executable = shutil.which("ollama")
    if not executable and os.name == "nt":
        candidate = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
        if candidate.is_file():
            executable = str(candidate)
    if not executable:
        print("Ollama is not installed. See README.md.")
        return 1
    environment = dict(os.environ)
    environment["OLLAMA_HOST"] = address.netloc
    if config.get("ollama_llm_library"):
        environment["OLLAMA_LLM_LIBRARY"] = config["ollama_llm_library"]
    if environment.get("OLLAMA_LLM_LIBRARY") == "vulkan":
        environment["OLLAMA_FLASH_ATTENTION"] = "0"
    cache = ROOT / ".cache"
    cache.mkdir(exist_ok=True)
    with (cache / "ollama-startup.log").open("ab") as log:
        subprocess.Popen(
            [executable, "serve"], env=environment, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, close_fds=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            start_new_session=os.name != "nt",
        )
    for _ in range(30):
        if ready():
            print("Ollama service started.")
            return 0
        time.sleep(1)
    print("Ollama startup timed out. See .cache/ollama-startup.log.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
