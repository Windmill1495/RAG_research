"""Small, dependency-free client for Ollama's documented HTTP API.

The retriever and the generator are intentionally separate. Nothing here
downloads a model, and an unavailable model never silently becomes a mock LLM.
"""

from __future__ import annotations

import json
import math
import socket
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .config import DEFAULT_MODEL

EMBEDDING_INPUT_POLICY = "ollama-context-truncate-v1"

QWEN3_REWRITE_TAGS = {
    f"qwen3:{size}{quantization}"
    for size in ("1.7b", "4b", "8b", "14b", "32b")
    for quantization in ("", "-q4_K_M")
}


class OllamaError(RuntimeError):
    """An actionable connection, model, or response-format error."""


def runtime_model_identity(config: dict, mode: str) -> dict:
    """Freeze the installed weight digest, not just a mutable Ollama tag.

    Demo runs do not require or contact Ollama. A real-model experiment fails
    early if its actual model identity cannot be verified, rather than silently
    resuming an old checkpoint after a model update.
    """
    if mode == "demo":
        return {"mode": "demo", "llm_called": False}
    if mode != "ollama":
        raise ValueError("Experiment mode must be demo or ollama")
    client = OllamaClient(config["base_url"], config["model"], config["embedding_model"], timeout=5)
    return {"mode": mode, "base_url": client.base_url, **_installed_identity(client, client.model),
            **client.thinking_option,
            "generation_options": {"temperature": 0.0, "seed": 42,
                                   "num_ctx": config.get("llm_num_ctx", 8192),
                                   "num_predict": config.get("llm_num_predict", 512)}}


def runtime_embedding_identity(config: dict) -> dict | None:
    if config["retriever_backend"] == "bm25":
        return None
    client = OllamaClient(config["base_url"], config["model"], config["embedding_model"], timeout=5)
    return {**_installed_identity(client, client.embedding_model),
            "input_policy": EMBEDDING_INPUT_POLICY}


def _installed_identity(client: OllamaClient, tag: str) -> dict:
    models = client._request("/api/tags").get("models")
    if not isinstance(models, list):
        raise OllamaError("Cannot verify the installed Ollama model identity")
    for model in models:
        if isinstance(model, dict) and model.get("name") in {tag, tag + ":latest"}:
            digest = model.get("digest")
            if not isinstance(digest, str) or not digest.strip():
                raise OllamaError("Ollama model has no weight digest; cannot safely identify this experiment")
            return {"name": model["name"], "digest": digest, "details": model.get("details", {})}
    raise OllamaError(f"Model {tag} is not installed; run ollama pull {tag}")


class OllamaClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        model: str = DEFAULT_MODEL,
        embedding_model: str = "bge-m3",
        timeout: float = 180,
        *, num_ctx: int = 8192, num_predict: int = 512,
    ) -> None:
        parsed = urlsplit(base_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.path.rstrip("/")
        ):
            raise ValueError("base_url 应为 Ollama 服务根地址，例如 http://127.0.0.1:11434")
        if not model.strip() or not embedding_model.strip():
            raise ValueError("模型名称不能为空")
        if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout 必须为正数秒数")
        self.base_url = base_url.rstrip("/")
        self.model = model
        # Keep the same non-thinking setting across the five model sizes.
        self.thinking_option = {"think": False} if model in QWEN3_REWRITE_TAGS else {}
        self.embedding_model = embedding_model
        self.timeout = float(timeout)
        if type(num_ctx) is not int or num_ctx < 1024 or type(num_predict) is not int or num_predict < 1:
            raise ValueError("num_ctx/num_predict 必须是有效的正整数")
        self.num_ctx = num_ctx
        self.num_predict = num_predict

    def _request(self, path: str, payload: dict | None = None) -> dict:
        body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(
            self.base_url + path,
            data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="GET" if payload is None else "POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")[:1200]
            try:
                detail = json.loads(raw).get("error", raw)
            except (ValueError, AttributeError):
                detail = raw
            suggestion = ""
            if exc.code == 404:
                requested_model = (payload or {}).get("model", self.model)
                suggestion = f"；请核对服务版本与模型名称，必要时运行 ollama pull {requested_model}"
            raise OllamaError(f"Ollama HTTP {exc.code}: {detail}{suggestion}") from exc
        except (URLError, TimeoutError, socket.timeout, ConnectionError, OSError) as exc:
            raise OllamaError(
                f"无法完成 Ollama 请求 ({self.base_url})：{exc}。"
                "请启动 Ollama；若模型加载较慢，可增大 config.json 中的 request_timeout。"
            ) from exc
        except (UnicodeError, ValueError) as exc:
            raise OllamaError("Ollama 返回了无效 JSON；请确认 base_url 指向 Ollama 服务") from exc
        if not isinstance(result, dict):
            raise OllamaError("Ollama 返回格式错误：顶层应为 JSON 对象")
        if result.get("error"):
            raise OllamaError(f"Ollama: {result['error']}")
        return result

    def generate(self, prompt: str, system: str = "", temperature: float = 0.0) -> str:
        """Return one complete response (streaming is explicitly disabled)."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt 不能为空")
        if not isinstance(system, str):
            raise ValueError("system 必须是字符串")
        if not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature 必须是非负有限数")
        result = self._request(
            "/api/generate",
            {
                "model": self.model,
                **self.thinking_option,
                "prompt": prompt,
                "system": system,
                "stream": False,
                "keep_alive": "5m",
                "options": {
                    "temperature": temperature,
                    "num_ctx": self.num_ctx,
                    "num_predict": self.num_predict,
                    "seed": 42,
                },
            },
        )
        answer = result.get("response")
        if not isinstance(answer, str) or not answer.strip() or result.get("done") is False:
            raise OllamaError("Ollama 未返回完整非空 response，请检查模型与服务日志")
        return answer.strip()

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Encode within Ollama's context window; keep source passages intact."""
        if not isinstance(texts, list) or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("texts 必须是非空字符串组成的列表")
        if not texts:
            return []
        result = self._request(
            "/api/embed",
            {
                "model": self.embedding_model,
                "input": texts,
                "truncate": True,
                "keep_alive": "5m",
            },
        )
        vectors = result.get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise OllamaError("Ollama embeddings 数量与输入数量不一致")
        dimension = None
        checked = []
        for vector in vectors:
            if not isinstance(vector, list) or not vector:
                raise OllamaError("Ollama 返回了空向量或无效向量")
            if dimension is None:
                dimension = len(vector)
            if len(vector) != dimension:
                raise OllamaError("Ollama embedding 维度不一致")
            if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in vector):
                raise OllamaError("Ollama embedding 包含非有限数值")
            checked.append([float(x) for x in vector])
        return checked

    def list_models(self) -> list[str]:
        result = self._request("/api/tags")
        models = result.get("models")
        if not isinstance(models, list):
            raise OllamaError("Ollama models 字段格式错误")
        names = []
        for model in models:
            if not isinstance(model, dict) or not isinstance(model.get("name"), str):
                raise OllamaError("Ollama 模型列表缺少 name 字段")
            names.append(model["name"])
        return names
