"""Load a small JSON configuration without third-party dependencies."""
import json
import math
from pathlib import Path

DEFAULT_MODEL = "qwen3:1.7b"

DEFAULTS = {
    "base_url": "http://127.0.0.1:11434",
    "model": DEFAULT_MODEL,
    "embedding_model": "bge-m3",
    "retriever_backend": "bm25",
    "chunk_size": 500,
    "overlap": 80,
    "top_k": 3,
    "context_char_budget": 4500,
    "request_timeout": 180,
    "history_turns": 6,
    "dataset": "qrecc",
    "corpus_format": "passages_jsonl",
    "corpus_path": "data/qrecc/passages.jsonl",
    "index_path": ".cache/qrecc/index.json",
    "eval_path": "data/qrecc/questions.jsonl",
    "answer_language": "en",
    "bm25_mode": "inverted",
    "hybrid_candidate_depth": 400,
    "compcqr_max_components": 5,
    "compcqr_retrieval_depth": 100,
    "compcqr_rrf_k": 0,
    "compcqr_selection_top_k": 5,
    "compcqr_passage_chars": 700,
    "compcqr_selection_char_budget": 6000,
    "compcqr_select": True,
    "llm_num_ctx": 8192,
    "llm_num_predict": 512,
}


def add_compcqr_arguments(parser) -> None:
    """Shared C settings; omitted flags preserve the JSON profile."""
    for flag in ("max-components", "retrieval-depth", "selection-top-k"):
        parser.add_argument("--compcqr-" + flag, type=int, default=None)
    parser.add_argument("--compcqr-rrf-k", type=float, default=None)
    parser.add_argument("--compcqr-no-selection", action="store_true",
                        help="Ablation: keep the fused ranking without the third LLM call")


def compcqr_overrides(args) -> dict:
    values = {key: getattr(args, key, None) for key in (
        "compcqr_max_components", "compcqr_retrieval_depth",
        "compcqr_selection_top_k", "compcqr_rrf_k")}
    if getattr(args, "compcqr_no_selection", False):
        values["compcqr_select"] = False
    return {key: value for key, value in values.items() if value is not None}


def offline_config(config: dict) -> dict:
    """Keep offline BM25 in a separate cache from model-backed retrieval."""
    index = Path(config["index_path"])
    if not index.stem.endswith("-bm25"):
        index = index.with_stem(f"{index.stem}-bm25")
    return {**config, "retriever_backend": "bm25",
            "index_path": str(index)}


def load_config(project_dir: Path, overrides: dict | None = None, profile: str | None = None) -> dict:
    """Common settings -> QReCC profile -> explicit overrides.

    Passing a complete config back into RAGEngine preserves all CLI overrides.
    """
    project_dir = Path(project_dir)
    config = dict(DEFAULTS)
    path = Path(project_dir) / "config.json"
    if path.exists():
        with path.open(encoding="utf-8-sig") as stream:
            values = json.load(stream)
        if not isinstance(values, dict):
            raise ValueError("config.json 必须是 JSON 对象")
        config.update(values)
    selected = profile or (overrides or {}).get("dataset", config["dataset"])
    if profile is not None and (overrides or {}).get("dataset", profile) != profile:
        raise ValueError("profile 与 overrides.dataset 冲突，请使用一致的数据集配置")
    if selected != "qrecc":
        raise ValueError("dataset/profile 必须为 qrecc")
    profile_path = project_dir / "configs" / "qrecc.json"
    if profile_path.is_file():
        values = json.loads(profile_path.read_text(encoding="utf-8-sig"))
        if not isinstance(values, dict):
            raise ValueError("数据集配置必须为JSON对象")
        config.update(values)
    config["dataset"] = selected
    if overrides:
        config.update({key: value for key, value in overrides.items() if value is not None})
    if "backend" in config:
        config["retriever_backend"] = config.pop("backend")
    for name in ("chunk_size", "top_k", "context_char_budget", "history_turns", "request_timeout"):
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError(f"{name} 必须是正整数")
    if type(config["overlap"]) is not int or not 0 <= config["overlap"] < config["chunk_size"]:
        raise ValueError("overlap 必须满足 0 <= overlap < chunk_size")
    if config["top_k"] > 20:
        raise ValueError("top_k 最多为20")
    if config["retriever_backend"] not in {"bm25", "dense", "hybrid"}:
        raise ValueError("retriever_backend 必须为 bm25、dense 或 hybrid")
    for name in ("base_url", "model", "embedding_model"):
        if not isinstance(config[name], str) or not config[name].strip():
            raise ValueError(f"{name} 必须是非空字符串")
    for name in ("corpus_path", "index_path", "eval_path"):
        if not isinstance(config[name], str) or not config[name].strip():
            raise ValueError(f"{name} 必须是非空路径")
    if config["corpus_format"] not in {"markdown", "passages_jsonl"}:
        raise ValueError("corpus_format 必须为 markdown 或 passages_jsonl")
    if config["bm25_mode"] not in {"scan", "inverted"}:
        raise ValueError("bm25_mode 必须为 scan 或 inverted")
    if config["answer_language"] not in {"auto", "zh", "en"}:
        raise ValueError("answer_language 必须为 auto、zh 或 en")
    limits = {"compcqr_max_components": (1, 5), "compcqr_retrieval_depth": (1, 1000),
              "hybrid_candidate_depth": (1, 10000),
              "compcqr_selection_top_k": (1, 50), "compcqr_passage_chars": (50, 4000),
              "compcqr_selection_char_budget": (200, 100000),
              "llm_num_ctx": (1024, 131072), "llm_num_predict": (128, 16384)}
    for name, (low, high) in limits.items():
        if type(config[name]) is not int or not low <= config[name] <= high:
            raise ValueError(f"{name} 必须是 {low} 到 {high} 的整数")
    if (isinstance(config["compcqr_rrf_k"], bool)
            or not isinstance(config["compcqr_rrf_k"], (int, float))
            or not math.isfinite(config["compcqr_rrf_k"]) or config["compcqr_rrf_k"] < 0):
        raise ValueError("compcqr_rrf_k 必须是非负有限数")
    if type(config["compcqr_select"]) is not bool:
        raise ValueError("compcqr_select 必须是布尔值")
    return config
