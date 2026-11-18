"""Load a small JSON configuration without third-party dependencies."""
import json
import math
import os
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
    "retrieval_store": "local",
    "evaluation_scope": "pilot",
    "full_corpus_source": "official_zip",
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
    local_path = project_dir / "config.local.json"
    if local_path.is_file():
        local = json.loads(local_path.read_text(encoding="utf-8-sig"))
        if not isinstance(local, dict):
            raise ValueError("config.local.json 必须是JSON对象")
        config.update(local)
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
    if config["corpus_format"] not in {"markdown", "passages_jsonl", "passages_parquet"}:
        raise ValueError("corpus_format 必须为 markdown、passages_jsonl 或 passages_parquet")
    if config.get('full_corpus_source', 'official_zip') not in {'official_zip', 'hf_qrecc_parquet'}:
        raise ValueError('Unsupported full_corpus_source')
    if config['corpus_format'] == 'passages_parquet' and config.get('retrieval_store') != 'opensearch':
        raise ValueError('Parquet corpus requires the streaming OpenSearch builder')
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
    if config.get("retrieval_store", "local") not in {"local", "opensearch"}:
        raise ValueError("retrieval_store 必须为 local 或 opensearch")
    for name, minimum in (('ollama_num_thread', 1), ('ollama_num_gpu', 0)):
        value = config.get(name)
        if value is not None and (type(value) is not int or not minimum <= value <= 1024):
            raise ValueError(f'{name} must be an integer in [{minimum},1024] or None')
    return config


def load_experiment_config(project_dir: Path, overrides: dict | None = None, *,
                           scope: str = "full", config_path: Path | str | None = None) -> dict:
    """Full experiments are explicit; local preview remains a separate pilot scope.

    Full profile -> machine-local config -> environment -> CLI. No machine-specific
    absolute path is committed. Supplying an absolute data root is optional.
    """
    root = Path(project_dir).resolve()
    if scope not in {"full", "pilot"}:
        raise ValueError("scope 必须为 full 或 pilot")
    values = {}
    if scope == "full":
        path = root / "configs/qrecc_full.json"
        if not path.is_file():
            raise ValueError("缺少 configs/qrecc_full.json，请完整复制项目")
        values.update(json.loads(path.read_text(encoding="utf-8-sig")))
    local = Path(config_path) if config_path else root / "config.local.json"
    if not local.is_absolute():
        local = root / local
    if config_path and not local.is_file():
        raise ValueError(f"机器配置文件不存在：{local}")
    if local.is_file():
        settings = json.loads(local.read_text(encoding="utf-8-sig"))
        if not isinstance(settings, dict):
            raise ValueError("机器配置必须是JSON对象")
        values.update(settings)
    for env, key in (("OLLAMA_BASE_URL", "base_url"), ("OLLAMA_EMBEDDING_URL", "embedding_base_url"), ("OPENSEARCH_URL", "opensearch_url"),
                     ("QRECC_MODEL", "model"), ("QRECC_EMBEDDING_MODEL", "embedding_model")):
        if os.environ.get(env):
            values[key] = os.environ[env]
    if scope == "full" and os.environ.get("QRECC_DATA_DIR"):
        data = Path(os.path.expandvars(os.path.expanduser(os.environ["QRECC_DATA_DIR"])))
        selected_source = (overrides or {}).get('full_corpus_source', values.get('full_corpus_source', 'official_zip'))
        corpus = data / 'hf_passages' if selected_source == 'hf_qrecc_parquet' else data / 'passages.zip'
        values.update(corpus_path=str(corpus), eval_path=str(data / "questions.jsonl"),
                      dataset_manifest_path=str(data / "manifest.json"))
    if os.environ.get("QRECC_CACHE_DIR"):
        values["index_path"] = str(Path(os.environ["QRECC_CACHE_DIR"]) / "index.json")
    values.update({k: v for k, v in (overrides or {}).items() if v is not None})
    values["evaluation_scope"] = scope
    result = load_config(root, values)
    if scope == "full" and (result["retrieval_store"] != "opensearch" or result["retriever_backend"] != "bm25"):
        raise ValueError("Full experiments require a verified OpenSearch BM25 index; use --scope pilot for diagnostics")
    return result


def validate_experiment_data(config: dict, data_path: Path | str) -> dict | None:
    """Require a verified full-dialogue preparation manifest before formal runs."""
    if config.get("evaluation_scope") != "full":
        return None
    from .run_cache import file_sha256
    root = Path(__file__).resolve().parents[1]
    manifest_path = Path(config.get("dataset_manifest_path", "data/qrecc_full/manifest.json"))
    manifest_path = manifest_path if manifest_path.is_absolute() else root / manifest_path
    if not manifest_path.is_file():
        raise ValueError("完整QReCC尚未准备。先运行 python prepare_qrecc_full.py --download-dialogs；详见 docs/服务器迁移.md")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset_scope") != "full_qrecc" or not manifest.get("complete_dialogues") or not manifest.get("official_sources_verified"):
        raise ValueError("当前数据不是完整QReCC对话准备结果，不能标为full实验")
    if file_sha256(data_path) != manifest.get("questions_sha256"):
        raise ValueError("题目文件与完整数据manifest不一致；不能用307题冒充完整实验")
    return manifest
