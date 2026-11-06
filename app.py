"""Local web application using only Python's standard library."""
import argparse
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from evaluate_rewrite import evaluate_rewrites
from rag.config import load_config, offline_config
from rag.llm import OllamaClient, OllamaError, runtime_model_identity
from rag.pipeline import RAGEngine, clean_history
from rag.rewrite_data import load_rewrite_examples
from rag.rewriters import AB_STRATEGIES

ROOT = Path(__file__).resolve().parent


def create_server(port: int, requested_mode: str = "auto", dataset: str = "qrecc", *,
                  rewrite_model: str | None = None, initial_page: str = "rag") -> ThreadingHTTPServer:
    config = load_config(ROOT, profile=dataset)
    probe = OllamaClient(config["base_url"], config["model"], config["embedding_model"], timeout=2)
    models, connection_error = [], ""
    try:
        models = probe.list_models()
    except Exception as error:
        connection_error = str(error)
    def missing_models(settings, installed):
        required = [settings["model"]]
        if settings["retriever_backend"] != "bm25":
            required.append(settings["embedding_model"])
        return [tag for tag in required if tag not in installed and tag + ":latest" not in installed]

    missing = missing_models(config, models)
    available = not missing
    if missing and not connection_error:
        connection_error = "缺少模型：" + "、".join(missing)
    default_mode = ("ollama" if available else "demo") if requested_mode == "auto" else requested_mode
    engines, operation_lock = {}, threading.RLock()
    initialization, initialization_lock = {}, threading.Lock()
    rewrite_engine, rewrite_lock = None, threading.Lock()

    def get_engine(mode):
        with operation_lock:
            if mode not in {"demo", "ollama"}:
                raise ValueError("mode 必须为 demo 或 ollama")
            if mode not in engines:
                settings = offline_config(config) if mode == "demo" else config
                with initialization_lock:
                    initialization[mode] = {"state": "building", "error": None}
                try:
                    engines[mode] = RAGEngine(ROOT, settings, mode)
                except Exception as error:
                    with initialization_lock:
                        initialization[mode] = {"state": "error", "error": str(error)}
                    raise
                with initialization_lock:
                    initialization[mode] = {"state": "ready", "error": None}
                print(f"[Index] {mode} index ready.", flush=True)
            return engines[mode]

    def prepare_engine(mode):
        # Binding the HTTP port must not wait for the first embedding index.
        with initialization_lock:
            if mode in engines or initialization.get(mode, {}).get("state") in {"building", "error"}:
                return
            initialization[mode] = {"state": "building", "error": None}

        def prepare():
            try:
                get_engine(mode)
            except Exception as error:
                print(f"[Index] Initialization failed: {error}", flush=True)

        threading.Thread(target=prepare, daemon=True).start()

    def get_rewrite_engine():
        # Rewriting uses its own model settings. The shared index is loaded on demand.
        nonlocal rewrite_engine
        with rewrite_lock:
            if rewrite_engine is None:
                rewrite_config = load_config(ROOT, {"model": rewrite_model}, profile="qrecc")
                mode = requested_mode
                if mode == "auto":
                    rewrite_models = models
                    if rewrite_config["base_url"] != config["base_url"]:
                        rewrite_probe = OllamaClient(rewrite_config["base_url"], rewrite_config["model"],
                                                     rewrite_config["embedding_model"], timeout=2)
                        try:
                            rewrite_models = rewrite_probe.list_models()
                        except OllamaError:
                            rewrite_models = []
                    mode = "demo" if missing_models(rewrite_config, rewrite_models) else "ollama"
                if mode == "demo":
                    rewrite_config = offline_config(rewrite_config)
                rewrite_engine = RAGEngine(ROOT, rewrite_config, mode, build_index=False)
            return rewrite_engine

    def rewrite_examples(engine):
        return load_rewrite_examples(ROOT / engine.config["eval_path"], "all")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            # Request bodies/questions are never logged to the terminal.
            print(f"[HTTP] {self.command} {self.path.split('?')[0]} {args[1] if len(args)>1 else ''}")

        def send_bytes(self, status, data, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'")
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def json_response(self, data, status=200):
            self.send_bytes(status, json.dumps(data, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def valid_origin(self):
            host = self.headers.get("Host", "")
            allowed = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            origin = self.headers.get("Origin")
            return host in allowed and (not origin or origin in {f"http://{value}" for value in allowed})

        def do_GET(self):
            if not self.valid_origin():
                self.json_response({"error": "仅接受本机同源访问"}, 403)
                return
            path = urlsplit(self.path).path
            if path == "/api/status":
                try:
                    prepare_engine(default_mode)
                    with initialization_lock:
                        state = dict(initialization[default_mode])
                    engine = engines.get(default_mode) if state["state"] == "ready" else None
                    settings = offline_config(config) if default_mode == "demo" else config
                    self.json_response({
                        "mode": default_mode, "model": config["model"], "models": models,
                        "dataset": config["dataset"],
                        "ollama_available": available, "connection_error": connection_error,
                        "config": {key: settings[key] for key in ("dataset", "corpus_format", "bm25_mode", "retriever_backend", "chunk_size", "overlap", "top_k", "context_char_budget", "history_turns")},
                        "initialization": state,
                        "index": engine.retriever.stats() if engine else {"built": False},
                    })
                except (OSError, RuntimeError, ValueError) as error:
                    self.json_response({"error": str(error)}, 503)
                return
            if path == "/api/rewrite/samples":
                try:
                    engine = get_rewrite_engine()
                    samples = [
                        {"id": row["id"], "source": row["source"], "turn": row["turn"], "split": row["split"],
                         "question": row["question"],
                         "history": clean_history(row["history"], engine.config["history_turns"])}
                        for row in rewrite_examples(engine)
                    ]
                    self.json_response({"samples": samples, "mode": engine.mode,
                                        "model": engine.config["model"],
                                        "retriever_backend": engine.config["retriever_backend"] if engine.mode == "ollama" else None,
                                        "history_turns": engine.config["history_turns"]})
                except (OSError, ValueError) as error:
                    self.json_response({"error": str(error)}, 503)
                return
            assets = {
                "/": ("index.html", "text/html"), "/app.js": ("app.js", "text/javascript"),
                "/style.css": ("style.css", "text/css"), "/rewrite": ("rewrite.html", "text/html"),
                "/rewrite.js": ("rewrite.js", "text/javascript"),
            }
            if path not in assets:
                self.json_response({"error": "页面不存在"}, 404)
                return
            filename, content_type = assets[path]
            self.send_bytes(200, (ROOT / "web" / filename).read_bytes(), content_type + "; charset=utf-8")

        def do_POST(self):
            if not self.valid_origin():
                self.json_response({"error": "仅接受本机同源请求"}, 403)
                return
            if self.path not in {"/api/answer", "/api/rebuild", "/api/rewrite/compare"}:
                self.json_response({"error": "接口不存在"}, 404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 100_000:
                    self.json_response({"error": "请求体需在1到100000字节之间"}, 413)
                    return
                if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                    self.json_response({"error": "需要application/json请求"}, 415)
                    return
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(body, dict):
                    raise ValueError("请求体必须为JSON对象")
                if self.path == "/api/rewrite/compare":
                    self.compare_rewrite(body)
                    return
                with operation_lock:
                    if self.path == "/api/rebuild":
                        engine = get_engine(default_mode)
                        result = engine.retriever.build(force=True)
                        engines.clear()
                        self.json_response({"message": "索引已重建", "index": result})
                    else:
                        mode = body.get("mode", default_mode)
                        engine = get_engine(mode)
                        result = engine.answer(body.get("question", ""), body.get("history", []), body.get("strategy", "rewrite"), body.get("top_k"))
                        self.json_response(result)
            except (ValueError, TypeError, UnicodeError) as error:
                self.json_response({"error": str(error)}, 400)
            except Exception as error:
                self.json_response({"error": str(error)}, 503)

        def compare_rewrite(self, body):
            sample_id = body.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError("sample_id 必须为非空字符串")
            engine = get_rewrite_engine()
            sample = next((row for row in rewrite_examples(engine) if row["id"] == sample_id), None)
            if sample is None:
                self.json_response({"error": "未找到指定的 QReCC 题目"}, 404)
                return
            with rewrite_lock, operation_lock:
                identity = runtime_model_identity(engine.config, engine.mode)
                retrieval = {}
                if engine.mode == "ollama" and sample.get("positive_ids"):
                    retrieval = {"retrieval_ids": {sample_id}, "retriever": get_engine("ollama").retriever}
                _, rows = evaluate_rewrites(engine, [sample], AB_STRATEGIES, progress=False, **retrieval)
            self.json_response({
                "sample_id": sample_id, "mode": engine.mode,
                "model": identity.get("name", engine.config["model"]),
                "model_digest": identity.get("digest"),
                "retriever_backend": engine.config["retriever_backend"] if retrieval else None,
                "reference_rewrite": sample["human_rewrite"], "results": rows,
            })

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    if initial_page == "rag":
        prepare_engine(default_mode)
    return server


def main():
    parser = argparse.ArgumentParser(description="本地多轮RAG演示页面")
    parser.add_argument("--mode", choices=["auto", "demo", "ollama"], default="auto")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--dataset", choices=["qrecc"], default="qrecc")
    parser.add_argument("--page", choices=["rag", "rewrite"], default="rag", help="浏览器打开的初始页面")
    parser.add_argument("--rewrite-model", help="覆盖重写 A/B 页使用的 Ollama 模型")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    try:
        server = create_server(args.port, args.mode, args.dataset, rewrite_model=args.rewrite_model,
                               initial_page=args.page)
    except Exception as error:
        raise SystemExit(f"启动失败：{error}\n若端口被占用，可使用 --port 8001；dense/hybrid需要Ollama嵌入模型。")
    url = f"http://127.0.0.1:{server.server_port}"
    if args.page == "rewrite":
        url += "/rewrite"
    print(f"RAG Research 已启动：{url}；数据集={args.dataset}\n按 Ctrl+C 停止。auto模式在检测不到配置模型时使用离线演示。")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已停止。")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
