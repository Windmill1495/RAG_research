"""Stream the full corpus into OpenSearch without loading corpus vectors into RAM.

Resume skips previously committed input ordinals; idempotent document IDs make
partially successful bulk requests safe to replay. An incomplete index cannot
be used by the evaluator. A new configuration requires a new index name.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

from rag.llm import EMBEDDING_INPUT_POLICY, OllamaClient, runtime_embedding_identity, runtime_options
from rag import official_index
from rag.opensearch_retrieval import OpenSearchRetriever, SearchServiceError, document_key, vector_mapping
from rag.retrieval import _normalized
from rag.run_cache import file_sha256, make_signature


INDEX_SCHEMA = "qrecc-opensearch-hybrid-v1"
BM25_INDEX_SCHEMA = "qrecc-opensearch-bm25-v1"
OFFICIAL_ARCHIVE_MD5 = official_index.ARCHIVE_MD5


class ParallelEmbedding:
    def __init__(self, config, urls, executor):
        self.executor = executor
        self.config, self.urls = config, urls
        self.clients = [OllamaClient(url, config["model"], config["embedding_model"],
                                    timeout=config["request_timeout"],
                                    num_thread=config.get("ollama_num_thread")) for url in urls]

    def verify_digest(self, digest):
        for url in self.urls:
            identity = runtime_embedding_identity({**self.config, "embedding_base_url": url})
            if identity["digest"] != digest:
                raise ValueError(f"Embedding weights differ at {url}")

    def embed(self, texts):
        # Striding keeps similarly long neighboring passages from all landing on one worker.
        if not texts:
            return []
        width = min(len(texts), len(self.clients))
        futures = [self.executor.submit(self.clients[i].embed, texts[i::width]) for i in range(width)]
        ordered = [None] * len(texts)
        for i, future in enumerate(futures):
            ordered[i::width] = future.result()
        return ordered


def source_identity(source: Path):
    source = Path(source).resolve()
    if source.is_dir():
        files = sorted(p for p in source.rglob("*") if p.is_file())
        if not files:
            raise ValueError("Source directory is empty")
        rows = [{"name": p.relative_to(source).as_posix(), "bytes": p.stat().st_size,
                 "sha256": file_sha256(p)} for p in files]
    elif source.is_file():
        sha, md5 = hashlib.sha256(), hashlib.md5()
        with source.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                sha.update(block)
                md5.update(block)
        rows = [{"name": source.name, "bytes": source.stat().st_size,
                 "sha256": sha.hexdigest(), "md5": md5.hexdigest()}]
    else:
        raise ValueError("Source corpus does not exist")
    return {"files": rows, "source_file_count": len(rows), "sha256": make_signature(rows)}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def build_full_index(config, source, checkpoint, *, resume=False, batch_size=16,
                     expected_count=None, client=None, passage_iterator=None,
                     allow_unverified_source=False, diagnostic_variants=None):
    lexical = config.get("retriever_backend") == "bm25"
    limit = 4096 if lexical else 256
    if type(batch_size) is not int or not 1 <= batch_size <= limit:
        raise ValueError(f"batch_size must be in [1,{limit}]")
    if lexical and client is not None:
        raise ValueError("BM25 builds do not accept embedding workers")
    if expected_count is not None and (type(expected_count) is not int or expected_count < 1):
        raise ValueError("expected_count must be positive")
    default_iterator = passage_iterator is None
    if default_iterator:
        from rag.full_data import iter_full_passages
        passage_iterator = iter_full_passages
    source, checkpoint = Path(source), Path(checkpoint)
    source_stamp = source.stat()
    config = dict(config)
    verify_runtime = not lexical and client is None
    if not lexical:
        if verify_runtime or not config.get("embedding_digest"):
            installed_digest = runtime_embedding_identity(config)["digest"]
            if config.get("embedding_digest") and config["embedding_digest"] != installed_digest:
                raise ValueError("Configured embedding digest differs from installed Ollama weights")
            config["embedding_digest"] = installed_digest
        client = client or OllamaClient(config["base_url"], config["model"], config["embedding_model"],
                                        timeout=config["request_timeout"],
                                        embedding_base_url=config.get("embedding_base_url"),
                                        num_thread=config.get("ollama_num_thread"), num_gpu=config.get("ollama_num_gpu"))
        if isinstance(client, ParallelEmbedding):
            client.verify_digest(config["embedding_digest"])
    retriever = OpenSearchRetriever(config, None if lexical else client.embed)
    print("Fingerprinting full corpus source; all source bytes are checked before indexing.", flush=True)
    hf = config.get('full_corpus_source', 'official_zip') == 'hf_qrecc_parquet'
    if hf:
        from rag.hf_corpus import verified_hf_identity, HF_SCOPE
        if Path(config['corpus_path']).resolve() != source.resolve():
            raise ValueError('HF --source differs from the explicitly configured corpus_path')
        sources = verified_hf_identity(source, config['eval_path'], hash_files=True)
        if default_iterator:
            from rag.hf_duplicates import iter_index_passages
            passage_iterator = lambda path: iter_index_passages(path, config['eval_path'])
        audited_count = sources['provenance']['expected_documents']
        if expected_count is not None and expected_count != audited_count:
            raise ValueError('--expected-count differs from the audited HF valid-passage count')
        expected_count = audited_count
        official = False
        corpus_scope = HF_SCOPE
    else:
        sources = source_identity(source)
        official = (source.is_file() and source.suffix.lower() == ".zip"
                    and official_index.is_official_archive(sources))
        corpus_scope = 'official_qrecc_full' if official else 'diagnostic_source'
    if not (official or hf) and not allow_unverified_source:
        raise ValueError("Full build requires the verified official passages.zip. "
                         "Other sources require --allow-unverified-source and remain diagnostic-only")
    if not (official or hf):
        # Only this explicit builder diagnostic can validate its test index.
        # Formal evaluator configs still reject this provenance unconditionally.
        retriever.config.update(evaluation_scope="pilot", allow_diagnostic_index=True)
    if diagnostic_variants is not None and (official or hf):
        raise ValueError("Diagnostic variants are only for a separate diagnostic index")
    if diagnostic_variants is not None and (type(diagnostic_variants) is not int or diagnostic_variants < 1):
        raise ValueError("Provide the positive expected distinct count of the diagnostic source")
    variants = official or diagnostic_variants is not None
    if official:
        if expected_count not in (None, official_index.VALID_ROWS):
            raise ValueError("--expected-count must match the audited official source row count")
        expected_count = official_index.VALID_ROWS
    vector = None if lexical else vector_mapping(config)
    identity = {"index_schema": BM25_INDEX_SCHEMA if lexical else INDEX_SCHEMA, "source_identity": sources,
                "corpus_scope": corpus_scope, "corpus_fingerprint": sources["sha256"],
                "expected_count_argument": expected_count,
                "text_analyzer": "standard", "bm25_k1": 1.5, "bm25_b": 0.75}
    if not lexical:
        identity.update(dimensions=retriever.dimension, embedding_model=config["embedding_model"],
                        embedding_digest=config["embedding_digest"], embedding_input_policy=EMBEDDING_INPUT_POLICY,
                        vector_config=vector, embedding_runtime_options=runtime_options(config))
    if variants:
        identity["document_key_policy"] = official_index.VARIANT_POLICY
    if official:
        identity["audited_variant_count"] = official_index.UNIQUE_VARIANTS
    elif diagnostic_variants is not None:
        identity["audited_variant_count"] = diagnostic_variants
    if isinstance(client, ParallelEmbedding):
        identity["embedding_runtime_options"] = dict(client.clients[0].runtime_options)
        identity["index_embedding_workers"] = {"count": len(client.clients), "num_gpu": "auto"}
    cursor = 0
    build_started = time.monotonic()
    first_cursor = 0
    try:
        mapping = retriever._request("GET", retriever._path("/_mapping"))
    except SearchServiceError as exc:
        if exc.status != 404:
            raise
        mapping = None
    if mapping:
        existing = mapping.get(retriever.index, {}).get("mappings", {}).get("_meta", {}).get("qrecc", {})
        if existing.get("build_identity") != make_signature(identity):
            raise ValueError("Existing index source/model/settings differ; use a new index name")
        if existing.get("complete") is True:
            # Recover a crash between committing completion metadata and the
            # final write block. The normal evaluator never performs writes.
            if resume:
                retriever._request("PUT", retriever._path("/_settings"),
                                   {"index": {"blocks.write": True, "refresh_interval": "1s"}})
            return retriever.build()
        if not resume or not checkpoint.is_file():
            raise ValueError("Index is incomplete; use its existing checkpoint with --resume")
        saved = json.loads(checkpoint.read_text(encoding="utf-8"))
        if saved.get("identity") != identity:
            raise ValueError("Checkpoint does not match corpus/model/configuration")
        settings = retriever._request("GET", retriever._path("/_settings"))
        if saved.get("uuid") != settings[retriever.index]["settings"]["index"].get("uuid"):
            raise ValueError("Checkpoint belongs to a different index instance")
        cursor = saved["committed_rows"]
        if type(cursor) is not int or cursor < 0:
            raise ValueError("Invalid committed row count in checkpoint")
        uuid = saved["uuid"]
        first_cursor = cursor
        meta = existing
    else:
        if checkpoint.exists():
            raise ValueError("Checkpoint exists but its server index is missing")
        meta = {**identity, "build_identity": make_signature(identity), "complete": False,
                "expected_documents": 0}
        settings = {"index": {"number_of_shards": config.get("opensearch_shards", 8),
                   "number_of_replicas": config.get("opensearch_replicas", 0),
                   "refresh_interval": "-1", "similarity": {"qrecc_bm25": {"type": "BM25", "k1": 1.5, "b": 0.75}}}}
        properties = {
                "passage_id": {"type": "keyword", "ignore_above": 32766},
                "title": {"type": "keyword", "index": False},
                "text": {"type": "text", "analyzer": "standard", "similarity": "qrecc_bm25"}}
        if not lexical:
            settings["index"]["knn"] = True
            properties["vector"] = vector
        if variants:
            properties.update(passage_key={"type": "keyword"},
                              variant_key={"type": "keyword", "index": False})
        retriever._request("PUT", retriever._path(), {"settings": settings, "mappings": {
            "_meta": {"qrecc": meta}, "_source": {"excludes": ["vector"]}, "properties": properties}})
        settings = retriever._request("GET", retriever._path("/_settings"))
        uuid = settings[retriever.index]["settings"]["index"]["uuid"]
        atomic_json(checkpoint, {"identity": identity, "uuid": uuid, "committed_rows": 0, "complete": False})

    def commit_batch(rows, ordinal):
        normalized = [None] * len(rows)
        if not lexical:
            vectors = client.embed([r["text"] for r in rows])
            if not isinstance(vectors, (list, tuple)) or len(vectors) != len(rows):
                raise ValueError("Embedding batch count differs from passage count")
            normalized = [_normalized(v) for v in vectors]
            if any(len(v) != retriever.dimension for v in normalized):
                raise ValueError("Embedding dimension differs from the index mapping")
        lines = []
        for row, embedding in zip(rows, normalized):
            pid, text = row["id"], row["text"]
            if not isinstance(pid, str) or not pid or not isinstance(text, str) or not text.strip():
                raise ValueError("Source passage ID/text is invalid")
            pid_key = document_key(pid)
            document = {"passage_id": pid, "text": text, "title": row.get("title", pid[:120])}
            if not lexical:
                document["vector"] = embedding
            key = pid_key
            if variants:
                body_key = hashlib.sha256(text.encode("utf-8")).hexdigest()
                key = pid_key + body_key
                document.update(passage_key=pid_key, variant_key=body_key)
            lines.extend([json.dumps({"index": {"_id": key}}), json.dumps(document, ensure_ascii=False)])
        result = retriever._request("POST", retriever._path("/_bulk"), "\n".join(lines) + "\n", ndjson=True)
        items = result.get("items")
        if result.get("errors") or not isinstance(items, list) or len(items) != len(rows):
            raise SearchServiceError("Bulk import failed or was partial; checkpoint not advanced")
        for item in items:
            value = item.get("index", {})
            if value.get("error") or value.get("status") not in {200, 201}:
                raise SearchServiceError("Bulk item failed; checkpoint not advanced")
        atomic_json(checkpoint, {"identity": identity, "uuid": uuid,
                                "committed_rows": ordinal, "complete": False})
        if lexical:
            rate = (ordinal - first_cursor) / max(time.monotonic() - build_started, 0.01)
            progress = f"/{expected_count:,} ({ordinal / expected_count:.2%})" if expected_count else ""
            print(f"[BM25] committed {ordinal:,}{progress} | {rate:.1f} passages/s", flush=True)
        else:
            print(f"[Full index] committed passages: {ordinal}", flush=True)

    pending, ordinal, pending_bytes = [], 0, 0
    for ordinal, row in enumerate(passage_iterator(source), 1):
        if ordinal <= cursor:
            continue
        if lexical:
            row_bytes = len(row["text"].encode("utf-8")) + len(row["id"].encode("utf-8")) + 512
            if pending and pending_bytes + row_bytes > 8 * 1024 * 1024:
                commit_batch(pending, ordinal - 1)
                pending.clear()
                pending_bytes = 0
            pending_bytes += row_bytes
        pending.append(row)
        if len(pending) >= batch_size:
            commit_batch(pending, ordinal)
            pending.clear()
            pending_bytes = 0
    if ordinal < cursor:
        raise ValueError("Source ended before the saved checkpoint")
    if pending:
        commit_batch(pending, ordinal)
    if ordinal < 1:
        raise ValueError("Source corpus contains no valid passages")
    if expected_count is not None and ordinal != expected_count:
        raise ValueError("Source EOF count differs from --expected-count; index remains incomplete")
    if official:
        current = source.stat()
        if (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns, current.st_ctime_ns) != (
                source_stamp.st_dev, source_stamp.st_ino, source_stamp.st_size,
                source_stamp.st_mtime_ns, source_stamp.st_ctime_ns):
            raise ValueError("Official archive changed during the build; index remains incomplete")
    if verify_runtime and runtime_embedding_identity(config)["digest"] != config["embedding_digest"]:
        raise ValueError("Embedding weights changed while indexing; use a new index with fixed weights")
    if isinstance(client, ParallelEmbedding):
        client.verify_digest(config["embedding_digest"])
    refresh = retriever._request("POST", retriever._path("/_refresh"))
    if refresh.get("_shards", {}).get("failed", 0):
        raise SearchServiceError("Refresh failed; index remains incomplete")
    count = retriever._request("GET", retriever._path("/_count"))
    documents = count.get("count")
    expected_documents = official_index.UNIQUE_VARIANTS if official else ordinal
    if diagnostic_variants is not None:
        expected_documents = diagnostic_variants
    if (type(documents) is not int or not 0 < documents <= ordinal or documents != expected_documents
            or count.get("_shards", {}).get("failed", 0)):
        raise ValueError("Indexed unique count differs from source rows (duplicates/partial writes); index remains incomplete")
    meta.update(complete=True, expected_documents=documents)
    retriever._request("PUT", retriever._path("/_mapping"), {"_meta": {"qrecc": meta}})
    retriever._request("PUT", retriever._path("/_settings"), {"index": {"blocks.write": True, "refresh_interval": "1s"}})
    result = retriever.build()
    atomic_json(checkpoint, {"identity": identity, "uuid": uuid, "committed_rows": ordinal, "complete": True})
    return result


def main():
    from rag.config import load_experiment_config
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Official passages.zip, explicitly configured audited HF directory, or diagnostic JSONL")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path(".cache/full-index-checkpoint.json"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--expected-count", type=int, help="Exact expected valid passage count, when independently known")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--embedding-worker", action="append", default=[], metavar="URL",
                        help="Repeat to distribute each batch across independent Ollama servers")
    parser.add_argument("--diagnostic-variants", type=int, metavar="UNIQUE_COUNT",
                        help="Expected distinct ID/text count for an explicitly unverified diagnostic source")
    parser.add_argument("--allow-unverified-source", action="store_true",
                        help="Diagnostic only: accept a source other than the verified official archive")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    config = load_experiment_config(root, scope="full", config_path=args.config)
    try:
        if len(set(args.embedding_worker)) != len(args.embedding_worker):
            raise ValueError("Embedding worker endpoints must be distinct")
        with ThreadPoolExecutor(max_workers=max(1, len(args.embedding_worker))) as executor:
            client = ParallelEmbedding(config, args.embedding_worker, executor) if args.embedding_worker else None
            result = build_full_index(config, args.source, args.checkpoint, resume=args.resume,
                                      batch_size=args.batch_size, expected_count=args.expected_count,
                                      allow_unverified_source=args.allow_unverified_source,
                                      diagnostic_variants=args.diagnostic_variants, client=client)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"Full index build failed: {exc}\n")


if __name__ == "__main__":
    main()
