"""Bounded-memory BM25 + ANN hybrid retrieval against a complete OpenSearch index.

OpenSearch 2.19+ APIs: knn_vector, _msearch, _mget, mappings _meta.
No index is built or silently substituted during evaluation.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen

from .llm import EMBEDDING_INPUT_POLICY, runtime_options
from .official_index import VARIANT_POLICY
from .retrieval import _normalized


class SearchServiceError(RuntimeError):
    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


def document_key(passage_id: str) -> str:
    return hashlib.sha256(passage_id.encode("utf-8")).hexdigest()


def vector_mapping(config):
    value = {"type": "knn_vector", "dimension": config.get("opensearch_dimension", 1024),
             "method": {"name": "hnsw", "engine": "faiss", "space_type": "cosinesimil",
                        "parameters": {"ef_construction": 128, "m": 16}}}
    mode = config.get("opensearch_vector_mode", "on_disk")
    if mode not in {"on_disk", "in_memory"}:
        raise ValueError("Invalid vector workload mode")
    value["mode"] = mode
    if mode == "on_disk":
        compression = config.get("opensearch_compression_level", "16x")
        if compression not in {"1x", "2x", "8x", "16x", "32x"}:
            raise ValueError("Invalid Faiss vector compression level (4x requires Lucene)")
        value["compression_level"] = compression
    return value


def mapping_contains(actual, expected):
    """Server mappings may serialize additional resolved default parameters."""
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(mapping_contains(actual.get(k), v) for k, v in expected.items())
    return actual == expected


class OpenSearchRetriever:
    def __init__(self, config: dict, embedding_fn):
        self.config = dict(config)
        self.url = config.get("opensearch_url", "http://127.0.0.1:9200").rstrip("/")
        parsed = urlsplit(self.url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("opensearch_url must be an HTTP(S) endpoint without credentials")
        if parsed.query or parsed.fragment:
            raise ValueError("opensearch_url cannot contain a query or fragment")
        self.index = config.get("opensearch_index", "qrecc-full")
        if not isinstance(self.index, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", self.index):
            raise ValueError("Use one concrete lowercase OpenSearch index name")
        self.backend = config.get("retriever_backend", "hybrid")
        if self.backend not in {"bm25", "hybrid"}:
            raise ValueError("OpenSearch retrieval requires BM25 or the legacy hybrid backend")
        self.embedding_fn = embedding_fn
        self.dimension = config.get("opensearch_dimension", 1024)
        self.depth = config.get("hybrid_candidate_depth", 400)
        self.batch = config.get("opensearch_msearch_batch", 8)
        if any(type(v) is not int or v < 1 for v in (self.dimension, self.depth, self.batch)) or self.depth > 10000:
            raise ValueError("Invalid OpenSearch dimensions/candidate/batch budgets")
        self.ef_search = config.get("opensearch_ef_search", 512)
        if type(self.ef_search) is not int or self.ef_search < 1:
            raise ValueError("opensearch_ef_search must be a positive integer")
        self.timeout = config.get("opensearch_timeout", config.get("request_timeout", 180))
        self._built = False
        self._identity = None
        self._meta = None
        self._cache_hit = False

    def _request(self, method: str, path: str, payload=None, *, ndjson=False):
        if ndjson:
            body = payload.encode("utf-8") if isinstance(payload, str) else payload
        else:
            body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Accept": "application/json", "Content-Type": "application/x-ndjson" if ndjson else "application/json"}
        user, password = os.environ.get("OPENSEARCH_USERNAME"), os.environ.get("OPENSEARCH_PASSWORD")
        if user:
            if password is None:
                raise SearchServiceError("OPENSEARCH_PASSWORD is required with OPENSEARCH_USERNAME")
            headers["Authorization"] = "Basic " + base64.b64encode((user + ":" + password).encode()).decode()
        request = Request(self.url + path, data=body, headers=headers, method=method)
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            raise SearchServiceError(f"OpenSearch HTTP {exc.code}: {detail}", status=exc.code) from exc
        except (URLError, TimeoutError, OSError, ValueError) as exc:
            raise SearchServiceError(f"OpenSearch request failed ({self.url}): {exc}") from exc
        if not isinstance(result, dict) or result.get("error"):
            raise SearchServiceError("OpenSearch returned an invalid/error response")
        return result

    def _path(self, suffix=""):
        return "/" + quote(self.index, safe="") + suffix

    @staticmethod
    def check_search(response):
        if not isinstance(response, dict):
            raise SearchServiceError("Invalid OpenSearch search response")
        if response.get("error") or response.get("timed_out") or response.get("terminated_early"):
            raise SearchServiceError("Search failed, timed out, or terminated early")
        shards = response.get("_shards", {})
        if (not isinstance(shards, dict) or type(shards.get("total")) is not int
                or shards["total"] < 1 or shards.get("failed", 0)
                or shards.get("successful", 0) + shards.get("skipped", 0) < shards["total"]):
            raise SearchServiceError("Search has unavailable or failed shards")
        if not isinstance(response.get("hits", {}).get("hits"), list):
            raise SearchServiceError("Missing OpenSearch hits")

    def build(self, force=False):
        if force:
            raise ValueError("Use build_full_index.py to create a new server index; evaluation never rebuilds it")
        self._built = False
        mapping = self._request("GET", self._path("/_mapping"))
        if set(mapping) != {self.index}:
            raise SearchServiceError("Index aliases/wildcards are not accepted for a frozen experiment")
        mappings = mapping[self.index].get("mappings", {})
        meta = mappings.get("_meta", {}).get("qrecc", {})
        if meta.get("complete") is not True:
            raise SearchServiceError("Full corpus index is incomplete; finish build_full_index.py before evaluation")
        schema = ("qrecc-opensearch-bm25-v1" if self.backend == "bm25" else "qrecc-opensearch-hybrid-v1")
        if meta.get("index_schema") != schema:
            raise SearchServiceError(f"Index does not match the requested {self.backend} backend")
        if self.backend == "bm25" and mappings.get("properties", {}).get("text", {}).get("type") != "text":
            raise SearchServiceError("BM25 requires a searchable text field")
        if meta.get("document_key_policy") == VARIANT_POLICY:
            properties = mappings.get("properties", {})
            if any(properties.get(k, {}).get("type") != "keyword" for k in ("passage_key", "variant_key")):
                raise SearchServiceError("Variant index lacks its passage and text keys")
        if (self.config.get('evaluation_scope', 'full') == 'full'
                or meta.get('corpus_scope') in {'official_qrecc_full', 'hf_qrecc_parquet_full'}):
            from .hf_corpus import validate_index_corpus
            try:
                validate_index_corpus(self.config, meta)
            except (OSError, ValueError, KeyError) as error:
                raise SearchServiceError(str(error)) from error
        elif not self.config.get('allow_diagnostic_index', False):
            raise SearchServiceError('Unverified or partial corpus index cannot be used as full QReCC')
        if self.backend == "hybrid":
            vector = mappings.get("properties", {}).get("vector", {})
            if vector.get("dimension") != self.dimension or meta.get("dimensions") != self.dimension:
                raise SearchServiceError("Index vector dimension differs from current configuration")
            expected_vector = vector_mapping(self.config)
            if meta.get("vector_config") != expected_vector or not mapping_contains(vector, expected_vector):
                raise SearchServiceError("Index ANN/compression configuration differs from the experiment")
            if meta.get("embedding_model") != self.config.get("embedding_model"):
                raise SearchServiceError("Index embedding model differs from current configuration")
            digest = self.config.get("embedding_digest")
            if not digest or digest != meta.get("embedding_digest"):
                raise SearchServiceError("Verify Ollama embedding weight digest before using the full index")
            if meta.get("embedding_input_policy") != EMBEDDING_INPUT_POLICY:
                raise SearchServiceError("Index embedding input policy differs")
            # A parallel build records its worker budgets independently of query serving.
            # The weight digest, truncation policy and vector dimensions must still match.
            if not meta.get('index_embedding_workers') and meta.get('embedding_runtime_options', {}) != runtime_options(self.config):
                raise SearchServiceError('Index embedding runtime options differ')
        if not meta.get("corpus_fingerprint") or not meta.get("source_identity"):
            raise SearchServiceError("Index lacks frozen corpus provenance")
        count = self._request("GET", self._path("/_count"))
        if count.get("_shards", {}).get("failed", 0):
            raise SearchServiceError("Index count has failed shards")
        expected = meta.get("expected_documents")
        if type(expected) is not int or expected < 1 or count.get("count") != expected:
            raise SearchServiceError("Index count differs from the completed corpus manifest")
        settings = self._request("GET", self._path("/_settings"))
        uuid = settings[self.index]["settings"]["index"].get("uuid")
        if not uuid:
            raise SearchServiceError("Index UUID is unavailable")
        if str(settings[self.index]["settings"]["index"].get("blocks", {}).get("write")).lower() != "true":
            raise SearchServiceError("Full index must be write-blocked before evaluation")
        self._identity = {"store": "opensearch", "url": self.url, "index": self.index, "uuid": uuid,
                          "documents": expected, "corpus_fingerprint": meta["corpus_fingerprint"],
                          "count": expected, "indexed_passage_count": expected,
                          "index_schema": meta.get("index_schema"), "vector_config": meta.get("vector_config"),
                          "corpus_scope": meta.get("corpus_scope"),
                          "source_identity": meta.get('source_identity'),
                          "corpus_provenance": meta.get('source_identity', {}).get('provenance'),
                          "backend": self.backend}
        if self.backend == "hybrid":
            self._identity.update({
                          "embedding_digest": digest, "dimensions": self.dimension,
                          "embedding_input_policy": EMBEDDING_INPUT_POLICY,
                          "embedding_runtime_options": meta.get('embedding_runtime_options', {}),
                          "hybrid_candidate_depth": self.depth,
                          "knn_ef_search": max(self.depth, self.ef_search),
                          "knn_rescore_oversample_factor": (2.0 if expected_vector["mode"] == "on_disk" else None)})
        if meta.get("document_key_policy") == VARIANT_POLICY:
            self._identity.update(document_key_policy=VARIANT_POLICY,
                                  expected_count_argument=meta.get("expected_count_argument"),
                                  expected_documents=expected)
        if "index_embedding_workers" in meta:
            self._identity["index_embedding_workers"] = meta["index_embedding_workers"]
        self._meta = meta
        self._built = self._cache_hit = True
        return self.stats()

    def index_identity(self):
        if not self._built:
            self.build()
        return dict(self._identity)

    def stats(self):
        return {"backend": self.backend, "retrieval_store": "opensearch", "built": self._built,
                "cache_hit": self._cache_hit, "documents": (self._identity or {}).get("documents", 0),
                "chunks": (self._identity or {}).get("documents", 0),
                "embedding_dimensions": self.dimension if self.backend == "hybrid" else 0,
                "embedding_model": self.config.get("embedding_model") if self.backend == "hybrid" else None,
                "hybrid_candidate_depth": self.depth if self.backend == "hybrid" else None, "index_path": self.url + self._path(),
                "index_identity": self._identity}

    def contains_passage_ids(self, ids):
        if not self._built:
            self.build()
        values = list(dict.fromkeys(ids))
        found = set()
        for offset in range(0, len(values), 500):
            group = values[offset:offset + 500]
            if (self._meta or {}).get("document_key_policy") == VARIANT_POLICY:
                by_key = {document_key(pid): pid for pid in group}
                response = self._request("POST", self._path("/_search"), {
                    "size": 0, "query": {"terms": {"passage_key": list(by_key)}},
                    "aggs": {"covered": {"terms": {"field": "passage_key", "size": len(by_key)}}},
                })
                self.check_search(response)
                covered = response.get("aggregations", {}).get("covered", {})
                buckets = covered.get("buckets")
                if (not isinstance(buckets, list) or covered.get("sum_other_doc_count") != 0
                        or any(not isinstance(bucket, dict) or bucket.get("key") not in by_key for bucket in buckets)):
                    raise SearchServiceError("Incomplete variant passage coverage response")
                found.update(by_key[bucket["key"]] for bucket in buckets)
                continue
            response = self._request("POST", self._path("/_mget"),
                                     {"docs": [{"_id": document_key(pid), "_source": ["passage_id"]}
                                               for pid in group]})
            docs = response.get("docs")
            if not isinstance(docs, list) or len(docs) != len(group):
                raise SearchServiceError("Incomplete passage coverage response")
            for expected, doc in zip(group, docs):
                if doc.get("error"):
                    raise SearchServiceError("Passage coverage lookup failed")
                if type(doc.get("found")) is not bool:
                    raise SearchServiceError("Passage coverage response lacks a found flag")
                if doc.get("found"):
                    if doc.get("_source", {}).get("passage_id") != expected:
                        raise SearchServiceError("Original passage ID mismatch")
                    found.add(expected)
        return found

    def search(self, query, top_k=3):
        return self.search_many([query], top_k)[0]

    def search_many(self, queries, top_k=100):
        if not isinstance(queries, list) or any(not isinstance(q, str) or not q.strip() for q in queries):
            raise ValueError("queries must be a list of nonempty strings")
        if type(top_k) is not int or not 1 <= top_k <= self.depth * 2:
            raise ValueError("Invalid final Top-K for the configured candidate pool")
        if not self._built:
            self.build()
        unique = list(dict.fromkeys(queries))
        if self.backend == "bm25":
            return self._search_bm25(queries, unique, top_k)
        vectors = []
        for offset in range(0, len(unique), 16):
            batch = self.embedding_fn(unique[offset:offset + 16])
            if not isinstance(batch, (list, tuple)) or len(batch) != len(unique[offset:offset + 16]):
                raise SearchServiceError("Query embedding count mismatch")
            vectors.extend(_normalized(v) for v in batch)
        if any(len(v) != self.dimension for v in vectors):
            raise SearchServiceError("Query embedding dimension differs from the index")
        output = {}
        for offset in range(0, len(unique), self.batch):
            group, lines = unique[offset:offset + self.batch], []
            for query, vector in zip(group, vectors[offset:offset + self.batch]):
                base = {"size": self.depth, "_source": ["passage_id", "text", "title"], "track_total_hits": False,
                        "sort": [{"_score": "desc"}, {"passage_id": "asc"}], "track_scores": True}
                if (self._meta or {}).get("document_key_policy") == VARIANT_POLICY:
                    base["sort"].append({"variant_key": "asc"})
                knn = {"vector": vector, "k": self.depth,
                       "method_parameters": {"ef_search": max(self.depth, self.ef_search)},
                       "rescore": ({"oversample_factor": 2.0}
                                   if vector_mapping(self.config)["mode"] == "on_disk" else False)}
                # dfs_query_then_fetch uses collection-wide lexical statistics.
                lines.extend([json.dumps({"search_type": "dfs_query_then_fetch"}),
                              json.dumps({**base, "query": {"match": {"text": {"query": query}}}}, ensure_ascii=False),
                              "{}", json.dumps({**base, "query": {"knn": {"vector": knn}}})])
            result = self._request("POST", self._path("/_msearch"), "\n".join(lines) + "\n", ndjson=True)
            responses = result.get("responses")
            if not isinstance(responses, list) or len(responses) != 2 * len(group):
                raise SearchServiceError("Multi-search returned an incomplete result set")
            for index, query in enumerate(group):
                lexical, dense = responses[index * 2:index * 2 + 2]
                self.check_search(lexical)
                self.check_search(dense)
                variants = (self._meta or {}).get("document_key_policy") == VARIANT_POLICY
                output[query] = self._fuse(lexical["hits"]["hits"], dense["hits"]["hits"], top_k, variants=variants)
        return [[dict(hit) for hit in output[q]] for q in queries]

    def _search_bm25(self, queries, unique, top_k):
        output = {}
        variants = (self._meta or {}).get("document_key_policy") == VARIANT_POLICY
        for offset in range(0, len(unique), self.batch):
            group = unique[offset:offset + self.batch]
            lines = []
            for query in group:
                order = [{"_score": "desc"}, {"passage_id": "asc"}]
                if variants:
                    order.append({"variant_key": "asc"})
                body = {"size": max(top_k, self.depth) if variants else top_k,
                        "_source": ["passage_id", "text", "title"], "sort": order, "track_total_hits": False,
                        "query": {"match": {"text": {"query": query}}}}
                lines.extend((json.dumps({"search_type": "dfs_query_then_fetch"}), json.dumps(body, ensure_ascii=False)))
            responses = self._request("POST", self._path("/_msearch"),
                                      "\n".join(lines) + "\n", ndjson=True).get("responses")
            if not isinstance(responses, list) or len(responses) != len(group):
                raise SearchServiceError("BM25 multi-search returned an incomplete result set")
            for query, response in zip(group, responses):
                self.check_search(response)
                hits, seen = [], set()
                for hit in response["hits"]["hits"]:
                    data = hit.get("_source", {})
                    pid, text, score = data.get("passage_id"), data.get("text"), hit.get("_score")
                    if not isinstance(pid, str) or not pid or not isinstance(text, str) or not text.strip():
                        raise SearchServiceError("BM25 hit lacks original passage ID or text")
                    if not isinstance(score, (float, int)):
                        raise SearchServiceError("BM25 hit lacks a numeric score")
                    if pid not in seen:
                        seen.add(pid)
                        hits.append({"id": "passage_" + document_key(pid)[:16], "passage_id": pid, "document": pid,
                                     "title": data.get("title", pid[:120]), "text": text, "start": 0, "end": len(text),
                                     "score": score, "bm25_score": score, "dense_score": 0.0})
                    if len(hits) == top_k:
                        break
                output[query] = hits
        return [[dict(hit) for hit in output[query]] for query in queries]

    @staticmethod
    def _fuse(lexical, dense, top_k, *, variants=False):
        totals, sources, best_rank, lexical_scores, dense_scores = {}, {}, {}, {}, {}
        for ranking, score_map in ((lexical, lexical_scores), (dense, dense_scores)):
            seen = set()
            for rank, hit in enumerate(ranking, 1):
                source = hit.get("_source", {})
                pid, text = source.get("passage_id"), source.get("text")
                if not isinstance(pid, str) or not pid or not isinstance(text, str) or not text.strip():
                    raise SearchServiceError("Indexed hit lacks original passage ID or text")
                if pid in seen:
                    continue
                seen.add(pid)
                if not variants or pid not in best_rank or rank < best_rank[pid]:
                    sources[pid], best_rank[pid] = source, rank
                score_map[pid] = hit.get("_score", 0.0)
                totals[pid] = totals.get(pid, 0.0) + 1 / (60 + rank)
        ordered = sorted(totals, key=lambda pid: (-totals[pid], pid))[:top_k]
        return [{"id": "passage_" + document_key(pid)[:16], "passage_id": pid,
                 "document": pid, "title": sources[pid].get("title", pid[:120]),
                 "text": sources[pid]["text"], "start": 0, "end": len(sources[pid]["text"]),
                 "score": totals[pid], "bm25_score": lexical_scores.get(pid, 0.0),
                 "dense_score": dense_scores.get(pid, 0.0)} for pid in ordered]
