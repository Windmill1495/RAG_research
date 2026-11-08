"""Small, readable retrieval implementation with no third-party dependencies.

The default retriever is lexical BM25. ``dense`` and ``hybrid`` accept an
embedding function, making the indexing/retrieval code independent of Ollama.
Offsets in Chunk refer to text after CRLF has been normalized to LF.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Callable


INDEX_VERSION = 2
TOKENIZER_VERSION = "chinese-unigram-bigram-english-v1"
EmbeddingFunction = Callable[[list[str]], list[list[float]]]


@dataclass(frozen=True)
class Chunk:
    id: str
    document: str
    title: str
    text: str
    start: int
    end: int
    # 已切分的段落语料保留原始 passage ID，文件切块时为 None。
    passage_id: str | None = None


def tokenize(text: str) -> list[str]:
    """English words plus Chinese characters and within-run character bigrams.

    This is a transparent teaching baseline, not a neural embedding model or
    a Chinese word segmenter. Unigrams also permit matching one-character terms.
    """
    tokens: list[str] = []
    for match in re.finditer(r"[\u3400-\u9fff]+|[a-z0-9]+(?:[_'-][a-z0-9]+)*", text.lower()):
        word = match.group(0)
        if "\u3400" <= word[0] <= "\u9fff":
            tokens.extend(word)
            tokens.extend(word[i:i + 2] for i in range(len(word) - 1))
        else:
            tokens.append(word)
    return tokens


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 80) -> list[tuple[str, int, int]]:
    """Split by character count, preferring nearby paragraph/sentence boundaries.

    Every non-whitespace character is covered; chunks never exceed chunk_size.
    ``overlap`` counts characters, not tokens. Trimming outer whitespace can
    make the visible overlap smaller than the requested maximum.
    """
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
        raise ValueError("chunk_size 必须是正整数（单位：字符）。")
    if not isinstance(overlap, int) or isinstance(overlap, bool) or not 0 <= overlap < chunk_size:
        raise ValueError("overlap 必须满足 0 <= overlap < chunk_size。")
    chunks: list[tuple[str, int, int]] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            # 设置一个下限 floor，只接受位于当前窗口后 40% 区域的边界
            floor = start + max(1, int(chunk_size * 0.6))
            boundaries = [m.end() for m in re.finditer(r"\n\s*\n|[。！？!?；;\n]", text[start:end])] # 找出窗口中的标点和换行边界
            candidates = [start + value for value in boundaries if start + value >= floor]
            if candidates:
                end = candidates[-1] # 筛选边界，并选择最靠后的一个
        raw = text[start:end]
        leading = len(raw) - len(raw.lstrip())
        trailing = len(raw) - len(raw.rstrip()) # 去除文本块首尾的空白，并修正位置
        visible_end = end - trailing
        visible_start = start + leading
        if visible_start < visible_end:
            chunks.append((text[visible_start:visible_end], visible_start, visible_end)) # 把有效文本块存入结果
        if end == len(text):
            break # 当前窗口已经到达文本末尾，就结束循环
        start = max(start + 1, end - overlap) # 移动窗口起点，确保至少前进一个字符，下一个起点是 end - overlap
    return chunks

def _normalized(vector: list[float]) -> list[float]: # 对向量进行归一化，返回单位向量
    if not isinstance(vector, (list, tuple)) or not vector:
        raise ValueError("Embedding 返回了空向量或无效向量。")
    try:
        numbers = [float(x) for x in vector]
    except (TypeError, ValueError) as exc:
        raise ValueError("Embedding 向量必须只包含数值。") from exc
    if any(not math.isfinite(x) for x in numbers):
        raise ValueError("Embedding 向量包含 NaN 或无穷大。")
    norm = math.sqrt(sum(x * x for x in numbers))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("Embedding 返回了零向量或溢出向量。")
    return [x / norm for x in numbers]


class Retriever:
    """A local Markdown corpus or pre-segmented JSONL passage collection.

    JSONL rows have ``id`` and ``text``. A passage is one retrieval unit, even
    when it exceeds chunk_size; passage IDs are labels, never filesystem paths.
    BM25's scan and inverted implementations use exactly the same formula.
    """

    def __init__(
        self,
        documents_dir: Path,
        index_path: Path,
        chunk_size: int = 500,
        overlap: int = 80,
        backend: str = "bm25",
        embedding_model: str = "bge-m3",
        embedding_fn: EmbeddingFunction | None = None,
        corpus_format: str = "markdown",
        bm25_mode: str = "inverted",
        *, embedding_digest: str | None = None, embedding_input_policy: str | None = None,
        hybrid_candidate_depth: int = 400,
    ) -> None:
        # Reuse the splitter's validation, including invalid empty input cases.
        chunk_text("", chunk_size, overlap)
        if backend not in {"bm25", "dense", "hybrid"}:
            raise ValueError("backend 必须为 bm25、dense 或 hybrid。")
        if corpus_format not in {"markdown", "passages_jsonl"}:
            raise ValueError("corpus_format 必须为 markdown 或 passages_jsonl。")
        if bm25_mode not in {"scan", "inverted"}:
            raise ValueError("bm25_mode 必须为 scan 或 inverted。")
        self.documents_dir = Path(documents_dir).resolve()
        self.index_path = Path(index_path).resolve()
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.backend = backend
        if type(hybrid_candidate_depth) is not int or hybrid_candidate_depth < 1:
            raise ValueError("hybrid_candidate_depth 必须是正整数")
        # Ranking-only budget: vectors and corpus index remain reusable.
        self.hybrid_candidate_depth = hybrid_candidate_depth
        self.embedding_model = embedding_model
        self.embedding_digest = embedding_digest
        self.embedding_input_policy = embedding_input_policy
        self.embedding_fn = embedding_fn
        self.corpus_format = corpus_format
        self.bm25_mode = bm25_mode
        self.chunks: list[Chunk] = []
        self.embeddings: list[list[float]] = []
        self._term_counts: list[Counter[str]] = []
        self._document_frequency: Counter[str] = Counter()
        self._lengths: list[int] = []
        self._average_length = 0.0
        self._postings: dict[str, list[tuple[int, int]]] = {}
        self._idf: dict[str, float] = {}
        self._length_normalizers: list[float] = []
        self._bm25_candidates: list[int] = []
        self._last_query_work: dict = {}
        self._documents: list[dict] = []
        self._corpus_sha256: str | None = None
        self._signature: str | None = None
        self._built = False
        self._cache_hit = False
        self._dense_matrix = None

    def _config(self) -> dict:
        return {
            "corpus_format": self.corpus_format,
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "backend": self.backend,
            "embedding_model": self.embedding_model if self.backend != "bm25" else None,
            **({"embedding_digest": self.embedding_digest} if self.backend != "bm25" and self.embedding_digest else {}),
            **({"embedding_input_policy": self.embedding_input_policy}
               if self.backend != "bm25" and self.embedding_input_policy else {}),
            "tokenizer": TOKENIZER_VERSION,
            "bm25_k1": 1.5,
            "bm25_b": 0.75,
        }

    def _read_documents(self) -> list[tuple[str, str, str]]:
        if self.corpus_format == "passages_jsonl":
            return self._read_passages()
        if not self.documents_dir.is_dir():
            raise ValueError(f"知识库目录不存在：{self.documents_dir}")
        documents = []
        for path in sorted(self.documents_dir.rglob("*")): # 遍历目录及子目录
            if not path.is_file() or path.suffix.lower() not in {".md", ".txt"}:
                continue # 只读取 .md、.txt 文件
            # Do not read knowledge files through symlinks outside the corpus.
            resolved = path.resolve()
            if path.is_symlink() or not resolved.is_relative_to(self.documents_dir):
                raise ValueError(f"知识库不允许外部链接文件：{path.name}")
            relative = path.relative_to(self.documents_dir).as_posix()
            raw = path.read_bytes()
            try:
                text = raw.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
            except UnicodeDecodeError as exc:
                raise ValueError(f"文件 {relative} 不是 UTF-8 编码，请另存为 UTF-8。") from exc
            documents.append((relative, text, hashlib.sha256(raw).hexdigest()))
        if not any(text.strip() for _, text, _ in documents):
            raise ValueError("知识库为空，请在知识库目录添加 UTF-8 编码的 .md 或 .txt 文件。")
        return documents

    def _read_passages(self) -> list[tuple[str, str, str]]:
        """Read actual passage text, retaining IDs exactly for gold matching."""
        if not self.documents_dir.is_file():
            raise ValueError(f"passages JSONL 文件不存在：{self.documents_dir}")
        raw = self.documents_dir.read_bytes()
        try:
            content = raw.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError("passages JSONL 必须使用 UTF-8 编码。") from exc
        # 原文件指纹还覆盖行顺序、JSON 字段等变化，不只覆盖处理后的正文。
        self._corpus_sha256 = hashlib.sha256(raw).hexdigest()
        seen: set[str] = set()
        passages = []
        for line_number, line in enumerate(content.splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"passages JSONL 第 {line_number} 行不是有效 JSON。") from exc
            if not isinstance(row, dict) or not {"id", "text"} <= row.keys():
                raise ValueError(f"passages JSONL 第 {line_number} 行需要 id、text 字段。")
            passage_id, text = row["id"], row["text"]
            if not isinstance(passage_id, str) or not passage_id.strip() or passage_id != passage_id.strip():
                raise ValueError(f"passages JSONL 第 {line_number} 行 id 必须为非空字符串，且首尾不能有空白。")
            if passage_id in seen:
                raise ValueError(f"passages JSONL 第 {line_number} 行出现重复 id：{passage_id}")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"passages JSONL 第 {line_number} 行 text 必须为非空字符串。")
            seen.add(passage_id)
            text = text.replace("\r\n", "\n").replace("\r", "\n")
            passages.append((passage_id, text, hashlib.sha256(text.encode("utf-8")).hexdigest()))
        if not passages:
            raise ValueError("passages JSONL 知识库为空。")
        return passages

    def _embed(self, texts: list[str]) -> list[list[float]]: # 调用向量化函数，并检查结果
        # 输入几段文字，就应返回几个向量
        if self.embedding_fn is None:
            raise RuntimeError("dense/hybrid 检索需要可用的 Embedding 服务；可切换为 bm25。")
        vectors = self.embedding_fn(texts)
        if not isinstance(vectors, (list, tuple)) or len(vectors) != len(texts):
            raise ValueError("Embedding 返回的向量数量与输入文本数量不一致。")
        normalized = [_normalized(v) for v in vectors]
        if len({len(v) for v in normalized}) > 1:
            raise ValueError("Embedding 返回了维度不一致的向量。")
        return normalized

    def _prepare_bm25(self) -> None: # 准备 BM25 所需的统计量
        self._term_counts = [Counter(tokenize(chunk.text)) for chunk in self.chunks] # 对每个文本块分词，再统计词项出现次数
        self._lengths = [sum(counts.values()) for counts in self._term_counts]
        self._average_length = sum(self._lengths) / max(1, len(self.chunks))
        self._document_frequency = Counter()
        for counts in self._term_counts: # 最后统计每个词项出现于多少个块
            self._document_frequency.update(counts.keys())
        # 倒排表：词项 -> [(文本块下标, 该词在块内的次数), ...]。
        # 它只改变访问方式，分词、k1、b 和 query_count 均沿用原实现。
        self._postings = {}
        for index, counts in enumerate(self._term_counts):
            for term, frequency in counts.items():
                self._postings.setdefault(term, []).append((index, frequency))
        size = len(self.chunks)
        self._idf = {
            term: math.log(1 + (size - df + 0.5) / (df + 0.5))
            for term, df in self._document_frequency.items()
        }
        self._length_normalizers = [
            1.5 * (1 - 0.75 + 0.75 * length / max(1, self._average_length))
            for length in self._lengths
        ]
        self._bm25_candidates = []
        self._last_query_work = {}

    def _try_load(self, signature: str) -> bool: # 尝试复用已有索引
        # 把上一次已经处理好的知识库数据保存下来，下次需要时直接加载，省去重复处理
        '''
        读取 index.json
            ↓
        检查索引格式版本是否一致
            ↓
        检查 signature 是否一致(signature 是本次根据知识库文件和配置计算的指纹)
            ↓
        恢复 Chunk 对象，并检查文本、位置、长度
            ↓
        向量检索模式下，再检查向量数量和维度
            ↓
        放入 self.chunks、self.embeddings
            ↓
        返回 True
        '''
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
            if data.get("version") != INDEX_VERSION or data.get("signature") != signature:
                return False
            chunks = [Chunk(**chunk) for chunk in data["chunks"]]
            if not chunks or any(
                not isinstance(chunk.text, str) or not chunk.text.strip()
                or type(chunk.start) is not int or type(chunk.end) is not int
                or chunk.start < 0
                or chunk.end <= chunk.start or len(chunk.text) != chunk.end - chunk.start
                or (self.corpus_format == "markdown" and
                    (len(chunk.text) > self.chunk_size or chunk.passage_id is not None))
                or (self.corpus_format == "passages_jsonl" and
                    (not isinstance(chunk.passage_id, str) or not chunk.passage_id
                     or chunk.document != chunk.passage_id or chunk.start != 0))
                or not isinstance(chunk.id, str) or not chunk.id
                or not isinstance(chunk.document, str) or not isinstance(chunk.title, str)
                for chunk in chunks
            ):
                return False
            if len({chunk.id for chunk in chunks}) != len(chunks):
                return False
            if self.corpus_format == "passages_jsonl" and len({chunk.passage_id for chunk in chunks}) != len(chunks):
                return False
            if self.corpus_format == "passages_jsonl":
                originals = {document["document"]: document for document in self._documents}
                if {chunk.passage_id for chunk in chunks} != originals.keys():
                    return False
                # 缓存中的原始 ID 与正文必须继续成对，不能把别段文字贴到正例 ID。
                if any(len(chunk.text) != originals[chunk.passage_id]["characters"]
                       or hashlib.sha256(chunk.text.encode("utf-8")).hexdigest() != originals[chunk.passage_id]["sha256"]
                       for chunk in chunks):
                    return False
            vectors = data.get("embeddings", [])
            if not isinstance(vectors, list):
                return False
            if self.backend == "bm25" and vectors:
                return False
            if self.backend != "bm25":
                if len(vectors) != len(chunks):
                    return False
                vectors = [_normalized(vector) for vector in vectors]
                if len({len(vector) for vector in vectors}) != 1:
                    return False
            self.chunks = chunks
            self.embeddings = vectors
            return True
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            # A stale, interrupted or corrupt index is safely rebuilt.
            return False

    def _save(self, signature: str) -> None: # 保存当前构建结果
        '''
        创建保存目录
            ↓
        写入临时文件
            ↓
        flush、fsync 刷新缓冲
            ↓
        os.replace() 用完整的新文件替换正式索引
            ↓
        finally 清理遗留临时文件
        '''
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": INDEX_VERSION, # 把数据整理成一个字典
            "signature": signature,
            "config": self._config(),
            "documents": self._documents,
            "chunks": [asdict(chunk) for chunk in self.chunks],
            "embeddings": self.embeddings,
        }
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", suffix=".tmp", dir=self.index_path.parent,
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.index_path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    def build(self, force: bool = False) -> dict:
        """Load a compatible index or rebuild after any corpus/config change."""
        # A failed rebuild must not leave a partly replaced in-memory index
        # looking ready to query (for example after an embedding outage).
        self._built = False
        self._dense_matrix = None
        documents = self._read_documents()
        self._documents = [
            {"document": name, "sha256": digest, "characters": len(text)}
            for name, text, digest in documents
        ]
        signature = hashlib.sha256(json.dumps( # SHA-256 是一种哈希算法，可以给数据计算出一个固定长度的“指纹”
            {"config": self._config(), "documents": self._documents,
             "corpus_sha256": self._corpus_sha256},
            sort_keys=True, ensure_ascii=False,
        ).encode("utf-8")).hexdigest()
        self._signature = signature
        self._cache_hit = not force and self._try_load(signature)
        if not self._cache_hit:
            chunks = []
            for name, text, _ in documents:
                heading = next((line.lstrip("# ").strip() for line in text.splitlines() if line.strip()), name)
                title = heading[:120]
                # passage 本来就是公开数据的检索单元，不再按 500 字符切割。
                units = ([(text, 0, len(text))] if self.corpus_format == "passages_jsonl"
                         else chunk_text(text, self.chunk_size, self.overlap))
                for content, start, end in units:
                    digest = hashlib.sha256(f"{name}\0{start}\0{end}\0{content}".encode("utf-8")).hexdigest()[:16]
                    chunks.append(Chunk(f"chunk_{digest}", name, title, content, start, end,
                                        name if self.corpus_format == "passages_jsonl" else None))
            self.chunks = chunks
            self.embeddings = []
            if self.backend != "bm25": # 需要向量检索时，就为文本块生成 Embedding，每批处理最多16个文本块
                print(f"[Index] Encoding {len(chunks)} passages with {self.embedding_model}...", flush=True)
                for offset in range(0, len(chunks), 16):
                    self.embeddings.extend(self._embed([chunk.text for chunk in chunks[offset:offset + 16]]))
                    completed = len(self.embeddings)
                    if offset == 0 or completed % 256 == 0 or completed == len(chunks):
                        print(f"[Index] Embeddings: {completed}/{len(chunks)} ({completed / len(chunks):.1%})", flush=True)
                if len({len(vector) for vector in self.embeddings}) != 1:
                    raise ValueError("Embedding 各批次返回的向量维度不同，请重建索引。")
            self._save(signature)
        self._prepare_bm25()
        self._built = True
        return self.stats()

    def stats(self) -> dict: # stats() 用来返回检索器的“状态报告”
        return {
            "backend": self.backend,
            "hybrid_candidate_depth": self.hybrid_candidate_depth if self.backend == "hybrid" else None,
            "corpus_format": self.corpus_format,
            "bm25_mode": self.bm25_mode,
            "documents": len(self._documents),
            "chunks": len(self.chunks),
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "embedding_model": self.embedding_model if self.backend != "bm25" else None,
            "embedding_dimensions": len(self.embeddings[0]) if self.embeddings else 0,
            "index_path": str(self.index_path),
            "corpus_path": str(self.documents_dir),
            "signature": self._signature,
            "corpus_sha256": self._corpus_sha256,
            "vocabulary_terms": len(self._postings),
            "posting_entries": sum(len(postings) for postings in self._postings.values()),
            "last_query_work": dict(self._last_query_work),
            "cache_hit": self._cache_hit,
            "built": self._built,
        }

    def _bm25_scores(self, query: str) -> list[float]:
        """Same BM25 values via a teaching scan or postings-based access."""
        if self.bm25_mode == "scan":
            return self._bm25_scores_scan(query)
        return self._bm25_scores_inverted(query)

    def _bm25_scores_inverted(self, query: str) -> list[float]:
        """Only matching postings receive a BM25 contribution."""
        terms = Counter(tokenize(query))
        sparse_scores: dict[int, float] = {}
        visited = 0
        for term, query_count in terms.items():
            idf = self._idf.get(term)
            if idf is None:
                continue
            for index, frequency in self._postings[term]:
                denominator = frequency + self._length_normalizers[index]
                sparse_scores[index] = sparse_scores.get(index, 0.0) + query_count * idf * frequency * 2.5 / denominator
                visited += 1
        scores = [0.0] * len(self.chunks)
        for index, score in sparse_scores.items():
            scores[index] = score
        self._bm25_candidates = [index for index, score in sparse_scores.items() if score > 0]
        self._last_query_work = {"posting_entries_visited": visited,
                                 "term_chunk_pairs_examined": visited,
                                 "scored_chunks": len(sparse_scores)}
        return scores

    def _bm25_scores_scan(self, query: str) -> list[float]:
        """
        与下面6个统计量有关：
            query_count	某词项在问题中出现的次数
            frequency	某词项在当前文本块中出现的次数
            df	包含该词项的文本块数量
            size	知识库的文本块总数
            length	当前文本块的词项总数
            self._average_length	所有文本块的平均词项数
        """
        terms = Counter(tokenize(query))
        size = len(self.chunks)
        scores = []
        # Standard BM25, k1=1.5, b=0.75, positive Robertson/Lucene IDF.
        for counts, length in zip(self._term_counts, self._lengths):
            score = 0.0
            for term, query_count in terms.items():
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                df = self._document_frequency[term]
                idf = math.log(1 + (size - df + 0.5) / (df + 0.5))
                denominator = frequency + 1.5 * (1 - 0.75 + 0.75 * length / max(1, self._average_length))
                score += query_count * idf * frequency * 2.5 / denominator
            scores.append(score)
        self._bm25_candidates = [index for index, score in enumerate(scores) if score > 0]
        self._last_query_work = {"posting_entries_visited": 0,
                                 "term_chunk_pairs_examined": size * len(terms),
                                 "scored_chunks": len(self._bm25_candidates)}
        return scores

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("检索问题不能为空。")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k 必须是正整数。")
        if not self._built:
            self.build()
        bm25 = self._bm25_scores(query)
        dense: list[float] = []
        if self.backend != "bm25": # 计算当前问题与每个文本块的向量相似度，bm25以外的方法需要这个
            query_vector = self._embed([query])[0]
            if len(query_vector) != len(self.embeddings[0]):
                raise ValueError("查询与索引的 Embedding 维度不同，请重建索引。")
            dense = [sum(a * b for a, b in zip(query_vector, vector)) for vector in self.embeddings]
        return self._rank_scores(bm25, dense, top_k)

    def search_many(self, queries: list[str], top_k: int = 100) -> list[list[dict]]:
        """Batch query embeddings, sharing the same index and per-query ranking rule.

        NumPy, when installed, accelerates dot products in float64. The standard
        library path remains available. Identical query strings are encoded once;
        BM25 work is reused for permutations with the same token multiset.
        """
        if not isinstance(queries, list) or any(not isinstance(q, str) or not q.strip() for q in queries):
            raise ValueError("queries 必须是非空查询字符串组成的列表")
        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k 必须是正整数")
        if not self._built:
            self.build()
        unique = list(dict.fromkeys(queries))
        vectors = []
        if self.backend != "bm25":
            for offset in range(0, len(unique), 16):
                vectors.extend(self._embed(unique[offset:offset + 16]))
            if any(len(v) != len(self.embeddings[0]) for v in vectors):
                raise ValueError("查询与索引的 Embedding 维度不同，请重建索引。")
        np = None
        if vectors:
            try:
                import numpy as np
            except ImportError:
                pass
        if np is not None and self._dense_matrix is None:
            self._dense_matrix = np.asarray(self.embeddings, dtype=np.float64)
        bm25_cache, output = {}, {}
        for index, query in enumerate(unique):
            key = tuple(sorted(Counter(tokenize(query)).items()))
            if key not in bm25_cache:
                bm25_cache[key] = (self._bm25_scores(query), list(self._bm25_candidates))
            bm25, candidates = bm25_cache[key]
            self._bm25_candidates = candidates
            dense = []
            if vectors:
                dense = ((self._dense_matrix @ np.asarray(vectors[index], dtype=np.float64)).tolist()
                         if np is not None else
                         [sum(a * b for a, b in zip(vectors[index], v)) for v in self.embeddings])
            output[query] = self._rank_scores(bm25, dense, top_k)
        return [[dict(hit) for hit in output[query]] for query in queries]

    def _rank_scores(self, bm25: list[float], dense: list[float], top_k: int) -> list[dict]:
        """Shared by scalar and batch search; hybrid RRF here stays at 60."""
        if self.backend == "bm25":
            scores = bm25
            candidates = self._bm25_candidates
        elif self.backend == "dense": # 使用向量分数，所有块参与排序
            scores = dense
            candidates = list(range(len(self.chunks)))
        else: # Hybrid 模式：融合两种方法的排名
            # Reciprocal Rank Fusion avoids comparing incompatible score scales.
            limit = self.hybrid_candidate_depth
            lexical_rank = sorted(self._bm25_candidates, key=lambda i: (-bm25[i], self.chunks[i].id))[:limit] #  # BM25 排名前 limit 的块下标
            dense_rank = sorted(range(len(dense)), key=lambda i: (-dense[i], self.chunks[i].id))[:limit] # # 向量排名前 limit 的块下标
            scores = [0.0] * len(self.chunks)
            for ranking in (lexical_rank, dense_rank):
                for rank, index in enumerate(ranking, start=1):
                    scores[index] += 1.0 / (60 + rank)
            candidates = [i for i, value in enumerate(scores) if value > 0]
        ranked = sorted(candidates, key=lambda i: (-scores[i], self.chunks[i].id))[:top_k] # 统一排序，取最终 Top-K
        results = []
        for index in ranked:
            result = asdict(self.chunks[index]) # 把 Chunk 对象转换为字典
            result["score"] = round(scores[index], 8)
            result["bm25_score"] = round(bm25[index], 8)
            if dense:
                result["dense_score"] = round(dense[index], 8)
            results.append(result) # 再加上分数
        return results # 把选中的文本块整理成字典返回
