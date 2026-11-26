"""Full QReCC passage streams and verified official downloads, without corpus lists."""
from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path
from urllib.request import Request, urlopen
import zipfile

OFFICIAL_FILES = {
    "qrecc-training.json": "72abc475c62b99546f4c61192cd721f3",
    "qrecc-test.json": "7ad29296a1dfd2f75dafd653a8822269",
    "passages.zip": "16c48fb9c4b9603b8ccc8ba3bd7d69c1",
}
OFFICIAL_BASE = "https://zenodo.org/records/5760304/files/"


def file_digest(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_official(name: str, directory: Path, *, opener=urlopen) -> Path:
    """Resume .part downloads only on a valid Range response; verify official MD5."""
    if name not in OFFICIAL_FILES:
        raise ValueError("Unknown official QReCC filename")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / name
    expected = OFFICIAL_FILES[name]
    if destination.exists():
        if file_digest(destination, "md5") != expected:
            raise FileExistsError(f"Existing {destination} has the wrong official MD5; input was not overwritten")
        return destination
    partial = destination.with_name(name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "RAG-Research-QReCC-preparation"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    with opener(Request(OFFICIAL_BASE + name + "?download=1", headers=headers), timeout=60) as response:
        status = getattr(response, "status", None) or response.getcode()
        if offset and status == 206:
            if not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                raise ValueError("Server returned a mismatched resume range")
            mode = "ab"
        elif status == 200:
            mode = "wb"  # A server that ignores Range must restart, never append.
        else:
            raise ValueError(f"Unexpected download status: {status}")
        with partial.open(mode) as handle:
            for block in iter(lambda: response.read(1024 * 1024), b""):
                handle.write(block)
    actual = file_digest(partial, "md5")
    if actual != expected:
        raise ValueError(f"Official MD5 mismatch for {name}; .part retained for inspection")
    partial.replace(destination)
    return destination


def _json_objects(handle, label: str):
    """Incrementally decode JSONL, one JSON object, or a JSON array of objects."""
    decoder = json.JSONDecoder()
    buffer, eof, started, array, after_item, closed = "", False, False, False, False, False
    require_value = False
    while True:
        buffer = buffer.lstrip("\ufeff \t\r\n")
        if not started and buffer:
            array = buffer.startswith("[")
            if array:
                buffer = buffer[1:].lstrip()
            started = True
        if started and array and after_item and buffer:
            if buffer[0] == ",":
                buffer, after_item, require_value = buffer[1:].lstrip(), False, True
                if buffer.startswith("]"):
                    raise ValueError(f"{label}: trailing JSON array comma")
            elif buffer[0] != "]":
                raise ValueError(f"{label}: missing JSON array comma")
        if started and array and buffer.startswith("]"):
            if require_value:
                raise ValueError(f"{label}: trailing JSON array comma")
            buffer, closed = buffer[1:].lstrip(), True
            if buffer:
                raise ValueError(f"{label}: trailing content after JSON array")
            for trailing in iter(lambda: handle.read(65536), ""):
                if trailing.strip():
                    raise ValueError(f"{label}: trailing content after JSON array")
            return
        if buffer and started:
            try:
                value, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                if eof:
                    raise ValueError(f"{label}: invalid or incomplete JSON") from None
            else:
                if not isinstance(value, dict):
                    raise ValueError(f"{label}: passage records must be JSON objects")
                yield value
                buffer, after_item, require_value = buffer[end:], True, False
                continue
        if eof:
            if buffer or array and not closed:
                raise ValueError(f"{label}: incomplete JSON array")
            return
        block = handle.read(65536)
        if block:
            buffer += block
            if len(buffer) > 64 * 1024 * 1024:
                raise ValueError(f"{label}: a single passage exceeds the 64 MiB parsing limit")
        else:
            eof = True


def _supported(name: str) -> bool:
    return name.lower().endswith((".jsonl", ".json", ".jsonl.gz", ".json.gz"))


def _canonical(rows, stats: dict):
    for row in rows:
        stats["rows_seen"] = stats.get("rows_seen", 0) + 1
        identifier = row.get("id")
        text = row.get("text", row.get("contents"))
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError("Full passage records require a nonempty original string id")
        if not isinstance(text, str):
            raise ValueError(f"Passage {identifier}: missing text/contents string")
        if not text.strip():
            stats["empty_text_skipped"] = stats.get("empty_text_skipped", 0) + 1
            continue
        stats["passages_yielded"] = stats.get("passages_yielded", 0) + 1
        yield {"id": identifier, "text": text}


def iter_full_passages(path: Path, *, stats: dict | None = None):
    """Stream original IDs and full text from official ZIP/JSONL/JSON/gzip shards.

    ZIP members are read in place, never extracted; corpus size does not become
    Python list/set memory. Callers indexing by ID must detect/handle duplicates.
    Official source format is {id, contents}; canonical {id, text} is also accepted.
    """
    path = Path(path)
    stats = stats if stats is not None else {}
    if not path.exists():
        raise FileNotFoundError(f"Full QReCC passages not found: {path}")
    if path.is_dir():
        parquet = sorted(path.glob('*.parquet')) or sorted((path / 'data').glob('*.parquet'))
        if parquet:
            for file in parquet:
                yield from iter_full_passages(file, stats=stats)
            return
        files = sorted(p for p in path.rglob("*") if p.is_file() and _supported(p.name))
        if not files:
            raise ValueError("Passage directory contains no JSON/JSONL/gzip shards")
        for file in files:
            yield from iter_full_passages(file, stats=stats)
        return
    if path.suffix.lower() == '.parquet':
        from .hf_corpus import iter_parquet_rows
        yield from _canonical(iter_parquet_rows(path), stats)
        return
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            members = sorted(name for name in archive.namelist() if _supported(name) and not name.endswith("/"))
            if not members:
                raise ValueError("Passage ZIP contains no supported JSON/JSONL shards")
            for name in members:
                with archive.open(name) as raw:
                    stream = gzip.GzipFile(fileobj=raw) if name.lower().endswith(".gz") else raw
                    with io.TextIOWrapper(stream, encoding="utf-8-sig") as handle:
                        yield from _canonical(_json_objects(handle, f"{path}!{name}"), stats)
        return
    if not _supported(path.name):
        raise ValueError("Passage input must be ZIP, JSONL, JSON, gzip, or a shard directory")
    opener = gzip.open if path.name.lower().endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8-sig") as handle:
        yield from _canonical(_json_objects(handle, str(path)), stats)
