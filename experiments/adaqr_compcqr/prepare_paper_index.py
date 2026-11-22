"""Extract the official QReCC collection and build the papers' Lucene index."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
from zipfile import ZipFile


OFFICIAL_MD5 = "16c48fb9c4b9603b8ccc8ba3bd7d69c1"
OFFICIAL_URL = "https://zenodo.org/records/5115890"


def extract(archive: Path, input_dir: Path):
    print(f"Checking official MD5 for {archive}", flush=True)
    with archive.open("rb") as stream:
        if hashlib.file_digest(stream, "md5").hexdigest() != OFFICIAL_MD5:
            raise ValueError("Archive differs from the official QReCC passages.zip")
    with ZipFile(archive) as source:
        members = [item for item in source.infolist() if not item.is_dir()]
        if len({item.filename for item in members}) != len(members):
            raise ValueError("Duplicate archive member name")
        for index, item in enumerate(members, 1):
            relative = PurePosixPath(item.filename)
            if (relative.is_absolute() or ".." in relative.parts or "\\" in item.filename
                    or relative.parts[0] != "collection-paragraph" or relative.suffix != ".jsonl"):
                raise ValueError(f"Unexpected official archive member {item.filename}")
            target = input_dir.joinpath(*relative.parts)
            if not target.is_file() or target.stat().st_size != item.file_size:
                target.parent.mkdir(parents=True, exist_ok=True)
                partial = target.with_suffix(".jsonl.part")
                with source.open(item) as stream, partial.open("wb") as output:
                    shutil.copyfileobj(stream, output, length=1024 * 1024)
                partial.replace(target)
            if index == 1 or index % 1000 == 0 or index == len(members):
                print(f"Extracted/verified {index:,} of {len(members):,} shards", flush=True)
    return input_dir / "collection-paragraph"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    index_method = parser.add_mutually_exclusive_group(required=True)
    index_method.add_argument("--index-python", type=Path)
    index_method.add_argument("--index-jar", type=Path, help="Use the Anserini JAR bundled with Pyserini 0.24.0")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--extract-only", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    with (work / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        collection = extract(args.archive.resolve(), work / "raw")
        if args.extract_only:
            return
        staging = work / "lucene.incomplete"
        final = work / "lucene"
        if final.exists() or staging.exists():
            raise FileExistsError("Index path already exists; inspect it before starting another build")
        if args.index_jar:
            command = ["java", "-Xmx16g", "-cp", str(args.index_jar.resolve(strict=True)),
                       "io.anserini.index.IndexCollection"]
        else:
            command = [str(args.index_python), "-m", "pyserini.index.lucene"]
        command += ["-collection", "JsonCollection",
                   "-generator", "DefaultLuceneDocumentGenerator", "-threads", str(args.threads),
                   "-input", str(collection), "-index", str(staging),
                   "-storePositions", "-storeDocvectors", "-storeRaw"]
        subprocess.run(command, check=True)
        if not list(staging.glob("segments_*")):
            raise ValueError("Pyserini did not produce a Lucene index")
        (staging / "source.json").write_text(json.dumps({"source": OFFICIAL_URL,
            "official_archive_md5": OFFICIAL_MD5, "build_command": command,
            "bm25_search": {"k1": 0.82, "b": 0.68}}, indent=2) + "\n", encoding="utf-8")
        staging.rename(final)
        print(f"QReCC Lucene index completed: {final}", flush=True)


if __name__ == "__main__":
    main()
