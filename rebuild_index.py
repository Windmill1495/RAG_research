import argparse
from pathlib import Path
from rag.config import load_config, offline_config
from rag.llm import runtime_embedding_identity
from rag.pipeline import RAGEngine

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="重建文档索引")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--mode", choices=["demo", "ollama"], default="ollama")
    parser.add_argument("--dataset", choices=["qrecc"], default="qrecc")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    config = load_config(root, profile=args.dataset)
    if args.mode == "demo":
        config = offline_config(config)
    identity = runtime_embedding_identity(config)
    if identity:
        config["embedding_digest"] = identity["digest"]
    engine = RAGEngine(root, config, args.mode, build_index=False)
    print(engine.retriever.build(force=args.force))
