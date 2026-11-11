"""Command line conversation: python chat.py --mode ollama --strategy rewrite."""
import argparse
from pathlib import Path

from rag.config import load_config, offline_config
from rag.pipeline import RAGEngine, STRATEGIES


def main():
    parser = argparse.ArgumentParser(description="简单多轮RAG命令行")
    parser.add_argument("--mode", choices=["demo", "ollama"], default="ollama")
    parser.add_argument("--dataset", choices=["qrecc"], default="qrecc")
    parser.add_argument("--strategy", choices=STRATEGIES, default="rewrite")
    parser.add_argument("--top-k", type=int)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    config = load_config(root, profile=args.dataset)
    engine = RAGEngine(root, offline_config(config) if args.mode == "demo" else config, args.mode)
    print(f"模式: {args.mode}；/clear 清空历史，/quit 退出。")
    if args.mode == "demo":
        print("离线模式仅摘录证据，查询重写为启发式规则，未调用大模型。")
    history = []
    while True:
        try:
            question = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question == "/quit":
            break
        if question == "/clear":
            history.clear()
            print("已清空历史。")
            continue
        if not question:
            continue
        try:
            result = engine.answer(question, history, args.strategy, args.top_k)
        except Exception as error:
            print(f"运行失败: {error}")
            continue
        print(f"检索Query: {result['search_query']}\n改写来源: {result['rewrite_source']}")
        print(result["answer"])
        print(f"耗时: {result['latencies_ms']}")
        for warning in result["warnings"]:
            print(f"提示: {warning}")
        history.extend([{"role": "user", "content": question}, {"role": "assistant", "content": result["answer"]}])
        history = history[-engine.config["history_turns"]*2:]


if __name__ == "__main__":
    main()
