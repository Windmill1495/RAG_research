"""CompCQR query composition, retrieval consensus and passage selection.

Uses the Baseline retriever and Ollama client; no second index or retrieval path.
"""
import json
from collections import defaultdict

from .compcqr_prompts import (
    FOLLOWUP_SYSTEM_PROMPT, SYSTEM_ASSISTANT, SYSTEM_SELECTOR,
    build_followup_prompt, build_rewrite_prompt, build_selection_prompt,
    load_demonstrations, parse_rewrite_outputs, parse_selected_passage, repair_json,
)
from .compcqr_variations import build_variations

COMPONENTS = ("oq", "mq", "ma", "mq2", "ma2")


def retrieve(engine, question, history, top_k):
    """Run the paper's three LLM calls and 325 ordered retrieval variations."""
    if engine.mode != "ollama":
        raise ValueError("CompCQR 需要 Ollama 模式")
    context = "\n".join(f"{m['role']}: {m['content']}" for m in history)
    demo_path = engine.root / "prompts" / "demonstrations.json"
    if not demo_path.is_file():
        raise FileNotFoundError(f"CompCQR few-shot demonstrations 缺失: {demo_path}")
    demo = load_demonstrations(demo_path)
    # 第一次调用 LLM
    rewrite_raw = engine.client.generate(
        build_rewrite_prompt(f"Context: {context}\nCurrent Question: {question}", demo),
        system=SYSTEM_ASSISTANT, temperature=0.7)
    rewrites, responses, _ = parse_rewrite_outputs([rewrite_raw])
    if not rewrites[0] or not responses[0]:
        raise ValueError(f"CompCQR Step 0 输出无法解析: {rewrite_raw[:300]}")
    # 第二次调用 LLM
    followup_raw = engine.client.generate(
        build_followup_prompt(context, question),
        system=FOLLOWUP_SYSTEM_PROMPT, temperature=0.7)
    try:
        followup = json.loads(repair_json(followup_raw))
        mq2, ma2 = followup["followup_query"], followup["next_answer"]
        if not isinstance(mq2, str) or not isinstance(ma2, str):
            raise TypeError("follow-up fields must be strings")
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"CompCQR Step 1 输出无法解析: {followup_raw[:300]}") from exc
    # 构造 325 种查询组合
    parts = {"oq": question, "mq": rewrites[0], "ma": responses[0],
             "mq2": mq2, "ma2": ma2}
    variations = build_variations(parts, COMPONENTS, 5, "+")
    # 执行 325 次检索
    ranks = defaultdict(list)
    documents = {}
    searched = 0
    for variation in variations:
        query = variation["text"]
        if not query.strip("+ "):
            continue  # The paper permits an empty follow-up when none is relevant.
        searched += 1
        hits = engine.retriever.search(query, 100)
        for rank, hit in enumerate(hits, 1):
            key = hit["id"]
            ranks[key].append(rank)
            documents[key] = hit
    if not ranks:
        raise ValueError("CompCQR Step 3 未检索到候选文档")
    # RRF 融合排序
    scores = {key: sum(1 / rank for rank in positions)
              for key, positions in ranks.items()}
    ordered = sorted(scores, key=lambda key: (-scores[key], key))
    candidates = ordered[:top_k]
    # 第三次调用 LLM
    selection_raw = engine.client.generate(
        build_selection_prompt(question, context,
                               [documents[key]["text"] for key in candidates], len(candidates)),
        system=SYSTEM_SELECTOR, temperature=0.7)
    selected = parse_selected_passage(selection_raw, 1, len(candidates))
    if selected is None:
        raise ValueError(f"CompCQR Step 5 选段无法解析: {selection_raw[:300]}")
    chosen = candidates[selected - 1]
    final_keys = [chosen] + [key for key in ordered if key != chosen][:top_k - 1]
    hits = [{**documents[key], "score": round(scores[key], 8)} for key in final_keys]
    return hits, {"components": parts, "variation_count": len(variations),
                  "searched_variation_count": searched,
                  "candidate_count": len(scores), "selected_passage_id": documents[chosen].get("passage_id"),
                  "selected_chunk_id": chosen, "selected_number": selected,
                  "selection_raw": selection_raw}
