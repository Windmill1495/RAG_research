"""Small documented metrics; lexical overlap is not semantic correctness."""
from __future__ import annotations

import math
import re
from collections import Counter

NORMALIZATION = "lowercase ASCII words/numbers and individual CJK characters; no stemming, stop-word or article removal"


def metric_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+|[\u3400-\u9fff]", text.casefold())


def rouge_scores(prediction: str, reference: str, *, include_l: bool = False) -> dict[str, float]:
    pred, gold = metric_tokens(prediction), metric_tokens(reference)
    overlap = sum((Counter(pred) & Counter(gold)).values())
    p = overlap / len(pred) if pred else 0.0
    r = overlap / len(gold) if gold else 0.0
    result = {"rouge1_precision": p, "rouge1_recall": r,
              "rouge1_f1": 2 * p * r / (p + r) if p + r else 0.0}
    if include_l:
        previous = [0] * (len(gold) + 1)
        for token in pred:
            current = [0]
            for j, expected in enumerate(gold, 1):
                current.append(previous[j - 1] + 1 if token == expected else max(previous[j], current[-1]))
            previous = current
        result["rougeL_f1"] = 2 * previous[-1] / (len(pred) + len(gold)) if pred or gold else 0.0
    return result


def document_metrics(hits: list[dict], gold_document: str, top_k: int) -> tuple[float, float]:
    for rank, hit in enumerate(hits[:top_k], 1):
        filename = str(hit.get("document", "")).replace("\\", "/").rsplit("/", 1)[-1]
        if filename == gold_document:
            return 1.0, 1.0 / rank
    return 0.0, 0.0


def passage_metrics(hits: list[dict], positive_ids: list[str], top_k: int) -> dict[str, float]:
    """Original result ranks; duplicate gold IDs contribute relevance only once."""
    if top_k < 1 or not positive_ids:
        raise ValueError("top_k and positive_ids must be nonempty")
    gold = set(positive_ids)
    seen, relevant_ranks = set(), []
    for rank, hit in enumerate(hits[:top_k], 1):
        pid = hit.get("passage_id")
        if not isinstance(pid, str) or not pid:
            raise ValueError("QReCC retrieval result lacks its original passage_id")
        if pid in gold and pid not in seen:
            relevant_ranks.append(rank)
            seen.add(pid)
    dcg = sum(1 / math.log2(rank + 1) for rank in relevant_ranks)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(gold), top_k) + 1))
    return {"hit_at_k": float(bool(relevant_ranks)),
            "gold_recall_at_k": len(seen) / len(gold),
            "mrr_at_k": 1 / relevant_ranks[0] if relevant_ranks else 0.0,
            "ndcg_at_k": dcg / ideal if ideal else 0.0}
