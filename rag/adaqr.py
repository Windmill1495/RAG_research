"""AdaQR (Zhang et al., EMNLP 2024): reward and preference construction.

Training and inference adapters live separately so importing this module never loads a model.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable


REWRITE_INSTRUCTION = (
    "Given a question and its context, decontextualize the question by addressing "
    "coreference and omission issues. The resulting question should retain its "
    "original meaning and be as informative as possible, and should not duplicate "
    "any previously asked questions in the context."
)


@dataclass(frozen=True)
class AdaQRSettings:
    candidates: int = 3
    top_k: int = 5
    temperature: float = 1.0
    delta: float = 0.1
    beta: float = 0.1

    def __post_init__(self):
        if self.candidates < 2 or self.top_k < 1 or self.temperature <= 0 or self.delta <= 0 or self.beta <= 0:
            raise ValueError("Invalid AdaQR hyperparameters")


def conversation_text(history: list[dict]) -> str:
    return " ".join(f"{'Q' if message['role'] == 'user' else 'A'}: {message['content']}"
                    for message in history)


def rewrite_prompt(history: list[dict], question: str) -> str:
    return (f"{REWRITE_INSTRUCTION}\nContext: {conversation_text(history)}\n"
            f"Question: {question}\nRewrite:")


def answer_prompt(history: list[dict], question: str, passage: str) -> str:
    # The original conversation, never a candidate rewrite, conditions the answer model.
    return (f"Context: {conversation_text(history)}\nQuestion: {question}\n"
            f"Passage: {passage}\nAnswer:")


def marginal_reward(hits: list[dict], log_probabilities: list[float]) -> float:
    """Equation (3): softmax(retrieval scores) weighted *log* answer likelihood.

    The paper calls this a marginal probability, but its printed equation is an
    expectation of log probabilities. Keep that equation, not logsumexp.
    """
    if not hits or len(hits) != len(log_probabilities):
        raise ValueError("Each retrieved passage needs one answer log probability")
    scores = [float(hit["score"]) for hit in hits]
    if any(not math.isfinite(value) for value in scores + log_probabilities):
        raise ValueError("Retrieval scores and log probabilities must be finite")
    maximum = max(scores)
    weights = [math.exp(value - maximum) for value in scores]
    total = sum(weights)
    return sum(weight / total * value for weight, value in zip(weights, log_probabilities))


def preference_pairs(candidates: list[dict], delta: float = 0.1) -> list[dict]:
    if delta <= 0:
        raise ValueError("delta must be positive")
    pairs = []
    for i, first in enumerate(candidates):
        for second in candidates[i + 1:]:
            gap = first["reward"] - second["reward"]
            if abs(gap) > delta and first["rewrite"] != second["rewrite"]:
                winner, loser = (first, second) if gap > 0 else (second, first)
                pairs.append({"chosen": winner["rewrite"], "rejected": loser["rewrite"],
                              "reward_gap": abs(gap)})
    return pairs


def collect_preferences(example: dict, sample: Callable, search: Callable,
                        answer_log_probability: Callable, settings: AdaQRSettings = AdaQRSettings()) -> dict:
    """Self-sample, retrieve, score, and form every qualifying pair for one turn."""
    history, question, answer = example["history"], example["question"], example["answer"]
    if not answer.strip() or not history:
        raise ValueError("AdaQR preference training needs a later turn with a gold answer")
    prompt = rewrite_prompt(history, question)
    rewrites = sample(prompt, settings.candidates, settings.temperature)
    if len(rewrites) != settings.candidates or any(not isinstance(r, str) or not r.strip() for r in rewrites):
        raise ValueError("SFT model must return exactly three nonempty sampled rewrites")
    candidates = []
    for rewrite in rewrites:
        hits = search(rewrite.strip(), settings.top_k)
        if not hits:
            raise ValueError("Retriever returned no passages; cannot compute AdaQR reward")
        probabilities = [answer_log_probability(answer_prompt(history, question, hit["text"]), answer)
                         for hit in hits]
        candidates.append({"rewrite": rewrite.strip(), "reward": marginal_reward(hits, probabilities),
                           "passage_ids": [hit.get("passage_id", hit.get("id")) for hit in hits]})
    pairs = [{"id": example["id"], "prompt": prompt, **pair}
             for pair in preference_pairs(candidates, settings.delta)]
    return {"id": example["id"], "candidates": candidates, "pairs": pairs}
