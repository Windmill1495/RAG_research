"""Compute AdaQR rewards with one evidence hypothesis per conversation turn."""

import math

from adaqr_support.adaqr import answer_prompt, marginal_reward, rewrite_prompt
from evidence_verifier import NLI_LABELS
from rag.compcqr import compose_queries, fuse_rankings
from train_fusion import SETTINGS, candidates_for


def _finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def apply_evidence_reward(baseline, answer, evidence_scores, weight, epsilon):
    """Apply shared evidence scores to the original single and fusion rewards."""
    status = baseline.get("status")
    if status == "component_error":
        return baseline
    if status != "ok":
        raise ValueError(f"Unexpected foundation reward status: {status}")
    if not _finite_number(weight) or weight < 0:
        raise ValueError("Evidence weight must be finite and nonnegative")
    if not _finite_number(epsilon) or not 0 < epsilon <= 1:
        raise ValueError("Evidence epsilon must be in (0, 1]")

    question = baseline["components"].get("mq")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("CompCQR standalone question is required for evidence scoring")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Training reference answer is required for evidence scoring")
    passages = baseline["evidence_passages"]
    if not passages or any(not isinstance(text, str) or not text.strip()
                           for text in passages.values()):
        raise ValueError("Foundation rewards must contain nonempty passage texts")

    texts = list(dict.fromkeys(passages.values()))
    scores = list(evidence_scores(texts, question, answer))
    if len(scores) != len(texts):
        raise ValueError("Expected one NLI result per distinct passage text")
    probabilities_by_text = {}
    for passage, score in zip(texts, scores):
        if not isinstance(score, dict) or any(label not in score for label in NLI_LABELS):
            raise ValueError("Each NLI result must contain all three label probabilities")
        probabilities = {label: score[label] for label in NLI_LABELS}
        if (any(not _finite_number(value) or not 0 <= value <= 1
                for value in probabilities.values())
                or not math.isclose(sum(probabilities.values()), 1, abs_tol=1e-6)):
            raise ValueError("Invalid NLI probability distribution")
        probabilities_by_text[passage] = probabilities
    probabilities_by_id = {key: probabilities_by_text[passage]
                           for key, passage in passages.items()}
    supports = {key: max(0.0, value["entailment"] - value["contradiction"])
                for key, value in probabilities_by_id.items()}

    candidates = []
    for original in baseline["candidates"]:
        candidate = {**original}
        for branch in ("single", "fusion"):
            records = original[f"{branch}_evidence"]
            weights = [record["weight"] for record in records]
            ids = [record["passage_id"] for record in records]
            if (ids != original[f"{branch}_passage_ids"]
                    or any(key not in supports for key in ids)
                    or any(not _finite_number(value) or value < 0 for value in weights)
                    or not math.isclose(sum(weights), 1, abs_tol=1e-8)):
                raise ValueError(f"Invalid stored evidence ranking or weights in the {branch} branch")
            cqa_reward = original.get(f"{branch}_cqa_reward", original[f"{branch}_reward"])
            if not _finite_number(cqa_reward):
                raise ValueError(f"Invalid foundation CQA reward in the {branch} branch")
            penalty = sum(value * math.log(epsilon + (1 - epsilon) * supports[key])
                          for key, value in zip(ids, weights))
            candidate[f"{branch}_cqa_reward"] = cqa_reward
            candidate[f"{branch}_evidence_penalty"] = penalty
            candidate[f"{branch}_reward"] = cqa_reward + weight * penalty
            candidate[f"{branch}_evidence"] = [
                {**record, "support_score": supports[record["passage_id"]]}
                for record in records
            ]
        candidates.append(candidate)
    return {**baseline, "candidates": candidates, "hypothesis_question": question,
            "support_by_passage_id": supports, "nli_probabilities_by_passage_id": probabilities_by_id}


def score_turn(row, *, retriever, components, answer_logp, evidence_scores,
               evidence_lambda, evidence_epsilon, config):
    """Retrieve and score both rankings, then apply the same evidence pass as reuse."""
    candidates = candidates_for(row)
    plans = {
        candidate: compose_queries({**components, "mq": candidate}, config["retriever_backend"],
                                   config["compcqr_max_components"])
        for candidate in candidates
    }
    depth = config["compcqr_retrieval_depth"]
    queries = list(dict.fromkeys([*candidates, *(query for plan in plans.values() for query in plan)]))
    ranks = retriever.search_many(queries, max(depth, SETTINGS.top_k))
    if len(queries) != len(ranks):
        raise ValueError("Search result count differs from the number of distinct queries")
    by_query = dict(zip(queries, ranks))
    text_by_id = {}
    logps_by_id = {}

    def branch(hits):
        if not hits:
            raise ValueError(f"No passages retrieved for training turn {row['id']}")
        ids, logps = [], []
        for hit in hits:
            key = hit.get("passage_id") or hit["id"]
            passage = hit.get("text") or retriever.passage_text(key)
            pair = (key, passage)
            if key in text_by_id and text_by_id[key] != passage:
                raise ValueError(f"Inconsistent evidence text for passage ID {key}")
            text_by_id[key] = passage
            if pair not in logps_by_id:
                prompt = answer_prompt(row["history"], row["question"], passage)
                logps_by_id[pair] = answer_logp(prompt, row["answer"])
            ids.append(key)
            logps.append(logps_by_id[pair])
        cqa_reward = marginal_reward(hits, logps)
        scores = [float(hit["score"]) for hit in hits]
        largest = max(scores)
        shifted = [math.exp(score - largest) for score in scores]
        total = sum(shifted)
        evidence = [{"passage_id": key, "weight": value / total, "answer_logp": logp}
                    for key, value, logp in zip(ids, shifted, logps)]
        return cqa_reward, ids, evidence

    records = []
    for candidate, plan in plans.items():
        single, single_ids, single_evidence = branch(by_query[candidate][:SETTINGS.top_k])
        fused = fuse_rankings([by_query[query][:depth] for query in plan],
                              config["compcqr_rrf_k"])
        multi, multi_ids, multi_evidence = branch(fused[:SETTINGS.top_k])
        records.append({"rewrite": candidate, "single_reward": single, "fusion_reward": multi,
                        "single_passage_ids": single_ids, "fusion_passage_ids": multi_ids,
                        "query_count": len(plan), "single_evidence": single_evidence,
                        "fusion_evidence": multi_evidence})
    baseline = {"id": row["id"], "prompt": rewrite_prompt(row["history"], row["question"]),
                "status": "ok", "components": components, "candidates": records,
                "distinct_queries": len(queries), "scored_passages": len(logps_by_id),
                "evidence_passages": text_by_id}
    return apply_evidence_reward(baseline, row["answer"], evidence_scores,
                                 evidence_lambda, evidence_epsilon)

