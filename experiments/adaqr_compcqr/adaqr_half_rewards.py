"""Compute the two AdaQR rewards and retain their passage scores for auditing."""

import math

from adaqr_support.adaqr import answer_prompt, marginal_reward, rewrite_prompt
from rag.compcqr import compose_queries, fuse_rankings
from train_fusion import SETTINGS, candidates_for


def score_turn(row, *, retriever, components, answer_logp, evidence_scores,
               evidence_lambda, evidence_epsilon, config):
    """Use the baseline's two rankings while saving their evidence and weights."""
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
    supports_by_candidate = {}

    def branch(hits, candidate):
        if not hits:
            raise ValueError(f"No passages retrieved for training turn {row['id']}")
        ids, logps, support_keys = [], [], []
        pending = {}
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
            support_key = (candidate, pair)
            if support_key not in supports_by_candidate:
                pending[support_key] = passage
            ids.append(key)
            logps.append(logps_by_id[pair])
            support_keys.append(support_key)
        if pending:
            supports_by_candidate.update(zip(
                pending, evidence_scores(list(pending.values()), candidate, row["answer"])))
        supports = [supports_by_candidate[key] for key in support_keys]
        cqa_reward = marginal_reward(hits, logps)
        scores = [float(hit["score"]) for hit in hits]
        largest = max(scores)
        shifted = [math.exp(score - largest) for score in scores]
        total = sum(shifted)
        evidence_penalty = sum(weight / total * math.log(evidence_epsilon + (1 - evidence_epsilon) * support)
                               for weight, support in zip(shifted, supports))
        evidence = [{"passage_id": key, "weight": weight / total, "answer_logp": logp,
                     "support_score": support}
                    for key, weight, logp, support in zip(ids, shifted, logps, supports)]
        return cqa_reward + evidence_lambda * evidence_penalty, cqa_reward, evidence_penalty, ids, evidence

    records = []
    for candidate, plan in plans.items():
        single, single_cqa, single_penalty, single_ids, single_evidence = branch(
            by_query[candidate][:SETTINGS.top_k], candidate)
        fused = fuse_rankings([by_query[query][:depth] for query in plan],
                              config["compcqr_rrf_k"])
        multi, multi_cqa, multi_penalty, multi_ids, multi_evidence = branch(
            fused[:SETTINGS.top_k], candidate)
        records.append({"rewrite": candidate, "single_reward": single, "fusion_reward": multi,
                        "single_cqa_reward": single_cqa, "fusion_cqa_reward": multi_cqa,
                        "single_evidence_penalty": single_penalty,
                        "fusion_evidence_penalty": multi_penalty,
                        "single_passage_ids": single_ids, "fusion_passage_ids": multi_ids,
                        "query_count": len(plan), "single_evidence": single_evidence,
                        "fusion_evidence": multi_evidence})
    return {"id": row["id"], "prompt": rewrite_prompt(row["history"], row["question"]),
            "status": "ok", "components": components, "candidates": records,
            "distinct_queries": len(queries), "scored_passages": len(logps_by_id),
            "evidence_passages": text_by_id}

