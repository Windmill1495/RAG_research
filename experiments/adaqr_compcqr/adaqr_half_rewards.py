"""Compute the two AdaQR rewards and retain their passage scores for auditing."""

import math

from adaqr_support.adaqr import answer_prompt, marginal_reward, rewrite_prompt
from rag.compcqr import compose_queries, fuse_rankings
from train_fusion import SETTINGS, candidates_for


def score_turn(row, *, retriever, components, answer_logp, config):
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
        reward = marginal_reward(hits, logps)
        scores = [float(hit["score"]) for hit in hits]
        largest = max(scores)
        shifted = [math.exp(score - largest) for score in scores]
        total = sum(shifted)
        evidence = [{"passage_id": key, "weight": weight / total, "answer_logp": logp}
                    for key, weight, logp in zip(ids, shifted, logps)]
        return reward, ids, evidence

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
    return {"id": row["id"], "prompt": rewrite_prompt(row["history"], row["question"]),
            "status": "ok", "components": components, "candidates": records,
            "distinct_queries": len(queries), "scored_passages": len(logps_by_id),
            "evidence_passages": text_by_id}

