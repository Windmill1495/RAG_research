"""Collect shared retrieval inputs and add an optional NLI evidence reward."""

import math
from pathlib import Path

from adaqr_support.adaqr import answer_prompt, marginal_reward, rewrite_prompt
from rag.compcqr import compose_queries, fuse_rankings
from train_fusion import SETTINGS, candidates_for


NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
NLI_REVISION = "6f5cf0a2b59cabb106aca4c287eed12e357e90eb"
NLI_REQUIRED = ("config.json", "model.safetensors", "spm.model", "tokenizer.json",
                "tokenizer_config.json")
HYPOTHESIS_VERSION = "same_turn_compcqr_standalone_question_plus_training_answer_v1"


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


def fetch_model(directory):
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=NLI_MODEL, revision=NLI_REVISION, local_dir=directory,
                      allow_patterns=[*NLI_REQUIRED, "added_tokens.json",
                                      "special_tokens_map.json", "README.md"])
    validate_model(directory)


def validate_model(directory):
    directory = Path(directory)
    missing = [name for name in NLI_REQUIRED if not (directory / name).is_file()
               or not (directory / name).stat().st_size]
    if missing:
        raise ValueError(f"Evidence scorer files missing at {directory}: {', '.join(missing)}")


class NliScorer:
    """Frozen NLI model. One turn uses the same hypothesis for all candidate rewrites."""

    def __init__(self, model_dir, batch_size=16):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        validate_model(model_dir)
        self.torch = torch
        self.batch_size = batch_size
        self.device = torch.device("cuda:0")
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir), local_files_only=True, use_safetensors=True).to(self.device).eval()
        labels = {name.strip().lower(): int(index) for index, name in self.model.config.id2label.items()}
        self.entailment = labels["entailment"]
        self.contradiction = labels["contradiction"]

    def __call__(self, passages, question, reference_answer):
        hypothesis = f'The answer to the question "{question}" is "{reference_answer}".'
        ids = self.tokenizer.encode(hypothesis, add_special_tokens=False)
        truncated = len(ids) > 240
        if truncated:
            hypothesis = self.tokenizer.decode(ids[:240], skip_special_tokens=True)
        supports = {}
        items = list(passages.items())
        with self.torch.inference_mode():
            for start in range(0, len(items), self.batch_size):
                batch = items[start:start + self.batch_size]
                encoded = self.tokenizer([text for _, text in batch], [hypothesis] * len(batch),
                                         truncation="only_first", max_length=512,
                                         padding=True, return_tensors="pt").to(self.device)
                probabilities = self.model(**encoded).logits.float().softmax(dim=-1)
                values = (probabilities[:, self.entailment] -
                          probabilities[:, self.contradiction]).clamp_min(0).cpu().tolist()
                supports.update((key, value) for (key, _), value in zip(batch, values))
        return supports, truncated


def evidence_adjusted(row, baseline, scorer, weight, epsilon):
    if row["id"] != baseline["id"]:
        raise ValueError("Evidence scoring input ID mismatch")
    if baseline["status"] != "ok":
        return baseline
    question = baseline["components"]["mq"]
    if not question.strip():
        raise ValueError("Successful CompCQR record requires a standalone hypothesis question")
    supports, truncated = scorer(baseline["evidence_passages"], question, row["answer"])
    if (supports.keys() != baseline["evidence_passages"].keys()
            or any(not math.isfinite(value) or not 0 <= value <= 1 for value in supports.values())):
        raise ValueError("Invalid evidence support distribution")

    def adjust(candidate, branch):
        records = candidate[f"{branch}_evidence"]
        if ([record["passage_id"] for record in records] != candidate[f"{branch}_passage_ids"]
                or not math.isclose(sum(record["weight"] for record in records), 1.0, abs_tol=1e-8)):
            raise ValueError("Stored evidence weights differ from the corresponding ranking")
        penalty = sum(record["weight"] * math.log(
            epsilon + (1 - epsilon) * supports[record["passage_id"]]) for record in records)
        return candidate[f"{branch}_reward"] + weight * penalty

    candidates = []
    for base in baseline["candidates"]:
        candidate = {key: value for key, value in base.items() if not key.endswith("_evidence")}
        for branch in ("single", "fusion"):
            candidate[f"{branch}_baseline_reward"] = base[f"{branch}_reward"]
            candidate[f"{branch}_reward"] = adjust(base, branch)
        candidates.append(candidate)
    result = {key: value for key, value in baseline.items() if key != "evidence_passages"}
    result.update(candidates=candidates, support_by_passage_id=supports,
                  hypothesis_truncated=truncated)
    return result
