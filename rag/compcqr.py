"""CompCQR adaptation: atomic components -> queries -> RRF -> best passage.

The paper's three-call algorithm is adapted to the project's Ollama client and
BM25/dense/hybrid retrievers. Generated answers are hypothetical query expansion,
never reference evidence. All functions accept only public conversational inputs.
"""
from __future__ import annotations

import itertools
import json
import math
import re
import time

from .rewriters import clean_query, heuristic_rewrite, parse_json_object, validation_issues
from .run_cache import make_signature


from .compcqr_prompts import original_rewrite_prompt, followup_prompt, selection_prompt, parse_rar

PROMPT_VERSION = "compcqr-paper-J-llm4cs-rar-cot-v2"
COMPONENT_NAMES = ("oq", "mq", "ma", "mq2", "ma2")


def compose_queries(components: dict, backend: str = "hybrid", max_components: int = 5) -> list[str]:
    """Enumerate stable unique nonempty sequences; BM25 ignores component order."""
    if backend not in {"bm25", "dense", "hybrid"}:
        raise ValueError("Unknown CompCQR retriever backend")
    if type(max_components) is not int or not 1 <= max_components <= 5:
        raise ValueError("compcqr_max_components must be between 1 and 5")
    atoms = list(dict.fromkeys(components.get(name, "").strip()
                              for name in COMPONENT_NAMES if components.get(name, "").strip()))
    combinations = itertools.combinations if backend == "bm25" else itertools.permutations
    queries, seen = [], set()
    for size in range(1, min(max_components, len(atoms)) + 1):
        for values in combinations(atoms, size):
            query = "\n".join(values)
            if query not in seen:
                seen.add(query)
                queries.append(query)
    return queries


ANSWER_TERM = re.compile(r"\b(?:[12]\d{3}|[A-Z]{2,}|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")


def supported_answer_queries(components: dict, question: str, history: list[dict],
                             passages_by_query: dict[str, list[str]], backend: str,
                             max_components: int) -> tuple[list[str], dict]:
    """Restrict unsupported answer terms to one intact paired query."""
    filtered, removed = components.copy(), {}
    for query_name, answer_name in (("mq", "ma"), ("mq2", "ma2")):
        answer = components.get(answer_name, "")
        if not answer or not components.get(query_name):
            continue
        source = "\n".join([question, *(item["content"] for item in history),
                            *passages_by_query[query_name][:5]])
        terms = []
        for match in ANSWER_TERM.finditer(answer):
            term = match.group()
            if match.start() == 0 and term.istitle() and " " not in term:
                continue  # Sentence-initial capitalization alone is not a name.
            pattern = r"(?<!\w)" + re.escape(term).replace(r"\ ", r"\s+") + r"(?!\w)"
            if not re.search(pattern, source, re.I):
                terms.append(term)
        if terms:
            removed[answer_name] = terms
            for term in terms:
                pattern = r"(?<!\w)" + re.escape(term).replace(r"\ ", r"\s+") + r"(?!\w)"
                answer = re.sub(pattern, "", answer)
            filtered[answer_name] = " ".join(answer.split())
    queries = compose_queries(filtered, backend, max_components)
    for query_name, answer_name in (("mq", "ma"), ("mq2", "ma2")):
        if answer_name in removed and components.get(query_name):
            queries.append(components[query_name] + "\n" + components[answer_name])
    return list(dict.fromkeys(queries)), removed


def _answer(value, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError("Component answer must be a string")
    value = " ".join(value.split())
    if not value and not allow_empty:
        raise ValueError("Component answer is empty")
    return value


def parse_followup_response(raw: str) -> tuple[dict, bool]:
    """Read original J.2 fields; tolerate only missing separators, never invent text."""
    if not isinstance(raw, str):
        raise ValueError("Follow-up output must be text")
    text = raw.strip()
    if text in {"", '""'}:
        return {"followup_query": "", "next_answer": ""}, False
    obj = parse_json_object(text) or parse_json_object("{" + text + "}")
    if obj is not None:
        if not {"followup_query", "next_answer"} <= obj.keys():
            raise ValueError("Both original follow-up fields are required")
        return obj, False
    # Some small models emit the paper's quoted fields on adjacent lines but
    # omit their comma. Decode each existing JSON string without changing it.
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I).strip()
    if text.startswith("{") and text.endswith("}"):
        text = text[1:-1].strip()
    decoder, values = json.JSONDecoder(), {}
    while text:
        key, end = decoder.raw_decode(text)
        if key not in {"followup_query", "next_answer"} or key in values:
            raise ValueError("Unexpected or duplicate follow-up field")
        text = text[end:].lstrip()
        if not text.startswith(":"):
            raise ValueError("Missing follow-up field colon")
        value, end = decoder.raw_decode(text[1:].lstrip())
        if not isinstance(value, str):
            raise ValueError("Follow-up values must be JSON strings")
        values[key] = value
        text = text[1:].lstrip()[end:].lstrip()
        if text.startswith(","):
            text = text[1:].lstrip()
    if set(values) != {"followup_query", "next_answer"}:
        raise ValueError("Both original follow-up fields are required")
    return values, True


def build_plan(question: str, history: list[dict], *, client, mode: str, config: dict | None = None) -> dict:
    """Generate components with two calls; failures remain explicit in source/trace."""
    config = config or {}
    components = dict.fromkeys(COMPONENT_NAMES, "")
    components["oq"] = question
    trace = {"strategy": "compcqr", "prompt_version": PROMPT_VERSION,
             "history_signature": make_signature(history), "visible_history_messages": len(history),
             "llm_calls": 0, "raw_responses": [], "fallback": False, "repaired": False,
             "component_errors": [], "stage_errors": [], "selection_error": None,
             "validation_policy": "heuristic rewrite checks only; generated answers are unverified",
             "adaptation": "Ollama + project retriever; not the original ANCE benchmark setting"}
    source, error = "compcqr_llm", None

    def call(prompt):
        trace["llm_calls"] += 1
        raw = client.generate(prompt, system="", temperature=0.0)
        trace["raw_responses"].append(raw)
        return raw

    def failure(stage, exception):
        message = f"{stage}: {type(exception).__name__}: {exception}"
        trace["component_errors"].append(message)
        trace["stage_errors"].append({"stage": stage, "error": message})
        trace["fallback"] = True
        return message

    if mode == "demo":
        components["mq"] = heuristic_rewrite(question, history)
        source = "compcqr_demo"
        trace["demo_note"] = "No LLM called; no hypothetical answers or LLM selection. Not a CompCQR model result."
    else:
        public_input = {"history": history, "current_question": question}
        try:
            cot, original_rewrite, original_answer = parse_rar(call(original_rewrite_prompt(question, history)))
            trace["rewrite_analysis"] = cot
            rewrite = clean_query(original_rewrite)
            issues = validation_issues(rewrite, question, history)
            trace["validation_issues"] = issues
            if issues:
                raise ValueError(", ".join(issues))
            answer = _answer(original_answer)
            components.update(mq=rewrite, ma=answer)
        except Exception as exc:
            error = failure("rewrite_answer", exc)
            components["mq"] = heuristic_rewrite(question, history)
            source = "compcqr_fallback"
        if not error:
            try:
                raw = call(followup_prompt(question, history))
                obj, format_repaired = parse_followup_response(raw)
                trace["followup_format_repaired"] = format_repaired
                proposed = obj.get("followup_query", obj.get("mq2"))
                query = clean_query(proposed)
                answer = _answer(obj.get("next_answer", obj.get("ma2")), allow_empty=True)
                if not isinstance(proposed, str) or (proposed.strip() and not query):
                    raise ValueError("Invalid follow-up query")
                if len(query) > 1200 or bool(query) != bool(answer):
                    raise ValueError("Follow-up query and answer must be a valid nonempty pair or both empty")
                components.update(mq2=query, ma2=answer)
            except Exception as exc:
                error = failure("followup", exc)
                source = "compcqr_partial"
    queries = compose_queries(components, config.get("retriever_backend", "hybrid"),
                              config.get("compcqr_max_components", 5))
    trace.update(components=components, query_count=len(queries), queries=queries,
                 composition_mode="subsets" if config.get("retriever_backend", "hybrid") == "bm25" else "ordered_permutations",
                 max_components=config.get("compcqr_max_components", 5),
                 nonempty_components=sum(bool(value) for value in components.values()),
                 hypothetical_answers_are_evidence=False)
    return {"query": components["mq"] or question, "queries": queries, "components": components,
            "source": source, "error": error, "trace": trace}


def _passage_key(hit: dict) -> str:
    value = hit.get("passage_id") or hit.get("id")
    if not isinstance(value, str) or not value:
        raise ValueError("CompCQR fusion requires passage_id or id")
    return value


def fuse_rankings(rankings: list[list[dict]], rrf_k: float = 0) -> list[dict]:
    """Aggregate unique passages per query, preserving raw ranks and deterministic ties."""
    if (type(rrf_k) not in {int, float} or not math.isfinite(rrf_k) or rrf_k < 0):
        raise ValueError("CompCQR RRF constant must be a finite nonnegative number")
    scores, frequencies, originals = {}, {}, {}
    for ranking in rankings:
        seen = set()
        for rank, hit in enumerate(ranking, 1):
            key = _passage_key(hit)
            if key in seen:
                continue
            seen.add(key)
            originals.setdefault(key, hit)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
            frequencies[key] = frequencies.get(key, 0) + 1
    order = sorted(scores, key=lambda key: (-scores[key], -frequencies[key], key))
    return [{**originals[key], "score": scores[key], "rrf_score": scores[key],
             "query_frequency": frequencies[key], "pre_selection_rank": rank}
            for rank, key in enumerate(order, 1)]


def _selection_prompt(question: str, history: list[dict], passages: list[dict]) -> str:
    return selection_prompt(question, history, passages)


def _selection_passages(hits: list[dict], config: dict, question: str, history: list[dict]) -> list[dict]:
    """Budget the complete prompt, preserving the same question and history window."""
    passages = []
    budget = config.get("compcqr_selection_char_budget", 6000)
    for hit in hits[:config.get("compcqr_selection_top_k", 5)]:
        item = {"number": len(passages) + 1, "passage_id": _passage_key(hit),
                "text": str(hit.get("text", ""))[:config.get("compcqr_passage_chars", 700)]}
        if len(_selection_prompt(question, history, [*passages, item])) > budget:
            break
        passages.append(item)
    return passages


def retrieve_plan(question: str, history: list[dict], queries: list[str], *, retriever,
                  client, mode: str, config: dict, trace: dict) -> dict:
    """Execute all queries, fuse by passage, then promote one valid LLM choice."""
    trace = {**trace, "stage_errors": list(trace.get("stage_errors", [])),
             "raw_responses": list(trace.get("raw_responses", []))}
    started = time.perf_counter()
    depth = config.get("compcqr_retrieval_depth", 100)
    removed = {}
    components = trace.get("components", {})
    if config.get("retriever_backend") == "bm25" and (components.get("ma") or components.get("ma2")):
        initial = list(dict.fromkeys(components[name] for name, answer_name in
            (("mq", "ma"), ("mq2", "ma2")) if components.get(name) and components.get(answer_name)))
        by_query = dict(zip(initial, retriever.search_many(initial, depth)))
        passages_by_query = {
            name: [hit.get("text") or retriever.passage_text(_passage_key(hit))
                   for hit in by_query[components[name]][:5]]
            for name, answer_name in (("mq", "ma"), ("mq2", "ma2"))
            if components.get(name) and components.get(answer_name)
        }
        queries, removed = supported_answer_queries(components, question, history,
            passages_by_query, "bm25", config.get("compcqr_max_components", 5))
        remaining = [query for query in queries if query not in by_query]
        by_query.update(zip(remaining, retriever.search_many(remaining, depth)))
        rankings = [by_query[query] for query in queries]
    else:
        rankings = retriever.search_many(queries, depth)
    retrieved = time.perf_counter()
    hits = fuse_rankings(rankings, config.get("compcqr_rrf_k", 0))[:config.get("compcqr_retrieval_depth", 100)]
    fused = time.perf_counter()
    trace.update(query_count=len(queries), queries=queries, retrieved_lists=len(rankings),
                 unsupported_answer_terms=removed,
                 fused_passage_count=len(hits), selection_error=None, selected_passage_id=None,
                 cross_query_rrf_k=config.get("compcqr_rrf_k", 0),
                 selection_applied=False, selection_candidate_count=0)
    candidates = _selection_passages(hits, config, question, history)
    trace["selection_candidate_count"] = len(candidates)
    trace["selection_candidates"] = candidates
    if mode == "demo":
        trace["selection_skipped"] = "demo_no_llm"
    elif not config.get("compcqr_select", True):
        trace["selection_skipped"] = "disabled_by_config"
    elif not candidates:
        trace["selection_skipped"] = "no_candidates_within_budget" if hits else "no_retrieved_passages"
        if hits:
            message = "Complete selector prompt and one passage exceed compcqr_selection_char_budget"
            trace.update(selection_error=message, fallback=True)
            trace["stage_errors"].append({"stage": "selection", "error": message})
    else:
        try:
            prompt = _selection_prompt(question, history, candidates)
            trace["selection_prompt_characters"] = len(prompt)
            trace["llm_calls"] = trace.get("llm_calls", 0) + 1
            raw = client.generate(prompt, system="",
                                  temperature=0.0)
            trace["raw_responses"].append(raw)
            obj = parse_json_object(raw) or {}
            number = obj.get("selected_passage_number")
            if isinstance(number, str) and re.fullmatch(r"[1-9][0-9]*", number.strip()):
                number = int(number.strip())
            if type(number) is not int or not 1 <= number <= len(candidates):
                raise ValueError("Selection must be a supplied passage number")
            chosen = hits[number - 1]
            hits = [chosen, *hits[:number - 1], *hits[number:]]
            trace.update(selected_passage_id=_passage_key(chosen), selection_applied=True,
                         selection_reason=str(obj.get("reason for selection", "")),
                         query_analysis=obj.get("query analysis"), passage_analysis=obj.get("passage analysis"))
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            trace.update(selection_error=message, fallback=True)
            trace["stage_errors"].append({"stage": "selection", "error": message})
            trace["selection_skipped"] = "invalid_or_failed_selection_keep_rrf"
    finished = time.perf_counter()
    return {"hits": hits, "trace": trace,
            "latencies_ms": {"retrieval": round((retrieved - started) * 1000, 2),
                             "fusion": round((fused - retrieved) * 1000, 2),
                             "selection": round((finished - fused) * 1000, 2)}}
