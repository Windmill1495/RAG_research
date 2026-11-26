"""Recover explicitly labelled CompCQR fields from model formatting variants."""

import json
import re

from rag.compcqr import build_plan, parse_followup_response
from rag.compcqr_prompts import parse_rar


RAR_INSTRUCTION = "For an information-seeking dialog"
FOLLOWUP_INSTRUCTION = "Generate a follow-up query and its answer"
RAR_SEPARATOR = "So the question should be rewritten as:"
FENCE = re.compile(r"\A```(?:[\w-]+)?[ \t]*\n(.*?)\n[ \t]*```\Z", re.S)
NO_FOLLOWUP = re.compile(
    r"no (?:follow[- ]?up|further) (?:(?:question|query) )?(?:is )?(?:needed|required|necessary)\.?\Z",
    re.I,
)


def section_pattern(names):
    return re.compile(
        rf"^[ \t]*(?:#{{1,6}}[ \t]+|[-*][ \t]+)?(?:\*\*|__)?[\"']?"
        rf"(?P<label>{names})[\"']?(?:\*\*|__)?[ \t]*:[ \t]*(?:\*\*|__)?[ \t]*",
        re.I | re.M,
    )


RAR_LABEL = section_pattern(r"rewrite|response")
FOLLOWUP_LABEL = section_pattern(r"followup_query|follow[- ]?up query|next_answer|next answer")
RAR_MARKER = re.compile(re.escape(RAR_SEPARATOR), re.I)


def unfence(raw):
    text = raw.strip()
    match = FENCE.fullmatch(text)
    return match.group(1).strip() if match else text


def recover_rar(raw):
    """Preserve the rewrite, explanation, and response from unique labelled sections."""
    text = unfence(raw)
    fields = list(RAR_LABEL.finditer(text))
    rewrites = [field for field in fields if field["label"].casefold() == "rewrite"]
    if len(rewrites) > 1:
        return None
    start = rewrites[0].end() if rewrites else 0
    markers = list(RAR_MARKER.finditer(text, start))
    if len(markers) != 1:
        return None
    marker = markers[0]
    responses = [field for field in fields
                 if field["label"].casefold() == "response" and field.start() >= marker.end()]
    if len(responses) != 1:
        return None
    response_label = responses[0]
    explanation = " ".join(text[start:marker.start()].split())
    rewrite = " ".join(text[marker.end():response_label.start()].split())
    response = text[response_label.end():].strip()
    if not rewrite or not response:
        return None
    return f"Rewrite: {explanation} {RAR_SEPARATOR} {rewrite}\nResponse: {response}"


def labelled_value(value):
    value = value.strip()
    if value.endswith(","):
        value = value[:-1].rstrip()
    if value.endswith('".') and value.startswith('"'):
        value = value[:-1]
    if value.startswith('"'):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            # The section label already delimits prose that contains literal quotes.
            if value.endswith('"'):
                value = value[1:-1]
            elif value[-1:] in {".", "!", "?", "。", "！", "？"}:
                if '"' not in value[1:]:
                    value = value[1:]
            else:
                return None
        else:
            if isinstance(decoded, str):
                return decoded
    elif len(value) >= 2 and value.startswith("'") and value.endswith("'"):
        value = value[1:-1]
    return value.strip()


def recover_followup(raw):
    text = unfence(raw)
    if text in {"", '""'} or NO_FOLLOWUP.fullmatch(text):
        return '""'
    if text.startswith("{") and text.endswith("}"):
        text = text[1:-1].strip()
    matches = list(FOLLOWUP_LABEL.finditer(text))
    fields = [match["label"].casefold().replace("_", " ").replace("-", "")
              for match in matches]
    if fields != ["followup query", "next answer"]:
        return None
    query = labelled_value(text[matches[0].end():matches[1].start()])
    answer = labelled_value(text[matches[1].end():])
    if query is None or answer is None or bool(query) != bool(answer) or len(query) > 1200:
        return None
    return json.dumps({"followup_query": query, "next_answer": answer}, ensure_ascii=False)


class RecoveringComponentClient:
    """Apply the original parser first and recover only explicit fields on failure."""

    def __init__(self, delegate):
        self.delegate = delegate
        self.recoveries = []

    def generate(self, prompt, **kwargs):
        followup = prompt.startswith(FOLLOWUP_INSTRUCTION)
        try:
            raw = self.delegate.generate(prompt, **kwargs)
        except ValueError as error:
            if not followup or str(error) != "Original auxiliary model generated empty output":
                raise
            self.recoveries.append("followup_empty")
            return '""'

        if prompt.startswith(RAR_INSTRUCTION):
            try:
                parse_rar(raw)
            except ValueError:
                repaired = recover_rar(raw)
                if repaired is not None:
                    self.recoveries.append("rewrite_answer")
                    return repaired
        if followup:
            try:
                parse_followup_response(raw)
            except (ValueError, TypeError) as error:
                if isinstance(error, TypeError) and "unhashable type" not in str(error):
                    raise
                repaired = recover_followup(raw)
                if repaired is not None:
                    self.recoveries.append("followup")
                    return repaired
                if isinstance(error, TypeError):
                    raise ValueError("Follow-up result contains a non-string field name") from error
        return raw


def recovered_plan(row, client, config):
    adapter = RecoveringComponentClient(client)
    plan = build_plan(row["question"], row["history"], client=adapter, mode="ollama", config=config)
    return plan, adapter.recoveries
