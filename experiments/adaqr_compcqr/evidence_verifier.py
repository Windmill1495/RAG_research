"""Frozen NLI model for checking whether a passage supports a candidate answer."""

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch


class EvidenceVerifier:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(model_path), local_files_only=True).to("cuda:0").eval()
        self.model.requires_grad_(False)
        self.label_ids = {label.lower(): int(index)
                          for index, label in self.model.config.id2label.items()}

    def score_many(self, passages, rewrite, answer):
        hypothesis = f'The answer to "{rewrite}" is "{answer}".'
        scores = []
        for start in range(0, len(passages), 8):
            batch = passages[start:start + 8]
            inputs = self.tokenizer(batch, [hypothesis] * len(batch), truncation="only_first",
                                    max_length=512, padding=True, return_tensors="pt").to("cuda:0")
            with torch.inference_mode():
                probabilities = self.model(**inputs).logits.float().softmax(dim=-1)
            support = (probabilities[:, self.label_ids["entailment"]]
                       - probabilities[:, self.label_ids["contradiction"]]).clamp_min(0)
            scores.extend(support.cpu().tolist())
        return scores
