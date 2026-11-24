"""Frozen NLI model for verifying one hypothesis against a turn's passages."""


NLI_LABELS = ("entailment", "neutral", "contradiction")
HYPOTHESIS_MAX_TOKENS = 240


class EvidenceVerifier:
    def __init__(self, model_path, device="cuda:0", batch_size=16):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("NLI batch size must be a positive integer")
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(model_path), local_files_only=True, use_safetensors=True).to(self.device).eval()
        self.model.requires_grad_(False)
        self.label_ids = {label.strip().lower(): int(index)
                          for index, label in self.model.config.id2label.items()}
        if set(self.label_ids) != set(NLI_LABELS):
            raise ValueError("Evidence verifier must expose entailment, neutral and contradiction labels")

    def score_many(self, passages, question, answer):
        hypothesis = f'The answer to the question "{question}" is "{answer}".'
        ids = self.tokenizer.encode(hypothesis, add_special_tokens=False)
        if len(ids) > HYPOTHESIS_MAX_TOKENS:
            hypothesis = self.tokenizer.decode(ids[:HYPOTHESIS_MAX_TOKENS], skip_special_tokens=True)
        scores = []
        with self.torch.inference_mode():
            for start in range(0, len(passages), self.batch_size):
                batch = passages[start:start + self.batch_size]
                inputs = self.tokenizer(batch, [hypothesis] * len(batch), truncation="only_first",
                                        max_length=512, padding=True, return_tensors="pt").to(self.device)
                probabilities = self.model(**inputs).logits.float().softmax(dim=-1).cpu().tolist()
                scores.extend({label: row[self.label_ids[label]] for label in NLI_LABELS}
                              for row in probabilities)
        return scores
