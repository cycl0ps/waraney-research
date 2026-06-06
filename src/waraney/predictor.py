import torch

class Predictor:

    def __init__(self, waraney_model):

        self.model = waraney_model.model
        self.tokenizer = waraney_model.tokenizer

    def predict(self, text):

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = torch.softmax(
            logits,
            dim=1
        )[0]

        label = int(torch.argmax(probs))

        return {
            "label": label,
            "confidence": float(probs[label])
        }