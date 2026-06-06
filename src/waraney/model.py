from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

class WaraneyModel:

    def __init__(self, model_name):
        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name
        )

    def save(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    @classmethod
    def load(cls, path):

        obj = cls.__new__(cls)

        obj.model_name = path

        obj.tokenizer = AutoTokenizer.from_pretrained(path)

        obj.model = AutoModelForSequenceClassification.from_pretrained(path)

        return obj