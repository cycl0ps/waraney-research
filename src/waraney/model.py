from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

class WaraneyModel:

    def __init__(self, config):

        self.config = config

        backbone = config["model"]["backbone"]

        self.tokenizer = AutoTokenizer.from_pretrained(
            backbone
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            backbone,
            num_labels=config["model"]["num_labels"]
        )