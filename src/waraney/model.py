import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

class WaraneyModel:

    def __init__(self, config):

        self.config = config

        backbone = config["model"]["backbone"]

        label2id = config["labels"]
        id2label = {v: k for k, v in label2id.items()}

        self.tokenizer = AutoTokenizer.from_pretrained(
            backbone
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            backbone,
            num_labels=config["model"]["num_labels"],
            label2id=label2id,
            id2label=id2label,
            torch_dtype=torch.float32
        )

        print(
            "MODEL DTYPE:",
            next(
                self.model.parameters()
            ).dtype
        )