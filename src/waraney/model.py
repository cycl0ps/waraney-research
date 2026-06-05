from transformers import AutoTokenizer


class WaraneyModel:

    def __init__(
        self,
        model_name: str
    ):

        self.model_name = model_name

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                model_name
            )
        )

    def info(self):

        return {
            "model_name": self.model_name
        }