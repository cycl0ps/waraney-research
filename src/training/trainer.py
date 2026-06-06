from transformers import Trainer

class WaraneyTrainer:

    def __init__(
        self,
        model,
        training_args,
        train_dataset,
        eval_dataset
    ):
        self.trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )

    def train(self):

        self.trainer.train()

    def evaluate(self):

        return self.trainer.evaluate()