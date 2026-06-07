from pathlib import Path

from transformers import (
    Trainer,
    TrainingArguments
)

from src.evaluation.metrics import (
    compute_metrics
)


def build_training_arguments(config):

    project_name = (
        f"{config['project']['name']}-"
        f"{config['project']['version']}"
    )

    output_dir = (
        Path(
            config["output"]["root_dir"]
        )
        / "checkpoints"
        / project_name
    )

    return TrainingArguments(

        output_dir=str(output_dir),

        learning_rate=float(
            config["training"]["learning_rate"]
        ),

        per_device_train_batch_size=int(
            config["training"]["batch_size"]
        ),

        per_device_eval_batch_size=int(
            config["training"]["batch_size"]
        ),

        num_train_epochs=int(
            config["training"]["epochs"]
        ),

        seed=int(
            config["training"]["seed"]
        ),

        eval_strategy=
            config["training"][
                "evaluation_strategy"
            ],

        save_strategy=
            config["training"][
                "save_strategy"
            ],

        save_total_limit=int(
            config["training"][
                "save_total_limit"
            ]
        ),

        logging_strategy=
            config["training"][
                "logging_strategy"
            ],

        report_to="none",

        dataloader_pin_memory=False
    )


class WaraneyTrainer:

    def __init__(
        self,
        model,
        training_args,
        train_dataset,
        validation_dataset,
        callbacks=None
    ):

        self.trainer = Trainer(

            model=model,

            args=training_args,

            train_dataset=train_dataset,

            eval_dataset=validation_dataset,

            compute_metrics=compute_metrics,

            callbacks=callbacks
        )

    def train(
        self,
        resume_from_checkpoint=None
    ):

        return self.trainer.train(
            resume_from_checkpoint=
            resume_from_checkpoint
        )

    def evaluate(self):

        return self.trainer.evaluate()

    def predict(
        self,
        dataset
    ):

        return self.trainer.predict(
            dataset
        )

    def get_state(self):

        return self.trainer.state