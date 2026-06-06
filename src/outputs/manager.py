from pathlib import Path
from datetime import datetime
import json
import yaml


class OutputManager:

    def __init__(self, config):

        project_name = config["project"]["name"]
        version = config["project"]["version"]

        root_dir = config["output"]["root_dir"]

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        self.output_dir = (
            Path(root_dir)
            / f"{project_name}-{version}"
            / timestamp
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    @property
    def model_dir(self):

        path = self.output_dir / "model"

        path.mkdir(
            exist_ok=True
        )

        return path

    def save_config(self, config):

        with open(
            self.output_dir / "config.yaml",
            "w",
            encoding="utf-8"
        ) as f:

            yaml.safe_dump(
                config,
                f,
                sort_keys=False
            )

    def save_metrics(self, metrics):

        with open(
            self.output_dir / "metrics.json",
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                metrics,
                f,
                indent=2
            )

    def save_split(
        self,
        train_df,
        validation_df,
        test_df
    ):

        train_df.to_csv(
            self.output_dir / "train.csv",
            index=False
        )

        validation_df.to_csv(
            self.output_dir / "validation.csv",
            index=False
        )

        test_df.to_csv(
            self.output_dir / "test.csv",
            index=False
        )

    def save_predictions(
        self,
        predictions_df
    ):

        predictions_df.to_csv(
            self.output_dir / "predictions.csv",
            index=False
        )

    def save_model(
        self,
        model,
        tokenizer
    ):

        model.save_pretrained(
            self.model_dir
        )

        tokenizer.save_pretrained(
            self.model_dir
        )