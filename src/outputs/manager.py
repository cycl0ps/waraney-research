from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import yaml


class OutputManager:

    def __init__(
        self,
        config,
        run_id=None
    ):

        self.config = config

        self.project_name = (
            config["project"]["name"]
        )

        self.version = (
            config["project"]["version"]
        )

        root_dir = (
            config["output"]["root_dir"]
        )

        timezone = (
            config
            .get("system", {})
            .get(
                "timezone",
                "Asia/Jakarta"
            )
        )

        if run_id is None:

            run_id = datetime.now(
                ZoneInfo(timezone)
            ).strftime(
                "%Y%m%d_%H%M%S"
            )

        self.run_id = run_id

        self.output_dir = (
            Path(root_dir)
            / f"{self.project_name}-{self.version}"
            / self.run_id
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

    def save_config(
        self,
        config
    ):

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

    def save_metrics(
        self,
        metrics
    ):

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

    def save_trainer_state(
        self,
        trainer
    ):

        trainer.state.save_to_json(
            str(
                self.output_dir
                / "trainer_state.json"
            )
        )

    def save_summary(
        self,
        summary
    ):

        with open(
            self.output_dir / "summary.json",
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                summary,
                f,
                indent=2
            )

    def create_summary(
        self,
        config,
        train_size=None,
        validation_size=None,
        test_size=None
    ):

        return {

            "project":
                self.project_name,

            "version":
                self.version,

            "run_id":
                self.run_id,

            "backbone":
                config["model"]["backbone"],

            "epochs":
                config["training"]["epochs"],

            "learning_rate":
                config["training"]["learning_rate"],

            "train_size":
                train_size,

            "validation_size":
                validation_size,

            "test_size":
                test_size
        }