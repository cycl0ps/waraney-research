from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo


class TrainingLog:

    def __init__(
        self,
        config,
        output_dir,
        filename="training_log.md"
    ):

        self.config = config

        self.output_dir = Path(
            output_dir
        )

        self.log_file = (
            self.output_dir
            / filename
        )

    def initialize(self):

        if self.log_file.exists():
            return

        with open(
            self.log_file,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(
                "# Training Log\n\n"
            )

    def append(
        self,
        content
    ):

        with open(
            self.log_file,
            "a",
            encoding="utf-8"
        ) as f:

            f.write(content)
            f.write("\n")

    def add_training_run(
        self,
        status="Planned"
    ):

        timezone = (
            self.config
            .get("system", {})
            .get(
                "timezone",
                "Asia/Jakarta"
            )
        )

        timestamp = datetime.now(
            ZoneInfo(timezone)
        ).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        project_name = (
            self.config["project"]["name"]
        )

        version = (
            self.config["project"]["version"]
        )

        content = f"""
## {project_name}-{version}

| Field | Value |
|---------|---------|
| Timestamp | {timestamp} |
| Status | {status} |
"""

        self.append(
            content
        )

    def add_configuration(self):

        training = self.config["training"]
        model = self.config["model"]

        content = f"""
### Configuration

| Parameter | Value |
|---------|---------|
| Backbone | {model["backbone"]} |
| Epochs | {training["epochs"]} |
| Batch Size | {training["batch_size"]} |
| Learning Rate | {training["learning_rate"]} |
| Max Length | {training["max_length"]} |
| Seed | {training["seed"]} |
"""

        self.append(
            content
        )

    def add_dataset_summary(
        self,
        train_size,
        validation_size,
        test_size
    ):

        dataset = self.config["dataset"]

        content = f"""
### Dataset

| Parameter | Value |
|---------|---------|
| Source | {dataset["source"]} |
| Train Samples | {train_size} |
| Validation Samples | {validation_size} |
| Test Samples | {test_size} |
| Stratify By | {dataset["stratify_by"]} |
"""

        self.append(
            content
        )

    def add_validation_metrics(
        self,
        metrics
    ):

        content = "\n### Validation Metrics\n\n"
        content += "| Metric | Value |\n"
        content += "|---------|---------|\n"

        for metric_name, value in metrics.items():

            content += (
                f"| {metric_name} | {value} |\n"
            )

        self.append(
            content
        )

    def add_test_metrics(
        self,
        metrics
    ):

        content = "\n### Test Metrics\n\n"
        content += "| Metric | Value |\n"
        content += "|---------|---------|\n"

        for metric_name, value in metrics.items():

            content += (
                f"| {metric_name} | {value} |\n"
            )

        self.append(
            content
        )

    def add_note(
        self,
        note
    ):

        content = f"""
### Notes

- {note}

---
"""

        self.append(
            content
        )