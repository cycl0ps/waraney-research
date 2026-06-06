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

        backbone = (
            self.config["model"]["backbone"]
        )

        content = f"""
## {project_name}-{version}

| Field | Value |
|---------|---------|
| Timestamp | {timestamp} |
| Status | {status} |
| Backbone | {backbone} |

### Notes

-
"""

        self.append(
            content
        )

    def add_metrics(
        self,
        metrics
    ):

        content = "\n### Metrics\n\n"
        content += "| Metric | Value |\n"
        content += "|---------|---------|\n"

        for metric_name, value in metrics.items():

            content += (
                f"| {metric_name} | {value} |\n"
            )

        content += "\n---\n"

        self.append(
            content
        )

    def add_note(
        self,
        note
    ):

        self.append(
            f"- {note}"
        )