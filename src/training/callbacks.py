import json

from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

from transformers import TrainerCallback


class ProgressCallback(
    TrainerCallback
):

    def __init__(
        self,
        progress_file,
        config
    ):

        self.progress_file = Path(
            progress_file
        )

        self.timezone = (
            config
            .get("system", {})
            .get(
                "timezone",
                "Asia/Jakarta"
            )
        )

    def _save(
        self,
        status,
        epoch,
        global_step,
        message=None
    ):

        payload = {

            "status":
                status,

            "epoch":
                (
                    round(epoch, 4)
                    if epoch is not None
                    else None
                ),

            "global_step":
                global_step,

            "timestamp":
                datetime.now(
                    ZoneInfo(
                        self.timezone
                    )
                ).isoformat(),

            "message":
                message
        }

        with open(
            self.progress_file,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                payload,
                f,
                indent=4
            )

    def on_train_begin(
        self,
        args,
        state,
        control,
        **kwargs
    ):

        self._save(
            status="RUNNING",
            epoch=0,
            global_step=0,
            message="Training started."
        )

    def on_epoch_end(
        self,
        args,
        state,
        control,
        **kwargs
    ):

        self._save(
            status="RUNNING",
            epoch=state.epoch,
            global_step=state.global_step,
            message="Training in progress."
        )

    def on_train_end(
        self,
        args,
        state,
        control,
        **kwargs
    ):

        self._save(
            status="COMPLETED",
            epoch=state.epoch,
            global_step=state.global_step,
            message="Training completed."
        )

    def on_log(
        self,
        args,
        state,
        control,
        logs=None,
        **kwargs
    ):

        self._save(
            status="RUNNING",
            epoch=state.epoch,
            global_step=state.global_step,
            message="Training in progress."
        )

    def on_save(
        self,
        args,
        state,
        control,
        **kwargs
    ):

        self._save(
            status="CHECKPOINT",
            epoch=state.epoch,
            global_step=state.global_step,
            message="Checkpoint saved."
        )