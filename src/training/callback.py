import json

from pathlib import Path
from datetime import datetime

from transformers import TrainerCallback


class ProgressCallback(
    TrainerCallback
):

    def __init__(
        self,
        progress_file
    ):

        self.progress_file = Path(
            progress_file
        )

    def _save(
        self,
        status,
        epoch,
        global_step
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
                datetime.now()
                .isoformat()
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
            global_step=0
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
            global_step=state.global_step
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
            global_step=state.global_step
        )