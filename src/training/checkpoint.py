from pathlib import Path


def find_latest_checkpoint(
    checkpoint_dir
):

    checkpoint_dir = Path(
        checkpoint_dir
    )

    checkpoints = list(
        checkpoint_dir.glob(
            "checkpoint-*"
        )
    )

    if not checkpoints:
        return None

    checkpoints = sorted(
        checkpoints,
        key=lambda x: int(
            x.name.split("-")[-1]
        )
    )

    return str(
        checkpoints[-1]
    )