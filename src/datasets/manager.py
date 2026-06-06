from .loader import load_dataset
from .splitter import split_dataset


def prepare_datasets(config):

    dataset_cfg = config["dataset"]

    if dataset_cfg["split_if_needed"]:

        df = load_dataset(
            dataset_cfg["source"]
        )

        return split_dataset(
            df,
            train_ratio=dataset_cfg["train_ratio"],
            validation_ratio=dataset_cfg["validation_ratio"],
            test_ratio=dataset_cfg["test_ratio"],
            random_state=dataset_cfg["random_state"],
            stratify_by=dataset_cfg.get(
                "stratify_by",
                "label"
            )
        )

    train_df = load_dataset(
        dataset_cfg["train"]
    )

    validation_df = load_dataset(
        dataset_cfg["validation"]
    )

    test_df = load_dataset(
        dataset_cfg["test"]
    )

    return (
        train_df,
        validation_df,
        test_df
    )