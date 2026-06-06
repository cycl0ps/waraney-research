from sklearn.model_selection import train_test_split


def _build_stratify_key(df, stratify_by):
    """
    Create stratification key from one or more columns.
    """

    if stratify_by is None:
        return None

    if isinstance(stratify_by, str):
        return df[stratify_by]

    if isinstance(stratify_by, list):

        missing = [
            col
            for col in stratify_by
            if col not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing stratify columns: {missing}"
            )

        return df[stratify_by].astype(str).agg(
            "_".join,
            axis=1
        )

    raise ValueError(
        "stratify_by must be string, list, or None"
    )


def split_dataset(
    df,
    train_ratio=0.8,
    validation_ratio=0.1,
    test_ratio=0.1,
    random_state=42,
    stratify_by="label"
):

    total_ratio = (
        train_ratio +
        validation_ratio +
        test_ratio
    )

    if round(total_ratio, 5) != 1.0:
        raise ValueError(
            "Ratios must sum to 1.0"
        )

    stratify_key = _build_stratify_key(
        df,
        stratify_by
    )

    train_df, temp_df = train_test_split(
        df,
        test_size=(1 - train_ratio),
        stratify=stratify_key,
        random_state=random_state
    )

    val_fraction = (
        validation_ratio /
        (validation_ratio + test_ratio)
    )

    temp_stratify = _build_stratify_key(
        temp_df,
        stratify_by
    )

    validation_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_fraction),
        stratify=temp_stratify,
        random_state=random_state
    )

    return (
        train_df.reset_index(drop=True),
        validation_df.reset_index(drop=True),
        test_df.reset_index(drop=True)
    )