from datasets import Dataset


def tokenize_dataframe(
    df,
    tokenizer,
    max_length
):

    dataset = Dataset.from_pandas(
        df,
        preserve_index=False
    )

    def tokenize(batch):

        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length
        )

    dataset = dataset.map(
        tokenize,
        batched=True
    )

    keep_columns = [
        "label",
        "input_ids",
        "attention_mask"
    ]

    if "token_type_ids" in dataset.column_names:
        keep_columns.append(
            "token_type_ids"
        )

    remove_columns = [
        col
        for col in dataset.column_names
        if col not in keep_columns
    ]

    dataset = dataset.remove_columns(
        remove_columns
    )

    return dataset