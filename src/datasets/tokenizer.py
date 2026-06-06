from datasets import Dataset


def tokenize_dataframe(
    df,
    tokenizer,
    max_length
):

    dataset = Dataset.from_pandas(df)

    def tokenize(batch):

        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length
        )

    dataset = dataset.map(
        tokenize,
        batched=True
    )

    return dataset