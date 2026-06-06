import pandas as pd

def load_dataset(path: str) -> pd.DataFrame:

    df = pd.read_csv(path)

    required_columns = [
        "text",
        "label"
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    return df