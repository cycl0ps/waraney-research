from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)


def compute_metrics(
    y_true,
    y_pred
):

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="binary"
        )
    )

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }