import pandas as pd
import numpy as np

from scipy.special import softmax

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


class WaraneyEvaluator:

    def __init__(
        self,
        trainer
    ):

        self.trainer = trainer

    def predict(
        self,
        dataset
    ):

        return self.trainer.predict(
            dataset
        )

    def _extract_labels(
        self,
        predictions
    ):

        y_pred = np.argmax(
            predictions.predictions,
            axis=1
        )

        y_true = predictions.label_ids

        return y_true, y_pred

    def evaluate(
        self,
        predictions
    ):

        y_true, y_pred = (
            self._extract_labels(
                predictions
            )
        )

        metrics = {

            "accuracy":
                accuracy_score(
                    y_true,
                    y_pred
                ),

            "precision":
                precision_score(
                    y_true,
                    y_pred,
                    zero_division=0
                ),

            "recall":
                recall_score(
                    y_true,
                    y_pred,
                    zero_division=0
                ),

            "f1":
                f1_score(
                    y_true,
                    y_pred,
                    zero_division=0
                )
        }

        return metrics

    def confusion_matrix(
        self,
        predictions
    ):

        y_true, y_pred = (
            self._extract_labels(
                predictions
            )
        )

        return confusion_matrix(
            y_true,
            y_pred
        )

    def classification_report(
        self,
        predictions
    ):

        y_true, y_pred = (
            self._extract_labels(
                predictions
            )
        )

        return classification_report(
            y_true,
            y_pred,
            digits=4,
            zero_division=0
        )

    def predictions_dataframe(
        self,
        predictions,
        dataset
    ):

        logits = predictions.predictions

        probabilities = softmax(
            logits,
            axis=1
        )

        y_pred = np.argmax(
            logits,
            axis=1
        )

        prob_human = probabilities[:, 0]
        prob_ai = probabilities[:, 1]

        confidence = np.max(
            probabilities,
            axis=1
        )

        records = []

        for i in range(
            len(dataset)
        ):

            row = dict(
                dataset[i]
            )

            row.pop(
                "text",
                None
            )

            row["y_pred"] = int(
                y_pred[i]
            )

            row["prob_human"] = float(
                prob_human[i]
            )

            row["prob_ai"] = float(
                prob_ai[i]
            )

            row["confidence"] = float(
                confidence[i]
            )

            records.append(
                row
            )

        return pd.DataFrame(
            records
        )