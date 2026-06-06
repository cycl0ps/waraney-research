from .metrics import classification_metrics

class Evaluator:

    @staticmethod
    def evaluate(
        y_true,
        y_pred
    ):

        return classification_metrics(
            y_true,
            y_pred
        )