import json
import os
from datetime import datetime


class EvaluationHistory:

    def __init__(self, file_path="ml/results/evaluation_history.json"):
        self.file_path = file_path
        self.history = []
        self.load_history()

    def load_history(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, "r") as file:
                self.history = json.load(file)
        else:
            self.history = []

        return self.history

    def save_history(self):
        directory = os.path.dirname(self.file_path)

        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(self.file_path, "w") as file:
            json.dump(self.history, file, indent=4)

    def add_evaluation(self, model_name, metrics):
        required_metrics = [
            "accuracy",
            "precision",
            "recall",
            "f1_score"
        ]

        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("Model name must be a non-empty string")

        if not isinstance(metrics, dict):
            raise ValueError("Metrics must be a dictionary")

        for metric in required_metrics:
            if metric not in metrics:
                raise ValueError(f"Missing metric: {metric}")

            value = metrics[metric]

            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"{metric} must be a number")

            if not 0 <= value <= 1:
                raise ValueError(f"{metric} must be between 0 and 1")

        record = {
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                metric: metrics[metric]
                for metric in required_metrics
            }
        }

        self.history.append(record)
        self.save_history()

        return record

    def get_history(self, model_name=None):
        if model_name is None:
            return self.history

        return [
            record for record in self.history
            if record["model_name"] == model_name
        ]

    def compare_latest(self, model_name):
        records = self.get_history(model_name)

        if len(records) < 2:
            return None

        previous = records[-2]["metrics"]
        latest = records[-1]["metrics"]

        changes = {}

        for metric in previous:
            changes[metric] = round(
                latest[metric] - previous[metric], 4
            )

        return {
            "model_name": model_name,
            "previous_metrics": previous,
            "latest_metrics": latest,
            "changes": changes
        }

    def clear_history(self):
        self.history = []
        self.save_history()
        return True