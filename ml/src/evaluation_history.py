import json
import os
import math
from numbers import Real
from datetime import datetime


class EvaluationHistory:

    REQUIRED_METRICS = [
        "accuracy",
        "precision",
        "recall",
        "f1_score"
    ]

    def __init__(self, file_path="ml/results/evaluation_history.json"):
        self.file_path = os.fspath(file_path)
        self.history = []
        self.load_history()

    def _validate_metrics(self, metrics):

        if not isinstance(metrics, dict):
            raise ValueError("Metrics must be a dictionary")

        validated = {}

        for metric in self.REQUIRED_METRICS:

            if metric not in metrics:
                raise ValueError(f"Missing metric: {metric}")

            value = metrics[metric]

            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"{metric} must be a number")

            value = float(value)

            if not math.isfinite(value):
                raise ValueError(f"{metric} must be finite")

            if not 0 <= value <= 1:
                raise ValueError(f"{metric} must be between 0 and 1")

            validated[metric] = value

        return validated

    def _validate_record(self, record):

        if not isinstance(record, dict):
            raise ValueError("Invalid history record")

        if not isinstance(record.get("model_name"), str):
            raise ValueError("Invalid model name in history")

        if not record["model_name"].strip():
            raise ValueError("Model name cannot be empty")

        if not isinstance(record.get("timestamp"), str):
            raise ValueError("Invalid timestamp in history")

        self._validate_metrics(record.get("metrics"))

    def load_history(self):

        if not os.path.exists(self.file_path):
            self.history = []
            return self.history

        try:
            with open(self.file_path, "r", encoding="utf-8") as file:
                loaded_history = json.load(file)

        except json.JSONDecodeError as error:
            raise ValueError("History file contains invalid JSON") from error

        if not isinstance(loaded_history, list):
            raise ValueError("History file must contain a list")

        for record in loaded_history:
            self._validate_record(record)

        self.history = loaded_history

        return self.history

    def save_history(self):

        directory = os.path.dirname(self.file_path)

        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(self.file_path, "w", encoding="utf-8") as file:
            json.dump(self.history, file, indent=4, allow_nan=False)

    def add_evaluation(self, model_name, metrics):

        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("Model name must be a non-empty string")

        model_name = model_name.strip()
        validated_metrics = self._validate_metrics(metrics)

        record = {
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "metrics": validated_metrics
        }

        self.history.append(record)

        try:
            self.save_history()
        except Exception:
            self.history.pop()
            raise

        return record

    def get_history(self, model_name=None):

        if model_name is None:
            return self.history

        if not isinstance(model_name, str):
            raise ValueError("Model name must be a string")

        model_name = model_name.strip()

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

        for metric in self.REQUIRED_METRICS:
            changes[metric] = round(
                latest[metric] - previous[metric], 4
            )

        return {
            "model_name": model_name.strip(),
            "previous_metrics": previous,
            "latest_metrics": latest,
            "changes": changes
        }

    def clear_history(self):

        previous_history = self.history
        self.history = []

        try:
            self.save_history()
        except Exception:
            self.history = previous_history
            raise

        return True