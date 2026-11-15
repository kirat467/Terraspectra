import json
from pathlib import Path
from datetime import datetime, timezone


def save_prediction_history(
    predictions,
    file_path="ml/results/prediction_history.json"
):
    if not isinstance(predictions, list):
        raise ValueError("Predictions must be a list")

    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    history = load_prediction_history(path)

    timestamp = datetime.now(timezone.utc).isoformat()

    for prediction in predictions:
        if not isinstance(prediction, dict):
            raise ValueError("Each prediction must be a dictionary")

        record = prediction.copy()
        record["timestamp"] = timestamp
        history.append(record)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=4)

    return history


def load_prediction_history(
    file_path="ml/results/prediction_history.json"
):
    path = Path(file_path)

    if not path.exists():
        return []

    with open(path, "r", encoding="utf-8") as file:
        history = json.load(file)

    if not isinstance(history, list):
        raise ValueError("Invalid prediction history format")

    return history


def clear_prediction_history(
    file_path="ml/results/prediction_history.json"
):
    path = Path(file_path)

    if path.exists():
        path.unlink()

    return True