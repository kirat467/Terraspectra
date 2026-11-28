import json
from pathlib import Path
from datetime import datetime, timezone


def save_prediction_history(
    predictions,
    file_path="ml/results/prediction_history.json"
):
    if not isinstance(predictions, list):
        raise ValueError("Predictions must be a list")

    validated_predictions = []

    for prediction in predictions:
        if not isinstance(prediction, dict):
            raise ValueError("Each prediction must be a dictionary")

        record = prediction.copy()

        try:
            json.dumps(record)
        except (TypeError, ValueError, OverflowError):
            raise ValueError("Prediction must contain valid JSON data")

        validated_predictions.append(record)

    path = Path(file_path)
    history = load_prediction_history(path)

    timestamp = datetime.now(timezone.utc).isoformat()

    for record in validated_predictions:
        record["timestamp"] = timestamp
        history.append(record)

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=4, allow_nan=False)

    return history


def load_prediction_history(
    file_path="ml/results/prediction_history.json"
):
    path = Path(file_path)

    if not path.exists():
        return []

    try:
        with open(path, "r", encoding="utf-8") as file:
            history = json.load(file)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise ValueError("Prediction history contains invalid JSON")
    except OSError as error:
        raise ValueError("Unable to read prediction history") from error

    if not isinstance(history, list):
        raise ValueError("Invalid prediction history format")

    if any(not isinstance(record, dict) for record in history):
        raise ValueError("Prediction history must contain dictionaries")

    return history


def clear_prediction_history(
    file_path="ml/results/prediction_history.json"
):
    path = Path(file_path)

    if path.exists():
        path.unlink()

    return True