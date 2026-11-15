import json
from prediction_history import (
    save_prediction_history,
    load_prediction_history,
    clear_prediction_history
)


def test_save_prediction_history(tmp_path):
    file_path = tmp_path / "history.json"

    predictions = [{
        "prediction": "Healthy",
        "confidence": 92.5,
        "top_predictions": [
            {"disease": "Healthy", "confidence": 92.5},
            {"disease": "Leaf Rust", "confidence": 5.0},
            {"disease": "Blight", "confidence": 2.5}
        ]
    }]

    history = save_prediction_history(predictions, file_path)

    assert len(history) == 1
    assert history[0]["prediction"] == "Healthy"
    assert history[0]["confidence"] == 92.5
    assert "timestamp" in history[0]


def test_load_prediction_history(tmp_path):
    file_path = tmp_path / "history.json"

    file_path.write_text(
        json.dumps([{"prediction": "Blight"}]),
        encoding="utf-8"
    )

    history = load_prediction_history(file_path)

    assert len(history) == 1
    assert history[0]["prediction"] == "Blight"


def test_clear_prediction_history(tmp_path):
    file_path = tmp_path / "history.json"
    file_path.write_text("[]", encoding="utf-8")

    result = clear_prediction_history(file_path)

    assert result is True
    assert not file_path.exists()


def test_load_missing_history(tmp_path):
    file_path = tmp_path / "missing.json"

    history = load_prediction_history(file_path)

    assert history == []