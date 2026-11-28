import json
import pytest

from prediction_history import (
    save_prediction_history,
    load_prediction_history,
    clear_prediction_history
)


def test_save_prediction_history(tmp_path):
    file_path = tmp_path / "history.json"

    predictions = [{
        "prediction": "Healthy",
        "confidence": 92.5
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

    assert load_prediction_history(file_path) == []


def test_save_multiple_predictions(tmp_path):
    file_path = tmp_path / "history.json"

    predictions = [
        {"prediction": "Healthy"},
        {"prediction": "Blight"}
    ]

    history = save_prediction_history(predictions, file_path)

    assert len(history) == 2
    assert history[0]["timestamp"] == history[1]["timestamp"]


def test_append_prediction_history(tmp_path):
    file_path = tmp_path / "history.json"

    save_prediction_history(
        [{"prediction": "Healthy"}], file_path
    )

    history = save_prediction_history(
        [{"prediction": "Blight"}], file_path
    )

    assert len(history) == 2


def test_invalid_predictions_type(tmp_path):
    with pytest.raises(ValueError):
        save_prediction_history("Healthy", tmp_path / "history.json")


def test_invalid_prediction_record(tmp_path):
    with pytest.raises(ValueError):
        save_prediction_history(
            [{"prediction": "Healthy"}, "Blight"],
            tmp_path / "history.json"
        )


def test_invalid_loaded_json(tmp_path):
    file_path = tmp_path / "history.json"
    file_path.write_text("{invalid json", encoding="utf-8")

    with pytest.raises(ValueError):
        load_prediction_history(file_path)


def test_invalid_history_format(tmp_path):
    file_path = tmp_path / "history.json"
    file_path.write_text(
        json.dumps({"prediction": "Healthy"}),
        encoding="utf-8"
    )

    with pytest.raises(ValueError):
        load_prediction_history(file_path)


def test_history_with_non_dictionary_records(tmp_path):
    file_path = tmp_path / "history.json"
    file_path.write_text(
        json.dumps([{"prediction": "Healthy"}, "Blight"]),
        encoding="utf-8"
    )

    with pytest.raises(ValueError):
        load_prediction_history(file_path)


def test_non_serializable_prediction(tmp_path):
    file_path = tmp_path / "history.json"

    with pytest.raises(ValueError):
        save_prediction_history(
            [{"prediction": object()}],
            file_path
        )

    assert not file_path.exists()


def test_empty_prediction_list(tmp_path):
    file_path = tmp_path / "history.json"

    history = save_prediction_history([], file_path)

    assert history == []
    assert file_path.exists()


def test_clear_missing_history(tmp_path):
    file_path = tmp_path / "missing.json"

    assert clear_prediction_history(file_path) is True


def test_saved_history_can_be_loaded(tmp_path):
    file_path = tmp_path / "history.json"

    save_prediction_history(
        [{"prediction": "Healthy", "confidence": 95}],
        file_path
    )

    history = load_prediction_history(file_path)

    assert history[0]["prediction"] == "Healthy"
    assert history[0]["confidence"] == 95