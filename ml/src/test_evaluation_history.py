import json
import numpy as np
import pytest

from evaluation_history import EvaluationHistory


@pytest.fixture
def history(tmp_path):
    file_path = tmp_path / "evaluation_history.json"
    return EvaluationHistory(str(file_path))


def sample_metrics(
    accuracy=0.90,
    precision=0.88,
    recall=0.89,
    f1_score=0.885
):
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }


def test_add_evaluation(history):
    result = history.add_evaluation("3D-CNN", sample_metrics())

    assert result["model_name"] == "3D-CNN"
    assert result["metrics"]["accuracy"] == 0.90
    assert len(history.history) == 1


def test_save_and_load_history(history):
    history.add_evaluation("3D-CNN", sample_metrics())

    new_history = EvaluationHistory(history.file_path)

    assert len(new_history.history) == 1
    assert new_history.history[0]["model_name"] == "3D-CNN"


def test_get_history_by_model(history):
    history.add_evaluation("3D-CNN", sample_metrics())
    history.add_evaluation("ViT", sample_metrics(0.85))

    result = history.get_history("3D-CNN")

    assert len(result) == 1
    assert result[0]["model_name"] == "3D-CNN"


def test_compare_latest(history):
    history.add_evaluation("3D-CNN", sample_metrics(0.80))
    history.add_evaluation("3D-CNN", sample_metrics(0.90))

    result = history.compare_latest("3D-CNN")

    assert result["changes"]["accuracy"] == 0.10
    assert result["changes"]["precision"] == 0.0


def test_compare_latest_without_enough_history(history):
    history.add_evaluation("3D-CNN", sample_metrics())

    assert history.compare_latest("3D-CNN") is None


def test_clear_history(history):
    history.add_evaluation("3D-CNN", sample_metrics())

    result = history.clear_history()

    assert result is True
    assert len(history.history) == 0


def test_missing_metric(history):
    metrics = sample_metrics()
    del metrics["accuracy"]

    with pytest.raises(ValueError):
        history.add_evaluation("3D-CNN", metrics)


def test_invalid_metric_value(history):
    with pytest.raises(ValueError):
        history.add_evaluation(
            "3D-CNN",
            sample_metrics(accuracy=1.5)
        )


def test_invalid_model_name(history):
    with pytest.raises(ValueError):
        history.add_evaluation("", sample_metrics())


def test_invalid_metrics_type(history):
    with pytest.raises(ValueError):
        history.add_evaluation("3D-CNN", [])


def test_model_name_whitespace_is_removed(history):
    result = history.add_evaluation("  3D-CNN  ", sample_metrics())

    assert result["model_name"] == "3D-CNN"


def test_empty_model_name_with_spaces(history):
    with pytest.raises(ValueError):
        history.add_evaluation("   ", sample_metrics())


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), float("-inf")]
)
def test_non_finite_metric(history, value):
    with pytest.raises(ValueError):
        history.add_evaluation(
            "3D-CNN",
            sample_metrics(accuracy=value)
        )


@pytest.mark.parametrize("value", [True, False])
def test_boolean_metric(history, value):
    with pytest.raises(ValueError):
        history.add_evaluation(
            "3D-CNN",
            sample_metrics(accuracy=value)
        )


@pytest.mark.parametrize(
    "value",
    ["0.9", None, complex(0.9, 0.1)]
)
def test_non_numeric_metric(history, value):
    with pytest.raises(ValueError):
        history.add_evaluation(
            "3D-CNN",
            sample_metrics(accuracy=value)
        )


def test_negative_metric(history):
    with pytest.raises(ValueError):
        history.add_evaluation(
            "3D-CNN",
            sample_metrics(accuracy=-0.1)
        )


def test_boundary_metric_values(history):
    result = history.add_evaluation(
        "3D-CNN",
        sample_metrics(accuracy=0, precision=1, recall=0, f1_score=1)
    )

    assert result["metrics"]["accuracy"] == 0.0
    assert result["metrics"]["precision"] == 1.0


def test_numpy_numeric_metrics(history):
    metrics = sample_metrics(
        accuracy=np.float64(0.9),
        precision=np.float64(0.8)
    )

    result = history.add_evaluation("3D-CNN", metrics)

    assert result["metrics"]["accuracy"] == 0.9


def test_empty_history_when_file_does_not_exist(history):
    assert history.history == []


def test_invalid_json_file(tmp_path):
    file_path = tmp_path / "invalid.json"
    file_path.write_text("{invalid json", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid JSON"):
        EvaluationHistory(str(file_path))


def test_json_root_must_be_list(tmp_path):
    file_path = tmp_path / "history.json"
    file_path.write_text('{"model": "3D-CNN"}', encoding="utf-8")

    with pytest.raises(ValueError, match="must contain a list"):
        EvaluationHistory(str(file_path))


def test_invalid_loaded_record(tmp_path):
    file_path = tmp_path / "history.json"
    file_path.write_text(
        json.dumps([{"model_name": "3D-CNN"}]),
        encoding="utf-8"
    )

    with pytest.raises(ValueError):
        EvaluationHistory(str(file_path))


def test_get_history_for_unknown_model(history):
    history.add_evaluation("3D-CNN", sample_metrics())

    assert history.get_history("ViT") == []


def test_get_history_invalid_model_name(history):
    with pytest.raises(ValueError):
        history.get_history(123)


def test_compare_latest_for_different_models(history):
    history.add_evaluation("3D-CNN", sample_metrics())
    history.add_evaluation("ViT", sample_metrics())

    assert history.compare_latest("3D-CNN") is None


def test_compare_latest_detects_all_metric_changes(history):
    history.add_evaluation(
        "3D-CNN",
        sample_metrics(0.5, 0.5, 0.5, 0.5)
    )

    history.add_evaluation(
        "3D-CNN",
        sample_metrics(0.9, 0.8, 0.7, 0.6)
    )

    result = history.compare_latest("3D-CNN")

    assert result["changes"] == {
        "accuracy": 0.4,
        "precision": 0.3,
        "recall": 0.2,
        "f1_score": 0.1
    }


def test_input_metrics_are_not_modified(history):
    metrics = sample_metrics()
    original = metrics.copy()

    history.add_evaluation("3D-CNN", metrics)

    assert metrics == original


def test_add_evaluation_persists_to_file(history):
    history.add_evaluation("3D-CNN", sample_metrics())

    with open(history.file_path, "r", encoding="utf-8") as file:
        saved_data = json.load(file)

    assert len(saved_data) == 1
    assert saved_data[0]["model_name"] == "3D-CNN"


def test_clear_history_persists_to_file(history):
    history.add_evaluation("3D-CNN", sample_metrics())
    history.clear_history()

    with open(history.file_path, "r", encoding="utf-8") as file:
        saved_data = json.load(file)

    assert saved_data == []