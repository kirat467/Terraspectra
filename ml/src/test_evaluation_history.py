import pytest
from evaluation_history import EvaluationHistory


@pytest.fixture
def history(tmp_path):
    file_path = tmp_path / "evaluation_history.json"
    return EvaluationHistory(str(file_path))


def sample_metrics(accuracy=0.90, precision=0.88, recall=0.89, f1_score=0.885):
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
    metrics = sample_metrics(accuracy=1.5)

    with pytest.raises(ValueError):
        history.add_evaluation("3D-CNN", metrics)


def test_invalid_model_name(history):
    with pytest.raises(ValueError):
        history.add_evaluation("", sample_metrics())


def test_invalid_metrics_type(history):
    with pytest.raises(ValueError):
        history.add_evaluation("3D-CNN", [])