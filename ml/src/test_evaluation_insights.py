import pytest
import numpy as np

from evaluation_insights import generate_evaluation_insights


def sample_metrics(
    accuracy=0.7,
    precision=0.8,
    recall=0.8,
    f1_score=0.8
):
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }


def test_strong_accuracy():
    result = generate_evaluation_insights(
        sample_metrics(accuracy=0.9)
    )

    assert "The model has strong overall accuracy" in result["insights"]


def test_low_accuracy():
    result = generate_evaluation_insights(
        sample_metrics(accuracy=0.3)
    )

    assert "Overall accuracy needs improvement" in result["insights"]


def test_moderate_accuracy():
    result = generate_evaluation_insights(
        sample_metrics(accuracy=0.7)
    )

    assert "Overall accuracy is moderate" in result["insights"]


def test_precision_recall_balance():
    result = generate_evaluation_insights(
        sample_metrics(precision=0.8, recall=0.8)
    )

    assert "Precision and recall are balanced" in result["insights"]


def test_false_positive_insight():
    result = generate_evaluation_insights(
        sample_metrics(precision=0.6, recall=0.8)
    )

    assert any("false positives" in item for item in result["insights"])


def test_false_negative_insight():
    result = generate_evaluation_insights(
        sample_metrics(precision=0.9, recall=0.6)
    )

    assert any("misses relatively more" in item for item in result["insights"])


def test_weak_f1_score():
    result = generate_evaluation_insights(
        sample_metrics(f1_score=0.3)
    )

    assert any("weak precision-recall balance" in item for item in result["insights"])


def test_strong_f1_score():
    result = generate_evaluation_insights(
        sample_metrics(f1_score=0.9)
    )

    assert any("strong precision-recall balance" in item for item in result["insights"])


def test_moderate_f1_score():
    result = generate_evaluation_insights(
        sample_metrics(f1_score=0.7)
    )

    assert "F1-score has room for improvement" in result["insights"]


def test_strongest_metric():
    metrics = sample_metrics(
        accuracy=0.9,
        precision=0.7,
        recall=0.6,
        f1_score=0.8
    )

    result = generate_evaluation_insights(metrics)

    assert result["strongest_metric"] == "accuracy"


def test_weakest_metric():
    metrics = sample_metrics(
        accuracy=0.9,
        precision=0.7,
        recall=0.6,
        f1_score=0.8
    )

    result = generate_evaluation_insights(metrics)

    assert result["weakest_metric"] == "recall"


def test_missing_metric():
    with pytest.raises(ValueError):
        generate_evaluation_insights({"accuracy": 0.8})


def test_invalid_input():
    with pytest.raises(TypeError):
        generate_evaluation_insights([])


def test_invalid_metric_range():
    with pytest.raises(ValueError):
        generate_evaluation_insights(
            sample_metrics(accuracy=1.5)
        )


def test_negative_metric():
    with pytest.raises(ValueError):
        generate_evaluation_insights(
            sample_metrics(accuracy=-0.1)
        )


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), float("-inf")]
)
def test_non_finite_metric(value):
    with pytest.raises(ValueError):
        generate_evaluation_insights(
            sample_metrics(accuracy=value)
        )


@pytest.mark.parametrize(
    "value",
    [True, False]
)
def test_boolean_metric(value):
    with pytest.raises(TypeError):
        generate_evaluation_insights(
            sample_metrics(accuracy=value)
        )


@pytest.mark.parametrize(
    "value",
    ["0.9", None, complex(0.8, 0.2)]
)
def test_non_numeric_metric(value):
    with pytest.raises(TypeError):
        generate_evaluation_insights(
            sample_metrics(accuracy=value)
        )


def test_boundary_metric_values():
    result = generate_evaluation_insights(
        sample_metrics(
            accuracy=0,
            precision=1,
            recall=0,
            f1_score=1
        )
    )

    assert result["strongest_metric"] == "precision"
    assert result["weakest_metric"] == "accuracy"


def test_numpy_numeric_values():
    metrics = sample_metrics(
        accuracy=np.float64(0.9),
        precision=np.float64(0.8),
        recall=np.float64(0.7),
        f1_score=np.float64(0.85)
    )

    result = generate_evaluation_insights(metrics)

    assert result["strongest_metric"] == "accuracy"


def test_result_structure():
    result = generate_evaluation_insights(sample_metrics())

    assert set(result.keys()) == {
        "insights",
        "strongest_metric",
        "weakest_metric"
    }


def test_insights_are_returned_as_list():
    result = generate_evaluation_insights(sample_metrics())

    assert isinstance(result["insights"], list)
    assert len(result["insights"]) == 3


def test_all_metrics_are_validated():
    metrics = sample_metrics()
    metrics["precision"] = 2

    with pytest.raises(ValueError):
        generate_evaluation_insights(metrics)


def test_input_metrics_are_not_modified():
    metrics = sample_metrics()
    original = metrics.copy()

    generate_evaluation_insights(metrics)

    assert metrics == original


def test_strongest_metric_tie():
    metrics = sample_metrics(
        accuracy=0.9,
        precision=0.9,
        recall=0.7,
        f1_score=0.8
    )

    result = generate_evaluation_insights(metrics)

    assert result["strongest_metric"] == "accuracy"


def test_weakest_metric_tie():
    metrics = sample_metrics(
        accuracy=0.5,
        precision=0.5,
        recall=0.8,
        f1_score=0.9
    )

    result = generate_evaluation_insights(metrics)

    assert result["weakest_metric"] == "accuracy"