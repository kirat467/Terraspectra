import pytest
import numpy as np

from model_reliability import analyze_model_reliability


def test_reliability_analysis():
    result = analyze_model_reliability([0.9, 0.8, 0.7, 0.4])

    assert result["total_predictions"] == 4
    assert result["reliable_predictions"] == 3
    assert result["uncertain_predictions"] == 1
    assert result["reliability_percentage"] == 75.0


def test_average_confidence():
    result = analyze_model_reliability([0.8, 0.6, 1.0])

    assert result["average_confidence"] == 0.8


def test_custom_threshold():
    result = analyze_model_reliability(
        [0.5, 0.7, 0.9],
        confidence_threshold=0.8
    )

    assert result["reliable_predictions"] == 1
    assert result["uncertain_predictions"] == 2


def test_low_confidence_recommendation():
    result = analyze_model_reliability([0.2, 0.3, 0.4])

    assert any(
        "Improve" in item
        for item in result["recommendations"]
    )


def test_uncertain_prediction_recommendation():
    result = analyze_model_reliability([0.2, 0.3, 0.4, 0.9])

    assert any(
        "manually" in item
        for item in result["recommendations"]
    )


def test_empty_predictions():
    with pytest.raises(ValueError):
        analyze_model_reliability([])


def test_invalid_confidence():
    with pytest.raises(ValueError):
        analyze_model_reliability([0.5, 1.5])


def test_invalid_threshold():
    with pytest.raises(ValueError):
        analyze_model_reliability([0.5], confidence_threshold=2)


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), float("-inf")]
)
def test_non_finite_confidence(value):
    with pytest.raises(ValueError):
        analyze_model_reliability([0.5, value])


@pytest.mark.parametrize(
    "value",
    [True, False, "0.5", None, 1 + 2j]
)
def test_invalid_confidence_type(value):
    with pytest.raises(TypeError):
        analyze_model_reliability([0.5, value])


@pytest.mark.parametrize(
    "value",
    [True, False, "0.5", None, 1 + 2j]
)
def test_invalid_threshold_type(value):
    with pytest.raises(TypeError):
        analyze_model_reliability(
            [0.5],
            confidence_threshold=value
        )


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), float("-inf")]
)
def test_non_finite_threshold(value):
    with pytest.raises(ValueError):
        analyze_model_reliability(
            [0.5],
            confidence_threshold=value
        )


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_confidence_out_of_range(value):
    with pytest.raises(ValueError):
        analyze_model_reliability([value])


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_threshold_out_of_range(value):
    with pytest.raises(ValueError):
        analyze_model_reliability(
            [0.5],
            confidence_threshold=value
        )


def test_numpy_numeric_values():
    result = analyze_model_reliability([
        np.float64(0.8),
        np.float64(0.9)
    ])

    assert result["reliable_predictions"] == 2


def test_all_predictions_reliable():
    result = analyze_model_reliability([0.8, 0.9, 1.0])

    assert result["reliable_predictions"] == 3
    assert result["uncertain_predictions"] == 0


def test_all_predictions_uncertain():
    result = analyze_model_reliability([0.1, 0.2, 0.3])

    assert result["reliable_predictions"] == 0
    assert result["uncertain_predictions"] == 3


def test_boundary_confidence():
    result = analyze_model_reliability(
        [0.6],
        confidence_threshold=0.6
    )

    assert result["reliable_predictions"] == 1


def test_zero_confidence():
    result = analyze_model_reliability([0.0])

    assert result["average_confidence"] == 0.0


def test_one_confidence():
    result = analyze_model_reliability([1.0])

    assert result["average_confidence"] == 1.0


def test_output_structure():
    result = analyze_model_reliability([0.8, 0.9])

    assert set(result.keys()) == {
        "total_predictions",
        "reliable_predictions",
        "uncertain_predictions",
        "reliability_percentage",
        "average_confidence",
        "confidence_threshold",
        "recommendations"
    }


def test_original_predictions_are_not_modified():
    predictions = [0.8, 0.9]
    original = predictions.copy()

    analyze_model_reliability(predictions)

    assert predictions == original


def test_custom_threshold_is_returned():
    result = analyze_model_reliability(
        [0.5, 0.9],
        confidence_threshold=0.75
    )

    assert result["confidence_threshold"] == 0.75