import pytest
from prediction_result import format_prediction


CLASS_NAMES = [
    "Healthy",
    "Alternaria alternata",
    "Alternaria solani",
    "Botrytis cinerea",
    "Fusarium oxysporum"
]


def test_format_prediction():
    result = format_prediction(0, 0.95, CLASS_NAMES)

    assert result["class_id"] == 0
    assert result["predicted_class"] == "Healthy"
    assert result["confidence"] == 95.0
    assert result["status"] == "Confident"


def test_uncertain_prediction():
    result = format_prediction(1, 0.35, CLASS_NAMES)

    assert result["status"] == (
        "Uncertain - Further analysis required"
    )


def test_confidence_threshold_boundary():
    result = format_prediction(
        0, 0.60, CLASS_NAMES, confidence_threshold=0.60
    )

    assert result["status"] == "Confident"


def test_custom_confidence_threshold():
    result = format_prediction(
        0, 0.70, CLASS_NAMES, confidence_threshold=0.80
    )

    assert result["status"] == (
        "Uncertain - Further analysis required"
    )


def test_invalid_class_id():
    with pytest.raises(ValueError):
        format_prediction(10, 0.90, CLASS_NAMES)


@pytest.mark.parametrize("class_id", [-1, 5, 1.5, "1", True])
def test_invalid_class_id_types(class_id):
    with pytest.raises(ValueError):
        format_prediction(class_id, 0.90, CLASS_NAMES)


@pytest.mark.parametrize(
    "confidence",
    [-0.1, 1.1, float("nan"), float("inf"),
     float("-inf"), "0.8", True, None]
)
def test_invalid_confidence(confidence):
    with pytest.raises(ValueError):
        format_prediction(0, confidence, CLASS_NAMES)


@pytest.mark.parametrize(
    "threshold",
    [-0.1, 1.1, float("nan"), float("inf"),
     float("-inf"), "0.6", True, None]
)
def test_invalid_threshold(threshold):
    with pytest.raises(ValueError):
        format_prediction(
            0, 0.90, CLASS_NAMES,
            confidence_threshold=threshold
        )


@pytest.mark.parametrize(
    "class_names",
    [None, "Healthy", [], [""], ["Healthy", None], [123]]
)
def test_invalid_class_names(class_names):
    with pytest.raises(ValueError):
        format_prediction(0, 0.90, class_names)


def test_confidence_zero():
    result = format_prediction(0, 0.0, CLASS_NAMES)

    assert result["confidence"] == 0.0
    assert result["status"] == (
        "Uncertain - Further analysis required"
    )


def test_confidence_one():
    result = format_prediction(0, 1.0, CLASS_NAMES)

    assert result["confidence"] == 100.0
    assert result["status"] == "Confident"


def test_class_name_whitespace():
    result = format_prediction(
        0, 0.90, [" Healthy "]
    )

    assert result["predicted_class"] == "Healthy"