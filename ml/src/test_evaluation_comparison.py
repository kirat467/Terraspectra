import numpy as np
import pytest

from evaluation_comparison import compare_evaluations


def create_model(accuracy=0.8, precision=0.8, recall=0.8, f1_score=0.8):
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }


def test_model_a_wins():
    a = create_model(0.9, 0.9, 0.8, 0.85)
    b = create_model(0.8, 0.8, 0.7, 0.75)

    assert compare_evaluations(a, b)["winner"] == "Model A"


def test_model_b_wins():
    a = create_model(0.9, 0.8, 0.7, 0.75)
    b = create_model(0.8, 0.9, 0.9, 0.85)

    assert compare_evaluations(a, b)["winner"] == "Model B"


def test_accuracy_breaks_f1_tie():
    a = create_model(0.9, 0.8, 0.8, 0.8)
    b = create_model(0.7, 0.8, 0.8, 0.8)

    assert compare_evaluations(a, b)["winner"] == "Model A"


def test_model_b_wins_accuracy_tie_breaker():
    a = create_model(0.7, 0.8, 0.8, 0.8)
    b = create_model(0.9, 0.8, 0.8, 0.8)

    assert compare_evaluations(a, b)["winner"] == "Model B"


def test_complete_tie():
    a = create_model()
    b = create_model()

    assert compare_evaluations(a, b)["winner"] == "Tie"


def test_missing_metric():
    with pytest.raises(ValueError):
        compare_evaluations(
            {"accuracy": 0.8},
            create_model()
        )


def test_missing_metric_in_model_b():
    with pytest.raises(ValueError):
        compare_evaluations(
            create_model(),
            {"accuracy": 0.8}
        )


def test_invalid_input():
    with pytest.raises(TypeError):
        compare_evaluations([], {})


def test_invalid_metric_range():
    a = create_model(accuracy=1.5)

    with pytest.raises(ValueError):
        compare_evaluations(a, create_model())


def test_negative_metric():
    a = create_model(accuracy=-0.1)

    with pytest.raises(ValueError):
        compare_evaluations(a, create_model())


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), float("-inf")]
)
def test_non_finite_metrics(value):
    a = create_model(accuracy=value)

    with pytest.raises(ValueError):
        compare_evaluations(a, create_model())


@pytest.mark.parametrize(
    "value",
    [True, False]
)
def test_boolean_metric(value):
    a = create_model(accuracy=value)

    with pytest.raises(TypeError):
        compare_evaluations(a, create_model())


def test_string_metric():
    a = create_model(accuracy="0.9")

    with pytest.raises(TypeError):
        compare_evaluations(a, create_model())


def test_none_metric():
    a = create_model(accuracy=None)

    with pytest.raises(TypeError):
        compare_evaluations(a, create_model())


def test_boundary_values():
    a = create_model(accuracy=0, f1_score=0)
    b = create_model(accuracy=1, f1_score=1)

    result = compare_evaluations(a, b)

    assert result["winner"] == "Model B"


def test_numpy_numeric_values():
    a = create_model(
        accuracy=np.float64(0.9),
        precision=np.float64(0.8),
        recall=np.float64(0.8),
        f1_score=np.float64(0.9)
    )

    result = compare_evaluations(a, create_model())

    assert result["winner"] == "Model A"


def test_complex_metric():
    a = create_model(accuracy=complex(0.8, 0.2))

    with pytest.raises(TypeError):
        compare_evaluations(a, create_model())


def test_result_structure():
    result = compare_evaluations(
        create_model(0.9, 0.8, 0.8, 0.9),
        create_model()
    )

    assert set(result.keys()) == {
        "model_a_f1",
        "model_b_f1",
        "model_a_accuracy",
        "model_b_accuracy",
        "winner"
    }


def test_input_models_are_not_modified():
    a = create_model(0.9, 0.8, 0.8, 0.9)
    b = create_model()

    original_a = a.copy()
    original_b = b.copy()

    compare_evaluations(a, b)

    assert a == original_a
    assert b == original_b


def test_all_metrics_are_validated():
    a = create_model()
    b = create_model()

    b["precision"] = 2

    with pytest.raises(ValueError):
        compare_evaluations(a, b)