import numpy as np
import pytest

from class_spectral_quality import compare_class_spectral_quality


def test_class_spectral_quality():
    cubes = {
        "Healthy": np.ones((10, 10, 5)),
        "Diseased": np.full((10, 10, 5), 2.0)
    }

    result = compare_class_spectral_quality(cubes)

    assert len(result) == 2
    assert "Healthy" in result
    assert "Diseased" in result

    assert result["Healthy"]["mean"] == 1.0
    assert result["Diseased"]["mean"] == 2.0

    assert len(result["Healthy"]["band_means"]) == 5


def test_band_means_are_calculated_correctly():
    cube = np.arange(60).reshape(3, 4, 5)

    result = compare_class_spectral_quality({"Healthy": cube})

    expected = cube.mean(axis=(0, 1))

    np.testing.assert_allclose(
        result["Healthy"]["band_means"],
        expected
    )


def test_spectral_statistics():
    cube = np.array(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
        dtype=np.float32
    )

    result = compare_class_spectral_quality({"Healthy": cube})
    stats = result["Healthy"]

    assert stats["mean"] == 4.5
    assert stats["minimum"] == 1.0
    assert stats["maximum"] == 8.0
    assert isinstance(stats["std"], float)


def test_empty_cubes():
    with pytest.raises(ValueError):
        compare_class_spectral_quality({})


def test_none_input():
    with pytest.raises(TypeError):
        compare_class_spectral_quality(None)


def test_invalid_input_type():
    with pytest.raises(TypeError):
        compare_class_spectral_quality([1, 2, 3])


def test_invalid_cube_dimensions():
    with pytest.raises(ValueError):
        compare_class_spectral_quality({
            "Healthy": np.random.rand(10, 10)
        })


def test_empty_cube_data():
    with pytest.raises(ValueError):
        compare_class_spectral_quality({
            "Healthy": np.empty((0, 10, 5))
        })


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_invalid_values(invalid_value):
    cube = np.ones((10, 10, 5))
    cube[0, 0, 0] = invalid_value

    with pytest.raises(ValueError):
        compare_class_spectral_quality({"Healthy": cube})


def test_non_numeric_cube():
    cube = np.full((2, 2, 2), "invalid")

    with pytest.raises(ValueError):
        compare_class_spectral_quality({"Healthy": cube})


def test_complex_cube():
    cube = np.ones((2, 2, 2), dtype=complex)

    with pytest.raises(ValueError):
        compare_class_spectral_quality({"Healthy": cube})


def test_invalid_class_name():
    with pytest.raises(ValueError):
        compare_class_spectral_quality({"": np.ones((2, 2, 2))})


def test_multiple_classes():
    cubes = {
        "Healthy": np.ones((3, 3, 4)),
        "Alternaria": np.full((3, 3, 4), 3.0),
        "Fusarium": np.full((3, 3, 4), 5.0)
    }

    result = compare_class_spectral_quality(cubes)

    assert len(result) == 3
    assert result["Healthy"]["mean"] == 1.0
    assert result["Alternaria"]["mean"] == 3.0
    assert result["Fusarium"]["mean"] == 5.0


def test_cube_values_are_converted_to_float32():
    cube = np.ones((2, 2, 3), dtype=np.float64)

    result = compare_class_spectral_quality({"Healthy": cube})

    assert result["Healthy"]["band_means"].dtype == np.float32