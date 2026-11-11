import pytest

from label_mapping import (
    get_label,
    get_class_name,
    get_all_classes
)


def test_healthy_label():
    assert get_label("Healthy") == 0


def test_disease_label():
    assert get_label("Alternaria alternata") == 1


def test_class_name():
    assert get_class_name(4) == "Fusarium oxysporum"


def test_all_classes():
    assert len(get_all_classes()) == 5


def test_invalid_class():
    with pytest.raises(ValueError):
        get_label("Unknown disease")


def test_invalid_label():
    with pytest.raises(ValueError):
        get_class_name(10)