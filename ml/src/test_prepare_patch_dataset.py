import numpy as np
import pytest

import ml.src.prepare_patch_dataset as prep


def test_prepare_patch_dataset(tmp_path, monkeypatch):

    classes = [
        "Healthy",
        "Alternaria alternata",
        "Alternaria solani",
        "Botrytis cinerea",
        "Fusarium oxysporum"
    ]

    inventory = {}

    for name in classes:
        filename = name.replace(" ", "_") + "_1.mat"
        (tmp_path / filename).touch()
        inventory[name] = [filename]

    monkeypatch.setattr(
        prep,
        "scan_dataset",
        lambda folder: inventory
    )

    monkeypatch.setattr(
        prep,
        "load_mat_cube",
        lambda path: np.random.rand(
            64, 256, 5
        ).astype(np.float32)
    )

    monkeypatch.setattr(
        prep,
        "get_label",
        lambda name: classes.index(name)
    )

    datasets, counts = prep.prepare_patch_dataset(
        tmp_path,
        normalization=None
    )

    # Check dataset sizes
    assert len(datasets["train"]) == 40
    assert len(datasets["val"]) == 10
    assert len(datasets["test"]) == 10

    # Check five original classes
    assert sum(counts.values()) == 5

    # Check patch dimensions
    image, label = datasets["train"][0]

    assert image.shape == (1, 5, 32, 32)
    assert label.item() in range(5)


def test_missing_class(tmp_path, monkeypatch):

    monkeypatch.setattr(
        prep,
        "scan_dataset",
        lambda folder: {}
    )

    with pytest.raises(
        ValueError,
        match="Missing dataset class"
    ):
        prep.prepare_patch_dataset(tmp_path)