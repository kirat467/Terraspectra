import numpy as np
import pytest

import ml.src.prepare_multiclass_data as prep


def test_prepare_multiclass_data(tmp_path, monkeypatch):

    (tmp_path / "Healthy_1.mat").touch()
    (tmp_path / "Alternaria_alternata_1.mat").touch()

    def fake_loader(path):
        return np.random.rand(
            10, 10, 5
        ).astype(np.float32)

    monkeypatch.setattr(
        prep,
        "load_mat_cube",
        fake_loader
    )

    dataset, counts = prep.prepare_multiclass_data(
        tmp_path,
        normalization="zscore"
    )

    assert len(dataset) == 2
    assert counts["Healthy"] == 1
    assert counts["Alternaria alternata"] == 1

    image, label = dataset[0]

    assert image.shape == (1, 5, 10, 10)
    assert label.item() == 0


def test_empty_dataset(tmp_path):

    with pytest.raises(ValueError):
        prep.prepare_multiclass_data(tmp_path)