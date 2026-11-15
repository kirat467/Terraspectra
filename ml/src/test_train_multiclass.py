import pytest
import ml.src.train_multiclass as training


def test_missing_classes(monkeypatch):

    monkeypatch.setattr(
        training,
        "prepare_patch_dataset",
        lambda folder: (
            {},
            {
                "Healthy": 1,
                "Alternaria alternata": 1
            }
        )
    )

    with pytest.raises(ValueError, match="Missing classes"):
        training.train_multiclass_3dcnn()


def test_all_classes_present(monkeypatch):

    class FakeModel:
        pass

    counts = {
        "Healthy": 1,
        "Alternaria alternata": 1,
        "Alternaria solani": 1,
        "Botrytis cinerea": 1,
        "Fusarium oxysporum": 1
    }

    datasets = {
        "train": ["fake train data"],
        "val": ["fake validation data"],
        "test": ["fake test data"]
    }

    monkeypatch.setattr(
        training,
        "prepare_patch_dataset",
        lambda folder: (datasets, counts)
    )

    monkeypatch.setattr(
        training,
        "Crop3DCNN",
        lambda num_classes: FakeModel()
    )

    monkeypatch.setattr(
        training,
        "train_model",
        lambda model, dataset, **kwargs: model
    )

    monkeypatch.setattr(
        training,
        "save_model",
        lambda model, path: None
    )

    model, result = training.train_multiclass_3dcnn()

    assert isinstance(model, FakeModel)
    assert len(result) == 5