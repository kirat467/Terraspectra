import torch
from torch.utils.data import TensorDataset

from model_3dcnn import Crop3DCNN
from train_3dcnn import train_model, save_model, set_seed


def test_train_model(tmp_path):
    images = torch.rand(8, 1, 16, 16, 16)
    labels = torch.randint(0, 3, (8,))
    dataset = TensorDataset(images, labels)

    model = Crop3DCNN(num_classes=3)

    trained_model = train_model(
        model,
        dataset,
        epochs=2,
        batch_size=2,
        checkpoint_path=tmp_path / "checkpoint.pth"
    )

    assert isinstance(trained_model, Crop3DCNN)
    assert (tmp_path / "checkpoint.pth").exists()


def test_save_model(tmp_path):
    model = Crop3DCNN(num_classes=3)
    save_path = tmp_path / "3dcnn.pth"

    save_model(model, save_path)

    assert save_path.exists()


def test_resume_training(tmp_path):
    images = torch.rand(8, 1, 16, 16, 16)
    labels = torch.randint(0, 3, (8,))
    dataset = TensorDataset(images, labels)

    checkpoint = tmp_path / "resume_checkpoint.pth"
    model = Crop3DCNN(num_classes=3)

    train_model(
        model,
        dataset,
        epochs=1,
        batch_size=2,
        checkpoint_path=checkpoint
    )

    resumed_model = train_model(
        model,
        dataset,
        epochs=2,
        batch_size=2,
        checkpoint_path=checkpoint,
        resume=True
    )

    saved = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False
    )

    assert isinstance(resumed_model, Crop3DCNN)
    assert saved["epoch"] == 1


def test_set_seed():
    set_seed(42)
    first = torch.rand(5)

    set_seed(42)
    second = torch.rand(5)

    assert torch.equal(first, second)