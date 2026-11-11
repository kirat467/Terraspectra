from dataset_inventory import scan_dataset


def test_inventory_counts(tmp_path):
    (tmp_path / "Healthy_1.mat").touch()
    (tmp_path / "Alternaria_alternata_1.mat").touch()
    (tmp_path / "Alternaria_alternata_2.mat").touch()

    result = scan_dataset(tmp_path)

    assert len(result["Healthy"]) == 1
    assert len(result["Alternaria alternata"]) == 2
    assert len(result["Botrytis cinerea"]) == 0


def test_inventory_empty_folder(tmp_path):
    result = scan_dataset(tmp_path)

    assert all(len(samples) == 0 for samples in result.values())