import pytest

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

    assert all(
        len(samples) == 0
        for samples in result.values()
    )


def test_all_classes_are_present_in_inventory(tmp_path):

    result = scan_dataset(tmp_path)

    expected_classes = [
        "Healthy",
        "Alternaria alternata",
        "Alternaria solani",
        "Botrytis cinerea",
        "Fusarium oxysporum"
    ]

    assert list(result.keys()) == expected_classes


def test_case_insensitive_filename_detection(tmp_path):

    (tmp_path / "HEALTHY_1.MAT").touch()
    (tmp_path / "Alternaria_Alternata_1.mat").touch()

    result = scan_dataset(tmp_path)

    assert len(result["Healthy"]) == 1
    assert len(result["Alternaria alternata"]) == 1


def test_unrelated_files_are_ignored(tmp_path):

    (tmp_path / "Healthy_1.mat").touch()
    (tmp_path / "readme.txt").touch()
    (tmp_path / "image.png").touch()

    result = scan_dataset(tmp_path)

    total = sum(
        len(samples)
        for samples in result.values()
    )

    assert total == 1


def test_unknown_mat_files_are_ignored(tmp_path):

    (tmp_path / "unknown_sample.mat").touch()

    result = scan_dataset(tmp_path)

    total = sum(
        len(samples)
        for samples in result.values()
    )

    assert total == 0


def test_inventory_files_are_sorted(tmp_path):

    (tmp_path / "Healthy_3.mat").touch()
    (tmp_path / "Healthy_1.mat").touch()
    (tmp_path / "Healthy_2.mat").touch()

    result = scan_dataset(tmp_path)

    assert result["Healthy"] == [
        "Healthy_1.mat",
        "Healthy_2.mat",
        "Healthy_3.mat"
    ]


def test_missing_folder_raises_error(tmp_path):

    missing_folder = tmp_path / "missing"

    with pytest.raises(FileNotFoundError):
        scan_dataset(missing_folder)


def test_file_instead_of_folder_raises_error(tmp_path):

    file_path = tmp_path / "sample.mat"
    file_path.touch()

    with pytest.raises(NotADirectoryError):
        scan_dataset(file_path)


def test_inventory_total_sample_count(tmp_path):

    (tmp_path / "Healthy_1.mat").touch()
    (tmp_path / "Healthy_2.mat").touch()
    (tmp_path / "Botrytis_cinerea_1.mat").touch()

    result = scan_dataset(tmp_path)

    total = sum(
        len(samples)
        for samples in result.values()
    )

    assert total == 3


def test_inventory_does_not_include_subfolder_files(tmp_path):

    subfolder = tmp_path / "nested"
    subfolder.mkdir()

    (subfolder / "Healthy_1.mat").touch()

    result = scan_dataset(tmp_path)

    assert len(result["Healthy"]) == 0