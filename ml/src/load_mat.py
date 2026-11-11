from pathlib import Path
import scipy.io
import h5py
import numpy as np
from .preprocessing import validate_hyperspectral_cube


def inspect_mat_file(file_path):
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"MATLAB file not found: {file_path}")

    datasets = []

    if h5py.is_hdf5(file_path):
        with h5py.File(file_path, "r") as file:
            def find_datasets(name, obj):
                if isinstance(obj, h5py.Dataset):
                    datasets.append({
                        "name": name,
                        "shape": obj.shape,
                        "dtype": str(obj.dtype)
                    })

            file.visititems(find_datasets)
    else:
        variables = scipy.io.whosmat(file_path)
        for name, shape, dtype in variables:
            datasets.append({
                "name": name,
                "shape": shape,
                "dtype": dtype
            })

    print("\n--- TerraSpectra MATLAB Dataset Report ---")
    for item in datasets:
        print(item)

    return datasets


def load_mat_cube(file_path, variable_name="Day_one_1"):
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    if h5py.is_hdf5(file_path):
        raise NotImplementedError(
            "MATLAB v7.3 loading needs separate dimension handling."
        )

    data = scipy.io.loadmat(
        file_path,
        variable_names=[variable_name]
    )

    if variable_name not in data:
        raise KeyError(f"Variable not found: {variable_name}")

    cube = validate_hyperspectral_cube(data[variable_name])

    print("MATLAB hyperspectral cube loaded successfully!")
    print("Shape:", cube.shape)
    print("Data type:", cube.dtype)

    return cube