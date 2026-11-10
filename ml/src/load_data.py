
from pathlib import Path

import rasterio
import h5py

from .preprocessing import validate_hyperspectral_cube


def load_tiff(file_path):

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with rasterio.open(file_path) as src:

        cube = src.read()
        cube = cube.transpose(1, 2, 0)

    cube = validate_hyperspectral_cube(cube)

    print("TIFF loaded successfully!")
    print("Image shape (H, W, Bands):", cube.shape)

    return cube


def inspect_hdf5(file_path):

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    datasets = []

    with h5py.File(file_path, "r") as file:

        def find_datasets(name, obj):

            if isinstance(obj, h5py.Dataset):

                datasets.append({
                    "name": name,
                    "shape": obj.shape,
                    "dtype": str(obj.dtype)
                })

        file.visititems(find_datasets)

    print("HDF5 datasets found:")

    for item in datasets:
        print(item)

    return datasets
