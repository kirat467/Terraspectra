import numpy as np


def analyze_data_quality(cube):
    cube = np.asarray(cube, dtype=np.float32)

    if cube.ndim != 3:
        raise ValueError("Cube must be 3-dimensional")

    if cube.size == 0:
        raise ValueError("Cube cannot be empty")

    if not np.isfinite(cube).all():
        raise ValueError("Cube contains NaN or infinite values")

    band_means = cube.mean(axis=(0, 1))
    band_stds = cube.std(axis=(0, 1))

    results = {
        "shape": cube.shape,
        "minimum": float(cube.min()),
        "maximum": float(cube.max()),
        "mean": float(cube.mean()),
        "standard_deviation": float(cube.std()),
        "band_means": band_means,
        "band_stds": band_stds,
        "constant_bands": np.where(band_stds == 0)[0].tolist()
    }

    print("\n--- TerraSpectra Data Quality Report ---")
    print("Image Shape:", results["shape"])
    print("Minimum:", results["minimum"])
    print("Maximum:", results["maximum"])
    print("Mean:", results["mean"])
    print("Standard Deviation:", results["standard_deviation"])
    print("Constant Bands:", results["constant_bands"])

    return results