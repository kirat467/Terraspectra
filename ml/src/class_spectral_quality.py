import numpy as np


def compare_class_spectral_quality(cubes):
    if not cubes:
        raise ValueError("No hyperspectral cubes provided")

    results = {}

    for class_name, cube in cubes.items():
        cube = np.asarray(cube, dtype=np.float32)

        if cube.ndim != 3:
            raise ValueError(f"Invalid cube dimensions: {class_name}")

        if cube.size == 0 or not np.isfinite(cube).all():
            raise ValueError(f"Invalid data: {class_name}")

        results[class_name] = {
            "mean": float(cube.mean()),
            "std": float(cube.std()),
            "minimum": float(cube.min()),
            "maximum": float(cube.max()),
            "band_means": cube.mean(axis=(0, 1))
        }

    print("\n--- Class Spectral Quality Comparison ---")

    for class_name, stats in results.items():
        print(f"\n{class_name}")
        print("Mean:", stats["mean"])
        print("Standard Deviation:", stats["std"])
        print("Minimum:", stats["minimum"])
        print("Maximum:", stats["maximum"])

    return results