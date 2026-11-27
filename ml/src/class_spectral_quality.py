from collections.abc import Mapping

import numpy as np


def compare_class_spectral_quality(cubes):
    if not isinstance(cubes, Mapping):
        raise TypeError("Cubes must be provided as a dictionary or mapping")

    if not cubes:
        raise ValueError("No hyperspectral cubes provided")

    results = {}

    for class_name, cube in cubes.items():
        if not isinstance(class_name, str) or not class_name.strip():
            raise ValueError("Class names must be non-empty strings")

        try:
            raw_cube = np.asarray(cube)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid cube data: {class_name}"
            ) from exc

        if raw_cube.ndim != 3:
            raise ValueError(
                f"Invalid cube dimensions: {class_name}"
            )

        if raw_cube.size == 0:
            raise ValueError(
                f"Empty cube data: {class_name}"
            )

        if (
            not np.issubdtype(raw_cube.dtype, np.number)
            or np.issubdtype(raw_cube.dtype, np.complexfloating)
        ):
            raise ValueError(
                f"Cube must contain real numeric values: {class_name}"
            )

        try:
            cube = raw_cube.astype(np.float32)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"Unable to convert cube data: {class_name}"
            ) from exc

        if not np.isfinite(cube).all():
            raise ValueError(
                f"Cube contains non-finite values: {class_name}"
            )

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