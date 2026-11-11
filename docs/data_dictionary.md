# Data Dictionary

## Files in data/processed
| File | Shape | Dtype | Description |
|---|---|---|---|
| X.npy | (N, B, 32, 32) | float32 | Preprocessed hyperspectral patches |
| y.npy | (N,) | int | Patch label (majority class) |
| coords.npy | (N, 2) | int | Patch top-left (row, col) in scene |
| split_indices.json | - | json | train/val/test indices (spatial block split) |

## Label mapping
| Value | Class |
|---|---|
| 0 | Healthy |
| 1 | Disease A |
| 2 | ... |

## Metadata (data/metadata/dataset_metadata.json)
wavelengths_nm, kept_band_indices, crs, patch_size, seed

## Preprocessing steps (order)
1. Reflectance scaling + clipping
2. Bad band removal
3. Cloud/shadow masking
4. Reprojection/resampling (GDAL)
5. Savitzky-Golay smoothing
6. Normalization
7. Patch extraction
8. Spatial split
