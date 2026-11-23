# First training run: 3D-CNN on the simulated patches (Nov 18)

`python ml/src/train_on_scenes.py` trains `Crop3DCNN` (Kirat's model, unchanged) on the 9 x 9 x 20 patches from `data/make_patches.py`, using `train_3dcnn.train_model` (early stopping, learning-rate scheduler, checkpoints). Runs in about 15 seconds on a CPU. Numbers are in `docs/cnn_run_results.json`, the loss curve in `docs/figures/cnn_loss_curve.png`.

## Setup
- Split: farm blocks, so test patches never touch training pixels (see `docs/week2_data_notes.md`). Train 3,984 patches, validation 1,080, test 1,116.
- Each PCA component is standardised with training statistics only.
- Rare classes are repeated in the training set only (early stress to 35% of the healthy count). Validation and test are untouched.
- Test blocks were used once, at the end.

## Results on the test blocks
| Class | Test patches | Precision | Recall | F1 |
|---|---|---|---|---|
| Healthy | 1,016 | 0.991 | 0.999 | 0.995 |
| Early stress | 71 | 0.738 | 0.873 | 0.800 |
| Visible stress | 29 | 1.000 | 0.276 | 0.432 |

Overall accuracy is 97.2%, but accuracy hides the problem: only 8 of the 29 visible-stress patches were found, and the other 21 were called early stress (confusion matrix rows are the true class: healthy 1015/1/0, early 9/62/0, visible 0/21/8). Numbers come from `docs/cnn_run_results.json` and were re-checked by evaluating the saved model on the test patches (Nov 23).

*Correction (Nov 23): an earlier version of this table showed accuracy 96.1% and zero visible-stress recall, which did not match the saved run.*

How well each method separates stressed patches from healthy ones (AUC, same test patches, 1.0 is perfect):
| Method | Early vs healthy | Visible vs healthy |
|---|---|---|
| 3D-CNN | 0.997 | 1.000 |
| Full-spectrum ridge | 0.997 | 0.999 |
| NDVI | 0.897 | 0.995 |
| Red-edge position | 0.914 | 0.861 |
| Water index (NDII) | 0.903 | 0.861 |

So the CNN finds early stress far better than NDVI, but a simple linear model on all bands does just as well. On this simulated data the extra power of the CNN is not needed; real data will decide.

## Why visible stress failed
Only 37 visible-stress patches exist in training and validation has none, so early stopping never saw the class and the model only partly learned to separate it from early stress. This is a data problem (6 simulated weeks, one main outbreak), not a model one. Fixes: simulate a longer season, or use more patch positions in the diseased blocks.

## CNN output in the dashboard
`python scripts/cnn_to_dashboard.py --out frontend/public/data/risk_cnn` runs the trained model over the six scans and writes risk files the dashboard reads (`docs/screenshots/07_cnn_output.png`). At risk 0.6 on the last scan it covers 98% of the truly stressed pixels with 87% precision.

Two problems visible in the picture:
- The CNN is over-confident, so the map is almost on or off and Field A3 (faint early stress) looks as severe as Field B2 (visible stress). The risk shown is "chance of stress", not how bad it is. A severity output or probability calibration would fix this.
- The 21-day forecast is still the simple trend rule from `risk_map.py`.

## Limits
Simulated data and simulated labels: the model learns the simulator's rules. Single seed, single run. The 3D-CNN architecture is Kirat's and was not changed.
