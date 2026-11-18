# Data notes, Nov 16: alignment, patches and the split

## 1. The cube and the dashboard use the same grid
`python data/check_alignment.py data/generated/scene_2026-11-10.tif` compares the GeoTIFF with `farm_meta.json`: the four edges match to 6 decimals, pixels are 15.00 m, and every field corner falls within 0.005 pixel of a pixel edge. A deliberately shifted scene (3 pixels east) is caught by the tests. When real scenes arrive, run this first: a shifted map would paint risk on the wrong field.

## 2. Why the split is by block
Neighbouring 9 x 9 patches overlap and look alike. With a random split, a patch in the test set has a near-twin in the training set, and accuracy looks far better than it is. `data/make_patches.py` instead:
- splits the farm into 18 x 18 pixel blocks and gives each whole block to train (60%), validation (20%) or test (20%);
- hands out diseased blocks first so every split contains early and visible stress;
- only takes patches that sit inside one block with a 2-pixel margin, so no pixel is shared between splits;
- keeps a place in the same split on all 6 dates, so the model is never tested on a place it trained on at another date;
- fits PCA on training pixels only.

Result on the simulated scans: 6,180 patches (train 3,984, val 1,080, test 1,116). Stress is rare: test has 71 early-stress and 29 visible-stress patches, validation has 75 early and none visible. Use class weights or oversampling, and report per-class results, not only accuracy.

`ml/src/split_data.py` splits at random (`train_test_split`). Use it only for data that is not made of overlapping patches.

## 3. PCA to 20 components
The first 3 components hold 99.8% of the variance on the simulated data (`docs/figures/pca_variance.png`). That is a property of the simulator: it mixes only a few processes (soil, leaf, water, stress). Real hyperspectral scenes usually need more components, and the disease signal may sit in components with small variance. Check on real data before fixing the number at 20, and standardise each component before training.

## 4. Using the patches with the CNN
`make_patches.to_cnn_layout(X)` turns (N, 9, 9, 20) into (N, 1, 20, 9, 9), the layout `Crop3DCNN` takes. This layout was checked against the model's input shape in `ml/src/test_predict.py` but the model itself has not been trained on these patches yet.
