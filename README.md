# TerraSpectra – Machine Learning Module

## Project Overview
TerraSpectra is a precision agriculture project focused on using hyperspectral imagery and machine learning to identify crop health conditions and support early disease detection.

Hyperspectral images contain multiple narrow spectral bands that can provide information about crop conditions.

## My Contribution
**Role: Machine Learning Developer – Kirat**

My responsibilities include:
- Developing a 3D Convolutional Neural Network (3D-CNN).
- Developing a Transformer-based classification model.
- Building hyperspectral image preprocessing utilities.
- Implementing Principal Component Analysis (PCA).
- Creating dataset analysis and splitting utilities.
- Developing model training, evaluation and prediction pipelines.

## Technologies Used
- Python
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Rasterio
- HDF5
- Spectral Python
- CUDA

## ML Module Structure

```text
ml/
├── dataset/
│   ├── raw/
│   ├── processed/
│   ├── README.md
│   └── data_info.txt
├── notebooks/
├── src/
│   ├── preprocessing.py
│   ├── pca.py
│   ├── explore_data.py
│   ├── analyze_dataset.py
│   ├── split_data.py
│   ├── dataset.py
│   ├── model_3dcnn.py
│   ├── train_3dcnn.py
│   ├── model_vit.py
│   ├── train_vit.py
│   ├── evaluate.py
│   └── predict.py
├── models/
├── requirements.txt
└── README.md
```

## Model Approaches

### 1. 3D-CNN
Designed to process spatial and spectral information from hyperspectral image patches.

### 2. Transformer-based Model
Uses a Transformer encoder to process sequences of spectral features for classification.

## Current Development Status

- [x] Python ML environment setup
- [x] Hyperspectral image validation
- [x] PCA implementation
- [x] Data exploration utilities
- [x] Dataset splitting utility
- [x] Dataset analysis utility
- [x] 3D-CNN architecture and training pipeline
- [x] Transformer architecture and training pipeline
- [x] Evaluation and prediction modules
- [x] Synthetic-data software tests
- [ ] Integration of actual hyperspectral dataset
- [ ] Real-data model training
- [ ] Model validation and performance analysis

## Dataset Status
The actual hyperspectral dataset and its labels are pending collection.

All current tests use synthetic data to verify software functionality. No real crop-disease accuracy or forecasting performance has been established.

## Future Work
- Integrate the collected hyperspectral dataset.
- Perform appropriate spatial train-validation-test splitting.
- Train and tune the classification models.
- Compare model performance using evaluation metrics.
- Generate crop-health prediction outputs for integration with the project dashboard.

---
**Project:** TerraSpectra – Hyperspectral Crop Disease Forecasting  
**ML Developer:** Kirat