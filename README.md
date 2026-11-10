
# TerraSpectra – Machine Learning Module

## 1. Project Overview

TerraSpectra is a precision agriculture project focused on hyperspectral crop health analysis.

The Machine Learning module aims to process hyperspectral images and develop deep learning models for crop health classification and early disease detection.

## 2. Technologies Used

- Python 3.12
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Rasterio
- H5py
- Spectral Python
- CUDA-enabled GPU support

## 3. Machine Learning Models

### 3D-CNN

A 3D Convolutional Neural Network designed to learn spatial and spectral features from hyperspectral image data.

File: `src/model_3dcnn.py`

### Vision Transformer (ViT)

A Transformer-based model that processes sequences of prepared image features for classification.

File: `src/model_vit.py`

Note: The current ViT implementation uses feature sequences as input. Raw hyperspectral image patch preparation is still pending.

## 4. Module Structure

```text
ml/
├── dataset/
│   ├── raw/
│   ├── processed/
│   ├── README.md
│   └── data_info.txt
├── notebooks/
│   └── data_exploration.ipynb
├── src/
│   ├── preprocessing.py
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

## 5. Implemented Features

- Hyperspectral image validation
- Dataset loading and tensor conversion
- 3D-CNN model architecture
- 3D-CNN training pipeline
- Vision Transformer architecture
- ViT training pipeline
- Model evaluation using accuracy, precision, recall and F1-score
- Model prediction utility
- Model saving functionality

## 6. Testing Status

The following software tests have been performed using synthetic data:

- Dataset loader integration test
- 3D-CNN forward-pass test
- 3D-CNN training test
- ViT architecture test
- ViT training test
- Model evaluation test
- Model saving test

These tests verify software functionality only. They do not establish real-world crop disease detection accuracy.

## 7. Dataset Status

The actual hyperspectral dataset is currently awaited from the data collection team.

Expected dataset information:

- Hyperspectral image files
- Spectral band information
- Crop types
- Disease labels
- Image dimensions
- Wavelength information
- Geographic metadata

Actual model training and performance evaluation will begin after the dataset is collected and validated.

## 8. Future Work

- Load and inspect real hyperspectral datasets
- Implement dataset-specific preprocessing
- Apply Principal Component Analysis (PCA)
- Prepare image patches and ViT input features
- Train models using labelled crop data
- Compare 3D-CNN and ViT performance
- Perform model validation and testing
- Integrate predictions with the TerraSpectra application

## 9. Team Responsibilities

- Kirat: Machine Learning and deep learning models
- Kailash: Hyperspectral data collection
- Sharath: Frontend development and GIS visualization

---

**Project:** TerraSpectra – Hyperspectral Crop Disease Forecasting
