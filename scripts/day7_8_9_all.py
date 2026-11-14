import os, sys, glob, json, subprocess
from datetime import date

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print("Working folder:", os.getcwd())

import numpy as np
import pandas as pd
import yaml
import scipy.io as sio
from scipy.ndimage import binary_dilation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import calculate_default_transform, reproject, Resampling

os.makedirs("reports/figures", exist_ok=True)
os.makedirs("data/interim", exist_ok=True)
os.makedirs("data/metadata", exist_ok=True)

cfg = yaml.safe_load(open("config/config.yaml"))
PP = cfg["preprocessing"]
BAD_RANGES = cfg["sensor"]["bad_band_ranges_nm"]
CROP_SIZE = 512