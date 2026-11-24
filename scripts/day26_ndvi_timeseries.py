"""Day 26: Sentinel-2 NDVI time series + weather lags -> forecasting-ready table."""
import os, sys, json, subprocess, urllib.request
from datetime import date, datetime, timedelta

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print("Working folder:", os.getcwd())