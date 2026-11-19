"""Day 21: real EMIT scene -> orthorectified, officially masked crop for the pipeline."""
import os, sys, glob, json, re, shutil, subprocess
from datetime import date

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print("Working folder:", os.getcwd())