"""Day 23: validate the final dataset with baseline models + leakage demo + week 3 write-up."""
import os, sys, json, subprocess
from datetime import date

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print("Working folder:", os.getcwd())