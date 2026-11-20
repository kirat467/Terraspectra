"""Day 22: patch pipeline for the real EMIT scene, run it, build label helpers, ingest labels."""
import os, sys, re, json, glob, shutil, subprocess
from datetime import datetime, timedelta, date

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print("Working folder:", os.getcwd())