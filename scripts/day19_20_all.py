import os, sys, json, glob, ast, hashlib, shutil, subprocess, zipfile
from datetime import date
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print("Working folder:", os.getcwd())