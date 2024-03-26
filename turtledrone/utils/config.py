import doit
from doit import get_var
from doit.tools import run_once
from doit import create_after
import glob
import yaml
import os
import pandas as pd
import shutil
import tkinter as tk
from tkinter import filedialog
from pathlib import Path



cfg = None
CATALOG_DIR = None
def read_config():
    global cfg  
    global CATALOG_DIR
    config = {"config": get_var('config', 'NO')}
    if config['config']=='NO':
        root = tk.Tk()
        root.withdraw()
        config['config'] = filedialog.askopenfilename()
    with open(config['config'], 'r') as ymlfile:
        cfg = yaml.load(ymlfile, yaml.SafeLoader)
    CATALOG_DIR = os.path.dirname(config['config'])

def geturl(key):
    global cfg
    global CATALOG_DIR
    if cfg is None:
        read_config()
    return Path(cfg[key].format(CATALOG_DIR=CATALOG_DIR))

def getdest(file):
    country = os.path.basename(file).split('_')[0]
    site = os.path.basename(file).split('_')[1]
    sitecode = '_'.join(os.path.basename(file).split('_')[1:3])
    return geturl('output') /country / site /sitecode     