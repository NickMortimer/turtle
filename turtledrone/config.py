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



# cfg = None
# CATALOG_DIR = None
# def read_config():
#     global cfg  
#     global CATALOG_DIR
#     config = {"config": get_var('config', 'NO')}
#     if config['config']=='NO':
#         root = tk.Tk()
#         root.withdraw()
#         config['config'] = filedialog.askopenfilename()
#     with open(config['config'], 'r') as ymlfile:
#         cfg = yaml.load(ymlfile, yaml.SafeLoader)
#     CATALOG_DIR = os.path.dirname(config['config'])

# def geturl(key):
#     global cfg
#     global CATALOG_DIR
#     if cfg is None:
#         read_config()
#     return Path(cfg[key].format(CATALOG_DIR=CATALOG_DIR))

# def getdest(file):
#     country = os.path.basename(file).split('_')[0]
#     site = os.path.basename(file).split('_')[1]
#     sitecode = '_'.join(os.path.basename(file).split('_')[1:3])
#     return geturl('output') /country / site /sitecode     


# config.py
from pathlib import Path
import yaml
import tkinter as tk
from tkinter import filedialog

class Config:
    def __init__(self, path=None):
        self.cfg = {}
        self.catalog_dir = None
        self.path = path
        if self.path:
            self.load(self.path)

    def ask_config_file(self):
        root = tk.Tk()
        root.withdraw()
        return filedialog.askopenfilename()

    def load(self, path):
        # this should update the cfg instead of replacing it
        if path:
            self.catalog_dir = Path(path).parent
            with open(path, "r") as ymlfile:
                cfg = yaml.load(ymlfile, Loader=yaml.SafeLoader)
            for key, value in cfg.items():
                self.cfg[key] = value

    def get(self, key, default=None):
        return self.cfg.get(key, default)

    def set(self, key, value):
        self.cfg[key] = value

    def get_url(self, key):
        value = self.cfg[key]
        return Path(value.format(CATALOG_DIR=self.catalog_dir))

    def get_destination(self, file_path):
        parts = Path(file_path).name.split('_')
        country = parts[0]
        site = parts[1]
        sitecode = "_".join(parts[1:3])
        return self.geturl("output") / country / site / sitecode

    def save(self, path=None):
        path = path or self.path
        if path:
            with open(path, "w") as ymlfile:
                yaml.dump(self.cfg, ymlfile)

# single shared instance
cfg = Config(path=get_var('config', None))
