import doit
from doit import get_var
from doit.tools import run_once
from doit import create_after
import glob
import yaml
import os
import pandas as pd
import shutil

cfg = None
basepath = None

def task_read_config():
    global cfg
    global basepath
    config = {"config": get_var('config', 'NO')}
    with open(config['config'], 'r') as ymlfile:
        cfg = yaml.load(ymlfile, yaml.SafeLoader)
    basepath = os.path.dirname(config['config'])