from ast import Pass
import doit
from doit import get_var
from doit.tools import run_once
from doit import create_after
import glob
from numpy import int32
import yaml
import os
import pandas as pd
import shutil
import config
import subprocess
import json
import shlex
import ast
from io import StringIO
import time

def task_scan_serialnumbers():
    def get_serialnumbers(dependencies, targets):
        mountpoints = pd.DataFrame(json.loads(subprocess.getoutput('lsblk -J -o  NAME,SIZE,FSTYPE,TYPE,MOUNTPOINT'))['blockdevices'])
        mountpoints = mountpoints[~mountpoints.children.isna()]
        mountpoints =pd.DataFrame(mountpoints.children.apply(lambda x: x[0]).to_list())[['name','mountpoint']]
        paths = pd.DataFrame(subprocess.getoutput('udevadm info -q path -n $(ls /dev/s*1)').splitlines(),columns=['Path'])
        paths[['host','dev']]=paths.Path.str.extract(r'(?P<host>host\d+).*block\/(?P<dev>([^\\]+$))')[['host','dev']]
        paths['name'] =paths.dev.str.split('/',expand=True)[1]
        mountpoints =mountpoints.merge(paths, on='name', how='inner')
        mountpoints['CardUUID']=mountpoints.mountpoint.str.extract(r'(?P<CardUUID>[0-9A-F]{4}-[0-9A-F]{4})')
        mountpoints = mountpoints[~mountpoints.CardUUID.isna()]
        os.makedirs(os.path.dirname(targets[0]),exist_ok=True)
        path = f'{os.path.dirname(targets[0])}/{os.path.splitext(os.path.basename(targets[0]))[0].split("_")[1]}'
        mountpoints['Destination'] = path
        mountpoints.to_csv(targets[0],index=False)

    t = time.localtime()
    target = f"{config.geturl('imagehead')}/backup_{time.strftime('%Y%m%dT%H%M%S', t)}.csv"
    return {
        'actions':[get_serialnumbers],
        'targets':[target],
        'uptodate':[False],
        'clean':True,
    }



         
if __name__ == '__main__':
    import doit
    DOIT_CONFIG = {'check_file_uptodate': 'timestamp'}
    #print(globals())
    doit.run(globals())