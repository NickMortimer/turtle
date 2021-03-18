import os
import glob
import doit
import glob
import os
import numpy as np
from pandas.core.arrays.integer import Int64Dtype
import yaml
import pandas as pd
from doit import get_var
from doit.tools import run_once
from doit import create_after
import numpy as np
import plotly
import plotly.express as px
import geopandas as gp
import json

def task_process_labelme():
        def loadshapes(file):
            with open(file, "r") as read_file:
                data = json.load(read_file)
            data =pd.DataFrame(data['shapes'])
            data['FilePath'] =file             
            return(data)
        def process_labelme(dependencies, targets):
            data = pd.concat([loadshapes(file) for file in dependencies])
            data.to_csv(targets[0],index=False)       
            
        config = {"config": get_var('config', 'NO')}
        with open(config['config'], 'r') as ymlfile:
            cfg = yaml.load(ymlfile, yaml.SafeLoader)
        basepath = os.path.dirname(config['config'])
        for item in glob.glob(os.path.join(basepath,cfg['paths']['labelmesource']),recursive=True):
            file_dep = glob.glob(os.path.join(os.path.dirname(item),'*.json'))
            target =   os.path.join(os.path.dirname(item),'labelme.csv')           
            yield {
                'name':item,
                'actions':[process_labelme],
                'file_dep':file_dep,
                'targets':[target],
                'clean':True,
            }
        
        
def task_process_mergelabel():
        def process_mergelabel(dependencies, targets):
            data = pd.concat([pd.read_csv(file) for file in dependencies])
            data.to_csv(targets[0],index=False)       
            
        config = {"config": get_var('config', 'NO')}
        with open(config['config'], 'r') as ymlfile:
            cfg = yaml.load(ymlfile, yaml.SafeLoader)
        basepath = os.path.dirname(config['config'])
        file_dep = glob.glob(os.path.join(basepath,cfg['paths']['labelmesource'],'labelme.csv'),recursive=True)
        target = os.path.join(basepath,cfg['paths']['process'],'labelme.csv')        
        return {
            'actions':[process_mergelabel],
            'file_dep':file_dep,
            'targets':[target],
            'clean':True,
        }
        
def task_extact_mergelabel():
        def process_mergelabel(dependencies, targets):
            data = pd.concat([pd.read_csv(file) for file in dependencies])
            data.to_csv(targets[0],index=False)       
            
        config = {"config": get_var('config', 'NO')}
        with open(config['config'], 'r') as ymlfile:
            cfg = yaml.load(ymlfile, yaml.SafeLoader)
        basepath = os.path.dirname(config['config'])
        file_dep = glob.glob(os.path.join(basepath,cfg['paths']['labelmesource'],'labelme.csv'),recursive=True)
        target = os.path.join(basepath,cfg['paths']['process'],'labelme.csv')        
        return {
            'actions':[process_mergelabel],
            'file_dep':file_dep,
            'targets':[target],
            'clean':True,
        }            
        

        
 





if __name__ == '__main__':
    import doit

    #print(globals())
    doit.run(globals())