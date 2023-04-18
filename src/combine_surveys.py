from genericpath import exists
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
from geopandas.tools import sjoin
from drone import P4rtk
from read_rtk import read_mrk
from pyproj import Proj 
from doit.tools import check_timestamp_unchanged
import shutil
from shapely.geometry import Polygon
import shapely.wkt
from shapely.geometry import MultiPoint
import xarray as xr
import rasterio as rio
from  utils import convert_wgs_to_utm
import config


def task_data_summary():
        def process_surveys(dependencies, targets):
            drone =pd.concat([pd.read_csv(file,index_col='SurveyId',parse_dates=['StartTime','EndTime']) for file in dependencies])
            drone['ElapsedTime'] = drone.EndTime - drone.StartTime 
            drone['ElapsedTime'] = (drone['ElapsedTime'].dt.total_seconds()).astype(int) /60
            drone['SurveyRate'] =  drone.Area /drone.ElapsedTime 
            drone = drone.sort_index()
            drone.to_csv(targets[0]) 
 
            
        surveys = os.path.join(config.geturl('surveysource'),'*survey_area_data_summary.csv')
        file_dep = glob.glob(surveys,recursive=True)
        os.makedirs(config.geturl('reports'),exist_ok=True)
        target = f"{config.geturl('reports')}/survey_area_data_summary.csv"
        return {
            'actions':[process_surveys],
            'file_dep':file_dep,
            'targets':[target],
            'clean':True,
        } 


if __name__ == '__main__':
    import doit
    DOIT_CONFIG = {'check_file_uptodate': 'timestamp'}
    #print(globals())
    doit.run(globals())
