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
from utils import convert_wgs_to_utm




def task_make_area_list():
        def split(path):
            print(path)
            keys =os.path.splitext(os.path.basename(path))[0].split('_')
            return {'SurveyCode':keys[0],'SurveyLongName':keys[1],'Type':keys[2],'File':path}
        
        def process_area_list(dependencies, targets):
            areas = pd.DataFrame.from_records([split(shape) for shape in dependencies])
            os.makedirs(os.path.dirname(targets[0]),exist_ok=True)
            areas.to_csv(targets[0],index=False)                
            
        config = {"config": get_var('config', 'NO')}
        with open(config['config'], 'r') as ymlfile:
            cfg = yaml.load(ymlfile, yaml.SafeLoader)
        basepath = os.path.dirname(config['config'])
        file_dep = glob.glob(os.path.join(basepath,os.path.dirname(cfg['paths']['surveyarea']),'**/*.shp'),recursive=True)
        target = os.path.join(basepath,cfg['paths']['process'],'surveyareas.csv')
        return {
            'actions':[process_area_list],
            'file_dep':file_dep,
            'targets':[target],
            'clean':True,
        } 
        
        
# def task_make_grids():
#     def process_grid(dependencies, targets,gridsize=25):
#         area = gp.read_file(dependencies[0])
#         utmcode = convert_wgs_to_utm(area.iloc[0].geometry.exterior.coords.xy[0][0],area.iloc[0].geometry.exterior.coords.xy[1][0])
#         crs = f'epsg:{utmcode}'
#         polygon = area.to_crs(crs).iloc[0].geometry
#         eastings =polygon.exterior.coords.xy[0]
#         northings =polygon.exterior.coords.xy[1]
#         easting =np.arange(np.min(eastings) -np.min(eastings) % gridsize,np.max(eastings) -np.max(eastings) % gridsize,gridsize)
#         northing=np.arange(np.min(northings) -np.min(northings) % gridsize,np.max(northings) -np.max(northings) % gridsize,gridsize)
#         xi,yi = np.meshgrid(easting,northing)
#         points =  MultiPoint(list(zip(xi.ravel(),yi.ravel())))
#         p =points.intersection(polygon)
#         d = {'Grid': ['25m'], 'geometry': [p]}
#         df =gp.GeoDataFrame(d, crs=crs)
#         df.to_file(targets[0])        
# # surveyarea =area.iloc[0].geometry
# # easting,northing =grid(surveyarea,20)
# # xi,yi = np.meshgrid(easting,northing)        


#     config = {"config": get_var('config', 'NO')}
#     with open(config['config'], 'r') as ymlfile:
#         cfg = yaml.load(ymlfile, yaml.SafeLoader)
#     basepath = os.path.dirname(config['config'])
#     file_dep = glob.glob(os.path.join(basepath,os.path.dirname(cfg['paths']['surveyarea']),'**/*_AOI.shp'),recursive=True)
#     for file in file_dep:
#         target = file.replace('_AOI.shp','_Grid.shp')
#         yield {
#             'name':target,
#             'actions':[process_grid],
#             'file_dep':[file],
#             'targets':[target],
#             'clean':True,
#         }         
        
if __name__ == '__main__':
    import doit
    DOIT_CONFIG = {'check_file_uptodate': 'timestamp'}
    #print(globals())
    doit.run(globals())           