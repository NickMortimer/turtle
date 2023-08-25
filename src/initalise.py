from genericpath import exists
import os
import glob
import doit
import glob
import os
import numpy as np
import pandas as pd
from doit import create_after
import numpy as np
import geopandas as gp
from drone import P4rtk
from read_rtk import read_mrk
from pyproj import Proj 
from shapely.geometry import MultiPoint
from utils import convert_wgs_to_utm
import config

def task_set_up():
    config.read_config()


def task_make_area_list():
        def split(path):
            print(path)
            keys =os.path.splitext(os.path.basename(path))[0].split('_')
            return {'SurveyCode':keys[0],'SurveyLongName':keys[1],'Type':keys[2],'File':path}
        
        def process_area_list(dependencies, targets):
            areas = pd.DataFrame.from_records([split(shape) for shape in dependencies])
            os.makedirs(os.path.dirname(targets[0]),exist_ok=True)
            areas.to_csv(targets[0],index=False)                
            
        file_dep = glob.glob(os.path.join(config.geturl('surveyarea'),'**/*.shp'),recursive=True)
        target = os.path.join(config.geturl('process'),'surveyareas.csv')
        return {
            'actions':[process_area_list],
            'file_dep':file_dep,
            'targets':[target],
            'clean':True,
        } 
        
@create_after(executed='make_area_list')         
def task_make_grids():
    def process_grid(dependencies, targets,gridsize=10):
        area = gp.read_file(dependencies[0])
        utmcode = convert_wgs_to_utm(area.iloc[0].geometry.exterior.coords.xy[0][0],area.iloc[0].geometry.exterior.coords.xy[1][0])
        crs = f'epsg:{utmcode}' 
        polygon = area.to_crs(crs).iloc[0].geometry
        eastings =polygon.exterior.coords.xy[0]
        northings =polygon.exterior.coords.xy[1]
        easting =np.arange(np.min(eastings) -np.min(eastings) % gridsize,np.max(eastings) -np.max(eastings) % gridsize,gridsize)
        northing=np.arange(np.min(northings) -np.min(northings) % gridsize,np.max(northings) -np.max(northings) % gridsize,gridsize)
        xi,yi = np.meshgrid(easting,northing)
        points =  MultiPoint(list(zip(xi.ravel(),yi.ravel())))
        p =points.intersection(polygon)
        d = {'Grid': ['25m'], 'geometry': [p]}
        df =gp.GeoDataFrame(d, crs=crs)
        df.to_file(targets[0])        
# surveyarea =area.iloc[0].geometry
# easting,northing =grid(surveyarea,20)
# xi,yi = np.meshgrid(easting,northing)        



    file_dep = glob.glob(os.path.join(config.geturl('surveyarea'),'**/*_AOI.shp'),recursive=True)
    for file in file_dep:
        target = file.replace('_AOI.shp','_Grid.shp')
        yield {
            'name':target,
            'actions':[(process_grid,[],{'gridsize':int(config.cfg['survey']['gridsize'])})],
            'file_dep':[file],
            'targets':[target],
            'clean':True,
        }         
        
if __name__ == '__main__':
    import doit
    DOIT_CONFIG = {'check_file_uptodate': 'timestamp'}
    #print(globals())
    doit.run(globals())           