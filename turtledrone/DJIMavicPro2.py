from genericpath import exists
import os
import glob
import doit
import glob
import os
import pandas as pd
from doit import create_after
import numpy as np
import geopandas as gp
from turtledrone.process.DJIMavic2 import Mavic2
from pyproj import Proj 
from utils.utils import convert_wgs_to_utm
import utils.config as config






 
def task_process_mergpos():
        def process_json(dependencies, targets):
            # dependencies.sort()
            source_file = list(filter(lambda x: 'exif.csv' in x, dependencies))[0]
            drone =pd.read_csv(source_file,index_col='Sequence',parse_dates=['TimeStamp'])
            utmcode =convert_wgs_to_utm(drone['Longitude'].mean(),drone['Latitude'].mean())
            utmproj =Proj(f'epsg:{utmcode:1.5}')            
            drone['Easting'],drone['Northing'] =utmproj(drone['Longitude'].values,drone['Latitude'].values)
            drone['LocalTime']=drone.TimeStamp
            drone['Interval']=drone.LocalTime.diff().dt.total_seconds()
            drone['GpsDist']=(drone['Northing'].diff()**2 + drone['Easting'].diff()**2)**0.5
            drone['GpsSpeed']=((drone['Northing'].diff()**2 + drone['Easting'].diff()**2)**0.5)/drone['Interval']
            drone['Leg'] =0
            drone['UtmCode'] =utmcode
            drone.set_index('TimeStamp',inplace=True)
            drone.sort_index(inplace=True)
            drone = drone[pd.notna(drone.index)]
            drone.to_csv(list(targets)[0],index=True)
            

        for item in glob.glob(config.geturl('imagesource'),recursive=True):
            source = item
            file_dep  =  list(filter(lambda x:  any(f in x for f in ['exif.csv','Timestamp']), glob.glob(os.path.join(source,'*.*'))))
            fild_dep = list(filter(lambda x:os.stat(x).st_size > 0,file_dep))
            if file_dep:
                target =   os.path.join(source,'position.csv')           
                yield {
                    'name':source,
                    'actions':[process_json],
                    'file_dep':file_dep,
                    'targets':[target],
                    'clean':True,
                }    
@create_after(executed='process_mergpos', target_regex='*\position.json')    
def task_addpolygons():
    def process_polygons(dependencies, targets,dewarp):
        def getpoly(item):
            drone.setdronepos(item.Easting,item.Northing,item.RelativeAltitude,
                                  (90+item.GimbalPitchDegree)*-1,item.GimbalRollDegree,item.GimbalYawDegree)
            return drone.getimagepolygon()
        data = pd.read_csv(dependencies[0],parse_dates=['TimeStamp'])
        crs = f'epsg:{int(data["UtmCode"][0])}'
        gdf = gp.GeoDataFrame(data, geometry=gp.points_from_xy(data.Easting, data.Northing),crs=crs)
        drone =Mavic2(crs)
        gdf['ImagePolygon'] = gdf.apply(getpoly,axis=1)
        gdf.to_csv(targets[0])
        
        
        

    dewarp = pd.to_numeric(config.cfg['survey']['dewarp'] )
    for file_dep in glob.glob(os.path.join(config.geturl('imagesource'),'position.csv'),recursive=True):
        target = os.path.join(os.path.dirname(file_dep),'polygons.csv')   
        yield {
            'name':file_dep,
            'actions':[(process_polygons, [],{'dewarp':dewarp})],
            'file_dep':[file_dep],
            'targets':[target],
            'clean':True,
        }       
@create_after(executed='addpolygons', target_regex='*\position.json')       
def task_merge_xif():
        def process_xif(dependencies, targets):
            target = list(targets)[0]
            os.makedirs(os.path.dirname(target),exist_ok=True)
            drone = pd.concat([pd.read_csv(file,index_col='TimeStamp',parse_dates=['TimeStamp']) 
                            for file in list(dependencies)]) 
            drone.sort_index(inplace=True)
            drone.to_csv(list(targets)[0],index=True)
            

        searchpath = os.path.join(config.geturl('imagesource'),'polygons.csv')
        file_dep = glob.glob(searchpath,recursive=True)
        processpath =config.geturl('process')
        os.makedirs(processpath,exist_ok=True)
        target = os.path.join(processpath,'imagedata.csv')
        return {
            'actions':[process_xif],
            'file_dep':file_dep,
            'targets':[target],
            'clean':True,
        }


if __name__ == '__main__':
    import doit
    DOIT_CONFIG = {'check_file_uptodate': 'timestamp'}
    #print(globals())
    doit.run(globals())
