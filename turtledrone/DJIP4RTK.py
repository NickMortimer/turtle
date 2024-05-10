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
from turtledrone.utils.utils import convert_wgs_to_utm
import config as config
from pathlib import Path
import re



def task_set_up():
    config.read_config()

 
def task_process_mergpos():
        def process_json(dependencies, targets):
            # dependencies.sort()
            source_file = list(filter(lambda x: 'exif.csv' in x, dependencies))[0]
            drone =pd.read_csv(source_file,index_col='Sequence',parse_dates=['TimeStamp']).sort_values('TimeStamp')
            mrk_file =list(filter(lambda x: '_Timestamp.MRK' in x, dependencies))
            mrk =read_mrk(mrk_file[0])
            utmcode =convert_wgs_to_utm(drone['Longitude'].mean(),drone['Latitude'].mean())
            utmproj =Proj(f'epsg:{utmcode:1.5}')            
            drone['Easting'],drone['Northing'] =utmproj(drone['Longitude'].values,drone['Latitude'].values)
            drone['LocalTime']=drone.TimeStamp
            drone['Interval']=drone.LocalTime.diff().dt.total_seconds()
            drone['GpsDist']=(drone['Northing'].diff()**2 + drone['Easting'].diff()**2)**0.5
            drone['GpsSpeed']=((drone['Northing'].diff()**2 + drone['Easting'].diff()**2)**0.5)/drone['Interval']
            drone['DroneSpeed'] = (drone['SpeedX']**2+drone['SpeedY']**2)**0.5
            drone['Leg'] =0
            drone.loc[drone['Interval']>8,'Leg'] =1
            drone['Leg'] = drone['Leg'].cumsum()
            drone['UtmCode'] =utmcode
            mrk['Easting'],mrk['Northing'] =utmproj(mrk['Longitude'].values,mrk['Latitude'].values)
            mrk['EllipsoideHight'] = pd.to_numeric(mrk.EllipsoideHight.str.split(',',expand=True)[0])
            mrk =mrk.add_suffix('Mrk')
            drone =drone.join(mrk[['UTCTimeMrk','EllipsoideHightMrk','LatitudeMrk','LongitudeMrk','EastingMrk','NorthingMrk']],rsuffix='Mrk')




            # g = drone.groupby('Leg')
            # drone =pd.concat([process_transect(leg) for name,leg in g])
            # mrk_file =list(filter(lambda x: '_Timestamp.MRK' in x, dependencies))
            # if mrk_file:
            #     mrk =read_mrk(mrk_file[0])
            #     if len(mrk)>0:
            # rtk_file=list(filter(lambda x: '_Timestamp.CSV' in x, dependencies))
            # if rtk_file:
            #     rtk =pd.read_csv(rtk_file[0],parse_dates=['GPST'],index_col=['Sequence'])
            #     rtk['Easting'],rtk['Northing'] =utmproj(rtk['longitude(deg)'].values,rtk['latitude(deg)'].values)
            #     rtk =rtk.add_suffix('Rtk')
            #     drone =drone.join(rtk,rsuffix='Rtk')
            # drone.set_index('TimeStamp',inplace=True)
            # drone.sort_index(inplace=True)
            # drone = drone[pd.notna(drone.index)]
            # drone.to_csv(targets[0],index=True)
            

        for item in config.geturl('imagesource').rglob('.'):
            source = list(item.glob('*.*'))
            if source:
                file_dep  =  list(filter(lambda x:  any(f in x.name for f in ['exif.csv','Timestamp']), source))
                fild_dep = list(filter(lambda x:os.stat(x).st_size > 0,file_dep))
                target =   item / 'position.csv'   
                if list(filter(lambda x: 'exif.csv' in x.name,file_dep)):        
                    yield {
                        'name':target,
                        'actions':[process_json],
                        'file_dep':file_dep,
                        'targets':[target],
                        'clean':True,
                    }    
@create_after(executed='process_mergpos', target_regex='*\position.json')    
def task_addpolygons():
    def process_polygons(dependencies, targets):
        def getpoly(item):
            if 'DewarpData' in item.keys():
                drone =P4rtk(pd.to_numeric(re.split(';|,',item.DewarpData)[1:]),crs,item.ImageWidth,item.ImageHeight)
            elif 'CalibratedOpticalCenterX' in item.keys():
                    drone =P4rtk([item.CalibratedFocalLength,
                                 item.CalibratedOpticalCenterX,item.CalibratedOpticalCenterY],crs,item.ImageWidth,item.ImageHeight)
            else:
                pass
            drone.setdronepos(item.Easting,item.Northing,item.RelativeAltitude,
                                  item.GimbalPitchDegree,item.GimbalRollDegree,item.GimbalYawDegree)
            return drone.getimagepolygon()
        data = pd.read_csv(dependencies[0],parse_dates=['TimeStamp'])
        crs = f'epsg:{int(data["UtmCode"][0])}'
        gdf = gp.GeoDataFrame(data, geometry=gp.points_from_xy(data.Easting, data.Northing),crs=crs)
        
        gdf['ImagePolygon'] = gdf.apply(getpoly,axis=1)
        gdf.to_csv(targets[0])
        
        
        

    # dewarp = pd.to_numeric(config.cfg['dewarp'] )
    # cal = pd.to_numeric(config.cfg['calibration'] )
    for file_dep in config.geturl('imagesource').rglob('position.csv'):
        target = file_dep.parent /'polygons.csv'
        yield {
            'name':file_dep,
            'actions':[process_polygons],
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

            # if 'LatitudeMrk' in drone.columns:
            #     rtkmask =drone.RtkFlag==1
            #     drone = drone[~drone.LatitudeMrk.isna()]
            drone.to_csv(list(targets)[0],index=True)
            
        file_dep = list(config.geturl('imagesource').rglob('polygons.csv'))
        if file_dep:
            processpath =config.geturl('process')
            os.makedirs(processpath,exist_ok=True)
            target = processpath / 'imagedata.csv'
            return {
                'actions':[process_xif],
                'file_dep':file_dep,
                'targets':[target],
                'clean':True,
            }
def run():
    import sys
    from doit.cmd_base import ModuleTaskLoader, get_loader
    from doit.doit_cmd import DoitMain
    DOIT_CONFIG = {'check_file_uptodate': 'timestamp',"continue": True}
    #print(globals())
    DoitMain(ModuleTaskLoader(globals())).run(sys.argv[1:])

if __name__ == '__main__':
    import doit
    DOIT_CONFIG = {'check_file_uptodate': 'timestamp'}
    #print(globals())
    doit.run(globals())
