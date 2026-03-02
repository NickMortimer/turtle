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
from pathlib import Path
import cameratransform as ct
import re
from shapely.geometry import point
from shapely.geometry import Polygon



 
def task_process_mergpos():
        def process_json(dependencies, targets):
            # dependencies.sort()
            source_file = next(f for f in dependencies if 'exif.csv' in str(f))
            drone =pd.read_csv(source_file,index_col='Sequence',parse_dates=['TimeStamp']).sort_values('TimeStamp')
            mrk_file = next((f for f in dependencies if '_Timestamp.MRK' in str(f)), None)
            rtk_file = next((f for f in dependencies if '_Timestamp.CSV' in str(f)), None)
            utmcode =convert_wgs_to_utm(drone['Longitude'].mean(),drone['Latitude'].mean())
            utmproj =Proj(f'epsg:{utmcode:1.5}')            
            drone['EastingXif'],drone['NorthingXif'] =utmproj(drone['Longitude'].values,drone['Latitude'].values)
            drone['LocalTime']=drone.TimeStamp
            drone['Interval']=drone.LocalTime.diff().dt.total_seconds()
            drone['GpsDist']=(drone['NorthingXif'].diff()**2 + drone['EastingXif'].diff()**2)**0.5
            drone['GpsSpeed']=((drone['NorthingXif'].diff()**2 + drone['EastingXif'].diff()**2)**0.5)/drone['Interval']
            if 'SpeedX' in drone.columns:
                drone['DroneSpeed'] = (drone['SpeedX']**2+drone['SpeedY']**2)**0.5
            drone['Leg'] =0
            drone.loc[drone['Interval']>8,'Leg'] =1
            drone['Leg'] = drone['Leg'].cumsum()
            drone['UtmCode'] =utmcode
            if rtk_file:
                rtk =pd.read_csv(rtk_file,parse_dates=['GPST'],index_col=['Sequence'])
                rtk['Easting'],rtk['Northing'] =utmproj(rtk['longitude(deg)'].values,rtk['latitude(deg)'].values)
                #ok lets replace the column names ( with _ characters
                rtk.columns = rtk.columns.str.replace(r'[(]', '_', regex=True)
                rtk.columns = rtk.columns.str.replace(r'[)]', '', regex=True)
                rtk =rtk.add_suffix('_rtk')
                drone =drone.join(rtk,rsuffix='_rtk')            
            if mrk_file:
                mrk =read_mrk(mrk_file)
                mrk['Easting'],mrk['Northing'] =utmproj(mrk['Longitude'].values,mrk['Latitude'].values)
                mrk['EllipsoideHight'] = pd.to_numeric(mrk.EllipsoideHight.str.split(',',expand=True)[0])
                mrk[['errX','errY','errZ']]=mrk.Error.str.split(',',expand=True).astype(float)
                mrk =mrk.add_suffix('Mrk')

                drone =drone.join(mrk[['UTCTimeMrk','EllipsoideHightMrk','LatitudeMrk','LongitudeMrk','EastingMrk','NorthingMrk','errXMrk','errYMrk','errZMrk']],rsuffix='Mrk')
                drone['ImageEasting']= drone['EastingMrk']
                drone['ImageNorthing'] = drone['NorthingMrk']
            else:
                drone['ImageEasting'] = drone['EastingXif']
                drone['ImageNorthing'] = drone['NorthingXif']





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
            drone.to_csv(targets[0],index=True)
        from turtledrone.config import cfg as config
        for item in config.get_imagesource_dirs():
            exiffile = item / 'exif.csv'
            if exiffile.exists():
                file_dep = list(item.glob('*.MRK')) + list(item.glob('*Timestamp.CSV'))
                file_dep.append(exiffile)
                file_dep = list(filter(lambda x:x.stat().st_size > 0,file_dep))
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
            if 'K1' in item.keys():
                cam = ct.Camera(ct.RectilinearProjection(focallength_x_px=item.CalibratedFocalLengthX,
                                                                     focallength_y_px=item.CalibratedFocalLengthY,
                                                                        center_x_px=item.CalibratedOpticalCenterX,
                                                                        center_y_px=item.CalibratedOpticalCenterY),
                                                                        orientation= ct.SpatialOrientation(tilt_deg=item.GimbalPitchDegree+90,
                                                                                                        elevation_m=item.RelativeAltitude,
                                                                                                        roll_deg=item.GimbalRollDegree,
                                                                                                        heading_deg=item.GimbalYawDegree),
                                                                        lens=ct.BrownLensDistortion(item.K1,item.K2,item.K3))
            else:
                cam = ct.Camera(ct.RectilinearProjection(focallength_px=item.CalibratedFocalLength,
                                                                    center_x_px=item.CalibratedOpticalCenterX,
                                                                    center_y_px=item.CalibratedOpticalCenterY),
                                                                    orientation= ct.SpatialOrientation(tilt_deg=item.GimbalPitchDegree+90,
                                                                                                    elevation_m=item.RelativeAltitude,
                                                                                                    roll_deg=item.GimbalRollDegree,
                                                                                                   heading_deg=item.GimbalYawDegree))
            perside=20 
            x = np.linspace(0, item.ImageWidth, num=perside)  
            y = np.linspace(0, item.ImageHeight, num=perside)
            bottom = np.dstack((x,np.ones(perside)*item.ImageHeight))[0]
            right = np.dstack((np.ones(perside)*item.ImageWidth,y[::-1]))[0]
            top = np.dstack((x[::-1],np.zeros(perside)))[0]
            left = np.dstack((np.zeros(perside),y))[0]
            points = np.vstack((bottom,right,top,left))
            polydata =cam.spaceFromImage(points)
            polydata[:,0] =item.ImageEasting + polydata[:,0] 
            polydata[:,1] =item.ImageNorthing +polydata[:,1]
            return Polygon(polydata[:,0:2])
        data = pd.read_csv(dependencies[0],parse_dates=['TimeStamp'])
        crs = f'epsg:{int(data["UtmCode"][0])}'
        gdf = gp.GeoDataFrame(data, geometry=gp.points_from_xy(data.ImageEasting, data.ImageNorthing),crs=crs)
        gdf = gdf[~gdf.geometry.is_empty]
        gdf['ImagePolygon'] = gdf.apply(getpoly,axis=1)
        gdf.to_csv(targets[0])
        
        
        

    # dewarp = pd.to_numeric(config.cfg['dewarp'] )
    # cal = pd.to_numeric(config.cfg['calibration'] )
    from turtledrone.config import cfg as config

    for item in config.get_imagesource_dirs():
        positionfile = item / 'position.csv'
        if positionfile.exists():
            file_dep  =  positionfile
            target =   item /'polygons.csv'           
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
        from turtledrone.config import cfg as config    
        polygons = [item / 'polygons.csv'
                    for item in config.get_imagesource_dirs()
                    if (item / 'polygons.csv').exists()]

        if polygons:
            processpath = config.get_url('process')
            os.makedirs(processpath, exist_ok=True)
            target = processpath / 'imagedata.csv'
            return {
                'actions': [process_xif],
                'file_dep': polygons,       # list of polygons.csv
                'targets': [target],
                'clean': True,
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
