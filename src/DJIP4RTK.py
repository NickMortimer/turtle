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
import utils





wanted ={"SourceFile","FileModifyDate","ImageDescription",
         "ExposureTime","FNumber","ExposureProgram","ISO",
         "DateTimeOriginal","Make","SpeedX","SpeedY","SpeedZ",
         "Pitch","Yaw","Roll","CameraPitch","CameraYaw","CameraRoll",
         "ExifImageWidth","ExifImageHeight","SerialNumber",
         "GPSLatitudeRef","GPSLongitudeRef","GPSAltitudeRef","AbsoluteAltitude",
         "RelativeAltitude","GimbalRollDegree","GimbalYawDegree",
         "GimbalPitchDegree","FlightRollDegree","FlightYawDegree",
         "FlightPitchDegree","CamReverse","GimbalReverse","CalibratedFocalLength",
         "CalibratedOpticalCenterX","CalibratedOpticalCenterY","ImageWidth",
         "ImageHeight","GPSAltitude","GPSLatitude","GPSLongitude","CircleOfConfusion",
         "FOV","Latitude",'Longitude'}



def task_create_json():
        config = {"config": get_var('config', 'NO')}
        with open(config['config'], 'r') as ymlfile:
            cfg = yaml.load(ymlfile, yaml.SafeLoader)
        basepath = os.path.dirname(config['config'])
        exifpath = os.path.join(basepath,cfg['paths']['exiftool'])
        for item in glob.glob(os.path.join(basepath,cfg['paths']['imagesource']),recursive=True):
            source = os.path.join(basepath,os.path.dirname(item))
            if glob.glob(os.path.join(source,'*.JPG')) or glob.glob(os.path.join(source,'*.jpg')):
                target  = os.path.join(source,'exif.json')
                filter = os.path.join(source,cfg['paths']['imagewild'])
                file_dep = glob.glob(filter)
                if file_dep:
                    yield {
                        'name':item,
                        'actions':[f'"{exifpath}" -json "{filter}" > "{target}"'],
                        'targets':[target],
                        'uptodate':[True],
#                        'uptodate': [check_timestamp_unchanged(file_dep, 'ctime')],
                        'clean':True,
                    }
    
@create_after(executed='create_json', target_regex='.*\exif.json')    
def task_process_json():
        def process_json(dependencies, targets):
            source_file =dependencies[0]
            print('source file is: {0}'.format(source_file))
            print('output dir is: {0}'.format(list(targets)[0]))
            drone = pd.read_json(source_file)
            def get_longitude(item):
                longitude =float(item[0]) + float(item[2][0:-1])/60 + float(item[3][0:-1])/3600
                return (longitude)
            drone['Longitude'] = np.nan
            drone['Latitude'] = np.nan
            drone.loc[ ~drone['GPSLongitude'].isna(),'Longitude']=drone.loc[ ~drone['GPSLongitude'].isna(),'GPSLongitude'].str.split(' ',expand=True).apply(get_longitude,axis=1)
            drone.loc[ ~drone['GPSLatitude'].isna(),'Latitude']=drone.loc[ ~drone['GPSLatitude'].isna(),'GPSLatitude'].str.split(' ',expand=True).apply(get_longitude,axis=1)
            drone.loc[drone['GPSLatitudeRef']=='South','Latitude'] =drone.loc[drone['GPSLatitudeRef']=='South','Latitude']*-1
            drone = drone[drone.columns[drone.columns.isin(wanted)]]
            drone['TimeStamp'] = pd.to_datetime(drone.DateTimeOriginal,format='%Y:%m:%d %H:%M:%S')
            drone['Sequence'] =drone.SourceFile.str.extract('(?P<Sequence>\d+)\.(jpg|JPG)')['Sequence']
            drone.set_index('Sequence',inplace=True)
            drone.to_csv(list(targets)[0],index=True)
            
        config = {"config": get_var('config', 'NO')}
        with open(config['config'], 'r') as ymlfile:
            cfg = yaml.load(ymlfile, yaml.SafeLoader)
        basepath = os.path.dirname(config['config'])
        for item in glob.glob(os.path.join(basepath,os.path.dirname(cfg['paths']['imagesource']),'exif.json'),recursive=True):
            source = os.path.join(basepath,os.path.dirname(item))
            file_dep  =  item
            target =   os.path.join(source,'exif.csv')           
            yield {
                'name':source,
                'actions':[process_json],
                'file_dep':[file_dep],
                'targets':[target],

                'clean':True,
            }

    

@create_after(executed='process_json', target_regex='.*\exif.json')    
def task_process_mergpos():
        def process_transect(leg):
            leg = leg.copy()
            leg['GPSTime'] = leg.TimeStamp
            leg = leg[~ leg.TimeStamp.isna()]
            leg.loc[leg['GPSTime'].duplicated(),'GPSTime']=leg.loc[leg['GPSTime'].duplicated(),'GPSTime']+pd.to_timedelta('500L')
            leg['GPSDistance'] =((leg['Northing'].diff()**2 + leg['Easting'].diff()**2)**0.5)
            leg['GpsSpeed']=(((leg['Northing'].diff()**2 + leg['Easting'].diff()**2)**0.5)/leg.GPSTime.diff().dt.total_seconds())
            leg['timedelta'] = leg.GPSTime.diff().dt.total_seconds()
            leg['GPSExcessTime'] = ((leg['GpsSpeed']/leg['DroneSpeed']) * leg['timedelta'])-leg['timedelta']
            leg['GPSExcessTime']=leg['GPSExcessTime'].replace([np.nan,np.inf],0)
            leg['ImageTime'] = leg.TimeStamp
            leg['ImageInterval']=leg['ImageTime'].diff().dt.total_seconds()
            leg.loc[leg['ImageInterval']==2,'ImageTime'] =leg.loc[leg['ImageInterval']==2,'ImageTime'] + pd.to_timedelta('500L')
            leg['ImageInterval']=leg['ImageTime'].diff().dt.total_seconds()
            for i in range(1,len(leg)):
                leg.iloc[i,leg.columns.get_loc('GPSTime')] = leg.iloc[i,leg.columns.get_loc('GPSTime')] + pd.to_timedelta(leg.iloc[i,leg.columns.get_loc('GPSExcessTime')],unit='s')
                leg['timedelta'] = leg.GPSTime.diff().dt.total_seconds()
                leg['GpsSpeed']=(((leg['Northing'].diff()**2 + leg['Easting'].diff()**2)**0.5)/leg.GPSTime.diff().dt.total_seconds())
                leg['GPSExcessTime'] = ((leg['GpsSpeed']/leg['DroneSpeed']) * leg['timedelta'])-leg['timedelta']
                leg['GPSExcessTime']=leg['GPSExcessTime'].replace([np.nan,np.inf],0)
            gpstime=leg.GPSTime.astype('int64').astype("float")
            imagetime= leg.ImageTime.astype('int64').astype('float')
            leg['ImageNorthing'] =np.interp(imagetime,gpstime,leg.Northing)
            leg['ImageEasting'] = np.interp(imagetime,gpstime,leg.Easting)
            leg['ImageDistance'] =((leg['ImageNorthing'].diff()**2 + leg['ImageEasting'].diff()**2)**0.5)
            leg['ImageSpeed'] =leg['ImageDistance']/leg['ImageInterval']
            return leg
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
            drone['DroneSpeed'] = (drone['SpeedX']**2+drone['SpeedY']**2)**0.5
            drone['Leg'] =0
            drone.loc[drone['Interval']>8,'Leg'] =1
            drone['Leg'] = drone['Leg'].cumsum()
            drone['UtmCode'] =utmcode
            g = drone.groupby('Leg')
            drone =pd.concat([process_transect(leg) for name,leg in g])
            mrk_file =list(filter(lambda x: '_Timestamp.MRK' in x, dependencies))
            if mrk_file:
                mrk =read_mrk(mrk_file[0])
                mrk['Easting'],mrk['Northing'] =utmproj(mrk['Longitude'].values,mrk['Latitude'].values)
                drone =drone.join(mrk,rsuffix='Mrk')
            rtk_file=list(filter(lambda x: '_Timestamp.CSV' in x, dependencies))
            if rtk_file:
                rtk =pd.read_csv(rtk_file[0],parse_dates=['GPST'],index_col=['Sequence'])
                rtk['Easting'],rtk['Northing'] =utmproj(rtk['longitude(deg)'].values,rtk['latitude(deg)'].values)
                drone =drone.join(rtk,rsuffix='rtk')
            drone.set_index('TimeStamp',inplace=True)
            drone.sort_index(inplace=True)
            drone = drone[pd.notna(drone.index)]
            drone.to_csv(list(targets)[0],index=True)
            
        config = {"config": get_var('config', 'NO')}
        with open(config['config'], 'r') as ymlfile:
            cfg = yaml.load(ymlfile, yaml.SafeLoader)
        basepath = os.path.dirname(config['config'])
        for item in glob.glob(os.path.join(basepath,cfg['paths']['imagesource']),recursive=True):
            source = os.path.join(basepath,os.path.dirname(item))
            file_dep  =  list(filter(lambda x:  any(f in x for f in ['exif.csv','Timestamp']), glob.glob(os.path.join(source,'*.*'))))
            if file_dep:
                target =   os.path.join(source,'merge.csv')           
                yield {
                    'name':source,
                    'actions':[process_json],
                    'file_dep':file_dep,
                    'targets':[target],
                    'clean':True,
                }    
    
def task_addpolygons():
    def process_polygons(dependencies, targets,dewarp):
        def getpoly(item):
            drone.setdronepos(item.Easting,item.Northing,item.RelativeAltitude,
                                  (90+item.GimbalPitchDegree)*-1,item.GimbalRollDegree,item.GimbalYawDegree)
            return drone.getimagepolygon()
        data = pd.read_csv(dependencies[0],parse_dates=['TimeStamp'])
        crs = f'epsg:{int(data["UtmCode"][0])}'
        gdf = gp.GeoDataFrame(data, geometry=gp.points_from_xy(data.Easting, data.Northing),crs=crs)
        drone =P4rtk(dewarp,crs)
        gdf['ImagePolygon'] = gdf.apply(getpoly,axis=1)
        gdf.to_csv(targets[0])
        
        
        
    config = {"config": get_var('config', 'NO')}
    with open(config['config'], 'r') as ymlfile:
        cfg = yaml.load(ymlfile, yaml.SafeLoader)
    basepath = os.path.dirname(config['config'])
    dewarp = pd.to_numeric(cfg['survey']['dewarp'] )
    for file_dep in glob.glob(os.path.join(basepath,cfg['paths']['imagesource'],'merge.csv'),recursive=True):
        target = os.path.join(basepath,os.path.dirname(file_dep),'merge_polygons.csv')   
        yield {
            'name':file_dep,
            'actions':[(process_polygons, [],{'dewarp':dewarp})],
            'file_dep':[file_dep],
            'targets':[target],
            'clean':True,
        }       
    
@create_after(executed='process_json', target_regex='.*\exif.csv') 
def task_merge_xif():
        def process_xif(dependencies, targets):
            target = list(targets)[0]
            os.makedirs(os.path.dirname(target),exist_ok=True)
            drone = pd.concat([pd.read_csv(file,index_col='TimeStamp',parse_dates=['TimeStamp']) 
                            for file in list(dependencies)]) 
            drone.sort_index(inplace=True)
            drone.to_csv(list(targets)[0],index=True)
            
        config = {"config": get_var('config', 'NO')}
        with open(config['config'], 'r') as ymlfile:
            cfg = yaml.load(ymlfile, yaml.SafeLoader)
        basepath = os.path.dirname(config['config'])
        searchpath = os.path.join(basepath,os.path.dirname(cfg['paths']['imagesource']),'merge_polygons.csv')
        file_dep = glob.glob(searchpath,recursive=True)
        processpath =os.path.join(basepath,cfg['paths']['process'])
        os.makedirs(processpath,exist_ok=True)
        target = os.path.join(processpath,'mergeall.csv')
        return {
            'actions':[process_xif],
            'file_dep':file_dep,
            'targets':[target],
            'clean':True,
        }

       

       

        

        
            
  
    
    
# def task_calculate_newname():
#     pass
#xifdata.apply(lambda item: f"{survey['dronetype']}_{survey['camera']}_{survey['country']}_{survey['surveycode']}_{survey['surveynumber']:03}_{item.LocalTime}_{item.Counter:04}.JPG", axis=1)       
 

if __name__ == '__main__':
    import doit
    DOIT_CONFIG = {'check_file_uptodate': 'timestamp'}
    #print(globals())
    doit.run(globals())
