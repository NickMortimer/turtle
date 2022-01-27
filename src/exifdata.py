import os
import glob
import doit
import numpy as np
import yaml
import pandas as pd
from doit import get_var
from doit.tools import run_once
from doit import create_after
import numpy as np


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
         "FOV","Latitude",'Longitude','SubSecDateTimeOriginal'}

def convert_wgs_to_utm(lon, lat):
    utm_band = str(int((np.floor((lon + 180) / 6 ) % 60) + 1))
    if len(utm_band) == 1:
        utm_band = '0'+utm_band
    if lat >= 0:
        epsg_code = '326' + utm_band
    else:
        epsg_code = '327' + utm_band
    return epsg_code

def task_create_json():
        config = {"config": get_var('config', 'NO')}
        with open(config['config'], 'r') as ymlfile:
            cfg = yaml.load(ymlfile, yaml.SafeLoader)
        basepath = os.path.dirname(config['config'])
        exifpath = os.path.join(basepath,cfg['paths']['exiftool'])
        for item in glob.glob(os.path.join(basepath,cfg['paths']['imagesource']),recursive=True):
            source = os.path.join(basepath,os.path.dirname(item))
            if glob.glob(os.path.join(source,cfg['paths']['imagewild'].upper())) or glob.glob(cfg['paths']['imagewild'].lower()):
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
            #drone = drone[drone.columns[drone.columns.isin(wanted)]]
            if 'SubSecDateTimeOriginal' in drone.columns:
                drone['TimeStamp'] = pd.to_datetime(drone.SubSecDateTimeOriginal,format='%Y:%m:%d %H:%M:%S.%f')
            else:
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
        searchpath = os.path.join(basepath,os.path.dirname(cfg['paths']['imagesource']),'exif.csv')
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
            
            
def task_split_surveys():
        def process_survey(dependencies, targets,timedelta,maxpitch):
            drone =pd.read_csv(list(dependencies)[0],index_col='TimeStamp',parse_dates=['TimeStamp'])
            #drone = drone[drone.GimbalPitchDegree<maxpitch].copy()
            os.makedirs(os.path.dirname(targets[0]),exist_ok=True)
            drone.sort_index(inplace=True)
            drone['Survey']=drone.index
            drone['Survey']=drone['Survey'].diff()>pd.Timedelta(timedelta)
            drone['Survey']=drone['Survey'].cumsum()+1
            drone.to_csv(targets[1],index=True)
            drone['StartTime'] =drone.index
            drone['EndTime'] =drone.index
            drone = drone[~drone.index.isna()]
            starttime = drone.groupby('Survey').min()['StartTime']
            endtime = drone.groupby('Survey').max()['EndTime']
            count =   drone.groupby('Survey').count()['SourceFile'].rename('ImageCount')
            position = drone.groupby('Survey').mean()[['Latitude','Longitude']]
            position =position.join([starttime,endtime,count])
            position.to_csv(targets[0],index=True)
            
            
        config = {"config": get_var('config', 'NO')}
        with open(config['config'], 'r') as ymlfile:
            cfg = yaml.load(ymlfile, yaml.SafeLoader)
        basepath = os.path.dirname(config['config'])
        file_dep = os.path.join(basepath,cfg['paths']['process'],'mergeall.csv')
        targets = (os.path.join(basepath,cfg['paths']['process'],'surveysummary.csv'),os.path.join(basepath,cfg['paths']['process'],'surveys.csv'))
        return {
            'actions':[(process_survey, [],{'timedelta':cfg['survey']['timedelta'],'maxpitch':cfg['survey']['maxpitch']})],
            'file_dep':[file_dep],
            'targets':targets,
            'clean':True,
        }              
        
if __name__ == '__main__':
    import doit
    DOIT_CONFIG = {'check_file_uptodate': 'timestamp'}
    #print(globals())
    doit.run(globals())        