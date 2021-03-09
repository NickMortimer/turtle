import os
import glob
import doit
import glob
import os
import numpy as np
import yaml
import pandas as pd
from doit import get_var
from doit.tools import run_once
from doit import create_after
import numpy as np
import plotly
import plotly.express as px
from getbase import getbasenames
from getbase import getbase
import geopandas as gp
from geopandas.tools import sjoin
from read_rtk import read_mrk





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
            target  = os.path.join(source,'exif.json')
            filter = os.path.join(source,cfg['paths']['imagewild'])
            yield {
                'name':item,
                'actions':[f'"{exifpath}" -json "{filter}" > "{target}"'],
                'targets':[target],
                'uptodate': [True],
                'clean':True,
            }
    
@create_after(executed='create_json', target_regex='.*\exif.json')    
def task_process_json():
        def process_json(dependencies, targets,rtk=False):
            # dependencies.sort()
            source_file = list(dependencies)[0]
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
            # if rtk:
            #     mrk =read_mrk(dependencies[0])
            #     drone['Sequence'] =drone.SourceFile.str.extract('(?P<Sequence>\d\d\d\d)\.JPG').astype(int)
            #     drone.set_index('Sequence',inplace=True)
            #     drone =drone.join(mrk,rsuffix='Mrk')
            # drone.set_index('TimeStamp',inplace=True)
            drone.sort_index(inplace=True)
            drone = drone[pd.notna(drone.index)]
            drone.to_csv(list(targets)[0],index=True)
            
        config = {"config": get_var('config', 'NO')}
        with open(config['config'], 'r') as ymlfile:
            cfg = yaml.load(ymlfile, yaml.SafeLoader)
        basepath = os.path.dirname(config['config'])
        for item in glob.glob(os.path.join(basepath,cfg['paths']['imagesource']),recursive=True):
            source = os.path.join(basepath,os.path.dirname(item))
            file_dep  = [os.path.join(source,'exif.json')]
            mark = glob.glob(os.path.join(source,'*Timestamp.MRK'))
            if mark:
                file_dep.append(mark[0])
                #file_dep = tuple(file_dep)
            target =   os.path.join(source,'exif.csv')           
            yield {
                'name':source,
                'actions':[(process_json,[],{'rtk':bool(mark)})],
                'file_dep':file_dep,
                'targets':[target],
                'clean':True,
            }

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
        target = os.path.join(basepath,os.path.dirname(cfg['paths']['output']),'merge/exif.csv')
        return {
            'actions':[process_xif],
            'file_dep':file_dep,
            'targets':[target],
            'clean':True,
        }

def task_split_surveys():
        def process_survey(dependencies, targets,timedelta):
            drone =pd.read_csv(list(dependencies)[0],index_col='TimeStamp',parse_dates=['TimeStamp'])
            os.makedirs(os.path.dirname(targets[0]),exist_ok=True)
            drone.sort_index(inplace=True)
            drone['Survey']=drone.index
            drone['Survey']=drone['Survey'].diff()>pd.Timedelta(timedelta)
            drone['Survey']=drone['Survey'].cumsum()+1
            drone.to_csv(targets[1],index=True)
            drone['StartTime'] =drone.index
            drone['EndTime'] =drone.index
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
        file_dep = os.path.join(basepath,os.path.dirname(cfg['paths']['output']),'merge/exif.csv')
        targets = (os.path.join(basepath,os.path.dirname(cfg['paths']['output']),'merge/surveysummary.csv'),os.path.join(basepath,os.path.dirname(cfg['paths']['output']),'merge/surveys.csv'))
        return {
            'actions':[(process_survey, [],{'timedelta':cfg['survey']['timedelta']})],
            'file_dep':[file_dep],
            'targets':targets,
            'clean':True,
        }  
        
          

def task_make_area_list():
        def split(path):
            keys =os.path.split(os.path.dirname(path))[-1].split('_')
            return {'SurveyCode':keys[0],'SurveyLongName':keys[1],'file':path}
        
        def process_area_list(dependencies, targets):
            areas = pd.DataFrame.from_records([split(shape) for shape in dependencies])
            areas.to_csv(targets[0],index=False)                
            
        config = {"config": get_var('config', 'NO')}
        with open(config['config'], 'r') as ymlfile:
            cfg = yaml.load(ymlfile, yaml.SafeLoader)
        basepath = os.path.dirname(config['config'])
        file_dep = glob.glob(os.path.join(basepath,os.path.dirname(cfg['paths']['surveyarea']),'**/*.shp'),recursive=True)
        target = os.path.join(basepath,os.path.dirname(cfg['paths']['output']),'merge/surveyareas.csv')
        return {
            'actions':[process_area_list],
            'file_dep':file_dep,
            'targets':[target],
            'clean':True,
        } 
           
def task_assign_area():
        def load_shape(row):
            area = gp.read_file(row.file)
            area.id = row.SurveyCode
            return area
        
        def process_assign_area(dependencies, targets):
            dependencies.sort()
            drone =pd.read_csv(dependencies[1],index_col='TimeStamp',parse_dates=['TimeStamp'])
            pnts = gp.GeoDataFrame(drone,geometry=gp.points_from_xy(drone.Longitude, drone.Latitude),crs='EPSG:4326')
            areas =pd.read_csv(dependencies[0])
            shapes =gp.GeoDataFrame(pd.concat([load_shape(row) for index,row in areas.iterrows()]))
            pnts = sjoin(pnts, shapes, how='left')
            pnts.to_csv(targets[0])
            
        config = {"config": get_var('config', 'NO')}
        with open(config['config'], 'r') as ymlfile:
            cfg = yaml.load(ymlfile, yaml.SafeLoader)
        basepath = os.path.dirname(config['config'])
        file_dep = [os.path.join(basepath,os.path.dirname(cfg['paths']['output']),'merge/surveys.csv'),
                    os.path.join(basepath,os.path.dirname(cfg['paths']['output']),'merge/surveyareas.csv')]
        target = os.path.join(basepath,os.path.dirname(cfg['paths']['output']),'merge/surveyswitharea.csv')
        return {
            'actions':[process_assign_area],
            'file_dep':file_dep,
            'targets':[target],
            'clean':True,
        }       
        
def task_make_surveys():
        def process_surveys(dependencies, targets,cfg):
            drone =pd.read_csv(dependencies[0],index_col='TimeStamp',parse_dates=['TimeStamp'])
            for name,data in drone.groupby('Survey'):
                data['Counter'] = 1
                data['Counter'] = data['Counter'].cumsum()
                data['NewName']=data.apply(lambda item: f"{cfg['survey']['dronetype']}_{cfg['survey']['cameratype']}_{cfg['survey']['country']}_{item.id}_{item.name.strftime('%Y%m%dT%H%M%S')}_{item.Counter:04}.JPG", axis=1)
                filename = os.path.join(basepath,os.path.dirname(cfg['paths']['output']),f'merge/Survey_{data.id.min()}_{data.index.min().strftime("%Y%m%dT%H%M%S")}.csv')                
                data.to_csv(filename,index=True)
            
        config = {"config": get_var('config', 'NO')}
        with open(config['config'], 'r') as ymlfile:
            cfg = yaml.load(ymlfile, yaml.SafeLoader)
        basepath = os.path.dirname(config['config'])
        file_dep = os.path.join(basepath,os.path.dirname(cfg['paths']['output']),'merge/surveyswitharea.csv')
        if os.path.exists(file_dep):
            surveys =pd.read_csv(file_dep,index_col='TimeStamp',parse_dates=['TimeStamp']).groupby('Survey')
            targets = [os.path.join(basepath,os.path.dirname(cfg['paths']['output']),f'merge/Survey_{data.id.min()}_{data.index.min().strftime("%Y%m%dT%H%M%S")}.csv') for name,data in surveys]
            return {
                'actions':[(process_surveys,[],{'cfg':cfg})],
                'file_dep':[file_dep],
                'targets':targets,
                'clean':True,
            }         
        
        

# def task_calculate_newname():
#     pass
#xifdata.apply(lambda item: f"{survey['dronetype']}_{survey['camera']}_{survey['country']}_{survey['surveycode']}_{survey['surveynumber']:03}_{item.LocalTime}_{item.Counter:04}.JPG", axis=1)       
 
def task_plot_surveys():
        def process_survey(dependencies, targets,apikey):
            drone =pd.read_csv(list(dependencies)[0],index_col='TimeStamp',parse_dates=['TimeStamp'])
            px.set_mapbox_access_token(apikey)
            fig = px.scatter_mapbox(drone, hover_name='Survey', lat="Latitude", lon="Longitude",  
                                    mapbox_style="satellite-streets",color="Survey", size_max=30, zoom=10)
            fig.update_layout(mapbox_style="satellite-streets")
            plotly.offline.plot(fig, filename=list(targets)[0],auto_open = False)
            
        config = {"config": get_var('config', 'NO')}
        with open(config['config'], 'r') as ymlfile:
            cfg = yaml.load(ymlfile, yaml.SafeLoader)
        basepath = os.path.dirname(config['config'])
        file_dep = os.path.join(basepath,os.path.dirname(cfg['paths']['output']),'merge/surveys.csv')
        targets = os.path.join(basepath,os.path.dirname(cfg['paths']['output']),'merge/surveys.html')
        return {
            'actions':[(process_survey, [],{'apikey':cfg['mapboxkey']})],
            'file_dep':[file_dep],
            'targets':[targets],
            'clean':True,
        }           
if __name__ == '__main__':
    import doit

    #print(globals())
    doit.run(globals())