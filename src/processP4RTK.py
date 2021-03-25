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
from read_rtk import read_mrk
from pyproj import Proj 
from doit.tools import check_timestamp_unchanged
import shutil





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
        def convert_wgs_to_utm(lon, lat):
            utm_band = str((np.floor((lon + 180) / 6 ) % 60) + 1)
            if len(utm_band) == 1:
                utm_band = '0'+utm_band
            if lat >= 0:
                epsg_code = '326' + utm_band
            else:
                epsg_code = '327' + utm_band
            return epsg_code
        def process_transect(leg):
            leg = leg.copy()
            leg['GPSTime'] = leg.TimeStamp
            leg.loc[leg['GPSTime'].duplicated(),'GPSTime']=leg.loc[leg['GPSTime'].duplicated(),'GPSTime']+pd.to_timedelta('500L')
            leg['GPSDistance'] =((leg['Northing'].diff()**2 + leg['Easting'].diff()**2)**0.5)
            leg['GpsSpeed']=(((leg['Northing'].diff()**2 + leg['Easting'].diff()**2)**0.5)/leg.GPSTime.diff().dt.total_seconds())
            leg['timedelta'] = leg.GPSTime.diff().dt.total_seconds()
            leg['GPSExcessTime'] = ((leg['GpsSpeed']/leg['DroneSpeed']) * leg['timedelta'])-leg['timedelta']
            leg['ImageTime'] = leg.TimeStamp
            leg['ImageInterval']=leg['ImageTime'].diff().dt.total_seconds()
            leg.loc[leg['ImageInterval']==2,'ImageTime'] =leg.loc[leg['ImageInterval']==2,'ImageTime'] + pd.to_timedelta('500L')
            leg['ImageInterval']=leg['ImageTime'].diff().dt.total_seconds()
            for i in range(1,len(leg)):
                leg.iloc[i,leg.columns.get_loc('GPSTime')] = leg.iloc[i,leg.columns.get_loc('GPSTime')] + pd.to_timedelta(leg.iloc[i,leg.columns.get_loc('GPSExcessTime')],unit='s')
                leg['timedelta'] = leg.GPSTime.diff().dt.total_seconds()
                leg['GpsSpeed']=(((leg['Northing'].diff()**2 + leg['Easting'].diff()**2)**0.5)/leg.GPSTime.diff().dt.total_seconds())
                leg['GPSExcessTime'] = ((leg['GpsSpeed']/leg['DroneSpeed']) * leg['timedelta'])-leg['timedelta']
                leg.replace([np.inf, -np.inf],0, inplace=True)
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
                rtk['Easting'],rtk['Northing'] =utmproj(rtk['Longitude'].values,rtk['Latitude'].values)
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
        searchpath = os.path.join(basepath,os.path.dirname(cfg['paths']['imagesource']),'merge.csv')
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
            'actions':[(process_survey, [],{'timedelta':cfg['survey']['timedelta']})],
            'file_dep':[file_dep],
            'targets':targets,
            'clean':True,
        }  
        
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
        file_dep = os.path.join(basepath,cfg['paths']['process'],'surveys.csv')
        targets = os.path.join(basepath,cfg['paths']['process'],'surveys.html')
        return {

            'actions':[(process_survey, [],{'apikey':cfg['mapboxkey']})],
            'file_dep':[file_dep],
            'targets':[targets],
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
        target = os.path.join(basepath,cfg['paths']['process'],'surveyareas.csv')
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
        
        def setarea(group):
            group['id'] = group['id'].value_counts().index[0]
            return group
        
        def process_assign_area(dependencies, targets):
            surveyfile = list(filter(lambda x: 'surveys.csv' in x, dependencies))[0]
            areafile = list(filter(lambda x: 'surveyareas.csv' in x, dependencies))[0]
            drone =pd.read_csv(surveyfile,index_col='TimeStamp',parse_dates=['TimeStamp'])
            pnts = gp.GeoDataFrame(drone,geometry=gp.points_from_xy(drone.Longitude, drone.Latitude),crs='EPSG:4326')
            pnts.Survey = pnts.Survey.astype('int')
            areas =pd.read_csv(areafile)
            shapes =gp.GeoDataFrame(pd.concat([load_shape(row) for index,row in areas.iterrows()]))
            pnts = sjoin(pnts, shapes, how='left')
            pnts.loc[pnts.id.isna(),'id']=''
            pnts =pnts.groupby('Survey').apply(setarea)
            pnts.loc[pnts.id=='','id'] ='NOAREA'
            pnts.to_csv(targets[0])
            
        config = {"config": get_var('config', 'NO')}
        with open(config['config'], 'r') as ymlfile:
            cfg = yaml.load(ymlfile, yaml.SafeLoader)
        basepath = os.path.dirname(config['config'])
        file_dep = [os.path.join(basepath,cfg['paths']['process'],'surveys.csv'),
                    os.path.join(basepath,cfg['paths']['process'],'surveyareas.csv')]
        target = os.path.join(basepath,cfg['paths']['process'],'surveyswitharea.csv')
        return {
            'actions':[process_assign_area],
            'file_dep':file_dep,
            'targets':[target],
            'clean':True,
        }       
@create_after(executed='assign_area', target_regex='.*\exif.csv')         
def task_make_surveys():
        def process_surveys(dependencies, targets,cfg):
            drone =pd.read_csv(dependencies[0],index_col='TimeStamp',parse_dates=['TimeStamp'])
            for name,data in drone.groupby('Survey'):
                data['Counter'] = 1
                data['Counter'] = data['Counter'].cumsum()
                data['SurveyId'] =f'{data.id.max()}_{data.index.min().strftime("%Y%m%dT%H%M")}'
                data['NewName']=data.apply(lambda item: f"{cfg['survey']['dronetype']}_{cfg['survey']['cameratype']}_{cfg['survey']['country']}_{item.id}_{item.name.strftime('%Y%m%dT%H%M%S')}_{item.Counter:04}.JPG", axis=1)
                filename = os.path.join(basepath,cfg['paths']['process'],f'{cfg["survey"]["country"]}_{data.id.max()}_{data.index.min().strftime("%Y%m%dT%H%M")}_survey.csv')                
                data.to_csv(filename,index=True)
            
        config = {"config": get_var('config', 'NO')}
        with open(config['config'], 'r') as ymlfile:
            cfg = yaml.load(ymlfile, yaml.SafeLoader)
        basepath = os.path.dirname(config['config'])
        file_dep = os.path.join(basepath,cfg['paths']['process'],'surveyswitharea.csv')
        if os.path.exists(file_dep):
            surveys =pd.read_csv(file_dep,index_col='TimeStamp',parse_dates=['TimeStamp']).groupby('Survey')
            targets = [os.path.join(basepath,cfg['paths']['process'],
                                    f'{cfg["survey"]["country"]}_{data.id.max()}_{data.index.min().strftime("%Y%m%dT%H%M")}_survey.csv') for name,data in surveys]
            return {
                'actions':[(process_surveys,[],{'cfg':cfg})],
                'file_dep':[file_dep],
                'targets':targets,
                'clean':True,
            }   
            
def task_file_images():
        def process_images(dependencies, targets):
            survey = pd.read_csv(dependencies[0])
            destpath = os.path.dirname(targets[0])
            os.makedirs(destpath,exist_ok=True)

            for index,row in survey.iterrows():
                dest =os.path.join(destpath,row.NewName)
                if not os.path.exists(dest):
                    shutil.copyfile(row.SourceFile,dest)
            shutil.copyfile(dependencies[0],targets[0])
            
        config = {"config": get_var('config', 'NO')}
        with open(config['config'], 'r') as ymlfile:
            cfg = yaml.load(ymlfile, yaml.SafeLoader)
        basepath = os.path.dirname(config['config'])
        file_dep = glob.glob(os.path.join(basepath,cfg['paths']['process'],'*_survey.csv'))
        for file in file_dep:
            country = os.path.basename(file).split('_')[0]
            sitecode = '_'.join(os.path.basename(file).split('_')[1:3])
            target = os.path.join(cfg['paths']['output'],country,sitecode,os.path.basename(file))
            yield {
                'name':file,
                'actions':[process_images],
                'file_dep':[file],
                'targets':[target],
                'uptodate': [True],
                'clean':True,
            }      
        
def task_check_survey():
        def process_check_survey(dependencies, targets):
            drone =pd.read_csv(dependencies[0],index_col='TimeStamp',parse_dates=['TimeStamp'])
            images = pd.DataFrame(glob.glob(os.path.join(os.path.dirname(dependencies[0]),'*.JPG')),
                                  columns=['Path'])
            images['NewName'] = images['Path'].apply(os.path.basename)
            d =drone.join(images.set_index('NewName'),how='left',on=['NewName'],rsuffix="f")
            missing =d.Path.isna().sum()
            expected = d.NewName.count()
            coverage = 100*(expected-missing)/expected
            pd.DataFrame([{'SurveyId':d.SurveyId.max(),
                           'StartTime':d.index.min(),'EndTime':d.index.max(),
                           'Latitude':d.Latitude.mean(),'Longitude':d.Longitude.mean(),
                           'Coverage':coverage,'Expected':expected,
                           'Missing':missing}]).to_csv(targets[0],index=False)
            
        config = {"config": get_var('config', 'NO')}
        with open(config['config'], 'r') as ymlfile:
            cfg = yaml.load(ymlfile, yaml.SafeLoader)
        basepath = os.path.dirname(config['config'])
        file_dep = glob.glob(os.path.join(cfg['paths']['output'],'**','*_survey.csv'),recursive=True)
        for file in file_dep:
            target = file.replace('_survey','_survey_summary')
            yield {
                'name':file,
                'actions':[process_check_survey],
                'file_dep':[file],
                'targets':[target],
                'uptodate': [True],
                'clean':True,
            }         

def task_concat_check_survey():
        def process_concat_check_survey(dependencies, targets):
            os.makedirs(os.path.dirname(targets[0]),exist_ok=True)
            surveys =pd.concat([pd.read_csv(file) for file in dependencies])
            surveys =surveys.set_index('SurveyId').sort_index()
            surveys.to_csv(targets[0])
            
        config = {"config": get_var('config', 'NO')}
        with open(config['config'], 'r') as ymlfile:
            cfg = yaml.load(ymlfile, yaml.SafeLoader)
        basepath = os.path.dirname(config['config'])
        file_dep = glob.glob(os.path.join(cfg['paths']['output'],'**','*_survey_summary.csv'),recursive=True)
        target = os.path.join(cfg['paths']['reports'],'image_coverage.csv')
        return {
            'actions':[process_concat_check_survey],
            'file_dep':file_dep,
            'targets':[target],
            'clean':True,
            }         
# def task_calculate_newname():
#     pass
#xifdata.apply(lambda item: f"{survey['dronetype']}_{survey['camera']}_{survey['country']}_{survey['surveycode']}_{survey['surveynumber']:03}_{item.LocalTime}_{item.Counter:04}.JPG", axis=1)       
 
         
if __name__ == '__main__':
    import doit

    #print(globals())
    doit.run(globals())