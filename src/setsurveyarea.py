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
import plotly.express as px
import geopandas as gp
import shutil
import shapely.wkt
from shapely.geometry import MultiPoint
from geopandas.tools import sjoin
import config

def task_set_up():
    config.read_config()
    
def task_detect_surveys():
        def process_survey(dependencies, targets,timedelta,maxpitch):
            drone =pd.read_csv(list(dependencies)[0],index_col='TimeStamp',parse_dates=['TimeStamp'])
            #drone = drone[drone.GimbalPitchDegree<maxpitch].copy()
            os.makedirs(os.path.dirname(targets[0]),exist_ok=True)
            drone.sort_index(inplace=True)
            #remove duplicated time stamps
            drone = drone[~drone.index.duplicated()]
            drone['Survey']=drone.index
            drone['Survey']=drone['Survey'].diff()>pd.Timedelta(timedelta)
            drone['Survey']=drone['Survey'].cumsum()+1
            drone.to_csv(targets[1],index=True)
            drone['StartTime'] =drone.index
            drone['EndTime'] =drone.index
            drone = drone[~drone.index.isna()]
            starttime = drone.groupby('Survey')['StartTime'].min()
            endtime = drone.groupby('Survey')['EndTime'].max()
            count =   drone.groupby('Survey').count()['SourceFile'].rename('ImageCount')
            position = drone.groupby('Survey')[['Latitude','Longitude']].mean()
            position =position.join([starttime,endtime,count])
            position.to_csv(targets[0],index=True)
            
        file_dep = os.path.join(config.geturl('process'),'imagedata.csv')
        targets = (os.path.join(config.geturl('process'),'surveysummary.csv'),os.path.join(config.geturl('process'),'surveys.csv'))
        return {
            'actions':[(process_survey, [],{'timedelta':config.cfg['survey']['timedelta'],'maxpitch':config.cfg['survey']['maxpitch']})],
            'file_dep':[file_dep],
            'targets':targets,
            'clean':True,
        }   

def task_assign_area():
        def load_shape(row):
            area = gp.read_file(row.File)
            area.id = row.SurveyCode
            return area
        
        def setarea(group):
            group['id'] = group['id'].value_counts().index[0]
            return group
        
        def process_assign_area(dependencies, targets, countrycode):
            surveyfile = list(filter(lambda x: 'surveys.csv' in x, dependencies))[0]
            areafile = list(filter(lambda x: 'surveyareas.csv' in x, dependencies))[0]
            drone =pd.read_csv(surveyfile,index_col='TimeStamp',parse_dates=['TimeStamp'])
            pnts = gp.GeoDataFrame(drone,geometry=gp.points_from_xy(drone.Longitude, drone.Latitude),crs='EPSG:4326')
            pnts.Survey = pnts.Survey.astype('int')
            try:
                areas =pd.read_csv(areafile)
                areas = areas[areas.Type=='SurveyArea']
                shapes =gp.GeoDataFrame(pd.concat([load_shape(row) for index,row in areas.iterrows()]))
                pnts = sjoin(pnts, shapes, how='left')
                pnts.loc[pnts.id.isna(),'id']=''
                ids = pnts.groupby('Survey')[['id']].apply(setarea)
                pnts.id = ids.id
                pnts.loc[pnts.id=='','id'] ='NOAREA'
            except:
                pnts = drone
                pnts['id'] ='NOAREA'
                          
            pnts['SurveyId']=countrycode+'_'+pnts['id']+'_'+pnts[['ImageHeight','Survey']].groupby('Survey').transform(lambda x: x.index.min().strftime("%Y%m%dT%H%M"))['ImageHeight']
            
            
                # data['Counter'] = 1
                # data['Counter'] = data['Counter'].cumsum()
                # data['SurveyId'] =f'{data.id.max()}_{data.index.min().strftime("%Y%m%dT%H%M")}'
                # data['Extension']=data['SourceFile'].apply(lambda x: os.path.splitext(x)[1]).str.upper()
                # data['NewName']=data.apply(lambda item: f"{cfg['survey']['dronetype']}_{cfg['survey']['cameratype']}_{cfg['survey']['country']}_{item.id}_{item.name.strftime('%Y%m%dT%H%M%S')}_{item.Counter:04}{item['Extension']}", axis=1)
                # filename = os.path.join(basepath,cfg['paths']['process'],f'{cfg["survey"]["country"]}_{data.id.max()}_{data.index.min().strftime("%Y%m%dT%H%M")}_survey.csv')                
                # data.to_csv(filename,index=True)
            
            
            pnts.to_csv(targets[0])
            

        file_dep = [os.path.join(config.geturl('process'),'surveys.csv'),
                    os.path.join(config.geturl('process'),'surveyareas.csv')]
        target = os.path.join(config.geturl('process'),'surveyswitharea.csv')
        return {
            'actions':[(process_assign_area,[],{'countrycode':config.cfg["survey"]["country"]})],
            'file_dep':file_dep,
            'targets':[target],
            'clean':True,
        }       

if __name__ == '__main__':
    import doit
    DOIT_CONFIG = {'check_file_uptodate': 'timestamp'}
    #print(globals())
    doit.run(globals())   