from genericpath import exists
import os
import glob
import doit
import glob
import os
import pandas as pd
import numpy as np
import geopandas as gp
from geopandas.tools import sjoin
import config
from utils import convert_wgs_to_utm
from doit import create_after

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
            drone['SourceDrive'] =drone.SourceFile.str.extract(r'^(.*?DCIM)')
            if timedelta == '0MIN':
                drone['Survey']=drone.groupby('SourceDrive').ngroup()
            else:
                drone['Survey']=drone.index
                drone['Survey']=drone['Survey'].diff()>pd.Timedelta(timedelta)
                drone['Survey']=drone['Survey'].cumsum()+1
                #all survey on one card!

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
         #   try:
            areas =pd.read_csv(areafile)
            areas = areas[areas.Type=='SurveyArea']
            shapes =gp.GeoDataFrame(pd.concat([load_shape(row) for index,row in areas.iterrows()]),crs='EPSG:4326')
            pnts = sjoin(pnts, shapes, how='left')
            pnts.loc[pnts.id.isna(),'id']=''
            ids = pnts.groupby('Survey')[['id']].apply(setarea).reset_index().set_index('TimeStamp')
            pnts.id = ids.id
            pnts.loc[pnts.id=='','id'] ='NOAREA'
            # except:
            #     pnts = drone
            #     pnts['id'] ='NOAREA'
                          
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