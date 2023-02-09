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



 
      
def task_make_surveys():
        def process_surveys(dependencies, targets,cfg):
            drone =pd.read_csv(dependencies[0],index_col='TimeStamp',parse_dates=['TimeStamp'])
            for name,data in drone.groupby('Survey'):
                data['Counter'] = 1
                data['Counter'] = data['Counter'].cumsum()
                #data['SurveyId'] =f'{data.id.max()}_{data.index.min().strftime("%Y%m%dT%H%M")}'
                data['Extension']=data['SourceFile'].apply(lambda x: os.path.splitext(x)[1]).str.upper()
                data['NewName']=data.apply(lambda item: f"{cfg['survey']['dronetype']}_{cfg['survey']['cameratype']}_{cfg['survey']['country']}_{item.id}_{item.name.strftime('%Y%m%dT%H%M%S')}_{item.Counter:04}{item['Extension']}", axis=1)
                filename = os.path.join(basepath,cfg['paths']['process'],f"{data['SurveyId'].min()}_survey.csv")                
                data.to_csv(filename,index=True,escapechar='"')
            
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
            
@create_after(executed='make_surveys', target_regex='.*\surveyswitharea.csv')             
def task_calculate_survey_areas():
    def poly_to_points(polygon):
        return np.dstack(polygon.exterior.coords.xy)
    
    def survey_area(grp):
        p=MultiPoint(np.hstack(grp['ImagePolygon'].apply(poly_to_points))[0]).convex_hull
        return p.area
    
    def calculate_area(dependencies, targets):
        data =pd.read_csv(dependencies[0],index_col='TimeStamp',parse_dates=['TimeStamp'])
        crs = f'epsg:{int(data["UtmCode"][0])}'
        survey = data['SurveyId'][0]
        gdf = gp.GeoDataFrame(data, geometry=data.ImagePolygon.apply(shapely.wkt.loads),crs=crs)
        gdf['ImagePolygon'] = data.ImagePolygon.apply(shapely.wkt.loads)
        gdf['SurveyAreaHec'] = survey_area(gdf)/10000
        gdf.to_csv(targets[0],index=True,escapechar='"')
        
    config = {"config": get_var('config', 'NO')}
    with open(config['config'], 'r') as ymlfile:
        cfg = yaml.load(ymlfile, yaml.SafeLoader)
    basepath = os.path.dirname(config['config'])
    file_dep = glob.glob(os.path.join(basepath,cfg['paths']['process'],'*_survey.csv'),recursive=True)
    for file in file_dep:
        target = file.replace('_survey','_survey_area')
        yield {
            'name':file,
            'actions':[calculate_area],
            'file_dep':[file],
            'targets':[target],
            'uptodate': [True],
            'clean':True,
        } 


@create_after(executed='calculate_survey_areas', target_regex='.*\surveyswitharea.csv')                  
def task_images_dest():
        def process_images(dependencies, targets,destination):
            survey = pd.read_csv(dependencies[0])
            os.makedirs(destination,exist_ok=True)
            survey['FileDest'] = survey['NewName'].apply(lambda x: os.path.join(destination,x))
            survey.to_csv(targets[0],index=False,escapechar='"')
          
            
            
        config = {"config": get_var('config', 'NO')}
        with open(config['config'], 'r') as ymlfile:
            cfg = yaml.load(ymlfile, yaml.SafeLoader)
        basepath = os.path.dirname(config['config'])
        file_dep = glob.glob(os.path.join(basepath,cfg['paths']['process'],'*_survey_area.csv'))
        for file in file_dep:
            country = os.path.basename(file).split('_')[0]
            sitecode = '_'.join(os.path.basename(file).split('_')[1:3])
            target = file.replace('_survey_area','_survey_data')
            dest = os.path.join(cfg['paths']['output'],country,sitecode)
            yield {
                'name':file,
                'actions':[(process_images, [],{'destination':dest})],
                'file_dep':[file],
                'targets':[target],
                'uptodate': [True],
                'clean':True,
            } 

@create_after(executed='images_dest', target_regex='.*\surveyswitharea.csv')                  
def task_file_images():
        def process_images(dependencies, targets):
            survey = pd.read_csv(dependencies[0])
            destination =os.path.dirname(targets[0])
            os.makedirs(destination,exist_ok=True)
            for index,row in survey.iterrows():
                if not os.path.exists(row.FileDest):
                    #os.symlink(row.SourceFile, row.FileDest)
                    if os.path.exists(row.SourceFile):
                        shutil.copyfile(row.SourceFile,row.FileDest)
            shutil.copyfile(dependencies[0],targets[0])
            
            
            
        config = {"config": get_var('config', 'NO')}
        with open(config['config'], 'r') as ymlfile:
            cfg = yaml.load(ymlfile, yaml.SafeLoader)
        basepath = os.path.dirname(config['config'])
        file_dep = glob.glob(os.path.join(basepath,cfg['paths']['process'],'*_survey_data.csv'))
        for file in file_dep:
            country = os.path.basename(file).split('_')[0]
            sitecode = '_'.join(os.path.basename(file).split('_')[1:3])
            dest = os.path.join(cfg['paths']['output'],country,sitecode)
            target = os.path.join(dest,os.path.basename(file))
            os.path.dirname
            yield {
                'name':file,
                'actions':[process_images],
                'file_dep':[file],
                'targets':[target],
                'uptodate': [True],
                'clean':True,
            } 


                      
            


if __name__ == '__main__':
    import doit
    DOIT_CONFIG = {'check_file_uptodate': 'timestamp'}
    #print(globals())
    doit.run(globals())   