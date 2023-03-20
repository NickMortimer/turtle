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
 
    
def task_make_surveys():
        def process_surveys(dependencies, targets):
            drone =pd.read_csv(dependencies[0],index_col='TimeStamp',parse_dates=['TimeStamp'])
            for name,data in drone.groupby('Survey'):
                data['Counter'] = 1
                data['Counter'] = data['Counter'].cumsum()
                #data['SurveyId'] =f'{data.id.max()}_{data.index.min().strftime("%Y%m%dT%H%M")}'
                data['Extension']=data['SourceFile'].apply(lambda x: os.path.splitext(x)[1]).str.upper()
                data['NewName']=data.apply(lambda item: f"{config.cfg['survey']['dronetype']}_{config.cfg['survey']['cameratype']}_{config.cfg['survey']['country']}_{item.id}_{item.name.strftime('%Y%m%dT%H%M%S')}_{item.Counter:04}{item['Extension']}", axis=1)
                filename = os.path.join(config.geturl('process'),f"{data['SurveyId'].min()}_survey.csv")                
                data.to_csv(filename,index=True)
            
        file_dep = os.path.join(config.geturl('process'),'surveyswitharea.csv')
        if os.path.exists(file_dep):
            surveys =pd.read_csv(file_dep,index_col='TimeStamp',parse_dates=['TimeStamp']).groupby('Survey')
            targets = [os.path.join(config.geturl('process'),
                                    f'{config.cfg["survey"]["country"]}_{data.id.max()}_{data.index.min().strftime("%Y%m%dT%H%M")}_survey.csv') for name,data in surveys]
            return {
                'actions':[(process_surveys,[])],
                'file_dep':[file_dep],
                'targets':targets,
                'clean':True,
            } 
            
@create_after(executed='make_surveys')             
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
        gdf.to_csv(targets[0],index=True)
        

    file_dep = glob.glob(os.path.join(config.geturl('process'),'*_survey.csv'),recursive=True)
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
            survey.to_csv(targets[0],index=False)

        file_dep = glob.glob(os.path.join(config.geturl('process'),'*_survey_area.csv'))
        for file in file_dep:
            target = file.replace('_survey_area','_survey_area_data')
            yield {
                'name':file,
                'actions':[(process_images, [],{'destination':config.getdest(os.path.basename(file))})],
                'file_dep':[file],
                'targets':[target],
                'uptodate': [True],
                'clean':True,
            } 

@create_after(executed='images_dest', target_regex='.*\surveyswitharea.csv')                  
def task_file_images():
        def process_images(dependencies, targets):
            destination =os.path.dirname(targets[0])
            os.makedirs(destination,exist_ok=True)
            survey = pd.read_csv(dependencies[0])
            for index,row in survey.iterrows():
                if not os.path.exists(row.FileDest):
                    if config.cfg['survey']['outputsymlink']:
                        os.symlink( os.path.join(config.CATALOG_DIR,row.SourceRel), row.FileDest )
                    else:
                        shutil.copyfile(row.SourceFile,row.FileDest)
            shutil.copyfile(dependencies[0],targets[0])
            

        file_dep = glob.glob(os.path.join(config.geturl('process'),'*_survey_area_data.csv'))
        for file in file_dep:
            target = os.path.join(config.getdest(os.path.basename(file)),os.path.basename(file))
            os.path.dirname
            yield {
                'name':file,
                'actions':[process_images],
                'file_dep':[file],
                'targets':[target],
                'uptodate': [True],
                'clean':True,
            } 

def task_move_summary():
    def move_smmary(dependencies, targets):
        shutil.copyfile(dependencies[0],targets[0])
        
    
    file_dep = glob.glob(os.path.join(config.geturl('process'),'*_survey_area_data_summary.csv'))
    for file in file_dep:
        target = os.path.join(config.getdest(os.path.basename(file)),os.path.basename(file))
        yield {
            'name':file,
            'actions':[move_smmary],
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