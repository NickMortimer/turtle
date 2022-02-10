import os
import glob
import doit
import glob
import os
import numpy as np
import yaml
import pandas as pd
from doit import get_var
from doit import create_after
import numpy as np
import plotly
import plotly.express as px
import geopandas as gp
import shapely.wkt
from shapely.geometry import MultiPoint


config = {"config": get_var('config', 'NO')}
with open(config['config'], 'r') as ymlfile:
    cfg = yaml.load(ymlfile, yaml.SafeLoader)
basepath = os.path.dirname(config['config'])


         
   
def task_check_survey():
    global cfg
    global basepath    
    def process_check_survey(dependencies, targets):
        drone =pd.read_csv(dependencies[0],index_col='TimeStamp',parse_dates=['TimeStamp'])
        images = pd.DataFrame(glob.glob(os.path.join(drone['FileDest'].min(),'*.JPG')),
                                columns=['Path'])
        images['NewName'] = images['Path'].apply(os.path.basename)
        d =drone.join(images.set_index('NewName'),how='left',on=['NewName'],rsuffix="f")
        missing =d.Path.isna().sum()
        expected = d.NewName.count()
        coverage = 100*(expected-missing)/expected
        df=pd.DataFrame([{'SurveyId':d.SurveyId.max(),
                        'StartTime':d.index.min(),'EndTime':d.index.max(),
                        'Latitude':d.Latitude.mean(),'Longitude':d.Longitude.mean(),
                        'Coverage':coverage,'Expected':expected,
                        'Area':d.SurveyAreaHec.mean(),
                        'Missing':missing}]).to_csv(targets[0],index=False)

    file_dep = glob.glob(os.path.join(basepath,cfg['paths']['process'],'*_survey_data.csv'),recursive=True)
    for file in file_dep:
        target = file.replace('_survey_data.csv','_survey_summary.csv')
        yield {
            'name':file,
            'actions':[process_check_survey],
            'file_dep':[file],
            'targets':[target],
            'uptodate': [True],
            'clean':True,
        } 
                
  
            
            
 
            
      
@create_after(executed='check_survey', target_regex='*_survey_summary')  
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
    file_dep = glob.glob(os.path.join(basepath,cfg['paths']['process'],'*_survey_summary.csv'),recursive=True)
    target = os.path.join(cfg['paths']['reports'],'image_coverage.csv')
    return {
        'actions':[process_concat_check_survey],
        'file_dep':file_dep,
        'targets':[target],
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
        

    basepath = os.path.dirname(config['config'])
    file_dep = os.path.join(basepath,cfg['paths']['process'],'surveys.csv')
    targets = os.path.join(basepath,cfg['paths']['process'],'surveys.html')
    return {

        'actions':[(process_survey, [],{'apikey':cfg['mapboxkey']})],
        'file_dep':[file_dep],
        'targets':[targets],
        'clean':True,
    }    
    
def task_geopgk_survey():
    def process_geo(dependencies, targets):
        data =pd.read_csv(dependencies[0],index_col='TimeStamp',parse_dates=['TimeStamp'])
        if "UtmCode" in data.columns:
            crs = f'epsg:{int(data["UtmCode"][0])}'
            survey = data['SurveyId'][0]
            gdf = gp.GeoDataFrame(data, geometry=data.ImagePolygon.apply(shapely.wkt.loads),crs=crs)
            gdf.to_file(targets[0], layer=survey, driver="GPKG")

        
    config = {"config": get_var('config', 'NO')}
    with open(config['config'], 'r') as ymlfile:
        cfg = yaml.load(ymlfile, yaml.SafeLoader)
    basepath = os.path.dirname(config['config'])
    file_dep = glob.glob(os.path.join(cfg['paths']['output'],cfg['survey']['country'],'**','*_survey.csv'),recursive=True)
    for file in file_dep:
        target = os.path.splitext(os.path.basename(file))[0]+'.gpkg'
        target = os.path.join(cfg['paths']['reports'],target)
        #countries_gdf
        yield {
            'name':file,
            'actions':[process_geo],
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
                    