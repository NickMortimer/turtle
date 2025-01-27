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
from shapely.geometry import MultiPolygon
from jinja2 import Environment, FileSystemLoader
from pathlib import Path
import turtledrone.config as config
import shutil


# config = {"config": get_var('config', 'NO')}
# with open(config['config'], 'r') as ymlfile:
#     cfg = yaml.load(ymlfile, yaml.SafeLoader)
# basepath = os.path.dirname(config['config'])

def task_set_up():
    config.read_config()
         
   
def task_check_survey(): 
    """
    Create a summary file for each survey
    """ 
    def process_check_survey(dependencies, targets):
        drone =pd.read_csv(dependencies[0],index_col='TimeStamp',parse_dates=['TimeStamp'])
        images = pd.DataFrame(glob.glob(os.path.join(os.path.dirname(drone['FileDest'].min()),'*.JPG')),
                                columns=['Path'])
        images['NewName'] = images['Path'].apply(os.path.basename)
        d =drone.join(images.set_index('NewName'),how='left',on=['NewName'],rsuffix="f")
        missing =d.Path.isna().sum()
        json=len(list(Path(dependencies[0]).parent.glob('*.json')))
        expected = d.NewName.count()
        coverage = 100*(expected-missing)/expected
        df=pd.DataFrame([{'SurveyId':d.SurveyId.max(),
                        'StartTime':d.index.min(),'EndTime':d.index.max(),
                        'Latitude':d.Latitude.mean(),'Longitude':d.Longitude.mean(),
                        'Coverage':coverage,'Expected':expected,
                        'Area':d.SurveyAreaHec.mean(),
                        'Missing':missing,
                        'JsonCount':json}]).to_csv(targets[0],index=False)
    file_dep =  list(config.geturl('output').rglob('*_survey_area_data.csv'))
    for file in file_dep:
        target = file.parent / file.name.replace('_survey_area_data.csv','_survey_area_data_summary.csv')
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
    """
    Concat each summary file into a single file 
    """
    def process_concat_check_survey(dependencies, targets):
        os.makedirs(os.path.dirname(targets[0]),exist_ok=True)
        surveys =pd.concat([pd.read_csv(file) for file in dependencies])
        surveys =surveys.set_index('SurveyId').sort_index()
        surveys.to_csv(targets[0])
        
    file_dep = list(config.geturl('output').rglob('*_survey_area_data_summary.csv'))
    target = config.geturl('reports') /'image_coverage.csv'
    return {
        'actions':[process_concat_check_survey],
        'file_dep':file_dep,
        'targets':[target],
        'clean':True,
        }     
        
def task_plot_surveys():
    """
    Plot each survey make sure your mapbox api key
    config
        mapboxkey : your mapboxkey
        reports : sets the destination of the files
    """
    def process_survey(dependencies, targets,apikey):
        drone =pd.concat([pd.read_csv(file,index_col='TimeStamp',parse_dates=['TimeStamp']) for file in dependencies])
        drone = drone[drone.Longitude>0]
        px.set_mapbox_access_token(apikey)
        max_bound = max(abs(drone.Longitude.max()-drone.Longitude.min()), abs(drone.Latitude.max()-drone.Latitude.min())) * 111
        zoom = 11.5 - np.log(max_bound)
        fig = px.scatter_mapbox(drone, hover_name='SurveyId', lat="Latitude", lon="Longitude",  
                                mapbox_style="satellite-streets",color="SurveyId", size_max=30, zoom=zoom)
        fig.update_layout(mapbox_style="satellite-streets")
        os.makedirs(os.path.dirname(targets[0]),exist_ok=True)
        html_file =list(filter(lambda x: 'html' in x, targets))[0]
        png_file =list(filter(lambda x: 'png' in x, targets))[0]
        plotly.offline.plot(fig, filename=html_file,auto_open = False)
        fig.write_image(png_file)
    file_dep = list(config.geturl('output').rglob('*_survey_area_data.csv'))
    targets = [os.path.join(config.geturl('reports'),'surveys.html'),
               os.path.join(config.geturl('reports'),'surveys.png')]
               
    return {

        'actions':[(process_survey, [],{'apikey':config.cfg['mapboxkey']})],
        'file_dep':file_dep,
        'targets':targets,
        'clean':True,
    }  
    
def task_plot_each_survey():
    """
    Plot each flight to a png file
    """
    def process_survey(dependencies, targets,apikey):
        drone =pd.read_csv(list(dependencies)[0],index_col='TimeStamp',parse_dates=['TimeStamp'])
        px.set_mapbox_access_token(apikey)
        max_bound = max(abs(drone.Longitude.max()-drone.Longitude.min()), abs(drone.Latitude.max()-drone.Latitude.min())) * 111
        zoom = 13.5 - np.log(max_bound)
        fig = px.scatter_mapbox(drone, hover_name='NewName', lat="Latitude", lon="Longitude",  
                                mapbox_style="satellite-streets",color="SurveyId", size_max=30, zoom=zoom)        
        fig.update_layout(mapbox_style="satellite-streets",autosize=False)
        html_file =list(filter(lambda x: 'html' in x, targets))[0]
        png_file =list(filter(lambda x: 'png' in x, targets))[0]
        plotly.offline.plot(fig, filename=html_file,auto_open = False)
        fig.update_layout(coloraxis_showscale=False,showlegend=False,autosize=False,margin = dict(t=10, l=10, r=10, b=10))
        fig.write_image(png_file)
        
        
    file_dep = list(config.geturl('output').rglob('*_survey_area_data.csv'))
    for inputfile in file_dep:
        targets = [config.geturl('reports') / inputfile.name.replace('csv','html'), config.geturl('reports') / inputfile.name.replace('csv','png')]
        yield {
            'name':targets[0],
            'actions':[(process_survey, [],{'apikey':config.cfg['mapboxkey']})],
            'file_dep':[inputfile],
            'targets':targets,
            'clean':True,
        }        
    
def task_geopgk_survey():
    """
    Create geopgk file for QGIS 
    """
    def process_geo(dependencies, targets):
        data =pd.read_csv(dependencies[0],index_col='TimeStamp',parse_dates=['TimeStamp'])
        if "UtmCode" in data.columns:
            crs = f'epsg:{int(data["UtmCode"].iloc[0])}'
            survey = data['SurveyId'].iloc[0]
            gdf = gp.GeoDataFrame(data, geometry=data.ImagePolygon.apply(lambda x: shapely.wkt.loads(x)),crs=crs)

            if os.path.exists(targets[0]):
                os.remove(targets[0])
            gdf.loc[(gdf.GimbalPitchDegree*-1)>80].to_file(targets[0], driver="GeoJSON")                
#            gdf.to_file(targets[0], driver="GPKG", layer=survey)

        
    file_dep = glob.glob(os.path.join(config.geturl('output'),config.cfg['country'],'**','*_survey_area_data.csv'),recursive=True)
    for file in file_dep:
        target = os.path.splitext(os.path.basename(file))[0]+'.gpkg'
        target = os.path.join(config.geturl('reports'),target)
        #countries_gdf
        yield {
            'name':file,
            'actions':[process_geo],
            'file_dep':[file],
            'targets':[target],
            'uptodate': [True],
            'clean':True,
        }  
    
def task_html_report():
    """
    Make a html report of the flight
    """
    def process_report(dependencies, targets):
        data =pd.read_csv(dependencies[0]).iloc[0]
        env = Environment(loader=FileSystemLoader(os.path.join(Path(os.path.abspath(__file__)).parent.parent,'templates')))
        template = env.get_template('report.html')
        html = template.render(SurveyId=data.SurveyId,
                               StartTime=data.StartTime,
                               EndTime=data.EndTime,
                               Coverage=data.Coverage,
                               Area=data.Area.round(1),
                               Missing=data.Missing,
                               Expected=data.Expected,
                               Longitude=data.Longitude.round(6),
                               Latitude=data.Latitude.round(6),
                               map=os.path.basename(dependencies[0].replace('_survey_area_data_summary.csv','_survey_area_data.png')))
        with open(targets[0], 'w') as f:
            f.write(html)        


        

    file_dep = glob.glob(os.path.join(config.cfg['output'],config.cfg['country'],'**','*_survey_area_data_summary.csv'),recursive=True)
    for file in file_dep:
        target = os.path.splitext(os.path.basename(file))[0]+'_report.html'
        target = os.path.join(config.cfg['reports'],target)
        #countries_gdf
        yield {
            'name':file,
            'actions':[process_report],
            'file_dep':[file],
            'targets':[target],
            'uptodate': [True],
            'clean':True,
        }  

# def task_data_summary():
#     """
#     create a summary list of the flights with 
#     """
#     def process_surveys(dependencies, targets):
#         drone =pd.concat([pd.read_csv(file,index_col='SurveyId',parse_dates=['StartTime','EndTime']) for file in dependencies])
#         drone['ElapsedTime'] = drone.EndTime - drone.StartTime 
#         drone['ElapsedTime'] = (drone['ElapsedTime'].dt.total_seconds()).astype(int) /60
#         drone['SurveyRate'] =  drone.Area /drone.ElapsedTime 
#         drone = drone.sort_index()
#         drone.to_csv(targets[0]) 

        
#     surveys = os.path.join(config.geturl('output'),'*survey_area_data_summary.csv')
#     file_dep = glob.glob(surveys,recursive=True)
#     os.makedirs(config.geturl('reports'),exist_ok=True)
#     target = f"{config.geturl('reports')}/survey_area_data_summary.csv"
#     return {
#         'actions':[process_surveys],
#         'file_dep':file_dep,
#         'targets':[target],
#         'clean':True,
#     } 
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
                    