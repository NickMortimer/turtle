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
from  utils import convert_wgs_to_utm


 

 
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


def task_survey_areas():
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
                           'Area':d.SurveyAreaHec.mean(),
                           'Missing':missing}]).to_csv(targets[0],index=False)
            
        config = {"config": get_var('config', 'NO')}
        with open(config['config'], 'r') as ymlfile:
            cfg = yaml.load(ymlfile, yaml.SafeLoader)
        basepath = os.path.dirname(config['config'])
        file_dep = glob.glob(os.path.join(cfg['paths']['output'],'**','*_survey_area.csv'),recursive=True)
        for file in file_dep:
            target = file.replace('_survey_area.csv','_survey_area_summary.csv')
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
        file_dep = glob.glob(os.path.join(cfg['paths']['output'],'**','*_survey_area_summary.csv'),recursive=True)
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
    DOIT_CONFIG = {'check_file_uptodate': 'timestamp'}
    #print(globals())
    doit.run(globals())
