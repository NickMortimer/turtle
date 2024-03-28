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
from shapely.geometry import MultiPoint
import utils.config as config
from utils.utils import convert_wgs_to_utm
from doit import create_after
from turtledrone.process.initalise import task_make_area_list
from turtledrone.process.initalise import task_make_grids

# def task_make_grids():
#     def process_grid(dependencies, targets,gridsize=10):
#         area = gp.read_file(dependencies[0])
#         utmcode = convert_wgs_to_utm(area.iloc[0].geometry.exterior.coords.xy[0][0],area.iloc[0].geometry.exterior.coords.xy[1][0])
#         crs = f'epsg:{utmcode}' 
#         polygon = area.to_crs(crs).iloc[0].geometry
#         eastings =polygon.exterior.coords.xy[0]
#         northings =polygon.exterior.coords.xy[1]
#         easting =np.arange(np.min(eastings) -np.min(eastings) % gridsize,np.max(eastings) -np.max(eastings) % gridsize,gridsize)
#         northing=np.arange(np.min(northings) -np.min(northings) % gridsize,np.max(northings) -np.max(northings) % gridsize,gridsize)
#         xi,yi = np.meshgrid(easting,northing)
#         points =  MultiPoint(list(zip(xi.ravel(),yi.ravel())))
#         p =points.intersection(polygon)
#         d = {'Grid': ['25m'], 'geometry': [p]}
#         df =gp.GeoDataFrame(d, crs=crs)
#         df.to_file(targets[0])        
      



#     file_dep = glob.glob(os.path.join(config.geturl('surveyarea'),'**/*_AOI.shp'),recursive=True)
#     for file in file_dep:
#         target = file.replace('_AOI.shp','_Grid.shp')
#         yield {
#             'name':target,
#             'actions':[(process_grid,[],{'gridsize':int(config.cfg['gridsize'])})],
#             'file_dep':[file],
#             'targets':[target],
#             'clean':True,
#         }   

def task_detect_surveys():
        """
        detect the surveys from the raw directory
        photos have to be in the DCIM directory
        config file
            timedelta : '20MIN'
        continous images if the gap is less than time delta then it
        s counted as one survey 
        """
        def process_survey(dependencies, targets,timedelta,maxpitch):
            drone =pd.read_csv(list(dependencies)[0],index_col='TimeStamp',parse_dates=['TimeStamp'])
            #drone = drone[drone.GimbalPitchDegree<maxpitch].copy()
            os.makedirs(os.path.dirname(targets[0]),exist_ok=True)
            drone.sort_index(inplace=True)
            #remove duplicated time stamps
            drone = drone[~drone.index.duplicated()]
            drone['SourceDrive'] =drone.SourceFile.str.extract(r'^(.*?DCIM)')
            if timedelta == '0MIN':
                drone=drone.groupby('SourceDrive').ngroup()
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
            'actions':[(process_survey, [],{'timedelta':config.cfg['timedelta'],'maxpitch':config.cfg['maxpitch']})],
            'file_dep':[file_dep],
            'targets':targets,
            'clean':True,
        }   

def task_assign_area():
        """
        load the shapes from the shaoe_list and compare the flights to the shapes.
        """
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
            areas =pd.read_csv(areafile)
            areas = areas[areas.Type=='SurveyArea']
            shapes =gp.GeoDataFrame(pd.concat([load_shape(row) for index,row in areas.iterrows()]),crs='EPSG:4326')
            pnts = sjoin(pnts, shapes, how='left')
            pnts.loc[pnts.id.isna(),'id']=''
            ids = pnts.groupby('Survey')[['id']].apply(setarea).reset_index().set_index('TimeStamp')
            pnts.id = ids.id
            pnts.loc[pnts.id=='','id'] ='NOAREA'              
            pnts['SurveyId']=countrycode+'_'+pnts['id']+'_'+pnts[['ImageHeight','Survey']].groupby('Survey').transform(lambda x: x.index.min().strftime("%Y%m%dT%H%M"))['ImageHeight']
            pnts.to_csv(targets[0])
            

        file_dep = [os.path.join(config.geturl('process'),'surveys.csv'),
                    os.path.join(config.geturl('process'),'surveyareas.csv')]
        target = os.path.join(config.geturl('process'),'surveyswitharea.csv')
        return {
            'actions':[(process_assign_area,[],{'countrycode':config.cfg["country"]})],
            'file_dep':file_dep,
            'targets':[target],
            'clean':True,
        }       

if __name__ == '__main__':
    import doit
    DOIT_CONFIG = {'check_file_uptodate': 'timestamp'}
    #print(globals())
    doit.run(globals())   