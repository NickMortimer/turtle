
import os
import glob
import doit
import glob
import os
import numpy as np
import pandas as pd
from doit import create_after
import numpy as np
import geopandas as gp
from drone import P4rtk
from pyproj import Proj 

from  turtledrone.utils.utils import convert_wgs_to_utm
import config as config


 

 
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
            drone.set_index('TimeStamp',inplace=True)
            drone.sort_index(inplace=True)
            drone = drone[pd.notna(drone.index)]
            drone.to_csv(list(targets)[0],index=True)

        for item in config.geturl('imagesource').rglob('exif.csv'):
            target =   item.parent / 'position.csv'           
            yield {
                'name':target,
                'actions':[process_json],
                'file_dep':[item],
                'targets':[target],
                'clean':True, 
               }    
@create_after(executed='process_mergpos', target_regex='*.csv')      
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
        
        
        

    dewarp = pd.to_numeric(config.cfg['dewarp'] )
    for file_dep in glob.glob(os.path.join(config.geturl('imagesource'),'position.csv'),recursive=True):
        target = os.path.join(os.path.dirname(file_dep),'polygons.csv')   
        yield {
            'name':file_dep,
            'actions':[(process_polygons, [],{'dewarp':dewarp})],
            'file_dep':[file_dep],
            'targets':[target],
            'clean':True,
        }       
@create_after(executed='addpolygons', target_regex='*.csv')          
def task_merge_xif():
        def process_xif(dependencies, targets):
            target = list(targets)[0]
            os.makedirs(os.path.dirname(target),exist_ok=True)
            drone = pd.concat([pd.read_csv(file,index_col='TimeStamp',parse_dates=['TimeStamp']) 
                            for file in list(dependencies)]) 
            drone.sort_index(inplace=True)
            drone['FileName'] =drone.SourceFile.apply(os.path.basename)
            drone =drone.drop_duplicates(subset= ['ImageTime','FileName'])
            drone.to_csv(list(targets)[0],index=True)
            
        searchpath = os.path.join(config.geturl('imagesource'),'polygons.csv')
        file_dep = glob.glob(searchpath,recursive=True)
        processpath =config.geturl('process')
        os.makedirs(processpath,exist_ok=True)
        target = os.path.join(processpath,'imagedata.csv')
        return {
            'actions':[process_xif],
            'file_dep':file_dep,
            'targets':[target],
            'clean':True,
        }



                
            
# def task_calculate_newname():
#     pass
#xifdata.apply(lambda item: f"{survey['dronetype']}_{survey['camera']}_{survey['country']}_{survey['surveycode']}_{survey['surveynumber']:03}_{item.LocalTime}_{item.Counter:04}.JPG", axis=1)       
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
