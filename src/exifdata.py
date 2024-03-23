import os
import glob
import doit
import numpy as np
import yaml
import pandas as pd
from doit import get_var
from doit.tools import run_once
from doit import create_after
import numpy as np
import config
from shutil import which



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
         "FOV","Latitude",'Longitude','SubSecDateTimeOriginal','FlightYSpeed','FlightXSpeed','FlightYSpeed'
         'Orientation','ShutterSpeedValue','ApertureValue','WhiteBalance','RtkFlag','DewarpData','DewarpFlag','Model'}

 					
										
								
													
								
							
										
	


def task_create_json():
        exifpath = os.path.join(config.geturl('exiftool'))
        for item in config.geturl('imagesource').rglob('.'):
            file_dep = list(item.glob(config.cfg['imagewild']))
            if len(file_dep)>0:
                target  = item /'exif.json'
                if file_dep:
                    if which('exiftool'):
                        yield {
                            'name':str(target),
                            'actions':[f'exiftool -ext JPG -ext jpg -json "{item.resolve()}" > "{target.resolve()}"'],
                            'targets':[target],
                            'uptodate':[True],
    #                        'uptodate': [check_timestamp_unchanged(file_dep, 'ctime')],
                            'clean':True,
                        }
                    else:
                        yield {
                             
                            'name':str(target),+
                            'actions':[f'"{exifpath}" -ext JPG -ext jpg -json "{os.path.abspath(item)}" > "{os.path.abspath(target)}"'],
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
            if os.stat(source_file).st_size > 0:
                drone = pd.read_json(source_file)
                if 'GPSLongitude' in drone.columns:
                    def get_longitude(item):
                        longitude =float(item[0]) + float(item[2][0:-1])/60 + float(item[3][0:-1])/3600
                        return (longitude)
                    drone['Longitude'] = np.nan
                    drone['Latitude'] = np.nan
                    drone.loc[ ~drone['GPSLongitude'].isna(),'Longitude']=drone.loc[ ~drone['GPSLongitude'].isna(),'GPSLongitude'].str.split(' ',expand=True).apply(get_longitude,axis=1)
                    drone.loc[ ~drone['GPSLatitude'].isna(),'Latitude']=drone.loc[ ~drone['GPSLatitude'].isna(),'GPSLatitude'].str.split(' ',expand=True).apply(get_longitude,axis=1)
                    drone.loc[drone['GPSLatitudeRef']=='South','Latitude'] =drone.loc[drone['GPSLatitudeRef']=='South','Latitude']*-1
                    drone = drone[drone.columns[drone.columns.isin(wanted)]]
                    if 'SubSecDateTimeOriginal' in drone.columns:
                        drone['TimeStamp'] = pd.to_datetime(drone.SubSecDateTimeOriginal,format='%Y:%m:%d %H:%M:%S.%f')
                    else:
                        drone['TimeStamp'] = pd.to_datetime(drone.DateTimeOriginal,format='%Y:%m:%d %H:%M:%S')
                    sourcepath = config.CATALOG_DIR
                    drone['SourceRel'] =drone.SourceFile.apply(lambda x: os.path.relpath(x,start=sourcepath))
                    drone['Sequence'] =drone.SourceFile.str.extract('(?P<Sequence>\d+)\.(jpg|JPG)')['Sequence']
                    drone.set_index('Sequence',inplace=True)
                    drone.to_csv(list(targets)[0],index=True)
            

        for item in list(config.geturl('imagesource').rglob('exif.json')):
            file_dep  =  item
            target =   item.with_suffix('.csv')           
            yield {
                'name':target,
                'actions':[process_json],
                'file_dep':[file_dep],
                'targets':[target],
                'clean':True,
            }
            


            
            
           
        
if __name__ == '__main__':
    import doit
    DOIT_CONFIG = {'check_file_uptodate': 'timestamp'}
    #print(globals())
    doit.run(globals())   