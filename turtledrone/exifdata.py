import os
import doit
import numpy as np
import pandas as pd
import numpy as np
import turtledrone.config as config
from shutil import which
from doit import create_after


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
        """
            Scan through the raw directories and make a json file with the xif data in
            from the config file
            imagesource :  "{CATALOG_DIR}/TR2023-02/instruments/P4RTK/work/raw"
            imagewild : "*_*_*.JPG"

            path to exiftool
            exiftool : exiftool
        """
        exifpath = os.path.join(config.geturl('exiftool'))
        for item in config.geturl('imagesource').rglob('.'):
            file_dep = list(item.glob(config.cfg['imagewild'].upper()))
            if len(file_dep)==0:
                file_dep = list(item.glob(config.cfg['imagewild'].lower()))
            if len(file_dep)>0:
                target  = item /'exif.json'
                if file_dep:
                    if which('exiftool'):
                        yield {
                            'name':str(target),
                            'actions':[f'exiftool -ext JPG -ext jpg -json "{item.resolve()}" > "{target.resolve()}"'],
                            'targets':[target],
                            'uptodate':[True],
                            'clean':True,
                        }
                    else:
                        yield {
                             
                            'name':str(target),+
                            'actions':[f'"{exifpath}" -ext JPG -ext jpg -json "{os.path.abspath(item)}" > "{os.path.abspath(target)}"'],
                            'targets':[target],
                            'uptodate':[True],
                            'clean':True,
                        }
 
    
@create_after(executed='create_json', target_regex='.*\exif.json')    
def task_process_json():
        """
            use to find all *.json files and convert them to csv files
            imagesource :  "{CATALOG_DIR}/TR2023-02/instruments/P4RTK/work/raw"
        """
        def process_json(dependencies, targets):
            source_file =dependencies[0]
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
                    # ok lets breakout the calibrations
                    if ('DewarpData' in drone.columns) and (~drone['DewarpData'].isna().max()):
                        drone[['CalibrationDate','CalibratedFocalLengthX','CalibratedFocalLengthY','CalibratedOpticalCenterX','CalibratedOpticalCenterY',
                                    'K1','K2','P1',"P2","K3"]] = drone['DewarpData'].str.split(r'[;,]',expand=True)
                        drone[['CalibratedFocalLengthX','CalibratedFocalLengthY','CalibratedOpticalCenterX','CalibratedOpticalCenterY',
                                    'K1','K2','P1',"P2","K3"]] = drone[['CalibratedFocalLengthX','CalibratedFocalLengthY','CalibratedOpticalCenterX','CalibratedOpticalCenterY',
                                    'K1','K2','P1',"P2","K3"]].astype(float)
                        drone['CalibratedOpticalCenterX'] = (drone['ImageWidth']/2)+drone['CalibratedOpticalCenterX']
                        drone['CalibratedOpticalCenterY'] = (drone['ImageHeight']/2)+drone['CalibratedOpticalCenterY']
                    elif not ('DewarpFlag' in drone.columns): 
                        drone[['CalibratedOpticalCenterX','CalibratedOpticalCenterY','CalibratedFocalLength']]  = [2432,1824,3666.665]
                    
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