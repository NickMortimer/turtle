import os
import glob
import doit
import glob
import os
import pandas as pd

import cameratransform as ct
from pathlib import Path
import numpy as np
from shapely import Polygon
from datetime import datetime

from turtledrone import drone

try:
    import geomag
    GEOMAG_AVAILABLE = True
except ImportError:
    GEOMAG_AVAILABLE = False
    print("Warning: geomag package not available. Magnetic declination correction will be skipped.")

def get_magnetic_declination(lat, lon, altitude_m, date):
    """
    Calculate magnetic declination at a given location and time.
    
    Parameters:
    -----------
    lat : float
        Latitude in degrees
    lon : float
        Longitude in degrees
    altitude_m : float
        Altitude in meters above sea level
    date : datetime or pandas.Timestamp
        Date and time of the measurement
        
    Returns:
    --------
    float
        Magnetic declination in degrees (positive East, negative West)
        Returns 0.0 if geomag is not available
    """
    if not GEOMAG_AVAILABLE:
        return 0.0
    
    # Convert altitude from meters to kilometers for geomag
    altitude_km = altitude_m / 1000.0
    
    # Convert pandas Timestamp to datetime.date if needed
    if hasattr(date, 'date'):
        date_obj = date.date()
    else:
        date_obj = date
    
    # Get magnetic declination
    mag = geomag.declination(lat, lon, altitude_km, date_obj)
    
    return mag

def task_make_geo():
    """
    Make geo file use by opendrone to enhance locations
    """
    def make_geo(dependencies, targets):
        data = pd.read_csv(dependencies[0])
        data['UTCTimeMrk'] = pd.to_datetime(data['UTCTimeMrk'], utc=True)
        
        # Calculate magnetic declination for each row and apply correction
        if GEOMAG_AVAILABLE:
            if 'latitude_deg_rtk' in data.columns:
                # Use RTK coordinates for declination calculation
                data['MagneticDeclination'] = data.apply(
                    lambda x: get_magnetic_declination(
                        x.latitude_deg_rtk, 
                        x.longitude_deg_rtk, 
                        x.height_m_rtk, 
                        x.UTCTimeMrk
                    ), axis=1
                )
                # Apply declination correction to Yaw (add declination to convert magnetic to true north)
                data['CameraYawCorrected'] = data['GimbalYawDegree'] + data['MagneticDeclination']
                # now wrap yaw to 0-360
                data['CameraYawCorrected'] = data['CameraYawCorrected'] % 360
            elif 'LatitudeMrk' in data.columns:
                # Use non-RTK coordinates for declination calculation
                data['MagneticDeclination'] = data.apply(
                    lambda x: get_magnetic_declination(
                        x.LatitudeMrk, 
                        x.LongitudeMrk, 
                        x.EllipsoideHightMrk, 
                        x.UTCTimeMrk
                    ), axis=1
                )
                # Apply declination correction to CameraYaw
                data['CameraYawCorrected'] = data['GimbalYawDegree'] + data['MagneticDeclination']

        if 'latitude_deg_rtk'  in data.columns:
             text = list(data.apply(lambda x:f'{x.NewName} {x.longitude_deg_rtk} {x.latitude_deg_rtk} {x.height_m_rtk} {x.CameraYawCorrected} {x.CameraPitch+90} {x.CameraRoll} {(x.sde_m_rtk+x.sdn_m_rtk)/2} {x.sdu_m_rtk} {x.UTCTimeMrk.isoformat()}\n',axis=1))
        elif 'LatitudeMrk' in data.columns:
            text = list(data.apply(lambda x:f'{x.NewName} {x.Longitude} {x.Latitude} {x.EllipsoideHightMrk} {x.CameraYawCorrected} {x.CameraPitch+90} {x.CameraRoll} {(x.errXMrk+x.errYMrk)/2} {x.errZMrk} {x.UTCTimeMrk.isoformat()}\n',axis=1))
        with open(targets[0], 'w') as f:
            f.write('EPSG:4326\n')
            f.writelines(text)
        
    from turtledrone.config import cfg as config
    file_dep = glob.glob(os.path.join(config.get_url('output'),'**','*_survey_area_data.csv'),recursive=True)
    for file in file_dep:
        target = os.path.join(config.get_destination(os.path.basename(file)),'geo.txt')
        yield {
            'name':target,
            'actions':[make_geo],
            'file_dep':[file],
            'targets':[target],
            'uptodate': [True],
            'clean':True,
        }            

def task_make_time():
    """
    Make time file use by opendrone to enhance locations
    """
    def make_time(dependencies, targets):
        data = pd.read_csv(dependencies[0])
        # Convert UTCTimeMrk to datetime with UTC timezone
        data['UTCTimeMrk'] = pd.to_datetime(data['UTCTimeMrk'], utc=True)
        # Format as ISO 8601
        text = list(data.apply(lambda x:f'{x.NewName} {x.UTCTimeMrk.isoformat()} \n',axis=1))   
        if 'LatitudeMrk' in data.columns:
            with open(targets[0], 'w') as f:
                f.writelines(text)
        
    from turtledrone.config import cfg as config
    file_dep = glob.glob(os.path.join(config.get_url('output'),'**','*_survey_area_data.csv'),recursive=True)
    for file in file_dep:
        target = os.path.join(config.get_destination(os.path.basename(file)),'time.txt')
        yield {
            'name':target,
            'actions':[make_time],
            'file_dep':[file],
            'targets':[target],
            'uptodate': [True],
            'clean':True,
        }                              


def task_make_cam_file():
    """
    Make time file use by opendrone to enhance locations
    """
    def make_time(dependencies, targets):
        data = pd.read_csv(dependencies[0])
        item=data.iloc[0]
        
        # Helper function to convert numpy/pandas types to native Python types
        def to_python_type(val):
            if pd.isna(val):
                return None
            if hasattr(val, 'item'):  # numpy scalar
                return val.item()
            return float(val) if isinstance(val, (np.integer, np.floating)) else val
        
        if item.DewarpFlag:
            camera_int ={'calibrated_focal_length_x': to_python_type(item.CalibratedFocalLength),
                                        'calibrated_focal_length_y': to_python_type(item.CalibratedFocalLength),
                                        'calibrated_optical_center_x': to_python_type(item.ImageWidth/2),
                                        'calibrated_optical_center_y': to_python_type(item.ImageHeight/2),
                                        'distortion_coefficients': {
                                            'k1': 0.0,
                                            'k2': 0.0,
                                            'p1': 0.0,
                                            'p2': 0.0,
                                            'k3': 0.0,
                                        }
                        }

        else:
            camera_int ={'calibrated_focal_length_x': to_python_type(item.CalibratedFocalLengthX),
                                        'calibrated_focal_length_y': to_python_type(item.CalibratedFocalLengthY),
                                        'calibrated_optical_center_x': to_python_type(item.CalibratedOpticalCenterX),
                                        'calibrated_optical_center_y': to_python_type(item.CalibratedOpticalCenterY),
                                        'distortion_coefficients': {
                                            'k1': to_python_type(item.K1),
                                            'k2': to_python_type(item.K2),
                                            'p1': to_python_type(item.P1),
                                            'p2': to_python_type(item.P2),
                                            'k3': to_python_type(item.K3),
                                        }
                        }



        camera_int['image_width_px'] = int(item.ImageWidth)
        camera_int['image_height_px'] = int(item.ImageHeight) 
        
        # Handle calibration date - save empty string if not a valid string
        if not isinstance(item.CalibrationDate, str):
            camera_int['calibration_date'] = ''
        else:
            camera_int['calibration_date'] = item.CalibrationDate
            
        camera_int['sensor_width_mm'] = 13.28
        camera_int['sensor_height_mm'] = 8.8
        camera_int['model'] = item.Model.lower()
        camera_int['make'] = item.Make.lower()
        camera_int['serial_number'] = item.SerialNumber
        # save this to a YAML file human readable
        import yaml
        with open(targets[0],'w') as f:
            yaml.dump(camera_int, f, default_flow_style=False, sort_keys=False, indent=2)  

        
    from turtledrone.config import cfg as config
    file_dep = glob.glob(os.path.join(config.get_url('output'),'**','*_survey_area_data.csv'),recursive=True)
    for file in file_dep:
        data = pd.read_csv(file)
        item=data.iloc[0]
        if item.DewarpFlag:
            cal_type ='dewarped'
        else:
            cal_type ='raw'
        

        target = config.get_destination(os.path.basename(file)) / f'{item.Make.lower()}_{item.Model.lower()}_{cal_type}_{item.ImageWidth}_{item.ImageHeight}_camera.yml'
        yield {
            'name':target,
            'actions':[make_time],
            'file_dep':[file],
            'targets':[target],
            'uptodate': [True],
            'clean':True,
        }     

# def task_make_labelgps():
#     """
#     make a location file to for use with modified labelme
#     """
#     def process_labelgps(dependencies, targets):
#     #     def get_border(item):
#     #         if 'K1' in item.keys():
#     #             cam = ct.Camera(ct.RectilinearProjection(focallength_x_px=item.CalibratedFocalLengthX,
#     #                                                                  focallength_y_px=item.CalibratedFocalLengthY,
#     #                                                                     center_x_px=item.CalibratedOpticalCenterX,
#     #                                                                     center_y_px=item.CalibratedOpticalCenterY),
#     #                                                                     orientation= ct.SpatialOrientation(tilt_deg=item.GimbalPitchDegree,
#     #                                                                                                     elevation_m=item.RelativeAltitude,
#     #                                                                                                     roll_deg=item.GimbalRollDegree,
#     #                                                                                                     heading_deg=item.GimbalYawDegree),
#     #                                                                     lens=ct.BrownLensDistortion(item.K1,item.K2,item.K3))
#     #         else:
#     #             cam = ct.Camera(ct.RectilinearProjection(focallength_px=item.CalibratedFocalLength,
#     #                                                                 center_x_px=item.CalibratedOpticalCenterX,
#     #                                                                 center_y_px=item.CalibratedOpticalCenterY),
#     #                                                                 orientation= ct.SpatialOrientation(tilt_deg=item.GimbalPitchDegree,
#     #                                                                                                 elevation_m=item.RelativeAltitude,
#     #                                                                                                 roll_deg=item.GimbalRollDegree,
#     #                                                                                                heading_deg=item.GimbalYawDegree))
#     #         perside=10 
#     #         x = np.linspace(0, item.ImageWidth, num=perside)  
#     #         y = np.linspace(0, item.ImageHeight, num=perside)
#     #         bottom = np.dstack((x,np.ones(perside)*item.ImageHeight-1))[0]
#     #         right = np.dstack((np.ones(perside)*item.ImageWidth-1,y[::-1]))[0]
#     #         top = np.dstack((x[::-1],np.zeros(perside)))[0]
#     #         left = np.dstack((np.zeros(perside),y))[0]
#     #         points = np.vstack((bottom,right,top,left))
#     #         polydata =cam.spaceFromImage(points)
#     #         polydata[:,0] =item.ImageEasting + polydata[:,0] 
#     #         polydata[:,1] =item.ImageNorthing +polydata[:,1]
#     #         item.ImagePolygon=Polygon(polydata[:,0:2])
#     #         return item


#         datafile = glob.glob(os.path.join(os.path.dirname(dependencies[0]),'*_survey_area_data.csv'))
#         if len(datafile)==0:
#             datafile = glob.glob(os.path.join(os.path.dirname(dependencies[0]),'*_survey_area.csv'))
#         if len(datafile)==0:
#             datafile = glob.glob(os.path.join(os.path.dirname(dependencies[0]),'*_survey_data.csv'))
        
#         source_file = pd.read_csv(datafile[0])
#         wanted =['TimeStamp','Longitude','Latitude','AbsoluteAltitude','RelativeAltitude','CalibratedFocalLength','CalibratedOpticalCenterX','CalibratedOpticalCenterY',
#                  'ImageHeight','ImageWidth','NewName','GimbalPitchDegree','GimbalRollDegree','GimbalYawDegree','DewarpData','CalibrationDate','CalibratedFocalLengthX','CalibratedFocalLengthY','CalibratedOpticalCenterX','CalibratedOpticalCenterY',
#                  'K1','K2','P1',"P2","K3",'UtmCode','ImageEasting','ImageNorthing','ImagePolygon','SurveyId']
#         if ('DewarpData' in source_file.columns) and (~source_file['DewarpData'].isna().max()):
#             source_file[['CalibrationDate','CalibratedFocalLengthX','CalibratedFocalLengthY','CalibratedOpticalCenterX','CalibratedOpticalCenterY',
#                          'K1','K2','P1',"P2","K3"]] = source_file['DewarpData'].str.split(r'[;,]',expand=True)
#             source_file[['CalibratedFocalLengthX','CalibratedFocalLengthY','CalibratedOpticalCenterX','CalibratedOpticalCenterY',
#                          'K1','K2','P1',"P2","K3"]] = source_file[['CalibratedFocalLengthX','CalibratedFocalLengthY','CalibratedOpticalCenterX','CalibratedOpticalCenterY',
#                          'K1','K2','P1',"P2","K3"]].astype(float)
#             source_file['CalibratedOpticalCenterX'] = (source_file['ImageWidth']/2)+source_file['CalibratedOpticalCenterX']
#             source_file['CalibratedOpticalCenterY'] = (source_file['ImageHeight']/2)+source_file['CalibratedOpticalCenterY']
#         source_file.loc[source_file['CalibratedFocalLength'].isna(),['CalibratedOpticalCenterX','CalibratedOpticalCenterY','CalibratedFocalLength']]  = [2432,1824,3666.665]

#         # if 'LatitudeMrk' in source_file.columns: 
#         #     source_file['Latitude'] =source_file['LatitudeMrk'].fillna(source_file['Latitude'])
#         #     source_file['Longitude'] =source_file['LongitudeMrk'].fillna(source_file['Longitude'])
#         #     utmproj =Proj(f'epsg:{int(source_file.UtmCode.median())}')            
#         #     source_file['ImageEasting'],source_file['ImageNorthing'] =utmproj(source_file['Longitude'].values,source_file['Latitude'].values)
#         #     # 
#         #     # utmproj =source_file          
#         #     # drone['Easting'],drone['Northing'] =utmproj(drone['Longitude'].values,drone['Latitude'].values)
#         gps =source_file[source_file.columns[source_file.columns.isin(wanted)]].rename(columns={'NewName':'FileName'}).set_index('FileName')
#         gps['GimbalPitchDegree'] = gps['GimbalPitchDegree'] + 90
#         gps['Key'] = gps.index
#         gps['Key'] = gps.Key.apply(lambda x: os.path.splitext(x)[0])
#         #gps =gps[gps.GimbalPitchDegree.abs()<40].apply(get_border,axis=1)
#         gps.to_csv(targets[0])
#         os.path.splitext
#     from turtledrone.config import cfg as config
#     file_dep =  Path(config.get_url('output')).glob('**/*_survey_area_data.csv')
#     for item in file_dep:
#         target = item.parent / 'location.csv'       
#         yield {
#             'name': item,
#             'file_dep':[item],
#             'actions':[process_labelgps],
#             'targets':[target],
#             'clean':True,
#         } 
        
if __name__ == '__main__':
    import doit
    DOIT_CONFIG = {'check_file_uptodate': 'timestamp'}
    #print(globals())
    doit.run(globals())   