import os
import glob
import doit
import glob
import os
import pandas as pd
import turtledrone.config as config
import cameratransform as ct
from pathlib import Path
import numpy as np
from shapely import Polygon

# def task_make_geo():
#     """
#     Make geo file use by opendrone to enhance locations
#     """
#     def make_geo(dependencies, targets):
#         data = pd.read_csv(dependencies[0])
#         text = list(data.apply(lambda x:f'{x.NewName}   {x.LongitudeMrk}   {x.LatitudeMrk}    {x.EllipsoideHightMrk} {x.CameraYaw} {x.CameraPitch} {x.CameraRoll}\n',axis=1))
#         if 'LatitudeMrk' in data.columns:
#             with open(targets[0], 'w') as f:
#                 f.write('EPSG:4326\n')
#                 f.writelines(text)
        
    
#     file_dep = glob.glob(os.path.join(config.geturl('output'),'**','*_survey_area_data.csv'),recursive=True)
#     for file in file_dep:
#         target = os.path.join(config.getdest(os.path.basename(file)),'geo.txt')
#         yield {
#             'name':target,
#             'actions':[make_geo],
#             'file_dep':[file],
#             'targets':[target],
#             'uptodate': [True],
#             'clean':True,
#         }                                
            

def task_make_labelgps():
    """
    make a location file to for use with modified labelme
    """
    def process_labelgps(dependencies, targets):
    #     def get_border(item):
    #         if 'K1' in item.keys():
    #             cam = ct.Camera(ct.RectilinearProjection(focallength_x_px=item.CalibratedFocalLengthX,
    #                                                                  focallength_y_px=item.CalibratedFocalLengthY,
    #                                                                     center_x_px=item.CalibratedOpticalCenterX,
    #                                                                     center_y_px=item.CalibratedOpticalCenterY),
    #                                                                     orientation= ct.SpatialOrientation(tilt_deg=item.GimbalPitchDegree,
    #                                                                                                     elevation_m=item.RelativeAltitude,
    #                                                                                                     roll_deg=item.GimbalRollDegree,
    #                                                                                                     heading_deg=item.GimbalYawDegree),
    #                                                                     lens=ct.BrownLensDistortion(item.K1,item.K2,item.K3))
    #         else:
    #             cam = ct.Camera(ct.RectilinearProjection(focallength_px=item.CalibratedFocalLength,
    #                                                                 center_x_px=item.CalibratedOpticalCenterX,
    #                                                                 center_y_px=item.CalibratedOpticalCenterY),
    #                                                                 orientation= ct.SpatialOrientation(tilt_deg=item.GimbalPitchDegree,
    #                                                                                                 elevation_m=item.RelativeAltitude,
    #                                                                                                 roll_deg=item.GimbalRollDegree,
    #                                                                                                heading_deg=item.GimbalYawDegree))
    #         perside=10 
    #         x = np.linspace(0, item.ImageWidth, num=perside)  
    #         y = np.linspace(0, item.ImageHeight, num=perside)
    #         bottom = np.dstack((x,np.ones(perside)*item.ImageHeight-1))[0]
    #         right = np.dstack((np.ones(perside)*item.ImageWidth-1,y[::-1]))[0]
    #         top = np.dstack((x[::-1],np.zeros(perside)))[0]
    #         left = np.dstack((np.zeros(perside),y))[0]
    #         points = np.vstack((bottom,right,top,left))
    #         polydata =cam.spaceFromImage(points)
    #         polydata[:,0] =item.ImageEasting + polydata[:,0] 
    #         polydata[:,1] =item.ImageNorthing +polydata[:,1]
    #         item.ImagePolygon=Polygon(polydata[:,0:2])
    #         return item


        datafile = glob.glob(os.path.join(os.path.dirname(dependencies[0]),'*_survey_area_data.csv'))
        if len(datafile)==0:
            datafile = glob.glob(os.path.join(os.path.dirname(dependencies[0]),'*_survey_area.csv'))
        if len(datafile)==0:
            datafile = glob.glob(os.path.join(os.path.dirname(dependencies[0]),'*_survey_data.csv'))
        
        source_file = pd.read_csv(datafile[0])
        wanted =['TimeStamp','Longitude','Latitude','AbsoluteAltitude','RelativeAltitude','CalibratedFocalLength','CalibratedOpticalCenterX','CalibratedOpticalCenterY',
                 'ImageHeight','ImageWidth','NewName','GimbalPitchDegree','GimbalRollDegree','GimbalYawDegree','DewarpData','CalibrationDate','CalibratedFocalLengthX','CalibratedFocalLengthY','CalibratedOpticalCenterX','CalibratedOpticalCenterY',
                 'K1','K2','P1',"P2","K3",'UtmCode','ImageEasting','ImageNorthing','ImagePolygon','SurveyId']
        if ('DewarpData' in source_file.columns) and (~source_file['DewarpData'].isna().max()):
            source_file[['CalibrationDate','CalibratedFocalLengthX','CalibratedFocalLengthY','CalibratedOpticalCenterX','CalibratedOpticalCenterY',
                         'K1','K2','P1',"P2","K3"]] = source_file['DewarpData'].str.split(r'[;,]',expand=True)
            source_file[['CalibratedFocalLengthX','CalibratedFocalLengthY','CalibratedOpticalCenterX','CalibratedOpticalCenterY',
                         'K1','K2','P1',"P2","K3"]] = source_file[['CalibratedFocalLengthX','CalibratedFocalLengthY','CalibratedOpticalCenterX','CalibratedOpticalCenterY',
                         'K1','K2','P1',"P2","K3"]].astype(float)
            source_file['CalibratedOpticalCenterX'] = (source_file['ImageWidth']/2)+source_file['CalibratedOpticalCenterX']
            source_file['CalibratedOpticalCenterY'] = (source_file['ImageHeight']/2)+source_file['CalibratedOpticalCenterY']
        source_file.loc[source_file['CalibratedFocalLength'].isna(),['CalibratedOpticalCenterX','CalibratedOpticalCenterY','CalibratedFocalLength']]  = [2432,1824,3666.665]

        # if 'LatitudeMrk' in source_file.columns: 
        #     source_file['Latitude'] =source_file['LatitudeMrk'].fillna(source_file['Latitude'])
        #     source_file['Longitude'] =source_file['LongitudeMrk'].fillna(source_file['Longitude'])
        #     utmproj =Proj(f'epsg:{int(source_file.UtmCode.median())}')            
        #     source_file['ImageEasting'],source_file['ImageNorthing'] =utmproj(source_file['Longitude'].values,source_file['Latitude'].values)
        #     # 
        #     # utmproj =source_file          
        #     # drone['Easting'],drone['Northing'] =utmproj(drone['Longitude'].values,drone['Latitude'].values)
        gps =source_file[source_file.columns[source_file.columns.isin(wanted)]].rename(columns={'NewName':'FileName'}).set_index('FileName')
        gps['GimbalPitchDegree'] = gps['GimbalPitchDegree'] + 90
        gps['Key'] = gps.index
        gps['Key'] = gps.Key.apply(lambda x: os.path.splitext(x)[0])
        #gps =gps[gps.GimbalPitchDegree.abs()<40].apply(get_border,axis=1)
        gps.to_csv(targets[0])
        os.path.splitext

    file_dep =  Path(config.geturl('output')).glob('**/*_survey_area_data.csv')
    for item in file_dep:
        target = item.parent / 'location.csv'       
        yield {
            'name': item,
            'file_dep':[item],
            'actions':[process_labelgps],
            'targets':[target],
            'clean':True,
        } 
        
if __name__ == '__main__':
    import doit
    DOIT_CONFIG = {'check_file_uptodate': 'timestamp'}
    #print(globals())
    doit.run(globals())   