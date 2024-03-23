from genericpath import exists
import os
import glob
import shutil
import doit
import glob
import os
import numpy as np
import pandas as pd
from doit import create_after
import numpy as np
import geopandas as gp
import json
import ast
from drone import P4rtk
from sklearn.cluster import MeanShift
import matplotlib.pyplot as plt
import shapely.wkt
import config
import shapely.wkt
from shapely.geometry import MultiPoint
from pathlib import Path
from doit.tools import run_once
from pyproj import Proj 
import cameratransform as ct
from shapely.geometry import point
from shapely.geometry import Polygon

def task_set_up():
    config.read_config()

# # def task_strip_image():
#     def strip_image(file):
#         try:
#             with open(file, "r") as read_file:
#                 data = json.load(read_file)
#             if data.get('imageData'):
#                 data['imageData'] = None
#                 # Serializing json
#                 json_object = json.dumps(data, indent=2)
#                 # Writing to sample.json
#                 with open(file, "w") as outfile:
#                     outfile.write(json_object)
#                 return {'file':file, 'Strip':'True'}
#             return {'file':file, 'Strip':'False'}
#         except:
#             return {'file':file, 'Strip':'Bad'}

#     def process_json(dependencies, targets):
#          output = pd.DataFrame.from_records([strip_image(file) for file in dependencies])
#          output.to_csv(targets[0])

# #     for item in glob.glob(os.path.join(config.geturl('labelmesource')),recursive=True):
# #             file_dep = glob.glob(os.path.join(os.path.dirname(item),'*.json'))
# #             if file_dep:
# #                 target =   os.path.join(os.path.dirname(item),'stripImage.csv')           
# #                 yield {
# #                     'name':item,
# #                     'file_dep': file_dep,
# #                     'actions':[process_json],
# #                     'targets':[target],
# #                     'clean':True,
# #                     'uptodate':[True],
                    
# #                 }


# def task_merge_strip():


#     def process_merge(dependencies, targets):
#          output = pd.concat([pd.read_csv(file) for file in dependencies])
#          output.to_csv(targets[0])
#     os.makedirs(config.geturl('process'),exist_ok=True)
#     file_dep = glob.glob(os.path.join(config.geturl('labelmesource'),"stripImage.csv"),recursive=True)
#     target = os.path.join(config.geturl('process'),"stripImage.csv")
#     return {
#                 'file_dep': file_dep,
#                 'actions':[process_merge],
#                 'targets':[target],
#                 'clean':True,
#                 'uptodate':[True],
                
#             }


def task_process_list_json():
        def process_labelme(dependencies, targets):
            data = pd.DataFrame(dependencies,columns=['JsonSource'])
            data['TimeStamp'] = data.JsonSource.str.extract(r'(?P<TimeStamp>\d{8}T\d{6})')
            data.sort_values('TimeStamp').to_csv(targets[0],index=False)       
            
        file_dep =list(config.geturl('labelsource').rglob('*.json'))
        target =   config.geturl('labelindex')        
        return {
            'file_dep':file_dep,
            'actions':[process_labelme],
            'targets':[target],
            'clean':True,
            'uptodate':[run_once],
        }



@create_after(executed='process_list_json', target_regex='.*\surveyswitharea.csv')   
def task_matchup_labelme():
    def process_labelmatch(dependencies, targets):
        jsonfiles = pd.read_csv(config.geturl('labelindex'),index_col='TimeStamp',parse_dates=['TimeStamp'])
        datafile = pd.read_csv(dependencies[0],index_col='TimeStamp',parse_dates=['TimeStamp'])
        datafile =datafile.join(jsonfiles,rsuffix='_label',how='left')
        datafile['LabelNewName']=datafile.NewName.str.replace('.JPG','.json',regex=False)
        datafile.to_csv(targets[0])


    file_dep =  list(config.geturl('output').rglob('*_survey_area_data.csv'))
    for item in file_dep:
        target = item.parent / item.name.replace('.csv','_labelmerge.csv')       
        yield {
            'name': item,
            'file_dep':[item],
            'actions':[process_labelmatch],
            'targets':[target],
            'clean':True,
        }  




@create_after(executed='matchup_labelme', target_regex='.*\surveyswitharea.csv')   
def task_move_labelme():
    def process_json(dependencies, targets): 
        def update_json(file):
            try:
                with open(file, "r") as read_file:
                    data = json.load(read_file)
                    data['imagePath'] = os.path.basename(file).replace('json','JPG')
                    json_object = json.dumps(data, indent=2)
                    with open(file, "w") as outfile:
                        outfile.write(json_object)
            except:
                print(file)
                
        destination =os.path.dirname(targets[0])
        os.makedirs(destination,exist_ok=True)
        survey = pd.read_csv(dependencies[0])
        if 'JsonSource' in survey.columns:
            survey = survey.loc[~survey.JsonSource.isna()]
            for index,row in survey.iterrows():
                    dest = os.path.join(os.path.dirname(row.FileDest),os.path.basename(row.LabelNewName))
                    source = Path(row.JsonSource)
                    if not source.exists():
                        source =Path(config.geturl('labelmesource')).parent / source.name
                    if not os.path.exists(dest):
                        if config.cfg['outputhardlink']:
                            os.link(source, dest)   
                        else:
                            shutil.copyfile(source,dest) 
                    update_json(dest)
                    
        shutil.copyfile(dependencies[0],targets[0])
            
    file_dep =  list(config.geturl('output').rglob('*_labelmerge.csv'))
    for item in file_dep:
        target = item.parent / item.name.replace('.csv','_move.csv')       
        yield {
            'name': item,
            'file_dep':[item],
            'actions':[process_json],
            'targets':[target],
            'clean':True,
        }   

@create_after(executed='move_labelme', target_regex='.*\surveyswitharea.csv')   
def task_report_labelme():
    def process_report(dependencies, targets): 
        found = list(filter(lambda x: '_labelmerge.csv' in x, dependencies))
        index = pd.read_csv(config.geturl('labelindex'),parse_dates=['TimeStamp'],index_col='TimeStamp')
        data = pd.concat([pd.read_csv(file,parse_dates=['TimeStamp'],index_col='TimeStamp')  for file in found])
        index.join(data[['SurveyId','NewName']]).to_csv(targets[0])
    file_dep =  list(config.geturl('output').rglob('*_labelmerge.csv'))
    file_dep.append(config.geturl('labelindex'))

    target = config.geturl('reports') / 'label_matchup_report.csv'   
    yield {
        'name': target,
        'file_dep':file_dep,
        'actions':[process_report],
        'targets':[target],
        'clean':True,
    }          
      
def task_make_labelgps():
    def process_labelgps(dependencies, targets):
        def get_border(item):
            if 'K1' in item.keys():
                cam = ct.Camera(ct.RectilinearProjection(focallength_x_px=item.CalibratedFocalLengthX,
                                                                     focallength_y_px=item.CalibratedFocalLengthY,
                                                                        center_x_px=item.CalibratedOpticalCenterX,
                                                                        center_y_px=item.CalibratedOpticalCenterY),
                                                                        orientation= ct.SpatialOrientation(tilt_deg=item.GimbalPitchDegree,
                                                                                                        elevation_m=item.RelativeAltitude,
                                                                                                        roll_deg=item.GimbalRollDegree,
                                                                                                        heading_deg=item.GimbalYawDegree),
                                                                        lens=ct.BrownLensDistortion(item.K1,item.K2,item.K3))
            else:
                cam = ct.Camera(ct.RectilinearProjection(focallength_px=item.CalibratedFocalLength,
                                                                    center_x_px=item.CalibratedOpticalCenterX,
                                                                    center_y_px=item.CalibratedOpticalCenterY),
                                                                    orientation= ct.SpatialOrientation(tilt_deg=item.GimbalPitchDegree,
                                                                                                    elevation_m=item.RelativeAltitude,
                                                                                                    roll_deg=item.GimbalRollDegree,
                                                                                                   heading_deg=item.GimbalYawDegree))
            perside=10 
            x = np.linspace(0, item.ImageWidth, num=perside)  
            y = np.linspace(0, item.ImageHeight, num=perside)
            bottom = np.dstack((x,np.ones(perside)*item.ImageHeight-1))[0]
            right = np.dstack((np.ones(perside)*item.ImageWidth-1,y[::-1]))[0]
            top = np.dstack((x[::-1],np.zeros(perside)))[0]
            left = np.dstack((np.zeros(perside),y))[0]
            points = np.vstack((bottom,right,top,left))
            polydata =cam.spaceFromImage(points)
            polydata[:,0] =item.ImageEasting + polydata[:,0] 
            polydata[:,1] =item.ImageNorthing +polydata[:,1]
            item.ImagePolygon=Polygon(polydata[:,0:2])
            return item


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

        if 'LatitudeMrk' in source_file.columns: 
            source_file['Latitude'] =source_file['LatitudeMrk'].fillna(source_file['Latitude'])
            source_file['Longitude'] =source_file['LongitudeMrk'].fillna(source_file['Longitude'])
            utmproj =Proj(f'epsg:{int(source_file.UtmCode.median())}')            
            source_file['ImageEasting'],source_file['ImageNorthing'] =utmproj(source_file['Longitude'].values,source_file['Latitude'].values)
            # 
            # utmproj =source_file          
            # drone['Easting'],drone['Northing'] =utmproj(drone['Longitude'].values,drone['Latitude'].values)
        gps =source_file[source_file.columns[source_file.columns.isin(wanted)]].rename(columns={'NewName':'FileName'}).set_index('FileName')
        gps['GimbalPitchDegree'] = gps['GimbalPitchDegree'] + 90
        gps['Key'] = gps.index
        gps['Key'] = gps.Key.apply(lambda x: os.path.splitext(x)[0])
        gps =gps[gps.GimbalPitchDegree.abs()<40].apply(get_border,axis=1)
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

    #print(globals())
    doit.run(globals())