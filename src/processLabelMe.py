from genericpath import exists
import os
import glob
import shutil
import doit
import glob
import os
import numpy as np
from pandas.core.arrays.integer import Int64Dtype
from pandas.io.parsers import read_csv
import yaml
import pandas as pd
from doit import get_var
from doit.tools import run_once
from doit import create_after
import numpy as np
import plotly
import plotly.express as px
import geopandas as gp
import json
import ast
from pyproj import Proj
from drone import P4rtk

# def task_process_list_json():
#         def process_labelme(dependencies, targets):
#             files =glob.glob(os.path.join(basepath,cfg['paths']['labelmesource']),recursive=True)
#             data = pd.DataFrame(dependencies,names='FilePaths')
#             data.to_csv(targets[0],index=False)       
            
#         config = {"config": get_var('config', 'NO')}
#         with open(config['config'], 'r') as ymlfile:
#             cfg = yaml.load(ymlfile, yaml.SafeLoader)
#         basepath = os.path.dirname(config['config'])

#         target =   os.path.join(basepath,cfg['paths']['process'],'jsonfiles.csv')        
#         return {
#             'actions':[process_labelme],
#             'targets':[target],
#             'clean':True,
#             'uptodate':[run_once],
            
#         }


def task_process_labelme():
        def loadshapes(file):
            lines =[]
            with open(file, "r") as read_file:
                while read_file:
                    line =read_file.readline()
                    if "imagePath" in line:
                        break
                    lines.append(line)
            lines[-1] ='  ]}\n'
            data = json.loads(''.join(lines).replace("\n", "").replace("'", '"').replace('u"', '"'))
            data =pd.DataFrame(data['shapes'])
            data['FilePath'] =file             
            return(data)
        def process_labelme(dependencies, targets):
            jsonfiles = glob.glob(os.path.join(os.path.dirname(targets[0]),'*.json'))
            if jsonfiles:
                data = pd.concat([loadshapes(file) for file in jsonfiles])
            else:
                data = pd.DataFrame()
            data.to_csv(targets[0],index=False)       
            
        config = {"config": get_var('config', 'NO')}
        with open(config['config'], 'r') as ymlfile:
            cfg = yaml.load(ymlfile, yaml.SafeLoader)
        basepath = os.path.dirname(config['config'])
        for item in glob.glob(os.path.join(basepath,cfg['paths']['labelmesource'])):
            #file_dep = glob.glob(os.path.join(os.path.dirname(item),'*.json'))
            target =   os.path.join(os.path.dirname(item),'labelme.csv')           
            yield {
                'name':item,
                'actions':[process_labelme],
                'targets':[target],
                'clean':True,
                'uptodate':[True],
                
            }
        
        
def task_process_mergelabel():
        def process_mergelabel(dependencies, targets):
            files = list(filter(lambda x:os.path.getsize(x)>10, dependencies))
            data = pd.concat([pd.read_csv(file) for file in files])
            data.to_csv(targets[0],index=False)       
            
        config = {"config": get_var('config', 'NO')}
        with open(config['config'], 'r') as ymlfile:
            cfg = yaml.load(ymlfile, yaml.SafeLoader)
        basepath = os.path.dirname(config['config'])
        file_dep = glob.glob(os.path.join(basepath,cfg['paths']['labelmesource'],'labelme.csv'),recursive=True)
        target = os.path.join(basepath,cfg['paths']['process'],'mergelabelme.csv')        
        return {
            'actions':[process_mergelabel],
            'file_dep':file_dep,
            'targets':[target],
            'clean':True,
        }
        
def task_process_matchup():
    def process_labelmatch(dependencies, targets):
        source_file = pd.concat([pd.read_csv(file) for file in filter(lambda x: '_survey.csv' in x, dependencies)])
        lableme = pd.concat([pd.read_csv(file) for file in filter(lambda x: 'mergelabelme' in x, dependencies)])
        source_file.index = source_file.NewName.apply(lambda x:os.path.splitext(x)[0])
        lableme.index = lableme.FilePath.apply(lambda x: os.path.splitext(os.path.basename(x))[0])
        output =lableme.join(source_file)
        output.index.name='Key'
        output.to_csv(targets[0])
    config = {"config": get_var('config', 'NO')}
    with open(config['config'], 'r') as ymlfile:
        cfg = yaml.load(ymlfile, yaml.SafeLoader)
    basepath = os.path.dirname(config['config'])
    file_dep =  glob.glob(os.path.join(basepath,cfg['paths']['process'],'*_survey.csv'),recursive=True)
    file_dep.append(os.path.join(basepath,cfg['paths']['process'],'mergelabelme.csv'))
    
    
    target = os.path.join(basepath,cfg['paths']['process'],'labelmematchup.csv')        
    return {
        'actions':[process_labelmatch],
        'file_dep':file_dep,
        'targets':[target],
        'clean':True,
    }   
    
# def task_process_movefiles():
#     def process_move(dependencies, targets,outputpath):
#         files = pd.read_csv(dependencies[0])
#         files['Destination'] = ''
#         for index,row in files.iterrows():
#             row['Destination'] = os.path.join(outputpath,row.SurveyId,os.path.splitext(row.NewName)[0])+ \
#                 os.path.splitext(row.FilePath)[1] 
#             if os.path.exists(row.FilePath): 
#                 shutil.move(row.FilePath,row.Destination)
#         files.to_csv(targets[0])
#     config = {"config": get_var('config', 'NO')}
#     with open(config['config'], 'r') as ymlfile:
#         cfg = yaml.load(ymlfile, yaml.SafeLoader)
#     basepath = os.path.dirname(config['config'])
#     file_dep = os.path.join(basepath,cfg['paths']['process'],'labelmematchup.csv')
#     target =os.path.join(basepath,cfg['paths']['process'],'labelmematchup_final.csv')
#     return {
#         'actions':[(process_move, [],
#                     {'outputpath':os.path.join(cfg['paths']['output'],cfg['survey']['country'])})],
#         'file_dep':[file_dep],
#         'targets':[target],
#         'clean':True,
#     }        

def task_calculate_positions():  

    def process_positions(dependencies, targets):
            def calcRealworld(item):
                #-14.772 hieght at Exmouth
                localdrone = P4rtk(data)
                localdrone.setdronepos(item.Eastingrtk,item.Northingrtk,item.EllipsoideHight+14.772+1.5,
                                  (90+item.GimbalPitchDegree)*-1,item.GimbalRollDegree,item.GimbalYawDegree+10) #item.GimbalPitchDegree-90,item.GimbalRollDegree,-item.GimbalYawDegree
                pos=localdrone.cameratorealworld(item.DewarpX,item.DewarpY)
                item.EastingPntD = pos[0]
                item.NorthingPntD = pos[1]
                pos=localdrone.cameratorealworld(item.JpegX,item.JpegY)
                #pos=localdrone.cameratorealworld(item.DewarpX,item.DewarpY)
                item.EastingPntJ = pos[0]
                item.NorthingPntJ = pos[1]
                return item           

            drone = pd.read_csv(dependencies[0])
            #drone=drone[~drone.label.isin(['done','don,e'])]
            drone= drone[drone.label.isin(['turtle_surface','turtle_jbs','gcp'])].copy()
            drone.points = drone.points.apply(ast.literal_eval)
            data = np.array([3706.080000000000,3692.930000000000,-34.370000000000,-34.720000000000,-0.271104000000,0.116514000000,0.001092580000,0.000348025000,-0.040583200000])
            p4rtk = P4rtk(data)
            #drone[['PointEasting','PontNorthing']]=drone.apply(process_row,axis=1,result_type='expand')
            jpegpoints = np.array(drone.points.apply(lambda x:x[0]).tolist())
            drone[['JpegX','JpegY']] =np.floor(jpegpoints)
            corrected =p4rtk.jpegtoreal(jpegpoints)
            drone[['DewarpX','DewarpY']] =corrected
            drone[['EastingPntJ','NorthingPntJ','EastingPntD','NorthingPntD']] = 0.
            drone['EllipsoideHight']= pd.to_numeric(drone['EllipsoideHight'].str.split(',',expand=True)[0])
            drone = drone.apply(calcRealworld,axis=1)
            drone.to_csv(targets[0],index=False) 

    config = {"config": get_var('config', 'NO')}
    with open(config['config'], 'r') as ymlfile:
        cfg = yaml.load(ymlfile, yaml.SafeLoader)
    basepath = os.path.dirname(config['config'])
    file_dep =os.path.join(basepath,cfg['paths']['process'],'labelmematchup.csv')
    target =os.path.join(basepath,cfg['paths']['process'],'labelmematchup_final_pointpos.csv')
    return {
        'actions':[process_positions],
        'file_dep':[file_dep],
        'targets':[target],
        'clean':True,
    }   



if __name__ == '__main__':
    import doit

    #print(globals())
    doit.run(globals())