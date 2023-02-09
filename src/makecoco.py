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
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from sklearn.cluster import MeanShift
import matplotlib.pyplot as plt
import shapely.wkt
import plotly
import plotly.express as px
from PIL import Image

# def task_create_json():
#         config = {"config": get_var('config', 'NO')}
#         with open(config['config'], 'r') as ymlfile:
#             cfg = yaml.load(ymlfile, yaml.SafeLoader)
#         basepath = os.path.dirname(config['config'])
#         exifpath = os.path.join(basepath,cfg['paths']['exiftool'])
#         for item in glob.glob(os.path.join(basepath,cfg['paths']['imagesource']),recursive=True):
#             if glob.glob(os.path.join(item,cfg['paths']['imagewild'].upper())) or glob.glob(cfg['paths']['imagewild'].lower()):
#                 target  = os.path.join(item,'exif.json')
#                 filter = os.path.join(item,cfg['paths']['imagewild'])
#                 file_dep = glob.glob(filter)
#                 if file_dep:
#                     yield { 
#                         'name':item,
#                         'actions':[f'exiftool -json {filter} > {target}'],
#                         'targets':[target],
#                         'uptodate':[True],
# #                        'uptodate': [check_timestamp_unchanged(file_dep, 'ctime')],
#                         'clean':True,
#                     }

def loadshapes(file):
    lines =[]
    try:
        #data=pd.read_json(file)
        with open(file, "r") as read_file:
            while read_file:
                line =read_file.readline()
                if (line ==''):
                    break
                if (line.startswith('  "imagePath')) :
                    lines.append(line[:-2]+'}')
                    break
                lines.append(line)
        if lines:            
            data = json.loads(''.join(lines).replace("\n", "").replace("'", '"').replace('u"', '"').replace('null', '-1'))
            if 'shapes' in data:
                data =pd.DataFrame(data['shapes'])
            else:
                data =pd.DataFrame(data)
            data['FilePath'] =file   
        else:
            data=pd.DataFrame(columns=['label', 'points', 'group_id', 'shape_type', 'flags'])          
    except:
        print(file)
    return(data)

def task_list_jason():
    def process_json(dependencies, targets):
        data =pd.concat([loadshapes(json) for json in dependencies])
        data[data['label']!='done'].to_csv(targets[0],index=False)
        
    config = {"config": get_var('config', 'NO')}
    with open(config['config'], 'r') as ymlfile:
        cfg = yaml.load(ymlfile, yaml.SafeLoader)
    basepath = os.path.dirname(config['config'])
    dest = os.path.join(cfg['paths']['output'],'AU','**','*.json')
    file_dep = glob.glob(dest,recursive=True)
    file_dep.sort()
    target = os.path.join(basepath,cfg['paths']['process'],'coco.csv')        
    return {
        'actions':[process_json],
        'file_dep':file_dep,
        'targets':[target],
        'clean':True,
    }  

def task_make_coco():
    def process_coco(dependencies, targets):
        def calc_bound(x):
            if len(x)==2:
                r =np.sqrt((x[0][0] -x[1][0])**2 + (x[0][1] -x[1][1])**2)
                box = np.hstack((np.array(np.ceil(x[0]-r)).astype(int),[r*2,r*2])).astype(int)
            else:
                box = x
            return(box)

        data =pd.read_csv(dependencies[0])
        keep =['turtle_surface', 'ray', 'turtle_deep', 'shark', 'turtle_jbs',
               'potential_turtle', 'turtle_diving', 'fish_school', 'weed',
               'mammal', 'dolphin', 'turtle_tracks','whale']

        output ={"categories": [{"supercategory": "Turtle","id": 1,"name": "turtle_surface"},
                                {"supercategory": "Turtle","id": 2,"name": "turtle_deep"},
                                {"supercategory": "Turtle","id": 3,"name": "potential_turtle"},
                                {"supercategory": "Turtle","id": 4,"name": "turtle_diving"},
                                {"supercategory": "Turtle","id": 5,"name": "turtle_jbs"},
                                {"supercategory": "Beach","id": 6,"name": "turtle_tracks"},
                                {"supercategory": "Fish","id": 7,"name": "fish_school"},
                                {"supercategory": "Fish","id": 8,"name": "ray"},
                                {"supercategory": "Fish","id": 9,"name": "shark"},
                                {"supercategory": "mammal","id": 10,"name": "whale"},
                                {"supercategory": "mammal","id": 11,"name": "mammal"},
                                {"supercategory": "mammal","id": 12,"name": "dolphin"},
                                {"supercategory": "weed","id": 13,"name": "weed"}]}
        
        data = data[data.label.isin(keep)]
        data.points =data.points.apply(ast.literal_eval).apply(calc_bound)
        images = pd.DataFrame(data.FilePath.unique(),columns=['FilePath'])
        images['ImageFile'] =images.FilePath.str.replace('json','JPG')
        images['Id'] = range(0,len(images))
        pd.merge(data,images,on='FilePath')
        data[data['label']!='done'].to_csv(targets[0],index=False)
        
    config = {"config": get_var('config', 'NO')}
    with open(config['config'], 'r') as ymlfile:
        cfg = yaml.load(ymlfile, yaml.SafeLoader)
    basepath = os.path.dirname(config['config'])
    dest = os.path.join(cfg['paths']['output'],'AU','**','*.json')

    file_dep = os.path.join(basepath,cfg['paths']['process'],'coco.csv')   
    target = os.path.join(basepath,cfg['paths']['process'],'coco.json')     
    return {
        'actions':[process_coco],
        'file_dep':[file_dep],
        'targets':[target],
        'clean':True,
    }  


# def task_make_yolo_training_images():
#     def process_yoloset(dependencies, targets):
#         os.makedirs(os.path.dirname(targets[0]),exist_ok=True)
#         imagedir = os.path.join(os.path.dirname(targets[0]),'images')
#         os.makedirs(imagedir,exist_ok=True)
#         sourcefile = pd.read_csv(dependencies[0])
#         good =sourcefile[~sourcefile.label.isin(['done','don,e','gcp'])]
#         good.points = good.points.apply(ast.literal_eval)
#         classes = list(good.label.unique())
#         classes.sort()
#         images =good.groupby('FilePath')
#         for file,data in images:
#             Imgdest = os.path.join(imagedir,os.path.splitext(os.path.basename(file))[0]+'.JPG')
#             if not os.path.exists(Imgdest):
#                 shutil.copy(os.path.splitext(file)[0]+'.JPG',
#                             os.path.join(imagedir,os.path.splitext(os.path.basename(file))[0]+'.JPG'))
#             jsondest = os.path.join(imagedir,os.path.splitext(os.path.basename(file))[0]+'.json')
#             if not os.path.exists(jsondest):
#                 shapes =loadshapes(file)
#                 shapes.drop(['FilePath','flags','group_id'],axis=1)[~shapes.label.isin(['done','don,e','gcp'])].to_json(jsondest)
#             txtdest = os.path.join(imagedir,os.path.splitext(os.path.basename(file))[0]+'.TXT')
#             if not os.path.exists(txtdest):
#                 with open(txtdest,'w') as datafile:
#                     for index,row in data.iterrows():
#                         cla = classes.index(row.label)
#                         points = row.points
#                         if len(points)==2: #one animal
#                             datafile.write(f'{classes.index(row.label)} {points[0][0]/row.ImageWidth} {points[0][1]/row.ImageHeight} '\
#                                             f'{2*abs(points[0][0]-points[1][0])/row.ImageWidth}' \
#                                             f'{2*abs(points[0][1]-points[1][1])/row.ImageHeight}\n')
                    
                 
                 
#         with open(targets[0],'w') as obnames:
#             obnames.writelines(map(lambda x:x+'\n', classes))
        
#     config = {"config": get_var('config', 'NO')}
#     with open(config['config'], 'r') as ymlfile:
#         cfg = yaml.load(ymlfile, yaml.SafeLoader)
#     basepath = os.path.dirname(config['config'])
#     file_dep = os.path.join(basepath,cfg['paths']['process'],'labelmematchup.csv')
#     target = os.path.join(basepath,cfg['paths']['output'],'yolo','data','obj.names')        
#     return {
#         'actions':[process_yoloset],
#         'file_dep':[file_dep],
#         'targets':[target],
#         'clean':True,
#     }   

if __name__ == '__main__':
    import doit
    DOIT_CONFIG = {'check_file_uptodate': 'timestamp'}
    #print(globals())
    doit.run(globals())