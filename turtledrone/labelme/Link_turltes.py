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
import json
from pathlib import Path
from doit.tools import run_once
from pyproj import Proj 
import cameratransform as ct6
from shapely.geometry import Polygon
from tqdm import tqdm
import turtledrone.config as config
from turtledrone.utils.yolo import circle_to_bbox
from turtledrone.utils.yolo import parse_json
from turtledrone.utils.yolo import  load_json
from turtledrone.utils.yolo import  find_label
from turtledrone.utils.yolo import  save_json
import ast



    
def task_process_turtles():
    def process_json(dependencies, targets):
        detections =[]
        for json_file in tqdm(dependencies, desc="Processing JPG/JSON files and creating dataframe"):
                for label, points, shape_type, imagefile,img_width,img_height in parse_json(json_file):
                    if 'turtle' in label:
                        if (shape_type == 'circle') & (len(points)==2):
                            bbox = circle_to_bbox(points, img_width,img_height )
                        elif (shape_type == 'polygon') & (len(points)==2):
                            # Assuming points for polygons are given as [[x_min, y_min], [x_max, y_max]]

                            bbox = [int(points[0][0]), int(points[0][1]), int(points[1][0]), int(points[1][1])]
                        else:
                            bbox = None
                        detections.append({'JsonFile':json_file, 'Label':label, 'Points':points, 'BoundingBox':bbox,'ImageFile':imagefile, 'Key':imagefile.stem})
        data = pd.DataFrame(detections)
        data[['Area','DateStamp']]=data.Key.str.split('_',expand=True)[[3,4]]
        data.sort_values('Key').to_csv(targets[0],index=False)

    file_dep =list((config.geturl('output') / config.cfg['country']).rglob('*.json'))
    target = config.geturl('output') / 'turtles' / 'turtle_list.csv'
    return {
        'actions':[process_json],
        'file_dep':file_dep,
        'targets':[target],
        'clean': True,
    } 

def task_file_turtles():
    def file_turtles(dependencies, targets):
        data = pd.read_csv(dependencies[0])
        data = data.drop_duplicates(subset='JsonFile')
        data = data.drop_duplicates(subset='ImageFile')
        dest_path = Path(dependencies[0]).parent
        for index,row in data.iterrows():
            Image_file =Path(row.ImageFile)
            Json_file = Path(row.JsonFile)
            Image_dest = dest_path / Image_file.name
            Json_dest = dest_path / Json_file.name
            if not Image_dest.exists():
                os.link(Image_file.absolute(),Image_dest)
            if not Json_dest.exists():
                os.link(Json_file.absolute(),dest_path / Json_file.name)



    file_dep = config.geturl('output') / 'turtles' / 'turtle_list.csv'
    return {
        'actions':[file_turtles],
        'file_dep':[file_dep],
        'uptodate':[run_once],
        'clean': True,
    } 

def task_add_box_turtles():
    def file_turtles(dependencies, targets):
        data = pd.read_csv(dependencies[0])
        data.Points =data.Points.apply(ast.literal_eval)
        for name,group in data.groupby('JsonFile'):
            json_data = load_json(name)
            if not find_label(json_data['shapes'],'yolo'):
                yolo=group.apply(lambda x:{"label":'turtle_yolo',"points":x.Points,"group_id": None,"shape_type": "rectangle","flags": {}},axis=1).tolist()
                json_data['shapes']=json_data['shapes']+yolo
                save_json(name,json_data)
            pass
            


    file_dep = config.geturl('output') / 'turtles' / 'turtle_list.csv'
    return {
        'actions':[file_turtles],
        'file_dep':[file_dep],
        'uptodate':[run_once],
        'clean': True,
    } 



if __name__ == '__main__':
    import doit
    DOIT_CONFIG = {'check_file_uptodate': 'timestamp'}
    #print(globals())
    doit.run(globals())  