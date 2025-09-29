import json
from pathlib import Path
import cv2
from sahi.slicing import slice_image
from sahi.utils.file import save_json
from sahi.utils.file import save_json
from sahi.utils.coco import CocoAnnotation
import numpy as np
from turtledrone.utils.yolo import circle_to_bbox
from segment_anything import SamPredictor, sam_model_registry
import torch
import doit 
import turtledrone.config as config
import pandas as pd
from tqdm import tqdm
from doit.tools import run_once
from PIL import Image
import ast
from PIL import ImageDraw
from itertools import chain
import os
import clip
import random
import shutil
import pickle
from turtledrone.utils.yolo import circle_to_bbox


# def task_process_jsonfiles():
#     def loadshapes(file):
#         lines =[]
#         with open(file, "r") as read_file:
#             data = json.load(read_file)
#         data =pd.DataFrame(data['shapes'])
#         data['FilePath'] =file             
#         return(data)
    
#     def calc_area(points):
#         points = np.array(points)
#         x =np.diff(np.sort(points[:,0])[[0,-1]])
#         y =np.diff(np.sort(points[:,1])[[0,-1]])
#         return (x * y)[0]
    
#     def process_labelme(dependencies, targets):
#         data = pd.concat([loadshapes(file) for file in dependencies])
#         data=data.loc[data.label.str.contains('turtle')]
#         data = data.sort_values('FilePath')
#         data['turtle_id'] = range(len(data))
#         data['area'] = data['points'].apply(calc_area)
#         data.to_csv(targets[0])       
        

#     with open('/home/mor582/process.txt','r') as process_dirs:
#         dirs = process_dirs.readlines()
#     for dir in dirs:
#         survey = Path(dir[:-1])
#         file_dep = list(survey.glob('*.json'))
#         if len(file_dep)>0:
#             target =  survey / 'annotations.csv'            
#             yield {
#                 'name': target,
#                 'file_dep' : file_dep,
#                 'actions':[process_labelme],
#                 'targets':[target],
#                 'clean':True,
#                 'uptodate':[False],
                
#             } 




def task_extract_thumbnails():
    def process_extract_turtle_3(dependencies, targets):
        def extact_image(image_file:Path,df:pd.DataFrame):
            image = Image.open(image_file.with_suffix('.JPG'))
            for index,row in df.iterrows():
                patch_path = config.geturl('trainpath') / image_file.parent.name / "raw_thumbs"
                patch_path.mkdir(exist_ok=True,parents=True)
                stamp =image.crop(np.hstack(row.points))
                patch_filename = patch_path / f"{image_file.stem}_{row.label}_{int(row.turtle_id):03}.JPG"
                stamp.save(patch_filename)
        
        detections_df = pd.read_csv(dependencies[0])
        detections_df.points =detections_df.points.apply(ast.literal_eval)
        detections_df = detections_df.loc[detections_df.label=='turtle_3_yolo']
        for filepath,df in detections_df.groupby('FilePath'):
            extact_image(Path(filepath),df)
             
    with open('/home/mor582/process.txt','r') as process_dirs:
        dirs = process_dirs.readlines()
    for dir in dirs:
        survey = Path(dir[:-1])
        file_dep =survey / 'annotations.csv'
        yield{
            'name' : file_dep,
            'actions':[process_extract_turtle_3],#CmdAction(, buffering=1)
            'file_dep':[file_dep],
            'clean': True
        } 


def task_extract_human_thumbnails():
    def process_extract_turtle_3(dependencies, targets):
        def extact_image(image_file:Path,df:pd.DataFrame):
            image = Image.open(image_file.with_suffix('.JPG'))
            for index,row in df.iterrows():
                patch_path = config.geturl('trainpath') / image_file.parent.name / "human_thumbs"
                patch_path.mkdir(exist_ok=True,parents=True)
                try:
                    if row.shape_type == 'circle':
                        points = circle_to_bbox(row.points, image.width,image.height )
                        stamp = image.crop(points)
                    else:
                        points = np.array(row.points)
                        x_min, y_min = points.min(axis=0)
                        x_max, y_max = points.max(axis=0)
                        stamp = image.crop((x_min, y_min, x_max, y_max))
                    patch_filename = patch_path / f"{image_file.stem}_{row.label}_{int(row.turtle_id):03}.JPG"
                    stamp.save(patch_filename)
                except:
                    pass
        
        detections_df = pd.read_csv(dependencies[0])
        detections_df.points =detections_df.points.apply(ast.literal_eval)
        detections_df = detections_df.loc[detections_df.label!='turtle_3_yolo']
        for filepath,df in detections_df.groupby('FilePath'):
            extact_image(Path(filepath),df)
             
    with open('/home/mor582/process.txt','r') as process_dirs:
        dirs = process_dirs.readlines()
    for dir in dirs:
        survey = Path(dir[:-1])
        file_dep =survey / 'annotations.csv'
        yield{
            'name' : file_dep,
            'actions':[process_extract_turtle_3],#CmdAction(, buffering=1)
            'file_dep':[file_dep],
            'clean': True
        } 

def task_apply_clip():

    def classify_images(dependencies, targets,batch_size=32):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        def classify_batch( image_paths):
            """
            Classify a batch of images (list of file paths).
            Returns list of tuples: (image_path, predicted_label, similarities_dict)
            """
            raw_images = [Image.open(p).convert("RGB") for p in image_paths]
            images = [preprocess(img) for img in raw_images]
            batch = torch.stack(images).to(device)

            with torch.no_grad():
                feats = model.encode_image(batch)
                feats /= feats.norm(dim=-1, keepdim=True)
            feats = feats.to(dtype=torch.float32)
            sims = feats @ proto_matrix.T  # (N, C)
            best_indices = sims.argmax(dim=1)

            results = []
            for i, idx in enumerate(best_indices):
                sim_dict = {label_list[j]: sims[i, j].item() for j in range(len(label_list))}
                width, height = raw_images[i].size
                results.append((image_paths[i], label_list[idx], sim_dict, width, height))
            return results
        
        proto_path = config.geturl('output') / 'yolo_finds' / 'clip_reference' / "prototypes.plk"
        with open(proto_path, "rb") as f:
            prototypes = pickle.load(f)
        label_list = list(prototypes.keys())
        proto_matrix = torch.stack([
            torch.tensor(prototypes[label], device=device, dtype=torch.float32)
            for label in label_list
        ])

        # Normalize just in case (they should be normalized already, but no harm)
        proto_matrix /= proto_matrix.norm(dim=1, keepdim=True)


        #prepare the batches
        def chunked(lst, size):
            for i in range(0, len(lst), size):
                yield lst[i:i + size]
        all_results = []
        for batch in chunked(dependencies, batch_size):
            batch_results = classify_batch(batch)
            all_results.extend(batch_results)

        df = pd.DataFrame(all_results,columns=['SourceFile','Label','Scores','Width','Height'])
        df['BestScore'] = df.apply(lambda x: x.Scores[x.Label],axis=1)
        df = pd.concat([df,df['Scores'].apply(pd.Series)],axis=1).drop('Scores',axis=1)
        df.columns = df.columns.str.capitalize()
        df.sort_values(['Label','Bestscore']).to_csv(targets[0],index=False)

    base_path = config.geturl('trainpath')
    file_dep = list(filter(lambda p: 'raw_thumbs' in p.parts,base_path.rglob('*.JPG')))
    df = pd.DataFrame(file_dep,columns=['SourceFile'])
    df['Survey']=df.SourceFile.apply(lambda x: x.parent.parent.name)
    for survey,df in df.groupby('Survey'):
        targets = config.geturl('trainpath') / survey / f"{survey}_thumbs_classified.csv"
        yield {
            'name' : targets,
            'file_dep' : df.SourceFile.to_list(),
            'targets' : [targets],
            'actions':[classify_images],
            'clean':True
            
        } 

def task_file_clip():
 
    def process_file_clip(dependencies, targets):
        data = pd.read_csv(dependencies[0])
        for index,row in data.iterrows():
            if (row.Width<110) and (row.Height <110):
                source = Path(row.Sourcefile)            
                output =source.parent.parent / 'sorted_thumbs'
                output.mkdir(exist_ok=True,parents=True)
                Image_dest = output / row.Label / f'{int(row.Bestscore*1000):03d}-{source.name}'
                Image_dest.parent.mkdir(exist_ok=True,parents=True)
                if not Image_dest.exists():                                                                                                                                                                                                                                                                                                                                                      
                    os.link(source.absolute(),Image_dest) 
                
    file_dep = config.geturl('trainpath').rglob("*_thumbs_classified.csv")
    for file in file_dep:
        yield {
            'name' : file,
            'file_dep' : [file],
            'actions':[process_file_clip],
            'clean':True,                                                                                                                                                                                
            'uptodate':[True],
            
        } 

def task_count_turtles():


    with open('/home/mor582/process.txt','r') as process_dirs:
        dirs = process_dirs.readlines()
    for dir in dirs:
        survey = Path(dir[:-1])
        file_dep = list(survey.glob('*.json'))
        if len(file_dep)>0:
            target =  survey / 'annotations.csv'            
            yield {
                'name': target,
                'file_dep' : file_dep,
                'actions':[process_count],
                'targets':[target],
                'clean':True,
                'uptodate':[False],
                
            } 

if __name__ == '__main__':
    import doit
    DOIT_CONFIG = {'check_file_uptodate': 'timestamp'}
    #print(globals())
    doit.run(globals())  