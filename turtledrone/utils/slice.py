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






def task_set_up():
    config.read_config()


# #@create_after(executed='move_labelme')   
# def task_process_labelme():
#     def loadshapes(file):
#         print(file)
#         lines =[]
#         with open(file, "r") as read_file:
#             data = json.load(read_file)

#         #data = json.loads(''.join(lines).replace("\n", "").replace("'", '"').replace('u"', '"'))
#         data =pd.DataFrame(data['shapes'])
#         data['FilePath'] =file             
#         return(data)
 
#     def process_labelme(dependencies, targets):
#         data = pd.concat([loadshapes(file) for file in dependencies])
#         data.to_csv(targets[0],index=False)       
        

#     file_dep = list((config.geturl('output') / 'turtles').glob('*.json'))
#     target =   config.geturl('output') / 'processing' /'shapes'           
#     return {
#         'file_dep' : file_dep,
#         'actions':[process_labelme],
#         'targets':[target],
#         'clean':True,
#         'uptodate':[True],
        
#     }    
# def task_slice_turtles():
#     def file_turtles(dependencies, targets):
#         data = pd.read_csv(dependencies[0])
#         data = data.drop_duplicates(subset='JsonFile')
#         data = data.drop_duplicates(subset='ImageFile')
#         dest_path = Path(dependencies[0]).parent
#         for index,row in data.iterrows():
#             Image_file =Path(row.ImageFile)
#             Json_file = Path(row.JsonFile)
#             Image_dest = dest_path / Image_file.name
#             Json_dest = dest_path / Json_file.name
#             if not Image_dest.exists():
#                 os.link(Image_file.absolute(),Image_dest)
#             if not Json_dest.exists():
#                 os.link(Json_file.absolute(),dest_path / Json_file.name)


#     source_files = (config.geturl('output') / 'turtles' / 'turtle_list.csv').glob("*.JPG")  # Change this
#     target = config.geturl('output') / 'turtles' / 'turtle_list.csv'
#     return {
#         'actions':[file_turtles],
#         'file_dep':[file_dep],
#         'uptodate':[run_once],
#         'clean': True,
#     } 




#  # Corresponding LabelMe JSON file


# def task_slice_images():
#     def process_image(input_image_path: Path,output_dir,id: Path):
#         # Load LabelMe JSON
#         output_dir.mkdir(exist_ok=True,parents=True)
#         try:
#             labelme_json_path = input_image_path.with_suffix(".json") 
#             if labelme_json_path.exists():
#                 with open(labelme_json_path, "r") as f:
#                     labelme_data = json.load(f)

#                 # Convert LabelMe annotations to SAHI CocoAnnotation format
#                 coco_annotations = []
#                 category_id=0
#                 cats= dict()
#                 for shape in labelme_data["shapes"]:
#                     if ('turtle_3' in shape["label"]) and not ('tracks' in shape["label"]) and ('yolo' in shape["label"]) and (np.array(shape["points"]).shape == (2,2)) :
#                         if not shape["label"] in cats.keys():
#                             cats[shape["label"]] = category_id
#                             category_id=category_id +1
#                         if shape['shape_type']=='circle':
#                             points =circle_to_bbox(shape["points"], labelme_data['imageWidth'],labelme_data['imageHeight'])   
#                             x_min = points[0]
#                             y_min = points[1]
#                             x_max = points[2]
#                             y_max = points[3]                 
#                         else:
#                             points = shape["points"]
#                             x_min = min(p[0] for p in points)
#                             y_min = min(p[1] for p in points)
#                             x_max = max(p[0] for p in points)
#                             y_max = max(p[1] for p in points)

#                         category_name = shape["label"]
#                         coco_annotation = CocoAnnotation(
#                             bbox=[x_min, y_min, x_max - x_min, y_max - y_min],  # (x, y, width, height)
#                             category_name=category_name,
#                             category_id =id
                            
#                         )
#                         id = id +1
#                         coco_annotations.append(coco_annotation)
#                 labels = dict()
#                 coco_annotations = list(filter(lambda ann:len(ann.bbox)==4,coco_annotations))
#                 for key in cats.keys():
#                     labels[key] = cats[key]
#                 if len(coco_annotations):
#                     # Tile the image while remapping objects
#                     slices = slice_image(
#                         image=str(input_image_path.with_suffix('.JPG')),
#                         output_file_name = input_image_path.stem,
#                         coco_annotation_list=coco_annotations,  # Now passing valid SAHI annotations
#                         slice_height=640,  # Adjustable tile size
#                         slice_width=640,
#                         overlap_height_ratio=0.2,
#                         overlap_width_ratio=0.2
#                     )
#                     slices = list(filter(lambda s:len(s['coco_image'].annotations),slices))
#                     annotation_idx = dict()
#                     for s in slices:
#                         for ann in s['coco_image'].annotations:
#                             if ann.category_id in annotation_idx.keys():

#                                 if ann.area > annotation_idx[ann.category_id].area:
#                                     annotation_idx[ann.category_id] = ann
#                             else:
#                                 annotation_idx[ann.category_id] = ann
                                

#                     used_id =[]
#                     for i, sliced_img in enumerate(slices):
#                         if len(sliced_img['coco_image'].annotations):  # Only save tiles with objects
#                             tile_path = output_dir / f"{sliced_img['filename']}"
#                             new_label=False
#                             if not tile_path.exists():
#                                 for ann in sliced_img['coco_image'].annotations:
#                                     if ann.category_id not in used_id:
#                                         if annotation_idx[ann.category_id].area==ann.area:
#                                             new_label=True
#                                             used_id.append(ann.category_id)
#                                 if new_label and not tile_path.exists():
#                                     cv2.imwrite(str(tile_path),cv2.cvtColor(sliced_img['image'], cv2.COLOR_RGB2BGR))

#                                     # Convert SAHI's annotations back to LabelMe format
#                                     new_labelme_json = {
#                                         "imagePath": str(tile_path.name), 
#                                         "imageHeight": sliced_img['image'].shape[0],
#                                         "imageWidth": sliced_img['image'].shape[1],
#                                         "version": "5.1.1",
#                                         "flags": {},
#                                         "imageData": None,
#                                         "shapes": [
#                                             {
#                                                 "label": ann.category_name,
#                                                 "points": [
#                                                     [ann.bbox[0], ann.bbox[1]],  # Top-left
#                                                     [ann.bbox[0] + ann.bbox[2], ann.bbox[1] + ann.bbox[3]]  # Bottom-right
#                                                 ],
#                                                 "shape_type": "rectangle",
#                                                 "group_id": None,
#                                                 "flags": {},
#                                                 "turtle_id" : ann.category_id
#                                             }
#                                             for ann in sliced_img['coco_image'].annotations
#                                         ]
#                                     }
#                                     save_json(new_labelme_json, tile_path.with_suffix(".json"))           

#         except:
#                 print(f"skipping")
#         return id

                
#             # Save only tiles containing objects
#             #sort by id

#   # Save JSON for the tile
#     def process_slice(dependencies, targets):
#         id = 0
#         for file in dependencies:
#             image =Path(file)
#             id =process_image(image,image.parent.parent / 'slices',id)

#     file_dep = list((config.geturl('output')).glob('**/images/*.JPG'))
#     return {
#         'file_dep' : file_dep,
#         'actions':[process_slice],
#         'clean':True,
#         'uptodate':[False],
#     }  


# def task_slice_empty_images():
#     def process_image(input_image_path: Path,output_dir,id: Path):
#         # Load LabelMe JSON
#         output_dir.mkdir(exist_ok=True,parents=True)
        
#         labelme_json_path = input_image_path.with_suffix(".json") 
#         if labelme_json_path.exists():
#             with open(labelme_json_path, "r") as f:
#                 labelme_data = json.load(f)
#             # Convert LabelMe annotations to SAHI CocoAnnotation format
#             coco_annotations = []
#             category_id=0
#             cats= dict()
#             for shape in labelme_data["shapes"]:
#                 if ('turtle' in shape["label"]) and not ('tracks' in shape["label"]) and not ('yolo' in shape["label"]) and (np.array(shape["points"]).shape == (2,2)) :
#                     if not shape["label"] in cats.keys():
#                         cats[shape["label"]] = category_id
#                         category_id=category_id +1
#                     if shape['shape_type']=='circle':
#                         points =circle_to_bbox(shape["points"], labelme_data['imageWidth'],labelme_data['imageHeight'])   
#                         x_min = points[0]
#                         y_min = points[1]
#                         x_max = points[2]
#                         y_max = points[3]                 
#                     else:
#                         points = shape["points"]
#                         x_min = min(p[0] for p in points)
#                         y_min = min(p[1] for p in points)
#                         x_max = max(p[0] for p in points)
#                         y_max = max(p[1] for p in points)

#                     category_name = shape["label"]
#                     coco_annotation = CocoAnnotation(
#                         bbox=[x_min, y_min, x_max - x_min, y_max - y_min],  # (x, y, width, height)
#                         category_name=category_name,
#                         category_id =id
                        
#                     )
#                     id = id +1
#                     coco_annotations.append(coco_annotation)
#             labels = dict()
#             coco_annotations = list(filter(lambda ann:len(ann.bbox)==4,coco_annotations))
#             for key in cats.keys():
#                 labels[key] = cats[key]
#             # Tile the image while remapping objects
#             print(input_image_path.stem)
#             if input_image_path.with_suffix('.JPG').exists():
#                 slices = slice_image(
#                     image=str(input_image_path.with_suffix('.JPG')),
#                     output_file_name = input_image_path.stem,
#                     coco_annotation_list=coco_annotations,  # Now passing valid SAHI annotations
#                     slice_height=640,  # Adjustable tile size
#                     slice_width=640,
#                     overlap_height_ratio=0.2,
#                     overlap_width_ratio=0.2
#                 )

#                 # Save only tiles containing objects
#                 #sort by id

#                 slices = list(filter(lambda s:len(s['coco_image'].annotations)==0,slices))
#                 annotation_idx = dict()
#                 for s in slices:
#                     for ann in s['coco_image'].annotations:
#                         if ann.category_id in annotation_idx.keys():

#                             if ann.area > annotation_idx[ann.category_id].area:
#                                 annotation_idx[ann.category_id] = ann
#                         else:
#                             annotation_idx[ann.category_id] = ann
                            

#                 used_id =[]
#                 for i, sliced_img in enumerate(slices):
#                     if len(sliced_img['coco_image'].annotations)==0:  # Only save with no bojects
#                         tile_path = output_dir / f"{sliced_img['filename']}"
#                         if not tile_path.exists():
#                             cv2.imwrite(str(tile_path),sliced_img['image'])
#         return id
#     def process_slice(dependencies, targets):
#         id = 0     
#         sample_size = min(len(dependencies), 500)
#         sampled_items = random.sample(dependencies, sample_size)
#         for file in sampled_items:
#             image =Path(file)
#             id =process_image(image,image.parent.parent / 'backgrounds',id)

#     file_dep = list((config.geturl('output')).glob('**/images/*.json'))
#     return {
#         'file_dep' : file_dep,
#         'actions':[process_slice],
#         'clean':True,
#         'uptodate':[True],
#     }  


# def task_file_blanks():
#     def process_slice(dependencies, targets):
#         tsample = min(len(dependencies), 300)
#         train = random.sample(dependencies, tsample)
#         training_path =Path('/home/mor582/turtles/yolo/images/train')
        
#         for file in train:
#             dest = training_path / Path(file).name
#             if not dest.exists():
#                 shutil.copy2(file,dest)
#         vsample = min(len(dependencies), 100)
#         val = random.sample(dependencies, vsample)
#         validation_path =Path('/home/mor582/turtles/yolo/images/val')
#         for file in val:
#             dest = validation_path / Path(file).name
#             if not dest.exists():
#                 shutil.copy2(file,dest)

#     file_dep = list((config.geturl('output')).glob('**/backgrounds/*.png'))
#     return {
#         'file_dep' : file_dep,
#         'actions':[process_slice],
#         'clean':True,
#         'uptodate':[True],
#     }  



# def task_process_catalog():
#     def loadshapes(file):
#         print(file)
#         lines =[]
#         with open(file, "r") as read_file:
#             data = json.load(read_file)

#         #data = json.loads(''.join(lines).replace("\n", "").replace("'", '"').replace('u"', '"'))
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
#         data = data.set_index('turtle_id').sort_values('FilePath')
#         data['area'] = data['points'].apply(calc_area)
#         data.to_csv(targets[0])       
        

#     dirs = directories = list((config.geturl('output') ).glob("**/slices"))
#     for directory in dirs:
#         file_dep = list(directory.glob('*.json'))
#         if len(file_dep)>0:
#             target =  directory / 'annotations.csv'            
#             yield {
#                 'name': directory.parent.stem,
#                 'file_dep' : file_dep,
#                 'actions':[process_labelme],
#                 'targets':[target],
#                 'clean':True,
#                 'uptodate':[False],
                
#             } 

# def task_process_tumbnails():
#     def loadshapes(file):
#         print(file)
#         lines =[]
#         with open(file, "r") as read_file:
#             data = json.load(read_file)

#         #data = json.loads(''.join(lines).replace("\n", "").replace("'", '"').replace('u"', '"'))
#         data =pd.DataFrame(data['shapes'])
#         data['FilePath'] =file             
#         return(data)
 
#     def process_tumbnails(dependencies, targets):
#         data = pd.read_csv(dependencies[0])
#         data = data[~data.turtle_id.isna()]
#         data['points'] = data['points'].apply(ast.literal_eval)
#         data['points'] = data['points'].apply(lambda x:np.array(x))
#         output = Path(dependencies[0]).parent.parent / 'thumbs'
#         output.mkdir(exist_ok=True)

        
#         for key, group in tqdm(data.groupby('FilePath'), desc="Processing background images and saving patches"):
#             im_path =Path(key).with_suffix('.png')
#             if im_path.exists():
#                 image = Image.open(im_path)
#                 for index,row in group.iterrows():
#                         x =np.sort(row.points[:,0])[[0,-1]]
#                         y =np.sort(row.points[:,1])[[0,-1]]
#                         points = [x[0],y[0],x[1],y[1]]
#                         stamp =image.crop(points)
#                         area =min(int(row.area // 1000) * 1000, 3000)
#                         output_dir = output / f'{area:04}'
#                         if not output_dir.exists():
#                             output_dir.mkdir(exist_ok=True)
#                             # (output_dir / 'sorted_1').mkdir(exist_ok=True)
#                             # (output_dir / 'sorted_2').mkdir(exist_ok=True)
#                             # (output_dir / 'sorted_3').mkdir(exist_ok=True)
#                             # (output_dir / 'zsorted_2_male').mkdir(exist_ok=True)
#                             # (output_dir / 'zsorted_3_male').mkdir(exist_ok=True)
#                             # (output_dir / 'sorted_delete').mkdir(exist_ok=True)
#                         patch_filename = output_dir /    f"{Path(key).stem}_{int(row.turtle_id):06}.JPG"
#                         draw = ImageDraw.Draw(stamp)
#                         #draw.rectangle(points, outline="red", width=2)
#                         try:
#                             stamp.save(patch_filename)
#                         except:
#                             print(patch_filename)



#             # for key, group in tqdm(df.sample(frac=0.1, random_state=42).groupby('Key'), desc="Processing background images and saving patches"):
#             #     # Open the image and convert to numpy array
#             #     image = Image.open(group.ImageFile.iloc[0])
#             #     if len(group) == 1:
#             #         bbox=group.iloc[0].BoundingBox
#             #         width = abs(bbox[0]-bbox[2])+5
#             #         height = abs(bbox[1]-bbox[3])+5
#             #         randx = np.random.randint(0,patch_size-width)
#             #         randy = np.random.randint(0,patch_size-height)
#             #         xmin = np.max([bbox[0],bbox[2]])+randx
#             #         ymin = np.min([bbox[1],bbox[3]])+randy
#             #         xmax = xmin+patch_size
#             #         ymax = ymin+patch_size
#             #         stamp =image.crop([xmin,ymin,xmax,ymax])
#             #         patch_filename = patch_path / f"{key}_patch_background.jpg"
#             #         if not bbox_to_patch_yolo(bbox,xmin,ymin,patch_size,patch_size,class_labels,group.iloc[0].Label):
#             #             stamp.save(patch_filename)
#             # for key, group in tqdm(df.groupby('Key'), desc="Processing images and saving patches"):
#             #     # Open the image and convert to numpy array
#             #     image = Image.open(group.ImageFile.iloc[0])
#             #     if len(group) == 1:
#             #         bbox=group.iloc[0].BoundingBox
#             #         width = abs(bbox[0]-bbox[2])+5
#             #         height = abs(bbox[1]-bbox[3])+5
#             #         randx = np.random.randint(0,patch_size-width)
#             #         randy = np.random.randint(0,patch_size-height)
#             #         xmin = np.min([bbox[0],bbox[2]])-randx
#             #         ymin = np.min([bbox[1],bbox[3]])-randy
#             #         xmax = xmin+patch_size
#             #         ymax = ymin+patch_size
#             #         stamp =image.crop([xmin,ymin,xmax,ymax])




     
        

#     files = list((config.geturl('output') ).rglob("annotations.csv"))
#     for file_dep in files:         
#         yield {
#             'name': file_dep.parent.parent.stem,
#             'file_dep' : [file_dep],
#             'actions':[process_tumbnails],
#             'clean':True,
#             'uptodate':[False],
            
#         }          


# def task_file_sorted():

#     def process_file_tumbnails(dependencies, targets):
#         data = pd.DataFrame(dependencies,columns=['FilePath'])
#         data[['Dir','Area','GroupId','ThumbFile']] = data['FilePath'].str.extract(r'(?P<Dir>.*?[A-Z]+_\d{8}T\d{4})/thumbs/(?P<Area>\d{4})/(?P<GroupId>[^/]+)/(?P<ThumbFile>[^/]+)')
#         data['ImageFile'] = data['ThumbFile'].str.extract(r'(?P<ImageFile>.*?_\d+T\d+_\d+_\d+_\d+_\d+_\d+)') + '.png'
#         for index,row in data.iterrows():
#             source_dir = Path(row.Dir)/ 'slices'
#             source_slice = source_dir / row.ImageFile
#             source_json = source_slice.with_suffix('.json')
#             dest = Path(row.Dir).parent / 'sorted' / row.GroupId / row.Area
#             Image_dest = dest / source_slice.name
#             Json_dest = dest / source_json.name
#             dest.mkdir(exist_ok=True,parents=True)
#             if not Image_dest.exists():
#                 os.link(source_slice.absolute(),Image_dest)
#             if not Json_dest.exists():
#                 os.link(source_json.absolute(),Json_dest)


#             # for key, group in tqdm(df.sample(frac=0.1, random_state=42).groupby('Key'), desc="Processing background images and saving patches"):
#             #     # Open the image and convert to numpy array
#             #     image = Image.open(group.ImageFile.iloc[0])
#             #     if len(group) == 1:
#             #         bbox=group.iloc[0].BoundingBox
#             #         width = abs(bbox[0]-bbox[2])+5
#             #         height = abs(bbox[1]-bbox[3])+5
#             #         randx = np.random.randint(0,patch_size-width)
#             #         randy = np.random.randint(0,patch_size-height)
#             #         xmin = np.max([bbox[0],bbox[2]])+randx
#             #         ymin = np.min([bbox[1],bbox[3]])+randy
#             #         xmax = xmin+patch_size
#             #         ymax = ymin+patch_size
#             #         stamp =image.crop([xmin,ymin,xmax,ymax])
#             #         patch_filename = patch_path / f"{key}_patch_background.jpg"
#             #         if not bbox_to_patch_yolo(bbox,xmin,ymin,patch_size,patch_size,class_labels,group.iloc[0].Label):
#             #             stamp.save(patch_filename)
#             # for key, group in tqdm(df.groupby('Key'), desc="Processing images and saving patches"):
#             #     # Open the image and convert to numpy array
#             #     image = Image.open(group.ImageFile.iloc[0])
#             #     if len(group) == 1:
#             #         bbox=group.iloc[0].BoundingBox
#             #         width = abs(bbox[0]-bbox[2])+5
#             #         height = abs(bbox[1]-bbox[3])+5
#             #         randx = np.random.randint(0,patch_size-width)
#             #         randy = np.random.randint(0,patch_size-height)
#             #         xmin = np.min([bbox[0],bbox[2]])-randx
#             #         ymin = np.min([bbox[1],bbox[3]])-randy
#             #         xmax = xmin+patch_size
#             #         ymax = ymin+patch_size
#             #         stamp =image.crop([xmin,ymin,xmax,ymax])




     
        

#     dirs = list((config.geturl('output') / 'turtles').rglob("*sorted*"))
#     file_dep = filter(lambda x:len(x),[list(d.glob('*.JPG')) for d in dirs])
#     file_dep = list(chain(*file_dep))
#     return {
#         'file_dep' : file_dep,
#         'actions':[process_file_tumbnails],
#         'clean':True,
#         'uptodate':[False],
        
#     }  
# for image in input_images:
#     process_image(image)


# # Load SAM model
# sam_checkpoint = "sam_vit_h_4b8939.pth"  # Ensure you've downloaded this
# sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
# sam.to(device="cuda" if torch.cuda.is_available() else "cpu")


# print(f"Filtered tiles and updated annotations saved in {output_dir}")

def task_make_stub_prompts():
    def create_stub_json(image_path:Path, prompt=[]):
        img = Image.open(image_path)
        width, height = img.size

        stub = {
            "version": "5.0.1",
            "flags": {},
            "shapes": [],
            "imagePath": image_path.name,
            "imageData": None,
            "imageHeight": height,
            "imageWidth": width,
            "metadata": {
                "prompt": prompt
            }
        }

        json_path = image_path.with_suffix('.json')
        with open(json_path, "w") as f:
            json.dump(stub, f, indent=2)

    def batch_stub_images(dependencies, targets):
        for img_path in dependencies:
            img_path = Path(img_path)
            if not img_path.with_suffix('.json').exists():
                create_stub_json(img_path, prompt=['Drone Image looking down on the ocean'])

    file_dep = list((config.geturl('output') / 'yolo_finds' / 'clip_reference' ).glob("*.JPG"))
    targets = [Path(file).with_suffix('.json') for file in file_dep]
    return {
        'file_dep' : file_dep,
        'targets' : targets,
        'actions':[batch_stub_images],
        'clean':True,
        'uptodate':[True],
        
    }   

def task_make_prototypes_prompts():
    def build_prototypes(dependencies, targets):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        def load_prompts(path:Path):
            if path.exists():
                print(path)
                with open(path, 'r') as f:
                    data = json.load(f)
                    return data['metadata']['prompt']
            return data.get("prompts", [])

        def encode_image(image_path):
            image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = model.encode_image(image)
                feat /= feat.norm(dim=-1, keepdim=True)
            return feat.cpu().numpy()

        def encode_prompts(prompts):

            tokens = clip.tokenize(prompts).to(device)
            with torch.no_grad():
                text_feats = model.encode_text(tokens)
                text_feats /= text_feats.norm(dim=-1, keepdim=True)
            return text_feats.cpu().numpy()

        # Step 1: Build class prototypes from few-shot
    
        prototypes = {}
        for class_dir in Path(dependencies[0]).parent.parent.iterdir():
            if not class_dir.is_dir():
                continue
            image_feats = []
            for image_path in class_dir.glob("*.JPG"):  # matches jpg, jpeg, png
                image_feats.append(encode_image(image_path))
            if image_feats:
                prototype = np.mean(np.vstack(image_feats), axis=0)
                prototype /= np.linalg.norm(prototype)
                prototypes[class_dir.name] = prototype
        
        # Save prototypes
        with open(targets[0], "wb") as f:
            pickle.dump(prototypes, f)
  

    file_dep = list((config.geturl('output') / 'yolo_finds' / 'clip_reference' ).rglob("*.JPG"))
    targets = config.geturl('output') / 'yolo_finds' / 'clip_reference' / "prototypes.plk"
    return {
        'file_dep' : file_dep,
        'targets' : [targets],
        'actions':[build_prototypes],
        'clean':True,
        'uptodate':[True],
        
    }  

# def task_sort_thumbs():
   
    
    
    
    
#     def classify_images(dependencies, targets,batch_size=32):
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         model, preprocess = clip.load("ViT-B/32", device=device)
#         def classify_batch( image_paths):
#             """
#             Classify a batch of images (list of file paths).
#             Returns list of tuples: (image_path, predicted_label, similarities_dict)
#             """
#             images = []
#             for p in image_paths:
#                 img = Image.open(p).convert("RGB")
#                 images.append(preprocess(img))
            
#             batch = torch.stack(images).to(device)

#             with torch.no_grad():
#                 feats = model.encode_image(batch)
#                 feats /= feats.norm(dim=-1, keepdim=True)

#             results = []
#             for i, feat in enumerate(feats):
#                 similarities = {
#                     label: (feat @ ref_feat.T).item()
#                     for label, ref_feat in prototypes.items()
#                 }
#                 best_label = max(similarities, key=similarities.get)
#                 results.append((image_paths[i], best_label, similarities))
#             return results 
        
#         proto_path = config.geturl('output') / 'yolo_finds' / 'clip_reference' / "prototypes.plk"
#         with open(proto_path, "rb") as f:
#             prototypes = pickle.load(f)
#         #prepare the batches
#         def chunked(lst, size):
#             for i in range(0, len(lst), size):
#                 yield lst[i:i + size]
#         all_results = []
#         for batch in chunked(dependencies, batch_size):
#             batch_results = classify_batch(batch)
#             all_results.extend(batch_results)

#         df = pd.DataFrame(all_results,columns=['SourceFile','Label','Scores'])
#         df['BestScore'] = df.apply(lambda x: x.Scores[x.Label],axis=1)
#         df = pd.concat([df,df['Scores'].apply(pd.Series)],axis=1).drop('Scores',axis=1)
#         df.columns = df.columns.str.capitalize()
#         df.sort_values(['Label','Bestscore']).to_csv(targets[0],index=False)

#     base_path = config.geturl('output') / 'yolo_finds'
#     file_dep = list(filter(lambda p: 'thumbs' in p.parts,base_path.rglob('*.JPG')))
#     targets = config.geturl('output') / 'yolo_finds' / 'clip_reference' / "thumbs_classified.csv"
#     return {
#         'file_dep' : file_dep,
#         'targets' : [targets],
#         'actions':[classify_images],
#         'clean':True,
#         'uptodate':[False],
        
#     }   


def task_file_clip():
 
    def process_file_clip(dependencies, targets):
        data = pd.read_csv(dependencies[0])
        output =config.geturl('output') / 'yolo_finds' / 'clip_turtle'
        output.mkdir(exist_ok=True,parents=True)
        for index,row in data.iterrows():
            source = Path(row.Sourcefile)
            Image_dest = output / row.Label / f'{int(row.Bestscore*1000):03d}-{source.name}'
            Image_dest.parent.mkdir(exist_ok=True,parents=True)
            if not Image_dest.exists():                                                                                                                                                                                                                                                                                                                                                        ``
                os.link(source.absolute(),Image_dest)  
    file_dep = config.geturl('output') / 'yolo_finds' / 'clip_reference' / "thumbs_classified.csv"
    return {
        'file_dep' : [file_dep],
        'actions':[process_file_clip],
        'clean':True,                                                                                                                                                                                
        'uptodate':[False],
        
    }   

if __name__ == '__main__':
    import doit
    DOIT_CONFIG = {'check_file_uptodate': 'timestamp'}
    #print(globals())
    doit.run(globals()) 