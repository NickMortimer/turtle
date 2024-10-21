from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import numpy as np
from pathlib import Path
import json
import pandas as pd
import shutil





# Access the object prediction list


def convert_box_to_labelme(annotation):
    bbox =np.array(annotation['bbox'])
    bbox[[2,3]] =bbox[0:2] + bbox[2:4]
    bbox = bbox.reshape([2,2])
    return {"label": annotation['category_name']+'_yolo',"points":bbox.tolist(),"group_id": None,"shape_type": "rectangle","flags": {}}


def yolo_to_labelimg(yolo_file,json_file, image_width, image_height, class_names):
    annotations = []

    # Read YOLO file
    with open(yolo_file, 'r') as file:
        lines = file.readlines()
        
        for line in lines:
            parts = line.strip().split()
            object_class = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            # Convert YOLO to LabelImg (Pascal VOC) coordinates
            xmin = (x_center - width / 2) * image_width
            ymin = (y_center - height / 2) * image_height
            xmax = (x_center + width / 2) * image_width
            ymax = (y_center + height / 2) * image_height
            
            # Create annotation for LabelImg
            bbox = [[xmin,ymin],[xmax,ymax]]
            annotation = {'label':'turtle',"points":bbox,"group_id": None,"shape_type": "rectangle","flags": {}}           
            annotations.append(annotation)
    # Create the final JSON structure
    labelimg_data = {
        "version": "4.5.7",  # You can put the version of LabelImg here
        "flags": {},
        "shapes": annotations,
        "imagePath": json_file.with_suffix('.jpg').name,  # Assuming image has the same name as the annotation file
        "imageData": None,
        "imageHeight": image_height,
        "imageWidth": image_width
    }
    # Write to JSON file
    with open(json_file, 'w') as outfile:
        json.dump(labelimg_data, outfile, indent=4)            


 
def covert_labelme(image_name,detections):
    shapes = [convert_box_to_labelme(detec) for detec in detections]
    return {"version": "4.5.6","flags": {},"shapes":shapes,  "imagePath": image_name,"imageData":None,"imageHeight": 3648,"imageWidth": 5472}

def parse_json(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
        detections = []
        try:
            for shape in data['shapes']:  # Use keys from the JSON file directly
                detections.append((shape['label'], shape['points'], shape['shape_type'],Path(json_path).parent / data['imagePath'],data['imageWidth'],data['imageHeight']))
        except:
            print(data)
        return detections




#get a list of all the images in QC
# qc_files = list(map(lambda x: x.name,list(Path('/media/mor582/Storage/Nick/surveys/train/yolo/train/qc').glob('*.jpg'))))
# keep_Images = [file for file in Path('/media/mor582/Storage/Nick/surveys/train/yolo/train/images').glob('*.jpg') if (file.name in qc_files) or ('background' in file.name)]
# clean =Path('/media/mor582/Storage/Nick/surveys/train/yolo/train/best/images')

# def yolo_to_labelimage(souce_path):
#     souce_path.mkdir(exist_ok=True)
#     class_names = ['turtle', 'turtle', 'turtle','turtle']
#     label_path = souce_path /'labels'
#     image_path = souce_path /'images'
#     yolo_source = label_path.glob('*.txt')
#     for file in yolo_source:
#         dest = image_path / file.with_suffix('.jpg').name
#         if dest.exists():
#             yolo_to_labelimg(file,dest.with_suffix('.json'),640,640,class_names)
        
# yolo_to_labelimage(Path('/media/mor582/Storage/Nick/surveys/train/yolo/val'))

def make_yolo(root_path):
    img_path = root_path / 'images'
    lbl_path = root_path / 'labels'
    json_files = img_path.glob('*.json')
    class_labels = {'turtle':0,'turtle_blob':1}
    lbl_path.mkdir(exist_ok=True)
    for json_file in json_files:
        yolo_annotations=[]
        for label, points, shape_type,image_path,image_width,image_height in parse_json(json_file):
            if shape_type == 'rectangle':
                xmin = min(points[0][0], points[1][0])
                ymin = min(points[0][1], points[1][1])
                xmax = max(points[0][0], points[1][0])
                ymax = max(points[0][1], points[1][1])
                # Convert to YOLO format
                x_center = (xmin + xmax) / 2 / 640
                y_center = (ymin + ymax) / 2 / 640
                width = (xmax - xmin) / 640
                height = (ymax - ymin) / 640
                yolo_annotations.append(f"{class_labels[label]} {x_center} {y_center} {width} {height}")
        with open(lbl_path / json_file.with_suffix('.txt').name , 'w') as f:
            f.writelines([f"{text}\n" for text in yolo_annotations])

make_yolo(Path('/media/mor582/Storage/Nick/surveys/train/yolo/val_best'))
make_yolo(Path('/media/mor582/Storage/Nick/surveys/train/yolo/train_best'))

# Convert to COCO annotation, COCO prediction, imantics, and fiftyone formats
input_files = list(map(str,list(Path('/media/mor582/Storage/Nick/surveys/AU/MWBAY/MWBAY_20230507T0907/').glob('*.JPG'))))

#results =[get_sliced_prediction(image,detection_model,slice_height=640,slice_width=640,overlap_height_ratio=0.2,overlap_width_ratio=0.2) for image in input_files]
for image in input_files:
    file_path =Path(image)
    result =get_sliced_prediction(image,detection_model,slice_height=640,slice_width=640,overlap_height_ratio=0.2,overlap_width_ratio=0.2)
    json_object =covert_labelme(file_path.name,result.to_coco_annotations()) 
    with open(file_path.with_suffix('.json'), "w") as outfile:
        outfile.write(json.dumps(json_object, indent=2))




