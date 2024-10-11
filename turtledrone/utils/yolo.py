import json
from pathlib import Path
import numpy as np

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
    
def circle_to_bbox(circle_ann, img_width, img_height):
    # Convert string representation of points to list if necessary
    if isinstance(circle_ann, str):
        circle_ann = json.loads(circle_ann.replace("'", "\""))
    # Get the center and edge coordinates of the circle
    center = circle_ann[0]
    edge = circle_ann[1]
    # Calculate the radius of the circle
    radius = ((center[0] - edge[0])**2 + (center[1] - edge[1])**2)**0.5

    # Calculate the bounding box coordinates
    x_min = center[0] - radius*1.3
    y_min = center[1] - radius*1.3
    x_max = center[0] + radius*1.3
    y_max = center[1] + radius*1.3

    # Convert the bounding box coordinates to integers
    bbox = [int(x_min), int(y_min), int(x_max), int(y_max)]

    # If the bounding box coordinates are outside the image, set them to the image boundary
    if bbox[0] < 0:
        bbox[0] = 0
    if bbox[1] < 0:
        bbox[1] = 0
    if bbox[2] > img_width:
        bbox[2] = img_width
    if bbox[3] > img_height:
        bbox[3] = img_height

    return bbox   

def convert_box_to_labelme(annotation,suffix='_yolo'):
    bbox =np.array(annotation['bbox'])
    bbox[[2,3]] =bbox[0:2] + bbox[2:4]
    bbox = bbox.reshape([2,2])
    return {"label": annotation['category_name']+suffix,"points":bbox.tolist(),"group_id": None,"shape_type": "rectangle","flags": {}}


def covert_labelme(image_name,detections):
    shapes = [convert_box_to_labelme(detec) for detec in detections]
    return {"version": "4.5.6","flags": {},"shapes":shapes,  "imagePath": image_name,"imageData":None,"imageHeight": 3648,"imageWidth": 5472}

def find_label(shapes,label_name):
    for shape in shapes:
        if label_name in shape['label']:
            return True
    return False

def load_json(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data

def save_json(json_path,data):
    with open(json_path, 'w') as outfile:
        json.dump(data, outfile, indent=4)  
        
def parse_json(json_path):
        data =load_json(json_path)
        detections = []
        try:
            for shape in data['shapes']:  # Use keys from the JSON file directly
                detections.append((shape['label'], shape['points'], shape['shape_type'],Path(json_path).parent / data['imagePath'],data['imageWidth'],data['imageHeight']))
        except:
            print(data)
        return detections 