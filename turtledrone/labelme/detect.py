from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import numpy as np
from pathlib import Path
import json






# Access the object prediction list


def convert_box_to_labelme(annotation):
    bbox =np.array(annotation['bbox'])
    bbox[[2,3]] =bbox[0:2] + bbox[2:4]
    bbox = bbox.reshape([2,2])
    return {"label": annotation['category_name']+'_yolo',"points":bbox.tolist(),"group_id": None,"shape_type": "rectangle","flags": {}}
 
def covert_labelme(image_name,detections):
    shapes = [convert_box_to_labelme(detec) for detec in detections]
    return {"version": "4.5.6","flags": {},"shapes":shapes,  "imagePath": image_name,"imageData":None,"imageHeight": 3648,"imageWidth": 5472}

detection_model = AutoDetectionModel.from_pretrained(
    model_type="yolov8",
    model_path='/media/mor582/Storage/Nick/surveys/train/yolo/runs/detect/train24/weights/best.pt',
    confidence_threshold=0.3,
    device='cuda:0'
)
# /home/mor582/deepthinker/surveys/test/TULKI_20231103T0817
# Convert to COCO annotation, COCO prediction, imantics, and fiftyone formats
input_files = list(map(str,list(Path('/media/mor582/Storage/Nick/surveys/AU/MANGB/MANGB_20221020T0949').glob('DJIP4*.JPG'))))
input_files.sort()
#results =[get_sliced_prediction(image,detection_model,slice_height=640,slice_width=640,overlap_height_ratio=0.2,overlap_width_ratio=0.2) for image in input_files]
for image in input_files:
    file_path =Path(image)
    result =get_sliced_prediction(image,detection_model,slice_height=640,slice_width=640,overlap_height_ratio=0.2,overlap_width_ratio=0.2)
    json_object =covert_labelme(file_path.name,result.to_coco_annotations())
    json_file =file_path.with_suffix('.json')
    if json_file.exists():
        with open(file_path.with_suffix('.json'), "r") as read_file:
            data = json.load(read_file)
            json_object['shapes'] =json_object['shapes'] +data['shapes']
    with open(file_path.with_suffix('.json'), "w") as outfile:
        outfile.write(json.dumps(json_object, indent=2))




