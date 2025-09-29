from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import cv2
import time
import torch






# Access the object prediction list


def convert_box_to_labelme(annotation):
    bbox =np.array(annotation['bbox'])
    bbox[[2,3]] =bbox[0:2] + bbox[2:4]
    bbox = bbox.reshape([2,2])
    return {"label": annotation['category_name']+'_yolo',"points":bbox.tolist(),"group_id": None,"shape_type": "rectangle","flags": {}}
 
def covert_labelme(image_name,detections):
    confidences=detections.filtered_confidences
    boxes=detections.filtered_boxes
    classes_ids=detections.filtered_classes_id
    classes_names=detections.filtered_classes_names

    shapes =[]
    for class_name, class_id, box, confidence in zip(classes_names, classes_ids, boxes, confidences):
        annotation = {
            'category_name': class_name,
            'category_id': class_id,
            'bbox': box,
            'score': confidence
        }
        shapes.append(convert_box_to_labelme(annotation))
    return {"version": "4.5.6","flags": {},"shapes":shapes,  "imagePath": image_name,"imageData":None,"imageHeight": 3648,"imageWidth": 5472}

# detection_model = AutoDetectionModel.from_pretrained(
#     model_type="yolov11",
#     model_path='/home/mor582/runs/detect/train16/weights/best.pt',
#     confidence_threshold=0.3,
#     device='cuda:0'
# )

from patched_yolo_infer import MakeCropsDetectThem, CombineDetections

# Check CUDA availability
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA not available - will use CPU")




# /home/mor582/deepthinker/surveys/test/TULKI_20231103T0817
# Convert to COCO annotation, COCO prediction, imantics, and fiftyone formats
with open('/home/mor582/process.txt','r') as process_dirs:
    dirs = process_dirs.readlines()
for path in dirs:
    input_files = list(map(str,list(Path(path[:-1]).glob('DJIP4*.JPG'))))
    input_files.sort()
    #results =[get_sliced_prediction(image,detection_model,slice_height=640,slice_width=640,overlap_height_ratio=0.2,overlap_width_ratio=0.2) for image in input_files
    for image in  tqdm(input_files, desc=f"Processing {Path(path).name}"):
        file_path =Path(image)
        json_file =file_path.with_suffix('.json')
        # if not json_file.exists():
        try:
            element_crops = MakeCropsDetectThem(
                image=cv2.imread(image),
                #model_path='/home/mor582/runs/detect/train16/weights/best.pt',
                model_path='/media/mor582/Timor2/yolov11l_dugong_aug_trained.pt',
                segment=False,
                shape_x=640,
                shape_y=640,
                overlap_x=25,
                overlap_y=25,
                conf=0.5,
                iou=0.7,
                batch_inference=True,
            )
            
            # Ensure the object has the _progress_bars attribute to avoid destructor issues
            if not hasattr(element_crops, '_progress_bars'):
                element_crops._progress_bars = {}
            
            # Time the patched_yolo_infer method
            start_time_patched = time.time()
            result_patched = CombineDetections(element_crops, nms_threshold=0.25)  
            end_time_patched = time.time()
            patched_time = end_time_patched - start_time_patched
            
            # Time the SAHI method
            start_time_sahi = time.time()
            #result_sahi = get_sliced_prediction(image, detection_model, slice_height=640, slice_width=640, overlap_height_ratio=0.2, overlap_width_ratio=0.2)
            end_time_sahi = time.time()
            sahi_time = end_time_sahi - start_time_sahi
            
            # print(f"Timing for {file_path.name}:")
            # print(f"  Patched YOLO: {patched_time:.3f}s")
            # print(f"  SAHI: {sahi_time:.3f}s")
            # print(f"  Speed difference: {sahi_time/patched_time:.2f}x")
            if len(result_patched.filtered_classes_names)>0:
                # Save crops of detected objects
                crops_dir = file_path.parent / 'crops'
                crops_dir.mkdir(exist_ok=True)
                img = cv2.imread(str(file_path))
                for idx, (class_name, box) in enumerate(zip(result_patched.filtered_classes_names, result_patched.filtered_boxes)):
                    x1, y1, x2, y2 = map(int, box)
                    crop = img[y1:y2, x1:x2]
                    crop_filename = crops_dir / f"{file_path.stem}_det_{idx}_{class_name}.jpg"
                    cv2.imwrite(str(crop_filename), crop)
                
            # Explicitly clean up the object
            del element_crops
        except Exception as e:
            print(f'Image problems {image}: {e}')




