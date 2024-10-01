import os
import pandas as pd
import numpy as np
import json
from PIL import Image
from tqdm import tqdm
import doit
from doit.action import CmdAction
from doit import create_after
from pathlib import Path
import ast



def list_and_count_files(image_folder, image_extension):
    jpg_files = [f for f in os.listdir(image_folder) if f.endswith(image_extension)]
    json_files = [f for f in os.listdir(image_folder) if f.endswith('.json')]
    print(f"Number of images found: {len(jpg_files)}")
    print(f"Number of JSON files found: {len(json_files)}")
    # If the number of images and JSON files don't match, exit the program
    if len(jpg_files) != len(json_files):
        print("The number of images and JSON files does not match. Please check the folder and try again.")
        exit()
    return jpg_files, json_files

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

def create_detections_dataframe(image_folder, jpg_files, image_extension):
    detections = []
    for jpg_file in tqdm(jpg_files, desc="Processing JPG/JSON files and creating dataframe"):
        # Extract survey information from the image filename
        survey_info = jpg_file.split('_')[3:5]
        datetime = pd.to_datetime(survey_info[1], format='%Y%m%dT%H%M%S')
        
        # Open image to get width and height
        img_path = os.path.join(image_folder, jpg_file)
        try:
            with Image.open(img_path) as img:
                img_width, img_height = img.size
        except OSError as e:
            print(f"Skipping {jpg_file} due to error: {e}")
            continue
        
        # Check for corresponding json file
        json_file = jpg_file.replace(image_extension, '.json')
        json_path = os.path.join(image_folder, json_file)
        if os.path.exists(json_path):
            # Parse JSON file and extend list with the detections
            for label, points, shape_type in parse_json(json_path):
                if shape_type == 'circle':
                    bbox = circle_to_bbox(points, img_width, img_height)
                elif shape_type == 'polygon':
                    # Assuming points for polygons are given as [[x_min, y_min], [x_max, y_max]]
                    bbox = [int(points[0][0]), int(points[0][1]), int(points[1][0]), int(points[1][1])]
                else:
                    bbox = None
                detections.append([jpg_file, survey_info[0], datetime, label, points, bbox])
        else:
            print(f'No JSON file for {jpg_file}')
            # If no JSON file exists, still append a row with None values for label, points, and bbox
            detections.append([jpg_file, survey_info[0], datetime, None, None, None])

    # Create a dataframe from the detections list
    columns = ['filename', 'survey_area', 'datetime', 'label', 'points', 'bbox']
    detections_df = pd.DataFrame(detections, columns=columns)

    # Sort the dataframe by datetime and reset index
    detections_df = detections_df.sort_values('datetime').reset_index(drop=True)

    # Drop any rows where the label does not contain 'turtle' and remove 'turtle_tracks' labels
    detections_df = detections_df[detections_df['label'].str.contains('turtle', na=False)].reset_index(drop=True)
    detections_df = detections_df[~detections_df['label'].str.contains('turtle_tracks', na=False)].reset_index(drop=True)

    print(f"Number of turtle detections: {len(detections_df)}")
    return detections_df

def generate_class_labels(detections_df):
    class_labels_list = detections_df['Label'].value_counts().index.tolist()
    class_labels_dict = {label: index for index, label in enumerate(class_labels_list)}
    print(f"Class labels: {class_labels_dict}")
    return class_labels_dict

def pad_to_nearest(image_np, patch_size, overlap):
    height, width = image_np.shape[:2]
    stride = patch_size - overlap

    # Calculate the number of patches to cover the entire image without padding
    num_patches_y = np.ceil(height / stride).astype(int)
    num_patches_x = np.ceil(width / stride).astype(int)

    # Determine the total length of the image after tiling, including overlaps
    total_y = stride * (num_patches_y - 1) + patch_size
    total_x = stride * (num_patches_x - 1) + patch_size

    # Calculate padding to add to reach the new total length
    pad_y = (total_y - height)
    pad_x = (total_x - width)

    # Even padding on both sides
    pad_height = (pad_y // 2, pad_y - (pad_y // 2))
    pad_width = (pad_x // 2, pad_x - (pad_x // 2))

    # Pad the image
    padded_image = np.pad(image_np, (pad_height, pad_width, (0, 0)), mode='constant')

    return padded_image, pad_height, pad_width

def split_image_into_patches(image_np, patch_size, overlap):
    stride = patch_size - overlap
    padded_image, pad_height, pad_width = pad_to_nearest(image_np, patch_size, overlap)
    
    patches = []
    padded_height, padded_width = padded_image.shape[:2]
    for i in range(0, padded_height - overlap, stride):
        for j in range(0, padded_width - overlap, stride):
            patch = padded_image[i:i + patch_size, j:j + patch_size]
            patches.append((patch, (j - pad_width[0], i - pad_height[0])))  # Subtract padding to get original image coordinates

    return patches

def bbox_to_yolo_format(x_min, y_min, x_max, y_max, patch_width, patch_height):
    bbox_width = x_max - x_min
    bbox_height = y_max - y_min
    center_x = x_min + (bbox_width / 2)
    center_y = y_min + (bbox_height / 2)
    return center_x / patch_width, center_y / patch_height, bbox_width / patch_width, bbox_height / patch_height

def bbox_to_patch_yolo(bbox, patch_x, patch_y, patch_width, patch_height, class_labels, label):
    x_min, y_min, x_max, y_max = bbox
    class_id = class_labels.get(label, None)
    # Clamp the coordinates to be within the patch
    x_min_clamped = max(x_min, patch_x)
    y_min_clamped = max(y_min, patch_y)
    x_max_clamped = min(x_max, patch_x + patch_width)
    y_max_clamped = min(y_max, patch_y + patch_height)

    # Check whether the adjusted bounding box has a non-zero area
    if x_max_clamped <= x_min_clamped or y_max_clamped <= y_min_clamped:
        # If the bounding box doesn't overlap at all, return empty list
        return []

    # Convert clamped coordinates to the patch's coordinate space
    x_min_rel = x_min_clamped - patch_x
    y_min_rel = y_min_clamped - patch_y
    x_max_rel = x_max_clamped - patch_x
    y_max_rel = y_max_clamped - patch_y

    # Convert to YOLO format (assumes bbox_to_yolo_format is defined)
    center_x, center_y, bbox_width, bbox_height = bbox_to_yolo_format(x_min_rel, y_min_rel, x_max_rel, y_max_rel, patch_width, patch_height)

    yolo_annotation = f"{class_id} {center_x} {center_y} {bbox_width} {bbox_height}"

    return [yolo_annotation]

def split_image_into_patches_with_annotations(detections_df, image_np, patch_size, overlap, class_labels):

    patches_with_coords = split_image_into_patches(image_np, patch_size, overlap)
    patch_annotations = [[] for _ in range(len(patches_with_coords))]
    patches = []

    for i, (patch, (origin_x, origin_y)) in enumerate(patches_with_coords):
        patches.append(patch)
        for _, row in detections_df.iterrows():
            label = row['Label']
            bbox = row['BoundingBox']
            patched_annotations = bbox_to_patch_yolo(bbox, origin_x, origin_y, patch_size, patch_size, class_labels, label)
            for patched_annotation in patched_annotations:
                if patched_annotation:
                    patch_annotations[i].append(patched_annotation)

    return patches, patch_annotations

def task_process_json():
    def process_json(dependencies, targets):
        detections =[]
        for json_file in tqdm(dependencies, desc="Processing JPG/JSON files and creating dataframe"):
                for label, points, shape_type, imagefile,img_width,img_height in parse_json(json_file):
                    if label in config.cfg['trainkeep']:
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
    target = config.geturl('output') / 'train' / 'object_index.csv'
    return {
        'actions':[process_json],
        'file_dep':file_dep,
        'targets':[target],
        'clean': True,
    } 


@create_after(executed='process_json', target_regex='*.csv')    
def task_make_taining():
    def process_images_and_save_patches(dependencies, targets):
        #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------_df, image_folder, patch_size, overlap, class_labels, output_folder):
        # Ensure output directories exist
        patch_path = config.geturl('trainpath') / 'yolo' / 'images'
        os.makedirs(patch_path, exist_ok=True)
        patch_annotation_path = config.geturl('trainpath') / 'yolo' /  'labels'
        os.makedirs(patch_annotation_path, exist_ok=True)
        patch_size = int(config.cfg['patch_size'])
        overlap = int(config.cfg['overlap'])

        # Iterate over images
        detections_df = pd.read_csv(dependencies[0]).dropna(subset='BoundingBox')
        detections_df['BoundingBox'] = detections_df['BoundingBox'].apply(ast.literal_eval)
        class_labels = generate_class_labels(detections_df)
        for key, group in tqdm(detections_df.groupby('Key'), desc="Processing images and saving patches"):
            # Open the image and convert to numpy array
            image_np = np.array(Image.open(group.ImageFile.iloc[0]))
            
            # Process and split each image into patches with annotations
            patches, patch_annotations = split_image_into_patches_with_annotations(group, image_np, patch_size, overlap, class_labels)
         
            # Save each patch and its annotations---------------------------------
            for i, (patch, annotation_texts) in enumerate(zip(patches, patch_annotations)):
                if len(annotation_texts)>0:
                    patch_filename = f"{key}_patch_{i}.jpg"
                    patch_annotation_filename = f"{key}_patch_{i}.txt"   
                    Image.fromarray(patch).save(patch_path / patch_filename )  # Save image patch
                    # Save YOLO annotations if available
                    with open(patch_annotation_path / patch_annotation_filename , 'w') as f:
                        f.writelines([f"{text}\n" for text in annotation_texts])
        detections_df.to_csv(targets[0],index=False)
    file_dep =config.geturl('output') / 'train' / 'object_index.csv'
    target = config.geturl('output') / 'train' / 'yolo.csv'
    return {
        'actions':[CmdAction(process_images_and_save_patches, buffering=1)],
        'file_dep':[file_dep],
        'targets':[target],
        'clean': True,
    } 


# def main():

#     # Define input and output folders, patch size, and overlap
#     image_folder = '/media/daniel/Storage/IMAGES/Turtles/drone/surveys/yolo/data/images'
#     image_extension = '.JPG'
#     patch_size = 1280
#     overlap = 40
#     output_folder = '/media/daniel/Storage/IMAGES/Turtles/drone/surveys/yolo/data/test_patches'

#     jpg_files, json_files = list_and_count_files(image_folder, image_extension)
#     detections_df = create_detections_dataframe(image_folder, jpg_files, image_extension)
#     class_labels = generate_class_labels(detections_df)
#     process_images_and_save_patches(detections_df, image_folder, patch_size, overlap, class_labels, output_folder)


if __name__ == '__main__':
    import doit
    DOIT_CONFIG = {'check_file_uptodate': 'timestamp'}
    #print(globals())
    doit.run(globals())   