import json
from pathlib import Path
import typer
import numpy as np

app = typer.Typer()

def convert_coco_to_anylabeling_0410(coco_json_path: Path, output_dir: Path):
    # Read the COCO JSON file
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Create a dictionary to hold image-specific data
    # image_data_map = {image['id']: {
    #     "version": "0.4.10",  # Set the AnyLabeling version
    #     "imagePath": Path(image['file_name']).name,
    #     "imageData": None,
    #     "imageHeight": 800,
    #     "imageWidth": 1280
    # } for image in coco_data['images']}
    image_data_map = {image['id']: {
        "file_name": image['file_name'],
        "width" : image['width'],
        "height" : image['height'],
        "annotations": []
    } for image in coco_data['images']}

    category_data_map = {name['id']: {
        'name':name['name']
    } for name in coco_data['categories']}   


    # Add annotations to the corresponding images
    for annotation in coco_data['annotations']:
        image_id = annotation['image_id']
        if image_id in image_data_map:
            image_data_map[image_id]['annotations'].append({
                "id": annotation['id'],
                "category_id": annotation['category_id'],
                "bbox": annotation['bbox'],
                "segmentation": annotation['segmentation'],
                "area": annotation['area'],
                "iscrowd": annotation['iscrowd']
            })
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write each image's data to a separate JSON file
    for image_id, image_data in image_data_map.items():
        image_file_name = Path(image_data['file_name']).stem
        anylabeling_json_path = output_dir / f"{image_file_name}.json"
        shapes =[]
        for annotation in image_data['annotations']:
            if annotation['category_id'] in category_data_map:
                shapes.append({
                    'label':category_data_map[annotation['category_id']]['name'],
                    'text':"",
                    'points': np.array(annotation['segmentation']).reshape(-1,2).tolist(),
                    "group_id": None,
                    "shape_type": "polygon",
                    "flags": {}
                })
        with open(anylabeling_json_path, 'w') as f:
            data = {
                "version": "0.4.10",
                "flags": {},
                "shapes":shapes,
                "imagePath": Path(image_data['file_name']).name,
                "imageData": None,
                "imageHeight": image_data['height'],
                "imageWidth": image_data['width']
            }
            json_object = json.dumps(data, indent=2)
            f.write(json_object)

@app.command()
def convert(
    coco_json_path: Path = typer.Argument(..., help="Path to the COCO JSON file"),
    output_dir: Path = typer.Argument(..., help="Output directory for AnyLabeling JSON files")
):
    """
    Convert COCO JSON annotations to AnyLabeling 0.4.10 format.
    """
    convert_coco_to_anylabeling_0410(coco_json_path, output_dir)
    typer.echo(f"Conversion complete. JSON files saved in {output_dir}")

if __name__ == "__main__":
    app()