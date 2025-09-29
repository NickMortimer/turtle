import typer
from pathlib import Path
import cv2
from tqdm import tqdm
from patched_yolo_infer import MakeCropsDetectThem, CombineDetections
import time
import csv
import pandas as pd

def detect(
    model_path: Path = typer.Option(..., help="Path to YOLO model .pt file"),
    image_dir: list[Path] = typer.Option(None, help="One or more directories containing images to process"),
    image_dir_file: Path = typer.Option(None, help="Text file with directories to process, one per line"),
    output_dir: Path = typer.Option(..., help="Directory to save crops"),
    shape_x: int = 640,
    shape_y: int = 640,
    overlap_x: int = 25,
    overlap_y: int = 25,
    conf: float = 0.5,
    iou: float = 0.7,
    batch_inference: bool = True,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    dirs_to_process = []
    if image_dir:
        dirs_to_process.extend(image_dir)
    if image_dir_file:
        with open(image_dir_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    dirs_to_process.append(Path(line))
    if not dirs_to_process:
        print("No image directories provided.")
        return
    crop_records = []
    for img_dir in dirs_to_process:
        img_dir = Path(img_dir)
        image_files = sorted(list(img_dir.glob('DJIP4*.JPG')))
        dir_crop_records = []
        dir_output_dir = output_dir / img_dir.name
        dir_output_dir.mkdir(exist_ok=True, parents=True)
        csv_path = dir_output_dir / "crops_metadata.csv"
        for image_path in tqdm(image_files, desc=f"Processing {img_dir.name}"):
            try:
                element_crops = MakeCropsDetectThem(
                    image=cv2.imread(str(image_path)),
                    model_path=str(model_path),
                    segment=False,
                    shape_x=shape_x,
                    shape_y=shape_y,
                    overlap_x=overlap_x,
                    overlap_y=overlap_y,
                    conf=conf,
                    iou=iou,
                    batch_inference=batch_inference,
                )
                if not hasattr(element_crops, '_progress_bars'):
                    element_crops._progress_bars = {}
                start_time = time.time()
                result_patched = CombineDetections(element_crops, nms_threshold=0.25)
                end_time = time.time()
                if len(result_patched.filtered_classes_names) > 0:
                    crops_dir = dir_output_dir / image_path.stem
                    crops_dir.mkdir(exist_ok=True, parents=True)
                    img = cv2.imread(str(image_path))
                    for idx, (class_name, box) in enumerate(zip(result_patched.filtered_classes_names, result_patched.filtered_boxes)):
                        x1, y1, x2, y2 = map(int, box)
                        crop = img[y1:y2, x1:x2]
                        class_dir = crops_dir / class_name
                        class_dir.mkdir(exist_ok=True, parents=True)
                        crop_filename = class_dir / f"{image_path.stem}_det_{idx}_{class_name}.jpg"
                        cv2.imwrite(str(crop_filename), crop)
                        dir_crop_records.append({
                            'crop_path': str(crop_filename.resolve()),
                            'original_image': str(image_path.resolve()),
                            'class_name': class_name,
                            'x1': x1,
                            'y1': y1,
                            'x2': x2,
                            'y2': y2
                        })
                del element_crops
                print(f"Processed {image_path.name} in {end_time - start_time:.2f}s")
            except Exception as e:
                print(f"Image problems {image_path}: {e}")
            # Save/update CSV after each image
            if dir_crop_records:
                df = pd.DataFrame(dir_crop_records)
                df.to_csv(csv_path, index=False)
    print(f"Wrote crop metadata to {csv_path}")

app = typer.Typer()
app.command()(detect)

if __name__ == "__main__":
    app()
