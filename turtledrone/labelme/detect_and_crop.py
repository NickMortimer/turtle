from pprint import pp
import typer
from pathlib import Path
import cv2
from tqdm import tqdm
from patched_yolo_infer import MakeCropsDetectThem, CombineDetections
import time
import csv
import pandas as pd
import ast
import numpy as np
from PIL import Image
import concurrent.futures

from turtledrone.labelme.detect_and_crop_transformer import (
    detect as detect_transformer,
)



app = typer.Typer()

@app.command('detect')
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
    run_number: int = typer.Option(None, help="Run number to continue from (overrides auto-increment)") ,
    require_cuda: bool = typer.Option(True, help="Fail if CUDA is not available")
):
    # Show device info
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if require_cuda and device != 'cuda':
        print("ERROR: --require-cuda was set but CUDA is not available. Exiting.")
        raise typer.Exit(1)
    print(f"Using device: {device.upper()} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU only'})")

    # Determine output directory name based on model and run_number
    model_stem = model_path.stem
    base_output_dir = Path(output_dir)
    if run_number is not None:
        output_dir = base_output_dir / f"{model_stem}_run{run_number:02d}"
    else:
        output_dir = base_output_dir / model_stem
        run_idx = 1
        while output_dir.exists():
            output_dir = base_output_dir / f"{model_stem}_run{run_idx:02d}"
            run_idx += 1
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving results to {output_dir}")
    
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
        csv_path = dir_output_dir / f"{img_dir.name}_crops_metadata.csv"
        processed_images = set()
        # If run_number is provided and CSV exists, skip already processed images
        if run_number is not None and csv_path.exists():
            try:
                df_existing = pd.read_csv(csv_path)
                processed_images = set(Path(p).stem for p in df_existing['original_image'])
                dir_crop_records = df_existing.to_dict('records')
                print(f"Skipping {len(processed_images)} images already in {csv_path}")
            except Exception as e:
                print(f"Warning: Could not read existing CSV {csv_path}: {e}")
        for image_path in tqdm(image_files, desc=f"Processing {img_dir.name}"):
            if run_number is not None and image_path.stem in processed_images:
                continue
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
                    img = cv2.imread(str(image_path))
                    for idx, (class_name, box, det_conf) in enumerate(zip(result_patched.filtered_classes_names, result_patched.filtered_boxes, result_patched.filtered_confidences)):
                        x1, y1, x2, y2 = map(int, box)
                        crop = img[y1:y2, x1:x2]
                        conf_int = int(round(float(det_conf) * 100))
                        class_dir = dir_output_dir / class_name
                        class_dir.mkdir(exist_ok=True, parents=True)
                        crop_filename = class_dir / f"{conf_int:03d}_{image_path.stem}_det_{idx}_{class_name}.jpg"
                        cv2.imwrite(str(crop_filename), crop)
                        dir_crop_records.append({
                            'crop_path': str(crop_filename.resolve()),
                            'original_image': str(image_path.resolve()),
                            'class_name': class_name,
                            'confidence': float(det_conf),
                            'x1': x1,
                            'y1': y1,
                            'x2': x2,
                            'y2': y2,
                            'no_detections': False
                        })
                else:
                    # No detections, mark as processed
                    dir_crop_records.append({
                        'crop_path': '',
                        'original_image': str(image_path.resolve()),
                        'class_name': '',
                        'confidence': '',
                        'x1': '',
                        'y1': '',
                        'x2': '',
                        'y2': '',
                        'no_detections': True
                    })
                del element_crops
                #print(f"Processed {image_path.name} in {end_time - start_time:.2f}s")
            except Exception as e:
                print(f"Image problems {image_path}: {e}")
            # Save/update CSV after each image
            if dir_crop_records:
                df = pd.DataFrame(dir_crop_records)
                df.to_csv(csv_path, index=False)
    print(f"Wrote crop metadata to {csv_path}")

@app.command('expand')
def expand_crops(
    csv_file: Path = typer.Option(..., help="Path to crops metadata CSV file or directory containing such CSV files"),
    expand_pct: float = typer.Option(20.0, help="Percentage to expand each crop bounding box (e.g. 20 for 20%)"),
    output_dir: Path = typer.Option(None, help="Directory to save expanded crops (default: sibling 'expanded_crops' next to CSV)"),
    fixed_size: int =typer.Option(0, help="If set, use fixed width,height for all crops instead of percentage-based expansion")

):
    """
    Expand each crop bounding box by a percentage and save the expanded crops.
    """
    import cv2
    import pandas as pd
    from pathlib import Path
    if output_dir is not None:
        output_dir = Path(output_dir)
    if csv_file.is_dir():
        input_dir = csv_file
        csv_files = list(csv_file.rglob('*_crops_metadata.csv'))
        if not csv_files:
            print(f"No *_crops_metadata.csv files found in directory {csv_file}")
            return
    else:
        csv_files = [csv_file]
    for cf in csv_files:
        
        df = pd.read_csv(cf).dropna(subset=['crop_path'])
        if df.empty:
            print(f"No records in {cf}, skipping.")
            continue
        #load the first image to get dimensions
        orig_img_path = str(df['original_image'].iloc[0])
        img = cv2.imread(orig_img_path)
        if img is None:
            print(f"Warning: Could not read {orig_img_path}")
            continue
        h, w = img.shape[:2]
        expand_pct = float(expand_pct)

        # Group by original image
        # apply the expaionsion to each row with lambda function
        def expand_row(row):
            if row.get('no_detections', False) or not row['crop_path']:
                return row
            orig_img_path = Path(row['original_image'])
            h, w = img.shape[:2]
            x1, y1, x2, y2 = map(int, [row['x1'], row['y1'], row['x2'], row['y2']])
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            bw = x2 - x1
            bh = y2 - y1
            if fixed_size > 0:
                expand_w = fixed_size
                expand_h = fixed_size
                # Adjust center so crop stays within image bounds
                half_w = expand_w / 2
                half_h = expand_h / 2
                cx = min(max(cx, half_w), w - half_w)
                cy = min(max(cy, half_h), h - half_h)
            else:
                expand_w = bw * (1 + expand_pct / 100)
                expand_h = bh * (1 + expand_pct / 100)
            new_x1 = int(round(max(0, cx - expand_w / 2)))
            new_y1 = int(round(max(0, cy - expand_h / 2)))
            new_x2 = int(round(min(w, cx + expand_w / 2)))
            new_y2 = int(round(min(h, cy + expand_h / 2)))
            # For fixed_size, force output size
            if fixed_size > 0:
                if new_x2 - new_x1 != fixed_size:
                    if new_x1 == 0:
                        new_x2 = min(w, new_x1 + fixed_size)
                    else:
                        new_x1 = max(0, new_x2 - fixed_size)
                if new_y2 - new_y1 != fixed_size:
                    if new_y1 == 0:
                        new_y2 = min(h, new_y1 + fixed_size)
                    else:
                        new_y1 = max(0, new_y2 - fixed_size)
            row['new_x1'] = new_x1
            row['new_y1'] = new_y1
            row['new_x2'] = new_x2
            row['new_y2'] = new_y2
            row['width'] = new_x2 - new_x1
            row['height'] = new_y2 - new_y1
            # Mirror original image's relative path under output_dir
            if output_dir is not None:
                # Find relative path from input root to original image
                rel_path = Path(row['crop_path']).parent.relative_to(input_dir.parent.parent)
                base_out = output_dir / rel_path.parent
            else:
                base_out = Path(row['crop_path']).parent.parent
            class_dir = base_out / (row['class_name'] if row['class_name'] else 'unknown')
            orig_stem = orig_img_path.stem
            shard = orig_stem[:2]
            if fixed_size > 0:
                crop_name = f"expanded_{fixed_size}px_{Path(row['crop_path']).name}"
                row['crop_path'] = class_dir / f'expanded_crops_{fixed_size}' / shard / crop_name
            else:
                crop_name = f"expanded_{expand_pct:.0f}pct_{Path(row['crop_path']).name}"
                row['crop_path'] = class_dir / f'expanded_crops_{expand_pct:.0f}' / shard / crop_name
            return row
        df = df.apply(expand_row, axis=1)
        if fixed_size > 0:
            df.to_csv(cf.with_name(cf.stem + f'{fixed_size}_with_expanded_boxes_{fixed_size}px.csv'), index=False)
        else:
            df.to_csv(cf.with_name(cf.stem + f'_with_expanded_boxes_{expand_pct:.0f}pct.csv'), index=False)
        groups = list(df.groupby('original_image'))
        # Prepare all crop jobs (one per crop, not per image)
        crop_jobs = []
        for image_path, crops in groups:
            crops = crops[crops['no_detections'] == False]
            if crops.empty:
                continue
            crop_jobs.append((crops, image_path))
        def process_crop(args):
            thumbs, image_path = args
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"Warning: Could not read {image_path}")
                return None
            h, w = img.shape[:2]
            orig_stem = Path(image_path).stem
            orig_shard = Path(orig_stem[:2])
            for idx, row in thumbs.iterrows():
                x1, y1, x2, y2 = map(int, [row['new_x1'], row['new_y1'], row['new_x2'], row['new_y2']])
                expanded_crop = img[y1:y2, x1:x2]
                output_path = Path(row['crop_path'])
                try:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    print(f"[ERROR] Could not create directory {output_path.parent}: {e}")
                    continue
                for attempt in range(3):
                    try:
                        success = cv2.imwrite(str(output_path), expanded_crop)
                        if not success:
                            raise OSError(f"cv2.imwrite failed for {output_path}")
                        break
                    except Exception as e:
                        print(f"[ERROR] Failed to write {output_path} (attempt {attempt+1}/3): {e}")
                        if attempt == 2:
                            continue
                        print(f"Shape for {output_path}: {expanded_crop.shape}")
            return image_path
        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as crop_executor:
            list(tqdm(crop_executor.map(process_crop, crop_jobs), total=len(crop_jobs), desc=f"Expanding crops for {cf.name}"))
    print(f"Expanded crops saved to {output_dir}")

@app.command('cuda')
def check_cuda():
    """Check if CUDA is available and print device info."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"CUDA is available: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA is NOT available.")
    except ImportError:
        print("PyTorch is not installed.")


app.command('detect-transformer')(detect_transformer)



if __name__ == "__main__":
    app()
