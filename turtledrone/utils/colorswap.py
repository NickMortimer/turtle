import cv2
from pathlib import Path
import numpy as np

def fix_bgr_saved_as_rgb(image_path: Path):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Could not read {image_path}")
        return

    # Swap R and B channels using numpy (BGR → RGB or vice versa)
    img_fixed = img[..., ::-1]  # Simple channel reversal

    cv2.imwrite(str(image_path), img_fixed)
    print(f"Fixed: {image_path}")


image_files = Path('/home/mor582/turtles/yolo').rglob('*.png')

for file in image_files:
    fix_bgr_saved_as_rgb(file)

