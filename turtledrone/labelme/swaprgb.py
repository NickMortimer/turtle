from pathlib import Path
import cv2

def swap_image_color_channels_in_slices(root_dir):
    root_dir = Path(root_dir)
    slices_dirs = list(root_dir.rglob('slices'))

    for slices_dir in slices_dirs:
        for img_path in slices_dir.glob('*'):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                # Read the image (OpenCV loads as BGR)
                image_bgr = cv2.imread(str(img_path))
                # if image_bgr is None:
                #     print(f"Skipping unreadable image: {img_path}")
                #     continue

                # Swap channels (BGR <-> RGB)
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_RGB2BGR)

                # Overwrite the image
                cv2.imwrite(str(img_path), cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB))  # writes in BGR

                print(f"Updated: {img_path}")

# Example usage
swap_image_color_channels_in_slices("/home/mor582/turtles")