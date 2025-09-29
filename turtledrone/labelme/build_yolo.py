import os
import json
import glob
import shutil
from PIL import Image
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

def convert_labelme_to_yolo_seg(json_folder, output_dir, train_ratio=0.8,valid_class=['turtle_3']):
    json_files = sorted(glob.glob(os.path.join(json_folder, "*.json")))
    data = pd.DataFrame(json_files,columns=['SourceFile'])
    data['FileName'] = data.SourceFile.apply(lambda x: Path(x).name)
    data[['Area','TimeStamp']]=data.FileName.str.extract(r'(?P<Make>.+?)_(?P<Camera>.+?)_(?P<Country>.+?)_(?P<Area>.+?)_(?P<Time>.+?)_')[['Area','Time']]
    data['TimeStamp'] =pd.to_datetime(data.TimeStamp)
    data['Date']=data.TimeStamp.dt.date
    data = data.sort_values('TimeStamp')
    # Calculate time difference from previous row
    data['TimeDiff'] = data['TimeStamp'].diff()
    data['Seen'] = data.TimeStamp < pd.to_datetime('2023-10-01')
    # Start new group if gap > 10 seconds
    data['Burst'] = (data['TimeDiff'] > pd.Timedelta(seconds=10)).cumsum()
    # Select from seen groups

    seen =data.loc[data.Seen].Burst.unique()
    train_files, val_files = train_test_split(seen, train_size=train_ratio, random_state=42)

    splits = {"train": train_files, "val": val_files}
    class_names = {}
    class_id = 0

    # Create output dirs
    for split in splits:
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

    for split, files in splits.items():
        for burst in tqdm(files, desc=f"Processing {split}"):
            for json_file in data.loc[data.Burst==burst,'SourceFile'].to_list():
                with open(json_file, 'r') as f:
                    json_data = json.load(f)

                img_name = json_data["imagePath"]
                img_path = os.path.join(os.path.dirname(json_file), img_name)
                if not os.path.exists(img_path):
                    print(f"⚠️ Missing image file: {img_path}")
                    continue

                img = Image.open(img_path)
                w, h = img.size

                # Copy image
                out_img_path = os.path.join(output_dir, "images", split, img_name)

                # Create label lines
                label_lines = []
                for shape in json_data["shapes"]:
                    label = shape["label"]
                    if label in valid_class:
                        if label not in class_names:
                            class_names[label] = class_id
                            class_id += 1
                        cls_id = class_names[label]

                        points = shape["points"]
                        flat_points = []
                        for x, y in points:
                            x_norm = x / w
                            y_norm = y / h
                            flat_points.extend([x_norm, y_norm])

                        if len(flat_points) >= 6:  # Must be at least 3 points (6 values)
                            label_lines.append(f"{cls_id} " + " ".join(f"{p:.6f}" for p in flat_points))

                    # Write label file
                if len(label_lines)>0:
                    label_filename = os.path.splitext(img_name)[0] + ".txt"
                    label_path = os.path.join(output_dir, "labels", split, label_filename)
                    with open(label_path, "w") as f:
                        f.write("\n".join(label_lines))
                    shutil.copy(img_path, out_img_path)
        data.to_csv(Path(output_dir) / 'allocation.csv',index=False)

    # Write dataset.yaml
    yaml_path = os.path.join(output_dir, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {os.path.abspath(output_dir)}\n")
        f.write(f"train: images/train\n")
        f.write(f"val: images/val\n")
        f.write("names:\n")
        for name, idx in sorted(class_names.items(), key=lambda x: x[1]):
            f.write(f"  {idx}: {name}\n")

    print(f"✅ Done! YOLOv8-ready dataset created at: {output_dir}")
    print(f"📄 dataset.yaml created at: {yaml_path}")

def create_unseen(data_file,output_dir,valid_class=['turtle_3']):
    data = pd.read_csv(data_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    splits = {"train": [], "val": data.loc[~data.Seen,'SourceFile'].to_list()}
    class_names = {}
    class_id = 0

    # Create output dirs
    for split in splits:
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

    for split, files in splits.items():
        for json_file in tqdm(files, desc=f"Processing {split}"):
            with open(json_file, 'r') as f:
                json_data = json.load(f)

            img_name = json_data["imagePath"]
            img_path = os.path.join(os.path.dirname(json_file), img_name)
            if not os.path.exists(img_path):
                print(f"⚠️ Missing image file: {img_path}")
                continue

            img = Image.open(img_path)
            w, h = img.size

            # Copy image
            out_img_path = os.path.join(output_dir, "images", split, img_name)

            # Create label lines
            label_lines = []
            for shape in json_data["shapes"]:
                label = shape["label"]
                if label in valid_class:
                    if label not in class_names:
                        class_names[label] = class_id
                        class_id += 1
                    cls_id = class_names[label]

                    points = shape["points"]
                    flat_points = []
                    for x, y in points:
                        x_norm = x / w
                        y_norm = y / h
                        flat_points.extend([x_norm, y_norm])

                    if len(flat_points) >= 6:  # Must be at least 3 points (6 values)
                        label_lines.append(f"{cls_id} " + " ".join(f"{p:.6f}" for p in flat_points))

                # Write label file
                label_filename = os.path.splitext(img_name)[0] + ".txt"
                label_path = os.path.join(output_dir, "labels", split, label_filename)
                with open(label_path, "w") as f:
                    f.write("\n".join(label_lines))
                shutil.copy(img_path, out_img_path)
    # Write dataset.yaml
    yaml_path = os.path.join(output_dir, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {os.path.abspath(output_dir)}\n")
        f.write(f"train: images/train\n")
        f.write(f"val: images/val\n")
        f.write("names:\n")
        for name, idx in sorted(class_names.items(), key=lambda x: x[1]):
            f.write(f"  {idx}: {name}\n")
convert_labelme_to_yolo_seg('/home/mor582/turtles/labeled/turtle_3/','/home/mor582/turtles/yolo')

#create_unseen('/home/mor582/turtles/yolo/allocation.csv','/home/mor582/turtles/yolo_unseen')