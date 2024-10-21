from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import numpy as np
from pathlib import Path
import json
import pandas as pd
import shutil
import numpy as np
import ast
import os




input_files = list(Path('/media/mor582/Storage/Nick/surveys/test/TULKI_20231103T0817/').glob('DJIP*.json'))

# def filter_big(shape):
#     if 'yolo' in shape['label']:
#         p =np.array(shape['points'])
#         box_len =np.linalg.norm(p[0] - p[1])
#         #2cm pixels
#         return (box_len>5) and (box_len <100)
#     return True


# for json_file in input_files:
#     if json_file.exists():
#         with open(json_file, "r") as read_file:
#             data = json.load(read_file)
#             data['shapes'] = list(filter(filter_big,data['shapes']))
#         with open(json_file, "w") as outfile:
#             outfile.write(json.dumps(data, indent=2))


turtle = Path('/media/mor582/Storage/Nick/surveys/test/TULKI_20231103T0817/turtles')
turtle.mkdir(exist_ok=True)

def find(shapes):
    for shape in shapes:
        if 'turtle_surface' in shape['label']:
            return 'turtle_surface'
    for shape in shapes:
        if 'turtle_jbs' in shape['label']:
            return 'turtle_jbs'
    for shape in shapes:
        if 'turtle_deep' in shape['label']:
            return 'turtle_deep'
       

for json_file in input_files:
    if json_file.exists():
        with open(json_file, "r") as read_file:
            data = json.load(read_file)
            dest =find(data['shapes'])
            if dest:
                dest_path =turtle / dest / json_file.name
                dest_path.parent.mkdir(exist_ok=True)
                if not dest_path.exists():
                    os.link(json_file,dest_path / json_file.name)
                    os.link(json_file.with_suffix('.JPG'),dest_path / json_file.with_suffix('.JPG').name)
