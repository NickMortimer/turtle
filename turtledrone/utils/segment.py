from ultralytics import SAM
import cv2
import json
import numpy as np
from pathlib import Path
import doit 
import turtledrone.config as config
import pandas as pd
import os
from turtledrone.utils.load_shapes import loadshapes
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils import TQDM




def task_set_up():
    config.read_config()

# link to origonal tile
# create new json file with the segmented outline
def task_process_tiles():
    def process_tiles(dependencies, targets,sam_model="sam_b.pt",device=None):
        df = pd.read_csv(dependencies[0])
        df['source_tile'] =df.image_path.apply(lambda x: config.geturl('output') / 'turtles' / Path(x).parts[1] / 'slices' / ('_'.join(Path(x).stem.split('_')[:-1])+'.png'))
        df['turtle_counts'] = df.groupby('source_tile')['labels'].transform('count')
        turtle3 =df.loc[(df.labels=='turtle_3') & (df.turtle_counts==1)]
        output =config.geturl('output') / 'turtles' / 'labeled' / 'turtle_3'
        output.mkdir(exist_ok=True,parents=True)
        sam_model = SAM(sam_model)
        for index,row in TQDM(turtle3.iterrows(), total=len(turtle3), desc="Generating segment labels"):
            source_tile = Path(row.source_tile)
            destination =output / source_tile.name
            if not destination.exists():
                os.link(source_tile.absolute(),destination)
            with open(source_tile.with_suffix('.json')) as f:
                data = json.load(f)
            for label in data['shapes']:
                if label['shape_type'] == 'rectangle':
                    boxes =np.hstack(np.array(label['points']))    
                    if len(boxes) !=4:  # skip empty labels
                        continue
                    im = cv2.imread(str(source_tile.absolute()))
                    sam_results = sam_model(im, bboxes=boxes, verbose=False, save=False, device=device)
                    newlabel =label.copy()
                    newlabel["points"] = sam_results[0].masks.xy[0].tolist()
                    newlabel['label'] = 'turtle_3'
                    newlabel['shape_type'] = 'polygon'
                    data['shapes'].append(newlabel)
            with open(destination.with_suffix('.json'), "w") as f:
                json.dump(data, f, indent=2)    
            

    file_dep = config.geturl('output') / 'turtles' / 'digikam.csv'
    return {
        'file_dep' : [file_dep],
        'actions':[process_tiles],
        'clean':True,
        'uptodate':[True],
    } 


if __name__ == '__main__':
    import doit
    DOIT_CONFIG = {'check_file_uptodate': 'timestamp'}
    #print(globals())
    doit.run(globals()) 