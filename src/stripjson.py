import json
import glob
from queue import Empty



files = glob.glob('d:/drone/**/*.json',recursive=True)
for file in files:
    with open(file) as f:
        data = json.load(f)
    if 'imageData' in data:
        data.pop('imageData')
        with open(file, 'w',encoding="utf-8") as f:
            json.dump(data, f)



