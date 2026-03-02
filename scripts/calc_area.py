from pathlib import Path
import pandas as pd
import json
import numpy as np
from shapely.geometry import Polygon
import typer
app = typer.Typer()

def loadshapes(file):
    print(file)
    lines =[]
    with open(file, "r") as read_file:
        data = json.load(read_file)

    #data = json.loads(''.join(lines).replace("\n", "").replace("'", '"').replace('u"', '"'))
    data =pd.DataFrame(data['shapes'])
    data['FilePath'] =str(file)             
    return(data)

def calc_shape_area(x):
    return Polygon(np.squeeze(np.array(x))).area



@app.command('area')
def calc_area(directory_path: Path = typer.Argument(..., help="source directory for json files"),
               destination: Path = typer.Argument(..., help="file path for results file")):
    files = list(directory_path.glob('*.json'))
    data = pd.concat([loadshapes(file) for file in files])
    data['area'] =data.points.apply(calc_shape_area)
    data['Key'] = data.FilePath.str.extract(r'(?P<QuadId>\d{2}_\d{5}_Q\d)')
    totals =data.groupby(['Key','label'])['area'].sum()
    totals.to_csv(destination)

if __name__ == "__main__":
    app()