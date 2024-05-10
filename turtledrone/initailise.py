import os
import glob
import doit
import glob
import os
import numpy as np
import pandas as pd
import numpy as np
import geopandas as gp
from shapely.geometry import MultiPoint
from turtledrone.utils.utils import convert_wgs_to_utm
import turtledrone.config as config
from doit import create_after
import typer

def task_set_up():
    config.read_config()


def task_make_area_list():
    """
        look for shape files that have the areas of interest in 

        path to survey area shape files is in config file:
            surveyarea : "{CATALOG_DIR}/survey_area/"
        example:
            TULKI_TulkiBeach_SurveyArea.shp
            {short_code}_{long_name}_SurveyArea

        target is sent to processing directory
            surveyareas.csv
    """
    def split(path):
        print(path)
        keys =os.path.splitext(os.path.basename(path))[0].split('_')
        return {'SurveyCode':keys[0],'SurveyLongName':keys[1],'Type':keys[2],'File':path}
    
    def process_area_list(dependencies, targets):
        areas = pd.DataFrame.from_records([split(shape) for shape in dependencies])
        os.makedirs(os.path.dirname(targets[0]),exist_ok=True)
        areas.to_csv(targets[0],index=False)                
        
    file_dep =list(config.geturl('surveyarea').resolve().rglob('*.shp'))
    target = config.geturl('process') / 'surveyareas.csv'
    if file_dep:
        return {
            'actions':[process_area_list],
            'file_dep':file_dep,
            'targets':[target],
            'clean':True,
        } 
        
@create_after(executed='make_area_list')         
def task_make_grids():
    """
        make a grid that can be used to extract tiles from the images
    """
    def process_grid(dependencies, targets,gridsize=10):
        area = gp.read_file(dependencies[0])
        if len(area.dropna())>0:
            utmcode = convert_wgs_to_utm(area.iloc[0].geometry.exterior.coords.xy[0][0],area.iloc[0].geometry.exterior.coords.xy[1][0])
            crs = f'epsg:{utmcode}' 
            polygon = area.to_crs(crs).iloc[0].geometry
            eastings =polygon.exterior.coords.xy[0]
            northings =polygon.exterior.coords.xy[1]
            easting =np.arange(np.min(eastings) -np.min(eastings) % gridsize,np.max(eastings) -np.max(eastings) % gridsize,gridsize)
            northing=np.arange(np.min(northings) -np.min(northings) % gridsize,np.max(northings) -np.max(northings) % gridsize,gridsize)
            xi,yi = np.meshgrid(easting,northing)
            points =  MultiPoint(list(zip(xi.ravel(),yi.ravel())))
            p =points.intersection(polygon)
            d = {'Grid': [f'{gridsize}m'], 'geometry': [p]}
            df =gp.GeoDataFrame(d, crs=crs)
            df.to_file(targets[0])    
        else:
            raise typer.Abort(f'No ploygon inside file {dependencies[0]}')

    file_dep = config.geturl('surveyarea').resolve().rglob('*_AOI.shp')
    for file in file_dep:
        target = file.parent / file.name.replace('_AOI.shp','_Grid.shp')
        yield {
            'name':target,
            'actions':[(process_grid,[],{'gridsize':int(config.cfg['gridsize'])})],
            'file_dep':[file],
            'targets':[target],
            'clean':True,
        }         
def run():
    import sys
    from doit.cmd_base import ModuleTaskLoader, get_loader
    from doit.doit_cmd import DoitMain
    DOIT_CONFIG = {'check_file_uptodate': 'timestamp',"continue": True}
    #print(globals())
    DoitMain(ModuleTaskLoader(globals())).run(sys.argv[1:])


if __name__ == '__main__':
    run()