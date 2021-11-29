from genericpath import exists
import os
import glob
import doit
import glob
import os
import numpy as np
from pandas.core.arrays.integer import Int64Dtype
import yaml
import pandas as pd
from doit import get_var
from doit.tools import run_once
from doit import create_after
import numpy as np
import plotly
import plotly.express as px
import geopandas as gp
from geopandas.tools import sjoin
from drone import P4rtk
from read_rtk import read_mrk
from pyproj import Proj 
from doit.tools import check_timestamp_unchanged
import shutil
from shapely.geometry import Polygon
import shapely.wkt
from shapely.geometry import MultiPoint
import xarray as xr
import rasterio as rio

def task_make_zarr():
    def process_zarr(dependencies, targets,cfg):
        def cut_tile(item,easting,northing,pix,x,y,pixeldim,imageheight,imagewidth,squaresize=32):
            ds = xr.Dataset()
            if (y+squaresize/2 < imageheight) & ( y-squaresize/2>0) & (x-squaresize/2>0) & (x+squaresize/2 < imagewidth):
                ds['image'] = xr.DataArray(pix[:,(y-squaresize//2):(y+squaresize//2),(x-squaresize//2):(x+squaresize//2)],
                                        dims=['rgb','dy','dx'],coords={'rgb':['r','g','b'],'dy':pixeldim,'dx':pixeldim})
                ds.coords['easting'] = easting
                ds.coords['northing'] =northing
                ds.coords['imagenumber'] = item.Counter
            return  ds
        
        
        def process_row(item,points):
            drone.setdronepos(item.Easting,item.Northing,item.RelativeAltitude,
                             (90+item.GimbalPitchDegree)*-1,item.GimbalRollDegree,item.GimbalYawDegree)
            img = xr.open_rasterio(item.ImagePath) 
            pixeldim=np.arange(-256,256)
            result =[]
            for point in points:
                imx,imy=drone.realwordtocamera(point[0],point[1])
                tile = cut_tile(item,point[0],point[1],img,int(imx),int(imy),pixeldim,item.ImageHeight,item.ImageWidth)
                if tile.variables:
                    #gcps = [rio.control.GroundControlPoint(row=0, col=0, x=100, y=1169) ]
                    #drone.jpegtoreal()
                    result.append(tile)
            if result:
                result=xr.concat(result,dim='tile')
            return result
        
        surveyfile = list(filter(lambda x: '.csv' in x, dependencies))[0]
        gridfile = list(filter(lambda x: '.shp' in x, dependencies))[0]
        grid =gp.read_file(gridfile)
        data = pd.read_csv(surveyfile,parse_dates=['TimeStamp'])
        sample = data.sample(25)
        sample['idx'] =sample.index
        data = data[data.index.isin(np.hstack(sample.idx.apply(lambda x:range(x-2,x+3))))]
        n =data.NewName.str.split('_',expand=True)
        data['ImagePath']=cfg['paths']['output']+'/'+n[2]+'/'+data.SurveyId+'/'+data.NewName
        crs = f'epsg:{int(data["UtmCode"].min())}'
        gdf = gp.GeoDataFrame(data, geometry=data.ImagePolygon.apply(shapely.wkt.loads),crs=crs)
        dewarp = pd.to_numeric(cfg['survey']['dewarp'] )
        drone =P4rtk(dewarp,crs)
        gridp = MultiPoint([(p.x,p.y) for p in grid.iloc[0].geometry])
        zarr = []
        for index,row in gdf.iterrows():
            intersetion = gridp.intersection(row.geometry.buffer(-10))
            if intersetion.geom_type=='Point':
                if intersetion.coords:
                    result=process_row(row,[(intersetion.x,intersetion.y)])
                    zarr.append(result)
            elif intersetion.geom_type=='MultiPoint':
                points=[(p.x,p.y) for p in intersetion]
                result=process_row(row,points)
                zarr.append(result)
        output =list(filter(lambda x: x,zarr))
        if output:
            output = xr.concat(output,dim='tile')
            output =output.chunk({'tile':20,'dx':512, 'dy':512,'rgb':3})
            output.to_zarr(targets[0])
            
            

            
    config = {"config": get_var('config', 'NO')}
    basepath = os.path.dirname(config['config'])
    with open(config['config'], 'r') as ymlfile:
        cfg = yaml.load(ymlfile, yaml.SafeLoader)
    file_dep = glob.glob(os.path.join(basepath,cfg['paths']['process'],'*_survey.csv'))
    areas = pd.read_csv(os.path.join(basepath,cfg['paths']['process'],'surveyareas.csv'),index_col='SurveyCode')
    areas = areas.loc[areas.Type=='Grid']
    for file in file_dep:
        surveyarea =os.path.basename(file).split('_')[1]
        if surveyarea in areas.index:
            file_dep =[file,areas.loc[surveyarea].File]
            target = os.path.join(basepath,cfg['paths']['zarrpath'],os.path.basename(file).replace('_survey.csv','.zarr'))
            yield {
                'name':file,
                'actions':[(process_zarr, [],{'cfg':cfg})],
                'file_dep':file_dep,
                'targets':[target],
                'uptodate': [True],
                'clean':True,
            }    

from sklearn.cluster import SpectralClustering
def task_sample_tiles():
    def process_tiles(dependencies, targets,cfg):
        tiles = xr.open_dataset(dependencies[0],engine='zarr')
        tiles.image.attrs['nodatavals'] =[0,0,0]
        tiles =tiles.rename({'dx':'x','dy':'y'})
        spectra =tiles.image.mean(dim=['x','y'])
        clustering = SpectralClustering(n_clusters=4,assign_labels='discretize',random_state=0).fit(spectra)
        os.makedirs(targets[0],exist_ok=True)
        for index in range(0,len(tiles.image)):
            tiles.image[index].rio.to_raster(os.path.join(targets[0],f'{clustering.labels_[index]:02d}_{index:04d}.JPG'), compress='zstd', zstd_level=1, num_threads='all_cpus', tiled=True, dtype='uint8', predictor=2)       
    config = {"config": get_var('config', 'NO')}
    basepath = os.path.dirname(config['config'])
    with open(config['config'], 'r') as ymlfile:
        cfg = yaml.load(ymlfile, yaml.SafeLoader)
    file_dep = glob.glob(os.path.join(os.path.join(basepath,cfg['paths']['zarrpath'],'*.zarr')))
    for file in file_dep:
        target =os.path.join(basepath,cfg['paths']['output'],'train','tiles',os.path.splitext(os.path.basename(file))[0])
        yield {
            'name':file,
            'actions':[(process_tiles, [],{'cfg':cfg})],
            'file_dep':[file],
            'targets':[target],
            'uptodate': [True],
            'clean':True,
        }  
         
if __name__ == '__main__':
    import doit
    DOIT_CONFIG = {'check_file_uptodate': 'timestamp'}
    #print(globals())
    doit.run(globals())