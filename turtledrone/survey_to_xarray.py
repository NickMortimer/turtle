import os
import re
import doit
import numpy as np
import pandas as pd
import numpy as np
from osgeo import gdal, osr

from shutil import which
from doit import create_after
import re
import json   
from turtledrone.utils.utils import convert_wgs_to_utm
from pathlib import Path
from PIL import Image
import concurrent.futures
from tqdm import tqdm

def task_survey_to_xarray():
    """
    Convert survey csv files to xarray datasets
    """
    def process_survey(dependencies, targets):
        import xarray as xr
        from pyproj import Transformer,CRS
        from turtledrone.utils.camera import getpoly


        def _to_snake(name: str) -> str:
            s = re.sub(r'[\s\-]+', '_', name)
            s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', s)
            s = re.sub(r'([a-z\d])([A-Z])', r'\1_\2', s)
            s = re.sub(r'__+', '_', s)
            return s.lower()
        
        transformer = Transformer.from_crs("EPSG:4979", "EPSG:4326+3855", always_xy=True)
        drone =pd.read_csv(dependencies[0],parse_dates=['TimeStamp'])
        if ('DewarpData' in drone.columns) and (~drone['DewarpData'].isna().max()):
            drone[['CalibrationDate','CalibratedFocalLengthX','CalibratedFocalLengthY','CalibratedOpticalCenterX','CalibratedOpticalCenterY',
                        'K1','K2','P1',"P2","K3"]] = drone['DewarpData'].str.split(r'[;,]',expand=True)
            drone[['CalibratedFocalLengthX','CalibratedFocalLengthY','CalibratedOpticalCenterX','CalibratedOpticalCenterY',
                        'K1','K2','P1',"P2","K3"]] = drone[['CalibratedFocalLengthX','CalibratedFocalLengthY','CalibratedOpticalCenterX','CalibratedOpticalCenterY',
                        'K1','K2','P1',"P2","K3"]].astype(float)
            drone['CalibratedOpticalCenterX'] = (drone['ImageWidth']/2)+drone['CalibratedOpticalCenterX']
            drone['CalibratedOpticalCenterY'] = (drone['ImageHeight']/2)+drone['CalibratedOpticalCenterY']
        else:
            pass
        
        drone.rename(columns={c: _to_snake(c) for c in drone.columns}, inplace=True)
        _, _, H = transformer.transform(
            drone["longitude"], 
            drone["latitude"], 
            drone["ellipsoide_hight_mrk"]
        )
        drone["orthometric_height_egm2008"] = H
        drone["geoid_separation_n"] = drone["ellipsoide_hight_mrk"] - H
        ds =xr.Dataset.from_dataframe(drone[config.cfg['export_nc_columns']])
        ds = ds.rename_vars({'new_name': 'image_name'})
        # add the UTM Zone to the attribues
        def utm_crs_from_lonlat(lon, lat):
            zone_number = int((lon + 180) / 6) + 1
            hemisphere = 'north' if lat >= 0 else 'south'
            epsg_code = 32600 + zone_number if hemisphere == 'north' else 32700 + zone_number
            return f"EPSG:{epsg_code}"
        
        ds['camera_pitch'] = ds['camera_pitch']+90 # adjust for DJI convention
        # pick a single zone for the whole survey
        ds.attrs['crs'] = 'EPSG:4326+3855'
        ds.attrs['utm_crs'] = f'EPSG:{drone.utm_code.iloc[0]}'
        ds.attrs['dewarp_flag'] = bool(drone.dewarp_flag.iloc[0])
        ds.attrs['image_width'] = int(drone.image_width.iloc[0])
        ds.attrs['image_height'] = int(drone.image_height.iloc[0])
        if ds.attrs['dewarp_flag']:
            camera_int ={'calibrated_focal_length_x': drone.get('calibrated_focal_length', pd.Series([None])).iloc[0],
                                        'calibrated_focal_length_y': drone.get('calibrated_focal_length', pd.Series([None])).iloc[0],
                                        'calibrated_optical_center_x': drone.image_width.iloc[0]/2,
                                        'calibrated_optical_center_y': drone.image_height.iloc[0]/2,
                                        'distortion_coefficients': {
                                            'k1': 0,
                                            'k2': 0,
                                            'p1': 0,
                                            'p2': 0,
                                            'k3': 0,
                                        }
                        }

        else:
            camera_int ={'calibrated_focal_length_x': drone.get('calibrated_focal_length_x', pd.Series([None])).iloc[0],
                                        'calibrated_focal_length_y': drone.get('calibrated_focal_length_y', pd.Series([None])).iloc[0],
                                        'calibrated_optical_center_x': drone.get('calibrated_optical_center_x', pd.Series([None])).iloc[0],
                                        'calibrated_optical_center_y': drone.get('calibrated_optical_center_y', pd.Series([None])).iloc[0],
                                        'distortion_coefficients': {
                                            'k1': drone.get('k1', pd.Series([None])).iloc[0],
                                            'k2': drone.get('k2', pd.Series([None])).iloc[0],
                                            'p1': drone.get('p1', pd.Series([None])).iloc[0],
                                            'p2': drone.get('p2', pd.Series([None])).iloc[0],
                                            'k3': drone.get('k3', pd.Series([None])).iloc[0],
                                        }
                        }
        
        center_lon = ds['longitude'].mean().item()
        center_lat = ds['latitude'].mean().item()
        crs_geo = CRS.from_user_input(ds.attrs['crs'])
        horizontal = CRS.from_proj4(
            f"+proj=stere +lat_0={center_lat} +lon_0={center_lon} +k=1 "
            "+x_0=0 +y_0=0 +units=m +ellps=WGS84"
        )
        vertical = CRS.from_epsg(3855)
        ds.attrs['horizontal_crs'] = horizontal.to_string()
        ds.attrs['vertical_crs'] = vertical.to_string()
        transformer = Transformer.from_crs("EPSG:4326+3855", horizontal, always_xy=True)
        x_vals, y_vals = transformer.transform(ds['longitude'], ds['latitude'])
        dims = ds["longitude"].dims  # e.g., ('TimeStamp',)
        ds["x"] = (dims, x_vals)
        ds["y"] = (dims, y_vals)
        # itterate over each row to calculate image polygon
        image_polygons = []
        import  cameratransform as ct
        camera = ct.Camera(
            ct.RectilinearProjection(
                focallength_x_px=camera_int['calibrated_focal_length_x'],
                focallength_y_px=camera_int['calibrated_focal_length_y'],
                center_x_px=camera_int['calibrated_optical_center_x'],
                center_y_px=camera_int['calibrated_optical_center_y']
            ),
            lens=ct.BrownLensDistortion(
                camera_int['distortion_coefficients']['k1'],
                camera_int['distortion_coefficients']['k2'],
                camera_int['distortion_coefficients']['k3']
            )
        )

        for index, row in ds.to_dataframe().iterrows():
            camera.orientation = ct.SpatialOrientation(elevation_m=row['relative_altitude'],
                                                        heading_deg=row['camera_yaw'],
                                                        tilt_deg=row['camera_pitch'],
                                                        roll_deg=row['camera_roll'], 
                                                        pos_x_m=row['x'], 
                                                        pos_y_m=row['y']) 
            poly = getpoly(
                camera,
                image_width=ds.attrs['image_width'],
                image_height=ds.attrs['image_height'],
                perside=20,
                crop=0,
                corners=0
            )
            image_polygons.append(poly.wkt)
        ds["image_polygon"] = (dims, image_polygons)

        # Helper to convert numpy scalars to Python scalars for JSON
        def _json_default(o):
            import numpy as np
            if isinstance(o, (np.floating, np.integer)):
                return o.item()
            return o
        ds.attrs['camera_intrinsics']=json.dumps(camera_int,ensure_ascii=False, default= _json_default)
        ds.to_netcdf(targets[0])       
        
    from turtledrone.config import cfg as config    
    file_dep = config.get_url('output').rglob('*_survey_area_data.csv')
    for file in file_dep:
        target = file.with_suffix('.nc')
        yield {
            'name':target,
            'actions':[process_survey],
            'file_dep':[file],
            'targets':[target],
            'uptodate': [True],
            'clean':True,
        }

@create_after(executed='survey_to_xarray', target_regex='*.nc')    
def task_xarray_to_gpkg():
    """
    Convert xarray datasets to geopackage files
    """
    def process_gpkg(dependencies, targets):
        import xarray as xr
        import geopandas as gp
        from shapely import wkt
        ds = xr.load_dataset(dependencies[0])
        df = ds.to_dataframe()
        gdf = gp.GeoDataFrame(df, geometry=df.image_polygon.apply(wkt.loads),crs=ds.attrs['horizontal_crs'])
        gdf.to_file(targets[0],driver='GPKG')
        
    from turtledrone.config import cfg as config    
    file_dep = config.get_url('output').rglob('*_survey_area_data.nc')
    for file in file_dep:
        target = file.with_suffix('.gpkg')
        yield {
            'name':target,
            'actions':[process_gpkg],
            'file_dep':[file],
            'targets':[target],
            'uptodate': [True],
            'clean':True,
        }

@create_after(executed='survey_to_xarray', target_regex='*.nc')        
def task_output_geotiff():
    """
    Generate geotiff files from xarray datasets
    for DJI pitch+90 correction is needed
    """
          
        
    def _grid_points_for_polygon(poly, resolution, crs):
        """
        Create a grid of points covering polygon bounding box.
        resolution: spacing in same units as polygon coordinates.
        Creates a mask where polygon interior = 1, exterior = 0
        Returns: xarray object with mask
        """

        from odc.geo.geobox import GeoBox
        from odc.geo.xr import xr_zeros
        import geopandas as gp
        from rasterio.features import rasterize as rio_rasterize
        import numpy as np
        import warnings
        
        grid = GeoBox.from_bbox(poly.bounds, crs, resolution=resolution)
        
        # Create a GeoDataFrame with the polygon to use with rasterio's rasterize
        gdf = gp.GeoDataFrame({'geometry': [poly]}, crs=crs)
        
        # Rasterize the polygon to create a mask
        # Suppress NotGeoreferencedWarning as we're explicitly providing the transform
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*Dataset has no geotransform.*')
            mask_array = rio_rasterize(
                [(geom, 1) for geom in gdf.geometry],
                out_shape=(grid.height, grid.width),
                transform=grid.affine,
                fill=0,
                dtype='uint8'
            )
        
        # Create the grid template
        grid_template = xr_zeros(grid)
        
        # Replace with the mask data
        grid_template.data = mask_array.astype(np.float32)
        
        return grid_template
    


    
    def process_geotiff(dependencies, targets, resolution=0.1):
        import xarray as xr
        import json
        from pyproj import CRS

        ds = xr.load_dataset(dependencies[0])
        crs_stere  = CRS.from_user_input(ds.attrs['horizontal_crs'])
        camera_int = json.loads(ds.attrs['camera_intrinsics'])
        target_dir = Path(targets[0]).parent    
        df = ds.to_dataframe()
        rows = [row for _, row in df.iterrows()]

        # Thread pool with 10 threads and tqdm progress bar
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(process_row, row, ds, camera_int, crs_stere, dependencies, resolution, target_dir)
                for row in rows
            ]
            for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Geotiffs"):
                pass
        
    def process_row(row, ds, camera_int, crs_stere, dependencies, resolution,target_dir):
        import cameratransform as ct
        from shapely import wkt
        from PIL import Image
        import numpy as np
        from pathlib import Path
        from odc.geo.cog import write_cog

        #convert image polygon to fiona polygon
        polygon = wkt.loads(row['image_polygon'])
        camera = ct.Camera(
            ct.RectilinearProjection(
                focallength_x_px=camera_int['calibrated_focal_length_x'],
                focallength_y_px=camera_int['calibrated_focal_length_y'],
                center_x_px=camera_int['calibrated_optical_center_x'],
                center_y_px=camera_int['calibrated_optical_center_y']
            ),
            lens=ct.BrownLensDistortion(
                camera_int['distortion_coefficients']['k1'],
                camera_int['distortion_coefficients']['k2'],
                camera_int['distortion_coefficients']['k3']
            )
        )
        camera.orientation = ct.SpatialOrientation(
            elevation_m=row['relative_altitude'],
            heading_deg=row['camera_yaw'], #-7.3
            tilt_deg=row['camera_pitch'],
            roll_deg=row['camera_roll'], 
            pos_x_m=row['x'], 
            pos_y_m=row['y']
        )             
        points = _grid_points_for_polygon(polygon, resolution, crs_stere)
        xx,yy = np.meshgrid(points.x, points.y)
        pts_xy = np.column_stack([xx.ravel(), yy.ravel()])
        idx_y, idx_x = np.unravel_index(np.arange(pts_xy.shape[0]), xx.shape)
        coords = np.column_stack([pts_xy,np.full(pts_xy.shape[0],0)])
        cam_points = camera.imageFromSpace(coords).astype(np.int32)
        img_w = int(ds.attrs['image_width']-20)
        img_h = int(ds.attrs['image_height']-20)
        u = cam_points[:, 0]
        v = cam_points[:, 1]
        in_frame = (u >= 0) & (v >= 0) & (u < img_w) & (v < img_h)
        image_path = Path(dependencies[0]).parent / row['image_name']
        with Image.open(image_path) as im:
            im = im.convert('RGB')
            arr = np.array(im)
        valid = np.where(in_frame)[0]
        if valid.size == 0 or len(points)==0:
            return
        samp_rgb = arr[v[valid], u[valid], :].astype(np.float32)
        Hgrid = points.sizes['y']
        Wgrid = points.sizes['x']
        rgb_grid = np.full((3, Hgrid, Wgrid), np.nan, dtype=np.float32)
        yi = idx_y[valid]
        xi = idx_x[valid]
        rgb_grid[0, yi, xi] = samp_rgb[:, 0]
        rgb_grid[1, yi, xi] = samp_rgb[:, 1]
        rgb_grid[2, yi, xi] = samp_rgb[:, 2]
        import xarray as xr
        rgb = xr.DataArray(
            rgb_grid,
            dims=('band', 'y', 'x'),
            coords={'band': ['R', 'G', 'B'], 'y': points.coords['y'], 'x': points.coords['x']},
            name='rgb_samples'
        )
        #ok so now use the mask in points to mask out rgb values outside polygon
        mask = points.data.astype(bool)
        rgb = rgb.where(mask)
        rgb['spatial_ref'] = points.spatial_ref
        target_image = target_dir / image_path.with_suffix('.tif').name
        write_cog(
            rgb,
            target_image,
            dtype='float32',
            compress='deflate',
            overview_levels=[2, 4, 8, 16, 32],
            overview_resampling='nearest',
            overwrite=True
        )


    from turtledrone.config import cfg as config    
    file_dep = config.get_url('output').rglob('*KEN*_survey_area_data.nc')
    for file in file_dep:
        target_dir = file.parent / 'geotiffs'
        os.makedirs(target_dir, exist_ok=True)
        #find all the jpeg files in the same directory
        jpeg_files = list(file.parent.glob('*.JPG'))
        #change all the jpeg files to geotiff files into a list
        targets = [target_dir / (jpeg.stem + '.tif') for jpeg in jpeg_files]

        yield {
            'name':str(target_dir),
            'actions':[process_geotiff],
            'file_dep':[file],
            'targets':targets,
            'uptodate': [True],
            'clean':True,
        }

if __name__ == '__main__':
    import doit
    DOIT_CONFIG = {'check_file_uptodate': 'timestamp'}
    #print(globals())
    doit.run(globals())