import numpy as np

def convert_wgs_to_utm(lon, lat):
    utm_band = str(int((np.floor((lon + 180) / 6 ) % 60) + 1))
    if len(utm_band) == 1:
        utm_band = '0'+utm_band
    if lat >= 0:
        epsg_code = '326' + utm_band
    else:
        epsg_code = '327' + utm_band
    return epsg_code