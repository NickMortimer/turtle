""" 
    Column 1: Image sequence.
    Column 2: Second of week of the exposure timestamp in UTC time, with seconds expressed in GPS
    time format.
    Column 3: GPS week of the exposure timestamp in UTC time, with seconds expressed in GPS time
    format.
    Column 4: Offset (in mm) of the antenna phase center to camera CMOS sensor center in the north
    (N) direction at the time of exposure of each photo. It is positive when the CMOS center is in the
    north direction of the antenna phase center and negative when in the south direction.
    Column 5: Offset (in mm) of the antenna phase center to camera CMOS sensor center in the east
    (E) direction at the time of exposure of each photo. It is positive when the CMOS center is in the
    east direction of the antenna phase center and negative when in the west direction.
    Column 6: Offset (in mm) of the antenna phase center to camera CMOS sensor center in the vertical
    (V) direction at the time of exposure of each photo. It is positive when the CMOS center is below
    the antenna phase center and negative when the former is above the latter.
    Column 7: real-time position latitude (Lat) of the CMOS center acquired at the time of exposure,
    in degrees. When the aircraft is in the RTK mode, its position is the RTK antenna phase center
    position plus the offset of the antenna phase center relative to the CMOS center at the time of
    exposure, with the RTK accuracy (centimeter level); and when the aircraft is in the GPS mode,
    its position is that detected by GPS single-point positioning plus the offset of the RTK antenna
    phase center relative to the CMOS center at the time of exposure, with GPS single-point
    positioning accuracy (meter level).
    Column 8: real-time position longitude (Lon) of the CMOS center acquired at the time of exposure,
    in degrees.
    Column 9: real-time height of the CMOS center acquired at the time of exposure, in meters. The
    height is a geodetic height (commonly known as the ellipsoid height), relative to the surface of
    the default reference ellipsoid (WGS84 is the default and can be set as other ellipsoids such as
    CGCS2000 via a different CORS station system/reference). Note that the above height is not based
    on the national 85 elevation datum or 56 elevation datum (normal height) or commonly used
    EGM96/2008 elevation datum (orthometric height) worldwide.
    For correlation of orthometric height, normal height, and geodetic height, refer to the following link:
    http://www.esri.com/news/arcuser/0703/geoid1of3.html
    Columns 10 to 12:
    Standard deviation (in meters) of positioning results in the north, east, and the vertical direction,
    describe the relative accuracy of positioning in the three directions.
    Column 13:
    RTK status flag. 0: no positioning; 16: single-point positioning mode; 34: RTK floating solution;
    50: RTK fixed solution. When the flag bit of a photo is not 50, it is recommended not to use this
    photo directly in map building.
    Rinex.Obs file: real-time decoded satellite observation file (GPS+GLO+BDS+GAL) received by
    the UAV, in RINEX 3.02 format. This file can be directly imported into PPK post-processing
    software for post-processing.
    
    1	268619.354051	[2129]	    24,N	     9,E	   193,V	-21.97273819,Lat	113.93433129,Lon	91.071,Ellh	1.056360, 1.085926, 3.245131	16,Q
    2	268625.283927	[2129]	    20,N	    15,E	   193,V	-21.97253139,Lat	113.93438555,Lon	91.182,Ellh	1.097883, 1.051731, 3.346695	16,Q
    3	268627.779356	[2129]	    10,N	    13,E	   194,V	-21.97234525,Lat	113.93440346,Lon	91.287,Ellh	1.095263, 1.051843, 3.344983	16,Q
    4	268630.314319	[2129]	    13,N	    10,E	   194,V	-21.97216604,Lat	113.93443131,Lon	89.959,Ellh	1.090093, 1.041215, 3.321528	16,Q
    5	268632.830491	[2129]	    15,N	     6,E	   194,V	-21.97199354,Lat	113.93445195,Lon	89.447,Ellh	1.081201, 1.031734, 3.327098	16,Q
    6	268635.351011	[2129]	    14,N	     8,E	   194,V	-21.97182339,Lat	113.93448809,Lon	92.148,Ellh	1.076505, 1.022308, 3.273235	16,Q
    7	268638.543134	[2129]	    66,N	    42,E	   178,V	-21.97168707,Lat	113.93449202,Lon	88.557,Ellh	1.110462, 1.036878, 3.367443	16,Q
    8	268642.289662	[2129]	    20,N	     3,E	   194,V	-21.97167044,Lat	113.93429231,Lon	91.825,Ellh	1.109429, 1.020208, 3.272965	16,Q


    https://hpiers.obspm.fr/eoppc/bul/bulc/Leap_Second.dat
"""

import pandas as pd
import os
import glob
import ftplib

"""
    Get 30 second data
"""
def get_base(date,station='EXMT00AUS',host='ftp://',destination='T:/drone/raw/gnss/'):
    for stamp in date:
        doy =pd.to_datetime(stamp).dayofyear
        year =pd.to_datetime(stamp).year
        file =f'{station}_R_{year}{doy:03d}0000_01D_30S_MO.crx.gz'
        gnss = f'{destination}{file}'
        if not os.path.exists(gnss):
            gFile = open(gnss, "wb")
            with ftplib.FTP('ftp.data.gnss.ga.gov.au',user='anonymous') as ftp:
                url = f'/daily/{year}/{doy:03d}/{file}'
                ftp.retrbinary(f'RETR {url}', gFile.write)
            gFile.close()

def read_mrk(filename,leapseconds=37):
    names=['Sequence','GPSSecondOfWeek','GPSWeekNumber','NorthOff','EastOff','VelOff','Latitude','Longitude','EllipsoideHight','Error','RTKFlag']
    data = pd.read_csv(filename,header=None,sep='\t',names=names)
    data['Latitude']=data.Latitude.str.split(',',expand=True)[0].astype(float)
    data['Longitude']=data.Longitude.str.split(',',expand=True)[0].astype(float)
    data['GPSWeekNumber']=data.GPSWeekNumber.str[1:-1].astype(int)
    data['UTCtime'] =data.apply(lambda x: pd.to_datetime("1980-01-06 00:00:00")+pd.Timedelta(weeks=x['GPSWeekNumber'])+pd.Timedelta(seconds=x.GPSSecondOfWeek)
                                -pd.Timedelta(seconds=leapseconds),axis=1)
    data.set_index('Sequence',inplace=True)
    return data 

def imagenames(filename):
    path =os.path.split(filename)[0]
    base =os.path.split(filename)[1].split('Timestamp')[0]+'*.JPG'
    images = glob.glob(os.path.join(path,base))
    data = pd.DataFrame({'Path':images})
    data['Sequence'] = data.Path.str.split('_').str[-1].str.split('.').str[0].astype(int)
    data.set_index('Sequence',inplace=True)
    return data

filename = r"T:/drone/raw/card0/SURVEY/100_0001/100_0001_Timestamp.MRK"
inputpath =os.path.split(filename)[0]
jsonfile = inputpath+'/exif.json'
merge = inputpath+'/merge.csv'
exiftool = f'""D:/exiftool/exiftool(-k).exe" -r -ext JPG -a -json {inputpath} > {jsonfile}"'
# if not os.path.exists(exiftool):
#     os.system(exiftool)
mrk =read_mrk(filename)
get_base(mrk.UTCtime.round('1D').unique())
images = imagenames(filename)
json = pd.read_json(jsonfile)
json['Squence']=json.SourceFile.str.extract('(?P<Sequence>\d\d\d\d)\.JPG').astype(int)
json.set_index('Squence',inplace=True)
json['Sequence'] =json.SourceFile.str.extract('(?P<Sequence>\d\d\d\d)\.JPG').astype(int)
json.set_index('Sequence',inplace=True)
json =json.join(mrk)
json.to_csv(merge,index=True)
json = json[json.SourceFile.str.contains('SURVEY')]