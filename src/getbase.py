from ftplib import FTP
from os.path import supports_unicode_filenames
import pandas as pd
import os


def getbase(times,destination):
    starttime =times.dt.floor('1H').min()-pd.Timedelta('1H')
    endtime = times.dt.ceil('1H').max()+pd.Timedelta('1H')
    filerange = pd.date_range(starttime,endtime,freq='15MIN')
    files =[{'path':f'/highrate/{item.year}/{item.day_of_year:03}/{item.hour:02}','file':f'{station}_S_{item.year}{item.day_of_year:03}{item.hour:02}{item.minute:02}_15M_01S_MO.crx.gz'} for item in filerange]
    ftp = FTP('ftp.data.gnss.ga.gov.au')
    ftp.login()
    for item in files:
        dest = os.path.join(destination,item['file'])
        print(dest)
        if not os.path.exists(dest):
            print(item['path'])
            ftp.cwd(item['path'])
            ftp.retrbinary("RETR " + item['file'], open(dest, 'wb').write)
    ftp.quit()
    
# EXMT00AUS_S_20203020100_15M_01S_MO.crx.gz

def buildpath(item):
    path = f'{filerange[1].year}/{filerange[1].day_of_year:03}/{filerange[1].hour:02}/{station}_S_{filerange[1].year}{filerange[1].day_of_year:03}{filerange[1].hour:02}{filerange[1].minute:02}_15M_01S_MO.crx.gz'
    return path


filename = r"T:/drone/raw/card0/SURVEY/100_0001/100_0001_Timestamp.MRK"
inputpath =os.path.split(filename)[0]
jsonfile = inputpath+'/exif.json'
merge = inputpath+'/merge.csv'

station = 'EXMT00AUS'
data = pd.read_csv(merge,parse_dates=['UTCtime'])
getbase(data.UTCtime,'T:/drone/raw/gnss')
