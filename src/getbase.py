from ftplib import FTP
from os.path import supports_unicode_filenames
import pandas as pd
import numpy as np
import yaml
import os
import glob
from doit import get_var
from doit.tools import run_once
from doit import create_after
from read_rtk import read_mrk_gpst
import shutil
from read_rtk import read_pos

def getbasenames(times,station,destination):
    """Create the names of the files neaded to get from Geoscience Australia
       times : pd series of times
       station : station that your interesting in
       destination : the path to where the files are to go
       
       eg.
       ftp.data.gnss.ga.gov.au
       EXMT00AUS_S_20203020100_15M_01S_MO.crx.gz
       
    """
    starttime =times.floor('1H').min()-pd.Timedelta('1H')
    endtime = times.ceil('1H').max()+pd.Timedelta('1H')
    filerange = pd.date_range(starttime,endtime,freq='15MIN')
    files =[{'TimeStamp':item,'path':f'/highrate/{item.year}/{item.day_of_year:03}/{item.hour:02}',
             'destination':destination,
             'file':f'{station}_S_{item.year}{item.day_of_year:03}{item.hour:02}{item.minute:02}_15M_01S_MO.crx.gz'} for item in filerange]
    for day in times.floor('1D').unique():
        files.append({'path':f'/daily/{day.year}/{day.day_of_year:03}',
                       'destination':destination,
                       'file':f'{station}_R_{day.year}{day.day_of_year:03}0000_01D_30S_MN.rnx.gz'})
    return pd.DataFrame(files)

def getbase(files):
    ftp = FTP('ftp.data.gnss.ga.gov.au')
    ftp.login()
    for item in files:
        dest = os.path.join(item['destination'],item['file'])
        print(dest)
        if not os.path.exists(dest):
            print(item['path'])
            ftp.cwd(item['path'])
            ftp.retrbinary("RETR " + item['file'], open(dest, 'wb').write)
    ftp.quit()


def task_calc_basefiles():
        def calc_basefiles(dependencies, targets,cfg,basepath):
            marks = [read_mrk_gpst(mark) for mark in dependencies]
            files =pd.concat([getbasenames(df.index,cfg['survey']['basestation'],
                                           os.path.join(basepath,cfg['paths']['gnssceche'])) for df in marks])
            files.drop_duplicates(inplace=True)
            files.to_csv(targets[0],index=False)
        config = {"config": get_var('config', 'NO')}
        with open(config['config'], 'r') as ymlfile:
            cfg = yaml.load(ymlfile, yaml.SafeLoader)
        basepath = os.path.dirname(config['config'])
        file_dep =glob.glob(os.path.join(basepath,cfg['paths']['marksource']),recursive=True)
        target = os.path.join(basepath,cfg['paths']['process'],'basefiles.csv')
        return {
            'actions':[(calc_basefiles, [],{'cfg':cfg,'basepath':basepath})],
            'file_dep':file_dep,
            'targets':[target],
            'uptodate':[True],
            'clean':True,
        }   
        
        
        
             
        
def task_get_basefiles():
        def calc_basefiles(dependencies, targets,basepath):
            basefiles = pd.read_csv(dependencies[0])
            getbase(basefiles.to_dict('records'))
        config = {"config": get_var('config', 'NO')}
        with open(config['config'], 'r') as ymlfile:
            cfg = yaml.load(ymlfile, yaml.SafeLoader)
        basepath = os.path.dirname(config['config'])
        file_dep = os.path.join(basepath,cfg['paths']['process'],'basefiles.csv')
        return {
            'actions':[(calc_basefiles, [],{'basepath':basepath})],
            'file_dep':[file_dep],
            'uptodate':[True],
            'clean':True,
        }         
def task_unzip_base():
        def calc_basefiles(dependencies, targets,cfg,basepath):
            surveys = pd.read_csv(dependencies[0],index_col='TimeStamp',parse_dates=['TimeStamp'])
            files =pd.concat([getbasenames(df.index,cfg['survey']['basestation'],
                                           os.path.join(basepath,cfg['paths']['gnssceche'])) for survey,df in surveys.groupby('Survey')])
            files.drop_duplicates(inplace=True)
            files.to_csv(targets[0],index=False)
        config = {"config": get_var('config', 'NO')}
        with open(config['config'], 'r') as ymlfile:
            cfg = yaml.load(ymlfile, yaml.SafeLoader)
        basepath = os.path.dirname(config['config'])
        crx2nxpath = os.path.join(basepath,cfg['paths']['rtklib'],'crx2rnx.exe')
        gnsspath = os.path.join(basepath,cfg['paths']['gnssceche'])
        file_dep = glob.glob(os.path.join(gnsspath,'*.gz'))
        zip = os.path.join(basepath,cfg['paths']['7zip'])
        for file in file_dep:
            yield {
                'name':file,
                'actions':[f'"{zip}" e "{file}" -o"{gnsspath}"'],
                'file_dep':[file],
                'targets' : [file[:-3]],
                'uptodate':[True],
                'clean':True,
            } 


def task_crx2rinx_base():
        def calc_basefiles(dependencies, targets,cfg,basepath):
            surveys = pd.read_csv(dependencies[0],index_col='TimeStamp',parse_dates=['TimeStamp'])
            files =pd.concat([getbasenames(df.index,cfg['survey']['basestation'],
                                           os.path.join(basepath,cfg['paths']['gnssceche'])) for survey,df in surveys.groupby('Survey')])
            files.drop_duplicates(inplace=True)
            files.to_csv(targets[0],index=False)
        config = {"config": get_var('config', 'NO')}
        with open(config['config'], 'r') as ymlfile:
            cfg = yaml.load(ymlfile, yaml.SafeLoader)
        basepath = os.path.dirname(config['config'])
        crx2nxpath = os.path.join(basepath,cfg['paths']['crx2rnx'])
        file_dep = glob.glob(os.path.join(basepath,cfg['paths']['gnssceche'],'*.crx'))
        targets=list(map (lambda x:x.replace('.crx','.rnx'),file_dep))
        for file,target in zip(file_dep,targets):
            yield {
                'name':file,
                'actions':[f'"{crx2nxpath}" "{file}"'],
                'file_dep':[file],
                'targets':[target],
                'clean':True,
            }          
        targets = os.path.join(basepath,os.path.dirname(cfg['paths']['output']),'merge/surveys.html')
        
            
def task_move_nav():
        def move_nav(dependencies, targets):
            gnss_file = targets[0]
            mrk =read_mrk(dependencies[0]).set_index('UTCtime')
            getbasenames(mrk.index,cfg['survey']['basestation'],os.path.dirname(gnss_file)).to_csv(gnss_file,index=True)
        config = {"config": get_var('config', 'NO')}
        with open(config['config'], 'r') as ymlfile:
            cfg = yaml.load(ymlfile, yaml.SafeLoader)
        basepath = os.path.dirname(config['config'])
        for item in glob.glob(os.path.join(basepath,cfg['paths']['imagesource']),recursive=True):
            source = os.path.join(basepath,os.path.dirname(item))
            mark = glob.glob(os.path.join(source,'*Timestamp.MRK'))
            if mark:
                yield {
                    'name':mark[0],
                    'actions':[move_nav],
                    'file_dep':mark,
                    'targets':[os.path.join(os.path.dirname(mark[0]),'gnss.csv')],
                    'clean':True,
                }          

def task_rtk():
        def process_rtk(dependencies, targets):
            gnss_file = targets[0]
            mrk =read_mrk_gpst(dependencies[0]).set_index('UTCtime')
            getbasenames(mrk.index,cfg['survey']['basestation'],os.path.dirname(gnss_file)).to_csv(gnss_file,index=True)
        config = {"config": get_var('config', 'NO')}
        with open(config['config'], 'r') as ymlfile:
            cfg = yaml.load(ymlfile, yaml.SafeLoader)
        basepath = os.path.dirname(config['config'])
        file_dep =glob.glob(os.path.join(basepath,cfg['paths']['obssource']),recursive=True)
        exepath = f'{os.path.join(basepath,cfg["paths"]["rtklib"],"rnx2rtkp.exe")}'
        rtkconfig = os.path.join(basepath,cfg['paths']['rtkconfig'])
        for file in file_dep:
            base = os.path.join(os.path.dirname(file),'*_15M_01S_MO.rnx')
            nav = os.path.join(os.path.dirname(file),'*_01D_30S_MN.rnx')
            yield {
                'name':file,
                'actions':[f'{exepath} -k {rtkconfig} -o {file.replace("obs","pos")} {file} {base} {nav} '],
                'file_dep':[file],
                'targets':[file.replace("obs","pos")],
                'uptodate':[True],
                'clean':True,
            }          

def task_calc_pic_pos():
        def process_pic_pos(dependencies, targets):
            markdata =read_mrk_gpst(list(filter(lambda x: '.MRK' in x, dependencies))[0])
            posdata = read_pos(list(filter(lambda x: '.pos' in x, dependencies))[0] )
            combined =pd.concat([markdata[['Sequence']],posdata]).sort_index()
            combined.loc[:,combined.columns !='Sequence']  = combined.loc[:,combined.columns !='Sequence'].interpolate(method='index')
            combined[~combined.Sequence.isna()].to_csv(targets[0],index=True)
        config = {"config": get_var('config', 'NO')}
        with open(config['config'], 'r') as ymlfile:
            cfg = yaml.load(ymlfile, yaml.SafeLoader)
        basepath = os.path.dirname(config['config'])
        pos_files =glob.glob(os.path.join(basepath,cfg['paths']['possource']),recursive=True)
        mark_files =glob.glob(os.path.join(basepath,cfg['paths']['marksource']),recursive=True)
        exepath = f'{os.path.join(basepath,cfg["paths"]["rtklib"],"rnx2rtkp.exe")}'
        rtkconfig = os.path.join(basepath,cfg['paths']['rtkconfig'])
        for mark,pos in zip(mark_files,pos_files):
            yield {
                'name':mark,
                'actions':[process_pic_pos],
                'file_dep':[mark,pos],
                'targets':[mark.replace('MRK','CSV')],
                'clean':True,
            } 
        
if __name__ == '__main__':
    import doit

    #print(globals())
    doit.run(globals())        
                    
# EXMT00AUS_S_20203020100_15M_01S_MO.crx.gz




# filename = r"T:/drone/raw/card0/SURVEY/100_0001/100_0001_Timestamp.MRK"
# inputpath =os.path.split(filename)[0]
# jsonfile = inputpath+'/exif.json'
# merge = inputpath+'/merge.csv'

# station = 'EXMT00AUS'
# data = pd.read_csv(merge,parse_dates=['UTCtime'])
# getbase(data.UTCtime,'T:/drone/raw/gnss')
