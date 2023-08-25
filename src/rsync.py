import doit
from doit import get_var
from doit.tools import run_once
from doit import create_after
import glob
from numpy import int32
import os
import pandas as pd
import config
import subprocess
import shlex
from io import StringIO
import time
import json

# def task_scan_serialnumbers():
#     def get_serialnumbers(dependencies, targets):
#         mountpoints = pd.DataFrame(json.loads(subprocess.getoutput('lsblk -J -o  NAME,SIZE,FSTYPE,TYPE,MOUNTPOINT'))['blockdevices'])
#         mountpoints = mountpoints[~mountpoints.children.isna()]
#         mountpoints =pd.DataFrame(mountpoints.children.apply(lambda x: x[0]).to_list())[['name','mountpoint']]
#         paths = pd.DataFrame(subprocess.getoutput('udevadm info -q path -n $(ls /dev/s*1)').splitlines(),columns=['Path'])
#         paths[['host','dev']]=paths.Path.str.extract(r'(?P<host>host\d+).*block\/(?P<dev>([^\\]+$))')[['host','dev']]
#         paths['name'] =paths.dev.str.split('/',expand=True)[1]
#         mountpoints =mountpoints.merge(paths, on='name', how='inner')
#         mountpoints['CardUUID']=mountpoints.mountpoint.str.extract(r'(?P<CardUUID>[0-9A-F]{4}-[0-9A-F]{4})')
#         mountpoints = mountpoints[~mountpoints.CardUUID.isna()]
#         os.makedirs(os.path.dirname(targets[0]),exist_ok=True)
#         path = f'{os.path.dirname(targets[0])}/{os.path.splitext(os.path.basename(targets[0]))[0].split("_")[1]}'
#         mountpoints['Destination'] = path
#         mountpoints.to_csv(targets[0],index=False)

#     t = time.localtime()
#     target = f"{config.geturl('imagehead')}/backup_{time.strftime('%Y%m%dT%H%M%S', t)}_source.csv"
#     return {
#         'actions':[get_serialnumbers],
#         'targets':[target],
#         'uptodate':[False],
#         'clean':True,
#     }

# @create_after(executed='scan_serialnumbers', target_regex='*.csv')   
def task_rsync_cards():
    def rsync_cards(dependencies, targets):
        cards = pd.read_csv(dependencies[0])
        cards = cards.groupby('mountpoint').first()
        hosts = cards.host.unique()
        cards['RsyncDone']= False
        processes = set()
        max_processes = 4                                                                                                               
        workon = True
        hosts = cards.host.unique()
        for host in hosts:
            item =cards[cards['host']==host].iloc[0]
            if os.path.exists(item.name):
                os.makedirs(item.Destination,  exist_ok=True)
                command = shlex.split(f"rsync -a -W  --exclude '*.LRV' --exclude '*.THM' --progress  {item.name} {item.Destination}")
                processes.add(subprocess.Popen(command))#,startupinfo=
                if len(processes) >= max_processes:
                    os.wait()
                processes.difference_update([p for p in processes if p.poll() is not None])
        for host in hosts:
            if len(cards[cards['host']==host])==2: #check to see if there were two cards on the host
                item =cards[cards['host']==host].iloc[-1]
                if os.path.exists(item.name):
                    os.makedirs(item.Destination, exist_ok=True)
                    command = shlex.split(f"rsync -a -W  --exclude '*.LRV' --exclude '*.THM' --progress  {item.name} {item.Destination}")
                    processes.add(subprocess.Popen(command))
                    if len(processes) >= max_processes:
                        os.wait()
                    processes.difference_update([p for p in processes if p.poll() is not None])
        for p in processes:
            if p.poll() is None:
                p.wait()
        cards['RsyncDone']= cards.Destination.apply(os.path.exists)
        cards.to_csv(targets[0])
    file_dep = glob.glob(f"{config.geturl('imagehead')}/backup_*_source.csv")
    for file in file_dep:
        target = file.replace('.csv','_cardsread.csv')
        yield {
            'name':file,
            'file_dep':[file],
            'actions':[rsync_cards],
            'targets':[target],
            'uptodate':[False],
            'clean':True,
        }


@create_after(executed='rsync_cards', target_regex='*.csv')   
def task_sync_backup():
    file_dep = glob.glob(f"{config.geturl('imagehead')}/*_cardsread.csv")
    t = time.localtime()
    target = f"{config.geturl('imagehead')}/log_{time.strftime('%Y%m%dT%H%M%S', t)}.log"
    return {
        'file_dep':file_dep,
        'actions':[f"rsync -av --progress {config.geturl('imagehead')}/  {config.geturl('backuphead')} > {target}"],
        'targets':[target],
        'uptodate':[True],
        'clean':True,
    }

@create_after(executed='sync_backup', target_regex='*.csv')  
def task_clean_cards():
    def rsync_cards(dependencies, targets):
        cards = pd.read_csv(dependencies[0])
        cards['Backup'] =cards.Destination.str.replace(config.geturl('imagehead'),config.geturl('backuphead'))
        cards = cards.groupby('mountpoint').first()
        hosts = cards.host.unique()
        cards['CleanDone']= False
        processes = set()
        max_processes = 4                                                                                                               
        workon = True
        hosts = cards.host.unique()
        for host in hosts:
            item =cards[cards['host']==host].iloc[0]
            if os.path.exists(item.name):
                os.makedirs(item.Destination,  exist_ok=True)
                command = shlex.split(f"rsync -a -W  --exclude '*.LRV' --exclude '*.THM' --progress --remove-source-files {item.name} {item.Backup}")
                processes.add(subprocess.Popen(command))#,startupinfo=
                if len(processes) >= max_processes:
                    os.wait()
                processes.difference_update([p for p in processes if p.poll() is not None])
        for host in hosts:
            if len(cards[cards['host']==host])==2: #check to see if there were two cards on the host
                item =cards[cards['host']==host].iloc[-1]
                if os.path.exists(item.name):
                    os.makedirs(item.Destination, exist_ok=True)
                    command = shlex.split(f"rsync -a -W  --exclude '*.LRV' --exclude '*.THM' --progress --remove-source-files {item.name} {item.Backup}")
                    processes.add(subprocess.Popen(command))
                    if len(processes) >= max_processes:
                        os.wait()
                    processes.difference_update([p for p in processes if p.poll() is not None])
        for p in processes:
            if p.poll() is None:
                p.wait()
        cards['CleanDone']= True                
        cards.to_csv(targets[0])
    file_dep = glob.glob(f"{config.geturl('imagehead')}/*_cardsread.csv")
    for file in file_dep:
        target = file.replace('.csv','_z`cleaned.csv')
        yield {
            'name':file,
            'file_dep':[file],
            'actions':[rsync_cards],
            'targets':[target],
            'uptodate':[False],
            'clean':True,
        }




    # def sync_dbackup()
    #         os.makedirs(item.Destination,  exist_ok=True)
    #         command = shlex.split(f"rsync -a -W  --exclude '*.LRV' --exclude '*.THM' --progress  {item.name} {item.Destination}")
    #         processes.add(subprocess.Popen(command))#,startupinfo=
# def task_scan_devs():
#     result = subprocess.getoutput('lsblk -J -o  NAME,SIZE,FSTYPE,TYPE,MOUNTPOINT')
    

# def task_process_cameras():
#     def process_cameras(dependencies, targets):
#         backupsessions = pd.read_csv(config.geturl('backupsessions'),index_col='TimeStamp',parse_dates=['TimeStamp']).sort_values('TimeStamp')
#         barnumbers = pd.read_csv(config.geturl('barnumbers'))
#         cameras = []
#         for item in dependencies:
#             filter = os.path.join(item,config.cfg['paths']['videowild'])
#             files = glob.glob(filter)
#             if files:
#                 command = f"exiftool -api largefilesupport=1 -u  -json -ext MP4 -q -CameraSerialNumber -CreateDate -SourceFile -Duration -FileSize -FieldOfView {item}"
#                 result =subprocess.getoutput(command)
#                 try:
#                     df =pd.read_json(result)
#                     cameras.append(df)
#                 except:
#                     print('No json')
#         cameras = pd.concat(cameras)
#         cameras['CreateDate'] = pd.to_datetime(cameras['CreateDate'],format='%Y:%m:%d %H:%M:%S')
#         cameras =cameras.merge(barnumbers, on='CameraSerialNumber', how='inner')
#         path = f"{config.geturl('cardstore')}/{backupsessions.iloc[-1].name:%Y%m%dT%H%M%S}"
#         os.makedirs(path,exist_ok=True)
#         cameras['Destination']=cameras.apply(lambda x: f"{path}/{x.CameraNumber}_{x.CameraSerialNumber}_{x.CreateDate:%Y%m%dT%H%M}",axis=1)
#         cameras.to_csv(targets[0],index=False)



#     gopro = glob.glob('/media/*/*/DCIM/100GOPRO')
#     backupsessions = pd.read_csv(config.geturl('backupsessions'),index_col='TimeStamp',parse_dates=['TimeStamp']).sort_values('TimeStamp')
#     target = f"{config.geturl('cardstore')}/backup_{backupsessions.iloc[-1].name:%Y%m%dT%H%M%S}.csv"
#     return {
#         'file_dep':gopro,
#         'actions':[get_serialnumbers],
#         'targets':[target],
#         'uptodate':[True],
#         'clean':True,
#     }


         
if __name__ == '__main__':
    import doit
    DOIT_CONFIG = {'check_file_uptodate': 'timestamp'}
    #print(globals())
    doit.run(globals())