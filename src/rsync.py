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
            os.makedirs(item.Destination,  exist_ok=True)
            command = shlex.split(f"rsync -a -W  --exclude '*.LRV' --exclude '*.THM' --progress  {item.name} {item.Destination}")
            processes.add(subprocess.Popen(command))#,startupinfo=
            if len(processes) >= max_processes:
                os.wait()
            processes.difference_update([p for p in processes if p.poll() is not None])
        for host in hosts:
            if len(cards[cards['host']==host])==2: #check to see if there were two cards on the host
                item =cards[cards['host']==host].iloc[-1]
                os.makedirs(item.Destination, exist_ok=True)
                command = shlex.split(f"rsync -a -W  --exclude '*.LRV' --exclude '*.THM' --progress  {item.name} {item.Destination}")
                processes.add(subprocess.Popen(command))
                if len(processes) >= max_processes:
                    os.wait()
                processes.difference_update([p for p in processes if p.poll() is not None])
        for p in processes:
            if p.poll() is None:
                p.wait()
    file_dep = glob.glob(f"{config.geturl('imagehead')}/backup_*.csv")
    for file in file_dep:
        target = file.replace('.csv','_complete.csv')
        yield {
            'name':file,
            'file_dep':[file],
            'actions':[rsync_cards],
            'targets':[target],
            'uptodate':[False],
            'clean':True,
        }




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