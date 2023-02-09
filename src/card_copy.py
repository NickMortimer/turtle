from pydoc import describe
import pandas as pd
import glob
import re
import os

import subprocess
import uuid
from datetime import datetime
import argparse





def rsync(sources,dest,move=False):
    output = []
    for source in sources:
        os.makedirs(dest,exist_ok=True)
        print(f'{source} {os.path.exists(source)}')
        if os.path.exists(source):
            if move:
                command =f"rsync -a -v -W --remove-source-files   {source} {dest}"
            else:
                command =f"rsync -v  -a -W  {source} {dest}"
            print(command)
            output.append(subprocess.call(command, shell=True))
    return output

def getdrives():
    command = "sudo blkid"
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    drives =output.decode().split('\n')
    drives.sort()
    drives = list(filter(lambda x: '/dev/sd' in x, drives))
    if drives:
        drivedf = pd.DataFrame({'blkid':drives})
        drivedf['UUID']=drivedf['blkid'].str.extract('UUID="(?P<UUID>([0-9a-fA-F]){4}-([0-9a-fA-F]){4})')['UUID']
        drivedf['PARTUUID']=drivedf['blkid'].str.extract('PARTUUID="(?P<PARTUUID>([0-9a-fA-F]){8}-([0-9a-fA-F]){4}-([0-9a-fA-F]){4}-([0-9a-fA-F]){4}-([0-9a-fA-F]){12})')['PARTUUID']
        drivedf['LABEL']=drivedf['blkid'].str.extract(' LABEL="(?P<LABEL>.*?)"')['LABEL']
        drivedf['DEV']=drivedf['blkid'].str.extract('/dev/(?P<DEV>.*?):')['DEV']
        drivedf.sort_values('DEV',inplace=True)
        drivedf['LETTER']=drivedf['DEV'].str[2]
        drivedf.set_index('LETTER',inplace=True)
    drivedf['PATH'] = drivedf['UUID']
    drivedf.loc[~pd.isna(drivedf.LABEL),'PATH'] = drivedf.loc[~pd.isna(drivedf.LABEL),'LABEL']
    drivedf = drivedf[pd.isna(drivedf.PARTUUID)]
    return drivedf

def namedrives(drivedf):
    for index,row in drivedf[~drivedf.UUID.isin(drivedf.LABEL)].iterrows():
        command =f"exfatlabel  /dev/{row.DEV} {row.UUID}"
        subprocess.call(command, shell=True)



parser = argparse.ArgumentParser(description='Copy to sdcards to file store',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('destination',type=str,help='primary path')
parser.add_argument('primarystore',type=str,help='primary path')
parser.add_argument('backupstore',type=str,help='backup path')
parser.add_argument('-t','--time',default=False,action='store_true',help='Force time of backup')
args = parser.parse_args()
primarydrive = args.primarystore
backupdrives = args.backupstore
if args.time:
    timest = args.time
else:
    timest = f'{datetime.now():%Y%m%dT%H%M%S}'
dest =f'{args.destination}{timest}'
local = f'{primarydrive}{dest}'
network =f'{backupdrives}{dest}'
drives = getdrives()
namedrives(drives)
drives = getdrives()
for letter,source in ('/media/card/'+drives.PATH).iteritems():
    rsync([source],local)
    rsync([local+'/'],network)
    rsync([source],network,move=True)
    command =f'find {source} -empty -type d -delete'
    subprocess.call(command, shell=True)

rsync([local.rsplit('/',1)[0]+'/'],network.rsplit('/',1)[0])



