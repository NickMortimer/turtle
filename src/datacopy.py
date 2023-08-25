import glob
import pandas as pd
import subprocess
import shlex
import os 
import shutil

#copy drone surveys over

outputdrives = glob.glob('/mnt/datasets/ibenthos/indonesia_sulawesi_2023_06/data_processed/*_N*')
output = pd.DataFrame(outputdrives,columns=['OutputDir'])
output[['Date','Station']] =output.OutputDir.str.extract(r'(?P<Date>\d{8})_(?P<Station>NS\d{2})')
output =output.set_index('Station')
raw = glob.glob('/mnt/datasets/ibenthos/indonesia_sulawesi_2023_06/data_source/DJIP4RTK/surveys/ID/*/*')
raw = pd.DataFrame(raw,columns=['InputDir'])
raw['Station']=raw.InputDir.str.extract(r'\/(?P<Station>NS\d{2})_')
raw =raw.set_index('Station')
raw =raw.join(output,on=['Station'])
raw.to_csv('/mnt/datasets/ibenthos/indonesia_sulawesi_2023_06/data_source/DJIP4RTK/mover.csv')
raw['Command'] = raw.apply(lambda x:f'rsync -avh {x.InputDir} {x.OutputDir}/DJIP4RTK/ --delete',axis=1)
processes = set()
max_processes = 4 
for index,row in raw.iterrows():
    command = shlex.split(row.Command)
    processes.add(subprocess.Popen(command))#,startupinfo=
    if len(processes) >= max_processes:
        os.wait()
        processes.difference_update([p for p in processes if p.poll() is not None])
ortho = glob.glob('/mnt/datasets/ibenthos/indonesia_sulawesi_2023_06/data_source/DJIP4RTK/maps/NS*')
ortho = pd.DataFrame(ortho,columns=['OrthoSource'])
ortho['Station'] = ortho.OrthoSource.str.extract(r'(?P<Station>NS\d{2})')
ortho = ortho.set_index('Station')
ortho = ortho.join(output)
ortho['Command'] = ortho.apply(lambda x:f'rsync -avh {x.OrthoSource} {x.OutputDir}/DJIP4RTK/ --delete',axis=1)
processes = set()
max_processes = 4 
for index,row in ortho.iterrows():
    command = shlex.split(row.Command)
    processes.add(subprocess.Popen(command))#,startupinfo=
    if len(processes) >= max_processes:
        os.wait()
        processes.difference_update([p for p in processes if p.poll() is not None])
os.wait()
