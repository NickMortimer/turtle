import pandas as pd 
import glob
import os
import json


files = glob.glob('/mnt/data/fielddata/Nick/turtle_stuff/NingalooOutlook/labelimage/*.json')
data = pd.DataFrame(files,columns=['FileName'])
data['BaseName']=data.FileName.apply(os.path.basename)
data['TimeStamp']=data.BaseName.str.extract(r'(?P<TimeStamp>\d{8}T\d{6})')
data.TimeStamp = pd.to_datetime(data['TimeStamp'])
data = data.set_index('TimeStamp')
data.to_csv('/mnt/data/fielddata/Nick/turtle_stuff/NingalooOutlook/labelimage/turtle_json.csv')




