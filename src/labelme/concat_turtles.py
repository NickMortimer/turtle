import pandas as pd 
import glob
import os
import json


files = glob.glob('/mnt/data/fielddata/Nick/turtle_stuff/NingalooOutlook/**/turtles_per_survey.csv',recursive=True)
data = pd.concat([pd.read_csv(file) for file in files])
data.to_csv('/mnt/data/fielddata/Nick/turtle_stuff/NingalooOutlook/labelimage/turtle_counts.csv')