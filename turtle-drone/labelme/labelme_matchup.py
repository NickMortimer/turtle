from genericpath import exists
import os
import glob
import shutil
import doit
import glob
import os
import numpy as np
import pandas as pd
from doit import create_after
import numpy as np
import json
import utils.config as config
from pathlib import Path
from doit.tools import run_once
from pyproj import Proj 
import cameratransform as ct
from shapely.geometry import Polygon

def task_set_up():
    config.read_config()


@create_after(executed='process_list_json', target_regex='.*\surveyswitharea.csv')   
def task_matchup_labelme():
    def process_labelmatch(dependencies, targets):
        jsonfiles = pd.read_csv(config.geturl('labelindex'),index_col='TimeStamp',parse_dates=['TimeStamp'])
        datafile = pd.read_csv(dependencies[0],index_col='TimeStamp',parse_dates=['TimeStamp'])
        datafile =datafile.join(jsonfiles,rsuffix='_label',how='left')
        datafile['LabelNewName']=datafile.NewName.str.replace('.JPG','.json',regex=False)
        datafile.to_csv(targets[0])


    file_dep =  list(config.geturl('output').rglob('*_survey_area_data.csv'))
    for item in file_dep:
        target = item.parent / item.name.replace('.csv','_labelmerge.csv')       
        yield {
            'name': item,
            'file_dep':[item],
            'actions':[process_labelmatch],
            'targets':[target],
            'clean':True,
        }  




@create_after(executed='matchup_labelme', target_regex='.*\surveyswitharea.csv')   
def task_move_labelme():
    def process_json(dependencies, targets): 
        def update_json(file):
            try:
                with open(file, "r") as read_file:
                    data = json.load(read_file)
                    data['imagePath'] = os.path.basename(file).replace('json','JPG')
                    json_object = json.dumps(data, indent=2)
                    with open(file, "w") as outfile:
                        outfile.write(json_object)
            except:
                print(file)
                
        destination =os.path.dirname(targets[0])
        os.makedirs(destination,exist_ok=True)
        survey = pd.read_csv(dependencies[0])
        if 'JsonSource' in survey.columns:
            survey = survey.loc[~survey.JsonSource.isna()]
            for index,row in survey.iterrows():
                    dest = os.path.join(os.path.dirname(row.FileDest),os.path.basename(row.LabelNewName))
                    source = Path(row.JsonSource)
                    if not source.exists():
                        source =Path(config.geturl('labelmesource')).parent / source.name
                    if not os.path.exists(dest):
                        if config.cfg['outputhardlink']:
                            os.link(source, dest)   
                        else:
                            shutil.copyfile(source,dest) 
                    update_json(dest)
                    
        shutil.copyfile(dependencies[0],targets[0])
            
    file_dep =  list(config.geturl('output').rglob('*_labelmerge.csv'))
    for item in file_dep:
        target = item.parent / item.name.replace('.csv','_move.csv')       
        yield {
            'name': item,
            'file_dep':[item],
            'actions':[process_json],
            'targets':[target],
            'clean':True,
        }   

@create_after(executed='move_labelme', target_regex='.*\surveyswitharea.csv')   
def task_report_labelme():
    def process_report(dependencies, targets): 
        found = list(filter(lambda x: '_labelmerge.csv' in x, dependencies))
        index = pd.read_csv(config.geturl('labelindex'),parse_dates=['TimeStamp'],index_col='TimeStamp')
        data = pd.concat([pd.read_csv(file,parse_dates=['TimeStamp'],index_col='TimeStamp')  for file in found])
        index.join(data[['SurveyId','NewName']]).to_csv(targets[0])
    file_dep =  list(config.geturl('output').rglob('*_labelmerge.csv'))
    file_dep.append(config.geturl('labelindex'))

    target = config.geturl('reports') / 'label_matchup_report.csv'   
    yield {
        'name': target,
        'file_dep':file_dep,
        'actions':[process_report],
        'targets':[target],
        'clean':True,
    }          
      


if __name__ == '__main__':
    import doit

    #print(globals())
    doit.run(globals())