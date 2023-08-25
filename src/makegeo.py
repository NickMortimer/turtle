import os
import glob
import doit
import glob
import os
import pandas as pd
import config

def task_make_geo():
    def make_geo(dependencies, targets):
        data = pd.read_csv(dependencies[0])
        text = list(data.apply(lambda x:f'{x.NewName}   {x.LongitudeMrk}   {x.LatitudeMrk}    {x.EllipsoideHightMrk} {x.CameraYaw} {x.CameraPitch} {x.CameraRoll}\n',axis=1))
        if 'LatitudeMrk' in data.columns:
            with open(targets[0], 'w') as f:
                f.write('EPSG:4326\n')
                f.writelines(text)
        
    
    file_dep = glob.glob(os.path.join(config.geturl('output'),'**','*_survey_area_data.csv'),recursive=True)
    for file in file_dep:
        target = os.path.join(config.getdest(os.path.basename(file)),'geo.txt')
        yield {
            'name':target,
            'actions':[make_geo],
            'file_dep':[file],
            'targets':[target],
            'uptodate': [True],
            'clean':True,
        }                                
            


if __name__ == '__main__':
    import doit
    DOIT_CONFIG = {'check_file_uptodate': 'timestamp'}
    #print(globals())
    doit.run(globals())   