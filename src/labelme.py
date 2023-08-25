from genericpath import exists
import os
import glob
import shutil
import doit
import glob
import os
import numpy as np
from pandas.core.arrays.integer import Int64Dtype
from pandas.io.parsers import read_csv
import yaml
import pandas as pd
from doit import get_var
from doit.tools import run_once
from doit import create_after
import numpy as np
import plotly
import plotly.express as px
import geopandas as gp
import json
import ast
from pyproj import Proj
from drone import P4rtk
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from sklearn.cluster import MeanShift
import matplotlib.pyplot as plt
import shapely.wkt
import plotly
import plotly.express as px
from PIL import Image
import config
import shapely.wkt
from shapely.geometry import MultiPoint

def task_set_up():
    config.read_config()

# # def task_strip_image():
#     def strip_image(file):
#         try:
#             with open(file, "r") as read_file:
#                 data = json.load(read_file)
#             if data.get('imageData'):
#                 data['imageData'] = None
#                 # Serializing json
#                 json_object = json.dumps(data, indent=2)
#                 # Writing to sample.json
#                 with open(file, "w") as outfile:
#                     outfile.write(json_object)
#                 return {'file':file, 'Strip':'True'}
#             return {'file':file, 'Strip':'False'}
#         except:
#             return {'file':file, 'Strip':'Bad'}

#     def process_json(dependencies, targets):
#          output = pd.DataFrame.from_records([strip_image(file) for file in dependencies])
#          output.to_csv(targets[0])

# #     for item in glob.glob(os.path.join(config.geturl('labelmesource')),recursive=True):
# #             file_dep = glob.glob(os.path.join(os.path.dirname(item),'*.json'))
# #             if file_dep:
# #                 target =   os.path.join(os.path.dirname(item),'stripImage.csv')           
# #                 yield {
# #                     'name':item,
# #                     'file_dep': file_dep,
# #                     'actions':[process_json],
# #                     'targets':[target],
# #                     'clean':True,
# #                     'uptodate':[True],
                    
# #                 }


# def task_merge_strip():


#     def process_merge(dependencies, targets):
#          output = pd.concat([pd.read_csv(file) for file in dependencies])
#          output.to_csv(targets[0])
#     os.makedirs(config.geturl('process'),exist_ok=True)
#     file_dep = glob.glob(os.path.join(config.geturl('labelmesource'),"stripImage.csv"),recursive=True)
#     target = os.path.join(config.geturl('process'),"stripImage.csv")
#     return {
#                 'file_dep': file_dep,
#                 'actions':[process_merge],
#                 'targets':[target],
#                 'clean':True,
#                 'uptodate':[True],
                
#             }


# def task_process_list_json():
#         def process_labelme(dependencies, targets):
#             files =glob.glob(os.path.join(basepath,cfg['paths']['labelmesource']),recursive=True)
#             data = pd.DataFrame(dependencies,names='FilePaths')
#             data.to_csv(targets[0],index=False)       
            
#         config = {"config": get_var('config', 'NO')}
#         with open(config['config'], 'r') as ymlfile:
#             cfg = yaml.load(ymlfile, yaml.SafeLoader)
#         basepath = os.path.dirname(config['config'])

#         target =   os.path.join(basepath,cfg['paths']['process'],'jsonfiles.csv')        
#         return {
#             'actions':[process_labelme],
#             'targets':[target],
#             'clean':True,
#             'uptodate':[run_once],
            
 

#        }


# def task_matchup_labelme():
#     def process_labelmatch(dependencies, targets):
#         jsonfiles = pd.read_csv(config.geturl('labelmesource'),index_col='TimeStamp')
#         datafile = pd.read_csv(dependencies[0],index_col='TimeStamp')
#         datafile =datafile.join(jsonfiles,rsuffix='_label')
#         datafile.to_csv(targets[0])


#     file_dep =  glob.glob(os.path.join(config.geturl('output'),'**/*_survey_area_data.csv'),recursive=True)
#     for item in file_dep:
#         target = item.replace('.csv','_labelmerge.csv')       
#         yield {
#             'name': item,
#             'file_dep':[item],
#             'actions':[process_labelmatch],
#             'targets':[target],
#             'clean':True,
#         }  

# @create_after(executed='matchup_labelme')   
# def task_move_labelme():
#     def process_json(dependencies, targets): 
#         destination =os.path.dirname(targets[0])
#         os.makedirs(destination,exist_ok=True)
#         survey = pd.read_csv(dependencies[0])
#         survey = survey.loc[~survey.FileName_label.isna()]
#         for index,row in survey.iterrows():
#                 print(row.name)
#                 dest = os.path.join(os.path.dirname(row.FileDest),os.path.basename(row.FileName_label))
#                 if not os.path.exists(dest):
#                     if config.cfg['survey']['outputsymlink']:
#                         relpath = os.path.join(os.path.relpath(os.path.dirname(row.FileName_label),start=os.path.dirname(row.FileDest)),os.path.basename(row.FileName_label))
#                         os.symlink(relpath, dest)
#                     else:
#                         shutil.copyfile(row.FileName_label,dest)
#         shutil.copyfile(dependencies[0],targets[0])
            
#     file_dep =  glob.glob(os.path.join(config.geturl('output'),'**/*_labelmerge.csv'),recursive=True)
#     for item in file_dep:
#         target = item.replace('.csv','_move.csv')       
#         yield {
#             'name': item,
#             'file_dep':[item],
#             'actions':[process_json],
#             'targets':[target],
#             'clean':True,
#         }  







#@create_after(executed='move_labelme')   
def task_process_labelme():
        
    def loadshapes(file):
        print(file)
        lines =[]
        with open(file, "r") as read_file:
            data = json.load(read_file)

        #data = json.loads(''.join(lines).replace("\n", "").replace("'", '"').replace('u"', '"'))
        data =pd.DataFrame(data['shapes'])
        data['FilePath'] =file             
        return(data)
 
    def process_labelme(dependencies, targets):
        jsonfiles = glob.glob(os.path.join(os.path.dirname(targets[0]),'*.json'))
        if jsonfiles:
            data = pd.concat([loadshapes(file) for file in jsonfiles])
        else:
            data = pd.DataFrame()
        data.to_csv(targets[0],index=False)       
        

    for item in glob.glob(os.path.join(config.geturl('output'),'AU/**/'),recursive=True):
        file_dep = glob.glob(os.path.join(os.path.dirname(item),'*.json'))
        if file_dep:
            target =   os.path.join(os.path.dirname(item),os.path.basename(os.path.dirname(item))+'_labelme.csv')           
            yield {
                'name':item,
                'actions':[process_labelme],
                'targets':[target],
                'clean':True,
                'uptodate':[True],
                
            }
    
def task_calculate_survey_areas():
    def poly_to_points(polygon):
        return np.dstack(polygon.exterior.coords.xy)
    
    def survey_area(grp):
        # switch to using image locations
        #p=MultiPoint(np.hstack(grp['ImagePolygon'].apply(poly_to_points))[0]).convex_hull
        p =MultiPoint(np.dstack((grp['ImageEasting'],grp['ImageNorthing']))[0]).convex_hull
        return p.area
    
    def load(file):
        data =pd.read_csv(file,index_col='TimeStamp',parse_dates=['TimeStamp'])
        crs = f'epsg:{int(data["UtmCode"][0])}'
        survey = data['SurveyId'][0]
        gdf = gp.GeoDataFrame(data, geometry=data.ImagePolygon.apply(shapely.wkt.loads),crs=crs)
        return {'SurveyId':data.SurveyId.min(),'Area':survey_area(gdf)/10000}


    def calculate_area(dependencies, targets):
        areas = [load(file) for file in dependencies]
        areas = pd.DataFrame(areas)
        areas.to_csv(targets[0],index=False)


         
    file_dep = glob.glob(os.path.join(config.geturl('output'),'**/*survey_data.csv'),recursive=True) + glob.glob(os.path.join(config.geturl('output'),'**/*survey_area.csv'),recursive=True)
    targets = os.path.join(config.geturl('process'),'areas.csv')
    return {
            'actions':[calculate_area],
            'file_dep':file_dep,
            'targets':[targets],
            'uptodate': [True],
            'clean': True,
    } 




@create_after(executed='process_labelme')        
def task_process_matchup():
    def process_labelmatch(dependencies, targets):
        datafile = glob.glob(os.path.join(os.path.dirname(dependencies[0]),'*_survey_area_data.csv'))
        if len(datafile)==0:
            datafile = glob.glob(os.path.join(os.path.dirname(dependencies[0]),'*_survey_area.csv'))
        if len(datafile)==0:
            datafile = glob.glob(os.path.join(os.path.dirname(dependencies[0]),'*_survey_data.csv'))
        
        source_file = pd.read_csv(datafile[0])
        lableme = pd.read_csv(dependencies[0])
        source_file =source_file.set_index(source_file.NewName.apply(lambda x:os.path.splitext(x)[0]))
        lableme = lableme.set_index(lableme.FilePath.apply(lambda x: os.path.splitext(os.path.basename(x))[0]))
        output =lableme.join(source_file)
        output.index.name='Key'     
        output[~output.UtmCode.isna()].to_csv(targets[0])

    file_dep =  glob.glob(os.path.join(config.geturl('output'),'**/*_labelme.csv'),recursive=True)
    for item in file_dep:
        target = item.replace('.csv','_merge.csv')       
        yield {
            'name': item,
            'file_dep':[item],
            'actions':[process_labelmatch],
            'targets':[target],
            'clean':True,
        }  

@create_after(executed='process_matchup')
def task_process_mergelabel():
        def process_mergelabel(dependencies, targets):
            os.makedirs(os.path.dirname(targets[0]),exist_ok=True)
            files = list(filter(lambda x:os.path.getsize(x)>10, dependencies))
            data = pd.concat([pd.read_csv(file) for file in files])
            data.to_csv(targets[0],index=False)       
            
        file_dep = glob.glob(os.path.join(config.geturl('output'),'AU/**/*_labelme_merge.csv'),recursive=True)
        if file_dep:
                target = os.path.join(config.geturl('process'),'mergelabelme.csv')        
                return {
                    'actions':[process_mergelabel],
                    'file_dep':file_dep,
                    'targets':[target],
                    'clean':True,
                }         

@create_after(executed='process_mergelabel')    
def task_process_turtle_mergelabel():
        def process_turtlelabel(dependencies, targets):
            os.makedirs(os.path.dirname(targets[0]),exist_ok=True)
            data = pd.read_csv(dependencies[0],index_col='Key')
            data[data.label.str.contains('turtle')].to_csv(targets[0])       
            
        file_dep = os.path.join(config.geturl('process'),'mergelabelme.csv')  
        target =  os.path.join(config.geturl('process'),'mergelabelme_turtles.csv')       
        return {
            'actions':[process_turtlelabel],
            'file_dep':[file_dep],
            'targets':[target],
            'clean':True,
        }     

@create_after(executed='process_turtle_mergelabel')  
def task_process_turtle_stats():
        def process_turtlelabel(dependencies, targets):
            os.makedirs(os.path.dirname(targets[0]),exist_ok=True)
            data = pd.read_csv(dependencies[0],index_col='Key')
            stats =data.groupby('label')['label'].count()
            stats.to_csv(targets[0])       
        file_dep = os.path.join(config.geturl('process'),'mergelabelme.csv')  
        target =  os.path.join(config.geturl('process'),'labelme_stats.csv')       
        return {
            'actions':[process_turtlelabel],
            'file_dep':[file_dep],
            'targets':[target],
            'clean':True,
        }   
    
# def task_make_yolo_training_images():
#     def process_yoloset(dependencies, targets):
#         os.makedirs(os.path.dirname(targets[0]),exist_ok=True)
#         imagedir = os.path.join(os.path.dirname(targets[0]),'images')
#         os.makedirs(imagedir,exist_ok=True)
#         sourcefile = pd.read_csv(dependencies[0])
#         good =sourcefile[~sourcefile.label.isin(['done','don,e','gcp'])]
#         good.points = good.points.apply(ast.literal_eval)
#         classes = list(good.label.unique())
#         classes.sort()
#         images =good.groupby('FilePath')
#         for file,data in images:
#             Imgdest = os.path.join(imagedir,os.path.splitext(os.path.basename(file))[0]+'.JPG')
#             if not os.path.exists(Imgdest):
#                 shutil.copy(os.path.splitext(file)[0]+'.JPG',
#                             os.path.join(imagedir,os.path.splitext(os.path.basename(file))[0]+'.JPG'))
#             jsondest = os.path.join(imagedir,os.path.splitext(os.path.basename(file))[0]+'.json')
#             if not os.path.exists(jsondest):
#                 shapes =loadshapes(file)
#                 shapes.drop(['FilePath','flags','group_id'],axis=1)[~shapes.label.isin(['done','don,e','gcp'])].to_json(jsondest)
#             txtdest = os.path.join(imagedir,os.path.splitext(os.path.basename(file))[0]+'.TXT')
#             if not os.path.exists(txtdest):
#                 with open(txtdest,'w') as datafile:
#                     for index,row in data.iterrows():
#                         cla = classes.index(row.label)
#                         points = row.points
#                         if len(points)==2: #one animal
#                             datafile.write(f'{classes.index(row.label)} {points[0][0]/row.ImageWidth} {points[0][1]/row.ImageHeight} '\
#                                             f'{2*abs(points[0][0]-points[1][0])/row.ImageWidth}' \
#                                             f'{2*abs(points[0][1]-points[1][1])/row.ImageHeight}\n')
                    
                 
                 
#         with open(targets[0],'w') as obnames:
#             obnames.writelines(map(lambda x:x+'\n', classes))
        
#     config = {"config": get_var('config', 'NO')}
#     with open(config['config'], 'r') as ymlfile:
#         cfg = yaml.load(ymlfile, yaml.SafeLoader)
#     basepath = os.path.dirname(config['config'])
#     file_dep = os.path.join(basepath,cfg['paths']['process'],'labelmematchup.csv')
#     target = os.path.join(basepath,cfg['paths']['output'],'yolo','data','obj.names')        
#     return {
#         'actions':[process_yoloset],
#         'file_dep':[file_dep],
#         'targets':[target],
#         'clean':True,
#     }   
    
    
# def task_crop_training_images():
#     def process_crop(dependencies, targets,size=256):
#         basepath = os.path.dirname(targets[0])
#         os.makedirs(basepath,exist_ok=True)
#         sourcefile = pd.read_csv(dependencies[0])
#         good =sourcefile[~sourcefile.label.isin(['done','don,e','gcp'])]
#         good.points = good.points.apply(ast.literal_eval)
#         classes = list(good.label.unique())
#         list(map(lambda x:os.makedirs(os.path.join(basepath,x),exist_ok=True), classes))
#         classes.sort()
#         images =good.groupby('FilePath')
#         for file,data in images:
#                 imageObject = Image.open(os.path.splitext(file)[0]+'.JPG')
#                 imagebase = os.path.splitext(os.path.basename(file))[0]
#                 counter=0
#                 for index,row in data.iterrows():
#                     cla = classes.index(row.label)
#                     points = row.points
#                     if len(points)==2: #one animal
#                         counter = counter+1
#                         output =os.path.join(basepath,row.label,f'{imagebase}_{counter:03d}.JPG')
#                         cropped = imageObject.crop((int(points[0][0] - size/2),
#                                                    int(points[0][1] - size/2),
#                                                    int(points[0][0] + size/2),
#                                                    int(points[0][1] + size/2)))

#                         cropped.save(output)                    
#         with open(targets[0],'w') as obnames:
#             obnames.writelines(map(lambda x:x+'\n', classes))
        
#     config = {"config": get_var('config', 'NO')}
#     with open(config['config'], 'r') as ymlfile:
#         cfg = yaml.load(ymlfile, yaml.SafeLoader)
#     basepath = os.path.dirname(config['config'])
#     file_dep = os.path.join(basepath,cfg['paths']['process'],'labelmematchup.csv')
#     target = os.path.join(basepath,cfg['paths']['output'],'train','obj.names')        
#     return {
#         'actions':[process_crop],
#         'file_dep':[file_dep],
#         'targets':[target],
#         'clean':True,
#     }     


# def task_make_training_images():
#     def process_yoloset(dependencies, targets):
#         os.makedirs(os.path.dirname(targets[0]),exist_ok=True)
#         imagedir = os.path.join(os.path.dirname(targets[0]),'images')
#         os.makedirs(imagedir,exist_ok=True)
#         sourcefile = pd.read_csv(dependencies[0])
#         good =sourcefile[~sourcefile.label.isin(['done','don,e','gcp'])]
#         good.points = good.points.apply(ast.literal_eval)
#         classes = list(good.label.unique())
#         classes.sort()
#         images =good.groupby('FilePath')
#         for file,data in images:
#             Imgdest = os.path.join(imagedir,os.path.splitext(os.path.basename(file))[0]+'.JPG')
#             if not os.path.exists(Imgdest):
#                 shutil.copy(os.path.splitext(file)[0]+'.JPG',
#                             os.path.join(imagedir,os.path.splitext(os.path.basename(file))[0]+'.JPG'))
#             jsondest = os.path.join(imagedir,os.path.splitext(os.path.basename(file))[0]+'.json')
#             if not os.path.exists(jsondest):
#                 shapes =loadshapes(file)
#                 shapes.drop(['FilePath','flags','group_id'],axis=1)[~shapes.label.isin(['done','don,e','gcp'])].to_json(jsondest)
#             txtdest = os.path.join(imagedir,os.path.splitext(os.path.basename(file))[0]+'.TXT')
#             if not os.path.exists(txtdest):
#                 with open(txtdest,'w') as datafile:
#                     for index,row in data.iterrows():
#                         cla = classes.index(row.label)
#                         points = row.points
#                         if len(points)==2: #one animal
#                             datafile.write(f'{classes.index(row.label)} {points[0][0]/row.ImageWidth} {points[0][1]/row.ImageHeight} '\
#                                             f'{2*abs(points[0][0]-points[1][0])/row.ImageWidth}' \
#                                             f'{2*abs(points[0][1]-points[1][1])/row.ImageHeight}\n')
                    
                 
                 
#         with open(targets[0],'w') as obnames:
#             obnames.writelines(map(lambda x:x+'\n', classes))
        
#     config = {"config": get_var('config', 'NO')}
#     with open(config['config'], 'r') as ymlfile:
#         cfg = yaml.load(ymlfile, yaml.SafeLoader)
#     basepath = os.path.dirname(config['config'])
#     file_dep = os.path.join(basepath,cfg['paths']['process'],'labelmematchup.csv')
#     target = os.path.join(basepath,cfg['paths']['output'],'yolo','obj.names')        
#     return {
#         'actions':[process_yoloset],
#         'file_dep':[file_dep],
#         'targets':[target],
#         'clean':True,
#     }      
    
# def task_process_movefiles():
#     def process_move(dependencies, targets,outputpath):
#         files = pd.read_csv(dependencies[0])
#         files['Destination'] = ''
#         for index,row in files.iterrows():
#             row['Destination'] = os.path.join(outputpath,row.SurveyId,os.path.splitext(row.NewName)[0])+ \
#                 os.path.splitext(row.FilePath)[1] 
#             if os.path.exists(row.FilePath): 
#                 shutil.move(row.FilePath,row.Destination)
#         files.to_csv(targets[0])
#     config = {"config": get_var('config', 'NO')}
#     with open(config['config'], 'r') as ymlfile:
#         cfg = yaml.load(ymlfile, yaml.SafeLoader)
#     basepath = os.path.dirname(config['config'])
#     file_dep = os.path.join(basepath,cfg['paths']['process'],'labelmematchup.csv')
#     target =os.path.join(basepath,cfg['paths']['process'],'labelmematchup_final.csv')
#     return {
#         'actions':[(process_move, [],
#                     {'outputpath':os.path.join(cfg['paths']['output'],cfg['survey']['country'])})],
#         'file_dep':[file_dep],
#         'targets':[target],
#         'clean':True,
#     }        

def task_calculate_positions():  
    def process_positions(dependencies, targets):
        def calcRealworld(item):
            #-14.772 hieght at Exmouth
            localdrone = P4rtk(cal,crs)

            if 'Eastingrtk' in item and not np.isnan(item.Eastingrtk):
                localdrone.setdronepos(item.Eastingrtk,item.Northingrtk,item.RelativeAltitude, #item.EllipsoideHight+20,
                                (90+item.GimbalPitchDegree),item.GimbalRollDegree,item.GimbalYawDegree) #item.GimbalPitchDegree-90,item.GimbalRollDegree,-item.GimbalYawDegree
            elif  'EastingMrk' in item and not np.isnan(item.EastingMrk):
                localdrone.setdronepos(item.EastingMrk,item.NorthingMrk,item.RelativeAltitude,
                                (90+item.GimbalPitchDegree),item.GimbalRollDegree,item.GimbalYawDegree)
            else:
                localdrone.setdronepos(item.Easting,item.Northing,item.RelativeAltitude,
                                (90+item.GimbalPitchDegree),item.GimbalRollDegree,item.GimbalYawDegree) #item.GimbalPitchDegree-90,item.GimbalRollDegree,-item.GimbalYawDegree
                
            pos=localdrone.cameratorealworld(item.DewarpX,item.DewarpY)
            item.EastingPntD = pos[0]
            item.NorthingPntD = pos[1]
            pos=localdrone.cameratorealworld(item.JpegX,item.JpegY)
            #pos=localdrone.cameratorealworld(item.DewarpX,item.DewarpY)
            item.EastingPntJ = pos[0]
            item.NorthingPntJ = pos[1]
            return item           

        drone = pd.read_csv(dependencies[0]).dropna(how='all',axis=1)
        if len(drone)>0:
            #drone=drone[~drone.label.isin(['done','don,e'])]
            drone.points = drone.points.apply(ast.literal_eval)
            data = np.array([3706.080000000000,3692.930000000000,-34.370000000000,-34.720000000000,-0.271104000000,0.116514000000,0.001092580000,0.000348025000,-0.040583200000])
            crs = f'epsg:{int(drone["UtmCode"].min())}'
            p4rtk = P4rtk(cal,crs)
            drone = drone[drone.points.apply(len)>0]
            #drone[['PointEasting','PontNorthing']]=drone.apply(process_row,axis=1,result_type='expand')
            jpegpoints = np.array(drone.points.apply(lambda x:x[0]).tolist())
            drone[['JpegX','JpegY']] =np.floor(jpegpoints)
            corrected =p4rtk.jpegtoreal(jpegpoints)
            drone[['DewarpX','DewarpY']] =corrected
            drone[['EastingPntJ','NorthingPntJ','EastingPntD','NorthingPntD']] = 0.
            if 'EllipsoideHight' in drone.columns and  len(drone.loc[~drone['EllipsoideHight'].isna(),'EllipsoideHight'])>0:
                drone.loc[~drone['EllipsoideHight'].isna(),'EllipsoideHight']= pd.to_numeric(drone.loc[~drone['EllipsoideHight'].isna(),'EllipsoideHight'].str.split(',',expand=True)[0])
            drone = drone.apply(calcRealworld,axis=1)
        drone.to_csv(targets[0],index=False) 
    dewarp = pd.to_numeric(config.cfg['survey']['dewarp'] )
    cal = pd.to_numeric(config.cfg['survey']['calibration'] )
    file_dep =  glob.glob(os.path.join(config.geturl('output'),'**/*labelme_merge.csv'),recursive=True)
    for item in file_dep:
        target =item.replace("merge.csv","merge_points.csv")
        yield {
            'name':item,
            'actions':[process_positions],
            'file_dep':[item],
            'targets':[target],
            'clean':True,
        }   

@create_after(executed='calculate_positions')  
def task_process_gcp():
    def process_turtles(dependencies, targets):
        def count_turtle(grp):
            clustering = MeanShift(bandwidth=10).fit(np.dstack([grp.EastingPntD.values,grp.NorthingPntD.values])[0])
            n_clusters_ = len(clustering.cluster_centers_)
            fig,ax = plt.subplots(figsize=(8,8))
            for index,row in grp.iterrows():
                (x,y)=row.ImagePolygon.exterior.xy
                ax.plot(x, y, color='#6699cc', alpha=0.7,
                    linewidth=3, solid_capstyle='round', zorder=2)
                ax.plot(row.EastingPntD,row.NorthingPntD,marker ='x',linestyle='')
            ax.set_aspect(1)
            plt.scatter(clustering.cluster_centers_[:, 0], clustering.cluster_centers_[:, 1], c='red', s=50);
            plt.savefig(os.path.join(plotpath,f'group_plot_{grp.groups.min()}'))
            plt.close()
            return {'count':n_clusters_,'centers':clustering.cluster_centers_}
        plotpath =os.path.dirname(targets[0])
        drone =pd.read_csv(dependencies[0],parse_dates=['TimeStamp'])
        # ["dolphin","fish_school","gcp","mammal","ray","shark","turtle_deep","turtle_diving","turtle_jbs","turtle_surface","turtle_tracks"]
        drone = drone[drone.label.isin(["gcp"])]
        if len(drone)>0:
            drone['ImagePolygon']=drone.ImagePolygon.apply(shapely.wkt.loads)
            drone.sort_values('TimeStamp',inplace=True)
            drone['groups']=0
            drone.loc[abs(drone.TimeStamp.diff().dt.total_seconds())>12,'groups']=1
            drone['groups']=drone['groups'].cumsum()
            tcounts =pd.DataFrame(drone.groupby('groups').apply(count_turtle),columns=['turtle_count'])
            counts=tcounts.turtle_count.apply(lambda x:x['count']).reset_index()     
            centers =tcounts.turtle_count.apply(lambda x:x['centers']).reset_index()   
            centers.name= 'centers'
            counts.name = 'counts'
            output1 =pd.merge(drone,counts,on=['groups'])
            output1 =pd.merge(output1,centers,on=['groups'])
            output1.to_csv(targets[0],index=True)
        else:
            drone.to_csv(targets[0],index=True)
        

    file_dep =  glob.glob(os.path.join(config.geturl('output'),'**/*labelme_merge_points.csv'),recursive=True)
    file_dep =list(filter(lambda x:os.stat(x).st_size > 100,file_dep))
    for item in file_dep:
        target =item.replace("points.csv","points_grouped.csv")
        yield {
            'name':target,
            'actions':[process_turtles],
            'file_dep':[item],
            'targets':[target],
            'clean':True,
        }          

# @create_after(executed='calculate_positions')  
# def task_process_turtles():
#     def process_turtles(dependencies, targets):
#         def count_turtle(grp):
#             clustering = MeanShift(bandwidth=10).fit(np.dstack([grp.EastingPntD.values,grp.NorthingPntD.values])[0])
#             n_clusters_ = len(clustering.cluster_centers_)
#             fig,ax = plt.subplots(figsize=(8,8))
#             for index,row in grp.iterrows():
#                 (x,y)=row.ImagePolygon.exterior.xy
#                 ax.plot(x, y, color='#6699cc', alpha=0.7,
#                     linewidth=3, solid_capstyle='round', zorder=2)
#                 ax.plot(row.EastingPntD,row.NorthingPntD,marker ='x',linestyle='')
#             ax.set_aspect(1)
#             plt.scatter(clustering.cluster_centers_[:, 0], clustering.cluster_centers_[:, 1], c='red', s=50);
#             plt.savefig(os.path.join(plotpath,f'group_plot_{grp.groups.min()}'))
#             plt.close()
#             return {'count':n_clusters_,'centers':clustering.cluster_centers_}
#         plotpath =os.path.dirname(targets[0])
#         drone =pd.read_csv(dependencies[0],parse_dates=['TimeStamp'])
#         # ["dolphin","fish_school","gcp","mammal","ray","shark","turtle_deep","turtle_diving","turtle_jbs","turtle_surface","turtle_tracks"]
#         drone = drone[drone.label.isin(["turtle_diving","turtle_jbs","turtle_surface"])]
#         if len(drone)>0:
#             drone['ImagePolygon']=drone.ImagePolygon.apply(shapely.wkt.loads)
#             drone.sort_values('TimeStamp',inplace=True)
#             drone['groups']=0
#             drone.loc[abs(drone.TimeStamp.diff().dt.total_seconds())>12,'groups']=1
#             drone['groups']=drone['groups'].cumsum()
#             tcounts =pd.DataFrame(drone.groupby('groups').apply(count_turtle),columns=['turtle_count'])
#             counts=tcounts.turtle_count.apply(lambda x:x['count']).reset_index()     
#             centers =tcounts.turtle_count.apply(lambda x:x['centers']).reset_index()   
#             centers.name= 'centers'
#             counts.name = 'counts'
#             output1 =pd.merge(drone,counts,on=['groups'])
#             output1 =pd.merge(output1,centers,on=['groups'])
#             output1.to_csv(targets[0],index=True)
#         else:
#             drone.to_csv(targets[0],index=True)
        

#     file_dep =  glob.glob(os.path.join(config.geturl('output'),'**/*labelme_merge_points.csv'),recursive=True)
#     file_dep =list(filter(lambda x:os.stat(x).st_size > 100,file_dep))
#     for item in file_dep:
#         target =item.replace("points.csv","points_grouped.csv")
#         yield {
#             'name':target,
#             'actions':[process_turtles],
#             'file_dep':[item],
#             'targets':[target],
#             'clean':True,
#         }   


# # @create_after(executed='process_turtles') 
# # def task_process_turtles_totals():
# #     def from_np_array(array_string):
# #         array_string = ','.join(array_string.replace('[ ', '[').split())
# #         return np.array(ast.literal_eval(array_string))
    
# #     def process_turtles_totals(dependencies, targets):
# #         def process_sruvey(grp):
# #             crs = f'epsg:{int(grp["UtmCode"].min())}'
# #             utmproj =Proj(crs)
# #             points = np.vstack(grp.turtle_count_y)
# #             data =pd.DataFrame(points,columns=['Easting','Norting'])
# #             data['Longitude'],data['Latitude']=utmproj(points[:,0],points[:,1],inverse=True)
# #             data['SurveyId'] = grp['SurveyId'].min()
# #             return data

# #         drone =pd.read_csv(dependencies[0],parse_dates=['TimeStamp'],converters={'turtle_count_y': from_np_array})
# #         if len(drone)>0:
# #             drone = drone[drone.label.isin(['turtle_surface','turtle_jbs'])]
# #             turtles =drone.groupby('groups').first().groupby('SurveyId').apply(process_sruvey)
# #             turtles.to_csv(targets[0],index=False)
# #         else:
# #             with open(targets[0], "w") as outfile:
# #                 outfile.write("Easting,Norting,Longitude,Latitude,SurveyId\n") 
                   

# #     file_dep =  glob.glob(os.path.join(config.geturl('output'),'**/*points_grouped.csv'),recursive=True)
# #     for item in file_dep:
# #         target =item.replace("grouped.csv","grouped_turtle.csv")
# #         yield {
# #             'name':target,
# #             'actions':[process_turtles_totals],
# #             'file_dep':[item],
# #             'targets':[target],
# #             'clean':True,
# #         }   
# # @create_after(executed='process_turtles_totals')             
# # def task_merge_turtle_totals():
# #     def process_merge(dependencies, targets):
# #         totals = pd.concat([pd.read_csv(file) for file in dependencies])
# #         totals.to_csv(targets[0],index=False)
# #     file_dep =  glob.glob(os.path.join(config.geturl('output'),'**/*points_grouped_turtle.csv'),recursive=True)
# #     target =os.path.join(config.geturl('process'),'turtles_totals.csv')
# #     return {
# #         'actions':[process_merge],
# #         'file_dep':file_dep,
# #         'targets':[target],
# #         'clean':True,
# #     }  

# # def task_plot_turtles():
# #         def process_survey(dependencies, targets,apikey):
# #             drone =pd.read_csv(list(dependencies)[0])
            
# #             px.set_mapbox_access_token(apikey)
# #             fig = px.scatter_mapbox(drone, hover_name='SurveyId', lat="Latitude", lon="Longitude",  
# #                                     mapbox_style="satellite-streets",color="SurveyId", size_max=30, zoom=10)
# #             fig.update_layout(mapbox_style="satellite-streets")
# #             plotly.offline.plot(fig, filename=list(targets)[0],auto_open = False)
            

# #         file_dep =os.path.join(config.geturl('process'),'turtles_totals.csv')
# #         targets = os.path.join(config.geturl('process'),'turtles_totals.html')
# #         return {

# #             'actions':[(process_survey, [],{'apikey':config.cfg['mapboxkey']})],
# #             'file_dep':[file_dep],
# #             'targets':[targets],
# #             'clean':True,
# #         }   


# # def task_plot_each_survey():
# #     def process_survey(dependencies, targets,apikey):
# #         drone =pd.read_csv(list(dependencies)[0])
# #         px.set_mapbox_access_token(apikey)
# #         if len(drone)>2:
# #             max_bound = max(abs(drone.Longitude.max()-drone.Longitude.min()), abs(drone.Latitude.max()-drone.Latitude.min())) * 111
# #             if max_bound>0:
# #                 zoom = 13.5 - np.log(max_bound)
# #             else:
# #                 zoom = 13.5
# #         else:
# #             zoom =14

# #         fig = px.scatter_mapbox(drone, lat="Latitude", lon="Longitude",  
# #                                 mapbox_style="satellite-streets",color="SurveyId", size_max=30, zoom=zoom)        
# #         fig.update_layout(mapbox_style="satellite-streets",autosize=False)
# #         html_file =list(filter(lambda x: 'html' in x, targets))[0]
# #         png_file =list(filter(lambda x: 'png' in x, targets))[0]
# #         plotly.offline.plot(fig, filename=html_file,auto_open = False)
# #         fig.update_layout(coloraxis_showscale=False,showlegend=False,autosize=False,margin = dict(t=10, l=10, r=10, b=10))
# #         fig.write_image(png_file)
        
        
# #     file_dep = glob.glob(os.path.join(config.geturl('output'),'**/*points_grouped_turtle.csv'),recursive=True)
# #     for item in file_dep:
# #         target = [item.replace('csv','png'),item.replace('csv','html')]
# #         yield {
# #             'name':target[0],
# #             'actions':[(process_survey, [],{'apikey':config.cfg['mapboxkey']})],
# #             'file_dep':[item],
# #             'targets':target,
# #             'clean':True,
# #         }   

# # def task_process_areas():
# #     def process_area(dependencies, targets):
# #         totals = pd.concat([pd.read_csv(file) for file in dependencies])
# #         output = totals.groupby('SurveyId').first()
# #         output.to_csv(targets[0],index=False)
# #     file_dep =  glob.glob(os.path.join(config.geturl('output'),'**/**_survey_area_data*.csv'),recursive=True)
# #     targets = os.path.join(config.geturl('process'),'areas.csv')
# #     return {

# #         'actions':[process_area],
# #         'file_dep':file_dep,
# #         'targets':[targets],
# #         'clean':True,
# #     } 

# def task_plot_surveys():
#     def process_survey(dependencies, targets,apikey):
#         drone =pd.read_csv(list(dependencies)[0],index_col='TimeStamp',parse_dates=['TimeStamp'])
#         drone = drone[drone.Longitude>0]
#         px.set_mapbox_access_token(apikey)
#         max_bound = max(abs(drone.Longitude.max()-drone.Longitude.min()), abs(drone.Latitude.max()-drone.Latitude.min())) * 111
#         zoom = 11.5 - np.log(max_bound)
#         fig = px.scatter_mapbox(drone, hover_name='SurveyId', lat="Latitude", lon="Longitude",  
#                                 mapbox_style="satellite-streets",color="SurveyId", size_max=30, zoom=zoom)
#         fig.update_layout(mapbox_style="satellite-streets")
#         os.makedirs(os.path.dirname(targets[0]),exist_ok=True)
#         html_file =list(filter(lambda x: 'html' in x, targets))[0]
#         png_file =list(filter(lambda x: 'png' in x, targets))[0]
#         plotly.offline.plot(fig, filename=html_file,auto_open = False)
#         fig.write_image(png_file)
#     file_dep = os.path.join(config.geturl('process'),'surveyswitharea.csv')
#     targets = [os.path.join(config.geturl('reports'),'surveys.html'),
#                os.path.join(config.geturl('reports'),'surveys.png')]


#     file_dep =  glob.glob(os.path.join(config.cfg['paths']['output'],config.cfg['survey']['country'],'**/*points_grouped.csv'),recursive=True)
#     for item in file_dep:
#         target =item.replace("grouped.csv","grouped_turtle.csv")          
#         yield {

#             'actions':[(process_survey, [],{'apikey':config.cfg['mapboxkey']})],
#             'file_dep':[file_dep],
#             'targets':targets,
#             'clean':True,
#         } 

        
# def task_turtles_report():
#         def process_survey(dependencies, targets):



#             turte_file = pd.read_csv(list(filter(lambda x: 'turtles_totals' in x, dependencies))[0])



#             turte_file['TotalTurtles'] =0
#             turtle = turte_file.groupby('SurveyId').count().reset_index(-1)
#             turtle.SurveyId =turtle.SurveyId.apply(lambda x: x.replace('AU_',''))
#             counts = turtle[['SurveyId','TotalTurtles']]
#             image_area = pd.read_csv(list(filter(lambda x: 'areas.csv' in x, dependencies))[0],index_col='SurveyId')
#             #image_area['SurveyId'] =image_area.NewName.str.split('_',expand=True)[[3,4]].apply(lambda x :"_".join(x)[:-2],axis=1)
#             #image_area = image_area.groupby('SurveyId').first().reset_index(-1)
#             #image_area = image_area[['SurveyId','SurveyAreaHec']].set_index('SurveyId')
#             counts = counts.set_index('SurveyId')
#             output =image_area.join(counts)
#             output['TurtlesPerHec'] = output['TotalTurtles']/output['Area']
#             output.to_csv(targets[0],index=True)

#         file_dep =[os.path.join(config.geturl('process'),'turtles_totals.csv'),
#                     os.path.join(config.geturl('process'),'areas.csv')]
#         os.makedirs(config.geturl('reports'),exist_ok=True)
#         targets = os.path.join(config.geturl('reports'),'turtles_per_survey.csv')
#         return {

#             'actions':[process_survey],
#             'file_dep':file_dep,
#             'targets':[targets],
#             'clean':True,
#         }          

if __name__ == '__main__':
    import doit

    #print(globals())
    doit.run(globals())