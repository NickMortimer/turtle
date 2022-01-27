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
            
#         }
def loadshapes(file):
    lines =[]
    with open(file, "r") as read_file:
        while read_file:
            line =read_file.readline()
            if "imagePath" in line:
                break
            lines.append(line)
    lines[-1] ='  ]}\n'
    data = json.loads(''.join(lines).replace("\n", "").replace("'", '"').replace('u"', '"'))
    data =pd.DataFrame(data['shapes'])
    data['FilePath'] =file             
    return(data)

def task_process_labelme():
 
        def process_labelme(dependencies, targets):
            jsonfiles = glob.glob(os.path.join(os.path.dirname(targets[0]),'*.json'))
            if jsonfiles:
                data = pd.concat([loadshapes(file) for file in jsonfiles])
            else:
                data = pd.DataFrame()
            data.to_csv(targets[0],index=False)       
            
        config = {"config": get_var('config', 'NO')}
        with open(config['config'], 'r') as ymlfile:
            cfg = yaml.load(ymlfile, yaml.SafeLoader)
        basepath = os.path.dirname(config['config'])
        for item in glob.glob(os.path.join(basepath,cfg['paths']['labelmesource'])):
            #file_dep = glob.glob(os.path.join(os.path.dirname(item),'*.json'))
            target =   os.path.join(os.path.dirname(item),'labelme.csv')           
            yield {
                'name':item,
                'actions':[process_labelme],
                'targets':[target],
                'clean':True,
                'uptodate':[True],
                
            }
        
        
def task_process_mergelabel():
        def process_mergelabel(dependencies, targets):
            files = list(filter(lambda x:os.path.getsize(x)>10, dependencies))
            data = pd.concat([pd.read_csv(file) for file in files])
            data.to_csv(targets[0],index=False)       
            
        config = {"config": get_var('config', 'NO')}
        with open(config['config'], 'r') as ymlfile:
            cfg = yaml.load(ymlfile, yaml.SafeLoader)
        basepath = os.path.dirname(config['config'])
        file_dep = glob.glob(os.path.join(basepath,cfg['paths']['labelmesource'],'labelme.csv'),recursive=True)
        target = os.path.join(basepath,cfg['paths']['process'],'mergelabelme.csv')        
        return {
            'actions':[process_mergelabel],
            'file_dep':file_dep,
            'targets':[target],
            'clean':True,
        }
        
def task_process_matchup():
    def process_labelmatch(dependencies, targets):
        source_file = pd.concat([pd.read_csv(file) for file in filter(lambda x: '_survey.csv' in x, dependencies)])
        lableme = pd.concat([pd.read_csv(file) for file in filter(lambda x: 'mergelabelme' in x, dependencies)])
        source_file.index = source_file.NewName.apply(lambda x:os.path.splitext(x)[0])
        lableme.index = lableme.FilePath.apply(lambda x: os.path.splitext(os.path.basename(x))[0])
        output =lableme.join(source_file)
        output.index.name='Key'
        output.to_csv(targets[0])
    config = {"config": get_var('config', 'NO')}
    with open(config['config'], 'r') as ymlfile:
        cfg = yaml.load(ymlfile, yaml.SafeLoader)
    basepath = os.path.dirname(config['config'])
    file_dep =  glob.glob(os.path.join(basepath,cfg['paths']['process'],'*_survey.csv'),recursive=True)
    file_dep.append(os.path.join(basepath,cfg['paths']['process'],'mergelabelme.csv'))
    
    
    target = os.path.join(basepath,cfg['paths']['process'],'labelmematchup.csv')        
    return {
        'actions':[process_labelmatch],
        'file_dep':file_dep,
        'targets':[target],
        'clean':True,
    }   
    
    
def task_make_yolo_training_images():
    def process_yoloset(dependencies, targets):
        os.makedirs(os.path.dirname(targets[0]),exist_ok=True)
        imagedir = os.path.join(os.path.dirname(targets[0]),'images')
        os.makedirs(imagedir,exist_ok=True)
        sourcefile = pd.read_csv(dependencies[0])
        good =sourcefile[~sourcefile.label.isin(['done','don,e','gcp'])]
        good.points = good.points.apply(ast.literal_eval)
        classes = list(good.label.unique())
        classes.sort()
        images =good.groupby('FilePath')
        for file,data in images:
            Imgdest = os.path.join(imagedir,os.path.splitext(os.path.basename(file))[0]+'.JPG')
            if not os.path.exists(Imgdest):
                shutil.copy(os.path.splitext(file)[0]+'.JPG',
                            os.path.join(imagedir,os.path.splitext(os.path.basename(file))[0]+'.JPG'))
            jsondest = os.path.join(imagedir,os.path.splitext(os.path.basename(file))[0]+'.json')
            if not os.path.exists(jsondest):
                shapes =loadshapes(file)
                shapes.drop(['FilePath','flags','group_id'],axis=1)[~shapes.label.isin(['done','don,e','gcp'])].to_json(jsondest)
            txtdest = os.path.join(imagedir,os.path.splitext(os.path.basename(file))[0]+'.TXT')
            if not os.path.exists(txtdest):
                with open(txtdest,'w') as datafile:
                    for index,row in data.iterrows():
                        cla = classes.index(row.label)
                        points = row.points
                        if len(points)==2: #one animal
                            datafile.write(f'{classes.index(row.label)} {points[0][0]/row.ImageWidth} {points[0][1]/row.ImageHeight} '\
                                            f'{2*abs(points[0][0]-points[1][0])/row.ImageWidth}' \
                                            f'{2*abs(points[0][1]-points[1][1])/row.ImageHeight}\n')
                    
                 
                 
        with open(targets[0],'w') as obnames:
            obnames.writelines(map(lambda x:x+'\n', classes))
        
    config = {"config": get_var('config', 'NO')}
    with open(config['config'], 'r') as ymlfile:
        cfg = yaml.load(ymlfile, yaml.SafeLoader)
    basepath = os.path.dirname(config['config'])
    file_dep = os.path.join(basepath,cfg['paths']['process'],'labelmematchup.csv')
    target = os.path.join(basepath,cfg['paths']['output'],'yolo','data','obj.names')        
    return {
        'actions':[process_yoloset],
        'file_dep':[file_dep],
        'targets':[target],
        'clean':True,
    }   
    
    
def task_crop_training_images():
    def process_crop(dependencies, targets,size=256):
        basepath = os.path.dirname(targets[0])
        os.makedirs(basepath,exist_ok=True)
        sourcefile = pd.read_csv(dependencies[0])
        good =sourcefile[~sourcefile.label.isin(['done','don,e','gcp'])]
        good.points = good.points.apply(ast.literal_eval)
        classes = list(good.label.unique())
        list(map(lambda x:os.makedirs(os.path.join(basepath,x),exist_ok=True), classes))
        classes.sort()
        images =good.groupby('FilePath')
        for file,data in images:
                imageObject = Image.open(os.path.splitext(file)[0]+'.JPG')
                imagebase = os.path.splitext(os.path.basename(file))[0]
                counter=0
                for index,row in data.iterrows():
                    cla = classes.index(row.label)
                    points = row.points
                    if len(points)==2: #one animal
                        counter = counter+1
                        output =os.path.join(basepath,row.label,f'{imagebase}_{counter:03d}.JPG')
                        cropped = imageObject.crop((int(points[0][0] - size/2),
                                                   int(points[0][1] - size/2),
                                                   int(points[0][0] + size/2),
                                                   int(points[0][1] + size/2)))

                        cropped.save(output)                    
        with open(targets[0],'w') as obnames:
            obnames.writelines(map(lambda x:x+'\n', classes))
        
    config = {"config": get_var('config', 'NO')}
    with open(config['config'], 'r') as ymlfile:
        cfg = yaml.load(ymlfile, yaml.SafeLoader)
    basepath = os.path.dirname(config['config'])
    file_dep = os.path.join(basepath,cfg['paths']['process'],'labelmematchup.csv')
    target = os.path.join(basepath,cfg['paths']['output'],'train','obj.names')        
    return {
        'actions':[process_crop],
        'file_dep':[file_dep],
        'targets':[target],
        'clean':True,
    }     


def task_make_training_images():
    def process_yoloset(dependencies, targets):
        os.makedirs(os.path.dirname(targets[0]),exist_ok=True)
        imagedir = os.path.join(os.path.dirname(targets[0]),'images')
        os.makedirs(imagedir,exist_ok=True)
        sourcefile = pd.read_csv(dependencies[0])
        good =sourcefile[~sourcefile.label.isin(['done','don,e','gcp'])]
        good.points = good.points.apply(ast.literal_eval)
        classes = list(good.label.unique())
        classes.sort()
        images =good.groupby('FilePath')
        for file,data in images:
            Imgdest = os.path.join(imagedir,os.path.splitext(os.path.basename(file))[0]+'.JPG')
            if not os.path.exists(Imgdest):
                shutil.copy(os.path.splitext(file)[0]+'.JPG',
                            os.path.join(imagedir,os.path.splitext(os.path.basename(file))[0]+'.JPG'))
            jsondest = os.path.join(imagedir,os.path.splitext(os.path.basename(file))[0]+'.json')
            if not os.path.exists(jsondest):
                shapes =loadshapes(file)
                shapes.drop(['FilePath','flags','group_id'],axis=1)[~shapes.label.isin(['done','don,e','gcp'])].to_json(jsondest)
            txtdest = os.path.join(imagedir,os.path.splitext(os.path.basename(file))[0]+'.TXT')
            if not os.path.exists(txtdest):
                with open(txtdest,'w') as datafile:
                    for index,row in data.iterrows():
                        cla = classes.index(row.label)
                        points = row.points
                        if len(points)==2: #one animal
                            datafile.write(f'{classes.index(row.label)} {points[0][0]/row.ImageWidth} {points[0][1]/row.ImageHeight} '\
                                            f'{2*abs(points[0][0]-points[1][0])/row.ImageWidth}' \
                                            f'{2*abs(points[0][1]-points[1][1])/row.ImageHeight}\n')
                    
                 
                 
        with open(targets[0],'w') as obnames:
            obnames.writelines(map(lambda x:x+'\n', classes))
        
    config = {"config": get_var('config', 'NO')}
    with open(config['config'], 'r') as ymlfile:
        cfg = yaml.load(ymlfile, yaml.SafeLoader)
    basepath = os.path.dirname(config['config'])
    file_dep = os.path.join(basepath,cfg['paths']['process'],'labelmematchup.csv')
    target = os.path.join(basepath,cfg['paths']['output'],'yolo','obj.names')        
    return {
        'actions':[process_yoloset],
        'file_dep':[file_dep],
        'targets':[target],
        'clean':True,
    }      
    
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
                localdrone = P4rtk(data,crs)
                localdrone.setdronepos(item.Eastingrtk,item.Northingrtk,item.EllipsoideHight+14.772,
                                  (90+item.GimbalPitchDegree)*-1,item.GimbalRollDegree,-item.GimbalYawDegree+8) #item.GimbalPitchDegree-90,item.GimbalRollDegree,-item.GimbalYawDegree
                pos=localdrone.cameratorealworld(item.DewarpX,item.DewarpY)
                item.EastingPntD = pos[0]
                item.NorthingPntD = pos[1]
                pos=localdrone.cameratorealworld(item.JpegX,item.JpegY)
                #pos=localdrone.cameratorealworld(item.DewarpX,item.DewarpY)
                item.EastingPntJ = pos[0]
                item.NorthingPntJ = pos[1]
                return item           

            drone = pd.read_csv(dependencies[0])
            #drone=drone[~drone.label.isin(['done','don,e'])]
            drone= drone[drone.label.isin(['turtle_surface','turtle_jbs','gcp'])].copy()
            drone.points = drone.points.apply(ast.literal_eval)
            data = np.array([3706.080000000000,3692.930000000000,-34.370000000000,-34.720000000000,-0.271104000000,0.116514000000,0.001092580000,0.000348025000,-0.040583200000])
            crs = f'epsg:{int(drone["UtmCode"].min())}'
            p4rtk = P4rtk(data,crs)
            #drone[['PointEasting','PontNorthing']]=drone.apply(process_row,axis=1,result_type='expand')
            jpegpoints = np.array(drone.points.apply(lambda x:x[0]).tolist())
            drone[['JpegX','JpegY']] =np.floor(jpegpoints)
            corrected =p4rtk.jpegtoreal(jpegpoints)
            drone[['DewarpX','DewarpY']] =corrected
            drone[['EastingPntJ','NorthingPntJ','EastingPntD','NorthingPntD']] = 0.
            drone['EllipsoideHight']= pd.to_numeric(drone['EllipsoideHight'].str.split(',',expand=True)[0])
            drone = drone.apply(calcRealworld,axis=1)
            drone.to_csv(targets[0],index=False) 

    config = {"config": get_var('config', 'NO')}
    with open(config['config'], 'r') as ymlfile:
        cfg = yaml.load(ymlfile, yaml.SafeLoader)
    basepath = os.path.dirname(config['config'])
    file_dep =os.path.join(basepath,cfg['paths']['process'],'labelmematchup.csv')
    target =os.path.join(basepath,cfg['paths']['process'],'labelmematchup_final_pointpos.csv')
    return {
        'actions':[process_positions],
        'file_dep':[file_dep],
        'targets':[target],
        'clean':True,
    }   

def task_process_turtles():

    
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
        
    config = {"config": get_var('config', 'NO')}
    with open(config['config'], 'r') as ymlfile:
        cfg = yaml.load(ymlfile, yaml.SafeLoader)
    basepath = os.path.dirname(config['config'])
    file_dep =os.path.join(basepath,cfg['paths']['process'],'labelmematchup_final_pointpos.csv')
    target =os.path.join(basepath,cfg['paths']['process'],'turtles_list.csv')
    return {
        'actions':[process_turtles],
        'file_dep':[file_dep],
        'targets':[target],
        'clean':True,
    }    



def task_process_turtles_totals():
    def from_np_array(array_string):
        array_string = ','.join(array_string.replace('[ ', '[').split())
        return np.array(ast.literal_eval(array_string))
    
    def process_turtles_totals(dependencies, targets):
        def process_sruvey(grp):
            crs = f'epsg:{int(grp["UtmCode"].min())}'
            utmproj =Proj(crs)
            points = np.vstack(grp.turtle_count_y)
            data =pd.DataFrame(points,columns=['Easting','Norting'])
            data['Longitude'],data['Latitude']=utmproj(points[:,0],points[:,1],inverse=True)
            data['SurveyId'] = grp['SurveyId'].min()
            return data

        drone =pd.read_csv(dependencies[0],parse_dates=['TimeStamp'],converters={'turtle_count_y': from_np_array})
        drone = drone[drone.label.isin(['turtle_surface','turtle_jbs'])]
        turtles =drone.groupby('groups').first().groupby('SurveyId').apply(process_sruvey)
        turtles.to_csv(targets[0],index=False)        
    config = {"config": get_var('config', 'NO')}
    with open(config['config'], 'r') as ymlfile:
        cfg = yaml.load(ymlfile, yaml.SafeLoader)
    basepath = os.path.dirname(config['config'])
    file_dep =os.path.join(basepath,cfg['paths']['process'],'turtles_list.csv')
    target =os.path.join(basepath,cfg['paths']['process'],'turtles_totals.csv')
    return {
        'actions':[process_turtles_totals],
        'file_dep':[file_dep],
        'targets':[target],
        'clean':True,
    }  

def task_plot_turtles():
        def process_survey(dependencies, targets,apikey):
            drone =pd.read_csv(list(dependencies)[0])
            
            px.set_mapbox_access_token(apikey)
            fig = px.scatter_mapbox(drone, hover_name='SurveyId', lat="Latitude", lon="Longitude",  
                                    mapbox_style="satellite-streets",color="SurveyId", size_max=30, zoom=10)
            fig.update_layout(mapbox_style="satellite-streets")
            plotly.offline.plot(fig, filename=list(targets)[0],auto_open = False)
            
        config = {"config": get_var('config', 'NO')}
        with open(config['config'], 'r') as ymlfile:
            cfg = yaml.load(ymlfile, yaml.SafeLoader)
        basepath = os.path.dirname(config['config'])
        file_dep =os.path.join(basepath,cfg['paths']['process'],'turtles_totals.csv')
        targets = os.path.join(basepath,cfg['paths']['process'],'turtles_totals.html')
        return {

            'actions':[(process_survey, [],{'apikey':cfg['mapboxkey']})],
            'file_dep':[file_dep],
            'targets':[targets],
            'clean':True,
        }   
        
def task_turtles_report():
        def process_survey(dependencies, targets):
            turte_file = pd.read_csv(list(filter(lambda x: 'turtles_totals' in x, dependencies))[0])
            turte_file['TotalTurtles'] =0
            turtle = turte_file.groupby('SurveyId').count()
            counts = turtle['TotalTurtles']
            image_area = pd.read_csv(list(filter(lambda x: 'image' in x, dependencies))[0],index_col='SurveyId')
            output =image_area.join(counts)
            output['TurtlesPerHec'] = output['TotalTurtles']/output['Area']
            output.to_csv(targets[0],index=True)
            

            
        config = {"config": get_var('config', 'NO')}
        with open(config['config'], 'r') as ymlfile:
            cfg = yaml.load(ymlfile, yaml.SafeLoader)
        basepath = os.path.dirname(config['config'])
        file_dep =[os.path.join(basepath,cfg['paths']['process'],'turtles_totals.csv'),
                   os.path.join(cfg['paths']['reports'],'image_coverage.csv')]
        targets = os.path.join(cfg['paths']['reports'],'turtles_per_survey.csv')
        return {

            'actions':[process_survey],
            'file_dep':file_dep,
            'targets':[targets],
            'clean':True,
        }          

if __name__ == '__main__':
    import doit

    #print(globals())
    doit.run(globals())