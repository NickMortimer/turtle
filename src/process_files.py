import os
import glob
import doit
import glob
import os
from bs4 import BeautifulSoup as b
from pyproj import Proj 
import numpy as np
import matplotlib.pyplot as plt
import yaml
import pandas as pd
from doit import get_var
import statsmodels.api as sm
from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely.geometry import MultiPoint
from bs4 import BeautifulSoup as bs
from sklearn.cluster import MeanShift
from functools import partial
import shapely.wkt
from matplotlib.patches import Circle




wanted ={"SourceFile","FileModifyDate","ImageDescription",
         "ExposureTime","FNumber","ExposureProgram","ISO",
         "DateTimeOriginal","Make","SpeedX","SpeedY","SpeedZ",
         "Pitch","Yaw","Roll","CameraPitch","CameraYaw","CameraRoll",
         "ExifImageWidth","ExifImageHeight","SerialNumber",
         "GPSLatitudeRef","GPSLongitudeRef","GPSAltitudeRef","AbsoluteAltitude",
         "RelativeAltitude","GimbalRollDegree","GimbalYawDegree",
         "GimbalPitchDegree","FlightRollDegree","FlightYawDegree",
         "FlightPitchDegree","CamReverse","GimbalReverse","CalibratedFocalLength",
         "CalibratedOpticalCenterX","CalibratedOpticalCenterY","ImageWidth",
         "ImageHeight","GPSAltitude","GPSLatitude","GPSLongitude","CircleOfConfusion",
         "FOV","Latitude",'Longitude'}

def task_process_json():
        def process_json(dependencies, targets):
            source_file = list(dependencies)[0]
            print('source file is: {0}'.format(source_file))
            print('output dir is: {0}'.format(list(targets)[0]))
            drone = pd.read_json(source_file)
            def get_longitude(item):
                longitude =float(item[0]) + float(item[2][0:-1])/60 + float(item[3][0:-1])/3600
                return (longitude)
            drone['Longitude'] = pd.np.NAN
            drone['Latitude'] = pd.np.NAN
            drone.loc[ ~drone['GPSLongitude'].isna(),'Longitude']=drone.loc[ ~drone['GPSLongitude'].isna(),'GPSLongitude'].str.split(' ',expand=True).apply(get_longitude,axis=1)
            drone.loc[ ~drone['GPSLatitude'].isna(),'Latitude']=drone.loc[ ~drone['GPSLatitude'].isna(),'GPSLatitude'].str.split(' ',expand=True).apply(get_longitude,axis=1)
            drone.loc[drone['GPSLatitudeRef']=='South','Latitude'] =drone.loc[drone['GPSLatitudeRef']=='South','Latitude']*-1
            drone = drone[wanted]
            drone['TimeStamp'] = pd.to_datetime(drone.DateTimeOriginal,format='%Y:%m:%d %H:%M:%S')
            drone.set_index('TimeStamp',inplace=True)
            drone.sort_index(inplace=True)
            drone = drone[pd.notna(drone.index)]
            os.makedirs(os.path.split(list(targets)[0])[0],exist_ok=True)
            drone.to_csv(list(targets)[0],index=True)
            
        config = {"config": get_var('con', 'NO')}
        with open(config['config'], 'r') as ymlfile:
            cfg = yaml.load(ymlfile, yaml.SafeLoader)

        return {'actions':[(process_json)],
                'file_dep':[cfg['locations']['input_exif']],
                'targets':[cfg['locations']['output_exif']],
                'verbosity':2,
                'clean': True,
                }

def task_process_surveys():
    def make_dir_name(project,starttime):
        return project + starttime.strftime('%Y%m%dT%H%M')
    def process_surveys(dependencies, targets,cfg):
        drone = pd.read_csv(list(dependencies)[0],index_col='TimeStamp',parse_dates=['TimeStamp'])
        myProj = Proj(cfg['processing']['proj'])
        drone['Easting'], drone['Northing'] = myProj(drone['Longitude'].values, drone['Latitude'].values)
        drone['DeltaT']=drone.index
        drone['DeltaT']=drone['DeltaT'].diff().dt.total_seconds()
        drone['Group'] = 0
        # group 
        drone.loc[drone['DeltaT']>cfg['processing']['min_time_delta'],'Group']=1
        drone['Group'] =drone['Group'].cumsum()
        drone['GroupCount'] = drone.groupby('Group')['Northing'].transform('count')
        drone = drone[drone['GroupCount']>cfg['processing']['min_pitures']]
        # group reprocess only big groups
        drone['Group'] = 0
        drone.sort_index(inplace=True)
        drone['DeltaT']=drone.index
        drone['DeltaT']=drone['DeltaT'].diff().dt.total_seconds()
        drone.loc[drone['DeltaT']>cfg['processing']['max_time_delta'],'Group']=1
        drone['Group'] =drone['Group'].cumsum()
        drone['AdjustedTime'] = drone.index
        drone['GroupCount'] = drone.groupby('Group')['Northing'].transform('count')
        drone['GroupId'] = cfg['processing']['project']+ drone.groupby('Group')['AdjustedTime'].transform('min').dt.strftime('%Y%m%dT%H%M')
        drone.to_csv(list(targets)[0],index=True)

        
    config = {"config": get_var('con', 'NO')}
    with open(config['config'], 'r') as ymlfile:
        cfg = yaml.load(ymlfile, yaml.SafeLoader)


    return {'actions':[(process_surveys, [],{'cfg':cfg})],
            'file_dep':[cfg['locations']['output_exif']],
            'targets':[cfg['locations']['output_survey_master']],
            'verbosity':2,
            'clean': True,
            }                
def task_process_build_surveys():
    def process_build_surveys(dependencies, targets,cfg):
        drone = pd.read_csv(list(dependencies)[0],index_col='TimeStamp',parse_dates=['TimeStamp'])
        grouped = drone.groupby('Group')
        for (name,data),target in zip(grouped,targets):
            os.makedirs(os.path.split(target)[0],exist_ok=True)
            data.to_csv(target,index=True)

        
    config = {"config": get_var('con', 'NO')}
    with open(config['config'], 'r') as ymlfile:
        cfg = yaml.load(ymlfile, yaml.SafeLoader)
    if os.path.exists(cfg['locations']['output_survey_master']):
        source = pd.read_csv(cfg['locations']['output_survey_master'],index_col='TimeStamp',parse_dates=['TimeStamp'])
        targets = list(cfg['locations']['output_survey']+source['GroupId'].unique()+
                      os.path.sep+'META_DATA' +os.path.sep+'DRN_XIF_'+
                      source['GroupId'].unique()+'.CSV')
        return {'actions':[(process_build_surveys, [],{'cfg':cfg})],
                'file_dep':[cfg['locations']['output_survey_master']],
                'targets':targets,
                'verbosity':2,
                'clean': True,
                'uptodate': [False],
                } 

import matplotlib.pyplot as plt
def task_process_plot_surveys():
    def process_plot_surveys(dependencies, targets):
        drone = pd.read_csv(list(dependencies)[0],index_col='TimeStamp',parse_dates=['TimeStamp'])
        plt.scatter(x=drone.Easting,y=drone.Northing)
        plt.savefig(list(targets)[0],dpi=300)
        plt.close()

        
    config = {"config": get_var('con', 'NO')}
    with open(config['config'], 'r') as ymlfile:
        cfg = yaml.load(ymlfile, yaml.SafeLoader)
    files =glob.glob(cfg['locations']['output_survey']+'/**/DRN_XIF*.CSV',recursive=True)
    for file in files:
        yield { 'name': 'plot1:{0}'.format(os.path.basename(file)),
                'actions':[(process_plot_surveys)],
                'file_dep':[file],
                'targets':[os.path.splitext(file)[0]+'.JPG'],
                'verbosity':2,
                'clean': True
                } 



def task_filter_surveys():
    def filter_low_high(x):
        if abs(x[0]-x[1])>2:
            if x[0]>x[1]:
                return pd.to_timedelta(-1000,'ms')
        return pd.to_timedelta(0,'ms')
    def process_plot_surveys(dependencies, targets):
        drone = pd.read_csv(list(dependencies)[0],index_col='TimeStamp',parse_dates=['TimeStamp'])
        drone = drone [~drone.index.duplicated()]
        drone['Distance'] = ((drone.Easting.diff(periods=-1)**2+drone.Northing.diff(periods=-1)**2)**0.5)
        drone['AdjustedTime'] = drone.index
        drone['GpsSpeed'] =0.
        drone['DeltaTime'] = (drone['AdjustedTime'].diff(periods=-1).dt.total_seconds())
        drone['Leg'] = 0
        drone.loc[(drone['DeltaTime'].shift(1))<-5,'Leg'] = 1
        drone['Leg'] = drone['Leg'].cumsum()
        def calc_distance(df):
            df['Distance']=(df.Easting.diff()**2+df.Northing.diff()**2)**0.5
            return df
        def cal_dt(df):
            df['DeltaTime'] = df['AdjustedTime'].diff().dt.total_seconds()
            return df
        def cal_speed(df):
            df['GpsSpeed']= (df.Distance/df.DeltaTime)
            df.iloc[0]['GpsSpeed'] = df['GpsSpeed'].mean()
            df['DjiSpeed'] = (df['SpeedX']**2+df['SpeedY']**2)**0.5
            df['Excess'] = ((df['DjiSpeed']*df.DeltaTime)-df.Distance)/df['DjiSpeed']
            df['Delta'] = 0
            df.loc[df['Excess'] <-0.7,'Delta']=-1000
            df.loc[df['Excess'] >0.7,'Delta']=1000 
            df['Delta'] = df['Delta'].cumsum()  
            df['FltAdjustedTime'] = df['AdjustedTime']+pd.to_timedelta(df['Delta'],'ms')
            return df
        drone=drone.groupby('Leg').apply(calc_distance).reset_index().set_index('TimeStamp')
        drone=drone.groupby('Leg').apply(cal_dt).reset_index().set_index('TimeStamp')
        drone=drone.groupby('Leg').apply(cal_speed).reset_index().set_index('TimeStamp')
        drone =drone.fillna(0)
        def process_leg(df):
            X =sm.add_constant((df['FltAdjustedTime']-df['FltAdjustedTime'].min()).dt.total_seconds())
            iX=sm.add_constant((df['AdjustedTime']-df['FltAdjustedTime'].min()).dt.total_seconds())
            model = sm.OLS(df['Northing'],X).fit() 
            df['FltNorthing'] = model.predict(iX)                       
            model = sm.OLS( df['Easting'],X).fit()
            df['FltEasting'] = model.predict(iX)
            df['DeltaTime'] = df.index
            df['DeltaTime'] =df['DeltaTime'].diff().dt.total_seconds()
            df['FltDistance']=(df.FltEasting.diff()**2+df.FltNorthing.diff()**2)**0.5
            df['FltGpsSpeed']= (df.FltDistance/df.DeltaTime)
            return df
        drone =drone.groupby('Leg').apply(process_leg).reset_index().set_index('TimeStamp')
        drone.to_csv(list(targets)[0],index=True)

        
    config = {"config": get_var('con', 'NO')}
    with open(config['config'], 'r') as ymlfile:
        cfg = yaml.load(ymlfile, yaml.SafeLoader)
    files =glob.glob(cfg['locations']['output_survey']+'/**/DRN_XIF*.CSV',recursive=True)
    targets = list(map( lambda s: s.replace('XIF','FLT'), files ))
    for file,target in zip(files,targets):
        yield { 'name': 'plot:{0}'.format(os.path.basename(file)),
                'actions':[(process_plot_surveys)],
                'file_dep':[file],
                'targets':[target],
                'verbosity':2,
                'clean': True
                } 

def task_process_plot_filter():
    def process_plot_surveys(dependencies, targets):
        drone = pd.read_csv(list(dependencies)[0],index_col='TimeStamp',parse_dates=['TimeStamp'])
        fig, ax = plt.subplots(figsize=(8,4))
        drone.GpsSpeed.plot(marker='o',legend=True)
        drone.FltGpsSpeed.plot(marker='o',legend=True)
        plt.savefig(list(targets)[0],dpi=300)
        plt.close()

        
    config = {"config": get_var('con', 'NO')}
    with open(config['config'], 'r') as ymlfile:
        cfg = yaml.load(ymlfile, yaml.SafeLoader)
    files =glob.glob(cfg['locations']['output_survey']+'/**/DRN_FLT*[0-9].CSV',recursive=True)
    for file in files:
        yield { 'name': 'plot:{0}'.format(os.path.basename(file)),
                'actions':[(process_plot_surveys)],
                'file_dep':[file],
                'targets':[os.path.splitext(file)[0]+'.JPG'],
                'verbosity':2,
                'clean': True
                } 

def task_process_plot_flt_surveys():
    def process_plot_flt_surveys(dependencies, targets):
        drone = pd.read_csv(list(dependencies)[0],index_col='TimeStamp',parse_dates=['TimeStamp'])
        plt.scatter(x=drone.Easting,y=drone.Northing,label='Raw')
        plt.scatter(x=drone.FltEasting,y=drone.FltNorthing,label='Corrected')
        plt.legend()
        plt.savefig(list(targets)[0],dpi=300)
        plt.close()

        
    config = {"config": get_var('con', 'NO')}
    with open(config['config'], 'r') as ymlfile:
        cfg = yaml.load(ymlfile, yaml.SafeLoader)
    files =glob.glob(cfg['locations']['output_survey']+'/**/DRN_FLT*[0-9].CSV',recursive=True)
    for file in files:
        yield { 'name': 'plot:{0}'.format(os.path.basename(file)),
                'actions':[(process_plot_flt_surveys)],
                'file_dep':[file],
                'targets':[os.path.splitext(file)[0]+'_FLIGHT.JPG'],
                'verbosity':2,
                'clean': True
                } 




def task_process_image_area():                
    def to_real_wrold(index,altitude,focallen=3666.666504):
        return (index/focallen)*altitude
    
    def make_image_poly(item):
        xw =float(item.ImageWidth)/2
        yw=float(item.ImageHeight)/2
        x = to_real_wrold(np.array([xw,xw,-xw,-xw]),item.RelativeAltitude)
        y = to_real_wrold(np.array([yw,-yw,-yw,yw]),item.RelativeAltitude)
        rads = np.deg2rad(item.GimbalYawDegree)
        xx = (x * np.cos(rads) +  y * np.sin(rads))+ item.FltEasting 
        yy = (-x * np.sin(rads)  +  y  * np.cos(rads))+ item.FltNorthing
        return Polygon(zip(xx,yy))

    def ploy_to_points(polygon):
        return np.dstack(polygon.exterior.coords.xy)

    def process_image_area(dependencies, targets):
        drone = pd.read_csv(list(dependencies)[0],index_col='TimeStamp',parse_dates=['TimeStamp'])
        drone['Polygon'] = drone.apply(make_image_poly,axis=1)
        drone['ImageArea'] = drone.Polygon.apply(lambda x: x.area)  
        drone['SurveyArea'] = MultiPoint(np.hstack(drone['Polygon'].apply(ploy_to_points))[0]).convex_hull.area/10000
        drone.to_csv(list(targets)[0],index=True)

    config = {"config": get_var('con', 'NO')}
    with open(config['config'], 'r') as ymlfile:
        cfg = yaml.load(ymlfile, yaml.SafeLoader)
    files =glob.glob(cfg['locations']['output_survey']+'/**/DRN_FLT*[0-9].CSV',recursive=True)
    for file in files:
        yield { 'name': 'AREA:{0}'.format(os.path.basename(file)),
                'actions':[(process_image_area)],
                'file_dep':[file],
                'targets':[os.path.splitext(file)[0]+'_AREA.CSV'],
                'verbosity':2,
                'clean': True
                } 
                

def task_process_summary_surveys():
    def process_build_surveys(dependencies, targets,cfg):
        if len(list(dependencies))>0:
            drone = pd.concat([pd.read_csv(file,index_col='TimeStamp',parse_dates=['TimeStamp']) 
                            for file in list(dependencies)]) 
            grouped = drone.groupby('GroupId').min()
            grouped.to_csv(list(target)[0],index=True)

        
    config = {"config": get_var('con', 'NO')}
    with open(config['config'], 'r') as ymlfile:
        cfg = yaml.load(ymlfile, yaml.SafeLoader)
    if os.path.exists(cfg['locations']['output_survey_master']):
        target=[cfg['locations']['output_survey']+'metadata' +
                 os.path.sep+'DRN_SUMMARY.CSV']
        files =glob.glob(cfg['locations']['output_survey']+'/**/DRN_FLT*AREA.CSV',recursive=True)
        return {'actions':[(process_build_surveys, [],{'cfg':cfg})],
                'file_dep':files,
                'targets':target,
                'verbosity':2,
                'clean': True,
                } 


def task_process_xml():
    def get_turtle_xy(path):
        #print(path)
        with open(path, "r") as f: # opening xml file
            content = f.read()
        soup = bs(content, "lxml")
        x = [];
        y = [];
        opath =[]
        objtype = []
        for obj in soup.findAll("object"):
            objtype.append(obj.find('name').text)
            opath.append(path) 
            x.append((int(obj.find('xmin').text) +int(obj.find('xmax').text))/2)
            y.append((int(obj.find('ymin').text) +int(obj.find('ymax').text))/2)
        return pd.DataFrame({'path':opath,'x':x,'y':y,'type':objtype})
    def process_xml(dependencies, targets,cfg):
        if len(list(dependencies))>0:
            output =pd.concat([get_turtle_xy(file) for file in list(dependencies)])
            output.to_csv(list(targets)[0],index=True)

        
    config = {"config": get_var('con', 'NO')}
    with open(config['config'], 'r') as ymlfile:
        cfg = yaml.load(ymlfile, yaml.SafeLoader)
    if os.path.exists(cfg['locations']['output_survey_master']):
        target=[cfg['locations']['output_survey']+'metadata' +
                    os.path.sep+'TURTLE_POSITIONS.CSV']
        files =glob.glob('z:/**/*.xml',recursive=True)
        return {'actions':[(process_xml, [],{'cfg':cfg})],
                'file_dep':files,
                'targets':target,
                'verbosity':2,
                'clean': True,
                }          
                
def task_process_match_turtle():
    def process_match_turtle(dependencies, targets):
        pos =pd.read_csv(list(filter(lambda x: 'TURTLE' in x,list(dependencies)))[0])
        pos['BaseName'] = pos.path.apply(os.path.basename)
        pos['BaseName']=  pos['BaseName'].str[0:-4]
        drone =pd.read_csv(list(filter(lambda x: 'AREA' in x,list(dependencies)))[0],parse_dates=['TimeStamp'])
        drone['BaseName']=drone.SourceFile.apply(os.path.basename)
        drone['BaseName']=drone['BaseName'].str[0:-4]
        drone =drone.merge(pos,on=['BaseName'])
        drone['StrikeDelta'] = drone['TimeStamp'].diff().dt.total_seconds()
        drone['TurEasting']=0
        drone['TurNorthing']=0
        drone['FltTurEasting']=0
        drone['FltTurNorthing']=0
        def to_real_wrold(index,altitude,focalen=3666.666504):
            return (index/focalen)*altitude
        def turtle_to_real(item):
            xw =float(item.ImageWidth)/2
            yw=float(item.ImageHeight)/2
            x = to_real_wrold(xw-item.x,item.RelativeAltitude)
            y = to_real_wrold(yw-item.y,item.RelativeAltitude)
            rads = np.deg2rad(item.GimbalYawDegree)
            item.FltTurEasting = (x * np.cos(rads) +  y * np.sin(rads))+ item.FltEasting 
            item.FltTurNorthing = (-x * np.sin(rads)  +  y  * np.cos(rads))+ item.FltNorthing
            item.TurEasting = (x * np.cos(rads) +  y * np.sin(rads))+ item.Easting 
            item.TurNorthing = (-x * np.sin(rads)  +  y  * np.cos(rads))+ item.Northing
            return item
        drone = drone.apply(turtle_to_real,axis=1)
        drone['Strike']=0
        drone.loc[drone.StrikeDelta>12,'Strike']=1
        drone['Strike']=drone['Strike'].cumsum()  
        drone['Sequence'] = 1
        drone['Sequence'] = drone.groupby('Strike')['Sequence'].cumsum()
        drone.to_csv(list(targets)[0],index=False)


        
    config = {"config": get_var('con', 'NO')}
    with open(config['config'], 'r') as ymlfile:
        cfg = yaml.load(ymlfile, yaml.SafeLoader)
    pos=cfg['locations']['output_survey']+'metadata' + os.path.sep+'TURTLE_POSITIONS.CSV'
    files = glob.glob(cfg['locations']['output_survey']+'/**/DRN_FLT*AREA.CSV',recursive=True)
    for file in files:
        yield { 'name': 'Match:{0}'.format(os.path.basename(file)),
                'actions':[(process_match_turtle)],
                'file_dep':[pos,file],
                'targets':[os.path.splitext(file)[0]+'_TURTLE.CSV'],
                'verbosity':2,
                'clean': True
                } 

                



            
def task_process_meanshift_turtle():
    def process_meanshift_turtle(dependencies, targets,bandwidth=6):
        def plot_grp(grp,plotfile=None):
            clustering = MeanShift(bandwidth).fit(np.dstack([grp.FltTurEasting.values,grp.FltTurNorthing.values])[0])
            # calculate the spread
            spread = np.ones(len(np.unique(clustering.labels_))) * np.nan
            looks = np.ones(len(np.unique(clustering.labels_))) * np.nan
            labels = clustering.labels_
            for k in np.unique(clustering.labels_):
                cx = clustering.cluster_centers_[k, 0]
                cy = clustering.cluster_centers_[k, 1]
                members = labels == k
                if np.max(members):
                    spread[k]=np.mean(np.power(np.power(grp.FltTurEasting[members].values-cx,2)+np.power(grp.FltTurNorthing[members].values-cy,2),0.5))
                    looks[k]= np.sum(members)
                else:
                    spread[k]=-1
                    looks[k]= -1
                
            
            clustering.labels_
            n_clusters_ = len(clustering.cluster_centers_)
            if plotfile!=None:
                fig,ax = plt.subplots(figsize=(10,10))
                i=0
                for index,row in grp.iterrows():
                    (x,y)=shapely.wkt.loads(row.Polygon).exterior.xy
                    l1=plt.plot(x, y, color='#6699cc', alpha=0.7,
                        linewidth=3, solid_capstyle='round', zorder=2)
                    l2=plt.plot(row.TurEasting,row.TurNorthing,marker ='x',linestyle='',c='black')
                    plt.text(row.TurEasting, row.TurNorthing  , row.Sequence, fontsize=12)
                    l2=plt.plot(row.FltTurEasting,row.FltTurNorthing,marker ='o',linestyle='',c='green')
                    plt.text(row.FltTurEasting, row.FltTurNorthing  , row.Sequence, fontsize=12)
                    i = i + 1
                ax.set_aspect(1)
                ax.tick_params(axis='both', which='major', labelsize=15)
                l3 =plt.scatter(clustering.cluster_centers_[:, 0], clustering.cluster_centers_[:, 1], c='red', s=50);
                for point in clustering.cluster_centers_:
                    circle = Circle((point[0], point[1]), 5, facecolor='none',
                                    edgecolor=(0, 0.8, 0.8), linewidth=3, alpha=0.5)
                    ax.add_patch(circle)
                    circle = Circle((point[0], point[1]), 2.5, facecolor='none',
                                    edgecolor=(0, 0.8, 0.8), linewidth=3, alpha=0.5)
                    ax.add_patch(circle)

                plt.legend([l1[0],l2[0],l3],['Images','Sightings','Turtles'],prop={'size': 15})
                plt.xlabel('Easting (m)',fontsize=20)
                plt.ylabel('Northing (m)',fontsize=20)
                plt.savefig(plotfile+'_%d.JPG' %(row.Strike),dpi=300)
                plt.close()
            return {'TrutleCount':n_clusters_,'Position':clustering.cluster_centers_,'spread':spread,'looks':looks}
        drone =pd.read_csv(list(dependencies)[0])
        if len(drone)>0:
            p = partial(plot_grp,plotfile =os.path.splitext(list(targets)[0])[0])
            tcounts =pd.DataFrame.from_records(drone.groupby('Strike').apply(p)).apply(pd.Series)
            tcounts.index.name='Strike'
            drone =pd.merge(drone,tcounts,on=['Strike'])
            drone =drone.set_index('TimeStamp').sort_index()
        drone.to_csv(list(targets)[0],index=True)

        
    config = {"config": get_var('con', 'NO')}
    with open(config['config'], 'r') as ymlfile:
        cfg = yaml.load(ymlfile, yaml.SafeLoader)
    files = glob.glob(cfg['locations']['output_survey']+'/**/DRN_FLT*AREA_TURTLE.CSV',recursive=True)
    for file in files:
        yield { 'name': 'Match:{0}'.format(os.path.basename(file)),
                'actions':[(process_meanshift_turtle)],
                'file_dep':[file],
                'targets':[os.path.splitext(file)[0]+'_MS.CSV'],
                'verbosity':2,
                'clean': True
                } 

    



if __name__ == '__main__':
    import doit

    #print(globals())
    doit.run(globals())