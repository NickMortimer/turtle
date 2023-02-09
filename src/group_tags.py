import pandas as pd
from pyproj import Proj 
from utils import convert_wgs_to_utm
from sklearn.cluster import MeanShift
import numpy as np
import matplotlib.pyplot as plt

marks = pd.read_csv('~/april_tag_locations.csv')
utmcode =convert_wgs_to_utm(marks['longitude'].mean(),marks['latitude'].mean())
utmproj =Proj(f'epsg:{utmcode:1.5}')          
marks['easting'],marks['northing'] =utmproj(marks['longitude'].values,marks['latitude'].values)
clustering = MeanShift(bandwidth=2).fit(np.dstack([marks['easting'].values,marks['northing'].values])[0])
n_clusters_ = len(clustering.cluster_centers_)
fig,ax = plt.subplots(figsize=(8,8))
for index,row in marks.iterrows():
    ax.plot(row.easting,row.northing,marker ='x',linestyle='')
ax.set_aspect(1)
plt.scatter(clustering.cluster_centers_[:, 0], clustering.cluster_centers_[:, 1], c='red', s=50);
print({'count':n_clusters_,'centers':clustering.cluster_centers_})
